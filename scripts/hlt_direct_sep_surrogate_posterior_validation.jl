#!/usr/bin/env julia

using AxisKeys
using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

import Distributions

using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))

const DEFAULT_RUN_ROOT = joinpath(
    ".local_artifacts",
    "hlt_validation_runs",
    "hlt3_20260302_153555",
)
const DEFAULT_SURROGATE = joinpath(DEFAULT_RUN_ROOT, "dataset", "hlt_sep_surrogate_trained.jls")
const DEFAULT_SYNTHETIC = joinpath(DEFAULT_RUN_ROOT, "synthetic", "hlt_sep_synth_data.jls")
const DEFAULT_CHAIN = joinpath(DEFAULT_RUN_ROOT, "synthetic", "hlt_sep_surrogate_estimation_chain.jls")
const DEFAULT_HISTORICAL_FOM = joinpath(DEFAULT_RUN_ROOT, "synthetic", "hlt_sep_fom_benchmark.jls")
const DEFAULT_OUT_DIR = joinpath(".local_artifacts", "hlt_direct_sep_surrogate_validation")

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function parse_arg_int_list(args::Vector{String}, key::String, default::Vector{Int})
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return default
    return parse.(Int, strip.(split(raw, ",")))
end

function parse_arg_float_list(args::Vector{String}, key::String, default::Vector{Float64})
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return default
    return parse.(Float64, strip.(split(raw, ",")))
end

function parse_bool_arg(args::Vector{String}, key::String, default::Bool)
    for arg in args
        if startswith(arg, key * "=")
            value = lowercase(strip(split(arg, "=", limit = 2)[2]))
            value in ("1", "true", "yes", "y", "on") && return true
            value in ("0", "false", "no", "n", "off") && return false
            error("Invalid boolean value for $(key): $(value)")
        elseif arg == key
            return true
        end
    end
    return default
end

function ensure_chain_deserialize_modules!()
    try
        @eval import Turing
        @eval import MCMCChains
    catch err
        error("Failed to load chain deserialization dependencies: $(sprint(showerror, err))")
    end
    return nothing
end

function inject_theta(base_params::AbstractVector, model, theta_names::Vector{Symbol}, theta_vals::AbstractVector)
    params = copy(base_params)
    idx_any = indexin(theta_names, model.parameters)
    any(isnothing, idx_any) && error("Theta names not found in $(model.model_name) parameters.")
    for (i, j) in enumerate(Int.(idx_any))
        params[j] = theta_vals[i]
    end
    return params
end

function make_legacy_prior(theta_names::Vector{Symbol};
                           cprobp_mu::Float64,
                           cprobp_sd::Float64,
                           cindp_mu::Float64,
                           cindp_sd::Float64,
                           curvp_mu::Float64,
                           curvp_sd::Float64)
    theta_names == [:cprobp, :cindp, :curvp] ||
        error("This validation harness currently expects HLT legacy theta_names [:cprobp, :cindp, :curvp], got $(theta_names).")

    dists = Distributions.Distribution[
        MacroModelling.Beta(cprobp_mu, cprobp_sd, 0.5, 0.95, μσ = true),
        MacroModelling.Beta(cindp_mu, cindp_sd, 0.01, 0.99, μσ = true),
        Distributions.Normal(curvp_mu, curvp_sd),
    ]
    bounds = [(0.5, 0.95), (0.01, 0.99), (2.0, 150.0)]
    return dists, bounds
end

function logprior_theta(theta::AbstractVector,
                        prior_dists::Vector{Distributions.Distribution},
                        prior_bounds::Vector{Tuple{Float64,Float64}})
    lp = 0.0
    for i in eachindex(theta)
        lb, ub = prior_bounds[i]
        if theta[i] < lb || theta[i] > ub
            return -Inf
        end
        lp += Distributions.logpdf(Distributions.truncated(prior_dists[i], lb, ub), theta[i])
    end
    return lp
end

function first_gated_block(mask::AbstractVector{Bool};
                           context_periods::Int,
                           max_periods::Int,
                           strategy::Symbol)
    selected, eval_idx, context_idx, note =
        MacroModelling.select_gated_block_periods(mask, strategy, context_periods, max_periods)
    isempty(selected) && error("No gated periods selected from chain gate_mask.")
    return selected, eval_idx, context_idx, note
end

function maybe_use_obs_sigma_from_surrogate!(obs_sigma::Vector{Float64},
                                             surrogate_payload::AbstractDict,
                                             d_obs::Int;
                                             mode::Symbol,
                                             scale::Float64,
                                             floor::Float64)
    if mode != :synthetic
        val_rmse = get(surrogate_payload, "validation_rmse", nothing)
        if val_rmse !== nothing
            length(val_rmse) < d_obs && error("validation_rmse length $(length(val_rmse)) < d_obs=$d_obs.")
            obs_rmse = Float64.(val_rmse[1:d_obs]) .* scale
            if mode == :surrogate
                obs_sigma .= obs_rmse
            elseif mode == :max
                obs_sigma .= max.(obs_sigma, obs_rmse)
            else
                error("Unknown --obs-sigma-mode=$(mode). Use synthetic, surrogate, or max.")
            end
        else
            @warn "Surrogate validation_rmse missing; retaining synthetic obs_sigma."
        end
    end
    if floor > 0
        obs_sigma .= max.(obs_sigma, floor)
    end
    return obs_sigma
end

function build_validation_predictors(model,
                                     surrogate_bundle,
                                     synthetic::AbstractDict,
                                     theta_names::Vector{Symbol};
                                     use_obc::Bool)
    surrogate_payload = surrogate_bundle.payload
    frozen = surrogate_bundle.frozen
    sur_meta = surrogate_bundle.meta
    rom_residual = get(sur_meta, "rom_residual", false)
    rom_order = Int(get(sur_meta, "rom_residual_order", 0))
    rom_mode_raw = get(sur_meta, "rom_mode", :baseline)
    rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)
    rom_residual || error("This validation requires a ROM1 residual surrogate.")
    rom_order == 1 || error("Expected rom_residual_order=1, got $(rom_order).")
    rom_mode == :baseline || error("Expected rom_mode=:baseline, got $(rom_mode).")

    observables = Symbol.(synthetic["observables"])
    state_names = Symbol.(synthetic["state_names"])
    obs_idx_any = indexin(observables, model.var)
    state_idx_any = indexin(state_names, model.var)
    any(isnothing, obs_idx_any) && error("Observable names not found in $(model.model_name).")
    any(isnothing, state_idx_any) && error("State names not found in $(model.model_name).")

    rom_predictor = RomPredictor(
        model,
        rom_order,
        rom_mode,
        use_obc,
        Int[],
        copy(model.parameter_values),
        nothing,
        nothing,
        Int.(state_idx_any),
        Int.(obs_idx_any),
    )
    ensure_rom_cache!(rom_predictor, Float64[])

    d_obs = length(observables)
    rom_full_predict = (state, shock_t, theta) -> rom_predict(rom_predictor, state, shock_t, theta)

    surrogate_theta_names = Symbol.(get(sur_meta, "theta_names", Symbol[]))
    if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
        base_params = copy(model.parameter_values)
        theta_baseline = zeros(Float64, length(surrogate_theta_names))
        theta_est_idx = zeros(Int, length(surrogate_theta_names))
        for (si, sname) in enumerate(surrogate_theta_names)
            ei = findfirst(==(sname), theta_names)
            if ei !== nothing
                theta_est_idx[si] = ei
            else
                pi = findfirst(==(sname), model.parameters)
                theta_baseline[si] = pi === nothing ? 0.0 : base_params[pi]
            end
        end
        function pad_theta(theta_local::AbstractVector)
            theta_full = copy(theta_baseline)
            for i in eachindex(theta_est_idx)
                if theta_est_idx[i] > 0
                    theta_full[i] = theta_local[theta_est_idx[i]]
                end
            end
            return theta_full
        end
        surrogate_residual_predict = (state, shock_t, theta) -> predict_frozen(frozen, vcat(state, shock_t, pad_theta(theta)))
    else
        surrogate_residual_predict = (state, shock_t, theta) -> predict_frozen(frozen, vcat(state, shock_t, theta))
    end

    surrogate_step_predict = (state, shock_t, theta) -> MacroModelling.predict_additive_residual(
        rom_full_predict,
        surrogate_residual_predict,
        state,
        shock_t,
        theta,
        d_obs;
        allow_full_residual = true,
    )
    rom_only_predict = (state, shock_t, theta) -> MacroModelling.predict_from_full(
        rom_full_predict,
        state,
        shock_t,
        theta,
        d_obs,
    )

    return (; surrogate_payload,
            rom_full_predict,
            surrogate_step_predict,
            surrogate_residual_predict,
            rom_only_predict)
end

function advance_surrogate_state(surrogate_step_predict,
                                 s0::AbstractVector,
                                 shocks::AbstractMatrix,
                                 theta::AbstractVector,
                                 stop_period::Int)
    stop_period <= 0 && return copy(s0)
    return MacroModelling.advance_state(
        surrogate_step_predict,
        s0,
        shocks[:, 1:stop_period],
        theta,
        stop_period,
    )
end

function summarize_chain(draws::AbstractMatrix, names::Vector{Symbol}, truth::AbstractVector)
    out = Vector{Dict{String,Any}}()
    for j in eachindex(names)
        vals = draws[:, j]
        q05, q95 = Statistics.quantile(vals, [0.05, 0.95])
        sdv = length(vals) > 1 ? Statistics.std(vals) : 0.0
        push!(out, Dict{String,Any}(
            "name" => String(names[j]),
            "truth" => truth[j],
            "mean" => Statistics.mean(vals),
            "sd" => sdv,
            "mcse" => sdv / sqrt(max(length(vals), 1)),
            "q05" => q05,
            "q95" => q95,
        ))
    end
    return out
end

function chain_draw_matrix(chain, theta_names::Vector{Symbol})
    arr = Array(chain[:, theta_names, :])
    if ndims(arr) == 3
        n_iter, n_param, n_chain = size(arr)
        return reshape(permutedims(arr, (1, 3, 2)), n_iter * n_chain, n_param)
    elseif ndims(arr) == 2
        return Matrix{Float64}(arr)
    else
        error("Unsupported chain array dimensions $(size(arr)).")
    end
end

function interval_overlap(a::AbstractDict, b::AbstractDict)
    return max(a["q05"], b["q05"]) <= min(a["q95"], b["q95"])
end

function run_rw_mh(logposterior::Function,
                   theta0::Vector{Float64};
                   samples::Int,
                   burnin::Int,
                   proposal_scales::Vector{Float64},
                   rng::AbstractRNG,
                   label::String)
    d = length(theta0)
    total = samples + burnin
    draws = zeros(Float64, samples, d)
    logposts = fill(-Inf, samples)
    accepted = falses(total)
    attempted_lp = fill(-Inf, total)
    current = copy(theta0)
    current_lp = logposterior(current)
    isfinite(current_lp) || error("$label initial log posterior is not finite at theta0=$(theta0): $(current_lp).")
    kept = 0
    for iter in 1:total
        proposal = current .+ proposal_scales .* randn(rng, d)
        proposal_lp = logposterior(proposal)
        attempted_lp[iter] = proposal_lp
        if isfinite(proposal_lp) && log(rand(rng)) < proposal_lp - current_lp
            current .= proposal
            current_lp = proposal_lp
            accepted[iter] = true
        end
        if iter > burnin
            kept += 1
            draws[kept, :] .= current
            logposts[kept] = current_lp
        end
        println("[$label] iter $iter/$total accept=$(accepted[iter]) lp=$(round(current_lp; digits=4))")
        flush(stdout)
    end
    return Dict{String,Any}(
        "draws" => draws,
        "logpost" => logposts,
        "accept_rate" => Statistics.mean(accepted),
        "accepted" => accepted,
        "attempted_logpost" => attempted_lp,
        "theta0" => theta0,
        "proposal_scales" => proposal_scales,
        "samples" => samples,
        "burnin" => burnin,
    )
end

function format_num(x; digits = 4)
    return @sprintf("%.*f", digits, Float64(x))
end

function write_latex_table(path::String,
                           theta_names::Vector{Symbol},
                           direct_summary::Vector{Dict{String,Any}},
                           surrogate_summary::Vector{Dict{String,Any}})
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & True & Direct mean & Direct 90\\% CI & Surrogate mean & Surrogate 90\\% CI \\\\")
        println(io, "\\midrule")
        for i in eachindex(theta_names)
            d = direct_summary[i]
            s = surrogate_summary[i]
            println(io,
                "\$\\", String(theta_names[i]), "\$ & ",
                format_num(d["truth"]), " & ",
                format_num(d["mean"]), " & [",
                format_num(d["q05"]), ", ",
                format_num(d["q95"]), "] & ",
                format_num(s["mean"]), " & [",
                format_num(s["q05"]), ", ",
                format_num(s["q95"]), "] \\\\",
            )
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function write_markdown_summary(path::String, payload::Dict{String,Any})
    direct_summary = payload["direct_summary"]
    surrogate_summary = payload["surrogate_summary"]
    theta_names = Symbol.(payload["theta_names"])
    open(path, "w") do io
        println(io, "# HLT Direct SEP vs Surrogate Posterior Validation")
        println(io)
        println(io, "- Created: `", payload["created"], "`")
        println(io, "- Status: `", payload["status"], "`")
        println(io, "- Scope: `", payload["scope"], "`")
        println(io, "- Synthetic: `", payload["synthetic_path"], "`")
        println(io, "- Surrogate: `", payload["surrogate_path"], "`")
        println(io, "- Chain source: `", payload["chain_path"], "`")
        println(io, "- Selected periods: `", payload["selected_periods"], "`")
        println(io, "- Evaluation periods: `", payload["evaluation_periods"], "`")
        println(io, "- Context periods: `", payload["context_periods"], "`")
        println(io, "- Period note: `", payload["period_selection_note"], "`")
        println(io, "- Sampler: `random_walk_mh`")
        println(io, "- Samples / burnin: `", payload["samples"], " / ", payload["burnin"], "`")
        println(io, "- Proposal scales: `", payload["proposal_scales"], "`")
        println(io, "- Direct elapsed seconds: `", payload["direct_elapsed_s"], "`")
        println(io, "- Surrogate elapsed seconds: `", payload["surrogate_elapsed_s"], "`")
        println(io, "- Direct accept rate: `", payload["direct_accept_rate"], "`")
        println(io, "- Surrogate accept rate: `", payload["surrogate_accept_rate"], "`")
        if payload["direct_error"] !== nothing
            println(io, "- Direct SEP current-run error: `", replace(String(payload["direct_error"]), "\n" => " "), "`")
            println(io, "- Direct last loglikelihood: `", payload["direct_last_loglik"], "`")
            if payload["direct_last_diagnostics"] !== nothing
                diag = payload["direct_last_diagnostics"]
                println(io, "- Direct failure diagnostics: status=`", get(diag, "status", nothing),
                    "`, code=`", get(diag, "failure_code", nothing),
                    "`, message=`", get(diag, "message", nothing), "`")
            end
        end
        if payload["historical_fom_status"] !== nothing
            println(io, "- Historical direct-FOM artifact: status=`", payload["historical_fom_status"], "`, loglik=`", payload["historical_fom_loglik"], "`, path=`", payload["historical_fom_path"], "`")
        end
        println(io, "- Surrogate source: `", payload["surrogate_source"], "`")
        if payload["surrogate_load_error"] !== nothing
            println(io, "- Surrogate load fallback: `", replace(String(payload["surrogate_load_error"]), "\n" => " "), "`")
        end
        println(io)
        println(io, "This is an HLT legacy three-parameter direct-SEP/MH validation harness. It is not the planned Galí direct SEP-HMC artifact. Direct SEP evaluation is expensive on the current OBC model, so small runs should be read as smoke/provenance checks; use larger `--samples` and `--burnin` values for a submission-grade posterior comparison. When the serialized HLT3 surrogate bundle cannot be loaded by the current Julia runtime, the surrogate side falls back to the existing surrogate-HMC chain payload and the status is marked accordingly.")
        println(io)
        println(io, "## Posterior Summary")
        println(io)
        println(io, "| Parameter | True | Direct mean | Direct 90% CI | Surrogate mean | Surrogate 90% CI | CI overlap |")
        println(io, "|---|---:|---:|---:|---:|---:|---|")
        for i in eachindex(theta_names)
            d = direct_summary[i]
            s = surrogate_summary[i]
            overlap = interval_overlap(d, s)
            println(io, "| `", theta_names[i], "` | ",
                format_num(d["truth"]), " | ",
                format_num(d["mean"]), " | [",
                format_num(d["q05"]), ", ",
                format_num(d["q95"]), "] | ",
                format_num(s["mean"]), " | [",
                format_num(s["q05"]), ", ",
                format_num(s["q95"]), "] | ",
                overlap ? "yes" : "no", " |")
        end
        println(io)
        println(io, "## Acceptance Criterion")
        println(io)
        println(io, "- Mean-distance diagnostics: `", payload["mean_distance_status"], "`")
        println(io, "- 90% interval overlap: `", payload["interval_overlap_status"], "`")
        println(io, "- Production threshold met: `", payload["production_threshold_met"], "`")
        println(io)
        println(io, "## Provenance")
        println(io)
        println(io, "- Raw payload: `", payload["raw_payload_path"], "`")
        println(io, "- LaTeX table: `", payload["latex_table_path"], "`")
        println(io, "- Direct likelihood settings: `", payload["direct_sep_settings"], "`")
        println(io, "- Surrogate inversion settings: `", payload["surrogate_inversion_settings"], "`")
    end
end

surrogate_path = parse_arg_string(ARGS, "--surrogate", DEFAULT_SURROGATE)
synthetic_path = parse_arg_string(ARGS, "--synthetic", DEFAULT_SYNTHETIC)
chain_path = parse_arg_string(ARGS, "--chain", DEFAULT_CHAIN)
historical_fom_path = parse_arg_string(ARGS, "--historical-fom", DEFAULT_HISTORICAL_FOM)
out_dir = parse_arg_string(ARGS, "--out-dir", DEFAULT_OUT_DIR)
samples = parse_arg_int(ARGS, "--samples", 3)
burnin = parse_arg_int(ARGS, "--burnin", 0)
seed = parse_arg_int(ARGS, "--seed", 20260503)
context_periods = parse_arg_int(ARGS, "--context-periods", 1)
max_periods = parse_arg_int(ARGS, "--max-periods", 1)
gated_block = parse_arg_symbol(ARGS, "--gated-block", :first)
proposal_scales = parse_arg_float_list(ARGS, "--proposal-scales", [1e-4, 5e-4, 5e-3])
theta0_mode = parse_arg_symbol(ARGS, "--theta0", Symbol("true"))
obs_sigma_mode = parse_arg_symbol(ARGS, "--obs-sigma-mode", :max)
obs_sigma_scale = parse_arg_float(ARGS, "--obs-sigma-scale", 1.0)
obs_sigma_floor = parse_arg_float(ARGS, "--obs-sigma-floor", 0.0)
use_obc = parse_bool_arg(ARGS, "--use-obc", true)
min_production_samples = parse_arg_int(ARGS, "--min-production-samples", 30)

sep_periods = parse_arg_int(ARGS, "--sep-periods", 4)
sep_order = parse_arg_int(ARGS, "--sep-order", 1)
sep_nnodes = parse_arg_int(ARGS, "--sep-nnodes", 3)
sep_sparse_tree = parse_bool_arg(ARGS, "--sep-sparse-tree", true)
sep_maxit = parse_arg_int(ARGS, "--sep-maxit", 20)
sep_tol = parse_arg_float(ARGS, "--sep-tol", 1e-4)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", 1.0)
sep_shock_scale = parse_arg_float(ARGS, "--sep-shock-scale", 0.5)
sep_inv_maxit = parse_arg_int(ARGS, "--sep-inv-maxit", 1)
sep_inv_step_tol = parse_arg_float(ARGS, "--sep-inv-step-tol", 1e-4)
sep_inv_resid_tol = parse_arg_float(ARGS, "--sep-inv-resid-tol", 1e-3)
sep_inv_lambda = parse_arg_float(ARGS, "--sep-inv-lambda", 1e-3)
sep_inv_predict_tol = parse_arg_float(ARGS, "--sep-inv-predict-tol", 1e-10)
sep_inv_logdet_method = parse_arg_symbol(ARGS, "--sep-inv-logdet-method", :exact)
sep_inv_logdet_sv_tol = parse_arg_float(ARGS, "--sep-inv-logdet-sv-tol", sqrt(eps(Float64)))

inversion_maxit = parse_arg_int(ARGS, "--inversion-maxit", 10)
inversion_tol = parse_arg_float(ARGS, "--inversion-tol", 1e-6)
inversion_lambda = parse_arg_float(ARGS, "--inversion-lambda", 1e-4)

samples > 0 || error("--samples must be positive.")
burnin >= 0 || error("--burnin must be nonnegative.")
length(proposal_scales) == 3 || error("--proposal-scales must contain exactly three comma-separated entries.")

mkpath(out_dir)

ensure_chain_deserialize_modules!()
chain_payload = MacroModelling.load_hlt_chain_payload(chain_path)
synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
surrogate_bundle = nothing
surrogate_load_error = nothing
try
    global surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
catch err
    global surrogate_load_error = sprint(showerror, err)
    println("Warning: could not load surrogate bundle; falling back to existing surrogate chain payload.")
    println(surrogate_load_error)
end

synthetic_model = String(get(synthetic, "model", "Smets_Wouters_2007_HLT_obc"))
model_name = use_obc ? "Smets_Wouters_2007_HLT_obc" : replace(synthetic_model, "_obc" => "")
hlt_model_file_and_symbol(model_name)
model = load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)

obs_data_full = Float64.(synthetic["obs_data"])
obs_sigma = Float64.(synthetic["obs_sigma"])
s0 = Float64.(synthetic["s0"])
shocks = Float64.(synthetic["shocks"])
shock_sigmas = Float64.(synthetic["shock_sigmas"])
theta_names = Symbol.(synthetic["theta_names"])
theta_true = Float64.(synthetic["theta_true"])
observables = Symbol.(synthetic["observables"])

predictors = nothing
if surrogate_bundle !== nothing
    predictors = build_validation_predictors(model, surrogate_bundle, synthetic, theta_names; use_obc = use_obc)
    maybe_use_obs_sigma_from_surrogate!(obs_sigma, predictors.surrogate_payload, size(obs_data_full, 1);
                                        mode = obs_sigma_mode,
                                        scale = obs_sigma_scale,
                                        floor = obs_sigma_floor)
elseif obs_sigma_floor > 0
    obs_sigma .= max.(obs_sigma, obs_sigma_floor)
end

prior_dists, prior_bounds = make_legacy_prior(
    theta_names;
    cprobp_mu = parse_arg_float(ARGS, "--prior-cprobp-mean", 0.5),
    cprobp_sd = parse_arg_float(ARGS, "--prior-cprobp-sd", 0.10),
    cindp_mu = parse_arg_float(ARGS, "--prior-cindp-mean", 0.5),
    cindp_sd = parse_arg_float(ARGS, "--prior-cindp-sd", 0.15),
    curvp_mu = parse_arg_float(ARGS, "--prior-curvp-mean", 75.0),
    curvp_sd = parse_arg_float(ARGS, "--prior-curvp-sd", 25.0),
)

gate_mask = Bool.(chain_payload["gate_mask"])
selected_periods, evaluation_periods, context_idx, period_note =
    first_gated_block(gate_mask; context_periods = context_periods, max_periods = max_periods, strategy = gated_block)
presample_periods = length(context_idx)
obs_selected = obs_data_full[:, selected_periods]
obs_selected_ka = KeyedArray(obs_selected; Variable = observables, Time = selected_periods)
first_selected = first(selected_periods)

theta0 = if theta0_mode == Symbol("true")
    copy(theta_true)
elseif theta0_mode == :post_mean
    Float64.(chain_payload["post_mean_theta"])
else
    error("Unknown --theta0=$(theta0_mode). Use true or post_mean.")
end

direct_sep_settings = Dict{String,Any}(
    "algorithm" => "stochastic_extended_path",
    "filter" => "inversion",
    "presample_periods" => presample_periods,
    "sep_periods" => sep_periods,
    "sep_order" => sep_order,
    "sep_nnodes" => sep_nnodes,
    "sep_sparse_tree" => sep_sparse_tree,
    "sep_maxit" => sep_maxit,
    "sep_tol" => sep_tol,
    "sep_accept_tol" => sep_accept_tol,
    "sep_shock_scale" => sep_shock_scale,
    "sep_inv_maxit" => sep_inv_maxit,
    "sep_inv_step_tol" => sep_inv_step_tol,
    "sep_inv_resid_tol" => sep_inv_resid_tol,
    "sep_inv_lambda" => sep_inv_lambda,
    "sep_inv_predict_tol" => sep_inv_predict_tol,
    "sep_inv_logdet_method" => String(sep_inv_logdet_method),
    "sep_inv_logdet_sv_tol" => sep_inv_logdet_sv_tol,
)
surrogate_inversion_settings = Dict{String,Any}(
    "presample_periods" => presample_periods,
    "inversion_maxit" => inversion_maxit,
    "inversion_tol" => inversion_tol,
    "inversion_lambda" => inversion_lambda,
    "state_initialization" => "surrogate advance through periods 1:$(first_selected - 1)",
)

const FAILURE_LL = -1e12
const DIRECT_LAST_DIAGNOSTICS = Ref{Any}(nothing)
const DIRECT_LAST_LOGLIK = Ref{Any}(nothing)

function direct_loglik(theta::Vector{Float64})
    params = inject_theta(model.parameter_values, model, theta_names, theta)
    ll = MacroModelling.get_loglikelihood(
        model,
        obs_selected_ka,
        params;
        algorithm = :stochastic_extended_path,
        filter = :inversion,
        verbose = false,
        on_failure_loglikelihood = FAILURE_LL,
        presample_periods = presample_periods,
        sep_periods = sep_periods,
        sep_order = sep_order,
        sep_nnodes = sep_nnodes,
        sep_sparse_tree = sep_sparse_tree,
        sep_maxit = sep_maxit,
        sep_tol = sep_tol,
        sep_accept_tol = sep_accept_tol,
        sep_shock_scale = sep_shock_scale,
        sep_inv_maxit = sep_inv_maxit,
        sep_inv_step_tol = sep_inv_step_tol,
        sep_inv_resid_tol = sep_inv_resid_tol,
        sep_inv_lambda = sep_inv_lambda,
        sep_inv_predict_tol = sep_inv_predict_tol,
        sep_inv_logdet_method = sep_inv_logdet_method,
        sep_inv_logdet_sv_tol = sep_inv_logdet_sv_tol,
    )
    DIRECT_LAST_LOGLIK[] = ll
    DIRECT_LAST_DIAGNOSTICS[] = MacroModelling.get_sep_inversion_last_diagnostics()
    (!isfinite(ll) || ll == FAILURE_LL) && return -Inf
    return Float64(ll)
end

function surrogate_loglik(theta::Vector{Float64})
    predictors === nothing && error("Surrogate bundle was not loaded; surrogate likelihood is unavailable.")
    s0_window = advance_surrogate_state(
        predictors.surrogate_step_predict,
        s0,
        shocks,
        theta,
        first_selected - 1,
    )
    ll_vec, _ = MacroModelling.inversion_loglik_per_period(
        predictors.rom_only_predict,
        s0_window,
        theta,
        obs_selected,
        obs_sigma,
        shock_sigmas;
        eval_predict_fn = predictors.surrogate_step_predict,
        maxit = inversion_maxit,
        tol = inversion_tol,
        lambda = inversion_lambda,
    )
    any(!isfinite, ll_vec) && return -Inf
    return sum(ll_vec[(presample_periods + 1):end])
end

function direct_logpost(theta::Vector{Float64})
    lp = logprior_theta(theta, prior_dists, prior_bounds)
    isfinite(lp) || return -Inf
    ll = direct_loglik(theta)
    isfinite(ll) || return -Inf
    return lp + ll
end

function surrogate_logpost(theta::Vector{Float64})
    lp = logprior_theta(theta, prior_dists, prior_bounds)
    isfinite(lp) || return -Inf
    ll = surrogate_loglik(theta)
    isfinite(ll) || return -Inf
    return lp + ll
end

println("HLT direct SEP vs surrogate posterior validation")
println("Model: $(model.model_name)")
println("Selected periods: $(selected_periods), evaluation periods: $(evaluation_periods), context=$(context_idx)")
println("Theta0 ($(theta0_mode)): $(theta0)")
println("Samples/burnin: $(samples)/$(burnin)")
println("Proposal scales: $(proposal_scales)")
flush(stdout)

rng_direct = MersenneTwister(seed)
rng_surrogate = MersenneTwister(seed)

t_direct = time()
direct_error = nothing
direct_chain = try
    run_rw_mh(
        direct_logpost,
        theta0;
        samples = samples,
        burnin = burnin,
        proposal_scales = proposal_scales,
        rng = rng_direct,
        label = "direct-sep",
    )
catch err
    global direct_error = sprint(showerror, err)
    println("Warning: direct SEP chain failed: $(direct_error)")
    Dict{String,Any}(
        "draws" => reshape(theta0, 1, length(theta0)),
        "logpost" => [-Inf],
        "accept_rate" => missing,
        "accepted" => Bool[],
        "attempted_logpost" => Float64[],
        "theta0" => theta0,
        "proposal_scales" => proposal_scales,
        "samples" => 0,
        "burnin" => burnin,
        "error" => direct_error,
        "last_loglik" => DIRECT_LAST_LOGLIK[],
        "last_diagnostics" => DIRECT_LAST_DIAGNOSTICS[],
    )
end
direct_elapsed = round(time() - t_direct; digits = 3)

t_surrogate = time()
surrogate_source = "surrogate_mh_same_window"
surrogate_chain = if predictors !== nothing
    run_rw_mh(
        surrogate_logpost,
        theta0;
        samples = samples,
        burnin = burnin,
        proposal_scales = proposal_scales,
        rng = rng_surrogate,
        label = "surrogate",
    )
else
    draws = chain_draw_matrix(chain_payload["chain"], theta_names)
    surrogate_source = "existing_surrogate_chain_payload"
    Dict{String,Any}(
        "draws" => draws,
        "logpost" => Float64[],
        "accept_rate" => missing,
        "accepted" => Bool[],
        "attempted_logpost" => Float64[],
        "theta0" => get(chain_payload, "init_params", nothing),
        "proposal_scales" => Float64[],
        "samples" => size(draws, 1),
        "burnin" => missing,
    )
end
surrogate_elapsed = round(time() - t_surrogate; digits = 3)

direct_summary = summarize_chain(direct_chain["draws"], theta_names, theta_true)
surrogate_summary = summarize_chain(surrogate_chain["draws"], theta_names, theta_true)

overlap_vec = [interval_overlap(direct_summary[i], surrogate_summary[i]) for i in eachindex(theta_names)]
mean_distance = Float64[]
for i in eachindex(theta_names)
    d = direct_summary[i]
    s = surrogate_summary[i]
    pooled = sqrt(d["mcse"]^2 + s["mcse"]^2 + eps(Float64))
    push!(mean_distance, abs(d["mean"] - s["mean"]) / pooled)
end
mean_distance_status = all(mean_distance .<= 2.0) ? "within_2_mcse" : "outside_2_mcse"
interval_overlap_status = all(overlap_vec) ? "all_overlap" : "not_all_overlap"
historical_fom_status, historical_fom_loglik = if isfile(historical_fom_path)
    try
        hist = deserialize(historical_fom_path)
        result = hist["results"]["true"]
        (get(result, "status", nothing), get(result, "fom_loglik", nothing))
    catch err
        ("unreadable: " * sprint(showerror, err), nothing)
    end
else
    (nothing, nothing)
end

production_threshold_met = samples >= min_production_samples
production_threshold_met = production_threshold_met && predictors !== nothing && direct_error === nothing
status = if direct_error !== nothing
    "direct_sep_backend_failed_current_runtime"
elseif production_threshold_met && mean_distance_status == "within_2_mcse" && interval_overlap_status == "all_overlap"
    "posterior_validation_pass"
elseif production_threshold_met
    "posterior_validation_attention_needed"
elseif predictors === nothing
    "direct_sep_smoke_surrogate_chain_fallback"
else
    "smoke_only_production_run_needed"
end

raw_payload_path = joinpath(out_dir, "hlt_direct_sep_surrogate_validation_payload.jls")
latex_table_path = joinpath(out_dir, "table_hlt_direct_sep_surrogate_validation.tex")
summary_path = joinpath(out_dir, "HLT_DIRECT_SEP_SURROGATE_VALIDATION_SUMMARY.md")

payload = Dict{String,Any}(
    "created" => string(now()),
    "status" => status,
    "scope" => "HLT legacy three-parameter direct-SEP/MH vs surrogate/MH gated-block validation",
    "synthetic_path" => synthetic_path,
    "surrogate_path" => surrogate_path,
    "chain_path" => chain_path,
    "historical_fom_path" => historical_fom_path,
    "historical_fom_status" => historical_fom_status,
    "historical_fom_loglik" => historical_fom_loglik,
    "theta_names" => String.(theta_names),
    "theta_true" => theta_true,
    "theta0" => theta0,
    "theta0_mode" => String(theta0_mode),
    "selected_periods" => selected_periods,
    "evaluation_periods" => evaluation_periods,
    "context_periods" => context_idx,
    "period_selection_note" => period_note,
    "samples" => samples,
    "burnin" => burnin,
    "seed" => seed,
    "proposal_scales" => proposal_scales,
    "direct_elapsed_s" => direct_elapsed,
    "surrogate_elapsed_s" => surrogate_elapsed,
    "direct_accept_rate" => direct_chain["accept_rate"],
    "direct_error" => direct_error,
    "direct_last_loglik" => DIRECT_LAST_LOGLIK[],
    "direct_last_diagnostics" => DIRECT_LAST_DIAGNOSTICS[],
    "surrogate_accept_rate" => surrogate_chain["accept_rate"],
    "surrogate_source" => surrogate_source,
    "surrogate_load_error" => surrogate_load_error,
    "direct_chain" => direct_chain,
    "surrogate_chain" => surrogate_chain,
    "direct_summary" => direct_summary,
    "surrogate_summary" => surrogate_summary,
    "mean_distance" => mean_distance,
    "mean_distance_status" => mean_distance_status,
    "interval_overlap" => overlap_vec,
    "interval_overlap_status" => interval_overlap_status,
    "production_threshold_met" => production_threshold_met,
    "min_production_samples" => min_production_samples,
    "direct_sep_settings" => direct_sep_settings,
    "surrogate_inversion_settings" => surrogate_inversion_settings,
    "raw_payload_path" => raw_payload_path,
    "latex_table_path" => latex_table_path,
    "summary_path" => summary_path,
)

serialize(raw_payload_path, payload)
write_latex_table(latex_table_path, theta_names, direct_summary, surrogate_summary)
write_markdown_summary(summary_path, payload)

println("Saved raw payload: $(raw_payload_path)")
println("Saved summary: $(summary_path)")
println("Saved LaTeX table: $(latex_table_path)")
