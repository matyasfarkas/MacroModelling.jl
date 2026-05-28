#!/usr/bin/env julia

using Dates
using AxisKeys
using LinearAlgebra
using MacroModelling
using Printf
using Random
using Serialization
using Statistics
using TOML

const INV_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

Base.@kwdef struct InversionBridgeOptions
    dataset::String = ""
    surrogate::String = ""
    out_dir::String = joinpath(INV_REPO_ROOT, ".local_artifacts", "hlt_reduced_bridge_validation", "inversion_bridge_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    param_set::Symbol = :investment_4p_supported
    periods::Int = 8
    truth_mode::String = "validation-nearest-center"
    truth_index::Int = 0
    split_seed::Int = 20260527
    obs_sigma_scale::Float64 = 1.0
    obs_sigma_floor::Float64 = 1.0e-3
    direct_eval_points::Int = 25
    sep_horizon::Int = 4
    sep_maxit::Int = 80
    inversion_maxit::Int = 10
    inversion_tol::Float64 = 1.0e-6
    inversion_lambda::Float64 = 1.0e-4
    dry_run::Bool = true
end

function parse_arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix) + 1:end]
    end
    return default
end

function parse_bool(x::AbstractString)
    v = lowercase(strip(x))
    v in ("1", "true", "yes", "y", "on") && return true
    v in ("0", "false", "no", "n", "off") && return false
    error("Cannot parse boolean: $x")
end

function parse_args(args::Vector{String})
    opts = InversionBridgeOptions()
    dataset = parse_arg(args, "--dataset", opts.dataset)
    surrogate = parse_arg(args, "--surrogate", opts.surrogate)
    isempty(dataset) && error("--dataset=<hlt_sep_surrogate_dataset.jls> is required")
    isempty(surrogate) && error("--surrogate=<hlt_sep_surrogate_trained.jls> is required")
    return InversionBridgeOptions(
        dataset = dataset,
        surrogate = surrogate,
        out_dir = parse_arg(args, "--out-dir", opts.out_dir),
        param_set = Symbol(parse_arg(args, "--param-set", String(opts.param_set))),
        periods = parse(Int, parse_arg(args, "--periods", string(opts.periods))),
        truth_mode = parse_arg(args, "--truth-mode", opts.truth_mode),
        truth_index = parse(Int, parse_arg(args, "--truth-index", string(opts.truth_index))),
        split_seed = parse(Int, parse_arg(args, "--split-seed", string(opts.split_seed))),
        obs_sigma_scale = parse(Float64, parse_arg(args, "--obs-sigma-scale", string(opts.obs_sigma_scale))),
        obs_sigma_floor = parse(Float64, parse_arg(args, "--obs-sigma-floor", string(opts.obs_sigma_floor))),
        direct_eval_points = parse(Int, parse_arg(args, "--direct-eval-points", string(opts.direct_eval_points))),
        sep_horizon = parse(Int, parse_arg(args, "--sep-horizon", string(opts.sep_horizon))),
        sep_maxit = parse(Int, parse_arg(args, "--sep-maxit", string(opts.sep_maxit))),
        inversion_maxit = parse(Int, parse_arg(args, "--inversion-maxit", string(opts.inversion_maxit))),
        inversion_tol = parse(Float64, parse_arg(args, "--inversion-tol", string(opts.inversion_tol))),
        inversion_lambda = parse(Float64, parse_arg(args, "--inversion-lambda", string(opts.inversion_lambda))),
        dry_run = parse_bool(parse_arg(args, "--dry-run", string(opts.dry_run))),
    )
end

function git_commit()
    try
        return readchomp(`git -C $INV_REPO_ROOT rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function matrix_from_theta_grid(raw)
    raw isa AbstractVector || error("theta_grid metadata must be a vector.")
    n = length(raw)
    n > 0 || error("theta_grid is empty.")
    p = length(raw[1])
    out = Matrix{Float64}(undef, n, p)
    for i in 1:n
        length(raw[i]) == p || error("theta_grid has ragged rows.")
        out[i, :] .= Float64.(raw[i])
    end
    return out
end

function training_split_indices(n::Int, seed::Int)
    Random.seed!(seed)
    n_train = n == 1 ? 1 : clamp(Int(floor(0.9 * n)), 1, n - 1)
    perm = randperm(n)
    return perm[1:n_train], perm[n_train + 1:end]
end

function nearest_index(theta_grid::Matrix{Float64}, target::Vector{Float64}, candidates::Vector{Int})
    isempty(candidates) && (candidates = collect(1:size(theta_grid, 1)))
    scales = vec(maximum(theta_grid, dims = 1) .- minimum(theta_grid, dims = 1))
    scales[scales .<= sqrt(eps(Float64))] .= 1.0
    scores = [sum(((theta_grid[i, :] .- target) ./ scales) .^ 2) for i in candidates]
    return candidates[argmin(scores)]
end

function nearest_indices(theta_grid::Matrix{Float64}, target::Vector{Float64}, n_keep::Int)
    n_keep = min(n_keep, size(theta_grid, 1))
    scales = vec(maximum(theta_grid, dims = 1) .- minimum(theta_grid, dims = 1))
    scales[scales .<= sqrt(eps(Float64))] .= 1.0
    scores = [sum(((theta_grid[i, :] .- target) ./ scales) .^ 2) for i in 1:size(theta_grid, 1)]
    return sortperm(scores)[1:n_keep]
end

function choose_truth_index(theta_grid::Matrix{Float64}, specs::Vector{ParameterSpec}, opts::InversionBridgeOptions)
    n = size(theta_grid, 1)
    train_idx, val_idx = training_split_indices(n, opts.split_seed)
    if opts.truth_index > 0
        1 <= opts.truth_index <= n || error("--truth-index must be between 1 and $n")
        return opts.truth_index, train_idx, val_idx
    end
    center = Float64[Float64(spec.prior_params.μ) for spec in specs]
    if opts.truth_mode == "validation-nearest-center"
        return nearest_index(theta_grid, center, val_idx), train_idx, val_idx
    elseif opts.truth_mode == "nearest-center"
        return nearest_index(theta_grid, center, collect(1:n)), train_idx, val_idx
    elseif opts.truth_mode == "first-validation"
        isempty(val_idx) && error("No validation points available.")
        return val_idx[1], train_idx, val_idx
    else
        error("Unknown --truth-mode=$(opts.truth_mode).")
    end
end

function fmt(x)
    return @sprintf("%.6g", Float64(x))
end

function logsumexp(x::AbstractVector{<:Real})
    finite = [Float64(v) for v in x if isfinite(Float64(v))]
    isempty(finite) && return -Inf
    m = maximum(finite)
    return m + log(sum(exp.(finite .- m)))
end

function weighted_quantile(vals::Vector{Float64}, weights::Vector{Float64}, p::Float64)
    order = sortperm(vals)
    v = vals[order]
    w = weights[order]
    total = sum(w)
    total > 0 || return NaN
    cdf = cumsum(w) ./ total
    idx = findfirst(>=(p), cdf)
    return v[idx === nothing ? length(v) : idx]
end

function prior_logpdf(theta::Vector{Float64}, specs::Vector{ParameterSpec})
    length(theta) == length(specs) || error("theta/spec length mismatch")
    lp = 0.0
    for (x, spec) in zip(theta, specs)
        lo, hi = spec.bounds
        (lo <= x <= hi) || return -Inf
        if spec.prior_type == :Normal
            μ = Float64(spec.prior_params.μ)
            σ = Float64(spec.prior_params.σ)
            lp += -0.5 * ((x - μ) / σ)^2 - log(σ) - 0.5 * log(2π)
        elseif spec.prior_type == :Uniform
            lp += -log(hi - lo)
        else
            error("Unsupported prior type: $(spec.prior_type)")
        end
    end
    return lp
end

function posterior_rows(theta_grid::Matrix{Float64},
                        candidate_idx::Vector{Int},
                        logpost::Vector{Float64},
                        theta_names::Vector{Symbol})
    lse = logsumexp(logpost)
    weights = isfinite(lse) ? exp.(logpost .- lse) : fill(NaN, length(logpost))
    map_local = argmax(logpost)
    rows = Vector{Dict{String,Any}}()
    for j in 1:size(theta_grid, 2)
        vals = Float64[theta_grid[i, j] for i in candidate_idx]
        mean_j = sum(weights .* vals)
        var_j = sum(weights .* (vals .- mean_j) .^ 2)
        push!(rows, Dict{String,Any}(
            "parameter" => String(theta_names[j]),
            "mean" => mean_j,
            "sd" => sqrt(max(var_j, 0.0)),
            "q05" => weighted_quantile(vals, weights, 0.05),
            "q95" => weighted_quantile(vals, weights, 0.95),
            "map" => vals[map_local],
        ))
    end
    return rows, weights
end

function shock_sigmas_for(model, shock_scale::Float64)
    shock_names = model.exo
    sigmas = zeros(Float64, length(shock_names))
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    for (i, name) in enumerate(shock_names)
        if !obc_mask[i]
            sigmas[i] = MacroModelling.sep_irf_shock_std(model, name)
        end
    end
    return sigmas .* shock_scale
end

function direct_params(model, theta_names::Vector{Symbol}, theta::AbstractVector)
    params = copy(model.parameter_values)
    idx = indexin(theta_names, model.parameters)
    any(isnothing, idx) && error("Theta names missing from direct model parameters.")
    params[Int.(idx)] .= theta
    return params
end

function write_latex_table(path::String, rows::Vector{Dict{String,Any}})
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & Direct mean & Direct 90\\% CI & Surrogate mean & Surrogate 90\\% CI & ROM1 mean & ROM1 90\\% CI \\\\")
        println(io, "\\midrule")
        for row in rows
            dci = "[$(fmt(row["direct_q05"])), $(fmt(row["direct_q95"]))]"
            sci = "[$(fmt(row["surrogate_q05"])), $(fmt(row["surrogate_q95"]))]"
            rci = "[$(fmt(row["rom1_q05"])), $(fmt(row["rom1_q95"]))]"
            pname = replace(row["parameter"], "_" => "\\_")
            println(io, "$pname & $(fmt(row["direct_mean"])) & $dci & $(fmt(row["surrogate_mean"])) & $sci & $(fmt(row["rom1_mean"])) & $rci \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function bridge_manifest(opts::InversionBridgeOptions)
    data = deserialize(opts.dataset)
    bundle = deserialize(opts.surrogate)
    for key in ("meta", "X", "Y", "Y_rom1")
        haskey(data, key) || error("Dataset missing key: $key")
    end
    haskey(bundle, "frozen") || error("Surrogate bundle missing frozen network.")
    meta = data["meta"]
    theta_grid = matrix_from_theta_grid(get(meta, "theta_grid", nothing))
    theta_names = Symbol.(get(meta, "theta_names", Symbol[]))
    isempty(theta_names) && error("Dataset metadata missing theta_names.")
    specs = get_parameter_specs(opts.param_set)
    spec_names = [s.name for s in specs]
    spec_names == theta_names ||
        error("Param set $(opts.param_set) names $spec_names do not match dataset theta_names $theta_names")
    X = Matrix{Float64}(data["X"])
    Y = Matrix{Float64}(data["Y"])
    Y_rom1 = Matrix{Float64}(data["Y_rom1"])
    d_obs = length(get(meta, "observables", Symbol[]))
    d_obs > 0 || error("Dataset metadata missing observables.")
    n = size(theta_grid, 1)
    n == size(X, 2) == size(Y, 2) == size(Y_rom1, 2) ||
        error("Dataset grid/sample count mismatch.")
    truth_idx, train_idx, val_idx = choose_truth_index(theta_grid, specs, opts)
    obs_sigma = max.(opts.obs_sigma_scale .* vec(Statistics.std(Y[1:d_obs, :], dims = 2; corrected = false)), opts.obs_sigma_floor)
    direct_eval_points = min(opts.direct_eval_points, n)
    panel_idx = val_idx[1:min(length(val_idx), opts.periods)]
    if length(panel_idx) < opts.periods
        panel_idx = vcat(panel_idx, train_idx[1:(opts.periods - length(panel_idx))])
    end

    return Dict{String,Any}(
        "created_at" => string(Dates.now()),
        "git_commit" => git_commit(),
        "dry_run" => opts.dry_run,
        "dataset" => opts.dataset,
        "surrogate" => opts.surrogate,
        "param_set" => String(opts.param_set),
        "theta_names" => String.(theta_names),
        "observables" => String.(get(meta, "observables", Symbol[])),
        "dataset_samples" => n,
        "feature_dim" => size(X, 1),
        "output_dim" => size(Y, 1),
        "obs_dim" => d_obs,
        "periods" => opts.periods,
        "truth_mode" => opts.truth_mode,
        "truth_index" => truth_idx,
        "truth_in_training_split" => truth_idx in train_idx,
        "truth_in_validation_split" => truth_idx in val_idx,
        "theta_true" => vec(theta_grid[truth_idx, :]),
        "panel_indices" => panel_idx,
        "obs_sigma" => obs_sigma,
        "direct_eval_points" => direct_eval_points,
        "sep_horizon" => opts.sep_horizon,
        "sep_maxit" => opts.sep_maxit,
        "inversion_maxit" => opts.inversion_maxit,
        "inversion_tol" => opts.inversion_tol,
        "inversion_lambda" => opts.inversion_lambda,
        "execution_plan" => [
            "construct a short synthetic HLT observation panel from a held-out validation sequence",
            "recover shocks with the ROM1 inversion filter under each candidate theta",
            "evaluate the direct SEP inversion objective on the same observation panel and parameter grid",
            "evaluate the ROM1-residual surrogate objective with the same recovered-shock architecture",
            "compare direct SEP, ROM1, and surrogate posterior surfaces by means, intervals, MAP ranking, and surface RMSE",
        ],
        "acceptance_criteria" => [
            "direct SEP inversion objective finite for the truth point and all direct evaluation anchors",
            "surrogate 90 percent intervals overlap direct SEP for every bridge parameter",
            "surrogate centered surface RMSE, after removing the mean objective offset, is materially below ROM1 centered surface RMSE",
            "local MAP ranking is reported as a diagnostic; this held-out-panel stress test is not a coherent synthetic time-series DGP",
        ],
        "artifact_schema" => [
            "manifest.toml",
            "SUMMARY.md",
            "synthetic_panel.jls",
            "direct_inversion_grid.jls",
            "surrogate_inversion_grid.jls",
            "comparison_table.tex",
        ],
    )
end

function write_summary(path::String, manifest::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT Multi-Period Inversion Bridge")
        println(io)
        println(io, "- Created: `$(manifest["created_at"])`")
        println(io, "- Git commit: `$(manifest["git_commit"])`")
        println(io, "- Dry run: `$(manifest["dry_run"])`")
        println(io, "- Dataset: `$(manifest["dataset"])`")
        println(io, "- Surrogate: `$(manifest["surrogate"])`")
        println(io, "- Parameter set: `$(manifest["param_set"])`")
        println(io, "- Periods: `$(manifest["periods"])`")
        println(io, "- Direct evaluation points: `$(manifest["direct_eval_points"])`")
        println(io, "- SEP horizon/maxit: `$(manifest["sep_horizon"]) / $(manifest["sep_maxit"])`")
        println(io, "- Inversion maxit/tol/lambda: `$(manifest["inversion_maxit"]) / $(manifest["inversion_tol"]) / $(manifest["inversion_lambda"])`")
        println(io, "- Truth index: `$(manifest["truth_index"])`")
        println(io, "- Truth theta: `$(join(["$(manifest["theta_names"][i])=$(fmt(manifest["theta_true"][i]))" for i in eachindex(manifest["theta_names"])], ", "))`")
        println(io, "- Truth in validation split: `$(manifest["truth_in_validation_split"])`")
        println(io, "- Observation sigma: `$(join(fmt.(manifest["obs_sigma"]), ", "))`")
        println(io)
        println(io, "## Execution Plan")
        println(io)
        for item in manifest["execution_plan"]
            println(io, "- $item")
        end
        println(io)
        println(io, "## Acceptance Criteria")
        println(io)
        for item in manifest["acceptance_criteria"]
            println(io, "- $item")
        end
        println(io)
        println(io, "## Current Status")
        println(io)
        if manifest["dry_run"]
            println(io, "Design scaffold only. The script has verified dataset/surrogate compatibility, parameter names, split provenance, and artifact schema. The executable inversion evaluator is the next code step.")
        else
            println(io, "Executable mode completed. See the posterior table and serialized payload in this directory. The panel is assembled from held-out one-step HLT bridge observations, so this is an inversion-objective stress test, not a coherent full-sample DGP.")
        end
    end
end

function run_executable_bridge(opts::InversionBridgeOptions, manifest::Dict{String,Any})
    data = deserialize(opts.dataset)
    bundle = deserialize(opts.surrogate)
    meta = data["meta"]
    theta_grid = matrix_from_theta_grid(get(meta, "theta_grid", nothing))
    theta_names = Symbol.(get(meta, "theta_names", Symbol[]))
    observables = Symbol.(get(meta, "observables", Symbol[]))
    state_names = Symbol.(get(meta, "state_names", Symbol[]))
    isempty(state_names) && error("Dataset metadata missing state_names; cannot build ROM predictor.")
    Y = Matrix{Float64}(data["Y"])
    d_obs = length(observables)
    specs = get_parameter_specs(opts.param_set)
    truth_idx = Int(manifest["truth_index"])
    theta_true = Float64.(manifest["theta_true"])
    candidate_idx = nearest_indices(theta_grid, theta_true, opts.direct_eval_points)
    truth_idx in candidate_idx || (candidate_idx[end] = truth_idx)
    candidate_idx = unique(candidate_idx)

    # Held-out short panel: nearest validation points around the selected truth.
    _, val_idx = training_split_indices(size(theta_grid, 1), opts.split_seed)
    panel_idx = nearest_indices(theta_grid[val_idx, :], theta_true, min(opts.periods, length(val_idx)))
    panel_idx = val_idx[panel_idx]
    obs_data = Matrix{Float64}(Y[1:d_obs, panel_idx])
    obs_sigma = Float64.(manifest["obs_sigma"])
    obs_ka = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))

    model_direct = load_hlt_model(INV_REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
    # The bridge surrogate dataset was generated on the OBC HLT state space.
    # Use the same model for ROM state propagation so state/shock dimensions
    # match the trained residual network.
    model_rom = model_direct
    obs_idx = indexin(observables, model_rom.var)
    state_idx = indexin(state_names, model_rom.var)
    any(isnothing, obs_idx) && error("Observables missing from ROM model.")
    any(isnothing, state_idx) && error("States missing from ROM model.")
    theta_param_idx = indexin(theta_names, model_rom.parameters)
    any(isnothing, theta_param_idx) && error("Theta names missing from ROM model parameters.")

    base_params = copy(model_rom.parameter_values)
    MacroModelling.write_parameters_input!(model_rom, base_params, verbose = false)
    # Avoid the generated state-transition closure used by `RomPredictor` here.
    # This executable bridge is loaded dynamically, and the closure can hit Julia
    # world-age errors after the HLT OBC model include. The matrix predictor uses
    # the same first-order solution matrix and is also the path used by the
    # inversion validation code where ForwardDiff Jacobians are required.
    Base.invokelatest(
        MacroModelling.solve!,
        model_rom;
        algorithm = :first_order,
        dynamics = true,
        obc = false,
        silent = true,
    )
    rom_full_predict, rom_predict_tuple, nsss = build_matrix_rom_predict(
        model_rom;
        state_idx = Int.(state_idx),
        obs_idx = Int.(obs_idx),
    )
    s0 = Float64.(nsss[Int.(state_idx)])
    shock_sigmas = shock_sigmas_for(model_rom, 1.0)
    frozen = bundle["frozen"]
    sur_meta = get(bundle, "meta", Dict{String,Any}())
    d_state = length(s0)
    d_eps = length(shock_sigmas)

    surrogate_theta_names = Symbol.(get(sur_meta, "theta_names", Symbol[]))
    if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
        sur_theta_baseline = zeros(Float64, length(surrogate_theta_names))
        sur_theta_est_idx = zeros(Int, length(surrogate_theta_names))
        for (si, sname) in enumerate(surrogate_theta_names)
            ei = findfirst(==(sname), theta_names)
            if ei !== nothing
                sur_theta_est_idx[si] = ei
            else
                pi = findfirst(==(sname), model_rom.parameters)
                sur_theta_baseline[si] = pi !== nothing ? base_params[pi] : 0.0
            end
        end
        function pad_theta(θ_local)
            θ_full = copy(sur_theta_baseline)
            for i in eachindex(sur_theta_est_idx)
                sur_theta_est_idx[i] > 0 && (θ_full[i] = θ_local[sur_theta_est_idx[i]])
            end
            return θ_full
        end
        residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, pad_theta(θ_local)))[1:d_obs]
    else
        residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, θ_local))[1:d_obs]
    end
    surrogate_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
        rom_full_predict,
        residual_predict,
        state,
        shock_t,
        θ_local,
        d_obs;
        allow_full_residual = false,
    )
    direct_ll = fill(-Inf, length(candidate_idx))
    surrogate_ll = fill(-Inf, length(candidate_idx))
    rom1_ll = fill(-Inf, length(candidate_idx))
    statuses = fill("pending", length(candidate_idx))
    elapsed_direct = zeros(Float64, length(candidate_idx))
    for (k, idx) in enumerate(candidate_idx)
        theta = vec(theta_grid[idx, :])
        lp = prior_logpdf(theta, specs)
        isfinite(lp) || (statuses[k] = "prior_out_of_bounds"; continue)

        t0 = time()
        try
            params = direct_params(model_direct, theta_names, theta)
            direct_ll[k] = MacroModelling.get_loglikelihood(
                model_direct,
                obs_ka,
                params;
                algorithm = :stochastic_extended_path,
                filter = :inversion,
                verbose = false,
                on_failure_loglikelihood = -1.0e12,
                presample_periods = 0,
                sep_periods = opts.sep_horizon,
                sep_order = 1,
                sep_nnodes = 3,
                sep_sparse_tree = true,
                sep_maxit = opts.sep_maxit,
                sep_tol = 1.0e-4,
                sep_accept_tol = 1.0e-2,
                sep_shock_scale = 0.5,
                sep_inv_maxit = 1,
                sep_inv_step_tol = 1.0e-4,
                sep_inv_resid_tol = 1.0e-3,
                sep_inv_lambda = 1.0e-3,
                sep_inv_predict_tol = 1.0e-10,
                sep_inv_logdet_method = :exact,
                sep_inv_logdet_sv_tol = sqrt(eps(Float64)),
            )
            statuses[k] = isfinite(direct_ll[k]) && direct_ll[k] > -1.0e11 ? "ok" : "direct_failure"
        catch err
            statuses[k] = "direct_error: " * sprint(showerror, err)
            direct_ll[k] = -Inf
        end
        elapsed_direct[k] = time() - t0

        try
            ll_sur, _ = MacroModelling.inversion_loglik_per_period(
                rom_predict_tuple,
                s0,
                theta,
                obs_data,
                obs_sigma,
                shock_sigmas;
                eval_predict_fn = surrogate_predict,
                maxit = opts.inversion_maxit,
                tol = opts.inversion_tol,
                lambda = opts.inversion_lambda,
            )
            surrogate_ll[k] = sum(ll_sur)
        catch err
            isfinite(direct_ll[k]) && (statuses[k] *= "; surrogate_error: " * sprint(showerror, err))
            surrogate_ll[k] = -Inf
        end

        try
            ll_rom, _ = MacroModelling.inversion_loglik_per_period(
                rom_predict_tuple,
                s0,
                theta,
                obs_data,
                obs_sigma,
                shock_sigmas;
                maxit = opts.inversion_maxit,
                tol = opts.inversion_tol,
                lambda = opts.inversion_lambda,
            )
            rom1_ll[k] = sum(ll_rom)
        catch
            rom1_ll[k] = -Inf
        end

        println("$(k)/$(length(candidate_idx)) idx=$idx direct=$(fmt(direct_ll[k])) surrogate=$(fmt(surrogate_ll[k])) rom1=$(fmt(rom1_ll[k])) status=$(statuses[k])")
    end

    prior = [prior_logpdf(vec(theta_grid[idx, :]), specs) for idx in candidate_idx]
    direct_post = direct_ll .+ prior
    surrogate_post = surrogate_ll .+ prior
    rom1_post = rom1_ll .+ prior
    direct_rows, _ = posterior_rows(theta_grid, candidate_idx, direct_post, theta_names)
    surrogate_rows, _ = posterior_rows(theta_grid, candidate_idx, surrogate_post, theta_names)
    rom1_rows, _ = posterior_rows(theta_grid, candidate_idx, rom1_post, theta_names)

    rows = Vector{Dict{String,Any}}()
    all_overlap = true
    for j in 1:length(theta_names)
        drow = direct_rows[j]
        srow = surrogate_rows[j]
        rrow = rom1_rows[j]
        overlap = max(drow["q05"], srow["q05"]) <= min(drow["q95"], srow["q95"])
        all_overlap &= overlap
        push!(rows, Dict{String,Any}(
            "parameter" => String(theta_names[j]),
            "direct_mean" => drow["mean"],
            "direct_q05" => drow["q05"],
            "direct_q95" => drow["q95"],
            "surrogate_mean" => srow["mean"],
            "surrogate_q05" => srow["q05"],
            "surrogate_q95" => srow["q95"],
            "rom1_mean" => rrow["mean"],
            "rom1_q05" => rrow["q05"],
            "rom1_q95" => rrow["q95"],
            "surrogate_interval_overlap" => overlap,
        ))
    end
    ok = isfinite.(direct_post) .& isfinite.(surrogate_post)
    surface_rmse_sur = any(ok) ? sqrt(mean((surrogate_post[ok] .- direct_post[ok]) .^ 2)) : Inf
    surface_offset_sur = any(ok) ? mean(surrogate_post[ok] .- direct_post[ok]) : NaN
    surface_centered_rmse_sur = any(ok) ? sqrt(mean(((surrogate_post[ok] .- direct_post[ok]) .- surface_offset_sur) .^ 2)) : Inf
    ok_rom = isfinite.(direct_post) .& isfinite.(rom1_post)
    surface_rmse_rom = any(ok_rom) ? sqrt(mean((rom1_post[ok_rom] .- direct_post[ok_rom]) .^ 2)) : Inf
    surface_offset_rom = any(ok_rom) ? mean(rom1_post[ok_rom] .- direct_post[ok_rom]) : NaN
    surface_centered_rmse_rom = any(ok_rom) ? sqrt(mean(((rom1_post[ok_rom] .- direct_post[ok_rom]) .- surface_offset_rom) .^ 2)) : Inf
    direct_map = argmax(direct_post)
    surrogate_map = argmax(surrogate_post)
    rom1_map = argmax(rom1_post)
    pass = all_overlap &&
        surface_centered_rmse_sur < surface_centered_rmse_rom &&
        count(==("ok"), statuses) >= min(3, length(statuses))

    result = Dict{String,Any}(
        "manifest" => manifest,
        "candidate_idx" => candidate_idx,
        "panel_idx" => panel_idx,
        "obs_data" => obs_data,
        "direct_ll" => direct_ll,
        "surrogate_ll" => surrogate_ll,
        "rom1_ll" => rom1_ll,
        "prior" => prior,
        "direct_logpost" => direct_post,
        "surrogate_logpost" => surrogate_post,
        "rom1_logpost" => rom1_post,
        "statuses" => statuses,
        "elapsed_direct_seconds" => elapsed_direct,
        "rows" => rows,
        "surface_rmse_surrogate_vs_direct" => surface_rmse_sur,
        "surface_rmse_rom1_vs_direct" => surface_rmse_rom,
        "surface_offset_surrogate_vs_direct" => surface_offset_sur,
        "surface_offset_rom1_vs_direct" => surface_offset_rom,
        "surface_centered_rmse_surrogate_vs_direct" => surface_centered_rmse_sur,
        "surface_centered_rmse_rom1_vs_direct" => surface_centered_rmse_rom,
        "direct_map_local_index" => direct_map,
        "surrogate_map_local_index" => surrogate_map,
        "rom1_map_local_index" => rom1_map,
        "surrogate_all_interval_overlap" => all_overlap,
        "comparison_pass" => pass,
    )
    payload_path = joinpath(opts.out_dir, "inversion_bridge_comparison.jls")
    table_path = joinpath(opts.out_dir, "comparison_table.tex")
    serialize(payload_path, result)
    write_latex_table(table_path, rows)
    open(joinpath(opts.out_dir, "SUMMARY.md"), "a") do io
        println(io)
        println(io, "## Executable Result")
        println(io)
        println(io, "- Candidate anchors: `$(length(candidate_idx))`")
        println(io, "- Direct finite anchors: `$(count(==("ok"), statuses)) / $(length(statuses))`")
        println(io, "- Surface RMSE, surrogate vs direct: `$(fmt(surface_rmse_sur))`")
        println(io, "- Surface RMSE, ROM1 vs direct: `$(fmt(surface_rmse_rom))`")
        println(io, "- Mean surface offset, surrogate minus direct: `$(fmt(surface_offset_sur))`")
        println(io, "- Mean surface offset, ROM1 minus direct: `$(fmt(surface_offset_rom))`")
        println(io, "- Centered surface RMSE, surrogate vs direct: `$(fmt(surface_centered_rmse_sur))`")
        println(io, "- Centered surface RMSE, ROM1 vs direct: `$(fmt(surface_centered_rmse_rom))`")
        println(io, "- Local MAP agreement, surrogate/direct: `$(surrogate_map == direct_map)`")
        println(io, "- Local MAP agreement, ROM1/direct: `$(rom1_map == direct_map)`")
        println(io, "- Surrogate intervals overlap direct: `$(all_overlap)`")
        println(io, "- Comparison pass: `$(pass)`")
        println(io, "- Payload: `$(payload_path)`")
        println(io, "- Table: `$(table_path)`")
        println(io)
        println(io, "### Posterior Marginals")
        println(io)
        println(io, "| Parameter | Direct mean | Direct 90% CI | Surrogate mean | Surrogate 90% CI | ROM1 mean | ROM1 90% CI |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for row in rows
            dci = "[$(fmt(row["direct_q05"])), $(fmt(row["direct_q95"]))]"
            sci = "[$(fmt(row["surrogate_q05"])), $(fmt(row["surrogate_q95"]))]"
            rci = "[$(fmt(row["rom1_q05"])), $(fmt(row["rom1_q95"]))]"
            println(io, "| `$(row["parameter"])` | $(fmt(row["direct_mean"])) | $dci | $(fmt(row["surrogate_mean"])) | $sci | $(fmt(row["rom1_mean"])) | $rci |")
        end
    end
    return result
end

function run_inversion_bridge(opts::InversionBridgeOptions)
    mkpath(opts.out_dir)
    manifest = bridge_manifest(opts)
    manifest_path = joinpath(opts.out_dir, "manifest.toml")
    summary_path = joinpath(opts.out_dir, "SUMMARY.md")
    open(manifest_path, "w") do io
        TOML.print(io, manifest)
    end
    write_summary(summary_path, manifest)
    println("Wrote manifest: $manifest_path")
    println("Wrote summary: $summary_path")
    if !opts.dry_run
        result = run_executable_bridge(opts, manifest)
        println("Executable comparison pass: $(result["comparison_pass"])")
    end
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_inversion_bridge(parse_args(ARGS))
end
