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
    panel_mode::String = "heldout-one-step"
    direct_objective::String = "exact-inversion"
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
        panel_mode = parse_arg(args, "--panel-mode", opts.panel_mode),
        direct_objective = parse_arg(args, "--direct-objective", opts.direct_objective),
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

function write_theta!(model, theta_names::Vector{Symbol}, theta::AbstractVector)
    params = direct_params(model, theta_names, theta)
    MacroModelling.write_parameters_input!(model, params, verbose = false)
    return params
end

function solve_matrix_rom_context!(model,
                                   theta_names::Vector{Symbol},
                                   theta::AbstractVector,
                                   state_idx::Vector{Int},
                                   obs_idx::Vector{Int})
    write_theta!(model, theta_names, theta)
    Base.invokelatest(
        MacroModelling.solve!,
        model;
        algorithm = :first_order,
        dynamics = true,
        obc = false,
        silent = true,
    )
    rom_full_predict, rom_predict_tuple, nsss = build_matrix_rom_predict(
        model;
        state_idx = state_idx,
        obs_idx = obs_idx,
    )
    return (
        rom_full_predict = rom_full_predict,
        rom_predict_tuple = rom_predict_tuple,
        nsss = Float64.(nsss),
        shock_sigmas = shock_sigmas_for(model, 1.0),
    )
end

function full_state_from_subset(nsss::Vector{Float64},
                                state_idx::Vector{Int},
                                state_subset::AbstractVector)
    length(state_subset) == length(state_idx) ||
        error("State subset length $(length(state_subset)) does not match state index length $(length(state_idx)).")
    state_full = copy(nsss)
    state_full[state_idx] .= Float64.(state_subset)
    return state_full
end

function direct_sep_predict_subset(model,
                                   theta_names::Vector{Symbol},
                                   theta::AbstractVector,
                                   obs_idx::Vector{Int},
                                   state_idx::Vector{Int},
                                   nsss::Vector{Float64},
                                   state_subset::AbstractVector,
                                   shock_full::AbstractVector,
                                   opts::InversionBridgeOptions)
    params = write_theta!(model, theta_names, theta)
    state_full = full_state_from_subset(nsss, state_idx, state_subset)
    shocks = reshape(Float64.(shock_full), :, 1)
    local res
    try
        res = MacroModelling.simulate_sep_extended_path(
            model;
            periods = 1,
            initial_state = state_full,
            shocks = shocks,
            burn_in = 0,
            sep_horizon = opts.sep_horizon,
            sep_order = 1,
            sep_nnodes = 3,
            sep_sparse_tree = true,
            sep_maxit = opts.sep_maxit,
            sep_tol = 1.0e-7,
            sep_accept_tol = 1.0e-2,
            sep_linear_solver = :qr,
            sep_fallback_solver = :normal_equations,
            sep_recovery = true,
            sep_recovery_scales = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
            shock_scaling = :none,
            silent = true,
        )
    catch err
        return (
            ok = false,
            obs = fill(NaN, length(obs_idx)),
            state_next = Float64.(state_subset),
            sep_err = Inf,
            message = sprint(showerror, err),
        )
    end
    errflag = hasproperty(res, :errorflag) ? Bool(res.errorflag) : false
    sep_err = if hasproperty(res, :sep_errors) && !isempty(res.sep_errors)
        last(res.sep_errors)
    else
        NaN
    end
    if errflag
        return (
            ok = false,
            obs = fill(NaN, length(obs_idx)),
            state_next = Float64.(state_subset),
            sep_err = sep_err,
            message = "simulate_sep_extended_path returned errorflag=true",
        )
    end
    sim = Float64.(Array(res.simulation))
    if size(sim, 2) < 2
        return (
            ok = false,
            obs = fill(NaN, length(obs_idx)),
            state_next = Float64.(state_subset),
            sep_err = Inf,
            message = "SEP simulation returned fewer than two columns",
        )
    end
    next_full = sim[:, 2]
    obs = next_full[obs_idx]
    state_next = next_full[state_idx]
    ok = all(isfinite, obs) && all(isfinite, state_next)
    return (
        ok = ok,
        obs = ok ? obs : fill(NaN, length(obs_idx)),
        state_next = ok ? state_next : Float64.(state_subset),
        sep_err = sep_err,
        message = ok ? "" : "Direct SEP prediction returned non-finite values",
    )
end

function make_direct_sep_eval_predict(model,
                                      theta_names::Vector{Symbol},
                                      obs_idx::Vector{Int},
                                      state_idx::Vector{Int},
                                      nsss::Vector{Float64},
                                      rom_predict_fn::Function,
                                      opts::InversionBridgeOptions)
    function direct_sep_eval_predict(state::AbstractVector, shock::AbstractVector, theta::AbstractVector)
        pred = direct_sep_predict_subset(
            model,
            theta_names,
            theta,
            obs_idx,
            state_idx,
            nsss,
            state,
            shock,
            opts,
        )
        if !pred.ok
            return fill(NaN, length(obs_idx)), fill(NaN, length(state_idx))
        end
        _, rom_state_next = rom_predict_fn(state, shock, theta)
        return pred.obs, rom_state_next
    end
    return direct_sep_eval_predict
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
    rom_mode_raw = get(meta, "rom_mode", :baseline)
    rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)
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
    rom_desc = rom_mode == :baseline ? "baseline ROM1" : "candidate-specific ROM1"

    execution_plan = opts.panel_mode == "surrogate-rollout" ? [
        "construct a deterministic dynamic HLT observation panel by rolling the trained ROM1-residual bridge at one truth theta",
        "recover shocks with the $(rom_desc) inversion filter under each candidate theta",
        "evaluate the direct SEP inversion objective on the same observation panel and parameter grid",
        "evaluate the ROM1-residual surrogate objective with the same recovered-shock architecture",
        "compare direct SEP, ROM1, and surrogate posterior surfaces by means, intervals, MAP ranking, and surface RMSE",
    ] : opts.panel_mode == "direct-sep-rollout" ? [
        "construct a deterministic dynamic HLT observation panel by rolling direct SEP at one truth theta",
        "recover shocks with the $(rom_desc) inversion filter under each candidate theta",
        "evaluate a direct-SEP measurement-error objective using the same recovered-shock architecture as the surrogate",
        "evaluate ROM1 and ROM1-residual surrogate objectives with $(rom_desc) matrices and shock scales",
        "compare centered objective surfaces, posterior intervals, and local MAP ranking",
    ] : [
        "construct a short synthetic HLT observation panel from a held-out validation sequence",
        "recover shocks with the $(rom_desc) inversion filter under each candidate theta",
        "evaluate the direct SEP inversion objective on the same observation panel and parameter grid",
        "evaluate the ROM1-residual surrogate objective with the same recovered-shock architecture",
        "compare direct SEP, ROM1, and surrogate posterior surfaces by means, intervals, MAP ranking, and surface RMSE",
    ]
    acceptance_criteria = opts.panel_mode == "surrogate-rollout" ? [
        "direct SEP inversion objective finite for the truth point and all direct evaluation anchors",
        "surrogate and ROM1 inversion objectives finite for all candidate anchors",
        "surrogate 90 percent intervals overlap direct SEP for every bridge parameter",
        "surface RMSE and MAP ranking are reported as diagnostics; this smoke validates dynamic bridge execution before a direct-SEP-generated panel is available",
    ] : opts.panel_mode == "direct-sep-rollout" ? [
        "direct SEP measurement-error objective finite for the truth point and all direct evaluation anchors",
        "surrogate 90 percent intervals overlap direct SEP for every bridge parameter",
        "surrogate centered surface RMSE, after removing the mean objective offset, is materially below ROM1 centered surface RMSE",
        "local MAP ranking is reported as a diagnostic; direct SEP is the DGP for this panel",
    ] : [
        "direct SEP inversion objective finite for the truth point and all direct evaluation anchors",
        "surrogate 90 percent intervals overlap direct SEP for every bridge parameter",
        "surrogate centered surface RMSE, after removing the mean objective offset, is materially below ROM1 centered surface RMSE",
        "local MAP ranking is reported as a diagnostic; this held-out-panel stress test is not a coherent synthetic time-series DGP",
    ]

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
        "rom_mode" => String(rom_mode),
        "periods" => opts.periods,
        "truth_mode" => opts.truth_mode,
        "truth_index" => truth_idx,
        "panel_mode" => opts.panel_mode,
        "direct_objective" => opts.direct_objective,
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
        "execution_plan" => execution_plan,
        "acceptance_criteria" => acceptance_criteria,
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
        println(io, "- ROM mode: `$(manifest["rom_mode"])`")
        println(io, "- Periods: `$(manifest["periods"])`")
        println(io, "- Direct evaluation points: `$(manifest["direct_eval_points"])`")
        println(io, "- SEP horizon/maxit: `$(manifest["sep_horizon"]) / $(manifest["sep_maxit"])`")
        println(io, "- Inversion maxit/tol/lambda: `$(manifest["inversion_maxit"]) / $(manifest["inversion_tol"]) / $(manifest["inversion_lambda"])`")
        println(io, "- Truth index: `$(manifest["truth_index"])`")
        println(io, "- Panel mode: `$(manifest["panel_mode"])`")
        println(io, "- Direct objective: `$(manifest["direct_objective"])`")
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
            if manifest["panel_mode"] == "surrogate-rollout"
                println(io, "Executable mode completed. See the posterior table and serialized payload in this directory. The panel is a deterministic dynamic rollout from the trained ROM1-residual bridge at one fixed truth theta, using shocks from the grid artifact.")
            elseif manifest["panel_mode"] == "direct-sep-rollout"
                println(io, "Executable mode completed. See the posterior table and serialized payload in this directory. The panel is a deterministic dynamic rollout from direct SEP at one fixed truth theta, using shocks from the grid artifact.")
            else
                println(io, "Executable mode completed. See the posterior table and serialized payload in this directory. The panel is assembled from held-out one-step HLT bridge observations, so this is an inversion-objective stress test, not a coherent full-sample DGP.")
            end
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
    rom_mode_raw = get(meta, "rom_mode", :baseline)
    rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)
    isempty(state_names) && error("Dataset metadata missing state_names; cannot build ROM predictor.")
    X = Matrix{Float64}(data["X"])
    Y = Matrix{Float64}(data["Y"])
    d_obs = length(observables)
    specs = get_parameter_specs(opts.param_set)
    truth_idx = Int(manifest["truth_index"])
    theta_true = Float64.(manifest["theta_true"])
    candidate_idx = nearest_indices(theta_grid, theta_true, opts.direct_eval_points)
    truth_idx in candidate_idx || (candidate_idx[end] = truth_idx)
    candidate_idx = unique(candidate_idx)

    # Held-out artifact indices used either as one-step observations or as
    # deterministic shock seeds for a coherent dynamic rollout.
    _, val_idx = training_split_indices(size(theta_grid, 1), opts.split_seed)
    panel_idx = nearest_indices(theta_grid[val_idx, :], theta_true, min(opts.periods, length(val_idx)))
    panel_idx = val_idx[panel_idx]
    obs_sigma = Float64.(manifest["obs_sigma"])

    model_direct = load_hlt_model(INV_REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
    # The bridge surrogate dataset was generated on the OBC HLT state space.
    # Use the same model for ROM state propagation so state/shock dimensions
    # match the trained residual network.
    model_rom = load_hlt_model(INV_REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
    obs_idx_raw = indexin(observables, model_rom.var)
    state_idx_raw = indexin(state_names, model_rom.var)
    any(isnothing, obs_idx_raw) && error("Observables missing from ROM model.")
    any(isnothing, state_idx_raw) && error("States missing from ROM model.")
    obs_idx = Int.(obs_idx_raw)
    state_idx = Int.(state_idx_raw)
    theta_param_idx_raw = indexin(theta_names, model_rom.parameters)
    any(isnothing, theta_param_idx_raw) && error("Theta names missing from ROM model parameters.")
    theta_param_idx = Int.(theta_param_idx_raw)

    base_params = copy(model_rom.parameter_values)
    baseline_theta = Float64[base_params[i] for i in theta_param_idx]
    baseline_rom = solve_matrix_rom_context!(model_rom, theta_names, baseline_theta, state_idx, obs_idx)
    truth_rom = rom_mode == :baseline ? baseline_rom :
        solve_matrix_rom_context!(model_rom, theta_names, theta_true, state_idx, obs_idx)
    frozen = bundle["frozen"]
    sur_meta = get(bundle, "meta", Dict{String,Any}())
    d_state = length(state_idx)
    d_eps = length(truth_rom.shock_sigmas)
    panel_initial_state = Float64.(X[1:d_state, panel_idx[1]])

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
    truth_surrogate_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
        truth_rom.rom_full_predict,
        residual_predict,
        state,
        shock_t,
        θ_local,
        d_obs;
        allow_full_residual = false,
    )

    obs_data = if opts.panel_mode == "heldout-one-step"
        Matrix{Float64}(Y[1:d_obs, panel_idx])
    elseif opts.panel_mode == "surrogate-rollout"
        size(X, 1) >= d_state + d_eps + length(theta_names) ||
            error("Dataset X has $(size(X, 1)) rows, expected at least state($d_state)+shock($d_eps)+theta($(length(theta_names))).")
        state_t = copy(panel_initial_state)
        shock_panel = Matrix{Float64}(X[d_state + 1:d_state + d_eps, panel_idx])
        out = Matrix{Float64}(undef, d_obs, length(panel_idx))
        for t in axes(out, 2)
            obs_t, state_next = truth_surrogate_predict(state_t, shock_panel[:, t], theta_true)
            out[:, t] .= Float64.(obs_t)
            state_t = Float64.(state_next)
        end
        serialize(joinpath(opts.out_dir, "synthetic_panel.jls"), Dict{String,Any}(
            "panel_mode" => opts.panel_mode,
            "dgp" => "ROM1-residual surrogate dynamic rollout",
            "truth_index" => truth_idx,
            "theta_true" => theta_true,
            "panel_idx" => panel_idx,
            "initial_state" => panel_initial_state,
            "shock_panel" => shock_panel,
            "obs_data" => out,
            "observables" => observables,
            "state_names" => state_names,
            "shock_names" => Symbol.(get(meta, "shock_names", Symbol[])),
        ))
        out
    elseif opts.panel_mode == "direct-sep-rollout"
        size(X, 1) >= d_state + d_eps + length(theta_names) ||
            error("Dataset X has $(size(X, 1)) rows, expected at least state($d_state)+shock($d_eps)+theta($(length(theta_names))).")
        shock_panel = Matrix{Float64}(X[d_state + 1:d_state + d_eps, panel_idx])
        write_theta!(model_direct, theta_names, theta_true)
        state0_full = full_state_from_subset(truth_rom.nsss, state_idx, panel_initial_state)
        local res
        try
            res = MacroModelling.simulate_sep_extended_path(
                model_direct;
                periods = size(shock_panel, 2),
                initial_state = state0_full,
                shocks = shock_panel,
                burn_in = 0,
                sep_horizon = opts.sep_horizon,
                sep_order = 1,
                sep_nnodes = 3,
                sep_sparse_tree = true,
                sep_maxit = opts.sep_maxit,
                sep_tol = 1.0e-7,
                sep_accept_tol = 1.0e-2,
                sep_linear_solver = :qr,
                sep_fallback_solver = :normal_equations,
                sep_recovery = true,
                sep_recovery_scales = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
                shock_scaling = :none,
                silent = true,
            )
        catch err
            error("Direct SEP panel generation failed: $(sprint(showerror, err))")
        end
        Bool(res.errorflag) && error("Direct SEP panel generation returned errorflag=true at period $(res.failure_period).")
        sim = Float64.(Array(res.simulation))
        size(sim, 2) >= size(shock_panel, 2) + 1 ||
            error("Direct SEP panel returned too few columns: $(size(sim, 2))")
        out = Matrix{Float64}(sim[obs_idx, 2:(size(shock_panel, 2) + 1)])
        serialize(joinpath(opts.out_dir, "synthetic_panel.jls"), Dict{String,Any}(
            "panel_mode" => opts.panel_mode,
            "dgp" => "direct SEP dynamic rollout",
            "truth_index" => truth_idx,
            "theta_true" => theta_true,
            "panel_idx" => panel_idx,
            "initial_state" => panel_initial_state,
            "initial_state_full" => state0_full,
            "shock_panel" => shock_panel,
            "obs_data" => out,
            "observables" => observables,
            "state_names" => state_names,
            "shock_names" => Symbol.(get(meta, "shock_names", Symbol[])),
            "sep_errors" => hasproperty(res, :sep_errors) ? res.sep_errors : Float64[],
            "sep_recovery_log" => hasproperty(res, :sep_recovery_log) ? res.sep_recovery_log : Any[],
        ))
        out
    else
        error("Unknown --panel-mode=$(opts.panel_mode). Supported modes: heldout-one-step, surrogate-rollout, direct-sep-rollout.")
    end
    obs_ka = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))
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
            rom_ctx = rom_mode == :baseline ? baseline_rom :
                solve_matrix_rom_context!(model_rom, theta_names, theta, state_idx, obs_idx)
            surrogate_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
                rom_ctx.rom_full_predict,
                residual_predict,
                state,
                shock_t,
                θ_local,
                d_obs;
                allow_full_residual = false,
            )

            if opts.direct_objective == "exact-inversion"
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
            elseif opts.direct_objective == "common-measurement-error"
                direct_predict = make_direct_sep_eval_predict(
                    model_direct,
                    theta_names,
                    obs_idx,
                    state_idx,
                    rom_ctx.nsss,
                    rom_ctx.rom_predict_tuple,
                    opts,
                )
                ll_direct, _ = MacroModelling.inversion_loglik_per_period(
                    rom_ctx.rom_predict_tuple,
                    panel_initial_state,
                    theta,
                    obs_data,
                    obs_sigma,
                    rom_ctx.shock_sigmas;
                    eval_predict_fn = direct_predict,
                    maxit = opts.inversion_maxit,
                    tol = opts.inversion_tol,
                    lambda = opts.inversion_lambda,
                )
                direct_ll[k] = sum(ll_direct)
            else
                error("Unknown --direct-objective=$(opts.direct_objective). Supported: exact-inversion, common-measurement-error.")
            end
            statuses[k] = isfinite(direct_ll[k]) && direct_ll[k] > -1.0e9 ? "ok" : "direct_failure"

            ll_sur, _ = MacroModelling.inversion_loglik_per_period(
                rom_ctx.rom_predict_tuple,
                panel_initial_state,
                theta,
                obs_data,
                obs_sigma,
                rom_ctx.shock_sigmas;
                eval_predict_fn = surrogate_predict,
                maxit = opts.inversion_maxit,
                tol = opts.inversion_tol,
                lambda = opts.inversion_lambda,
            )
            surrogate_ll[k] = sum(ll_sur)

            ll_rom, _ = MacroModelling.inversion_loglik_per_period(
                rom_ctx.rom_predict_tuple,
                panel_initial_state,
                theta,
                obs_data,
                obs_sigma,
                rom_ctx.shock_sigmas;
                maxit = opts.inversion_maxit,
                tol = opts.inversion_tol,
                lambda = opts.inversion_lambda,
            )
            rom1_ll[k] = sum(ll_rom)
        catch err
            statuses[k] = "error: " * sprint(showerror, err)
            direct_ll[k] = -Inf
            surrogate_ll[k] = -Inf
            rom1_ll[k] = -Inf
        end
        elapsed_direct[k] = time() - t0

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
    dynamic_rollout_smoke = opts.panel_mode == "surrogate-rollout"
    pass = all_overlap &&
        count(==("ok"), statuses) >= min(3, length(statuses)) &&
        (dynamic_rollout_smoke || surface_centered_rmse_sur < surface_centered_rmse_rom)

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
        "smoke_pass" => pass,
        "dynamic_rollout_smoke" => dynamic_rollout_smoke,
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
        println(io, "- Dynamic rollout smoke: `$(dynamic_rollout_smoke)`")
        if dynamic_rollout_smoke
            println(io, "- Dynamic rollout smoke pass: `$(pass)`")
            println(io, "- Accuracy comparison pass: not assessed by smoke criteria")
        else
            println(io, "- Comparison pass: `$(pass)`")
        end
        dynamic_rollout_smoke && println(io, "- Synthetic panel: `$(joinpath(opts.out_dir, "synthetic_panel.jls"))`")
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
