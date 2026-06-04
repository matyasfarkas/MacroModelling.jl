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
    panel_selection::String = "nearest-theta"
    panel_shock_weight::Float64 = 1.0
    initial_state_mode::String = "panel-feature"
    direct_panel_solver::String = "full-path"
    direct_objective::String = "exact-inversion"
    surrogate_objective::String = "batch-obs-residual"
    split_seed::Int = 20260527
    obs_sigma_scale::Float64 = 1.0
    obs_sigma_floor::Float64 = 1.0e-3
    direct_eval_points::Int = 25
    profile_repeats::Int = 1
    profile_direct_repeats::Int = 1
    profile_fast_repeats::Int = 1
    profile_warmup::Int = 0
    profile_projection_periods::Int = 265
    dynamic_calibration_train_points::Int = 0
    dynamic_ridge_lambda::Float64 = 1.0e-4
    dynamic_feature_mode::String = "state-shock-theta-time"
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

meta_symbol(x, default::Symbol) = x === nothing ? default : (x isa Symbol ? x : Symbol(x))

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
        panel_selection = parse_arg(args, "--panel-selection", opts.panel_selection),
        panel_shock_weight = parse(Float64, parse_arg(args, "--panel-shock-weight", string(opts.panel_shock_weight))),
        initial_state_mode = parse_arg(args, "--initial-state-mode", opts.initial_state_mode),
        direct_panel_solver = parse_arg(args, "--direct-panel-solver", opts.direct_panel_solver),
        direct_objective = parse_arg(args, "--direct-objective", opts.direct_objective),
        surrogate_objective = parse_arg(args, "--surrogate-objective", opts.surrogate_objective),
        split_seed = parse(Int, parse_arg(args, "--split-seed", string(opts.split_seed))),
        obs_sigma_scale = parse(Float64, parse_arg(args, "--obs-sigma-scale", string(opts.obs_sigma_scale))),
        obs_sigma_floor = parse(Float64, parse_arg(args, "--obs-sigma-floor", string(opts.obs_sigma_floor))),
        direct_eval_points = parse(Int, parse_arg(args, "--direct-eval-points", string(opts.direct_eval_points))),
        profile_repeats = parse(Int, parse_arg(args, "--profile-repeats", string(opts.profile_repeats))),
        profile_direct_repeats = parse(Int, parse_arg(args, "--profile-direct-repeats", parse_arg(args, "--profile-repeats", string(opts.profile_direct_repeats)))),
        profile_fast_repeats = parse(Int, parse_arg(args, "--profile-fast-repeats", parse_arg(args, "--profile-repeats", string(opts.profile_fast_repeats)))),
        profile_warmup = parse(Int, parse_arg(args, "--profile-warmup", string(opts.profile_warmup))),
        profile_projection_periods = parse(Int, parse_arg(args, "--profile-projection-periods", string(opts.profile_projection_periods))),
        dynamic_calibration_train_points = parse(Int, parse_arg(args, "--dynamic-calibration-train-points", string(opts.dynamic_calibration_train_points))),
        dynamic_ridge_lambda = parse(Float64, parse_arg(args, "--dynamic-ridge-lambda", string(opts.dynamic_ridge_lambda))),
        dynamic_feature_mode = parse_arg(args, "--dynamic-feature-mode", opts.dynamic_feature_mode),
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

function choose_panel_indices(theta_grid::Matrix{Float64},
                              X::Matrix{Float64},
                              theta_true::Vector{Float64},
                              train_idx::Vector{Int},
                              val_idx::Vector{Int},
                              opts::InversionBridgeOptions,
                              d_state::Int,
                              d_eps::Int)
    n_keep = min(opts.periods, size(theta_grid, 1))
    candidates = copy(val_idx)
    if length(candidates) < n_keep
        candidates = vcat(candidates, train_idx[1:min(length(train_idx), n_keep - length(candidates))])
    end
    isempty(candidates) && error("No panel candidates available.")

    scales = vec(maximum(theta_grid, dims = 1) .- minimum(theta_grid, dims = 1))
    scales[scales .<= sqrt(eps(Float64))] .= 1.0
    theta_scores = [sum(((theta_grid[i, :] .- theta_true) ./ scales) .^ 2) for i in candidates]

    if opts.panel_selection == "nearest-theta"
        order = sortperm(theta_scores)
    elseif opts.panel_selection == "small-shocks" || opts.panel_selection == "small-shocks-nearest-theta"
        d_state > 0 && d_eps > 0 && size(X, 1) >= d_state + d_eps ||
            error("--panel-selection=$(opts.panel_selection) requires state and shock features in X.")
        shock_block = X[(d_state + 1):(d_state + d_eps), candidates]
        shock_norms = vec(sum(abs2, shock_block, dims = 1))
        scale_shock = maximum(shock_norms)
        scale_shock <= sqrt(eps(Float64)) && (scale_shock = 1.0)
        shock_scores = shock_norms ./ scale_shock
        if opts.panel_selection == "small-shocks"
            order = sortperm(shock_scores)
        else
            theta_scale = maximum(theta_scores)
            theta_scale <= sqrt(eps(Float64)) && (theta_scale = 1.0)
            combined = theta_scores ./ theta_scale .+ opts.panel_shock_weight .* shock_scores
            order = sortperm(combined)
        end
    else
        error("Unknown --panel-selection=$(opts.panel_selection). Supported: nearest-theta, small-shocks, small-shocks-nearest-theta.")
    end
    return candidates[order[1:min(n_keep, length(order))]]
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

function fmt_seconds(x)
    isfinite(Float64(x)) || return "NaN"
    return @sprintf("%.3f", Float64(x))
end

function finite_values(x)
    return Float64[v for v in x if isfinite(Float64(v))]
end

function mean_finite(x)
    vals = finite_values(x)
    return isempty(vals) ? NaN : mean(vals)
end

function median_finite(x)
    vals = sort(finite_values(x))
    isempty(vals) && return NaN
    n = length(vals)
    isodd(n) ? vals[(n + 1) ÷ 2] : 0.5 * (vals[n ÷ 2] + vals[n ÷ 2 + 1])
end

function ratio_or_nan(num, den)
    num_f = Float64(num)
    den_f = Float64(den)
    return isfinite(num_f) && isfinite(den_f) && den_f > 0 ? num_f / den_f : NaN
end

function timed_eval(fn::Function, repeats::Int, warmup::Int)
    repeats = max(repeats, 1)
    warmup = max(warmup, 0)
    for _ in 1:warmup
        fn()
    end
    times = Vector{Float64}(undef, repeats)
    value = nothing
    for r in 1:repeats
        t0 = time()
        value = fn()
        times[r] = time() - t0
    end
    return value, mean(times), times
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

function draw_hlt_shocks(rng::AbstractRNG,
                         model,
                         total_periods::Int,
                         shock_scaling::Symbol,
                         shock_scale::Float64)
    shock_names = model.exo
    shocks = zeros(Float64, length(shock_names), total_periods)
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    isempty(structural_idx) && return shocks
    sigmas = if shock_scaling == :parameter
        [MacroModelling.sep_irf_shock_std(model, shock_names[idx]) for idx in structural_idx]
    else
        ones(Float64, length(structural_idx))
    end
    sigmas .*= shock_scale
    shocks[structural_idx, :] .= Diagonal(sigmas) * randn(rng, length(structural_idx), total_periods)
    return shocks
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

function write_speed_profile_csv(path::String,
                                 candidate_idx::Vector{Int},
                                 statuses::Vector{String},
                                 direct_ll::Vector{Float64},
                                 surrogate_ll::Vector{Float64},
                                 rom1_ll::Vector{Float64},
                                 elapsed_rom_setup::Vector{Float64},
                                 elapsed_direct::Vector{Float64},
                                 elapsed_surrogate::Vector{Float64},
                                 elapsed_rom1::Vector{Float64},
                                 elapsed_total::Vector{Float64},
                                 periods::Int)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join([
            "local_candidate",
            "grid_index",
            "status",
            "direct_ll",
            "surrogate_ll",
            "rom1_ll",
            "rom_setup_seconds",
            "direct_seconds",
            "surrogate_seconds",
            "rom1_seconds",
            "total_seconds",
            "direct_seconds_per_period",
            "surrogate_seconds_per_period",
            "rom1_seconds_per_period",
            "direct_to_surrogate_speedup",
            "direct_to_rom1_speedup",
        ], ","))
        for k in eachindex(candidate_idx)
            direct_per = ratio_or_nan(elapsed_direct[k], periods)
            sur_per = ratio_or_nan(elapsed_surrogate[k], periods)
            rom_per = ratio_or_nan(elapsed_rom1[k], periods)
            row = [
                string(k),
                string(candidate_idx[k]),
                replace(statuses[k], "," => ";"),
                fmt(direct_ll[k]),
                fmt(surrogate_ll[k]),
                fmt(rom1_ll[k]),
                fmt_seconds(elapsed_rom_setup[k]),
                fmt_seconds(elapsed_direct[k]),
                fmt_seconds(elapsed_surrogate[k]),
                fmt_seconds(elapsed_rom1[k]),
                fmt_seconds(elapsed_total[k]),
                fmt_seconds(direct_per),
                fmt_seconds(sur_per),
                fmt_seconds(rom_per),
                fmt(ratio_or_nan(elapsed_direct[k], elapsed_surrogate[k])),
                fmt(ratio_or_nan(elapsed_direct[k], elapsed_rom1[k])),
            ]
            println(io, join(row, ","))
        end
    end
end

function speed_summary(elapsed_direct::Vector{Float64},
                       elapsed_surrogate::Vector{Float64},
                       elapsed_rom1::Vector{Float64},
                       elapsed_total::Vector{Float64},
                       periods::Int,
                       projection_periods::Int,
                       projection_points::Int)
    direct_med = median_finite(elapsed_direct)
    sur_med = median_finite(elapsed_surrogate)
    rom_med = median_finite(elapsed_rom1)
    total_med = median_finite(elapsed_total)
    direct_mean = mean_finite(elapsed_direct)
    sur_mean = mean_finite(elapsed_surrogate)
    rom_mean = mean_finite(elapsed_rom1)
    scale = periods > 0 ? projection_periods / periods : NaN
    direct_projection_hours = isfinite(direct_med) ? direct_med * scale * projection_points / 3600 : NaN
    surrogate_projection_hours = isfinite(sur_med) ? sur_med * scale * projection_points / 3600 : NaN
    rom_projection_hours = isfinite(rom_med) ? rom_med * scale * projection_points / 3600 : NaN
    return Dict{String,Any}(
        "periods_profiled" => periods,
        "projection_periods" => projection_periods,
        "projection_points" => projection_points,
        "direct_mean_seconds" => direct_mean,
        "surrogate_mean_seconds" => sur_mean,
        "rom1_mean_seconds" => rom_mean,
        "direct_median_seconds" => direct_med,
        "surrogate_median_seconds" => sur_med,
        "rom1_median_seconds" => rom_med,
        "total_median_seconds" => total_med,
        "direct_median_seconds_per_period" => ratio_or_nan(direct_med, periods),
        "surrogate_median_seconds_per_period" => ratio_or_nan(sur_med, periods),
        "rom1_median_seconds_per_period" => ratio_or_nan(rom_med, periods),
        "direct_to_surrogate_speedup_median" => ratio_or_nan(direct_med, sur_med),
        "direct_to_rom1_speedup_median" => ratio_or_nan(direct_med, rom_med),
        "projected_direct_hours" => direct_projection_hours,
        "projected_surrogate_hours" => surrogate_projection_hours,
        "projected_rom1_hours" => rom_projection_hours,
    )
end

function split_predict_output(y, d_obs::Int; label::AbstractString)
    if y isa Tuple && length(y) == 2
        return y[1], y[2]
    end
    return MacroModelling.split_observation_state(y, d_obs; label = label)
end

function rollout_path_cache(predict_fn::Function,
                            s0::AbstractVector,
                            shocks::AbstractMatrix,
                            theta::AbstractVector,
                            d_obs::Int)
    T = size(shocks, 2)
    d_state = length(s0)
    states = Matrix{Float64}(undef, d_state, T)
    rom_obs = Matrix{Float64}(undef, d_obs, T)
    state = Float64.(s0)
    for t in 1:T
        states[:, t] .= state
        y_full = predict_fn(state, shocks[:, t], theta)
        obs_t, state_next = split_predict_output(
            y_full,
            d_obs;
            label = "dynamic calibration ROM1 output",
        )
        rom_obs[:, t] .= Float64.(obs_t)
        state = Float64.(state_next)
    end
    return states, rom_obs
end

function dynamic_feature_matrix(states::AbstractMatrix,
                                shocks::AbstractMatrix,
                                theta::AbstractVector,
                                mode::AbstractString)
    T = size(states, 2)
    size(shocks, 2) == T ||
        error("Dynamic feature states/shocks length mismatch: $(size(states,2)) vs $(size(shocks,2)).")
    theta_block = repeat(reshape(Float64.(theta), :, 1), 1, T)
    if mode == "state-shock-theta"
        return vcat(Float64.(states), Float64.(shocks), theta_block)
    elseif mode == "shock-theta"
        return vcat(Float64.(shocks), theta_block)
    elseif mode == "state-shock-theta-time"
        time_row = reshape(T == 1 ? [0.0] : collect(range(0.0, 1.0; length = T)), 1, T)
        return vcat(Float64.(states), Float64.(shocks), theta_block, time_row)
    else
        error("Unknown --dynamic-feature-mode=$mode. Supported: state-shock-theta-time, state-shock-theta, shock-theta.")
    end
end

function fit_dynamic_ridge_residual(X_train::AbstractMatrix,
                                    Y_train::AbstractMatrix,
                                    λ::Float64)
    size(X_train, 2) == size(Y_train, 2) ||
        error("Dynamic ridge X/Y sample mismatch: $(size(X_train,2)) vs $(size(Y_train,2)).")
    size(X_train, 2) > 0 || error("Dynamic ridge requires at least one training sample.")
    λ >= 0 || error("Dynamic ridge lambda must be nonnegative.")
    μ = vec(mean(X_train, dims = 2))
    σ = vec(Statistics.std(X_train, dims = 2; corrected = false))
    σ[σ .<= sqrt(eps(Float64))] .= 1.0
    Xz = (Float64.(X_train) .- μ) ./ σ
    Z = vcat(ones(Float64, 1, size(Xz, 2)), Xz)
    penalty = Diagonal(vcat(0.0, fill(λ, size(Xz, 1))))
    coef = (Float64.(Y_train) * Z') / (Z * Z' + penalty)
    return (
        coef = coef,
        μ = μ,
        σ = σ,
        λ = λ,
    )
end

function predict_dynamic_ridge(model, X::AbstractMatrix)
    size(X, 1) == length(model.μ) ||
        error("Dynamic ridge feature dimension mismatch: got $(size(X,1)), expected $(length(model.μ)).")
    Xz = (Float64.(X) .- model.μ) ./ model.σ
    Z = vcat(ones(Float64, 1, size(Xz, 2)), Xz)
    return model.coef * Z
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
    opts.panel_mode in ("heldout-one-step", "surrogate-rollout", "direct-sep-rollout") ||
        error("Unknown --panel-mode=$(opts.panel_mode). Supported modes: heldout-one-step, surrogate-rollout, direct-sep-rollout.")
    opts.panel_selection in ("nearest-theta", "small-shocks", "small-shocks-nearest-theta") ||
        error("Unknown --panel-selection=$(opts.panel_selection). Supported modes: nearest-theta, small-shocks, small-shocks-nearest-theta.")
    opts.initial_state_mode in ("panel-feature", "truth-feature", "steady-state") ||
        error("Unknown --initial-state-mode=$(opts.initial_state_mode). Supported modes: panel-feature, truth-feature, steady-state.")
    opts.direct_panel_solver in ("full-path", "sequential-one-step", "generator-path") ||
        error("Unknown --direct-panel-solver=$(opts.direct_panel_solver). Supported modes: full-path, sequential-one-step, generator-path.")
    opts.direct_objective in ("exact-inversion", "common-measurement-error") ||
        error("Unknown --direct-objective=$(opts.direct_objective). Supported modes: exact-inversion, common-measurement-error.")
    opts.surrogate_objective in ("batch-obs-residual", "step-predict", "dynamic-ridge-residual") ||
        error("Unknown --surrogate-objective=$(opts.surrogate_objective). Supported modes: batch-obs-residual, step-predict, dynamic-ridge-residual.")
    opts.profile_repeats >= 1 || error("--profile-repeats must be positive.")
    opts.profile_direct_repeats >= 1 || error("--profile-direct-repeats must be positive.")
    opts.profile_fast_repeats >= 1 || error("--profile-fast-repeats must be positive.")
    opts.profile_warmup >= 0 || error("--profile-warmup must be nonnegative.")
    opts.profile_projection_periods > 0 || error("--profile-projection-periods must be positive.")
    opts.dynamic_calibration_train_points >= 0 || error("--dynamic-calibration-train-points must be nonnegative.")
    opts.dynamic_ridge_lambda >= 0 || error("--dynamic-ridge-lambda must be nonnegative.")
    opts.dynamic_feature_mode in ("state-shock-theta-time", "state-shock-theta", "shock-theta") ||
        error("Unknown --dynamic-feature-mode=$(opts.dynamic_feature_mode). Supported modes: state-shock-theta-time, state-shock-theta, shock-theta.")

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
    d_state = length(get(meta, "state_names", Symbol[]))
    d_eps = length(get(meta, "shock_names", Symbol[]))
    rom_mode_raw = get(meta, "rom_mode", :baseline)
    rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)
    d_obs > 0 || error("Dataset metadata missing observables.")
    n = size(theta_grid, 1)
    n == size(X, 2) == size(Y, 2) == size(Y_rom1, 2) ||
        error("Dataset grid/sample count mismatch.")
    truth_idx, train_idx, val_idx = choose_truth_index(theta_grid, specs, opts)
    obs_sigma = max.(opts.obs_sigma_scale .* vec(Statistics.std(Y[1:d_obs, :], dims = 2; corrected = false)), opts.obs_sigma_floor)
    direct_eval_points = min(opts.direct_eval_points, n)
    panel_idx = choose_panel_indices(theta_grid, X, vec(theta_grid[truth_idx, :]), train_idx, val_idx, opts, d_state, d_eps)
    rom_desc = rom_mode == :baseline ? "baseline ROM1" : "candidate-specific ROM1"

    surrogate_desc = opts.surrogate_objective == "batch-obs-residual" ?
        "the HLT estimator's ROM1 state propagation with batched obs-only NN residual correction" :
        opts.surrogate_objective == "dynamic-ridge-residual" ?
        "a held-out dynamic ridge residual calibrated on direct SEP one-step residuals along ROM1 inversion paths" :
        "a sequential ROM1-residual step predictor"

    execution_plan = opts.panel_mode == "surrogate-rollout" ? [
        "construct a deterministic dynamic HLT observation panel by rolling the trained ROM1-residual bridge at one truth theta",
        "recover shocks with the $(rom_desc) inversion filter under each candidate theta",
        "evaluate the direct SEP inversion objective on the same observation panel and parameter grid",
        "evaluate the ROM1-residual surrogate objective using $(surrogate_desc)",
        "profile direct SEP, surrogate, and ROM1 inversion likelihood runtimes on the same dynamic panels",
        "compare direct SEP, ROM1, and surrogate posterior surfaces by means, intervals, MAP ranking, and surface RMSE",
    ] : opts.panel_mode == "direct-sep-rollout" ? [
        "construct a deterministic dynamic HLT observation panel by rolling direct SEP at one truth theta",
        "recover shocks with the $(rom_desc) inversion filter under each candidate theta",
        "evaluate a direct-SEP measurement-error objective using the same recovered-shock architecture as the surrogate",
        "evaluate ROM1 and ROM1-residual surrogate objectives with $(rom_desc) matrices and shock scales",
        "use $(surrogate_desc) for the surrogate objective",
        "profile direct SEP, surrogate, and ROM1 inversion likelihood runtimes on the same dynamic panels",
        "compare centered objective surfaces, posterior intervals, and local MAP ranking",
    ] : [
        "construct a short synthetic HLT observation panel from a held-out validation sequence",
        "recover shocks with the $(rom_desc) inversion filter under each candidate theta",
        "evaluate the direct SEP inversion objective on the same observation panel and parameter grid",
        "evaluate the ROM1-residual surrogate objective using $(surrogate_desc)",
        "profile direct SEP, surrogate, and ROM1 inversion likelihood runtimes on the same dynamic panels",
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
        "panel_selection" => opts.panel_selection,
        "panel_shock_weight" => opts.panel_shock_weight,
        "initial_state_mode" => opts.initial_state_mode,
        "direct_panel_solver" => opts.direct_panel_solver,
        "direct_objective" => opts.direct_objective,
        "surrogate_objective" => opts.surrogate_objective,
        "truth_in_training_split" => truth_idx in train_idx,
        "truth_in_validation_split" => truth_idx in val_idx,
        "theta_true" => vec(theta_grid[truth_idx, :]),
        "panel_indices" => panel_idx,
        "obs_sigma" => obs_sigma,
        "direct_eval_points" => direct_eval_points,
        "profile_repeats" => opts.profile_repeats,
        "profile_direct_repeats" => opts.profile_direct_repeats,
        "profile_fast_repeats" => opts.profile_fast_repeats,
        "profile_warmup" => opts.profile_warmup,
        "profile_projection_periods" => opts.profile_projection_periods,
        "dynamic_calibration_train_points" => opts.dynamic_calibration_train_points,
        "dynamic_ridge_lambda" => opts.dynamic_ridge_lambda,
        "dynamic_feature_mode" => opts.dynamic_feature_mode,
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
            "dynamic_ridge_residual.jls",
            "speed_profile.csv",
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
        println(io, "- Panel selection: `$(manifest["panel_selection"])`")
        println(io, "- Initial state mode: `$(manifest["initial_state_mode"])`")
        println(io, "- Direct panel solver: `$(manifest["direct_panel_solver"])`")
        println(io, "- Direct objective: `$(manifest["direct_objective"])`")
        println(io, "- Surrogate objective: `$(manifest["surrogate_objective"])`")
        if manifest["surrogate_objective"] == "dynamic-ridge-residual"
            println(io, "- Dynamic calibration train points: `$(manifest["dynamic_calibration_train_points"])`")
            println(io, "- Dynamic ridge lambda: `$(manifest["dynamic_ridge_lambda"])`")
            println(io, "- Dynamic feature mode: `$(manifest["dynamic_feature_mode"])`")
        end
        println(io, "- Profile repeats direct/fast/warmup: `$(manifest["profile_direct_repeats"]) / $(manifest["profile_fast_repeats"]) / $(manifest["profile_warmup"])`")
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
    train_idx, val_idx = training_split_indices(size(theta_grid, 1), opts.split_seed)
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
    panel_idx = choose_panel_indices(theta_grid, X, theta_true, train_idx, val_idx, opts, d_state, d_eps)
    panel_initial_state = if opts.initial_state_mode == "panel-feature"
        Float64.(X[1:d_state, panel_idx[1]])
    elseif opts.initial_state_mode == "truth-feature"
        Float64.(X[1:d_state, truth_idx])
    elseif opts.initial_state_mode == "steady-state"
        Float64.(truth_rom.nsss[state_idx])
    else
        error("Unknown --initial-state-mode=$(opts.initial_state_mode).")
    end

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
        function batch_nn_residual(X_nn::AbstractMatrix)
            d_prefix = d_state + d_eps
            T_batch = size(X_nn, 2)
            X_padded = Matrix{eltype(X_nn)}(undef, frozen.d_in, T_batch)
            X_padded[1:d_prefix, :] .= X_nn[1:d_prefix, :]
            for t in 1:T_batch
                X_padded[(d_prefix + 1):end, t] .= pad_theta(X_nn[(d_prefix + 1):end, t])
            end
            return predict_frozen_batch(frozen, X_padded)
        end
        function single_nn_residual(x_nn::AbstractVector)
            d_prefix = d_state + d_eps
            x_padded = vcat(x_nn[1:d_prefix], pad_theta(x_nn[(d_prefix + 1):end]))
            return predict_frozen(frozen, x_padded)
        end
    else
        batch_nn_residual = (X_nn::AbstractMatrix) -> predict_frozen_batch(frozen, X_nn)
        single_nn_residual = (x_nn::AbstractVector) -> predict_frozen(frozen, x_nn)
    end
    residual_predict = (state, shock_t, θ_local) -> single_nn_residual(vcat(state, shock_t, θ_local))[1:d_obs]
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
        sep_errors = Float64[]
        sep_recovery_log = Any[]
        native_panel_info = Dict{String,Any}()
        out = Matrix{Float64}(undef, d_obs, size(shock_panel, 2))
        if opts.direct_panel_solver == "generator-path"
            native_burn_in = Int(get(meta, "burn_in", 1))
            native_seed_vec = get(meta, "theta_seeds", Int[])
            native_seed = length(native_seed_vec) >= truth_idx ? Int(native_seed_vec[truth_idx]) : opts.split_seed
            native_shock_scale = Float64(get(meta, "shock_scale", 0.05))
            native_shock_scaling = meta_symbol(get(meta, "shock_scaling", :parameter), :parameter)
            native_fallback_solver = get(meta, "sep_fallback_solver", nothing)
            native_fallback_solver = native_fallback_solver == :none ? nothing : native_fallback_solver
            native_total_periods = opts.periods + native_burn_in
            native_panel_info = Dict{String,Any}(
                "seed" => native_seed,
                "burn_in" => native_burn_in,
                "shock_scale" => native_shock_scale,
                "shock_scaling" => String(native_shock_scaling),
                "total_periods_drawn" => native_total_periods,
            )
            shocks_native = draw_hlt_shocks(
                MersenneTwister(native_seed),
                model_direct,
                native_total_periods,
                native_shock_scaling,
                native_shock_scale,
            )
            local res
            try
                res = MacroModelling.simulate_sep_extended_path(
                    model_direct;
                    periods = opts.periods,
                    burn_in = native_burn_in,
                    shocks = shocks_native,
                    random_seed = native_seed,
                    sep_horizon = opts.sep_horizon,
                    sep_order = Int(get(meta, "sep_order", 1)),
                    sep_nnodes = Int(get(meta, "sep_nnodes", 3)),
                    sep_sparse_tree = Bool(get(meta, "sep_sparse_tree", true)),
                    sep_maxit = opts.sep_maxit,
                    sep_tol = Float64(get(meta, "sep_tol", 1.0e-5)),
                    sep_accept_tol = get(meta, "sep_accept_tol", nothing),
                    sep_linear_solver = meta_symbol(get(meta, "sep_linear_solver", :normal_equations), :normal_equations),
                    sep_fallback_solver = native_fallback_solver,
                    sep_stall_iters = Int(get(meta, "sep_stall_iters", 25)),
                    sep_stall_rel_tol = Float64(get(meta, "sep_stall_rel_tol", 1.0e-4)),
                    sep_stall_abs_tol = Float64(get(meta, "sep_stall_abs_tol", 1.0e-10)),
                    sep_line_search = Bool(get(meta, "sep_line_search", true)),
                    sep_line_search_maxit = Int(get(meta, "sep_line_search_maxit", 6)),
                    sep_line_search_factor = Float64(get(meta, "sep_line_search_factor", 0.5)),
                    sep_line_search_min_alpha = Float64(get(meta, "sep_line_search_min_alpha", 1.0e-4)),
                    sep_lm_lambda = Float64(get(meta, "sep_lm_lambda", 1.0e-8)),
                    sep_lm_lambda_scale = Float64(get(meta, "sep_lm_lambda_scale", 10.0)),
                    sep_lm_lambda_min = Float64(get(meta, "sep_lm_lambda_min", 1.0e-12)),
                    sep_lm_lambda_max = Float64(get(meta, "sep_lm_lambda_max", 1.0e4)),
                    shock_scaling = native_shock_scaling,
                    silent = true,
                )
            catch err
                error("Direct SEP generator-path panel generation failed: $(sprint(showerror, err))")
            end
            Bool(res.errorflag) && error("Direct SEP generator-path panel generation returned errorflag=true at period $(res.failure_period).")
            sim = Float64.(Array(res.simulation))
            shocks_res = Float64.(Array(res.shocks))
            size(sim, 2) >= opts.periods + 1 ||
                error("Direct SEP generator-path panel returned too few simulation columns: $(size(sim, 2))")
            size(shocks_res, 2) >= opts.periods ||
                error("Direct SEP generator-path panel returned too few shock columns: $(size(shocks_res, 2))")
            panel_initial_state = Float64.(sim[state_idx, 1])
            state0_full = Float64.(sim[:, 1])
            shock_panel = Matrix{Float64}(shocks_res[:, 1:opts.periods])
            out .= Matrix{Float64}(sim[obs_idx, 2:(opts.periods + 1)])
            sep_errors = hasproperty(res, :sep_errors) ? Float64.(res.sep_errors) : Float64[]
            sep_recovery_log = hasproperty(res, :sep_recovery_log) ? res.sep_recovery_log : Any[]
        elseif opts.direct_panel_solver == "full-path"
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
            out .= Matrix{Float64}(sim[obs_idx, 2:(size(shock_panel, 2) + 1)])
            sep_errors = hasproperty(res, :sep_errors) ? Float64.(res.sep_errors) : Float64[]
            sep_recovery_log = hasproperty(res, :sep_recovery_log) ? res.sep_recovery_log : Any[]
        elseif opts.direct_panel_solver == "sequential-one-step"
            state_t = copy(panel_initial_state)
            for t in axes(shock_panel, 2)
                pred = direct_sep_predict_subset(
                    model_direct,
                    theta_names,
                    theta_true,
                    obs_idx,
                    state_idx,
                    truth_rom.nsss,
                    state_t,
                    shock_panel[:, t],
                    opts,
                )
                pred.ok || error("Sequential direct SEP panel generation failed at period $t: $(pred.message)")
                out[:, t] .= pred.obs
                state_t = Float64.(pred.state_next)
                push!(sep_errors, Float64(pred.sep_err))
            end
        else
            error("Unknown --direct-panel-solver=$(opts.direct_panel_solver).")
        end
        serialize(joinpath(opts.out_dir, "synthetic_panel.jls"), Dict{String,Any}(
            "panel_mode" => opts.panel_mode,
            "dgp" => "direct SEP dynamic rollout",
            "direct_panel_solver" => opts.direct_panel_solver,
            "initial_state_mode" => opts.initial_state_mode,
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
            "sep_errors" => sep_errors,
            "sep_recovery_log" => sep_recovery_log,
            "native_panel_info" => native_panel_info,
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
    elapsed_rom_setup = fill(NaN, length(candidate_idx))
    elapsed_direct = fill(NaN, length(candidate_idx))
    elapsed_surrogate = fill(NaN, length(candidate_idx))
    elapsed_rom1 = fill(NaN, length(candidate_idx))
    elapsed_total = fill(NaN, length(candidate_idx))
    path_states_cache = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(candidate_idx))
    path_shocks_cache = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(candidate_idx))
    path_rom_obs_cache = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(candidate_idx))
    dynamic_feature_cache = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(candidate_idx))
    dynamic_target_cache = Vector{Union{Nothing,Matrix{Float64}}}(nothing, length(candidate_idx))
    dynamic_target_status = fill("not_requested", length(candidate_idx))
    use_dynamic_ridge = opts.surrogate_objective == "dynamic-ridge-residual"
    for (k, idx) in enumerate(candidate_idx)
        theta = vec(theta_grid[idx, :])
        lp = prior_logpdf(theta, specs)
        isfinite(lp) || (statuses[k] = "prior_out_of_bounds"; continue)

        t_total = time()
        try
            t_rom = time()
            rom_ctx = rom_mode == :baseline ? baseline_rom :
                solve_matrix_rom_context!(model_rom, theta_names, theta, state_idx, obs_idx)
            elapsed_rom_setup[k] = time() - t_rom
            surrogate_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
                rom_ctx.rom_full_predict,
                residual_predict,
                state,
                shock_t,
                θ_local,
                d_obs;
                allow_full_residual = false,
            )

            direct_fn = if opts.direct_objective == "exact-inversion"
                params = direct_params(model_direct, theta_names, theta)
                () -> MacroModelling.get_loglikelihood(
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
                () -> begin
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
                    return sum(ll_direct)
                end
            else
                error("Unknown --direct-objective=$(opts.direct_objective). Supported: exact-inversion, common-measurement-error.")
            end
            direct_ll[k], elapsed_direct[k], _ = timed_eval(direct_fn, opts.profile_direct_repeats, opts.profile_warmup)
            statuses[k] = isfinite(direct_ll[k]) && direct_ll[k] > -1.0e9 ? "ok" : "direct_failure"

            rom1_fn = () -> begin
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
                return sum(ll_rom)
            end
            rom1_ll[k], elapsed_rom1[k], _ = timed_eval(rom1_fn, opts.profile_fast_repeats, opts.profile_warmup)

            ll_rom_path, shocks_rom_path = MacroModelling.inversion_loglik_per_period(
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
            isfinite(rom1_ll[k]) || (rom1_ll[k] = sum(ll_rom_path))
            states_path, rom_obs_path = rollout_path_cache(
                rom_ctx.rom_predict_tuple,
                panel_initial_state,
                shocks_rom_path,
                theta,
                d_obs,
            )
            path_states_cache[k] = states_path
            path_shocks_cache[k] = Float64.(shocks_rom_path)
            path_rom_obs_cache[k] = rom_obs_path

            if use_dynamic_ridge && statuses[k] == "ok"
                target = Matrix{Float64}(undef, d_obs, size(obs_data, 2))
                target_ok = true
                for t in axes(target, 2)
                    pred = direct_sep_predict_subset(
                        model_direct,
                        theta_names,
                        theta,
                        obs_idx,
                        state_idx,
                        rom_ctx.nsss,
                        states_path[:, t],
                        shocks_rom_path[:, t],
                        opts,
                    )
                    if !pred.ok
                        target_ok = false
                        dynamic_target_status[k] = "direct_residual_failure_t$t: $(pred.message)"
                        break
                    end
                    target[:, t] .= pred.obs .- rom_obs_path[:, t]
                end
                if target_ok
                    dynamic_feature_cache[k] = dynamic_feature_matrix(
                        states_path,
                        shocks_rom_path,
                        theta,
                        opts.dynamic_feature_mode,
                    )
                    dynamic_target_cache[k] = target
                    dynamic_target_status[k] = "ok"
                end
            end

            if !use_dynamic_ridge
                surrogate_fn = if opts.surrogate_objective == "batch-obs-residual"
                    () -> begin
                        ll_sur, _ = MacroModelling.inversion_loglik_per_period(
                            rom_ctx.rom_predict_tuple,
                            panel_initial_state,
                            theta,
                            obs_data,
                            obs_sigma,
                            rom_ctx.shock_sigmas;
                            batch_eval_residual_fn = batch_nn_residual,
                            maxit = opts.inversion_maxit,
                            tol = opts.inversion_tol,
                            lambda = opts.inversion_lambda,
                        )
                        return sum(ll_sur)
                    end
                elseif opts.surrogate_objective == "step-predict"
                    () -> begin
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
                        return sum(ll_sur)
                    end
                else
                    error("Unknown --surrogate-objective=$(opts.surrogate_objective). Supported: batch-obs-residual, step-predict, dynamic-ridge-residual.")
                end
                surrogate_ll[k], elapsed_surrogate[k], _ = timed_eval(surrogate_fn, opts.profile_fast_repeats, opts.profile_warmup)
            end
        catch err
            statuses[k] = "error: " * sprint(showerror, err)
            direct_ll[k] = -Inf
            surrogate_ll[k] = -Inf
            rom1_ll[k] = -Inf
        end
        elapsed_total[k] = time() - t_total

        println("$(k)/$(length(candidate_idx)) idx=$idx direct=$(fmt(direct_ll[k])) surrogate=$(fmt(surrogate_ll[k])) rom1=$(fmt(rom1_ll[k])) t_direct=$(fmt_seconds(elapsed_direct[k]))s t_sur=$(fmt_seconds(elapsed_surrogate[k]))s status=$(statuses[k])")
    end

    dynamic_calibration = Dict{String,Any}("enabled" => use_dynamic_ridge)
    dynamic_eval_mask = trues(length(candidate_idx))
    if use_dynamic_ridge
        usable = findall(i -> dynamic_target_status[i] == "ok" && dynamic_feature_cache[i] !== nothing, eachindex(candidate_idx))
        length(usable) >= 2 || error("Dynamic ridge residual requires at least two usable direct residual anchors; got $(length(usable)).")
        requested_train = opts.dynamic_calibration_train_points > 0 ?
            min(opts.dynamic_calibration_train_points, length(usable) - 1) :
            clamp(length(usable) ÷ 2, 1, length(usable) - 1)
        train_local = usable[1:2:end]
        if length(train_local) < requested_train
            train_local = unique(vcat(train_local, usable))
        end
        train_local = train_local[1:requested_train]
        eval_local = setdiff(usable, train_local)
        isempty(eval_local) && error("Dynamic ridge residual has no held-out anchors.")

        X_train = hcat([dynamic_feature_cache[i] for i in train_local]...)
        Y_train = hcat([dynamic_target_cache[i] for i in train_local]...)
        ridge_model = fit_dynamic_ridge_residual(X_train, Y_train, opts.dynamic_ridge_lambda)
        ridge_payload_path = joinpath(opts.out_dir, "dynamic_ridge_residual.jls")

        function dynamic_batch_residual(X_nn::AbstractMatrix)
            X_dyn = dynamic_feature_matrix(
                X_nn[1:d_state, :],
                X_nn[(d_state + 1):(d_state + d_eps), :],
                vec(X_nn[(d_state + d_eps + 1):end, 1]),
                opts.dynamic_feature_mode,
            )
            return predict_dynamic_ridge(ridge_model, X_dyn)
        end

        for k in eachindex(candidate_idx)
            statuses[k] == "ok" || continue
            theta = vec(theta_grid[candidate_idx[k], :])
            try
                t_rom = time()
                rom_ctx = rom_mode == :baseline ? baseline_rom :
                    solve_matrix_rom_context!(model_rom, theta_names, theta, state_idx, obs_idx)
                elapsed_rom_setup[k] = isfinite(elapsed_rom_setup[k]) ? elapsed_rom_setup[k] : time() - t_rom
                dynamic_fn = () -> begin
                    ll_sur, _ = MacroModelling.inversion_loglik_per_period(
                        rom_ctx.rom_predict_tuple,
                        panel_initial_state,
                        theta,
                        obs_data,
                        obs_sigma,
                        rom_ctx.shock_sigmas;
                        batch_eval_residual_fn = dynamic_batch_residual,
                        maxit = opts.inversion_maxit,
                        tol = opts.inversion_tol,
                        lambda = opts.inversion_lambda,
                    )
                    return sum(ll_sur)
                end
                surrogate_ll[k], elapsed_surrogate[k], _ = timed_eval(dynamic_fn, opts.profile_fast_repeats, opts.profile_warmup)
            catch err
                surrogate_ll[k] = -Inf
                statuses[k] = "dynamic_surrogate_error: " * sprint(showerror, err)
            end
        end

        dynamic_eval_mask .= false
        dynamic_eval_mask[eval_local] .= true
        train_pred = predict_dynamic_ridge(ridge_model, X_train)
        train_rmse = sqrt(mean((train_pred .- Y_train) .^ 2))
        X_eval = hcat([dynamic_feature_cache[i] for i in eval_local]...)
        Y_eval = hcat([dynamic_target_cache[i] for i in eval_local]...)
        eval_pred = predict_dynamic_ridge(ridge_model, X_eval)
        eval_rmse = sqrt(mean((eval_pred .- Y_eval) .^ 2))
        dynamic_calibration = Dict{String,Any}(
            "enabled" => true,
            "train_local_indices" => train_local,
            "eval_local_indices" => eval_local,
            "train_grid_indices" => candidate_idx[train_local],
            "eval_grid_indices" => candidate_idx[eval_local],
            "train_points" => length(train_local),
            "eval_points" => length(eval_local),
            "train_samples" => size(X_train, 2),
            "eval_samples" => size(X_eval, 2),
            "feature_mode" => opts.dynamic_feature_mode,
            "ridge_lambda" => opts.dynamic_ridge_lambda,
            "train_rmse" => train_rmse,
            "eval_rmse" => eval_rmse,
            "target_status" => dynamic_target_status,
            "ridge_payload" => ridge_payload_path,
        )
        serialize(ridge_payload_path, Dict{String,Any}(
            "model" => Dict{String,Any}(
                "coef" => ridge_model.coef,
                "mu" => ridge_model.μ,
                "sigma" => ridge_model.σ,
                "lambda" => ridge_model.λ,
            ),
            "calibration" => dynamic_calibration,
            "theta_names" => theta_names,
            "observables" => observables,
            "state_names" => state_names,
            "shock_names" => Symbol.(get(meta, "shock_names", Symbol[])),
        ))
        println("Dynamic ridge residual: train anchors=$(length(train_local)) heldout anchors=$(length(eval_local)) train_rmse=$(fmt(train_rmse)) heldout_rmse=$(fmt(eval_rmse))")
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
    ok = isfinite.(direct_post) .& isfinite.(surrogate_post) .& dynamic_eval_mask
    surface_rmse_sur = any(ok) ? sqrt(mean((surrogate_post[ok] .- direct_post[ok]) .^ 2)) : Inf
    surface_offset_sur = any(ok) ? mean(surrogate_post[ok] .- direct_post[ok]) : NaN
    surface_centered_rmse_sur = any(ok) ? sqrt(mean(((surrogate_post[ok] .- direct_post[ok]) .- surface_offset_sur) .^ 2)) : Inf
    ok_rom = isfinite.(direct_post) .& isfinite.(rom1_post) .& dynamic_eval_mask
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

    speed_csv_path = joinpath(opts.out_dir, "speed_profile.csv")
    write_speed_profile_csv(
        speed_csv_path,
        candidate_idx,
        statuses,
        direct_ll,
        surrogate_ll,
        rom1_ll,
        elapsed_rom_setup,
        elapsed_direct,
        elapsed_surrogate,
        elapsed_rom1,
        elapsed_total,
        size(obs_data, 2),
    )
    speed = speed_summary(
        elapsed_direct,
        elapsed_surrogate,
        elapsed_rom1,
        elapsed_total,
        size(obs_data, 2),
        opts.profile_projection_periods,
        length(candidate_idx),
    )

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
        "elapsed_rom_setup_seconds" => elapsed_rom_setup,
        "elapsed_direct_seconds" => elapsed_direct,
        "elapsed_surrogate_seconds" => elapsed_surrogate,
        "elapsed_rom1_seconds" => elapsed_rom1,
        "elapsed_total_seconds" => elapsed_total,
        "speed_profile_csv" => speed_csv_path,
        "speed_summary" => speed,
        "dynamic_calibration" => dynamic_calibration,
        "dynamic_metric_mask" => collect(dynamic_eval_mask),
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
        if get(dynamic_calibration, "enabled", false)
            println(io, "- Dynamic calibration train anchors: `$(dynamic_calibration["train_points"])`")
            println(io, "- Dynamic calibration held-out anchors: `$(dynamic_calibration["eval_points"])`")
            println(io, "- Dynamic residual train RMSE: `$(fmt(dynamic_calibration["train_rmse"]))`")
            println(io, "- Dynamic residual held-out RMSE: `$(fmt(dynamic_calibration["eval_rmse"]))`")
            println(io, "- Surface metrics use held-out dynamic anchors: `true`")
            println(io, "- Dynamic ridge residual artifact: `$(dynamic_calibration["ridge_payload"])`")
        end
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
        println(io)
        println(io, "## Speed Profile")
        println(io)
        println(io, "- Profiled periods: `$(speed["periods_profiled"])`")
        println(io, "- Median direct SEP seconds/candidate: `$(fmt_seconds(speed["direct_median_seconds"]))`")
        println(io, "- Median surrogate seconds/candidate: `$(fmt_seconds(speed["surrogate_median_seconds"]))`")
        println(io, "- Median ROM1 seconds/candidate: `$(fmt_seconds(speed["rom1_median_seconds"]))`")
        println(io, "- Median direct SEP seconds/period: `$(fmt_seconds(speed["direct_median_seconds_per_period"]))`")
        println(io, "- Median surrogate seconds/period: `$(fmt_seconds(speed["surrogate_median_seconds_per_period"]))`")
        println(io, "- Median ROM1 seconds/period: `$(fmt_seconds(speed["rom1_median_seconds_per_period"]))`")
        println(io, "- Direct/surrogate median speed ratio: `$(fmt(speed["direct_to_surrogate_speedup_median"]))`")
        println(io, "- Direct/ROM1 median speed ratio: `$(fmt(speed["direct_to_rom1_speedup_median"]))`")
        println(io, "- Projected direct SEP hours for $(speed["projection_points"]) anchors × $(speed["projection_periods"]) periods: `$(fmt(speed["projected_direct_hours"]))`")
        println(io, "- Projected surrogate hours for $(speed["projection_points"]) anchors × $(speed["projection_periods"]) periods: `$(fmt(speed["projected_surrogate_hours"]))`")
        println(io, "- Speed CSV: `$(speed_csv_path)`")
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
