#!/usr/bin/env julia

# Two-parameter Galí actual-floor inversion-grid validation.
#
# This escalates the one-parameter std_z check by estimating std_z together
# with a configurable second shock scale, defaulting to std_nu.
# The DGP keeps the corrected adverse eps_z ELB block and adds a small
# deterministic second-shock pulse away from the binding window so the second
# shock scale is locally identified.  The validation is still grid-based: ROM1
# inversion recovers shocks from the OBC observations, and the common recovered
# shocks are evaluated under direct OBC and ROM1 plus an interpolated
# OBC-minus-ROM1 residual surrogate.

ENV["GKSwstype"] = "100"

using AxisKeys
using Distributions
using LinearAlgebra
using Logging
using MacroModelling
using Printf
using Random
using Serialization
using Statistics

const TWOP_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(TWOP_REPO_ROOT, "scripts", "gali_obc_actual_floor_inversion_grid_validation.jl"))

const TWOP_THETA_NAMES = [:std_z, :std_nu]
const TWOP_BASELINE = [0.05, 0.0025]
const TWOP_TRUE = [0.05, 0.0025]
const TWOP_PRIOR_LOG_SD = 0.35
const TWOP_LOWER = 0.95 .* TWOP_BASELINE
const TWOP_UPPER = 1.05 .* TWOP_BASELINE

Base.@kwdef struct TwoParamOptions
    periods::Int = 24
    seed::Int = 20260515
    shock_scale::Float64 = 0.0
    elb_period::Int = 6
    elb_span::Int = 5
    elb_decay::Float64 = 1.0
    elb_shock::Float64 = 0.8
    z_ident_period::Int = 0
    z_ident_span::Int = 0
    z_ident_shock::Float64 = 0.0
    z_ident_pattern::String = "alternating"
    policy_shock_period::Int = 2
    policy_shock_span::Int = 2
    policy_shock::Float64 = 2.0
    policy_shock_pattern::String = "constant"
    second_std_name::Symbol = :std_nu
    second_shock_name::Symbol = :eps_nu
    second_baseline::Float64 = 0.0025
    second_true::Float64 = 0.0025
    axis_size::Int = 7
    train_axis_size::Int = 5
    floor_tolerance_pct::Float64 = 0.25
    inv_maxit::Int = 20
    inv_tol::Float64 = 1e-9
    inv_lambda::Float64 = 1e-6
    hmc_warmup::Int = 0
    hmc_draws::Int = 0
    hmc_chains::Int = 1
    hmc_step_size::Float64 = 0.08
    hmc_max_depth::Int = 2
    hmc_fd_eps::Float64 = 1e-4
    hmc_target_accept::Float64 = 0.8
    out_dir::String = joinpath(TWOP_REPO_ROOT, ".local_artifacts", "gali_actual_floor_twoparam_inversion_grid")
    run_id::String = "actual_floor_twoparam_inversion_grid"
end

struct TwoParamResidualSurrogate
    log_axis_z::Vector{Float64}
    log_axis_nu::Vector{Float64}
    residuals::Array{Float64,4}
    train_rmse::Vector{Float64}
    holdout_rmse::Vector{Float64}
end

function parse_twoparam_args(args)
    opts = TwoParamOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    return TwoParamOptions(
        periods = parse(Int, get(values, "periods", string(opts.periods))),
        seed = parse(Int, get(values, "seed", string(opts.seed))),
        shock_scale = parse(Float64, get(values, "shock-scale", string(opts.shock_scale))),
        elb_period = parse(Int, get(values, "elb-period", string(opts.elb_period))),
        elb_span = parse(Int, get(values, "elb-span", string(opts.elb_span))),
        elb_decay = parse(Float64, get(values, "elb-decay", string(opts.elb_decay))),
        elb_shock = parse(Float64, get(values, "elb-shock", string(opts.elb_shock))),
        z_ident_period = parse(Int, get(values, "z-ident-period", string(opts.z_ident_period))),
        z_ident_span = parse(Int, get(values, "z-ident-span", string(opts.z_ident_span))),
        z_ident_shock = parse(Float64, get(values, "z-ident-shock", string(opts.z_ident_shock))),
        z_ident_pattern = get(values, "z-ident-pattern", opts.z_ident_pattern),
        policy_shock_period = parse(Int, get(values, "policy-shock-period", string(opts.policy_shock_period))),
        policy_shock_span = parse(Int, get(values, "policy-shock-span", string(opts.policy_shock_span))),
        policy_shock = parse(Float64, get(values, "policy-shock", string(opts.policy_shock))),
        policy_shock_pattern = get(values, "policy-shock-pattern", opts.policy_shock_pattern),
        second_std_name = Symbol(get(values, "second-std-name", string(opts.second_std_name))),
        second_shock_name = Symbol(get(values, "second-shock-name", string(opts.second_shock_name))),
        second_baseline = parse(Float64, get(values, "second-baseline", string(opts.second_baseline))),
        second_true = parse(Float64, get(values, "second-true", string(opts.second_true))),
        axis_size = parse(Int, get(values, "axis-size", string(opts.axis_size))),
        train_axis_size = parse(Int, get(values, "train-axis-size", string(opts.train_axis_size))),
        floor_tolerance_pct = parse(Float64, get(values, "floor-tolerance-pct", string(opts.floor_tolerance_pct))),
        inv_maxit = parse(Int, get(values, "inv-maxit", string(opts.inv_maxit))),
        inv_tol = parse(Float64, get(values, "inv-tol", string(opts.inv_tol))),
        inv_lambda = parse(Float64, get(values, "inv-lambda", string(opts.inv_lambda))),
        hmc_warmup = parse(Int, get(values, "hmc-warmup", string(opts.hmc_warmup))),
        hmc_draws = parse(Int, get(values, "hmc-draws", string(opts.hmc_draws))),
        hmc_chains = parse(Int, get(values, "hmc-chains", string(opts.hmc_chains))),
        hmc_step_size = parse(Float64, get(values, "hmc-step-size", string(opts.hmc_step_size))),
        hmc_max_depth = parse(Int, get(values, "hmc-max-depth", string(opts.hmc_max_depth))),
        hmc_fd_eps = parse(Float64, get(values, "hmc-fd-eps", string(opts.hmc_fd_eps))),
        hmc_target_accept = parse(Float64, get(values, "hmc-target-accept", string(opts.hmc_target_accept))),
        out_dir = get(values, "out-dir", opts.out_dir),
        run_id = get(values, "run-id", opts.run_id),
    )
end

theta_names(opts::TwoParamOptions) = [:std_z, opts.second_std_name]
theta_baseline(opts::TwoParamOptions) = [TWOP_BASELINE[1], opts.second_baseline]
theta_true(opts::TwoParamOptions) = [TWOP_TRUE[1], opts.second_true]
theta_lower(opts::TwoParamOptions) = 0.95 .* theta_baseline(opts)
theta_upper(opts::TwoParamOptions) = 1.05 .* theta_baseline(opts)

function axis_grid(i::Int, n::Int, opts::TwoParamOptions)
    lower = theta_lower(opts)
    upper = theta_upper(opts)
    truth = theta_true(opts)
    vals = exp.(collect(range(log(lower[i]), log(upper[i]); length = n)))
    if !any(isapprox.(vals, truth[i]; rtol = 1e-12, atol = 1e-14))
        push!(vals, truth[i])
    end
    return sort(vals)
end

function log_prior(theta::AbstractVector, opts::TwoParamOptions)
    lower = theta_lower(opts)
    upper = theta_upper(opts)
    baseline = theta_baseline(opts)
    for i in eachindex(theta)
        tol = 64 * eps(Float64) * max(1.0, abs(theta[i]), abs(baseline[i]))
        (theta[i] < lower[i] - tol || theta[i] > upper[i] + tol) && return -Inf
    end
    theta_clamped = clamp.(Float64.(theta), lower, upper)
    lp = 0.0
    for i in eachindex(theta)
        lp += Distributions.logpdf(Distributions.Normal(log(baseline[i]), TWOP_PRIOR_LOG_SD), log(theta_clamped[i]))
    end
    return lp
end

function deterministic_shock_value(amplitude::Float64, pattern::String, j::Int)
    if pattern == "constant"
        return amplitude
    elseif pattern == "alternating"
        return isodd(j) ? amplitude : -amplitude
    else
        error("Unsupported shock pattern=$pattern. Use `constant` or `alternating`.")
    end
end

deterministic_second_shock_value(opts::TwoParamOptions, j::Int) =
    deterministic_shock_value(opts.policy_shock, opts.policy_shock_pattern, j)

function base_two_param_shocks(model, opts::TwoParamOptions)
    stoch_opts = StochCompareOptions(
        periods = opts.periods,
        seed = opts.seed,
        shock_scale = opts.shock_scale,
        elb_period = opts.elb_period,
        elb_shock_name = :eps_z,
        elb_shock = opts.elb_shock,
        elb_span = opts.elb_span,
        elb_decay = opts.elb_decay,
        tail_periods = 0,
        floor_tolerance_pct = opts.floor_tolerance_pct,
        out_dir = opts.out_dir,
    )
    shocks = build_stochastic_shocks(model, stoch_opts)
    if opts.z_ident_period > 0 && opts.z_ident_span > 0 && opts.z_ident_shock != 0.0
        eps_z_idx = find_idx(model.exo, :eps_z)
        for j in 1:opts.z_ident_span
            t = opts.z_ident_period + j - 1
            t <= opts.periods || break
            shocks[eps_z_idx, t] = deterministic_shock_value(opts.z_ident_shock, opts.z_ident_pattern, j)
        end
    end
    second_idx = find_idx(model.exo, opts.second_shock_name)
    for j in 1:opts.policy_shock_span
        t = opts.policy_shock_period + j - 1
        t <= opts.periods || break
        shocks[second_idx, t] = deterministic_second_shock_value(opts, j)
    end
    return shocks
end

function scale_two_param_shocks(model, shocks::Matrix{Float64}, theta::AbstractVector, opts::TwoParamOptions)
    baseline = theta_baseline(opts)
    scaled = copy(shocks)
    scaled[find_idx(model.exo, :eps_z), :] .*= theta[1] / baseline[1]
    scaled[find_idx(model.exo, opts.second_shock_name), :] .*= theta[2] / baseline[2]
    return scaled
end

function simulate_two_param_observables(model, opts::TwoParamOptions, theta::AbstractVector,
                                        shocks::Matrix{Float64}; ignore_obc::Bool,
                                        warnings::Union{Nothing,Vector{String}} = nothing)
    local_model = deepcopy(model)
    scaled_shocks = scale_two_param_shocks(local_model, shocks, theta, opts)
    keyed_shocks = KeyedArray(scaled_shocks; Shocks = local_model.timings.exo, Periods = 1:opts.periods)
    logger = WarningCaptureLogger(String[])
    irf = Logging.with_logger(logger) do
        MacroModelling.get_irf(
            local_model;
            algorithm = :first_order,
            shocks = keyed_shocks,
            periods = 0,
            variables = FLOOR_OBS,
            levels = true,
            ignore_obc = ignore_obc,
            verbose = false,
        )
    end
    warnings !== nothing && append!(warnings, logger.messages)
    ss = MacroModelling.get_steady_state(local_model; derivatives = false, verbose = false)
    log_y_ss = Float64(ss[find_idx(local_model.var, :log_y), 1])
    obs = zeros(Float64, length(FLOOR_OBS), opts.periods)
    obs[1, :] .= 100 .* (keyed_series(irf, :log_y, opts.periods) .- log_y_ss)
    obs[2, :] .= 100 .* keyed_series(irf, :pi_ann, opts.periods)
    obs[3, :] .= 100 .* keyed_series(irf, :i_ann, opts.periods)
    return obs
end

function build_two_param_linear_predict(model, opts::TwoParamOptions)
    local_model = deepcopy(model)
    base_params = Float64.(local_model.parameter_values)
    MacroModelling.write_parameters_input!(local_model, base_params, verbose = false)
    MacroModelling.solve!(local_model; algorithm = :first_order, dynamics = true, obc = false, silent = true)

    state_idx = collect(1:length(local_model.var))
    obs_idx = [find_idx(local_model.var, obs) for obs in FLOOR_OBS]
    _, raw_predict_tuple, nsss = build_matrix_rom_predict(local_model; state_idx = state_idx, obs_idx = obs_idx)
    log_y_ss = Float64(nsss[find_idx(local_model.var, :log_y)])
    eps_z_idx = find_idx(local_model.exo, :eps_z)
    second_idx = find_idx(local_model.exo, opts.second_shock_name)
    baseline = theta_baseline(opts)

    function predict_tuple(state::AbstractVector, shock::AbstractVector, theta::AbstractVector)
        shock_scaled = copy(shock)
        shock_scaled[eps_z_idx] *= theta[1] / baseline[1]
        shock_scaled[second_idx] *= theta[2] / baseline[2]
        raw_obs, state_next = raw_predict_tuple(state, shock_scaled, theta)
        return transformed_observables_from_raw(raw_obs, log_y_ss), state_next
    end

    structural_idx = [find_idx(local_model.exo, s) for s in (:eps_a, :eps_z, :eps_nu)]
    return predict_tuple, Float64.(nsss), structural_idx
end

function recover_two_param_path(predict_tuple::Function, state0::Vector{Float64},
                                structural_idx::Vector{Int}, obs_data::Matrix{Float64},
                                obs_sigma::Vector{Float64}, theta::AbstractVector,
                                opts::TwoParamOptions)
    shock_sigmas = zeros(Float64, length(Gali_2015_chapter_3_obc.exo))
    shock_sigmas[structural_idx] .= 1.0
    state = copy(state0)
    states = zeros(Float64, length(state0), size(obs_data, 2))
    shocks = zeros(Float64, length(shock_sigmas), size(obs_data, 2))
    residual_norms = zeros(Float64, size(obs_data, 2))
    logdet_terms = zeros(Float64, size(obs_data, 2))
    eps_init = zeros(Float64, length(structural_idx))
    for t in 1:size(obs_data, 2)
        states[:, t] .= state
        local eps_full, state_next, ll_t
        try
            eps_full, state_next, ll_t = MacroModelling.inversion_step(
                predict_tuple,
                state,
                obs_data[:, t],
                theta,
                obs_sigma,
                shock_sigmas,
                structural_idx;
                eps_init = eps_init,
                maxit = opts.inv_maxit,
                tol = opts.inv_tol,
                lambda = opts.inv_lambda,
            )
        catch e
            return (ok = false, states = states, shocks = shocks,
                    residual_norms = residual_norms, logdet_terms = logdet_terms,
                    logdet_total = -Inf,
                    message = sprint(showerror, e), fail_period = t)
        end
        if !all(isfinite, eps_full) || !all(isfinite, state_next) || !isfinite(ll_t)
            return (ok = false, states = states, shocks = shocks, residual_norms = residual_norms,
                    logdet_terms = logdet_terms, logdet_total = -Inf,
                    message = "nonfinite inversion result", fail_period = t)
        end
        pred_obs, _ = predict_tuple(state, eps_full, theta)
        logdet_t = rom1_inversion_logdet(predict_tuple, state, eps_full, theta, structural_idx)
        isfinite(logdet_t) || return (ok = false, states = states, shocks = shocks,
                                      residual_norms = residual_norms, logdet_terms = logdet_terms,
                                      logdet_total = -Inf,
                                      message = "nonfinite ROM1 inversion log determinant", fail_period = t)
        shocks[:, t] .= eps_full
        state = Float64.(state_next)
        eps_init .= eps_full[structural_idx]
        residual_norms[t] = norm((obs_data[:, t] .- pred_obs) ./ obs_sigma)
        logdet_terms[t] = logdet_t
    end
    return (ok = true, states = states, shocks = shocks, residual_norms = residual_norms,
            logdet_terms = logdet_terms, logdet_total = sum(logdet_terms),
            message = "", fail_period = 0)
end

function rom1_inversion_logdet(predict_tuple::Function,
                               state::AbstractVector,
                               eps_full::AbstractVector,
                               theta::AbstractVector,
                               structural_idx::Vector{Int})
    base_obs, _ = predict_tuple(state, eps_full, theta)
    J = zeros(Float64, length(base_obs), length(structural_idx))
    for (jcol, jshock) in pairs(structural_idx)
        h = max(cbrt(eps(Float64)) * max(1.0, abs(eps_full[jshock])), 1e-6)
        eps_p = copy(eps_full)
        eps_p[jshock] += h
        obs_p, _ = predict_tuple(state, eps_p, theta)
        J[:, jcol] .= (Float64.(obs_p) .- Float64.(base_obs)) ./ h
    end
    val, sign = logabsdet(J)
    sign == 0 && return -Inf
    return val
end

function bracket_index(axis::Vector{Float64}, x::Float64)
    if x <= first(axis)
        return 1, 1, 0.0
    elseif x >= last(axis)
        n = length(axis)
        return n, n, 0.0
    end
    hi = searchsortedfirst(axis, x)
    lo = hi - 1
    w = (x - axis[lo]) / (axis[hi] - axis[lo])
    return lo, hi, w
end

function interpolate_residual(s::TwoParamResidualSurrogate, theta::AbstractVector)
    xz, xnu = log(theta[1]), log(theta[2])
    zlo, zhi, wz = bracket_index(s.log_axis_z, xz)
    nlo, nhi, wnu = bracket_index(s.log_axis_nu, xnu)
    r00 = s.residuals[:, :, zlo, nlo]
    r10 = s.residuals[:, :, zhi, nlo]
    r01 = s.residuals[:, :, zlo, nhi]
    r11 = s.residuals[:, :, zhi, nhi]
    return (1 - wz) * (1 - wnu) .* r00 .+
           wz * (1 - wnu) .* r10 .+
           (1 - wz) * wnu .* r01 .+
           wz * wnu .* r11
end

function direct_and_linear_paths(model, opts::TwoParamOptions, theta::AbstractVector,
                                 shocks::Matrix{Float64}, warnings::Vector{String})
    direct = simulate_two_param_observables(model, opts, theta, shocks; ignore_obc = false, warnings = warnings)
    linear = simulate_two_param_observables(model, opts, theta, shocks; ignore_obc = true, warnings = warnings)
    return direct, linear
end

function train_twoparam_surrogate(model, opts::TwoParamOptions, predict_tuple::Function,
                                  state0::Vector{Float64}, structural_idx::Vector{Int},
                                  obs_data::Matrix{Float64}, obs_sigma::Vector{Float64},
                                  warnings::Vector{String})
    axis_z = axis_grid(1, opts.train_axis_size, opts)
    axis_nu = axis_grid(2, opts.train_axis_size, opts)
    residuals = zeros(Float64, length(FLOOR_OBS), opts.periods, length(axis_z), length(axis_nu))
    for (i, ztheta) in pairs(axis_z), (j, nutheta) in pairs(axis_nu)
        theta = [ztheta, nutheta]
        recovered = recover_two_param_path(predict_tuple, state0, structural_idx, obs_data, obs_sigma, theta, opts)
        recovered.ok || error("Training inversion failed at theta=$(theta): $(recovered.message)")
        direct, linear = direct_and_linear_paths(model, opts, theta, recovered.shocks, warnings)
        residuals[:, :, i, j] .= direct .- linear
    end
    s_tmp = TwoParamResidualSurrogate(log.(axis_z), log.(axis_nu), residuals, zeros(length(FLOOR_OBS)), zeros(length(FLOOR_OBS)))
    train_err = zeros(Float64, length(FLOOR_OBS), length(axis_z) * length(axis_nu) * opts.periods)
    col = 1
    for ztheta in axis_z, nutheta in axis_nu
        theta = [ztheta, nutheta]
        i = findfirst(isapprox(ztheta; rtol = 1e-12, atol = 1e-14), axis_z)
        j = findfirst(isapprox(nutheta; rtol = 1e-12, atol = 1e-14), axis_nu)
        train_err[:, col:col+opts.periods-1] .= interpolate_residual(s_tmp, theta) .- residuals[:, :, i, j]
        col += opts.periods
    end
    train_rmse = sqrt.(vec(Statistics.mean(train_err .^ 2, dims = 2)))

    mid_z = exp.((log.(axis_z[1:end-1]) .+ log.(axis_z[2:end])) ./ 2)
    mid_nu = exp.((log.(axis_nu[1:end-1]) .+ log.(axis_nu[2:end])) ./ 2)
    holdout_err = zeros(Float64, length(FLOOR_OBS), length(mid_z) * length(mid_nu) * opts.periods)
    col = 1
    for ztheta in mid_z, nutheta in mid_nu
        theta = [ztheta, nutheta]
        recovered = recover_two_param_path(predict_tuple, state0, structural_idx, obs_data, obs_sigma, theta, opts)
        recovered.ok || error("Holdout inversion failed at theta=$(theta): $(recovered.message)")
        direct, linear = direct_and_linear_paths(model, opts, theta, recovered.shocks, warnings)
        holdout_err[:, col:col+opts.periods-1] .= interpolate_residual(s_tmp, theta) .- (direct .- linear)
        col += opts.periods
    end
    holdout_rmse = sqrt.(vec(Statistics.mean(holdout_err .^ 2, dims = 2)))
    return TwoParamResidualSurrogate(log.(axis_z), log.(axis_nu), residuals, train_rmse, holdout_rmse)
end

function posterior_summary(theta_points::Matrix{Float64}, logpost::Vector{Float64}, opts::TwoParamOptions)
    finite = isfinite.(logpost)
    weights = zeros(Float64, length(logpost))
    weights[finite] .= exp.(logpost[finite] .- maximum(logpost[finite]))
    weights ./= sum(weights)
    means = vec(sum(theta_points .* weights', dims = 2))
    rows = Vector{Dict{String,Any}}()
    for p in 1:size(theta_points, 1)
        order = sortperm(theta_points[p, :])
        vals = theta_points[p, order]
        cdf = cumsum(weights[order])
        push!(rows, Dict{String,Any}(
            "parameter" => string(theta_names(opts)[p]),
            "mean" => means[p],
            "sd" => sqrt(sum(weights .* (theta_points[p, :] .- means[p]) .^ 2)),
            "q05" => vals[findfirst(>=(0.05), cdf)],
            "q95" => vals[findfirst(>=(0.95), cdf)],
            "map" => theta_points[p, argmax(logpost)],
        ))
    end
    return Dict{String,Any}("weights" => weights, "rows" => rows, "map_theta" => theta_points[:, argmax(logpost)])
end

function twoparam_objective_at_theta(model, opts::TwoParamOptions,
                                     predict_tuple::Function, state0::Vector{Float64},
                                     structural_idx::Vector{Int}, dgp_obs::Matrix{Float64},
                                     obs_sigma::Vector{Float64},
                                     surrogate::TwoParamResidualSurrogate,
                                     theta::AbstractVector, warnings::Vector{String};
                                     objective::Symbol = :both)
    lp = log_prior(theta, opts)
    isfinite(lp) || return (ok = false, direct_logpost = -Inf, surrogate_logpost = -Inf,
                            recovered = nothing, direct_path = nothing, surrogate_path = nothing,
                            message = "nonfinite prior")
    recovered = recover_two_param_path(predict_tuple, state0, structural_idx, dgp_obs, obs_sigma, theta, opts)
    recovered.ok || return (ok = false, direct_logpost = -Inf, surrogate_logpost = -Inf,
                            recovered = recovered, direct_path = nothing, surrogate_path = nothing,
                            message = recovered.message)
    direct = nothing
    surrogate_path = nothing
    direct_lp = -Inf
    surrogate_lp = -Inf
    if objective in (:both, :direct)
        direct = simulate_two_param_observables(model, opts, theta, recovered.shocks; ignore_obc = false, warnings = warnings)
        direct_lp = lp + inversion_loglik(dgp_obs, direct, obs_sigma, recovered.shocks, structural_idx) - recovered.logdet_total
    end
    if objective in (:both, :surrogate)
        linear = simulate_two_param_observables(model, opts, theta, recovered.shocks; ignore_obc = true, warnings = warnings)
        surrogate_path = linear .+ interpolate_residual(surrogate, theta)
        surrogate_lp = lp + inversion_loglik(dgp_obs, surrogate_path, obs_sigma, recovered.shocks, structural_idx) - recovered.logdet_total
    end
    ok = objective == :both ? (isfinite(direct_lp) && isfinite(surrogate_lp)) :
         objective == :direct ? isfinite(direct_lp) :
         objective == :surrogate ? isfinite(surrogate_lp) :
         error("Unsupported objective=$objective")
    return (ok = ok,
            direct_logpost = direct_lp,
            surrogate_logpost = surrogate_lp,
            recovered = recovered,
            direct_path = direct,
            surrogate_path = surrogate_path,
            message = "")
end

function twop_sigmoid_stable(x::Float64)
    if x >= 0
        ex = exp(-x)
        return 1 / (1 + ex)
    end
    ex = exp(x)
    return ex / (1 + ex)
end

function twop_hmc_z_to_theta(z::AbstractVector, opts::TwoParamOptions)
    lower = theta_lower(opts)
    upper = theta_upper(opts)
    theta = similar(Float64.(z))
    for i in eachindex(theta)
        s = twop_sigmoid_stable(Float64(z[i]))
        theta[i] = lower[i] + (upper[i] - lower[i]) * s
    end
    return theta
end

function twop_theta_to_hmc_z(theta::AbstractVector, opts::TwoParamOptions)
    lower = theta_lower(opts)
    upper = theta_upper(opts)
    z = similar(Float64.(theta))
    for i in eachindex(z)
        s = clamp((theta[i] - lower[i]) / (upper[i] - lower[i]), 1e-10, 1 - 1e-10)
        z[i] = log(s / (1 - s))
    end
    return z
end

function twop_hmc_logjac(z::AbstractVector, opts::TwoParamOptions)
    lower = theta_lower(opts)
    upper = theta_upper(opts)
    lj = 0.0
    for i in eachindex(z)
        s = twop_sigmoid_stable(Float64(z[i]))
        if !(0 < s < 1)
            return -Inf
        end
        lj += log(upper[i] - lower[i]) + log(s) + log1p(-s)
    end
    return lj
end

struct TwoParamHMCLogDensity
    logpost_z::Function
    fd_eps::Float64
end

LogDensityProblems.logdensity(p::TwoParamHMCLogDensity, z) = p.logpost_z(Vector{Float64}(z))
LogDensityProblems.dimension(::TwoParamHMCLogDensity) = 2
LogDensityProblems.capabilities(::Type{TwoParamHMCLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function twop_fd_gradient!(grad::Vector{Float64}, f::Function, z::Vector{Float64}, h::Float64)
    f0 = f(z)
    isfinite(f0) || (fill!(grad, NaN); return f0)
    for i in eachindex(z)
        zp = copy(z)
        zm = copy(z)
        zp[i] += h
        zm[i] -= h
        fp = f(zp)
        fm = f(zm)
        grad[i] = if isfinite(fp) && isfinite(fm)
            (fp - fm) / (2 * h)
        elseif isfinite(fp)
            (fp - f0) / h
        elseif isfinite(fm)
            (f0 - fm) / h
        else
            NaN
        end
    end
    return f0
end

function LogDensityProblems.logdensity_and_gradient(p::TwoParamHMCLogDensity, z)
    zf = Vector{Float64}(z)
    grad = zeros(Float64, length(zf))
    val = twop_fd_gradient!(grad, p.logpost_z, zf, p.fd_eps)
    return val, grad
end

function make_twop_hmc_logpost(logpost_theta::Function, opts::TwoParamOptions)
    return function (z::Vector{Float64})
        theta = twop_hmc_z_to_theta(z, opts)
        lj = twop_hmc_logjac(z, opts)
        isfinite(lj) || return -Inf
        lp = logpost_theta(theta)
        isfinite(lp) || return -Inf
        return lp + lj
    end
end

function run_twoparam_hmc(name::String, logpost_theta::Function, opts::TwoParamOptions; seed::Int)
    opts.hmc_draws > 0 || return nothing
    logpost_z = make_twop_hmc_logpost(logpost_theta, opts)
    z_init = twop_theta_to_hmc_z(theta_true(opts), opts)
    logdensity = TwoParamHMCLogDensity(logpost_z, opts.hmc_fd_eps)
    metric = UnitEuclideanMetric(2)
    hamiltonian = Hamiltonian(metric, logdensity)
    integrator = Leapfrog(opts.hmc_step_size)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth = opts.hmc_max_depth)))
    adaptor = StepSizeAdaptor(opts.hmc_target_accept, integrator)
    chains = Vector{Dict{String,Any}}()

    println("Running $name two-parameter NUTS: chains=$(opts.hmc_chains), warmup=$(opts.hmc_warmup), draws=$(opts.hmc_draws)")
    for chain in 1:opts.hmc_chains
        Random.seed!(seed + 1000 * chain)
        local samples_z, stats
        elapsed = @elapsed begin
            samples_z, stats = sample(
                hamiltonian,
                kernel,
                z_init,
                opts.hmc_warmup + opts.hmc_draws,
                adaptor,
                opts.hmc_warmup;
                progress = false,
                verbose = false,
            )
        end
        theta_all = hcat([twop_hmc_z_to_theta(Vector{Float64}(s), opts) for s in samples_z]...)'
        theta_post = theta_all[(opts.hmc_warmup + 1):end, :]
        stats_post = stats[(opts.hmc_warmup + 1):end]
        numerical_errors = [getproperty(s, :numerical_error) for s in stats_post]
        accept_rates = [getproperty(s, :acceptance_rate) for s in stats_post]
        tree_depths = [getproperty(s, :tree_depth) for s in stats_post]
        push!(chains, Dict{String,Any}(
            "name" => name,
            "chain" => chain,
            "theta_post" => Matrix{Float64}(theta_post),
            "theta_all" => Matrix{Float64}(theta_all),
            "n_numerical_errors" => sum(numerical_errors),
            "mean_accept" => Statistics.mean(accept_rates),
            "max_tree_depth" => maximum(tree_depths),
            "elapsed_s" => elapsed,
        ))
        println("  $name chain $chain complete: elapsed=$(round(elapsed, digits = 2))s, numerical_errors=$(sum(numerical_errors))")
    end
    return chains
end

function twop_hmc_summary(chains::Vector{Dict{String,Any}})
    theta = vcat([c["theta_post"] for c in chains]...)
    n = size(theta, 1)
    return Dict{String,Any}(
        "mean" => vec(Statistics.mean(theta, dims = 1)),
        "mcse" => [Statistics.std(theta[:, i]) / sqrt(n) for i in 1:size(theta, 2)],
        "q05" => [Statistics.quantile(theta[:, i], 0.05) for i in 1:size(theta, 2)],
        "q95" => [Statistics.quantile(theta[:, i], 0.95) for i in 1:size(theta, 2)],
        "n_draws" => n,
        "n_numerical_errors" => sum(c["n_numerical_errors"] for c in chains),
        "mean_accept" => Statistics.mean([c["mean_accept"] for c in chains]),
        "max_tree_depth" => maximum([c["max_tree_depth"] for c in chains]),
        "elapsed_s" => sum(c["elapsed_s"] for c in chains),
    )
end

function twop_hmc_comparison(direct_chains::Vector{Dict{String,Any}},
                             surrogate_chains::Vector{Dict{String,Any}},
                             opts::TwoParamOptions)
    direct = twop_hmc_summary(direct_chains)
    surrogate = twop_hmc_summary(surrogate_chains)
    truth = theta_true(opts)
    names = theta_names(opts)
    rows = Vector{Dict{String,Any}}()
    pass_rows = true
    for i in eachindex(truth)
        denom = sqrt(direct["mcse"][i]^2 + surrogate["mcse"][i]^2)
        diff_mcse = denom > 0 ? (surrogate["mean"][i] - direct["mean"][i]) / denom : NaN
        overlap = max(direct["q05"][i], surrogate["q05"][i]) <= min(direct["q95"][i], surrogate["q95"][i])
        direct_cover = direct["q05"][i] <= truth[i] <= direct["q95"][i]
        surrogate_cover = surrogate["q05"][i] <= truth[i] <= surrogate["q95"][i]
        pass_rows &= overlap && direct_cover && surrogate_cover && abs(diff_mcse) <= 2.0
        push!(rows, Dict{String,Any}(
            "parameter" => string(names[i]),
            "true" => truth[i],
            "direct_mean" => direct["mean"][i],
            "direct_mcse" => direct["mcse"][i],
            "direct_q05" => direct["q05"][i],
            "direct_q95" => direct["q95"][i],
            "surrogate_mean" => surrogate["mean"][i],
            "surrogate_mcse" => surrogate["mcse"][i],
            "surrogate_q05" => surrogate["q05"][i],
            "surrogate_q95" => surrogate["q95"][i],
            "mean_diff_combined_mcse" => diff_mcse,
            "interval_overlap" => overlap,
            "direct_cover" => direct_cover,
            "surrogate_cover" => surrogate_cover,
        ))
    end
    return Dict{String,Any}(
        "direct" => direct,
        "surrogate" => surrogate,
        "rows" => rows,
        "pass_rows" => pass_rows,
    )
end

function negligible_hmc_issues(summary::Dict{String,Any}; rate_tol::Float64 = 0.005)
    n_draws = max(Int(summary["n_draws"]), 1)
    allowed = max(1, ceil(Int, rate_tol * n_draws))
    return Int(summary["n_numerical_errors"]) <= allowed
end

function run_twoparam_validation(opts::TwoParamOptions)
    out_dir = joinpath(opts.out_dir, opts.run_id)
    mkpath(out_dir)
    model = Gali_2015_chapter_3_obc
    warnings = String[]

    println("Generating two-parameter actual-floor OBC DGP...")
    names = theta_names(opts)
    truth = theta_true(opts)
    base_shocks = base_two_param_shocks(model, opts)
    dgp_obs = simulate_two_param_observables(model, opts, truth, base_shocks; ignore_obc = false, warnings = warnings)
    lin_true = simulate_two_param_observables(model, opts, truth, base_shocks; ignore_obc = true, warnings = warnings)
    obs_sigma = max.(0.05 .* vec(Statistics.std(dgp_obs, dims = 2; corrected = false)), 1e-3)
    floor_pct = 400 * log(Float64(model.parameter_values[find_idx(model.parameters, :R̄)]))
    actual_floor = dgp_obs[3, :] .<= floor_pct + opts.floor_tolerance_pct
    linear_violates = lin_true[3, :] .< floor_pct

    println("Building two-parameter linear ROM1 inversion predictor...")
    predict_tuple, state0, structural_idx = build_two_param_linear_predict(model, opts)

    println("Training two-parameter inversion-path residual surrogate...")
    surrogate = train_twoparam_surrogate(model, opts, predict_tuple, state0, structural_idx, dgp_obs, obs_sigma, warnings)

    axis_z = axis_grid(1, opts.axis_size, opts)
    axis_nu = axis_grid(2, opts.axis_size, opts)
    n_points = length(axis_z) * length(axis_nu)
    theta_points = zeros(Float64, 2, n_points)
    direct_logpost = fill(-Inf, n_points)
    surrogate_logpost = fill(-Inf, n_points)
    recovery_failures = 0
    max_resid_norm = 0.0
    max_shock = 0.0
    k = 1
    for ztheta in axis_z, nutheta in axis_nu
        theta = [ztheta, nutheta]
        theta_points[:, k] .= theta
        eval = twoparam_objective_at_theta(
            model,
            opts,
            predict_tuple,
            state0,
            structural_idx,
            dgp_obs,
            obs_sigma,
            surrogate,
            theta,
            warnings,
        )
        if eval.ok
            recovered = eval.recovered
            direct_logpost[k] = eval.direct_logpost
            surrogate_logpost[k] = eval.surrogate_logpost
            max_resid_norm = max(max_resid_norm, maximum(recovered.residual_norms))
            max_shock = max(max_shock, maximum(abs.(recovered.shocks[structural_idx, :])))
        else
            recovery_failures += 1
        end
        k += 1
    end

    direct = posterior_summary(theta_points, direct_logpost, opts)
    surrogate_summary = posterior_summary(theta_points, surrogate_logpost, opts)
    rows = Vector{Dict{String,Any}}()
    pass_rows = true
    for p in 1:2
        drow = direct["rows"][p]
        srow = surrogate_summary["rows"][p]
        overlap = max(drow["q05"], srow["q05"]) <= min(drow["q95"], srow["q95"])
        dcover = drow["q05"] <= truth[p] <= drow["q95"]
        scover = srow["q05"] <= truth[p] <= srow["q95"]
        diff_sd = abs(drow["mean"] - srow["mean"]) / max(drow["sd"], eps(Float64))
        pass_rows &= overlap && dcover && scover && diff_sd <= 0.25
        push!(rows, Dict{String,Any}(
            "parameter" => string(names[p]),
            "true" => truth[p],
            "direct_mean" => drow["mean"],
            "surrogate_mean" => srow["mean"],
            "direct_q05" => drow["q05"],
            "direct_q95" => drow["q95"],
            "surrogate_q05" => srow["q05"],
            "surrogate_q95" => srow["q95"],
            "mean_diff_direct_sd" => diff_sd,
            "interval_overlap" => overlap,
            "direct_cover" => dcover,
            "surrogate_cover" => scover,
        ))
    end
    hmc_warnings = String[]
    direct_hmc_chains = nothing
    surrogate_hmc_chains = nothing
    hmc_cmp = nothing
    if opts.hmc_draws > 0
        direct_logpost_theta = function (theta::Vector{Float64})
            eval = twoparam_objective_at_theta(
                model,
                opts,
                predict_tuple,
                state0,
                structural_idx,
                dgp_obs,
                obs_sigma,
                surrogate,
                theta,
                hmc_warnings;
                objective = :direct,
            )
            return eval.direct_logpost
        end
        surrogate_logpost_theta = function (theta::Vector{Float64})
            eval = twoparam_objective_at_theta(
                model,
                opts,
                predict_tuple,
                state0,
                structural_idx,
                dgp_obs,
                obs_sigma,
                surrogate,
                theta,
                hmc_warnings;
                objective = :surrogate,
            )
            return eval.surrogate_logpost
        end
        hmc_seed = opts.seed + 202
        direct_hmc_chains = run_twoparam_hmc("direct_obc_twoparam_inversion", direct_logpost_theta, opts; seed = hmc_seed)
        surrogate_hmc_chains = run_twoparam_hmc("surrogate_twoparam_inversion", surrogate_logpost_theta, opts; seed = hmc_seed)
        hmc_cmp = twop_hmc_comparison(direct_hmc_chains, surrogate_hmc_chains, opts)
        serialize(joinpath(out_dir, "direct_twoparam_hmc_chains.jls"), direct_hmc_chains)
        serialize(joinpath(out_dir, "surrogate_twoparam_hmc_chains.jls"), surrogate_hmc_chains)
        serialize(joinpath(out_dir, "twoparam_hmc_comparison_payload.jls"), hmc_cmp)
    end
    grid_validation_pass = isempty(warnings) && recovery_failures == 0 && pass_rows
    hmc_strict_zero_issue_pass = hmc_cmp === nothing ||
        (isempty(hmc_warnings) &&
         hmc_cmp["direct"]["n_numerical_errors"] == 0 &&
         hmc_cmp["surrogate"]["n_numerical_errors"] == 0)
    hmc_negligible_issue_pass = hmc_cmp === nothing ||
        (length(hmc_warnings) <= max(1, ceil(Int, 0.005 * max(hmc_cmp["direct"]["n_draws"], 1))) &&
         negligible_hmc_issues(hmc_cmp["direct"]) &&
         negligible_hmc_issues(hmc_cmp["surrogate"]))
    hmc_validation_pass = hmc_cmp === nothing || (hmc_negligible_issue_pass && hmc_cmp["pass_rows"])
    validation_pass = grid_validation_pass && hmc_validation_pass
    payload = Dict{String,Any}(
        "options" => opts,
        "policy_shock_pattern" => opts.policy_shock_pattern,
        "theta_names" => names,
        "theta_true" => truth,
        "base_shocks" => base_shocks,
        "dgp_obs" => dgp_obs,
        "linear_true" => lin_true,
        "obs_sigma" => obs_sigma,
        "actual_floor" => actual_floor,
        "linear_violates" => linear_violates,
        "surrogate" => surrogate,
        "theta_points" => theta_points,
        "direct_logpost" => direct_logpost,
        "surrogate_logpost" => surrogate_logpost,
        "direct" => direct,
        "surrogate_summary" => surrogate_summary,
        "rows" => rows,
        "finite_direct_grid_points" => count(isfinite, direct_logpost),
        "finite_surrogate_grid_points" => count(isfinite, surrogate_logpost),
        "recovery_failures" => recovery_failures,
        "max_linear_inversion_residual_norm" => max_resid_norm,
        "max_abs_recovered_structural_shock" => max_shock,
        "solver_warning_count_total" => length(warnings),
        "solver_warnings_unique" => sort(unique(warnings)),
        "hmc_warning_count_total" => length(hmc_warnings),
        "hmc_warnings_unique" => sort(unique(hmc_warnings)),
        "direct_hmc_chains" => direct_hmc_chains,
        "surrogate_hmc_chains" => surrogate_hmc_chains,
        "hmc_comparison" => hmc_cmp,
        "grid_validation_pass" => grid_validation_pass,
        "hmc_strict_zero_issue_pass" => hmc_strict_zero_issue_pass,
        "hmc_negligible_issue_pass" => hmc_negligible_issue_pass,
        "hmc_validation_pass" => hmc_validation_pass,
        "validation_pass" => validation_pass,
    )
    serialize(joinpath(out_dir, "actual_floor_twoparam_inversion_grid_payload.jls"), payload)

    open(joinpath(out_dir, "SUMMARY.md"), "w") do io
        println(io, "# Galí Actual-Floor Two-Parameter Inversion Grid Validation")
        println(io)
        println(io, "- Parameters: `$(join(string.(names), "`, `"))`")
        println(io, "- True values: $(join(truth, ", "))")
        println(io, "- DGP: adverse `eps_z` ELB block plus deterministic `$(opts.second_shock_name)` identification pulse")
        println(io, "- Objective: profiled Gaussian inversion criterion with ROM1 shock-to-observable log-determinant correction")
        if opts.z_ident_period > 0 && opts.z_ident_span > 0 && opts.z_ident_shock != 0.0
            println(io, "- Post-ELB `eps_z` identification block: `eps_z[$(opts.z_ident_period):$(min(opts.periods, opts.z_ident_period + opts.z_ident_span - 1))]`, pattern `$(opts.z_ident_pattern)`, amplitude `$(opts.z_ident_shock)`")
        end
        println(io, "- Second-shock block: `$(opts.second_shock_name)[$(opts.policy_shock_period):$(min(opts.periods, opts.policy_shock_period + opts.policy_shock_span - 1))]`, pattern `$(opts.policy_shock_pattern)`, amplitude `$(opts.policy_shock)`")
        println(io, "- Actual floor periods: $(sum(actual_floor)) / $(opts.periods)")
        println(io, "- Linear sub-floor periods: $(sum(linear_violates)) / $(opts.periods)")
        println(io, "- Observation sigma: $(join(round.(obs_sigma, digits = 5), ", "))")
        println(io, "- Finite direct grid points: $(count(isfinite, direct_logpost)) / $(n_points)")
        println(io, "- Finite surrogate grid points: $(count(isfinite, surrogate_logpost)) / $(n_points)")
        println(io, "- Recovery failures: $(recovery_failures)")
        println(io, "- Solver warning count: $(length(warnings))")
        println(io, "- HMC requested: $(opts.hmc_draws > 0)")
        if opts.hmc_draws > 0
            println(io, "- HMC chains/warmup/draws: $(opts.hmc_chains) / $(opts.hmc_warmup) / $(opts.hmc_draws)")
            println(io, "- HMC warning count: $(length(hmc_warnings))")
        end
        println(io, "- Max linear-inversion residual norm: $(@sprintf("%.4g", max_resid_norm))")
        println(io, "- Max absolute recovered structural shock: $(@sprintf("%.4g", max_shock))")
        println(io)
        println(io, "## Surrogate Fit")
        println(io)
        println(io, "- Train RMSE: $(join(round.(surrogate.train_rmse, digits = 6), ", "))")
        println(io, "- Holdout-midpoint RMSE: $(join(round.(surrogate.holdout_rmse, digits = 6), ", "))")
        println(io)
        println(io, "## Posterior Grid")
        println(io)
        println(io, "| Parameter | True | Direct mean | Direct 90% interval | Surrogate mean | Surrogate 90% interval | Mean diff./sd | Overlap | Coverage |")
        println(io, "|---|---:|---:|---|---:|---|---:|:---:|:---:|")
        for row in rows
            dci = "[$(@sprintf("%.6g", row["direct_q05"])), $(@sprintf("%.6g", row["direct_q95"]))]"
            sci = "[$(@sprintf("%.6g", row["surrogate_q05"])), $(@sprintf("%.6g", row["surrogate_q95"]))]"
            coverage = row["direct_cover"] && row["surrogate_cover"] ? "Both" : row["direct_cover"] ? "Direct" : row["surrogate_cover"] ? "Surrogate" : "Neither"
            println(io, "| `$(row["parameter"])` | $(@sprintf("%.6g", row["true"])) | $(@sprintf("%.6g", row["direct_mean"])) | $dci | $(@sprintf("%.6g", row["surrogate_mean"])) | $sci | $(@sprintf("%.4g", row["mean_diff_direct_sd"])) | $(row["interval_overlap"]) | $coverage |")
        end
        println(io)
        if hmc_cmp !== nothing
            println(io, "## Matched Two-Parameter HMC")
            println(io)
            println(io, "| Parameter | True | Direct mean | Direct MCSE | Direct 90% interval | Surrogate mean | Surrogate MCSE | Surrogate 90% interval | Diff/MCSE | Overlap | Coverage |")
            println(io, "|---|---:|---:|---:|---|---:|---:|---|---:|:---:|:---:|")
            for row in hmc_cmp["rows"]
                dci = "[$(@sprintf("%.6g", row["direct_q05"])), $(@sprintf("%.6g", row["direct_q95"]))]"
                sci = "[$(@sprintf("%.6g", row["surrogate_q05"])), $(@sprintf("%.6g", row["surrogate_q95"]))]"
                coverage = row["direct_cover"] && row["surrogate_cover"] ? "Both" : row["direct_cover"] ? "Direct" : row["surrogate_cover"] ? "Surrogate" : "Neither"
                println(io, "| `$(row["parameter"])` | $(@sprintf("%.6g", row["true"])) | $(@sprintf("%.6g", row["direct_mean"])) | $(@sprintf("%.4g", row["direct_mcse"])) | $dci | $(@sprintf("%.6g", row["surrogate_mean"])) | $(@sprintf("%.4g", row["surrogate_mcse"])) | $sci | $(@sprintf("%.4g", row["mean_diff_combined_mcse"])) | $(row["interval_overlap"]) | $coverage |")
            end
            println(io)
            println(io, "- Direct post-warmup numerical errors: $(hmc_cmp["direct"]["n_numerical_errors"])")
            println(io, "- Surrogate post-warmup numerical errors: $(hmc_cmp["surrogate"]["n_numerical_errors"])")
            println(io, "- HMC warnings: $(length(hmc_warnings))")
            println(io, "- Direct mean acceptance: $(@sprintf("%.4g", hmc_cmp["direct"]["mean_accept"]))")
            println(io, "- Surrogate mean acceptance: $(@sprintf("%.4g", hmc_cmp["surrogate"]["mean_accept"]))")
            println(io, "- Strict zero-issue HMC diagnostic pass: $(hmc_strict_zero_issue_pass)")
            println(io, "- Negligible-issue HMC diagnostic pass: $(hmc_negligible_issue_pass)")
            println(io)
        end
        println(io, "## Checks")
        println(io)
        println(io, "- Two-parameter inversion-grid validation pass: $(grid_validation_pass)")
        if hmc_cmp !== nothing
            println(io, "- Matched two-parameter HMC validation pass: $(hmc_validation_pass)")
        end
        println(io, "- Overall validation pass: $(validation_pass)")
    end

    open(joinpath(out_dir, "comparison_table.tex"), "w") do io
        println(io, "\\begin{tabular}{lrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & True & Direct mean & Surrogate mean & Mean diff./sd \\\\")
        println(io, "\\midrule")
        for row in rows
            label = row["parameter"] == "std_z" ? "\$\\sigma_z\$" :
                    row["parameter"] == "std_a" ? "\$\\sigma_a\$" : "\$\\sigma_\\nu\$"
            println(io, "$label & $(@sprintf("%.6g", row["true"])) & $(@sprintf("%.6g", row["direct_mean"])) & $(@sprintf("%.6g", row["surrogate_mean"])) & $(@sprintf("%.4g", row["mean_diff_direct_sd"])) \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
    println("Wrote summary: $(joinpath(out_dir, "SUMMARY.md"))")
    println("Actual floor periods: $(sum(actual_floor)) / $(opts.periods)")
    println("Validation pass: $(validation_pass)")
    return payload
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_twoparam_validation(parse_twoparam_args(ARGS))
end
