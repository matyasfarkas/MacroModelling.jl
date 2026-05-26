#!/usr/bin/env julia

# Galí actual-floor inversion-grid validation.
#
# This is the next stage after `gali_obc_actual_floor_residual_grid_validation.jl`.
# The DGP is the corrected adverse eps_z episode where output and inflation fall
# and the OBC policy rate binds for several periods.  Unlike the known-shock
# check, this runner recovers shocks from the observations with the linear ROM1
# inversion filter, then evaluates the same recovered shocks through:
#
#   1. direct first-order OBC path evaluation, and
#   2. linear ROM1 plus an interpolated OBC-minus-ROM1 residual path.
#
# The object is still a one-parameter grid over std_z.  It validates the
# inversion-filter layer before launching any matched HMC exercise.

ENV["GKSwstype"] = "100"

using AxisKeys
using AdvancedHMC
using Distributions
using LogDensityProblems
using LinearAlgebra
using MacroModelling
using Printf
using Random
using Serialization
using Statistics

const INV_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(INV_REPO_ROOT, "scripts", "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(INV_REPO_ROOT, "scripts", "gali_obc_actual_floor_residual_grid_validation.jl"))

Base.@kwdef struct InversionGridOptions
    periods::Int = 24
    seed::Int = 20260515
    shock_scale::Float64 = 0.0
    elb_period::Int = 6
    elb_span::Int = 5
    elb_decay::Float64 = 1.0
    elb_shock_name::Symbol = FLOOR_DRIVER_SHOCK
    elb_shock::Float64 = 0.8
    grid_size::Int = 17
    train_thetas::Int = 9
    floor_tolerance_pct::Float64 = 0.25
    inv_maxit::Int = 20
    inv_tol::Float64 = 1e-9
    inv_lambda::Float64 = 1e-6
    hmc_warmup::Int = 0
    hmc_draws::Int = 0
    hmc_chains::Int = 1
    hmc_step_size::Float64 = 0.05
    hmc_max_depth::Int = 4
    hmc_fd_eps::Float64 = 1e-4
    out_dir::String = joinpath(INV_REPO_ROOT, ".local_artifacts", "gali_actual_floor_inversion_grid")
    run_id::String = "actual_floor_inversion_grid"
end

function parse_inversion_args(args)
    opts = InversionGridOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    return InversionGridOptions(
        periods = parse(Int, get(values, "periods", string(opts.periods))),
        seed = parse(Int, get(values, "seed", string(opts.seed))),
        shock_scale = parse(Float64, get(values, "shock-scale", string(opts.shock_scale))),
        elb_period = parse(Int, get(values, "elb-period", string(opts.elb_period))),
        elb_span = parse(Int, get(values, "elb-span", string(opts.elb_span))),
        elb_decay = parse(Float64, get(values, "elb-decay", string(opts.elb_decay))),
        elb_shock_name = Symbol(get(values, "elb-shock-name", string(opts.elb_shock_name))),
        elb_shock = parse(Float64, get(values, "elb-shock", string(opts.elb_shock))),
        grid_size = parse(Int, get(values, "grid-size", string(opts.grid_size))),
        train_thetas = parse(Int, get(values, "train-thetas", string(opts.train_thetas))),
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
        out_dir = get(values, "out-dir", opts.out_dir),
        run_id = get(values, "run-id", opts.run_id),
    )
end

function floor_options(opts::InversionGridOptions)
    return FloorGridOptions(
        periods = opts.periods,
        seed = opts.seed,
        shock_scale = opts.shock_scale,
        elb_period = opts.elb_period,
        elb_span = opts.elb_span,
        elb_decay = opts.elb_decay,
        elb_shock_name = opts.elb_shock_name,
        elb_shock = opts.elb_shock,
        grid_size = opts.grid_size,
        train_thetas = opts.train_thetas,
        floor_tolerance_pct = opts.floor_tolerance_pct,
        out_dir = opts.out_dir,
        run_id = opts.run_id,
    )
end

function transformed_observables_from_raw(raw_obs::AbstractVector, log_y_ss::Float64)
    return [
        100 * (raw_obs[1] - log_y_ss),
        100 * raw_obs[2],
        100 * raw_obs[3],
    ]
end

function build_linear_inversion_predict(model)
    local_model = deepcopy(model)
    base_params = Float64.(local_model.parameter_values)
    MacroModelling.write_parameters_input!(local_model, base_params, verbose = false)
    MacroModelling.solve!(local_model; algorithm = :first_order, dynamics = true, obc = false, silent = true)

    state_idx = collect(1:length(local_model.var))
    obs_idx = [find_idx(local_model.var, obs) for obs in FLOOR_OBS]
    _, raw_predict_tuple, nsss = build_matrix_rom_predict(local_model; state_idx = state_idx, obs_idx = obs_idx)
    log_y_ss = Float64(nsss[find_idx(local_model.var, :log_y)])
    eps_z_idx = find_idx(local_model.exo, FLOOR_DRIVER_SHOCK)

    function predict_tuple(state::AbstractVector, shock::AbstractVector, theta_vec::AbstractVector)
        shock_scaled = copy(shock)
        shock_scaled[eps_z_idx] *= theta_vec[1] / FLOOR_THETA_BASELINE
        raw_obs, state_next = raw_predict_tuple(state, shock_scaled, theta_vec)
        return transformed_observables_from_raw(raw_obs, log_y_ss), state_next
    end

    return predict_tuple, Float64.(nsss), [find_idx(local_model.exo, s) for s in (:eps_a, :eps_z, :eps_nu)]
end

function recover_linear_inversion_path(predict_tuple::Function,
                                       state0::Vector{Float64},
                                       structural_idx::Vector{Int},
                                       obs_data::Matrix{Float64},
                                       obs_sigma::Vector{Float64},
                                       theta::Float64,
                                       opts::InversionGridOptions)
    shock_sigmas = zeros(Float64, length(Gali_2015_chapter_3_obc.exo))
    shock_sigmas[structural_idx] .= 1.0
    state = copy(state0)
    states = zeros(Float64, length(state0), size(obs_data, 2))
    shocks = zeros(Float64, length(shock_sigmas), size(obs_data, 2))
    iterations = zeros(Int, size(obs_data, 2))
    residual_norms = zeros(Float64, size(obs_data, 2))
    eps_init = zeros(Float64, length(structural_idx))

    for t in 1:size(obs_data, 2)
        states[:, t] .= state
        local eps_full, state_next, ll_t
        try
            eps_full, state_next, ll_t = MacroModelling.inversion_step(
                predict_tuple,
                state,
                obs_data[:, t],
                [theta],
                obs_sigma,
                shock_sigmas,
                structural_idx;
                eps_init = eps_init,
                maxit = opts.inv_maxit,
                tol = opts.inv_tol,
                lambda = opts.inv_lambda,
            )
        catch e
            return (ok = false, states = states, shocks = shocks, iterations = iterations,
                    residual_norms = residual_norms, message = sprint(showerror, e), fail_period = t)
        end
        if !all(isfinite, eps_full) || !all(isfinite, state_next) || !isfinite(ll_t)
            return (ok = false, states = states, shocks = shocks, iterations = iterations,
                    residual_norms = residual_norms, message = "nonfinite inversion result", fail_period = t)
        end
        pred_obs, _ = predict_tuple(state, eps_full, [theta])
        shocks[:, t] .= eps_full
        state = Float64.(state_next)
        eps_init .= eps_full[structural_idx]
        iterations[t] = opts.inv_maxit
        residual_norms[t] = norm((obs_data[:, t] .- pred_obs) ./ obs_sigma)
    end
    return (ok = true, states = states, shocks = shocks, iterations = iterations,
            residual_norms = residual_norms, message = "", fail_period = 0)
end

function simulate_with_supplied_shocks(model, opts::InversionGridOptions, theta::Float64,
                                       shocks::Matrix{Float64}; ignore_obc::Bool,
                                       warnings::Union{Nothing,Vector{String}} = nothing)
    local_model = deepcopy(model)
    shocks_scaled = copy(shocks)
    eps_z_idx = find_idx(local_model.exo, FLOOR_DRIVER_SHOCK)
    shocks_scaled[eps_z_idx, :] .*= theta / FLOOR_THETA_BASELINE
    keyed_shocks = KeyedArray(shocks_scaled; Shocks = local_model.timings.exo, Periods = 1:opts.periods)
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

function inversion_loglik(obs::Matrix{Float64}, pred::Matrix{Float64}, sigma::Vector{Float64},
                          shocks::Matrix{Float64}, structural_idx::Vector{Int})
    resid = (obs .- pred) ./ sigma
    structural_shocks = shocks[structural_idx, :]
    n_obs = length(obs)
    n_shock = length(structural_shocks)
    obs_const = n_obs * log(2pi) + 2 * size(obs, 2) * sum(log.(sigma))
    shock_const = n_shock * log(2pi)
    return -0.5 * (sum(resid .^ 2) + sum(structural_shocks .^ 2) + obs_const + shock_const)
end

function inversion_log_prior(theta::Float64)
    tol = 64 * eps(Float64) * max(1.0, abs(theta), abs(FLOOR_THETA_BASELINE))
    (theta < FLOOR_PRIOR_LOWER - tol || theta > FLOOR_PRIOR_UPPER + tol) && return -Inf
    theta_clamped = clamp(theta, FLOOR_PRIOR_LOWER, FLOOR_PRIOR_UPPER)
    return Distributions.logpdf(Distributions.Normal(log(FLOOR_THETA_BASELINE), FLOOR_PRIOR_LOG_SD), log(theta_clamped))
end

struct InversionResidualSurrogate
    log_thetas::Vector{Float64}
    residuals::Array{Float64,3}
    train_rmse::Vector{Float64}
    holdout_rmse::Vector{Float64}
    train_recovery_failures::Int
    holdout_recovery_failures::Int
end

function interpolate_residual(s::InversionResidualSurrogate, theta::Float64)
    x = log(theta)
    if x <= first(s.log_thetas)
        return s.residuals[:, :, 1]
    elseif x >= last(s.log_thetas)
        return s.residuals[:, :, end]
    end
    hi = searchsortedfirst(s.log_thetas, x)
    lo = hi - 1
    w = (x - s.log_thetas[lo]) / (s.log_thetas[hi] - s.log_thetas[lo])
    return (1 - w) .* s.residuals[:, :, lo] .+ w .* s.residuals[:, :, hi]
end

function direct_and_linear_paths(model, opts::InversionGridOptions, theta::Float64,
                                 shocks::Matrix{Float64};
                                 warnings::Union{Nothing,Vector{String}} = nothing)
    direct = simulate_with_supplied_shocks(model, opts, theta, shocks; ignore_obc = false, warnings = warnings)
    linear = simulate_with_supplied_shocks(model, opts, theta, shocks; ignore_obc = true, warnings = warnings)
    return direct, linear
end

function train_inversion_residual_surrogate(model, opts::InversionGridOptions,
                                            predict_tuple::Function,
                                            state0::Vector{Float64},
                                            structural_idx::Vector{Int},
                                            obs_data::Matrix{Float64},
                                            obs_sigma::Vector{Float64};
                                            warnings::Union{Nothing,Vector{String}} = nothing)
    thetas = theta_grid(opts.train_thetas)
    residuals = zeros(Float64, length(FLOOR_OBS), opts.periods, length(thetas))
    recovery_failures = 0
    for (i, theta) in pairs(thetas)
        recovered = recover_linear_inversion_path(predict_tuple, state0, structural_idx, obs_data, obs_sigma, theta, opts)
        if !recovered.ok
            recovery_failures += 1
            residuals[:, :, i] .= NaN
            continue
        end
        direct, linear = direct_and_linear_paths(model, opts, theta, recovered.shocks; warnings = warnings)
        residuals[:, :, i] .= direct .- linear
    end
    if any(!isfinite, residuals)
        error("Inversion residual surrogate training failed at $recovery_failures theta grid points.")
    end
    log_thetas = log.(thetas)
    train_err = zeros(Float64, length(FLOOR_OBS), length(thetas) * opts.periods)
    col = 1
    tmp = InversionResidualSurrogate(log_thetas, residuals, zeros(length(FLOOR_OBS)), zeros(length(FLOOR_OBS)), recovery_failures, 0)
    for (i, theta) in pairs(thetas)
        train_err[:, col:col+opts.periods-1] .= interpolate_residual(tmp, theta) .- residuals[:, :, i]
        col += opts.periods
    end
    train_rmse = sqrt.(vec(Statistics.mean(train_err .^ 2, dims = 2)))

    midpoints = exp.((log_thetas[1:end-1] .+ log_thetas[2:end]) ./ 2)
    valid_err = zeros(Float64, length(FLOOR_OBS), length(midpoints) * opts.periods)
    holdout_failures = 0
    col = 1
    for theta in midpoints
        recovered = recover_linear_inversion_path(predict_tuple, state0, structural_idx, obs_data, obs_sigma, theta, opts)
        if !recovered.ok
            holdout_failures += 1
            valid_err[:, col:col+opts.periods-1] .= Inf
        else
            direct, linear = direct_and_linear_paths(model, opts, theta, recovered.shocks; warnings = warnings)
            valid_err[:, col:col+opts.periods-1] .= interpolate_residual(tmp, theta) .- (direct .- linear)
        end
        col += opts.periods
    end
    holdout_failures > 0 && error("Inversion residual surrogate holdout failed at $holdout_failures midpoint(s).")
    holdout_rmse = sqrt.(vec(Statistics.mean(valid_err .^ 2, dims = 2)))
    return InversionResidualSurrogate(log_thetas, residuals, train_rmse, holdout_rmse, recovery_failures, holdout_failures)
end

function inversion_objective_at_theta(model, opts::InversionGridOptions,
                                      predict_tuple::Function,
                                      state0::Vector{Float64},
                                      structural_idx::Vector{Int},
                                      obs_data::Matrix{Float64},
                                      obs_sigma::Vector{Float64},
                                      surrogate::InversionResidualSurrogate,
                                      theta::Float64;
                                      warnings::Union{Nothing,Vector{String}} = nothing)
    lp = inversion_log_prior(theta)
    isfinite(lp) || return (ok = false, direct_logpost = -Inf, surrogate_logpost = -Inf,
                            recovered = nothing, direct_path = nothing, surrogate_path = nothing,
                            message = "prior support")
    recovered = recover_linear_inversion_path(predict_tuple, state0, structural_idx, obs_data, obs_sigma, theta, opts)
    if !recovered.ok
        return (ok = false, direct_logpost = -Inf, surrogate_logpost = -Inf,
                recovered = recovered, direct_path = nothing, surrogate_path = nothing,
                message = recovered.message)
    end
    direct_path, linear_path = direct_and_linear_paths(model, opts, theta, recovered.shocks; warnings = warnings)
    surrogate_path = linear_path .+ interpolate_residual(surrogate, theta)
    direct_logpost = lp + inversion_loglik(obs_data, direct_path, obs_sigma, recovered.shocks, structural_idx)
    surrogate_logpost = lp + inversion_loglik(obs_data, surrogate_path, obs_sigma, recovered.shocks, structural_idx)
    return (ok = isfinite(direct_logpost) && isfinite(surrogate_logpost),
            direct_logpost = direct_logpost,
            surrogate_logpost = surrogate_logpost,
            recovered = recovered,
            direct_path = direct_path,
            surrogate_path = surrogate_path,
            message = "")
end

sigmoid_stable(x::Float64) = x >= 0 ? 1 / (1 + exp(-x)) : begin
    ex = exp(x)
    ex / (1 + ex)
end

function hmc_z_to_theta(z::Float64)
    s = sigmoid_stable(z)
    return FLOOR_PRIOR_LOWER + (FLOOR_PRIOR_UPPER - FLOOR_PRIOR_LOWER) * s
end

function theta_to_hmc_z(theta::Float64)
    s = clamp((theta - FLOOR_PRIOR_LOWER) / (FLOOR_PRIOR_UPPER - FLOOR_PRIOR_LOWER),
              1e-10, 1 - 1e-10)
    return log(s / (1 - s))
end

function hmc_logjac(z::Float64)
    s = sigmoid_stable(z)
    if !(0 < s < 1)
        return -Inf
    end
    return log(FLOOR_PRIOR_UPPER - FLOOR_PRIOR_LOWER) + log(s) + log1p(-s)
end

struct OneParamLogDensity
    logpost_z::Function
    fd_eps::Float64
end

LogDensityProblems.logdensity(p::OneParamLogDensity, z) = p.logpost_z(Float64(z[1]))
LogDensityProblems.dimension(::OneParamLogDensity) = 1
LogDensityProblems.capabilities(::Type{OneParamLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::OneParamLogDensity, z)
    z0 = Float64(z[1])
    f0 = p.logpost_z(z0)
    zp = z0 + p.fd_eps
    zm = z0 - p.fd_eps
    fp = p.logpost_z(zp)
    fm = p.logpost_z(zm)
    grad = if isfinite(fp) && isfinite(fm)
        (fp - fm) / (2 * p.fd_eps)
    elseif isfinite(fp) && isfinite(f0)
        (fp - f0) / p.fd_eps
    elseif isfinite(fm) && isfinite(f0)
        (f0 - fm) / p.fd_eps
    else
        NaN
    end
    return f0, [grad]
end

function make_hmc_logpost(logpost_theta::Function)
    return function (z::Float64)
        theta = hmc_z_to_theta(z)
        lj = hmc_logjac(z)
        isfinite(lj) || return -Inf
        lp = logpost_theta(theta)
        isfinite(lp) || return -Inf
        return lp + lj
    end
end

function run_one_param_hmc(name::String, logpost_theta::Function, opts::InversionGridOptions; seed::Int)
    opts.hmc_draws > 0 || return nothing
    logpost_z = make_hmc_logpost(logpost_theta)
    z_init = [theta_to_hmc_z(FLOOR_THETA_TRUE)]
    logdensity = OneParamLogDensity(logpost_z, opts.hmc_fd_eps)
    metric = UnitEuclideanMetric(1)
    hamiltonian = Hamiltonian(metric, logdensity)
    integrator = Leapfrog(opts.hmc_step_size)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth = opts.hmc_max_depth)))
    adaptor = StepSizeAdaptor(0.8, integrator)
    chains = Vector{Dict{String,Any}}()

    println("Running $name one-parameter NUTS: chains=$(opts.hmc_chains), warmup=$(opts.hmc_warmup), draws=$(opts.hmc_draws)")
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
        theta_all = [hmc_z_to_theta(Float64(s[1])) for s in samples_z]
        theta_post = theta_all[(opts.hmc_warmup + 1):end]
        stats_post = stats[(opts.hmc_warmup + 1):end]
        numerical_errors = [getproperty(s, :numerical_error) for s in stats_post]
        accept_rates = [getproperty(s, :acceptance_rate) for s in stats_post]
        tree_depths = [getproperty(s, :tree_depth) for s in stats_post]
        push!(chains, Dict{String,Any}(
            "name" => name,
            "chain" => chain,
            "theta_post" => theta_post,
            "theta_all" => theta_all,
            "n_numerical_errors" => sum(numerical_errors),
            "mean_accept" => Statistics.mean(accept_rates),
            "max_tree_depth" => maximum(tree_depths),
            "elapsed_s" => elapsed,
        ))
        println("  $name chain $chain complete: elapsed=$(round(elapsed, digits = 2))s, numerical_errors=$(sum(numerical_errors))")
    end
    return chains
end

function hmc_summary(chains::Vector{Dict{String,Any}})
    theta = vcat([c["theta_post"] for c in chains]...)
    n = length(theta)
    return Dict{String,Any}(
        "mean" => Statistics.mean(theta),
        "mcse" => Statistics.std(theta) / sqrt(n),
        "q05" => Statistics.quantile(theta, 0.05),
        "q95" => Statistics.quantile(theta, 0.95),
        "n_draws" => n,
        "n_numerical_errors" => sum(c["n_numerical_errors"] for c in chains),
        "mean_accept" => Statistics.mean([c["mean_accept"] for c in chains]),
        "max_tree_depth" => maximum([c["max_tree_depth"] for c in chains]),
        "elapsed_s" => sum(c["elapsed_s"] for c in chains),
    )
end

function hmc_comparison(direct_chains::Vector{Dict{String,Any}},
                        surrogate_chains::Vector{Dict{String,Any}})
    direct = hmc_summary(direct_chains)
    surrogate = hmc_summary(surrogate_chains)
    denom = sqrt(direct["mcse"]^2 + surrogate["mcse"]^2)
    diff_mcse = denom > 0 ? (surrogate["mean"] - direct["mean"]) / denom : NaN
    overlap = max(direct["q05"], surrogate["q05"]) <= min(direct["q95"], surrogate["q95"])
    direct_cover = direct["q05"] <= FLOOR_THETA_TRUE <= direct["q95"]
    surrogate_cover = surrogate["q05"] <= FLOOR_THETA_TRUE <= surrogate["q95"]
    return Dict{String,Any}(
        "direct" => direct,
        "surrogate" => surrogate,
        "mean_diff_combined_mcse" => diff_mcse,
        "interval_overlap" => overlap,
        "direct_cover" => direct_cover,
        "surrogate_cover" => surrogate_cover,
    )
end

function run_inversion_grid_validation(opts::InversionGridOptions)
    opts.elb_shock_name == FLOOR_DRIVER_SHOCK || error("This validation estimates $(FLOOR_THETA_NAME); use --elb-shock-name=$(FLOOR_DRIVER_SHOCK).")
    out_dir = joinpath(opts.out_dir, opts.run_id)
    mkpath(out_dir)
    model = Gali_2015_chapter_3_obc
    floor_opts = floor_options(opts)
    all_warnings = String[]

    println("Generating actual-floor OBC DGP...")
    dgp_obs, dgp_shocks = simulate_observables(model, floor_opts, FLOOR_THETA_TRUE; ignore_obc = false, warnings = all_warnings)
    lin_true, _ = simulate_observables(model, floor_opts, FLOOR_THETA_TRUE; ignore_obc = true, warnings = all_warnings)
    floor_pct = 400 * log(Float64(model.parameter_values[find_idx(model.parameters, :R̄)]))
    actual_floor = dgp_obs[3, :] .<= floor_pct + opts.floor_tolerance_pct
    linear_violates = lin_true[3, :] .< floor_pct
    obs_sigma = max.(0.05 .* vec(Statistics.std(dgp_obs, dims = 2; corrected = false)), 1e-3)

    println("Building linear ROM1 inversion predictor...")
    predict_tuple, state0, structural_idx = build_linear_inversion_predict(model)

    println("Training inversion-path OBC-minus-linear residual surrogate...")
    surrogate = train_inversion_residual_surrogate(
        model,
        opts,
        predict_tuple,
        state0,
        structural_idx,
        dgp_obs,
        obs_sigma;
        warnings = all_warnings,
    )

    grid = theta_grid(opts.grid_size)
    direct_logpost = fill(-Inf, length(grid))
    surrogate_logpost = fill(-Inf, length(grid))
    recovery_failures = 0
    max_residual_norm = 0.0
    max_recovered_shock = 0.0
    direct_paths = zeros(Float64, length(FLOOR_OBS), opts.periods, length(grid))
    surrogate_paths = zeros(Float64, length(FLOOR_OBS), opts.periods, length(grid))

    for (i, theta) in pairs(grid)
        eval = inversion_objective_at_theta(
            model,
            opts,
            predict_tuple,
            state0,
            structural_idx,
            dgp_obs,
            obs_sigma,
            surrogate,
            theta;
            warnings = all_warnings,
        )
        if !eval.ok
            recovery_failures += 1
            continue
        end
        recovered = eval.recovered
        max_residual_norm = max(max_residual_norm, maximum(recovered.residual_norms))
        max_recovered_shock = max(max_recovered_shock, maximum(abs.(recovered.shocks[structural_idx, :])))
        direct_paths[:, :, i] .= eval.direct_path
        surrogate_paths[:, :, i] .= eval.surrogate_path
        direct_logpost[i] = eval.direct_logpost
        surrogate_logpost[i] = eval.surrogate_logpost
    end

    direct = posterior_summary(grid, direct_logpost)
    surrogate_summary = posterior_summary(grid, surrogate_logpost)
    overlap = max(direct["q05"], surrogate_summary["q05"]) <= min(direct["q95"], surrogate_summary["q95"])
    tol = 1e-12
    direct_cover = direct["q05"] <= FLOOR_THETA_TRUE + tol && FLOOR_THETA_TRUE <= direct["q95"] + tol
    surrogate_cover = surrogate_summary["q05"] <= FLOOR_THETA_TRUE + tol && FLOOR_THETA_TRUE <= surrogate_summary["q95"] + tol
    mean_diff_sd = abs(direct["mean"] - surrogate_summary["mean"]) / max(direct["sd"], eps(Float64))
    finite_direct = count(isfinite, direct_logpost)
    finite_surrogate = count(isfinite, surrogate_logpost)
    hmc_warnings = String[]
    direct_hmc_chains = nothing
    surrogate_hmc_chains = nothing
    hmc_cmp = nothing
    if opts.hmc_draws > 0
        direct_logpost_theta = function (theta::Float64)
            eval = inversion_objective_at_theta(
                model,
                opts,
                predict_tuple,
                state0,
                structural_idx,
                dgp_obs,
                obs_sigma,
                surrogate,
                theta;
                warnings = hmc_warnings,
            )
            return eval.direct_logpost
        end
        surrogate_logpost_theta = function (theta::Float64)
            eval = inversion_objective_at_theta(
                model,
                opts,
                predict_tuple,
                state0,
                structural_idx,
                dgp_obs,
                obs_sigma,
                surrogate,
                theta;
                warnings = hmc_warnings,
            )
            return eval.surrogate_logpost
        end
        hmc_seed = opts.seed + 101
        direct_hmc_chains = run_one_param_hmc("direct_obc_inversion", direct_logpost_theta, opts; seed = hmc_seed)
        surrogate_hmc_chains = run_one_param_hmc("surrogate_inversion", surrogate_logpost_theta, opts; seed = hmc_seed)
        hmc_cmp = hmc_comparison(direct_hmc_chains, surrogate_hmc_chains)
        serialize(joinpath(out_dir, "direct_hmc_chains.jls"), direct_hmc_chains)
        serialize(joinpath(out_dir, "surrogate_hmc_chains.jls"), surrogate_hmc_chains)
        serialize(joinpath(out_dir, "hmc_comparison_payload.jls"), hmc_cmp)
    end
    unique_warnings = sort(unique(all_warnings))
    unique_hmc_warnings = sort(unique(hmc_warnings))
    grid_pass = isempty(all_warnings) && recovery_failures == 0 && overlap && direct_cover && surrogate_cover
    hmc_pass = hmc_cmp === nothing ||
        (isempty(hmc_warnings) &&
         hmc_cmp["direct"]["n_numerical_errors"] == 0 &&
         hmc_cmp["surrogate"]["n_numerical_errors"] == 0 &&
         hmc_cmp["interval_overlap"] &&
         hmc_cmp["direct_cover"] &&
         hmc_cmp["surrogate_cover"] &&
         abs(hmc_cmp["mean_diff_combined_mcse"]) <= 2.0)

    payload = Dict{String,Any}(
        "options" => opts,
        "dgp_obs" => dgp_obs,
        "dgp_shocks" => dgp_shocks,
        "linear_true_obs" => lin_true,
        "obs_sigma" => obs_sigma,
        "actual_floor" => actual_floor,
        "linear_violates" => linear_violates,
        "surrogate" => surrogate,
        "grid" => grid,
        "direct" => direct,
        "surrogate_summary" => surrogate_summary,
        "direct_logpost" => direct_logpost,
        "surrogate_logpost" => surrogate_logpost,
        "direct_paths" => direct_paths,
        "surrogate_paths" => surrogate_paths,
        "mean_diff_direct_sd" => mean_diff_sd,
        "interval_overlap" => overlap,
        "direct_cover" => direct_cover,
        "surrogate_cover" => surrogate_cover,
        "finite_direct_grid_points" => finite_direct,
        "finite_surrogate_grid_points" => finite_surrogate,
        "recovery_failures" => recovery_failures,
        "max_linear_inversion_residual_norm" => max_residual_norm,
        "max_abs_recovered_structural_shock" => max_recovered_shock,
        "solver_warning_count_total" => length(all_warnings),
        "solver_warnings_unique" => unique_warnings,
        "hmc_warning_count_total" => length(hmc_warnings),
        "hmc_warnings_unique" => unique_hmc_warnings,
        "direct_hmc_chains" => direct_hmc_chains,
        "surrogate_hmc_chains" => surrogate_hmc_chains,
        "hmc_comparison" => hmc_cmp,
        "grid_validation_pass" => grid_pass,
        "hmc_validation_pass" => hmc_pass,
    )
    serialize(joinpath(out_dir, "actual_floor_inversion_grid_payload.jls"), payload)

    open(joinpath(out_dir, "SUMMARY.md"), "w") do io
        println(io, "# Galí Actual-Floor Inversion Grid Validation")
        println(io)
        println(io, "- Parameter: `$(FLOOR_THETA_NAME)`")
        println(io, "- True value: $(FLOOR_THETA_TRUE)")
        println(io, "- DGP: first-order OBC path with adverse `$(opts.elb_shock_name)` block")
        println(io, "- Inversion architecture: linear ROM1 shock recovery, common recovered shocks for direct and surrogate evaluations")
        println(io, "- Surrogate target: inversion-path `OBC observables - linear observables`")
        println(io, "- Prior bounds: [$(FLOOR_PRIOR_LOWER), $(FLOOR_PRIOR_UPPER)]")
        println(io, "- Actual floor periods: $(sum(actual_floor)) / $(opts.periods)")
        println(io, "- Linear sub-floor periods: $(sum(linear_violates)) / $(opts.periods)")
        println(io, "- Observation sigma: $(join(round.(obs_sigma, digits = 5), ", "))")
        println(io, "- Finite direct grid points: $(finite_direct) / $(length(grid))")
        println(io, "- Finite surrogate grid points: $(finite_surrogate) / $(length(grid))")
        println(io, "- Linear inversion recovery failures on posterior grid: $(recovery_failures)")
        println(io, "- Max linear-inversion residual norm: $(@sprintf("%.4g", max_residual_norm))")
        println(io, "- Max absolute recovered structural shock: $(@sprintf("%.4g", max_recovered_shock))")
        println(io, "- Solver warning count: $(length(all_warnings))")
        println(io, "- HMC requested: $(opts.hmc_draws > 0)")
        if opts.hmc_draws > 0
            println(io, "- HMC chains/warmup/draws: $(opts.hmc_chains) / $(opts.hmc_warmup) / $(opts.hmc_draws)")
            println(io, "- HMC warning count: $(length(hmc_warnings))")
        end
        if !isempty(unique_warnings)
            println(io, "- Unique solver warnings: $(join(unique_warnings, " | "))")
        end
        if !isempty(unique_hmc_warnings)
            println(io, "- Unique HMC-time solver warnings: $(join(unique_hmc_warnings, " | "))")
        end
        println(io)
        println(io, "## Surrogate Fit")
        println(io)
        println(io, "- Train RMSE: $(join(round.(surrogate.train_rmse, digits = 6), ", "))")
        println(io, "- Holdout-midpoint RMSE: $(join(round.(surrogate.holdout_rmse, digits = 6), ", "))")
        println(io)
        println(io, "## Posterior Grid")
        println(io)
        println(io, "| Quantity | Direct OBC, common ROM1 inversion | Linear+residual surrogate |")
        println(io, "|---|---:|---:|")
        println(io, "| Mean | $(@sprintf("%.6g", direct["mean"])) | $(@sprintf("%.6g", surrogate_summary["mean"])) |")
        println(io, "| MAP | $(@sprintf("%.6g", direct["map"])) | $(@sprintf("%.6g", surrogate_summary["map"])) |")
        println(io, "| 90% interval | [$(@sprintf("%.6g", direct["q05"])), $(@sprintf("%.6g", direct["q95"]))] | [$(@sprintf("%.6g", surrogate_summary["q05"])), $(@sprintf("%.6g", surrogate_summary["q95"]))] |")
        println(io)
        if hmc_cmp !== nothing
            println(io, "## Matched One-Parameter HMC")
            println(io)
            println(io, "| Quantity | Direct OBC, common ROM1 inversion | Linear+residual surrogate |")
            println(io, "|---|---:|---:|")
            println(io, "| Draws | $(hmc_cmp["direct"]["n_draws"]) | $(hmc_cmp["surrogate"]["n_draws"]) |")
            println(io, "| Mean | $(@sprintf("%.6g", hmc_cmp["direct"]["mean"])) | $(@sprintf("%.6g", hmc_cmp["surrogate"]["mean"])) |")
            println(io, "| MCSE | $(@sprintf("%.4g", hmc_cmp["direct"]["mcse"])) | $(@sprintf("%.4g", hmc_cmp["surrogate"]["mcse"])) |")
            println(io, "| 90% interval | [$(@sprintf("%.6g", hmc_cmp["direct"]["q05"])), $(@sprintf("%.6g", hmc_cmp["direct"]["q95"]))] | [$(@sprintf("%.6g", hmc_cmp["surrogate"]["q05"])), $(@sprintf("%.6g", hmc_cmp["surrogate"]["q95"]))] |")
            println(io, "| Numerical errors | $(hmc_cmp["direct"]["n_numerical_errors"]) | $(hmc_cmp["surrogate"]["n_numerical_errors"]) |")
            println(io, "| Mean acceptance | $(@sprintf("%.4g", hmc_cmp["direct"]["mean_accept"])) | $(@sprintf("%.4g", hmc_cmp["surrogate"]["mean_accept"])) |")
            println(io)
            println(io, "- HMC mean difference / combined MCSE: $(@sprintf("%.4g", hmc_cmp["mean_diff_combined_mcse"]))")
            println(io, "- HMC interval overlap: $(hmc_cmp["interval_overlap"])")
            println(io, "- HMC direct true coverage: $(hmc_cmp["direct_cover"])")
            println(io, "- HMC surrogate true coverage: $(hmc_cmp["surrogate_cover"])")
            println(io)
        end
        println(io, "## Checks")
        println(io)
        println(io, "- Mean difference / direct posterior sd: $(@sprintf("%.4g", mean_diff_sd))")
        println(io, "- Interval overlap: $(overlap)")
        println(io, "- Direct true coverage: $(direct_cover)")
        println(io, "- Surrogate true coverage: $(surrogate_cover)")
        println(io, "- Inversion-grid validation pass: $(grid_pass)")
        if hmc_cmp !== nothing
            println(io, "- Matched one-parameter HMC validation pass: $(hmc_pass)")
        end
    end

    open(joinpath(out_dir, "comparison_table.tex"), "w") do io
        println(io, "\\begin{tabular}{lrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & True & Direct mean & Surrogate mean & Mean diff./sd \\\\")
        println(io, "\\midrule")
        println(io, "\$\\sigma_z\$ & $(@sprintf("%.6g", FLOOR_THETA_TRUE)) & $(@sprintf("%.6g", direct["mean"])) & $(@sprintf("%.6g", surrogate_summary["mean"])) & $(@sprintf("%.4g", mean_diff_sd)) \\\\")
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end

    println("Wrote summary: $(joinpath(out_dir, "SUMMARY.md"))")
    println("Actual floor periods: $(sum(actual_floor)) / $(opts.periods)")
    println("Direct mean: $(direct["mean"]), surrogate mean: $(surrogate_summary["mean"])")
    return payload
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_inversion_grid_validation(parse_inversion_args(ARGS))
end
