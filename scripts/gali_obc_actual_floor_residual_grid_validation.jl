#!/usr/bin/env julia

# Path-level validation for the Galí actual-floor OBC episode.
#
# This is a fast first check after constructing a stochastic path where the
# actual OBC policy rate stays at the floor for several periods.  It treats the
# OBC first-order simulator as the nonlinear benchmark and learns the
# observable residual:
#
#     y_obc_same_shocks(theta) - y_linear_same_shocks(theta)
#
# for output, inflation, and the policy rate.  The validation compares a direct
# OBC posterior grid for std_z to a linear+residual-surrogate posterior grid.

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

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "scripts", "gali_obc_stochastic_comparison_plot.jl"))

const FLOOR_OBS = [:log_y, :pi_ann, :i_ann]
const FLOOR_DRIVER_SHOCK = :eps_z
const FLOOR_THETA_NAME = :std_z
const FLOOR_THETA_BASELINE = 0.05
const FLOOR_THETA_TRUE = 0.05
const FLOOR_PRIOR_LOG_SD = 0.35
const FLOOR_PRIOR_LOWER = 0.95 * FLOOR_THETA_BASELINE
const FLOOR_PRIOR_UPPER = 1.05 * FLOOR_THETA_BASELINE

Base.@kwdef struct FloorGridOptions
    periods::Int = 24
    seed::Int = 20260515
    shock_scale::Float64 = 0.0
    elb_period::Int = 6
    elb_span::Int = 5
    elb_decay::Float64 = 1.0
    elb_shock_name::Symbol = FLOOR_DRIVER_SHOCK
    elb_shock::Float64 = 0.8
    grid_size::Int = 9
    train_thetas::Int = 9
    floor_tolerance_pct::Float64 = 0.25
    out_dir::String = joinpath(REPO_ROOT, ".local_artifacts", "gali_actual_floor_residual_grid")
    run_id::String = "actual_floor_grid"
end

struct PathResidualSurrogate
    log_thetas::Vector{Float64}
    residuals::Array{Float64,3}
    train_rmse::Vector{Float64}
    loo_rmse::Vector{Float64}
end

mutable struct WarningCaptureLogger <: Logging.AbstractLogger
    messages::Vector{String}
end

Logging.min_enabled_level(::WarningCaptureLogger) = Logging.Warn
Logging.shouldlog(::WarningCaptureLogger, level, _module, group, id) = level >= Logging.Warn
Logging.catch_exceptions(::WarningCaptureLogger) = false

function Logging.handle_message(logger::WarningCaptureLogger, level, message, _module, group, id, file, line; kwargs...)
    level >= Logging.Warn && push!(logger.messages, string(message))
    return nothing
end

function parse_floor_args(args)
    opts = FloorGridOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    return FloorGridOptions(
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
        out_dir = get(values, "out-dir", opts.out_dir),
        run_id = get(values, "run-id", opts.run_id),
    )
end

function theta_grid(n::Int)
    lo, hi = log(FLOOR_PRIOR_LOWER), log(FLOOR_PRIOR_UPPER)
    vals = exp.(collect(range(lo, hi; length = n)))
    if !any(isapprox.(vals, FLOOR_THETA_TRUE; rtol = 1e-12, atol = 1e-14))
        push!(vals, FLOOR_THETA_TRUE)
    end
    return sort(vals)
end

function log_prior(theta::Float64)
    (theta < FLOOR_PRIOR_LOWER || theta > FLOOR_PRIOR_UPPER) && return -Inf
    return Distributions.logpdf(Distributions.Normal(log(FLOOR_THETA_BASELINE), FLOOR_PRIOR_LOG_SD), log(theta))
end

function effective_shock_matrix(model, opts::FloorGridOptions, theta::Float64)
    stoch_opts = StochCompareOptions(
        periods = opts.periods,
        seed = opts.seed,
        shock_scale = opts.shock_scale,
        elb_period = opts.elb_period,
        elb_shock_name = opts.elb_shock_name,
        elb_shock = opts.elb_shock,
        elb_span = opts.elb_span,
        elb_decay = opts.elb_decay,
        tail_periods = 0,
        floor_tolerance_pct = opts.floor_tolerance_pct,
        out_dir = opts.out_dir,
    )
    shocks = build_stochastic_shocks(model, stoch_opts)
    driver_idx = find_idx(model.exo, opts.elb_shock_name)
    shocks[driver_idx, :] .*= theta / FLOOR_THETA_BASELINE
    return shocks
end

function simulate_observables(model, opts::FloorGridOptions, theta::Float64; ignore_obc::Bool,
                              warnings::Union{Nothing,Vector{String}} = nothing)
    # MacroModelling models carry mutable solution state. Use a fresh copy for
    # each path evaluation so repeated OBC solves do not contaminate the grid.
    local_model = deepcopy(model)
    shocks = effective_shock_matrix(local_model, opts, theta)
    keyed_shocks = KeyedArray(shocks; Shocks = local_model.timings.exo, Periods = 1:opts.periods)
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
    return obs, shocks
end

function interpolate_residual(s::PathResidualSurrogate, theta::Float64)
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

function path_residual(model, opts::FloorGridOptions, theta::Float64; warnings::Union{Nothing,Vector{String}} = nothing)
    obc_obs, _ = simulate_observables(model, opts, theta; ignore_obc = false, warnings = warnings)
    lin_obs, _ = simulate_observables(model, opts, theta; ignore_obc = true, warnings = warnings)
    return obc_obs .- lin_obs
end

function train_path_residual_surrogate(model, opts::FloorGridOptions;
                                       train_warnings::Union{Nothing,Vector{String}} = nothing,
                                       holdout_warnings::Union{Nothing,Vector{String}} = nothing)
    thetas = theta_grid(opts.train_thetas)
    residuals = zeros(Float64, length(FLOOR_OBS), opts.periods, length(thetas))
    for (i, theta) in pairs(thetas)
        residuals[:, :, i] .= path_residual(model, opts, theta; warnings = train_warnings)
    end
    s = PathResidualSurrogate(log.(thetas), residuals, zeros(length(FLOOR_OBS)), zeros(length(FLOOR_OBS)))

    train_err = zeros(Float64, length(FLOOR_OBS), length(thetas) * opts.periods)
    col = 1
    for (i, theta) in pairs(thetas)
        train_err[:, col:col+opts.periods-1] .= interpolate_residual(s, theta) .- residuals[:, :, i]
        col += opts.periods
    end
    train_rmse = sqrt.(vec(Statistics.mean(train_err .^ 2, dims = 2)))

    midpoints = exp.((log.(thetas[1:end-1]) .+ log.(thetas[2:end])) ./ 2)
    valid_err = zeros(Float64, length(FLOOR_OBS), length(midpoints) * opts.periods)
    col = 1
    for theta in midpoints
        valid_err[:, col:col+opts.periods-1] .= interpolate_residual(s, theta) .- path_residual(model, opts, theta; warnings = holdout_warnings)
        col += opts.periods
    end
    loo_rmse = sqrt.(vec(Statistics.mean(valid_err .^ 2, dims = 2)))
    return PathResidualSurrogate(log.(thetas), residuals, train_rmse, loo_rmse)
end

function surrogate_observables(model, opts::FloorGridOptions, theta::Float64, surrogate::PathResidualSurrogate;
                               warnings::Union{Nothing,Vector{String}} = nothing)
    lin_obs, _ = simulate_observables(model, opts, theta; ignore_obc = true, warnings = warnings)
    return lin_obs .+ interpolate_residual(surrogate, theta)
end

function loglik(obs::Matrix{Float64}, pred::Matrix{Float64}, sigma::Vector{Float64})
    resid = (obs .- pred) ./ sigma
    return -0.5 * sum(resid .^ 2) - size(obs, 2) * sum(log.(sqrt(2pi) .* sigma))
end

function posterior_summary(grid::Vector{Float64}, logpost::Vector{Float64})
    finite = isfinite.(logpost)
    weights = zeros(Float64, length(grid))
    weights[finite] .= exp.(logpost[finite] .- maximum(logpost[finite]))
    weights ./= sum(weights)
    cdf = cumsum(weights)
    mean_theta = sum(weights .* grid)
    return Dict{String,Any}(
        "grid" => grid,
        "logpost" => logpost,
        "weights" => weights,
        "mean" => mean_theta,
        "sd" => sqrt(sum(weights .* (grid .- mean_theta) .^ 2)),
        "q05" => grid[findfirst(>=(0.05), cdf)],
        "q95" => grid[findfirst(>=(0.95), cdf)],
        "map" => grid[argmax(logpost)],
    )
end

function run_floor_grid_validation(opts::FloorGridOptions)
    out_dir = joinpath(opts.out_dir, opts.run_id)
    mkpath(out_dir)
    model = Gali_2015_chapter_3_obc
    opts.elb_shock_name == FLOOR_DRIVER_SHOCK || error("This validation estimates $(FLOOR_THETA_NAME); use --elb-shock-name=$(FLOOR_DRIVER_SHOCK).")
    dgp_warnings = String[]
    train_warnings = String[]
    holdout_warnings = String[]
    posterior_warnings = String[]

    println("Generating actual-floor OBC DGP...")
    dgp_obs, dgp_shocks = simulate_observables(model, opts, FLOOR_THETA_TRUE; ignore_obc = false, warnings = dgp_warnings)
    lin_true, _ = simulate_observables(model, opts, FLOOR_THETA_TRUE; ignore_obc = true, warnings = dgp_warnings)
    floor_pct = 400 * log(Float64(model.parameter_values[find_idx(model.parameters, :R̄)]))
    actual_floor = dgp_obs[3, :] .<= floor_pct + opts.floor_tolerance_pct
    linear_violates = lin_true[3, :] .< floor_pct
    obs_sigma = max.(0.05 .* vec(Statistics.std(dgp_obs, dims = 2; corrected = false)), 1e-3)

    println("Training path-level OBC-minus-linear residual surrogate...")
    surrogate = train_path_residual_surrogate(model, opts; train_warnings = train_warnings, holdout_warnings = holdout_warnings)

    grid = theta_grid(opts.grid_size)
    direct_logpost = similar(grid)
    surrogate_logpost = similar(grid)
    for i in eachindex(grid)
        theta = grid[i]
        lp = log_prior(theta)
        direct_obs, _ = simulate_observables(model, opts, theta; ignore_obc = false, warnings = posterior_warnings)
        surr_obs = surrogate_observables(model, opts, theta, surrogate; warnings = posterior_warnings)
        direct_logpost[i] = lp + loglik(dgp_obs, direct_obs, obs_sigma)
        surrogate_logpost[i] = lp + loglik(dgp_obs, surr_obs, obs_sigma)
    end

    direct = posterior_summary(grid, direct_logpost)
    surrogate_summary = posterior_summary(grid, surrogate_logpost)
    overlap = max(direct["q05"], surrogate_summary["q05"]) <= min(direct["q95"], surrogate_summary["q95"])
    tol = 1e-12
    direct_cover = direct["q05"] <= FLOOR_THETA_TRUE + tol && FLOOR_THETA_TRUE <= direct["q95"] + tol
    surrogate_cover = surrogate_summary["q05"] <= FLOOR_THETA_TRUE + tol && FLOOR_THETA_TRUE <= surrogate_summary["q95"] + tol
    mean_diff_sd = abs(direct["mean"] - surrogate_summary["mean"]) / max(direct["sd"], eps(Float64))
    solver_warnings = vcat(dgp_warnings, train_warnings, holdout_warnings, posterior_warnings)
    unique_warnings = sort(unique(solver_warnings))
    core_clean_solver = isempty(dgp_warnings) && isempty(train_warnings) && isempty(posterior_warnings)
    fully_clean_solver = isempty(solver_warnings)

    payload = Dict{String,Any}(
        "options" => opts,
        "dgp_obs" => dgp_obs,
        "linear_true_obs" => lin_true,
        "dgp_shocks" => dgp_shocks,
        "obs_sigma" => obs_sigma,
        "actual_floor" => actual_floor,
        "linear_violates" => linear_violates,
        "surrogate" => surrogate,
        "direct" => direct,
        "surrogate_summary" => surrogate_summary,
        "mean_diff_direct_sd" => mean_diff_sd,
        "interval_overlap" => overlap,
        "direct_cover" => direct_cover,
        "surrogate_cover" => surrogate_cover,
        "dgp_warning_count" => length(dgp_warnings),
        "train_warning_count" => length(train_warnings),
        "holdout_warning_count" => length(holdout_warnings),
        "posterior_warning_count" => length(posterior_warnings),
        "solver_warning_count_total" => length(solver_warnings),
        "solver_warnings_unique" => unique_warnings,
        "core_clean_solver" => core_clean_solver,
        "fully_clean_solver" => fully_clean_solver,
    )
    serialize(joinpath(out_dir, "actual_floor_residual_grid_payload.jls"), payload)

    open(joinpath(out_dir, "SUMMARY.md"), "w") do io
        println(io, "# Galí Actual-Floor Residual Grid Validation")
        println(io)
        println(io, "- Parameter: `$(FLOOR_THETA_NAME)`")
        println(io, "- True value: $(FLOOR_THETA_TRUE)")
        println(io, "- OBC DGP: first-order simulator with `ignore_obc=false`")
        println(io, "- Linear benchmark: same shocks with `ignore_obc=true`")
        println(io, "- Parameterization: `$(FLOOR_THETA_NAME)` enters by scaling the fixed standardized `$(FLOOR_DRIVER_SHOCK)` path for this known-shock validation")
        println(io, "- Prior bounds: [$(FLOOR_PRIOR_LOWER), $(FLOOR_PRIOR_UPPER)]")
        println(io, "- Surrogate target: `OBC observables - linear observables`")
        println(io, "- Surrogate type: period-wise log-theta interpolation of the residual path")
        println(io, "- Actual floor periods: $(sum(actual_floor)) / $(opts.periods)")
        println(io, "- Linear sub-floor periods: $(sum(linear_violates)) / $(opts.periods)")
        println(io, "- Observation sigma: $(join(round.(obs_sigma, digits = 5), ", "))")
        println(io, "- DGP solver warning count: $(length(dgp_warnings))")
        println(io, "- Training-grid solver warning count: $(length(train_warnings))")
        println(io, "- Holdout-midpoint solver warning count: $(length(holdout_warnings))")
        println(io, "- Posterior-grid solver warning count: $(length(posterior_warnings))")
        println(io, "- Total solver warning count: $(length(solver_warnings))")
        println(io, "- Core posterior-comparison solver path clean: $(core_clean_solver)")
        println(io, "- Fully clean including holdout diagnostics: $(fully_clean_solver)")
        if !isempty(unique_warnings)
            println(io, "- Unique solver warnings: $(join(unique_warnings, " | "))")
        end
        println(io)
        println(io, "## Surrogate Fit")
        println(io)
        println(io, "- Train RMSE: $(join(round.(surrogate.train_rmse, digits = 5), ", "))")
        println(io, "- Holdout RMSE: $(join(round.(surrogate.loo_rmse, digits = 5), ", "))")
        println(io)
        println(io, "## Posterior Grid")
        println(io)
        println(io, "| Quantity | Direct OBC | Linear+residual surrogate |")
        println(io, "|---|---:|---:|")
        println(io, "| Mean | $(@sprintf("%.6g", direct["mean"])) | $(@sprintf("%.6g", surrogate_summary["mean"])) |")
        println(io, "| MAP | $(@sprintf("%.6g", direct["map"])) | $(@sprintf("%.6g", surrogate_summary["map"])) |")
        println(io, "| 90% interval | [$(@sprintf("%.6g", direct["q05"])), $(@sprintf("%.6g", direct["q95"]))] | [$(@sprintf("%.6g", surrogate_summary["q05"])), $(@sprintf("%.6g", surrogate_summary["q95"]))] |")
        println(io)
        println(io, "## Checks")
        println(io)
        println(io, "- Mean difference / direct posterior sd: $(@sprintf("%.4g", mean_diff_sd))")
        println(io, "- Interval overlap: $(overlap)")
        println(io, "- Direct true coverage: $(direct_cover)")
        println(io, "- Surrogate true coverage: $(surrogate_cover)")
        println(io, "- Core posterior-comparison validation pass: $(core_clean_solver && overlap && direct_cover && surrogate_cover)")
        println(io, "- Paper-ready clean validation including holdout: $(fully_clean_solver && overlap && direct_cover && surrogate_cover)")
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
    run_floor_grid_validation(parse_floor_args(ARGS))
end
