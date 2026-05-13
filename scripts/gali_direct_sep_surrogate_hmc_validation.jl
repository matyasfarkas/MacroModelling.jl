#!/usr/bin/env julia

# Direct SEP-HMC vs surrogate-HMC validation for the current-repo Gali OBC model.
#
# The script is intentionally staged.  Use --stage=smoke first, then --stage=pilot,
# and only launch --stage=full after finite likelihood, finite finite-difference
# gradient, and short-chain diagnostics pass.

using AdvancedHMC
using AxisKeys
using Dates
using Distributions
using LinearAlgebra
using LogDensityProblems
using MacroModelling
using Printf
using Random
using Serialization
using Statistics
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(REPO_ROOT, "models", "Gali_2015_chapter_3_obc.jl"))
const GALI_VALIDATION_MODEL = Gali_2015_chapter_3_obc

const OBSERVABLES = [:log_y, :pi_ann, :i_ann]
const THETA_NAMES = [:std_a, :std_z, :std_nu]
const THETA_BASELINE = [0.01, 0.05, 0.0025]
const THETA_TRUE = [0.012, 0.0425, 0.00275]
const PRIOR_LOG_SD = 0.35
const PRIOR_LOWER = 0.25 .* THETA_BASELINE
const PRIOR_UPPER = 4.0 .* THETA_BASELINE
const FAILURE_LL = -1.0e12
const ROM_FILTER_ANCHOR_REPEATS = 3
const DIRECT_LAST_DIAGNOSTICS = Ref{Any}(nothing)

Base.@kwdef struct StageConfig
    name::Symbol
    periods::Int
    train_samples::Int
    hidden_units::Int
    warmup::Int
    draws::Int
    chains::Int
    sep_horizon::Int
    sep_maxit::Int
    sep_tol::Float64
    sep_accept_tol::Float64
    inv_maxit::Int
    inv_tol::Float64
    inv_lambda::Float64
    hmc_max_depth::Int
    hmc_target_accept::Float64
    hmc_initial_step_size::Float64
    fd_eps::Float64
end

Base.@kwdef struct CliOptions
    stage::Symbol = :smoke
    dry_run::Bool = false
    likelihood_smoke_only::Bool = false
    run_id::String = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out_dir::String = ""
    seed::Int = 20260507
    periods_override::Union{Nothing,Int} = nothing
    train_samples_override::Union{Nothing,Int} = nothing
    warmup_override::Union{Nothing,Int} = nothing
    draws_override::Union{Nothing,Int} = nothing
    chains_override::Union{Nothing,Int} = nothing
    hmc_step_size_override::Union{Nothing,Float64} = nothing
    sep_horizon_override::Union{Nothing,Int} = nothing
    sep_maxit_override::Union{Nothing,Int} = nothing
    sep_accept_tol_override::Union{Nothing,Float64} = nothing
    hidden_units_override::Union{Nothing,Int} = nothing
    surrogate_design::Symbol = :local_path
    hmc_objectives::Symbol = :both
    direct_objective::Symbol = :measurement_error
    direct_logdet_method::Symbol = :exact
    direct_audit_draws::Int = 0
end

struct RandomFeatureSurrogate
    feature_kind::Symbol
    W::Matrix{Float64}
    b::Vector{Float64}
    bandwidth::Float64
    beta::Matrix{Float64}
    x_mean::Vector{Float64}
    x_scale::Vector{Float64}
    y_mean::Vector{Float64}
    train_rmse::Vector{Float64}
    valid_rmse::Vector{Float64}
    train_rrmse_obs::Vector{Float64}
    valid_rrmse_obs::Vector{Float64}
    train_rrmse_resid::Vector{Float64}
    valid_rrmse_resid::Vector{Float64}
end

struct ValidationLogDensity
    dim::Int
    fd_eps::Float64
    logpost_z::Function
end

LogDensityProblems.logdensity(p::ValidationLogDensity, z) = p.logpost_z(Vector{Float64}(z))
LogDensityProblems.dimension(p::ValidationLogDensity) = p.dim
LogDensityProblems.capabilities(::Type{ValidationLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::ValidationLogDensity, z)
    zf = Vector{Float64}(z)
    grad = zeros(Float64, p.dim)
    val = fd_gradient!(grad, p.logpost_z, zf, p.fd_eps)
    return val, grad
end

function parse_bool(s::AbstractString)
    sl = lowercase(strip(s))
    sl in ("true", "1", "yes", "y") && return true
    sl in ("false", "0", "no", "n") && return false
    error("Cannot parse boolean value: $s")
end

function parse_args(args)
    opts = CliOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    direct_logdet_method = Symbol(get(values, "direct-logdet-method", string(opts.direct_logdet_method)))
    direct_logdet_method in (:exact, :svd_pseudodet, :pseudodet) ||
        error("Unknown --direct-logdet-method=$direct_logdet_method. Use exact or svd_pseudodet.")
    direct_objective = Symbol(get(values, "direct-objective", string(opts.direct_objective)))
    direct_objective in (:measurement_error, :profiled_measurement_error, :exact_inversion) ||
        error("Unknown --direct-objective=$direct_objective. Use measurement_error, profiled_measurement_error, or exact_inversion.")
    hmc_objectives = Symbol(get(values, "hmc-objectives", string(opts.hmc_objectives)))
    hmc_objectives in (:both, :direct, :surrogate, :none) ||
        error("Unknown --hmc-objectives=$hmc_objectives. Use both, direct, surrogate, or none.")
    surrogate_design = Symbol(get(values, "surrogate-design", string(opts.surrogate_design)))
    surrogate_design in (:local_path, :global) ||
        error("Unknown --surrogate-design=$surrogate_design. Use local_path or global.")
    return CliOptions(
        stage = Symbol(get(values, "stage", string(opts.stage))),
        dry_run = parse_bool(get(values, "dry-run", string(opts.dry_run))),
        likelihood_smoke_only = parse_bool(get(values, "likelihood-smoke-only", string(opts.likelihood_smoke_only))),
        run_id = get(values, "run-id", opts.run_id),
        out_dir = get(values, "out-dir", opts.out_dir),
        seed = parse(Int, get(values, "seed", string(opts.seed))),
        periods_override = haskey(values, "periods") ? parse(Int, values["periods"]) : nothing,
        train_samples_override = haskey(values, "train-samples") ? parse(Int, values["train-samples"]) : nothing,
        warmup_override = haskey(values, "warmup") ? parse(Int, values["warmup"]) : nothing,
        draws_override = haskey(values, "draws") ? parse(Int, values["draws"]) : nothing,
        chains_override = haskey(values, "chains") ? parse(Int, values["chains"]) : nothing,
        hmc_step_size_override = haskey(values, "hmc-step-size") ? parse(Float64, values["hmc-step-size"]) : nothing,
        sep_horizon_override = haskey(values, "sep-horizon") ? parse(Int, values["sep-horizon"]) : nothing,
        sep_maxit_override = haskey(values, "sep-maxit") ? parse(Int, values["sep-maxit"]) : nothing,
        sep_accept_tol_override = haskey(values, "sep-accept-tol") ? parse(Float64, values["sep-accept-tol"]) : nothing,
        hidden_units_override = haskey(values, "hidden-units") ? parse(Int, values["hidden-units"]) : nothing,
        surrogate_design = surrogate_design,
        hmc_objectives = hmc_objectives,
        direct_objective = direct_objective,
        direct_logdet_method = direct_logdet_method,
        direct_audit_draws = parse(Int, get(values, "direct-audit-draws", string(opts.direct_audit_draws))),
    )
end

function stage_config(stage::Symbol)
    if stage == :smoke
        return StageConfig(
            name = stage,
            periods = 4,
            train_samples = 24,
            hidden_units = 32,
            warmup = 4,
            draws = 6,
            chains = 1,
            sep_horizon = 6,
            sep_maxit = 50,
            sep_tol = 1e-6,
            sep_accept_tol = 1e-3,
            inv_maxit = 8,
            inv_tol = 1e-6,
            inv_lambda = 1e-4,
            hmc_max_depth = 1,
            hmc_target_accept = 0.8,
            hmc_initial_step_size = 0.001,
            fd_eps = 1e-4,
        )
    elseif stage == :pilot
        return StageConfig(
            name = stage,
            periods = 40,
            train_samples = 600,
            hidden_units = 96,
            warmup = 100,
            draws = 200,
            chains = 1,
            sep_horizon = 12,
            sep_maxit = 80,
            sep_tol = 1e-7,
            sep_accept_tol = 1e-3,
            inv_maxit = 10,
            inv_tol = 1e-6,
            inv_lambda = 1e-4,
            hmc_max_depth = 6,
            hmc_target_accept = 0.8,
            hmc_initial_step_size = 0.001,
            fd_eps = 5e-5,
        )
    elseif stage == :full
        return StageConfig(
            name = stage,
            periods = 80,
            train_samples = 4000,
            hidden_units = 192,
            warmup = 500,
            draws = 1000,
            chains = 4,
            sep_horizon = 16,
            sep_maxit = 100,
            sep_tol = 1e-8,
            sep_accept_tol = 1e-5,
            inv_maxit = 12,
            inv_tol = 1e-6,
            inv_lambda = 1e-4,
            hmc_max_depth = 7,
            hmc_target_accept = 0.8,
            hmc_initial_step_size = 0.001,
            fd_eps = 2e-5,
        )
    else
        error("Unknown stage=$stage. Use smoke, pilot, or full.")
    end
end

function apply_overrides(cfg::StageConfig, opts::CliOptions)
    return StageConfig(
        name = cfg.name,
        periods = something(opts.periods_override, cfg.periods),
        train_samples = something(opts.train_samples_override, cfg.train_samples),
        hidden_units = something(opts.hidden_units_override, cfg.hidden_units),
        warmup = something(opts.warmup_override, cfg.warmup),
        draws = something(opts.draws_override, cfg.draws),
        chains = something(opts.chains_override, cfg.chains),
        sep_horizon = something(opts.sep_horizon_override, cfg.sep_horizon),
        sep_maxit = something(opts.sep_maxit_override, cfg.sep_maxit),
        sep_tol = cfg.sep_tol,
        sep_accept_tol = something(opts.sep_accept_tol_override, cfg.sep_accept_tol),
        inv_maxit = cfg.inv_maxit,
        inv_tol = cfg.inv_tol,
        inv_lambda = cfg.inv_lambda,
        hmc_max_depth = cfg.hmc_max_depth,
        hmc_target_accept = cfg.hmc_target_accept,
        hmc_initial_step_size = something(opts.hmc_step_size_override, cfg.hmc_initial_step_size),
        fd_eps = cfg.fd_eps,
    )
end

function default_out_dir(opts::CliOptions)
    isempty(opts.out_dir) || return opts.out_dir
    return joinpath(REPO_ROOT, ".local_artifacts", "gali_direct_sep_surrogate_hmc", opts.run_id)
end

function load_model()
    return GALI_VALIDATION_MODEL
end

function parameter_indices(model, names::Vector{Symbol})
    syms = Symbol.(model.parameters)
    idx = indexin(names, syms)
    any(isnothing, idx) && error("Missing parameters: $(names[findall(isnothing, idx)])")
    return Int.(idx)
end

function variable_indices(model, names::Vector{Symbol})
    syms = Symbol.(model.var)
    idx = indexin(names, syms)
    any(isnothing, idx) && error("Missing variables: $(names[findall(isnothing, idx)])")
    return Int.(idx)
end

function structural_shock_indices(model)
    return findall(!, contains.(string.(model.exo), "ᵒᵇᶜ"))
end

function inject_theta(base_params::Vector{Float64}, theta_idx::Vector{Int}, theta::Vector{Float64})
    params = copy(base_params)
    params[theta_idx] .= theta
    return params
end

logtheta_to_theta(z::AbstractVector) = exp.(z)
theta_to_logtheta(theta::AbstractVector) = log.(theta)

function log_prior_z(z::AbstractVector)
    theta = exp.(z)
    all(theta .>= PRIOR_LOWER) || return -Inf
    all(theta .<= PRIOR_UPPER) || return -Inf
    lp = 0.0
    for i in eachindex(z)
        lp += Distributions.logpdf(Distributions.Normal(log(THETA_BASELINE[i]), PRIOR_LOG_SD), z[i])
    end
    return lp
end

function draw_theta_from_prior(rng::AbstractRNG)
    z = similar(THETA_BASELINE)
    for i in eachindex(z)
        lo, hi = log(PRIOR_LOWER[i]), log(PRIOR_UPPER[i])
        dist = Distributions.Normal(log(THETA_BASELINE[i]), PRIOR_LOG_SD)
        zi = rand(rng, dist)
        z[i] = clamp(zi, lo, hi)
    end
    return exp.(z)
end

function unit_structural_shocks(rng::AbstractRNG, model, periods::Int)
    shocks = zeros(Float64, length(model.exo), periods)
    structural_idx = structural_shock_indices(model)
    shocks[structural_idx, :] .= randn(rng, length(structural_idx), periods)
    return shocks
end

function simulate_sep_dataset(model, theta::Vector{Float64}, cfg::StageConfig; seed::Int)
    rng = MersenneTwister(seed)
    base_params = Float64.(model.parameter_values)
    theta_idx = parameter_indices(model, THETA_NAMES)
    params = inject_theta(base_params, theta_idx, theta)
    MacroModelling.write_parameters_input!(model, params, verbose = false)

    shocks = unit_structural_shocks(rng, model, cfg.periods)
    res = MacroModelling.simulate_sep_extended_path(
        model;
        periods = cfg.periods,
        shocks = shocks,
        burn_in = 0,
        sep_horizon = cfg.sep_horizon,
        sep_order = 1,
        sep_nnodes = 3,
        sep_sparse_tree = true,
        sep_maxit = cfg.sep_maxit,
        sep_tol = cfg.sep_tol,
        sep_accept_tol = cfg.sep_accept_tol,
        shock_scaling = :none,
        random_seed = seed,
        silent = true,
    )
    hasproperty(res, :errorflag) && res.errorflag && error("SEP simulation failed for DGP.")

    obs_idx = variable_indices(model, OBSERVABLES)
    sim = Array(res.simulation)
    T_avail = min(cfg.periods, size(sim, 2) - 1)
    T_avail >= 1 || error("SEP simulation returned no usable periods.")
    obs = Float64.(sim[obs_idx, 2:(T_avail + 1)])
    state0 = Float64.(sim[:, 1])
    state_path = Float64.(sim[:, 1:(T_avail + 1)])
    obs_sigma = max.(0.05 .* vec(Statistics.std(obs, dims = 2; corrected = false)), 1e-3)

    return Dict{String,Any}(
        "theta" => theta,
        "theta_names" => string.(THETA_NAMES),
        "observables" => string.(OBSERVABLES),
        "obs_data" => obs,
        "obs_sigma" => obs_sigma,
        "state0" => state0,
        "state_path" => state_path,
        "shocks" => Float64.(shocks[:, 1:T_avail]),
        "periods" => T_avail,
        "sep_errorflag" => hasproperty(res, :errorflag) ? res.errorflag : false,
    )
end

function build_baseline_rom(model, base_params::Vector{Float64}, obs_idx::Vector{Int})
    MacroModelling.write_parameters_input!(model, base_params, verbose = false)
    MacroModelling.solve!(model; algorithm = :first_order, dynamics = true, obc = true, silent = true)
    state_idx = collect(1:length(model.var))
    cache = build_rom_cache(model, 1; params = base_params, use_obc = true)
    _, raw_predict_tuple, nsss = build_matrix_rom_predict(model; state_idx = state_idx, obs_idx = obs_idx)
    predict_tuple = make_theta_scaled_rom_predict(raw_predict_tuple, shock_theta_parameter_map(model))
    return cache, predict_tuple, nsss, state_idx
end

function shock_theta_parameter_map(model)
    shock_syms = Symbol.(model.exo)
    map = zeros(Int, length(shock_syms))
    pairs = Dict(:eps_a => 1, :eps_z => 2, :eps_nu => 3)
    for (shock_name, theta_pos) in pairs
        idx = findfirst(==(shock_name), shock_syms)
        idx === nothing && continue
        map[idx] = theta_pos
    end
    return map
end

function scale_shock_for_theta(shock::AbstractVector, theta::AbstractVector, shock_theta_map::Vector{Int})
    scaled = copy(shock)
    for j in eachindex(shock_theta_map)
        theta_pos = shock_theta_map[j]
        theta_pos == 0 && continue
        scaled[j] = shock[j] * (theta[theta_pos] / THETA_BASELINE[theta_pos])
    end
    return scaled
end

function make_theta_scaled_rom_predict(raw_predict_tuple::Function, shock_theta_map::Vector{Int})
    function theta_scaled_predict(state::AbstractVector, shock::AbstractVector, theta::AbstractVector)
        scaled_shock = scale_shock_for_theta(shock, theta, shock_theta_map)
        return raw_predict_tuple(state, scaled_shock, theta)
    end
    return theta_scaled_predict
end

function rom_step_full_scaled(cache::RomCache, state::AbstractVector, shock::AbstractVector,
                              theta::AbstractVector, shock_theta_map::Vector{Int})
    scaled_shock = scale_shock_for_theta(shock, theta, shock_theta_map)
    return rom_step_full(cache, state, scaled_shock)
end

function collect_surrogate_training_data(model, cfg::StageConfig, cache::RomCache,
                                         obs_idx::Vector{Int}; seed::Int)
    rng = MersenneTwister(seed)
    base_params = Float64.(model.parameter_values)
    theta_idx = parameter_indices(model, THETA_NAMES)
    shock_theta_map = shock_theta_parameter_map(model)
    n_state = length(model.var)
    n_shock = length(model.exo)
    x_cols = Vector{Vector{Float64}}()
    y_cols = Vector{Vector{Float64}}()
    y_sep_cols = Vector{Vector{Float64}}()

    attempts = 0
    while length(x_cols) < cfg.train_samples && attempts < max(20, 10 * cfg.train_samples)
        attempts += 1
        theta = draw_theta_from_prior(rng)
        params = inject_theta(base_params, theta_idx, theta)
        MacroModelling.write_parameters_input!(model, params, verbose = false)

        periods_this = min(max(2, cfg.periods), max(2, cfg.train_samples - length(x_cols)))
        shocks = unit_structural_shocks(rng, model, periods_this)
        local res
        try
            res = MacroModelling.simulate_sep_extended_path(
                model;
                periods = periods_this,
                shocks = shocks,
                burn_in = 0,
                sep_horizon = cfg.sep_horizon,
                sep_order = 1,
                sep_nnodes = 3,
                sep_sparse_tree = true,
                sep_maxit = cfg.sep_maxit,
                sep_tol = cfg.sep_tol,
                sep_accept_tol = cfg.sep_accept_tol,
                shock_scaling = :none,
                random_seed = seed + attempts,
                silent = true,
            )
        catch
            continue
        end
        hasproperty(res, :errorflag) && res.errorflag && continue
        sim = Float64.(Array(res.simulation))
        T_avail = min(periods_this, size(sim, 2) - 1)
        for t in 1:T_avail
            state = sim[:, t]
            shock = shocks[:, t]
            next_full = rom_step_full_scaled(cache, state, shock, theta, shock_theta_map)
            y_sep = sim[obs_idx, t + 1]
            y_rom = next_full[obs_idx]
            push!(x_cols, vcat(state, shock, log.(theta)))
            push!(y_cols, y_sep .- y_rom)
            push!(y_sep_cols, y_sep)
            length(x_cols) >= cfg.train_samples && break
        end
    end
    length(x_cols) >= max(8, min(cfg.train_samples, 8)) ||
        error("Could not collect enough surrogate training samples; got $(length(x_cols)).")

    X = hcat(x_cols...)
    Y = hcat(y_cols...)
    Ysep = hcat(y_sep_cols...)
    @assert size(X, 1) == n_state + n_shock + length(THETA_NAMES)
    return X, Y, Ysep
end

function feature_matrix(W::Matrix{Float64}, b::Vector{Float64}, Xn::Matrix{Float64})
    return tanh.(W * Xn .+ b)
end

function rbf_feature_matrix(centers::Matrix{Float64}, bandwidth::Float64, Xn::Matrix{Float64})
    center_norm = vec(sum(centers .^ 2, dims = 2))
    x_norm = vec(sum(Xn .^ 2, dims = 1))
    d2 = center_norm .+ x_norm' .- 2 .* (centers * Xn)
    return exp.(-0.5 .* max.(d2, 0.0) ./ (bandwidth^2))
end

function median_pairwise_distance(X::Matrix{Float64}; max_pairs::Int = 2000)
    n = size(X, 2)
    n <= 1 && return sqrt(size(X, 1))
    rng = MersenneTwister(1776)
    dists = Float64[]
    for _ in 1:min(max_pairs, n * (n - 1) ÷ 2)
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        i == j && continue
        push!(dists, norm(X[:, i] .- X[:, j]))
    end
    isempty(dists) && return sqrt(size(X, 1))
    return Statistics.median(dists)
end

function train_random_feature_surrogate(X::Matrix{Float64}, Y::Matrix{Float64}, Ysep::Matrix{Float64},
                                        hidden_units::Int; seed::Int, ridge::Float64 = 1e-6,
                                        score_X::Union{Nothing,Matrix{Float64}} = nothing,
                                        score_Y::Union{Nothing,Matrix{Float64}} = nothing,
                                        score_Ysep::Union{Nothing,Matrix{Float64}} = nothing)
    rng = MersenneTwister(seed)
    n = size(X, 2)
    order = randperm(rng, n)
    n_valid = max(1, floor(Int, 0.2 * n))
    valid_idx = order[1:n_valid]
    train_idx = order[(n_valid + 1):end]
    isempty(train_idx) && (train_idx = valid_idx)

    x_mean = vec(Statistics.mean(X[:, train_idx], dims = 2))
    x_scale = vec(Statistics.std(X[:, train_idx], dims = 2; corrected = false))
    x_scale .= max.(x_scale, 1e-8)
    y_mean = vec(Statistics.mean(Y[:, train_idx], dims = 2))

    Xn = (X .- x_mean) ./ x_scale
    center_count = min(hidden_units, length(train_idx))
    center_idx = train_idx[randperm(rng, length(train_idx))[1:center_count]]
    W = Matrix{Float64}(Xn[:, center_idx]')
    b = zeros(Float64, center_count)
    obs_scale_train = max.(sqrt.(vec(Statistics.mean(Ysep[:, train_idx] .^ 2, dims = 2))), 1e-10)
    obs_scale_valid = max.(sqrt.(vec(Statistics.mean(Ysep[:, valid_idx] .^ 2, dims = 2))), 1e-10)
    resid_scale_train = max.(sqrt.(vec(Statistics.mean(Y[:, train_idx] .^ 2, dims = 2))), 1e-10)
    resid_scale_valid = max.(sqrt.(vec(Statistics.mean(Y[:, valid_idx] .^ 2, dims = 2))), 1e-10)

    score_has_data = score_X !== nothing && score_Y !== nothing && score_Ysep !== nothing
    Xn_score = score_has_data ? (score_X .- x_mean) ./ x_scale : Xn[:, valid_idx]
    Y_score = score_has_data ? score_Y : Y[:, valid_idx]
    Ysep_score = score_has_data ? score_Ysep : Ysep[:, valid_idx]
    obs_scale_score = max.(sqrt.(vec(Statistics.mean(Ysep_score .^ 2, dims = 2))), 1e-10)

    base_bandwidth = max(median_pairwise_distance(Xn[:, center_idx]), 1e-3)
    Yc_train = Y[:, train_idx] .- y_mean
    best_score = Inf
    best_bandwidth = base_bandwidth
    best_beta = zeros(Float64, size(Y, 1), center_count + 1)
    for bw_mult in (0.15, 0.25, 0.4, 0.6, 1.0)
        bandwidth_i = max(bw_mult * base_bandwidth, 1e-3)
        Phi_train = vcat(ones(1, length(train_idx)), rbf_feature_matrix(W, bandwidth_i, Xn[:, train_idx]))
        Phi_score = vcat(ones(1, size(Xn_score, 2)), rbf_feature_matrix(W, bandwidth_i, Xn_score))
        gram = Phi_train * Phi_train'
        for ridge_i in (1e-8, ridge, 1e-4)
            beta_i = (Yc_train * Phi_train') / (gram + ridge_i * I)
            pred_score = y_mean .+ beta_i * Phi_score
            resid_score_i = pred_score .- Y_score
            score_rmse_i = sqrt.(vec(Statistics.mean(resid_score_i .^ 2, dims = 2)))
            score = maximum(score_rmse_i ./ obs_scale_score) + 0.1 * Statistics.mean(score_rmse_i ./ obs_scale_score)
            if score < best_score
                best_score = score
                best_bandwidth = bandwidth_i
                best_beta = beta_i
            end
        end
    end
    bandwidth = best_bandwidth
    beta = best_beta

    pred_all = y_mean .+ beta * vcat(ones(1, n), rbf_feature_matrix(W, bandwidth, Xn))
    resid_train = pred_all[:, train_idx] .- Y[:, train_idx]
    resid_valid = pred_all[:, valid_idx] .- Y[:, valid_idx]
    train_rmse = sqrt.(vec(Statistics.mean(resid_train .^ 2, dims = 2)))
    valid_rmse = sqrt.(vec(Statistics.mean(resid_valid .^ 2, dims = 2)))

    return RandomFeatureSurrogate(
        :rbf,
        W,
        b,
        bandwidth,
        beta,
        x_mean,
        x_scale,
        y_mean,
        train_rmse,
        valid_rmse,
        train_rmse ./ obs_scale_train,
        valid_rmse ./ obs_scale_valid,
        train_rmse ./ resid_scale_train,
        valid_rmse ./ resid_scale_valid,
    )
end

function predict_residual(s::RandomFeatureSurrogate,
                          state::AbstractVector,
                          shock::AbstractVector,
                          theta::AbstractVector)
    x = Float64.(vcat(state, shock, log.(Float64.(theta))))
    xn = (x .- s.x_mean) ./ s.x_scale
    if s.feature_kind == :rbf
        d2 = vec(sum((s.W .- reshape(xn, 1, :)) .^ 2, dims = 2))
        phi = vcat(1.0, exp.(-0.5 .* d2 ./ (s.bandwidth^2)))
    else
        phi = vcat(1.0, tanh.(s.W * xn .+ s.b))
    end
    return s.y_mean .+ s.beta * phi
end

function surrogate_path_diagnostics(s::RandomFeatureSurrogate, cache::RomCache,
                                    obs_idx::Vector{Int}, dgp::Dict{String,Any})
    haskey(dgp, "state_path") || return nothing
    state_path = Matrix{Float64}(dgp["state_path"])
    shocks = Matrix{Float64}(dgp["shocks"])
    obs_data = Matrix{Float64}(dgp["obs_data"])
    T_path = size(obs_data, 2)
    shock_theta_map = shock_theta_parameter_map(GALI_VALIDATION_MODEL)
    errors = zeros(Float64, length(obs_idx), T_path)
    residuals = zeros(Float64, length(obs_idx), T_path)
    for t in 1:T_path
        state = state_path[:, t]
        shock = shocks[:, t]
        rom_obs = rom_step_full_scaled(cache, state, shock, THETA_TRUE, shock_theta_map)[obs_idx]
        true_resid = obs_data[:, t] .- rom_obs
        pred_resid = predict_residual(s, state, shock, THETA_TRUE)
        errors[:, t] .= pred_resid .- true_resid
        residuals[:, t] .= true_resid
    end
    rmse = sqrt.(vec(Statistics.mean(errors .^ 2, dims = 2)))
    obs_scale = max.(sqrt.(vec(Statistics.mean(obs_data .^ 2, dims = 2))), 1e-10)
    resid_scale = max.(sqrt.(vec(Statistics.mean(residuals .^ 2, dims = 2))), 1e-10)
    return Dict{String,Any}(
        "path_rmse" => rmse,
        "path_rrmse_obs" => rmse ./ obs_scale,
        "path_rrmse_resid" => rmse ./ resid_scale,
    )
end

function surrogate_rom_filter_diagnostics(s::RandomFeatureSurrogate, model, base_params::Vector{Float64},
                                          theta_idx::Vector{Int}, cache::RomCache,
                                          obs_idx::Vector{Int}, predict_tuple::Function,
                                          dgp::Dict{String,Any}, cfg::StageConfig)
    rom_states, rom_shocks = build_rom_filter_path(predict_tuple, dgp, THETA_TRUE, cfg)
    shock_theta_map = shock_theta_parameter_map(model)
    T_path = size(rom_shocks, 2)
    errors = zeros(Float64, length(obs_idx), T_path)
    residuals = zeros(Float64, length(obs_idx), T_path)
    targets = zeros(Float64, length(obs_idx), T_path)
    for t in 1:T_path
        state = rom_states[:, t]
        shock = rom_shocks[:, t]
        pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx,
                                        state, shock, THETA_TRUE, cfg)
        pred.ok || continue
        rom_obs = rom_step_full_scaled(cache, state, shock, THETA_TRUE, shock_theta_map)[obs_idx]
        true_resid = pred.obs .- rom_obs
        pred_resid = predict_residual(s, state, shock, THETA_TRUE)
        errors[:, t] .= pred_resid .- true_resid
        residuals[:, t] .= true_resid
        targets[:, t] .= pred.obs
    end
    rmse = sqrt.(vec(Statistics.mean(errors .^ 2, dims = 2)))
    obs_scale = max.(sqrt.(vec(Statistics.mean(targets .^ 2, dims = 2))), 1e-10)
    resid_scale = max.(sqrt.(vec(Statistics.mean(residuals .^ 2, dims = 2))), 1e-10)
    return Dict{String,Any}(
        "rom_path_rmse" => rmse,
        "rom_path_rrmse_obs" => rmse ./ obs_scale,
        "rom_path_rrmse_resid" => rmse ./ resid_scale,
    )
end

function make_surrogate_predict(predict_tuple::Function, surrogate::RandomFeatureSurrogate)
    function surrogate_predict(state::AbstractVector, shock::AbstractVector, theta::AbstractVector)
        obs_rom, state_next = predict_tuple(state, shock, theta)
        resid = predict_residual(surrogate, state, shock, theta)
        return obs_rom .+ resid, state_next
    end
    return surrogate_predict
end

function make_shock_sigmas(model)
    sigmas = zeros(Float64, length(model.exo))
    sigmas[structural_shock_indices(model)] .= 1.0
    return sigmas
end

function direct_exact_inversion_loglik(model, obs_ka, base_params::Vector{Float64},
                                       theta_idx::Vector{Int}, theta::Vector{Float64}, cfg::StageConfig,
                                       direct_logdet_method::Symbol)
    params = inject_theta(base_params, theta_idx, theta)
    ll = MacroModelling.get_loglikelihood(
        model,
        obs_ka,
        params;
        algorithm = :stochastic_extended_path,
        filter = :inversion,
        verbose = false,
        on_failure_loglikelihood = FAILURE_LL,
        presample_periods = 0,
        sep_periods = cfg.sep_horizon,
        sep_order = 1,
        sep_nnodes = 3,
        sep_sparse_tree = true,
        sep_maxit = cfg.sep_maxit,
        sep_tol = cfg.sep_tol,
        sep_accept_tol = cfg.sep_accept_tol,
        sep_inv_maxit = cfg.inv_maxit,
        sep_inv_resid_tol = cfg.inv_tol,
        sep_inv_step_tol = cfg.inv_tol,
        sep_inv_lambda = cfg.inv_lambda,
        sep_inv_predict_tol = 1e-10,
        sep_inv_logdet_method = direct_logdet_method,
    )
    DIRECT_LAST_DIAGNOSTICS[] = MacroModelling.get_sep_inversion_last_diagnostics()
    (!isfinite(ll) || ll == FAILURE_LL) && return -Inf
    return Float64(ll)
end

function direct_sep_predict_level(model, base_params::Vector{Float64}, theta_idx::Vector{Int},
                                  obs_idx::Vector{Int}, state_level::Vector{Float64},
                                  shock_full::Vector{Float64}, theta::Vector{Float64},
                                  cfg::StageConfig)
    params = inject_theta(base_params, theta_idx, theta)
    MacroModelling.write_parameters_input!(model, params, verbose = false)
    shocks = reshape(shock_full, :, 1)
    local res
    try
        res = MacroModelling.simulate_sep_extended_path(
            model;
            periods = 1,
            initial_state = state_level,
            shocks = shocks,
            burn_in = 0,
            sep_horizon = cfg.sep_horizon,
            sep_order = 1,
            sep_nnodes = 3,
            sep_sparse_tree = true,
            sep_maxit = cfg.sep_maxit,
            sep_tol = cfg.sep_tol,
            sep_accept_tol = cfg.sep_accept_tol,
            shock_scaling = :none,
            silent = true,
        )
    catch e
        return (ok = false, obs = zeros(Float64, length(obs_idx)), state_next = copy(state_level),
                sep_flag = -1, sep_err = Inf, message = sprint(showerror, e))
    end
    errflag = hasproperty(res, :errorflag) ? Bool(res.errorflag) : false
    if errflag
        return (ok = false, obs = zeros(Float64, length(obs_idx)), state_next = copy(state_level),
                sep_flag = 1, sep_err = Inf, message = "simulate_sep_extended_path returned errorflag=true")
    end
    sim = Float64.(Array(res.simulation))
    size(sim, 2) >= 2 || return (ok = false, obs = zeros(Float64, length(obs_idx)), state_next = copy(state_level),
                                 sep_flag = 2, sep_err = Inf, message = "SEP simulation returned fewer than two columns")
    state_next = sim[:, 2]
    obs = state_next[obs_idx]
    if !all(isfinite, obs) || !all(isfinite, state_next)
        return (ok = false, obs = zeros(Float64, length(obs_idx)), state_next = copy(state_level),
                sep_flag = 3, sep_err = Inf, message = "Direct SEP prediction returned non-finite values")
    end
    return (ok = true, obs = obs, state_next = state_next, sep_flag = 0, sep_err = 0.0, message = "")
end

function draw_local_theta(rng::AbstractRNG)
    center = rand(rng) < 0.90 ? THETA_TRUE : THETA_BASELINE
    local_sd = rand(rng) < 0.90 ? 0.05 : 0.12
    z = log.(center) .+ local_sd .* randn(rng, length(center))
    return clamp.(exp.(z), PRIOR_LOWER, PRIOR_UPPER)
end

function build_rom_filter_path(predict_tuple::Function, dgp::Dict{String,Any},
                               theta::Vector{Float64}, cfg::StageConfig)
    obs_data = Matrix{Float64}(dgp["obs_data"])
    obs_sigma = Vector{Float64}(dgp["obs_sigma"])
    shock_sigmas = make_shock_sigmas(GALI_VALIDATION_MODEL)
    structural_idx = findall(shock_sigmas .> 0)
    state = Vector{Float64}(dgp["state0"])
    states = zeros(Float64, length(state), size(obs_data, 2))
    shocks = zeros(Float64, length(shock_sigmas), size(obs_data, 2))
    eps_init = zeros(Float64, length(structural_idx))
    for t in 1:size(obs_data, 2)
        states[:, t] .= state
        eps_full, state_next, _ = MacroModelling.inversion_step(
            predict_tuple,
            state,
            obs_data[:, t],
            theta,
            obs_sigma,
            shock_sigmas,
            structural_idx;
            eps_init = eps_init,
            maxit = cfg.inv_maxit,
            tol = cfg.inv_tol,
            lambda = cfg.inv_lambda,
        )
        shocks[:, t] .= eps_full
        state = Float64.(state_next)
        eps_init .= eps_full[structural_idx]
    end
    return states, shocks
end

function collect_local_path_surrogate_training_data(model, cfg::StageConfig, cache::RomCache,
                                                    obs_idx::Vector{Int}, dgp::Dict{String,Any};
                                                    predict_tuple::Function, seed::Int)
    rng = MersenneTwister(seed)
    base_params = Float64.(model.parameter_values)
    theta_idx = parameter_indices(model, THETA_NAMES)
    state_path = Matrix{Float64}(dgp["state_path"])
    dgp_shocks = Matrix{Float64}(dgp["shocks"])
    T_path = size(dgp_shocks, 2)
    rom_states, rom_shocks = build_rom_filter_path(predict_tuple, dgp, THETA_TRUE, cfg)
    n_state = length(model.var)
    n_shock = length(model.exo)
    structural_idx = structural_shock_indices(model)
    shock_theta_map = shock_theta_parameter_map(model)

    state_scale = vec(Statistics.std(state_path[:, 1:T_path], dims = 2; corrected = false))
    state_scale .= max.(state_scale, 1e-5)

    x_cols = Vector{Vector{Float64}}()
    y_cols = Vector{Vector{Float64}}()
    y_sep_cols = Vector{Vector{Float64}}()

    attempts = 0
    while length(x_cols) < cfg.train_samples && attempts < max(100, 25 * cfg.train_samples)
        attempts += 1
        t = rand(rng, 1:T_path)
        use_rom_path = rand(rng) < 0.85
        state = use_rom_path ? copy(rom_states[:, t]) : copy(state_path[:, t])
        if rand(rng) < 0.35
            state .+= 0.01 .* state_scale .* randn(rng, n_state)
        end

        shock = use_rom_path ? copy(rom_shocks[:, t]) : copy(dgp_shocks[:, t])
        if rand(rng) < 0.95
            shock[structural_idx] .+= 0.10 .* randn(rng, length(structural_idx))
        else
            shock[structural_idx] .= randn(rng, length(structural_idx))
        end

        theta = draw_local_theta(rng)
        pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx, state, shock, theta, cfg)
        pred.ok || continue
        local next_full
        try
            next_full = rom_step_full_scaled(cache, state, shock, theta, shock_theta_map)
        catch
            continue
        end
        y_sep = pred.obs
        y_rom = next_full[obs_idx]
        all(isfinite, y_sep) || continue
        all(isfinite, y_rom) || continue
        push!(x_cols, vcat(state, shock, log.(theta)))
        push!(y_cols, y_sep .- y_rom)
        push!(y_sep_cols, y_sep)
    end

    length(x_cols) >= max(8, min(cfg.train_samples, 8)) ||
        error("Could not collect enough local-path surrogate training samples; got $(length(x_cols)).")

    X = hcat(x_cols...)
    Y = hcat(y_cols...)
    Ysep = hcat(y_sep_cols...)
    @assert size(X, 1) == n_state + n_shock + length(THETA_NAMES)
    return X, Y, Ysep
end

function collect_training_data(model, cfg::StageConfig, cache::RomCache,
                               obs_idx::Vector{Int}, dgp::Dict{String,Any},
                               design::Symbol; predict_tuple::Function, seed::Int)
    if design == :local_path
        return collect_local_path_surrogate_training_data(model, cfg, cache, obs_idx, dgp;
                                                          predict_tuple = predict_tuple,
                                                          seed = seed)
    elseif design == :global
        return collect_surrogate_training_data(model, cfg, cache, obs_idx; seed = seed)
    else
        error("Unsupported surrogate design: $design")
    end
end

function collect_rom_filter_surrogate_data(model, cfg::StageConfig, cache::RomCache,
                                           obs_idx::Vector{Int}, dgp::Dict{String,Any},
                                           predict_tuple::Function,
                                           base_params::Vector{Float64},
                                           theta_idx::Vector{Int})
    rom_states, rom_shocks = build_rom_filter_path(predict_tuple, dgp, THETA_TRUE, cfg)
    shock_theta_map = shock_theta_parameter_map(model)
    x_cols = Vector{Vector{Float64}}()
    y_cols = Vector{Vector{Float64}}()
    y_sep_cols = Vector{Vector{Float64}}()
    for t in 1:size(rom_shocks, 2)
        state = rom_states[:, t]
        shock = rom_shocks[:, t]
        pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx,
                                        state, shock, THETA_TRUE, cfg)
        pred.ok || continue
        next_full = rom_step_full_scaled(cache, state, shock, THETA_TRUE, shock_theta_map)
        y_rom = next_full[obs_idx]
        push!(x_cols, vcat(state, shock, log.(THETA_TRUE)))
        push!(y_cols, pred.obs .- y_rom)
        push!(y_sep_cols, pred.obs)
    end
    isempty(x_cols) && return nothing
    return hcat(x_cols...), hcat(y_cols...), hcat(y_sep_cols...)
end

function direct_sep_fd_jacobian(model, base_params::Vector{Float64}, theta_idx::Vector{Int},
                                obs_idx::Vector{Int}, state_level::Vector{Float64},
                                eps_struct::Vector{Float64}, structural_idx::Vector{Int},
                                shock_sigmas::Vector{Float64}, theta::Vector{Float64},
                                cfg::StageConfig)
    d_eps = length(shock_sigmas)
    n_obs = length(obs_idx)
    n_struct = length(structural_idx)
    base_full = zeros(Float64, d_eps)
    base_full[structural_idx] .= eps_struct
    base_pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx, state_level,
                                         base_full, theta, cfg)
    J = zeros(Float64, n_obs, n_struct)
    base_pred.ok || return (ok = false, J = J, base = base_pred, message = base_pred.message)

    for j in 1:n_struct
        step_scale = max(1.0, abs(eps_struct[j]), shock_sigmas[structural_idx[j]])
        h = max(cbrt(eps(Float64)) * step_scale, 1e-6)
        pert = copy(eps_struct)
        pert[j] += h
        eps_full = zeros(Float64, d_eps)
        eps_full[structural_idx] .= pert
        pred_p = direct_sep_predict_level(model, base_params, theta_idx, obs_idx, state_level,
                                          eps_full, theta, cfg)
        pred_p.ok || return (ok = false, J = J, base = base_pred, message = pred_p.message)
        J[:, j] .= (pred_p.obs .- base_pred.obs) ./ h
    end
    return (ok = true, J = J, base = base_pred, message = "")
end

function direct_measurement_error_step(model, base_params::Vector{Float64}, theta_idx::Vector{Int},
                                       obs_idx::Vector{Int}, state_level::Vector{Float64},
                                       y_obs::Vector{Float64}, obs_sigma::Vector{Float64},
                                       shock_sigmas::Vector{Float64}, theta::Vector{Float64},
                                       cfg::StageConfig; eps_init::Union{Nothing,Vector{Float64}} = nothing)
    structural_idx = findall(shock_sigmas .> 0)
    n_struct = length(structural_idx)
    n_struct > 0 || error("Measurement-error direct SEP objective needs at least one structural shock.")
    shock_std = shock_sigmas[structural_idx]
    eps_struct = eps_init === nothing ? zeros(Float64, n_struct) : copy(eps_init)
    obs_log_norm_const = sum(log.(2 * pi .* obs_sigma .^ 2))
    shock_log_norm_const = sum(log.(2 * pi .* shock_std .^ 2))
    lambda_eff = cfg.inv_lambda
    final_pred = nothing
    final_J = nothing
    final_iter = 0

    for inv_iter in 1:cfg.inv_maxit
        final_iter = inv_iter
        jac = direct_sep_fd_jacobian(model, base_params, theta_idx, obs_idx, state_level,
                                    eps_struct, structural_idx, shock_sigmas, theta, cfg)
        jac.ok || return (ok = false, ll = -Inf, state_next = copy(state_level), eps_full = zeros(Float64, length(shock_sigmas)),
                          failure_code = "fd_jacobian_failed", iteration = inv_iter, message = jac.message)
        pred = jac.base
        J = jac.J
        final_pred = pred
        final_J = J
        if !all(isfinite, J) || !all(isfinite, pred.obs)
            return (ok = false, ll = -Inf, state_next = copy(state_level), eps_full = zeros(Float64, length(shock_sigmas)),
                    failure_code = "nonfinite_prediction_or_jacobian", iteration = inv_iter, message = "")
        end
        resid = (y_obs .- pred.obs) ./ obs_sigma
        r = vcat(resid, eps_struct ./ shock_std)
        J_obs = -(J ./ obs_sigma)
        J_prior = Diagonal(1.0 ./ shock_std)
        J_aug = vcat(J_obs, J_prior)
        lhs = J_aug' * J_aug + lambda_eff * I
        rhs = -J_aug' * r
        step = try
            lhs \ rhs
        catch e
            return (ok = false, ll = -Inf, state_next = copy(state_level), eps_full = zeros(Float64, length(shock_sigmas)),
                    failure_code = "linear_solve_failed", iteration = inv_iter, message = sprint(showerror, e))
        end
        all(isfinite, step) || return (ok = false, ll = -Inf, state_next = copy(state_level), eps_full = zeros(Float64, length(shock_sigmas)),
                                       failure_code = "nonfinite_step", iteration = inv_iter, message = "")
        eps_struct .+= step
        if norm(step) <= cfg.inv_tol * (1 + norm(eps_struct))
            break
        end
    end

    eps_full = zeros(Float64, length(shock_sigmas))
    eps_full[structural_idx] .= eps_struct
    pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx, state_level,
                                    eps_full, theta, cfg)
    pred.ok || return (ok = false, ll = -Inf, state_next = copy(state_level), eps_full = eps_full,
                       failure_code = "final_prediction_failed", iteration = final_iter, message = pred.message)
    resid = (y_obs .- pred.obs) ./ obs_sigma
    ll = -0.5 * (sum(resid .^ 2) +
                 sum((eps_struct ./ shock_std) .^ 2) +
                 obs_log_norm_const +
                 shock_log_norm_const)
    return (ok = true, ll = ll, state_next = pred.state_next, eps_full = eps_full,
            failure_code = "none", iteration = final_iter, message = "",
            residual_norm = norm(resid), shock_norm = norm(eps_struct ./ shock_std),
            jacobian_rank = final_J === nothing ? missing : rank(final_J))
end

function direct_measurement_error_loglik(model, base_params::Vector{Float64},
                                         theta_idx::Vector{Int}, obs_idx::Vector{Int},
                                         dgp::Dict{String,Any}, theta::Vector{Float64},
                                         cfg::StageConfig)
    obs_data = Matrix{Float64}(dgp["obs_data"])
    obs_sigma = Vector{Float64}(dgp["obs_sigma"])
    shock_sigmas = make_shock_sigmas(model)
    state = Vector{Float64}(dgp["state0"])
    eps_init = zeros(Float64, count(>(0), shock_sigmas))
    ll_total = 0.0
    shocks_out = zeros(Float64, length(shock_sigmas), size(obs_data, 2))
    iterations = zeros(Int, size(obs_data, 2))
    residual_norms = zeros(Float64, size(obs_data, 2))
    shock_norms = zeros(Float64, size(obs_data, 2))
    ranks = Vector{Any}(undef, size(obs_data, 2))

    for t in 1:size(obs_data, 2)
        step = direct_measurement_error_step(model, base_params, theta_idx, obs_idx, state,
                                             obs_data[:, t], obs_sigma, shock_sigmas, theta, cfg;
                                             eps_init = eps_init)
        if !step.ok || !isfinite(step.ll)
            DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
                "kind" => "direct_sep_measurement_error",
                "status" => "failure",
                "failure_code" => step.failure_code,
                "period_index" => t,
                "iteration" => step.iteration,
                "message" => step.message,
            )
            return -Inf
        end
        ll_total += step.ll
        state = step.state_next
        shocks_out[:, t] .= step.eps_full
        eps_init .= step.eps_full[findall(shock_sigmas .> 0)]
        iterations[t] = step.iteration
        residual_norms[t] = step.residual_norm
        shock_norms[t] = step.shock_norm
        ranks[t] = step.jacobian_rank
    end
    DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
        "kind" => "direct_sep_measurement_error",
        "status" => "ok",
        "failure_code" => "none",
        "n_periods" => size(obs_data, 2),
        "max_iteration" => maximum(iterations),
        "mean_residual_norm" => Statistics.mean(residual_norms),
        "mean_shock_norm" => Statistics.mean(shock_norms),
        "jacobian_ranks" => ranks,
    )
    return ll_total
end

function make_direct_sep_eval_predict(model, base_params::Vector{Float64},
                                      theta_idx::Vector{Int}, obs_idx::Vector{Int},
                                      cfg::StageConfig)
    function direct_sep_eval_predict(state::AbstractVector, shock::AbstractVector, theta::AbstractVector)
        pred = direct_sep_predict_level(
            model,
            base_params,
            theta_idx,
            obs_idx,
            Float64.(state),
            Float64.(shock),
            Float64.(theta),
            cfg,
        )
        if !pred.ok
            return fill(NaN, length(obs_idx)), fill(NaN, length(state))
        end
        return pred.obs, pred.state_next
    end
    return direct_sep_eval_predict
end

function direct_rom_inversion_loglik(model, base_params::Vector{Float64},
                                     theta_idx::Vector{Int}, obs_idx::Vector{Int},
                                     predict_tuple::Function, dgp::Dict{String,Any},
                                     theta::Vector{Float64}, cfg::StageConfig)
    obs_data = Matrix{Float64}(dgp["obs_data"])
    obs_sigma = Vector{Float64}(dgp["obs_sigma"])
    state = Vector{Float64}(dgp["state0"])
    shock_sigmas = make_shock_sigmas(model)
    structural_idx = findall(shock_sigmas .> 0)
    shock_std = shock_sigmas[structural_idx]
    obs_log_norm_const = sum(log.(2 * pi .* obs_sigma .^ 2))
    shock_log_norm_const = sum(log.(2 * pi .* shock_std .^ 2))
    eps_init = zeros(Float64, length(structural_idx))
    shocks = zeros(Float64, length(shock_sigmas), size(obs_data, 2))
    ll_vec = zeros(Float64, size(obs_data, 2))
    residual_norms = zeros(Float64, size(obs_data, 2))

    for t in 1:size(obs_data, 2)
        local eps_full
        try
            eps_full, _, _ = MacroModelling.inversion_step(
                predict_tuple,
                state,
                obs_data[:, t],
                theta,
                obs_sigma,
                shock_sigmas,
                structural_idx;
                eps_init = eps_init,
                maxit = cfg.inv_maxit,
                tol = cfg.inv_tol,
                lambda = cfg.inv_lambda,
            )
        catch e
            DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
                "kind" => "direct_sep_rom_inversion_measurement_error",
                "status" => "failure",
                "failure_code" => "rom_inversion_failed",
                "period_index" => t,
                "message" => sprint(showerror, e),
            )
            return -Inf
        end
        all(isfinite, eps_full) || begin
            DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
                "kind" => "direct_sep_rom_inversion_measurement_error",
                "status" => "failure",
                "failure_code" => "nonfinite_rom_shock",
                "period_index" => t,
            )
            return -Inf
        end

        pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx, state,
                                        Float64.(eps_full), theta, cfg)
        if !pred.ok
            DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
                "kind" => "direct_sep_rom_inversion_measurement_error",
                "status" => "failure",
                "failure_code" => "direct_sep_prediction_failed",
                "period_index" => t,
                "message" => pred.message,
            )
            return -Inf
        end
        resid = (obs_data[:, t] .- pred.obs) ./ obs_sigma
        ll_vec[t] = -0.5 * (sum(resid .^ 2) +
                             sum((eps_full[structural_idx] ./ shock_std) .^ 2) +
                             obs_log_norm_const +
                             shock_log_norm_const)
        isfinite(ll_vec[t]) || begin
            DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
                "kind" => "direct_sep_rom_inversion_measurement_error",
                "status" => "failure",
                "failure_code" => "nonfinite_ll",
                "period_index" => t,
            )
            return -Inf
        end
        shocks[:, t] .= eps_full
        residual_norms[t] = norm(resid)
        state = pred.state_next
        eps_init .= eps_full[structural_idx]
    end

    DIRECT_LAST_DIAGNOSTICS[] = Dict{String,Any}(
        "kind" => "direct_sep_rom_inversion_measurement_error",
        "status" => "ok",
        "failure_code" => "none",
        "n_periods" => size(obs_data, 2),
        "mean_ll" => Statistics.mean(ll_vec),
        "mean_residual_norm" => Statistics.mean(residual_norms),
        "mean_abs_structural_shock" => Statistics.mean(abs.(shocks[shock_sigmas .> 0, :])),
    )
    return sum(ll_vec)
end

function direct_loglik(model, obs_ka, base_params::Vector{Float64},
                       theta_idx::Vector{Int}, obs_idx::Vector{Int}, dgp::Dict{String,Any},
                       predict_tuple::Function, theta::Vector{Float64}, cfg::StageConfig,
                       direct_objective::Symbol, direct_logdet_method::Symbol)
    if direct_objective == :measurement_error
        return direct_rom_inversion_loglik(model, base_params, theta_idx, obs_idx,
                                           predict_tuple, dgp, theta, cfg)
    elseif direct_objective == :profiled_measurement_error
        return direct_measurement_error_loglik(model, base_params, theta_idx, obs_idx, dgp, theta, cfg)
    elseif direct_objective == :exact_inversion
        return direct_exact_inversion_loglik(model, obs_ka, base_params, theta_idx, theta, cfg, direct_logdet_method)
    else
        error("Unsupported direct objective: $direct_objective")
    end
end

function surrogate_loglik(predict_tuple::Function, surrogate_predict::Function,
                         state0::Vector{Float64}, obs_data::Matrix{Float64},
                         obs_sigma::Vector{Float64}, shock_sigmas::Vector{Float64},
                         theta::Vector{Float64}, cfg::StageConfig)
    ll_vec, _ = MacroModelling.inversion_loglik_per_period(
        predict_tuple,
        state0,
        theta,
        obs_data,
        obs_sigma,
        shock_sigmas;
        eval_predict_fn = surrogate_predict,
        maxit = cfg.inv_maxit,
        tol = cfg.inv_tol,
        lambda = cfg.inv_lambda,
    )
    any(!isfinite, ll_vec) && return -Inf
    return sum(ll_vec)
end

function make_logposts(model, obs_ka, base_params, theta_idx, obs_idx, predict_tuple, surrogate_predict,
                       dgp::Dict{String,Any}, shock_sigmas, cfg::StageConfig,
                       direct_objective::Symbol, direct_logdet_method::Symbol)
    obs_data = Matrix{Float64}(dgp["obs_data"])
    obs_sigma = Vector{Float64}(dgp["obs_sigma"])
    state0 = Vector{Float64}(dgp["state0"])

    function direct_logpost_z(z::Vector{Float64})
        lp = log_prior_z(z)
        isfinite(lp) || return -Inf
        theta = exp.(z)
        ll = direct_loglik(model, obs_ka, base_params, theta_idx, obs_idx, dgp,
                           predict_tuple, theta, cfg, direct_objective, direct_logdet_method)
        isfinite(ll) || return -Inf
        return lp + ll
    end

    function surrogate_logpost_z(z::Vector{Float64})
        lp = log_prior_z(z)
        isfinite(lp) || return -Inf
        theta = exp.(z)
        ll = surrogate_loglik(predict_tuple, surrogate_predict, state0, obs_data, obs_sigma,
                              shock_sigmas, theta, cfg)
        isfinite(ll) || return -Inf
        return lp + ll
    end

    return direct_logpost_z, surrogate_logpost_z
end

function fd_gradient!(grad::Vector{Float64}, f::Function, x::Vector{Float64}, h::Float64)
    fx = f(x)
    isfinite(fx) || (fill!(grad, NaN); return fx)
    for i in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        xp[i] += h
        xm[i] -= h
        fp = f(xp)
        fm = f(xm)
        if isfinite(fp) && isfinite(fm)
            grad[i] = (fp - fm) / (2h)
        elseif isfinite(fp)
            grad[i] = (fp - fx) / h
        elseif isfinite(fm)
            grad[i] = (fx - fm) / h
        else
            grad[i] = NaN
        end
    end
    return fx
end

function check_finite_gradient(name::String, f::Function, z::Vector{Float64}, cfg::StageConfig)
    grad = zeros(Float64, length(z))
    val = fd_gradient!(grad, f, z, cfg.fd_eps)
    ok = isfinite(val) && all(isfinite, grad)
    println("  $name log posterior at theta_true: $(@sprintf("%.6f", val)); finite gradient=$(ok)")
    if !ok && startswith(name, "direct") && DIRECT_LAST_DIAGNOSTICS[] !== nothing
        diag = DIRECT_LAST_DIAGNOSTICS[]
        diag_status = get(diag, "status", "missing")
        diag_code = get(diag, "failure_code", get(diag, "code", "missing"))
        diag_rank = get(diag, "sep_inv_logdet_rank", "missing")
        diag_size = get(diag, "jacobian_size", "missing")
        println("    direct diagnostics: status=$diag_status, code=$diag_code, rank=$diag_rank, jacobian_size=$diag_size")
    end
    diag_copy = startswith(name, "direct") && DIRECT_LAST_DIAGNOSTICS[] !== nothing ?
        deepcopy(DIRECT_LAST_DIAGNOSTICS[]) : nothing
    return Dict{String,Any}("name" => name, "value" => val, "gradient" => grad, "ok" => ok, "diagnostics" => diag_copy)
end

function run_hmc_objective(name::String, logpost_z::Function, cfg::StageConfig; seed::Int)
    Random.seed!(seed)
    z_init = log.(THETA_BASELINE)
    logdensity = ValidationLogDensity(length(z_init), cfg.fd_eps, logpost_z)
    metric = UnitEuclideanMetric(length(z_init))
    hamiltonian = Hamiltonian(metric, logdensity)
    initial_eps = if cfg.hmc_initial_step_size > 0
        cfg.hmc_initial_step_size
    else
        try
            find_good_stepsize(hamiltonian, z_init)
        catch e
            @warn "$name find_good_stepsize failed; using 0.01" exception = (e, catch_backtrace())
            0.01
        end
    end
    integrator = Leapfrog(initial_eps)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth = cfg.hmc_max_depth)))
    adaptor = StepSizeAdaptor(cfg.hmc_target_accept, integrator)

    println("  Running $name NUTS: chains=$(cfg.chains), warmup=$(cfg.warmup), draws=$(cfg.draws)")
    chain_payloads = Vector{Dict{String,Any}}()
    for c in 1:cfg.chains
        seed_c = seed + 1000 * c
        Random.seed!(seed_c)
        local samples_z, stats
        elapsed = @elapsed begin
            samples_z, stats = sample(
                hamiltonian,
                kernel,
                z_init,
                cfg.warmup + cfg.draws,
                adaptor,
                cfg.warmup;
                progress = false,
                verbose = false,
            )
        end
        theta_chain = hcat([exp.(Vector{Float64}(s)) for s in samples_z]...)'
        theta_post = theta_chain[(cfg.warmup + 1):end, :]
        stats_post = stats[(cfg.warmup + 1):end]
        numerical_errors = [getproperty(s, :numerical_error) for s in stats_post]
        accept_rates = [getproperty(s, :acceptance_rate) for s in stats_post]
        tree_depths = [getproperty(s, :tree_depth) for s in stats_post]
        push!(chain_payloads, Dict{String,Any}(
            "name" => name,
            "chain" => c,
            "seed" => seed_c,
            "theta_post" => Matrix{Float64}(theta_post),
            "theta_all" => Matrix{Float64}(theta_chain),
            "n_numerical_errors" => sum(numerical_errors),
            "mean_accept" => Statistics.mean(accept_rates),
            "max_tree_depth" => maximum(tree_depths),
            "elapsed_s" => elapsed,
        ))
        println("    chain $c complete: elapsed=$(round(elapsed, digits=2))s, numerical_errors=$(sum(numerical_errors))")
    end
    return chain_payloads
end

function mcse(x::AbstractVector)
    n = length(x)
    n <= 1 && return NaN
    return Statistics.std(x) / sqrt(n)
end

function summarize_chains(chains::Vector{Dict{String,Any}})
    theta_post = vcat([c["theta_post"] for c in chains]...)
    return Dict{String,Any}(
        "mean" => vec(Statistics.mean(theta_post, dims = 1)),
        "mcse" => [mcse(theta_post[:, i]) for i in 1:size(theta_post, 2)],
        "q05" => [Statistics.quantile(theta_post[:, i], 0.05) for i in 1:size(theta_post, 2)],
        "q95" => [Statistics.quantile(theta_post[:, i], 0.95) for i in 1:size(theta_post, 2)],
        "n_draws" => size(theta_post, 1),
        "n_numerical_errors" => sum(c["n_numerical_errors"] for c in chains),
        "mean_accept" => Statistics.mean([c["mean_accept"] for c in chains]),
        "elapsed_s" => sum(c["elapsed_s"] for c in chains),
    )
end

function comparison_payload(direct_chains, surrogate_chains)
    ds = summarize_chains(direct_chains)
    ss = summarize_chains(surrogate_chains)
    rows = Vector{Dict{String,Any}}()
    for i in eachindex(THETA_NAMES)
        denom = sqrt(ds["mcse"][i]^2 + ss["mcse"][i]^2)
        diff_mcse = denom > 0 ? (ss["mean"][i] - ds["mean"][i]) / denom : NaN
        overlap = max(ds["q05"][i], ss["q05"][i]) <= min(ds["q95"][i], ss["q95"][i])
        direct_cover = ds["q05"][i] <= THETA_TRUE[i] <= ds["q95"][i]
        surrogate_cover = ss["q05"][i] <= THETA_TRUE[i] <= ss["q95"][i]
        push!(rows, Dict{String,Any}(
            "parameter" => string(THETA_NAMES[i]),
            "true" => THETA_TRUE[i],
            "direct_mean" => ds["mean"][i],
            "direct_mcse" => ds["mcse"][i],
            "direct_q05" => ds["q05"][i],
            "direct_q95" => ds["q95"][i],
            "surrogate_mean" => ss["mean"][i],
            "surrogate_mcse" => ss["mcse"][i],
            "surrogate_q05" => ss["q05"][i],
            "surrogate_q95" => ss["q95"][i],
            "mean_diff_combined_mcse" => diff_mcse,
            "interval_overlap" => overlap,
            "direct_true_coverage" => direct_cover,
            "surrogate_true_coverage" => surrogate_cover,
        ))
    end
    return Dict{String,Any}("direct" => ds, "surrogate" => ss, "rows" => rows)
end

function stacked_theta_post(chains::Vector{Dict{String,Any}})
    return vcat([c["theta_post"] for c in chains]...)
end

function select_audit_thetas(surrogate_chains::Vector{Dict{String,Any}}, n_draws::Int; seed::Int)
    theta_post = stacked_theta_post(surrogate_chains)
    n_post = size(theta_post, 1)
    n_take = min(max(n_draws, 0), n_post)
    selected = Vector{Vector{Float64}}()
    labels = String[]

    push!(selected, copy(THETA_TRUE))
    push!(labels, "theta_true")
    push!(selected, copy(THETA_BASELINE))
    push!(labels, "prior_center")

    n_take == 0 && return selected, labels
    rng = MersenneTwister(seed)
    draw_idx = sort(randperm(rng, n_post)[1:n_take])
    for idx in draw_idx
        push!(selected, vec(theta_post[idx, :]))
        push!(labels, "surrogate_draw_$idx")
    end
    return selected, labels
end

function direct_audit_payload(direct_logpost_z::Function, surrogate_logpost_z::Function,
                              surrogate_chains::Vector{Dict{String,Any}}, n_draws::Int; seed::Int)
    thetas, labels = select_audit_thetas(surrogate_chains, n_draws; seed = seed)
    rows = Vector{Dict{String,Any}}()
    direct_elapsed = 0.0
    surrogate_elapsed = 0.0
    for (label, theta) in zip(labels, thetas)
        z = log.(theta)
        local direct_lp, surrogate_lp
        direct_t = @elapsed direct_lp = direct_logpost_z(z)
        surrogate_t = @elapsed surrogate_lp = surrogate_logpost_z(z)
        direct_elapsed += direct_t
        surrogate_elapsed += surrogate_t
        push!(rows, Dict{String,Any}(
            "label" => label,
            "theta" => theta,
            "direct_logpost" => direct_lp,
            "surrogate_logpost" => surrogate_lp,
            "surrogate_minus_direct" => surrogate_lp - direct_lp,
            "direct_elapsed_s" => direct_t,
            "surrogate_elapsed_s" => surrogate_t,
            "direct_diagnostics" => DIRECT_LAST_DIAGNOSTICS[] === nothing ? nothing : deepcopy(DIRECT_LAST_DIAGNOSTICS[]),
        ))
        println("  audit $label: direct=$(fmt(direct_lp)), surrogate=$(fmt(surrogate_lp)), diff=$(fmt(surrogate_lp - direct_lp)), direct_elapsed=$(round(direct_t, digits=2))s")
    end
    diffs = [row["surrogate_minus_direct"] for row in rows if isfinite(row["surrogate_minus_direct"])]
    return Dict{String,Any}(
        "n_requested_posterior_draws" => n_draws,
        "n_rows" => length(rows),
        "rows" => rows,
        "mean_surrogate_minus_direct" => isempty(diffs) ? NaN : Statistics.mean(diffs),
        "max_abs_surrogate_minus_direct" => isempty(diffs) ? NaN : maximum(abs.(diffs)),
        "direct_elapsed_s" => direct_elapsed,
        "surrogate_elapsed_s" => surrogate_elapsed,
    )
end

latex_escape(s::AbstractString) = replace(s, "_" => "\\_")

function write_direct_audit_table(path::String, audit::Dict{String,Any})
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrrr}")
        println(io, "\\toprule")
        println(io, "Point & \$\\sigma_a\$ & \$\\sigma_z\$ & \$\\sigma_\\nu\$ & Direct log post. & Surrogate log post. & Difference \\\\")
        println(io, "\\midrule")
        for row in audit["rows"]
            theta = row["theta"]
            println(io, "$(latex_escape(row["label"])) & $(fmt(theta[1])) & $(fmt(theta[2])) & $(fmt(theta[3])) & $(fmt(row["direct_logpost"])) & $(fmt(row["surrogate_logpost"])) & $(fmt(row["surrogate_minus_direct"])) \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function write_manifest(path::String, opts::CliOptions, cfg::StageConfig, model)
    mkpath(path)
    manifest = Dict{String,Any}(
        "run_id" => opts.run_id,
        "stage" => string(cfg.name),
        "dry_run" => opts.dry_run,
        "likelihood_smoke_only" => opts.likelihood_smoke_only,
        "model" => "Gali_2015_chapter_3_obc",
        "model_file" => joinpath(REPO_ROOT, "models", "Gali_2015_chapter_3_obc.jl"),
        "observables" => string.(OBSERVABLES),
        "theta_names" => string.(THETA_NAMES),
        "theta_true" => THETA_TRUE,
        "theta_baseline" => THETA_BASELINE,
        "prior_log_sd" => PRIOR_LOG_SD,
        "prior_lower" => PRIOR_LOWER,
        "prior_upper" => PRIOR_UPPER,
        "periods" => cfg.periods,
        "train_samples" => cfg.train_samples,
        "hidden_units" => cfg.hidden_units,
        "warmup" => cfg.warmup,
        "draws" => cfg.draws,
        "chains" => cfg.chains,
        "sep_horizon" => cfg.sep_horizon,
        "sep_maxit" => cfg.sep_maxit,
        "sep_tol" => cfg.sep_tol,
        "sep_accept_tol" => cfg.sep_accept_tol,
        "inv_maxit" => cfg.inv_maxit,
        "inv_tol" => cfg.inv_tol,
        "inv_lambda" => cfg.inv_lambda,
        "hmc_max_depth" => cfg.hmc_max_depth,
        "hmc_target_accept" => cfg.hmc_target_accept,
        "hmc_initial_step_size" => cfg.hmc_initial_step_size,
        "hmc_objectives" => string(opts.hmc_objectives),
        "surrogate_design" => string(opts.surrogate_design),
        "rom_filter_anchor_repeats" => opts.surrogate_design == :local_path ? ROM_FILTER_ANCHOR_REPEATS : 0,
        "direct_objective" => string(opts.direct_objective),
        "direct_logdet_method" => string(opts.direct_logdet_method),
        "direct_audit_draws" => opts.direct_audit_draws,
        "shock_scaling" => "none; supplied shocks are unit structural innovations",
        "artifact_schema" => [
            "manifest.toml",
            "synthetic_dgp.jls",
            "surrogate_bundle.jls",
            "direct_chains.jls",
            "surrogate_chains.jls",
            "direct_audit_payload.jls",
            "direct_audit_table.tex",
            "comparison_payload.jls",
            "SUMMARY.md",
            "comparison_table.tex",
        ],
        "available_model_variables" => string.(model.var),
        "available_model_shocks" => string.(model.exo),
    )
    open(joinpath(path, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    return manifest
end

function fmt(x; digits = 5)
    isfinite(x) || return "NA"
    return @sprintf("%.*g", digits, x)
end

function write_latex_table(path::String, cmp::Dict{String,Any})
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrrrrrc}")
        println(io, "\\toprule")
        println(io, "Parameter & True & Direct mean & Direct MCSE & Direct 90\\% CI & Surrogate mean & Surrogate MCSE & Surrogate 90\\% CI & Diff/MCSE & Overlap \\\\")
        println(io, "\\midrule")
        for row in cmp["rows"]
            direct_ci = "[$(fmt(row["direct_q05"])), $(fmt(row["direct_q95"]))]"
            surrogate_ci = "[$(fmt(row["surrogate_q05"])), $(fmt(row["surrogate_q95"]))]"
            overlap = row["interval_overlap"] ? "Yes" : "No"
            println(io, "$(row["parameter"]) & $(fmt(row["true"])) & $(fmt(row["direct_mean"])) & $(fmt(row["direct_mcse"])) & $direct_ci & $(fmt(row["surrogate_mean"])) & $(fmt(row["surrogate_mcse"])) & $surrogate_ci & $(fmt(row["mean_diff_combined_mcse"], digits=3)) & $overlap \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function write_summary(path::String, cfg::StageConfig, surrogate::RandomFeatureSurrogate,
                       checks::Vector{Dict{String,Any}}, cmp::Union{Nothing,Dict{String,Any}};
                       surrogate_design::Symbol = :unknown,
                       path_diagnostics::Union{Nothing,Dict{String,Any}} = nothing,
                       direct_audit::Union{Nothing,Dict{String,Any}} = nothing)
    open(path, "w") do io
        println(io, "# Gali Direct SEP-HMC vs Surrogate-HMC Validation")
        println(io)
        println(io, "**Stage**: `$(cfg.name)`")
        println(io, "**Model**: `models/Gali_2015_chapter_3_obc.jl`")
        println(io, "**Observables**: `$(join(string.(OBSERVABLES), "`, `"))`")
        println(io, "**Parameters**: `$(join(string.(THETA_NAMES), "`, `"))`")
        println(io)
        println(io, "## Surrogate Fit")
        println(io)
        println(io, "- Training design: `$(surrogate_design)`")
        if surrogate_design == :local_path
            println(io, "- ROM-filter anchor repeats: $(ROM_FILTER_ANCHOR_REPEATS)")
        end
        println(io, "- Feature kind: `$(surrogate.feature_kind)`")
        println(io, "- Validation RMSE: $(join(fmt.(surrogate.valid_rmse), ", "))")
        println(io, "- Validation RRMSE on observable scale: $(join(fmt.(surrogate.valid_rrmse_obs), ", "))")
        println(io, "- Validation RRMSE on residual scale: $(join(fmt.(surrogate.valid_rrmse_resid), ", "))")
        if path_diagnostics !== nothing
            println(io, "- DGP-path RRMSE on observable scale: $(join(fmt.(path_diagnostics["path_rrmse_obs"]), ", "))")
            println(io, "- DGP-path RRMSE on residual scale: $(join(fmt.(path_diagnostics["path_rrmse_resid"]), ", "))")
            if haskey(path_diagnostics, "rom_path_rrmse_obs")
                println(io, "- ROM-filter-path RRMSE on observable scale: $(join(fmt.(path_diagnostics["rom_path_rrmse_obs"]), ", "))")
                println(io, "- ROM-filter-path RRMSE on residual scale: $(join(fmt.(path_diagnostics["rom_path_rrmse_resid"]), ", "))")
            end
        end
        println(io)
        println(io, "## Smoke Checks")
        for check in checks
            println(io, "- `$(check["name"])`: log posterior=$(fmt(check["value"])); finite gradient=$(check["ok"])")
            diag = get(check, "diagnostics", nothing)
            if diag !== nothing
                kind = get(diag, "kind", "missing")
                status = get(diag, "status", "missing")
                code = get(diag, "failure_code", "none")
                rank = get(diag, "sep_inv_logdet_rank", get(diag, "jacobian_ranks", "missing"))
                jsize = get(diag, "jacobian_size", "missing")
                method = get(diag, "sep_inv_logdet_method", "missing")
                max_iter = get(diag, "max_iteration", "missing")
                resid_norm = get(diag, "mean_residual_norm", "missing")
                shock_norm = get(diag, "mean_shock_norm", "missing")
                println(io, "  - direct diagnostics: kind=`$kind`, status=`$status`, failure_code=`$code`, logdet_method=`$method`, rank=`$rank`, jacobian_size=`$jsize`, max_iteration=`$max_iter`, mean_residual_norm=`$resid_norm`, mean_shock_norm=`$shock_norm`")
            end
        end
        if direct_audit !== nothing
            println(io)
            println(io, "## Direct SEP Audit")
            println(io)
            println(io, "- Requested surrogate posterior draws: $(direct_audit["n_requested_posterior_draws"])")
            println(io, "- Audit rows including reference points: $(direct_audit["n_rows"])")
            println(io, "- Mean surrogate-minus-direct log posterior: $(fmt(direct_audit["mean_surrogate_minus_direct"]))")
            println(io, "- Max absolute surrogate-minus-direct log posterior: $(fmt(direct_audit["max_abs_surrogate_minus_direct"]))")
            println(io, "- Direct audit elapsed time: $(fmt(direct_audit["direct_elapsed_s"], digits=4)) seconds")
            println(io)
            println(io, "| Point | `std_a` | `std_z` | `std_nu` | Direct log post. | Surrogate log post. | Difference |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|")
            for row in direct_audit["rows"]
                theta = row["theta"]
                println(io, "| $(row["label"]) | $(fmt(theta[1])) | $(fmt(theta[2])) | $(fmt(theta[3])) | $(fmt(row["direct_logpost"])) | $(fmt(row["surrogate_logpost"])) | $(fmt(row["surrogate_minus_direct"])) |")
            end
        end
        cmp === nothing && return
        println(io)
        println(io, "## Posterior Comparison")
        println(io)
        println(io, "| Parameter | True | Direct mean | Direct MCSE | Direct 90% CI | Surrogate mean | Surrogate MCSE | Surrogate 90% CI | Diff/MCSE | Overlap | Coverage |")
        println(io, "|---|---:|---:|---:|---|---:|---:|---|---:|:---:|:---:|")
        for row in cmp["rows"]
            direct_ci = "[$(fmt(row["direct_q05"])), $(fmt(row["direct_q95"]))]"
            surrogate_ci = "[$(fmt(row["surrogate_q05"])), $(fmt(row["surrogate_q95"]))]"
            overlap = row["interval_overlap"] ? "Yes" : "No"
            cover = (row["direct_true_coverage"] && row["surrogate_true_coverage"]) ? "Both" :
                    row["direct_true_coverage"] ? "Direct" :
                    row["surrogate_true_coverage"] ? "Surrogate" : "Neither"
            println(io, "| $(row["parameter"]) | $(fmt(row["true"])) | $(fmt(row["direct_mean"])) | $(fmt(row["direct_mcse"])) | $direct_ci | $(fmt(row["surrogate_mean"])) | $(fmt(row["surrogate_mcse"])) | $surrogate_ci | $(fmt(row["mean_diff_combined_mcse"], digits=3)) | $overlap | $cover |")
        end
        println(io)
        println(io, "## Diagnostics")
        println(io)
        println(io, "- Direct post-warmup numerical errors: $(cmp["direct"]["n_numerical_errors"])")
        println(io, "- Surrogate post-warmup numerical errors: $(cmp["surrogate"]["n_numerical_errors"])")
        println(io, "- Direct mean acceptance: $(fmt(cmp["direct"]["mean_accept"], digits=4))")
        println(io, "- Surrogate mean acceptance: $(fmt(cmp["surrogate"]["mean_accept"], digits=4))")
    end
end

function run_validation(opts::CliOptions)
    cfg = apply_overrides(stage_config(opts.stage), opts)
    out_dir = default_out_dir(opts)
    mkpath(out_dir)
    model = load_model()
    theta_idx = parameter_indices(model, THETA_NAMES)
    obs_idx = variable_indices(model, OBSERVABLES)
    manifest = write_manifest(out_dir, opts, cfg, model)
    println("Wrote manifest: $(joinpath(out_dir, "manifest.toml"))")

    if opts.dry_run
        println("Dry run complete. No SEP simulations or HMC chains were launched.")
        return Dict{String,Any}("out_dir" => out_dir, "manifest" => manifest)
    end

    base_params = Float64.(model.parameter_values)
    println("Building baseline ROM1 predictor...")
    cache, predict_tuple, _, _ = build_baseline_rom(model, base_params, obs_idx)

    println("Generating common synthetic SEP DGP...")
    dgp = simulate_sep_dataset(model, THETA_TRUE, cfg; seed = opts.seed)
    serialize(joinpath(out_dir, "synthetic_dgp.jls"), dgp)
    obs_ka = KeyedArray(Matrix{Float64}(dgp["obs_data"]); Variable = OBSERVABLES, Time = 1:Int(dgp["periods"]))

    println("Training obs-only ROM1-residual surrogate ($(opts.surrogate_design) design)...")
    X, Y, Ysep = collect_training_data(model, cfg, cache, obs_idx, dgp, opts.surrogate_design;
                                       predict_tuple = predict_tuple,
                                       seed = opts.seed + 10)
    score_data = opts.surrogate_design == :local_path ?
        collect_rom_filter_surrogate_data(model, cfg, cache, obs_idx, dgp, predict_tuple, base_params, theta_idx) :
        nothing
    if score_data !== nothing
        anchor_X, anchor_Y, anchor_Ysep = score_data
        for _ in 1:ROM_FILTER_ANCHOR_REPEATS
            X = hcat(X, anchor_X)
            Y = hcat(Y, anchor_Y)
            Ysep = hcat(Ysep, anchor_Ysep)
        end
    end
    surrogate = if score_data === nothing
        train_random_feature_surrogate(X, Y, Ysep, cfg.hidden_units; seed = opts.seed + 20)
    else
        score_X, score_Y, score_Ysep = score_data
        train_random_feature_surrogate(
            X,
            Y,
            Ysep,
            cfg.hidden_units;
            seed = opts.seed + 20,
            score_X = score_X,
            score_Y = score_Y,
            score_Ysep = score_Ysep,
        )
    end
    surrogate_predict = make_surrogate_predict(predict_tuple, surrogate)
    path_diagnostics = surrogate_path_diagnostics(surrogate, cache, obs_idx, dgp)
    rom_path_diagnostics = surrogate_rom_filter_diagnostics(
        surrogate,
        model,
        base_params,
        theta_idx,
        cache,
        obs_idx,
        predict_tuple,
        dgp,
        cfg,
    )
    if path_diagnostics !== nothing
        merge!(path_diagnostics, rom_path_diagnostics)
    else
        path_diagnostics = rom_path_diagnostics
    end
    serialize(joinpath(out_dir, "surrogate_bundle.jls"), Dict{String,Any}(
        "surrogate" => surrogate,
        "observables" => OBSERVABLES,
        "theta_names" => THETA_NAMES,
        "train_samples" => size(X, 2),
        "feature_dimension" => size(X, 1),
        "surrogate_design" => opts.surrogate_design,
        "rom_filter_anchor_repeats" => score_data === nothing ? 0 : ROM_FILTER_ANCHOR_REPEATS,
        "surrogate_feature_kind" => surrogate.feature_kind,
        "surrogate_bandwidth" => surrogate.bandwidth,
        "path_diagnostics" => path_diagnostics,
    ))
    println("  validation RRMSE (observable scale): $(join(fmt.(surrogate.valid_rrmse_obs), ", "))")
    path_diagnostics === nothing ||
        println("  DGP-path RRMSE (observable scale): $(join(fmt.(path_diagnostics["path_rrmse_obs"]), ", "))")
    path_diagnostics === nothing ||
        println("  ROM-filter-path RRMSE (observable scale): $(join(fmt.(path_diagnostics["rom_path_rrmse_obs"]), ", "))")

    full_validation_rrmse = if path_diagnostics !== nothing && haskey(path_diagnostics, "rom_path_rrmse_obs")
        maximum(path_diagnostics["rom_path_rrmse_obs"])
    else
        maximum(surrogate.valid_rrmse_obs)
    end
    if cfg.name == :full && full_validation_rrmse > 0.005
        write_summary(joinpath(out_dir, "SUMMARY.md"), cfg, surrogate, Dict{String,Any}[], nothing;
                      surrogate_design = opts.surrogate_design,
                      path_diagnostics = path_diagnostics)
        error("Full-stage ROM-filter-path surrogate validation RRMSE exceeds 0.005; refusing paper-stage run.")
    end

    shock_sigmas = make_shock_sigmas(model)
    direct_logpost_z, surrogate_logpost_z = make_logposts(
        model,
        obs_ka,
        base_params,
        theta_idx,
        obs_idx,
        predict_tuple,
        surrogate_predict,
        dgp,
        shock_sigmas,
        cfg,
        opts.direct_objective,
        opts.direct_logdet_method,
    )

    println("Checking finite likelihoods and finite-difference gradients...")
    checks = Vector{Dict{String,Any}}()
    if opts.hmc_objectives in (:both, :direct, :none)
        push!(checks, check_finite_gradient("direct at theta_true", direct_logpost_z, log.(THETA_TRUE), cfg))
        push!(checks, check_finite_gradient("direct at prior_center", direct_logpost_z, log.(THETA_BASELINE), cfg))
    end
    if opts.hmc_objectives in (:both, :surrogate, :none)
        push!(checks, check_finite_gradient("surrogate at theta_true", surrogate_logpost_z, log.(THETA_TRUE), cfg))
        push!(checks, check_finite_gradient("surrogate at prior_center", surrogate_logpost_z, log.(THETA_BASELINE), cfg))
    end
    serialize(joinpath(out_dir, "smoke_checks.jls"), checks)
    if !all(c -> c["ok"], checks)
        write_summary(joinpath(out_dir, "SUMMARY.md"), cfg, surrogate, checks, nothing;
                      surrogate_design = opts.surrogate_design,
                      path_diagnostics = path_diagnostics)
        error("Finite likelihood/gradient smoke checks failed. See $(joinpath(out_dir, "SUMMARY.md")).")
    end

    if opts.likelihood_smoke_only || opts.hmc_objectives == :none
        write_summary(joinpath(out_dir, "SUMMARY.md"), cfg, surrogate, checks, nothing;
                      surrogate_design = opts.surrogate_design,
                      path_diagnostics = path_diagnostics)
        println("Likelihood smoke complete. No NUTS chains were launched.")
        return Dict{String,Any}("out_dir" => out_dir, "checks" => checks)
    end

    direct_chains = nothing
    surrogate_chains = nothing
    direct_audit = nothing
    if opts.hmc_objectives in (:both, :direct)
        direct_chains = run_hmc_objective("direct_sep", direct_logpost_z, cfg; seed = opts.seed + 100)
        serialize(joinpath(out_dir, "direct_chains.jls"), direct_chains)
    end
    if opts.hmc_objectives in (:both, :surrogate)
        surrogate_chains = run_hmc_objective("surrogate", surrogate_logpost_z, cfg; seed = opts.seed + 200)
        serialize(joinpath(out_dir, "surrogate_chains.jls"), surrogate_chains)
    end
    if opts.direct_audit_draws > 0
        if surrogate_chains === nothing
            @warn "direct-audit-draws was requested, but no surrogate chains are available; skipping direct audit."
        else
            println("  Running direct SEP audit on surrogate posterior draws: requested=$(opts.direct_audit_draws)")
            direct_audit = direct_audit_payload(
                direct_logpost_z,
                surrogate_logpost_z,
                surrogate_chains,
                opts.direct_audit_draws;
                seed = opts.seed + 300,
            )
            serialize(joinpath(out_dir, "direct_audit_payload.jls"), direct_audit)
            write_direct_audit_table(joinpath(out_dir, "direct_audit_table.tex"), direct_audit)
        end
    end

    cmp = nothing
    partial = Dict{String,Any}()
    if direct_chains !== nothing
        partial["direct"] = summarize_chains(direct_chains)
    end
    if surrogate_chains !== nothing
        partial["surrogate"] = summarize_chains(surrogate_chains)
    end
    if direct_chains !== nothing && surrogate_chains !== nothing
        cmp = comparison_payload(direct_chains, surrogate_chains)
        serialize(joinpath(out_dir, "comparison_payload.jls"), cmp)
        write_latex_table(joinpath(out_dir, "comparison_table.tex"), cmp)
    elseif !isempty(partial)
        serialize(joinpath(out_dir, "partial_hmc_summary.jls"), partial)
    end
    write_summary(joinpath(out_dir, "SUMMARY.md"), cfg, surrogate, checks, cmp;
                  surrogate_design = opts.surrogate_design,
                  path_diagnostics = path_diagnostics,
                  direct_audit = direct_audit)
    println("Wrote summary: $(joinpath(out_dir, "SUMMARY.md"))")
    return Dict{String,Any}("out_dir" => out_dir, "checks" => checks, "comparison" => cmp, "partial" => partial)
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_args(ARGS)
    result = run_validation(opts)
    println("Artifacts: $(result["out_dir"])")
end
