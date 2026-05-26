#!/usr/bin/env julia

# First-stage Galí ELB validation for the ROM1-residual surrogate pipeline.
#
# This runner starts with one parameter, std_nu, and a deliberately binding ELB
# episode.  It trains a surrogate on the observable residual
#
#     direct SEP observable - ROM1 observable
#
# and compares direct-SEP and ROM1+residual posterior grids on the same
# synthetic path.  The first stage conditions on the DGP state and shock path,
# which isolates the nonlinear transition approximation before adding the
# inversion-filter layer.

using Dates
using Distributions
using LinearAlgebra
using MacroModelling
using Printf
using Random
using Serialization
using Statistics
using TOML

const ELB_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

# Reuse the maintained Galí model, SEP one-step evaluator, ROM cache utilities,
# and small random-feature residual learner used by the existing validation
# runner.  The program guard in the included file prevents it from executing.
include(joinpath(ELB_REPO_ROOT, "scripts", "gali_direct_sep_surrogate_hmc_validation.jl"))

const ELB_MODEL = GALI_VALIDATION_MODEL
const ELB_OBSERVABLES = [:log_y, :pi_ann, :i_ann]
const ELB_THETA_NAMES = [:std_nu]
const ELB_THETA_BASELINE = [0.0025]
const ELB_THETA_TRUE = [0.0050]
const ELB_PRIOR_LOG_SD = 0.35
const ELB_PRIOR_LOWER = 0.25 .* ELB_THETA_BASELINE
const ELB_PRIOR_UPPER = 4.0 .* ELB_THETA_BASELINE

Base.@kwdef struct ElbValidationConfig
    stage::Symbol
    periods::Int
    train_samples::Int
    hidden_units::Int
    grid_size::Int
    sep_horizon::Int
    sep_maxit::Int
    sep_tol::Float64
    sep_accept_tol::Float64
    measurement_noise_share::Float64
    min_elb_share::Float64
    posterior_mean_tolerance_sd::Float64
end

Base.@kwdef struct ElbValidationOptions
    stage::Symbol = :smoke
    dry_run::Bool = false
    run_id::String = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    out_dir::String = ""
    seed::Int = 20260513
    periods_override::Union{Nothing,Int} = nothing
    train_samples_override::Union{Nothing,Int} = nothing
    grid_size_override::Union{Nothing,Int} = nothing
    hidden_units_override::Union{Nothing,Int} = nothing
    sep_maxit_override::Union{Nothing,Int} = nothing
    sep_accept_tol_override::Union{Nothing,Float64} = nothing
end

function elb_stage_config(stage::Symbol)
    if stage == :smoke
        return ElbValidationConfig(
            stage = stage,
            periods = 4,
            train_samples = 8,
            hidden_units = 8,
            grid_size = 5,
            sep_horizon = 6,
            sep_maxit = 100,
            sep_tol = 1e-6,
            sep_accept_tol = 5e-2,
            measurement_noise_share = 0.05,
            min_elb_share = 0.25,
            posterior_mean_tolerance_sd = 0.50,
        )
    elseif stage == :grid1
        return ElbValidationConfig(
            stage = stage,
            periods = 12,
            train_samples = 120,
            hidden_units = 48,
            grid_size = 21,
            sep_horizon = 8,
            sep_maxit = 120,
            sep_tol = 1e-7,
            sep_accept_tol = 1e-2,
            measurement_noise_share = 0.05,
            min_elb_share = 0.20,
            posterior_mean_tolerance_sd = 0.25,
        )
    elseif stage == :paper1
        return ElbValidationConfig(
            stage = stage,
            periods = 24,
            train_samples = 500,
            hidden_units = 96,
            grid_size = 41,
            sep_horizon = 12,
            sep_maxit = 100,
            sep_tol = 1e-8,
            sep_accept_tol = 1e-5,
            measurement_noise_share = 0.05,
            min_elb_share = 0.20,
            posterior_mean_tolerance_sd = 0.20,
        )
    else
        error("Unknown --stage=$stage. Use smoke, grid1, or paper1.")
    end
end

function apply_elb_overrides(cfg::ElbValidationConfig, opts::ElbValidationOptions)
    return ElbValidationConfig(
        stage = cfg.stage,
        periods = something(opts.periods_override, cfg.periods),
        train_samples = something(opts.train_samples_override, cfg.train_samples),
        hidden_units = something(opts.hidden_units_override, cfg.hidden_units),
        grid_size = something(opts.grid_size_override, cfg.grid_size),
        sep_horizon = cfg.sep_horizon,
        sep_maxit = something(opts.sep_maxit_override, cfg.sep_maxit),
        sep_tol = cfg.sep_tol,
        sep_accept_tol = something(opts.sep_accept_tol_override, cfg.sep_accept_tol),
        measurement_noise_share = cfg.measurement_noise_share,
        min_elb_share = cfg.min_elb_share,
        posterior_mean_tolerance_sd = cfg.posterior_mean_tolerance_sd,
    )
end

function parse_elb_bool(s::AbstractString)
    sl = lowercase(strip(s))
    sl in ("true", "1", "yes", "y") && return true
    sl in ("false", "0", "no", "n") && return false
    error("Cannot parse boolean value: $s")
end

function parse_elb_args(args)
    defaults = ElbValidationOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    stage = Symbol(get(values, "stage", string(defaults.stage)))
    stage in (:smoke, :grid1, :paper1) || error("Unknown --stage=$stage. Use smoke, grid1, or paper1.")
    return ElbValidationOptions(
        stage = stage,
        dry_run = parse_elb_bool(get(values, "dry-run", string(defaults.dry_run))),
        run_id = get(values, "run-id", defaults.run_id),
        out_dir = get(values, "out-dir", defaults.out_dir),
        seed = parse(Int, get(values, "seed", string(defaults.seed))),
        periods_override = haskey(values, "periods") ? parse(Int, values["periods"]) : nothing,
        train_samples_override = haskey(values, "train-samples") ? parse(Int, values["train-samples"]) : nothing,
        grid_size_override = haskey(values, "grid-size") ? parse(Int, values["grid-size"]) : nothing,
        hidden_units_override = haskey(values, "hidden-units") ? parse(Int, values["hidden-units"]) : nothing,
        sep_maxit_override = haskey(values, "sep-maxit") ? parse(Int, values["sep-maxit"]) : nothing,
        sep_accept_tol_override = haskey(values, "sep-accept-tol") ? parse(Float64, values["sep-accept-tol"]) : nothing,
    )
end

function default_elb_out_dir(opts::ElbValidationOptions)
    isempty(opts.out_dir) || return opts.out_dir
    return joinpath(ELB_REPO_ROOT, ".local_artifacts", "gali_elb_rom1_residual_validation", opts.run_id)
end

function find_symbol_index(names, target::Symbol, label::AbstractString)
    idx = findfirst(==(target), Symbol.(names))
    idx === nothing && error("Could not find $target in $label.")
    return idx
end

function parameter_value(model, params::Vector{Float64}, name::Symbol)
    idx = find_symbol_index(model.parameters, name, "model parameters")
    return params[idx]
end

function elb_prior_logpost_z(z::Real)
    theta = exp(Float64(z))
    (theta < ELB_PRIOR_LOWER[1] || theta > ELB_PRIOR_UPPER[1]) && return -Inf
    return Distributions.logpdf(Distributions.Normal(log(ELB_THETA_BASELINE[1]), ELB_PRIOR_LOG_SD), Float64(z))
end

function elb_grid(cfg::ElbValidationConfig)
    lo, hi = log(ELB_PRIOR_LOWER[1]), log(ELB_PRIOR_UPPER[1])
    return exp.(collect(range(lo, hi; length = cfg.grid_size)))
end

function base_stage_config(cfg::ElbValidationConfig)
    return StageConfig(
        name = cfg.stage,
        periods = cfg.periods,
        train_samples = cfg.train_samples,
        hidden_units = cfg.hidden_units,
        warmup = 0,
        draws = 0,
        chains = 1,
        sep_horizon = cfg.sep_horizon,
        sep_maxit = cfg.sep_maxit,
        sep_tol = cfg.sep_tol,
        sep_accept_tol = cfg.sep_accept_tol,
        inv_maxit = 0,
        inv_tol = 1e-6,
        inv_lambda = 1e-4,
        hmc_max_depth = 1,
        hmc_target_accept = 0.8,
        hmc_initial_step_size = 0.001,
        fd_eps = 1e-4,
    )
end

function build_elb_shocks(model, cfg::ElbValidationConfig; seed::Int)
    rng = MersenneTwister(seed)
    shocks = zeros(Float64, length(model.exo), cfg.periods)
    eps_a_idx = find_symbol_index(model.exo, :eps_a, "model shocks")
    eps_z_idx = find_symbol_index(model.exo, :eps_z, "model shocks")
    eps_nu_idx = find_symbol_index(model.exo, :eps_nu, "model shocks")

    # Keep technology shocks small and use a negative monetary-policy shock block
    # to force the max() lower-bound kink to bind.
    shocks[eps_a_idx, :] .= 0.05 .* randn(rng, cfg.periods)
    shocks[eps_z_idx, :] .= 0.05 .* randn(rng, cfg.periods)
    pattern = [-22.0, -14.0, -8.0, -4.0]
    for t in 1:cfg.periods
        if t <= length(pattern)
            shocks[eps_nu_idx, t] = pattern[t]
        else
            shocks[eps_nu_idx, t] = 0.15 * randn(rng)
        end
    end
    return shocks
end

function simulate_elb_dgp(model, cfg::ElbValidationConfig; seed::Int)
    base_params = Float64.(model.parameter_values)
    theta_idx = parameter_indices(model, ELB_THETA_NAMES)
    obs_idx = variable_indices(model, ELB_OBSERVABLES)
    r_idx = find_symbol_index(model.var, :R, "model variables")
    y_idx = find_symbol_index(model.var, :Y, "model variables")
    pi_idx = find_symbol_index(model.var, :Pi, "model variables")
    nu_idx = find_symbol_index(model.var, :nu, "model variables")
    i_ann_idx = find_symbol_index(model.var, :i_ann, "model variables")
    params = inject_theta(base_params, theta_idx, ELB_THETA_TRUE)
    MacroModelling.write_parameters_input!(model, params, verbose = false)

    shocks = build_elb_shocks(model, cfg; seed = seed)
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
    hasproperty(res, :errorflag) && res.errorflag && error("SEP DGP simulation failed.")
    sim = Float64.(Array(res.simulation))
    T_avail = min(cfg.periods, size(sim, 2) - 1)
    T_avail >= 1 || error("SEP DGP returned no usable simulated periods.")
    obs = sim[obs_idx, 2:(T_avail + 1)]
    obs_sigma = max.(cfg.measurement_noise_share .* vec(Statistics.std(obs, dims = 2; corrected = false)), 1e-4)
    r_path = vec(sim[r_idx, 2:(T_avail + 1)])
    y_path = vec(sim[y_idx, 2:(T_avail + 1)])
    pi_path = vec(sim[pi_idx, 2:(T_avail + 1)])
    nu_path = vec(sim[nu_idx, 2:(T_avail + 1)])
    i_ann_path = vec(sim[i_ann_idx, 2:(T_avail + 1)])
    y_ss = sim[y_idx, 1]
    rbar = parameter_value(model, params, :R̄)
    beta = parameter_value(model, params, :β)
    phi_pi = parameter_value(model, params, :ϕᵖⁱ)
    phi_y = parameter_value(model, params, :ϕʸ)
    shadow_rate = (1 / beta) .* (pi_path .^ phi_pi) .* ((y_path ./ y_ss) .^ phi_y) .* exp.(nu_path)
    elb_binding = shadow_rate .<= rbar

    return Dict{String,Any}(
        "theta_true" => copy(ELB_THETA_TRUE),
        "theta_names" => string.(ELB_THETA_NAMES),
        "observables" => string.(ELB_OBSERVABLES),
        "obs_data" => obs,
        "obs_sigma" => obs_sigma,
        "state0" => sim[:, 1],
        "state_path" => sim[:, 1:(T_avail + 1)],
        "shocks" => shocks[:, 1:T_avail],
        "periods" => T_avail,
        "r_path" => r_path,
        "shadow_rate_path" => shadow_rate,
        "i_ann_path" => i_ann_path,
        "elb_binding" => elb_binding,
        "elb_share" => mean(elb_binding),
    )
end

function scale_std_nu_shock(model, shock::AbstractVector, theta::AbstractVector)
    scaled = copy(Float64.(shock))
    eps_nu_idx = find_symbol_index(model.exo, :eps_nu, "model shocks")
    scaled[eps_nu_idx] *= theta[1] / ELB_THETA_BASELINE[1]
    return scaled
end

function rom_obs_std_nu(model, cache::RomCache, obs_idx::Vector{Int},
                        state::AbstractVector, shock::AbstractVector, theta::AbstractVector)
    scaled_shock = scale_std_nu_shock(model, shock, theta)
    return rom_step_full(cache, state, scaled_shock)[obs_idx]
end

function collect_elb_residual_training_data(model, cfg::ElbValidationConfig, cache::RomCache,
                                            obs_idx::Vector{Int}, dgp::Dict{String,Any};
                                            seed::Int)
    rng = MersenneTwister(seed)
    base_params = Float64.(model.parameter_values)
    sep_cfg = base_stage_config(cfg)
    theta_idx = parameter_indices(model, ELB_THETA_NAMES)
    structural_idx = structural_shock_indices(model)
    state_path = Matrix{Float64}(dgp["state_path"])
    dgp_shocks = Matrix{Float64}(dgp["shocks"])
    state_scale = max.(vec(Statistics.std(state_path[:, 1:end-1], dims = 2; corrected = false)), 1e-5)
    x_cols = Vector{Vector{Float64}}()
    y_cols = Vector{Vector{Float64}}()
    y_sep_cols = Vector{Vector{Float64}}()
    attempts = 0

    while length(x_cols) < cfg.train_samples && attempts < max(50, 25 * cfg.train_samples)
        attempts += 1
        t = rand(rng, 1:size(dgp_shocks, 2))
        state = copy(state_path[:, t])
        if rand(rng) < 0.35
            state .+= 0.002 .* state_scale .* randn(rng, length(state))
        end
        shock = copy(dgp_shocks[:, t])
        shock[structural_idx] .+= 0.05 .* randn(rng, length(structural_idx))
        z = log(ELB_THETA_TRUE[1]) + 0.12 * randn(rng)
        theta = [clamp(exp(z), ELB_PRIOR_LOWER[1], ELB_PRIOR_UPPER[1])]

        pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx, state, shock, theta, sep_cfg)
        pred.ok || continue
        local y_rom
        try
            y_rom = rom_obs_std_nu(model, cache, obs_idx, state, shock, theta)
        catch
            continue
        end
        all(isfinite, pred.obs) || continue
        all(isfinite, y_rom) || continue
        push!(x_cols, vcat(state, shock, log.(theta)))
        push!(y_cols, pred.obs .- y_rom)
        push!(y_sep_cols, pred.obs)
    end

    length(x_cols) >= cfg.train_samples ||
        error("Could not collect enough ELB residual training samples; got $(length(x_cols)) of $(cfg.train_samples).")
    return hcat(x_cols...), hcat(y_cols...), hcat(y_sep_cols...)
end

function known_path_loglik_direct(model, cfg::ElbValidationConfig, dgp::Dict{String,Any},
                                  obs_idx::Vector{Int}, theta::Vector{Float64})
    base_params = Float64.(model.parameter_values)
    sep_cfg = base_stage_config(cfg)
    theta_idx = parameter_indices(model, ELB_THETA_NAMES)
    obs_data = Matrix{Float64}(dgp["obs_data"])
    obs_sigma = Vector{Float64}(dgp["obs_sigma"])
    state_path = Matrix{Float64}(dgp["state_path"])
    shocks = Matrix{Float64}(dgp["shocks"])
    ll = 0.0
    norm_const = sum(log.(2pi .* obs_sigma .^ 2))
    for t in 1:size(obs_data, 2)
        pred = direct_sep_predict_level(model, base_params, theta_idx, obs_idx,
                                        state_path[:, t], shocks[:, t], theta, sep_cfg)
        pred.ok || return -Inf
        resid = (obs_data[:, t] .- pred.obs) ./ obs_sigma
        ll += -0.5 * (sum(resid .^ 2) + norm_const)
    end
    return ll
end

function known_path_loglik_surrogate(model, cfg::ElbValidationConfig, dgp::Dict{String,Any},
                                     cache::RomCache, surrogate::RandomFeatureSurrogate,
                                     obs_idx::Vector{Int}, theta::Vector{Float64})
    obs_data = Matrix{Float64}(dgp["obs_data"])
    obs_sigma = Vector{Float64}(dgp["obs_sigma"])
    state_path = Matrix{Float64}(dgp["state_path"])
    shocks = Matrix{Float64}(dgp["shocks"])
    ll = 0.0
    norm_const = sum(log.(2pi .* obs_sigma .^ 2))
    for t in 1:size(obs_data, 2)
        state = state_path[:, t]
        shock = shocks[:, t]
        local y_rom
        try
            y_rom = rom_obs_std_nu(model, cache, obs_idx, state, shock, theta)
        catch
            return -Inf
        end
        y_hat = y_rom .+ predict_residual(surrogate, state, shock, theta)
        all(isfinite, y_hat) || return -Inf
        resid = (obs_data[:, t] .- y_hat) ./ obs_sigma
        ll += -0.5 * (sum(resid .^ 2) + norm_const)
    end
    return ll
end

function posterior_grid_summary(theta_grid::Vector{Float64}, logpost::Vector{Float64})
    finite = isfinite.(logpost)
    any(finite) || error("No finite grid log posterior values.")
    shifted = fill(0.0, length(logpost))
    shifted[finite] .= exp.(logpost[finite] .- maximum(logpost[finite]))
    weights = shifted ./ sum(shifted)
    cdf = cumsum(weights)
    q05 = theta_grid[findfirst(>=(0.05), cdf)]
    q95 = theta_grid[findfirst(>=(0.95), cdf)]
    mean_theta = sum(weights .* theta_grid)
    sd_theta = sqrt(sum(weights .* (theta_grid .- mean_theta) .^ 2))
    map_theta = theta_grid[argmax(logpost)]
    return Dict{String,Any}(
        "theta_grid" => theta_grid,
        "logpost" => logpost,
        "weights" => weights,
        "mean" => mean_theta,
        "sd" => sd_theta,
        "q05" => q05,
        "q95" => q95,
        "map" => map_theta,
    )
end

function run_posterior_grids(model, cfg::ElbValidationConfig, dgp::Dict{String,Any},
                             cache::RomCache, surrogate::RandomFeatureSurrogate,
                             obs_idx::Vector{Int})
    theta_grid = elb_grid(cfg)
    direct_logpost = similar(theta_grid)
    surrogate_logpost = similar(theta_grid)
    for i in eachindex(theta_grid)
        theta = [theta_grid[i]]
        lp = elb_prior_logpost_z(log(theta[1]))
        direct_ll = known_path_loglik_direct(model, cfg, dgp, obs_idx, theta)
        surrogate_ll = known_path_loglik_surrogate(model, cfg, dgp, cache, surrogate, obs_idx, theta)
        direct_logpost[i] = lp + direct_ll
        surrogate_logpost[i] = lp + surrogate_ll
    end
    direct = posterior_grid_summary(theta_grid, direct_logpost)
    surrogate_summary = posterior_grid_summary(theta_grid, surrogate_logpost)
    return Dict{String,Any}(
        "theta_grid" => theta_grid,
        "direct" => direct,
        "surrogate" => surrogate_summary,
        "objective" => "known_state_known_shock_measurement_error",
    )
end

function elb_acceptance(cfg::ElbValidationConfig, dgp::Dict{String,Any},
                        surrogate::RandomFeatureSurrogate, grid_payload::Dict{String,Any})
    direct = grid_payload["direct"]
    surrogate_summary = grid_payload["surrogate"]
    pooled_sd = max(direct["sd"], eps(Float64))
    mean_diff_sd = abs(surrogate_summary["mean"] - direct["mean"]) / pooled_sd
    interval_overlap = max(direct["q05"], surrogate_summary["q05"]) <= min(direct["q95"], surrogate_summary["q95"])
    cover_tol = max(1e-12, 1e-10 * ELB_THETA_TRUE[1])
    true_direct_cover = direct["q05"] <= ELB_THETA_TRUE[1] + cover_tol &&
                        ELB_THETA_TRUE[1] <= direct["q95"] + cover_tol
    true_surrogate_cover = surrogate_summary["q05"] <= ELB_THETA_TRUE[1] + cover_tol &&
                           ELB_THETA_TRUE[1] <= surrogate_summary["q95"] + cover_tol
    min_finite = min(length(grid_payload["theta_grid"]), 3)
    finite_grid_ok = count(isfinite, grid_payload["direct"]["logpost"]) >= min_finite &&
                     count(isfinite, grid_payload["surrogate"]["logpost"]) >= min_finite
    return Dict{String,Any}(
        "elb_share_ok" => dgp["elb_share"] >= cfg.min_elb_share,
        "max_valid_rrmse_obs" => maximum(surrogate.valid_rrmse_obs),
        "finite_grid_ok" => finite_grid_ok,
        "mean_diff_direct_sd" => mean_diff_sd,
        "posterior_mean_ok" => mean_diff_sd <= cfg.posterior_mean_tolerance_sd,
        "interval_overlap" => interval_overlap,
        "true_direct_cover" => true_direct_cover,
        "true_surrogate_cover" => true_surrogate_cover,
        "stage_pass" => dgp["elb_share"] >= cfg.min_elb_share &&
                        finite_grid_ok &&
                        isfinite(maximum(surrogate.valid_rrmse_obs)) &&
                        mean_diff_sd <= cfg.posterior_mean_tolerance_sd &&
                        interval_overlap &&
                        true_direct_cover &&
                        true_surrogate_cover,
    )
end

function write_elb_manifest(path::String, opts::ElbValidationOptions, cfg::ElbValidationConfig, model)
    mkpath(path)
    manifest = Dict{String,Any}(
        "run_id" => opts.run_id,
        "stage" => string(cfg.stage),
        "dry_run" => opts.dry_run,
        "model" => "Gali_2015_chapter_3_obc",
        "model_file" => joinpath(ELB_REPO_ROOT, "models", "Gali_2015_chapter_3_obc.jl"),
        "hard_nonlinearity" => "ELB max operator in R equation",
        "elb_binding_definition" => "shadow Taylor-rule rate below Rbar; the solved R path may be slightly above the bound when SEP is accepted with a loose smoke tolerance",
        "first_parameter_block" => string.(ELB_THETA_NAMES),
        "theta_true" => ELB_THETA_TRUE,
        "theta_baseline" => ELB_THETA_BASELINE,
        "prior_lower" => ELB_PRIOR_LOWER,
        "prior_upper" => ELB_PRIOR_UPPER,
        "prior_log_sd" => ELB_PRIOR_LOG_SD,
        "observables" => string.(ELB_OBSERVABLES),
        "target_definition" => "observable residual: direct SEP observable minus ROM1 observable",
        "rom_state_propagation" => "ROM1 state path; residual correction is observable-only in this first stage",
        "posterior_objective" => "known-state, known-shock measurement-error grid; inversion-filter validation is the next stage",
        "periods" => cfg.periods,
        "train_samples" => cfg.train_samples,
        "hidden_units" => cfg.hidden_units,
        "grid_size" => cfg.grid_size,
        "sep_horizon" => cfg.sep_horizon,
        "sep_maxit" => cfg.sep_maxit,
        "sep_tol" => cfg.sep_tol,
        "sep_accept_tol" => cfg.sep_accept_tol,
        "measurement_noise_share" => cfg.measurement_noise_share,
        "acceptance_criteria" => Dict(
            "min_elb_share" => cfg.min_elb_share,
            "posterior_mean_tolerance_direct_sd" => cfg.posterior_mean_tolerance_sd,
            "requires_direct_and_surrogate_90pct_interval_overlap" => true,
            "requires_true_value_in_both_90pct_intervals" => true,
        ),
        "scale_up_sequence" => [
            "stage=smoke: finite direct SEP and surrogate grid on a tiny ELB path",
            "stage=grid1: one-parameter std_nu posterior grid suitable for replication checks",
            "stage=paper1: longer one-parameter ELB posterior-grid table",
            "next: replace known shocks with ROM1/surrogate inversion-filter shocks",
            "next: add a second parameter only after the one-parameter ELB grid passes",
        ],
        "artifact_schema" => [
            "manifest.toml",
            "synthetic_dgp.jls",
            "surrogate_bundle.jls",
            "grid_payload.jls",
            "SUMMARY.md",
            "comparison_table.tex",
        ],
    )
    open(joinpath(path, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    return manifest
end

function fmt_elb(x)
    x isa Missing && return "missing"
    isfinite(Float64(x)) ? @sprintf("%.6g", Float64(x)) : string(x)
end

function write_elb_table(path::String, grid_payload::Dict{String,Any}, acceptance::Dict{String,Any})
    direct = grid_payload["direct"]
    surrogate_summary = grid_payload["surrogate"]
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & True & Direct SEP mean & ROM1-residual mean & Mean diff./direct sd \\\\")
        println(io, "\\midrule")
        println(io, "\$\\sigma_\\nu\$ & $(fmt_elb(ELB_THETA_TRUE[1])) & $(fmt_elb(direct["mean"])) & $(fmt_elb(surrogate_summary["mean"])) & $(fmt_elb(acceptance["mean_diff_direct_sd"])) \\\\")
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function write_elb_summary(path::String, cfg::ElbValidationConfig, dgp::Dict{String,Any},
                           surrogate::RandomFeatureSurrogate, grid_payload::Dict{String,Any},
                           acceptance::Dict{String,Any})
    direct = grid_payload["direct"]
    surrogate_summary = grid_payload["surrogate"]
    open(path, "w") do io
        println(io, "# Galí ELB ROM1-Residual Validation")
        println(io)
        println(io, "- Stage: `$(cfg.stage)`")
        println(io, "- Parameter block: `std_nu` only")
        println(io, "- Nonlinearity: ELB max operator in the Galí policy rule")
        println(io, "- Surrogate target: `direct SEP observable - ROM1 observable`")
        println(io, "- Objective: known-state, known-shock measurement-error posterior grid")
        println(io, "- SEP settings: horizon=$(cfg.sep_horizon), maxit=$(cfg.sep_maxit), accept_tol=$(cfg.sep_accept_tol)")
        println(io)
        println(io, "## DGP")
        println(io)
        println(io, "- Periods: $(dgp["periods"])")
        println(io, "- ELB binding share: $(fmt_elb(dgp["elb_share"]))")
        println(io, "- Minimum shadow policy rate: $(fmt_elb(minimum(dgp["shadow_rate_path"])))")
        println(io, "- Observation sigma: $(join(fmt_elb.(dgp["obs_sigma"]), ", "))")
        println(io)
        println(io, "## Surrogate Fit")
        println(io)
        println(io, "- Train RRMSE relative to observables: $(join(fmt_elb.(surrogate.train_rrmse_obs), ", "))")
        println(io, "- Validation RRMSE relative to observables: $(join(fmt_elb.(surrogate.valid_rrmse_obs), ", "))")
        println(io, "- Validation RRMSE relative to residuals: $(join(fmt_elb.(surrogate.valid_rrmse_resid), ", "))")
        println(io)
        println(io, "## Posterior Grid")
        println(io)
        println(io, "| Parameter | True | Direct mean | Direct 90% interval | Surrogate mean | Surrogate 90% interval |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        println(io, "| `std_nu` | $(fmt_elb(ELB_THETA_TRUE[1])) | $(fmt_elb(direct["mean"])) | [$(fmt_elb(direct["q05"])), $(fmt_elb(direct["q95"]))] | $(fmt_elb(surrogate_summary["mean"])) | [$(fmt_elb(surrogate_summary["q05"])), $(fmt_elb(surrogate_summary["q95"]))] |")
        println(io)
        println(io, "## Acceptance")
        println(io)
        for key in sort(collect(keys(acceptance)))
            println(io, "- `$key`: $(acceptance[key])")
        end
        println(io)
        println(io, "## Interpretation")
        println(io)
        println(io, "This is the first validation layer. Passing it shows that the ROM1-residual surrogate can recover the one-parameter ELB transition posterior when the same state and shock path is supplied to direct SEP and the surrogate. It does not yet validate the full inversion-filter posterior; that is the next staged exercise after this grid passes.")
    end
end

function run_elb_validation(opts::ElbValidationOptions)
    cfg = apply_elb_overrides(elb_stage_config(opts.stage), opts)
    out_dir = default_elb_out_dir(opts)
    model = ELB_MODEL
    manifest = write_elb_manifest(out_dir, opts, cfg, model)
    opts.dry_run && return Dict{String,Any}("out_dir" => out_dir, "manifest" => manifest)

    obs_idx = variable_indices(model, ELB_OBSERVABLES)
    base_params = Float64.(model.parameter_values)
    cache = build_rom_cache(model, 1; params = base_params, use_obc = true)

    println("Simulating Galí ELB DGP...")
    dgp = simulate_elb_dgp(model, cfg; seed = opts.seed)
    serialize(joinpath(out_dir, "synthetic_dgp.jls"), dgp)
    println("  ELB share: $(round(dgp["elb_share"], digits = 3))")
    MacroModelling.write_parameters_input!(model, base_params, verbose = false)

    println("Collecting SEP-ROM1 residual training data...")
    X, Y, Ysep = collect_elb_residual_training_data(model, cfg, cache, obs_idx, dgp; seed = opts.seed + 11)
    surrogate = train_random_feature_surrogate(X, Y, Ysep, cfg.hidden_units; seed = opts.seed + 17)
    serialize(joinpath(out_dir, "surrogate_bundle.jls"), Dict(
        "surrogate" => surrogate,
        "X" => X,
        "Y_residual" => Y,
        "Y_sep" => Ysep,
        "target_definition" => "direct SEP observable - ROM1 observable",
    ))
    println("  validation RRMSE(obs): $(join(round.(surrogate.valid_rrmse_obs, digits = 5), ", "))")

    println("Evaluating direct SEP and ROM1-residual posterior grids...")
    grid_payload = run_posterior_grids(model, cfg, dgp, cache, surrogate, obs_idx)
    serialize(joinpath(out_dir, "grid_payload.jls"), grid_payload)

    acceptance = elb_acceptance(cfg, dgp, surrogate, grid_payload)
    serialize(joinpath(out_dir, "acceptance_payload.jls"), acceptance)
    write_elb_table(joinpath(out_dir, "comparison_table.tex"), grid_payload, acceptance)
    write_elb_summary(joinpath(out_dir, "SUMMARY.md"), cfg, dgp, surrogate, grid_payload, acceptance)
    println("Wrote summary: $(joinpath(out_dir, "SUMMARY.md"))")
    return Dict{String,Any}(
        "out_dir" => out_dir,
        "manifest" => manifest,
        "dgp" => dgp,
        "surrogate" => surrogate,
        "grid" => grid_payload,
        "acceptance" => acceptance,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    opts = parse_elb_args(ARGS)
    result = run_elb_validation(opts)
    println("Artifacts: $(result["out_dir"])")
end
