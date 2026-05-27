#!/usr/bin/env julia
using MacroModelling
using Random
using Serialization
using Dates
using Statistics
using Distributions
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function parse_float_list(arg::AbstractString)
    if isempty(arg)
        return Float64[]
    end
    return [parse(Float64, strip(x)) for x in split(String(arg), ",")]
end

function parse_int_list(arg::AbstractString)
    if isempty(arg)
        return Int[]
    end
    return [parse(Int, strip(x)) for x in split(String(arg), ",")]
end

grid_points = parse_arg_int(ARGS, "--grid", 5)
theta_samples = parse_arg_int(ARGS, "--theta-samples", grid_points^3)
theta_max_attempts = parse_arg_int(ARGS, "--theta-max-attempts", theta_samples * 10)
samples_per_theta = parse_arg_int(ARGS, "--samples-per-theta", 0)
burn_in = parse_arg_int(ARGS, "--burn-in", 100)
sample_start = parse_arg_int(ARGS, "--sample-start", 47)
sample_length = parse_arg_int(ARGS, "--sample-length", 0)
irf_augment = "--irf-augment" in ARGS
irf_periods = parse_arg_int(ARGS, "--irf-periods", 20)
irf_shocks_arg = parse_arg_string(ARGS, "--irf-shocks", "all")
irf_sizes_arg = parse_arg_string(ARGS, "--irf-sizes", "")
irf_sizes_override = !isempty(irf_sizes_arg)
irf_sizes = parse_float_list(irf_sizes_override ? irf_sizes_arg : "0.5,2.0")
irf_accept_tol = parse_arg_float(ARGS, "--irf-accept-tol", NaN)
irf_accept_tol = isnan(irf_accept_tol) ? nothing : irf_accept_tol
stable_prefix = "--stable-prefix" in ARGS
stable_min_periods = parse_arg_int(ARGS, "--stable-min-periods", 0)
theta_attempts_per_theta = parse_arg_int(ARGS, "--theta-attempts-per-theta", stable_prefix ? 3 : 1)
seed_attempt_stride = parse_arg_int(ARGS, "--seed-attempt-stride", 10_000)
retry_on_early_failure = parse_arg_bool(ARGS, "--retry-on-early-failure", stable_prefix)
retry_shock_scale_backoff = parse_arg_float(ARGS, "--retry-shock-scale-backoff", 1.0)
allow_empty_dataset = "--allow-empty-dataset" in ARGS
min_total_samples = parse_arg_int(ARGS, "--min-total-samples", 1)
sep_horizon = parse_arg_int(ARGS, "--sep-horizon", 20)
sep_order = parse_arg_int(ARGS, "--sep-order", 1)
sep_nnodes = parse_arg_int(ARGS, "--sep-nnodes", 3)
sep_maxit = parse_arg_int(ARGS, "--sep-maxit", 80)
sep_tol = parse_arg_float(ARGS, "--sep-tol", 1e-5)
sep_linear_solver = parse_arg_symbol(ARGS, "--sep-linear-solver", :normal_equations)
sep_fallback_solver_arg = parse_arg_symbol(ARGS, "--sep-fallback-solver", :none)
sep_fallback_solver = sep_fallback_solver_arg == :none ? nothing : sep_fallback_solver_arg
sep_stall_iters = parse_arg_int(ARGS, "--sep-stall-iters", 25)
sep_stall_rel_tol = parse_arg_float(ARGS, "--sep-stall-rel-tol", 1e-4)
sep_stall_abs_tol = parse_arg_float(ARGS, "--sep-stall-abs-tol", 1e-10)
sep_line_search = parse_arg_bool(ARGS, "--sep-line-search", true)
sep_line_search_maxit = parse_arg_int(ARGS, "--sep-line-search-maxit", 6)
sep_line_search_factor = parse_arg_float(ARGS, "--sep-line-search-factor", 0.5)
sep_line_search_min_alpha = parse_arg_float(ARGS, "--sep-line-search-min-alpha", 1e-4)
sep_lm_lambda = parse_arg_float(ARGS, "--sep-lm-lambda", 1e-8)
sep_lm_lambda_scale = parse_arg_float(ARGS, "--sep-lm-lambda-scale", 10.0)
sep_lm_lambda_min = parse_arg_float(ARGS, "--sep-lm-lambda-min", 1e-12)
sep_lm_lambda_max = parse_arg_float(ARGS, "--sep-lm-lambda-max", 1e4)
sep_shock_scale = parse_arg_float(ARGS, "--sep-shock-scale", 1.0)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", NaN)
sep_accept_tol = isnan(sep_accept_tol) ? nothing : sep_accept_tol
sep_expectation_method = parse_arg_symbol(ARGS, "--sep-expectation-method", :gauss_hermite)
hmc_samples = parse_arg_int(ARGS, "--hmc-samples", 100)
hmc_warmup = parse_arg_int(ARGS, "--hmc-warmup", 50)
hmc_leapfrog_steps = parse_arg_int(ARGS, "--hmc-leapfrog-steps", 15)
hmc_step_size = parse_arg_float(ARGS, "--hmc-step-size", 0.1)
hmc_use_tempering = parse_arg_bool(ARGS, "--hmc-use-tempering", false)
hmc_verbose = parse_arg_bool(ARGS, "--hmc-verbose", true)
use_subdifferential = parse_arg_bool(ARGS, "--use-subdifferential", false)
subdiff_kink_tol = parse_arg_float(ARGS, "--subdiff-kink-tol", 1e-6)
subdiff_alpha_maxit = parse_arg_int(ARGS, "--subdiff-alpha-maxit", 20)
subdiff_alpha_tol = parse_arg_float(ARGS, "--subdiff-alpha-tol", 1e-3)
subdiff_verbose = parse_arg_bool(ARGS, "--subdiff-verbose", false)
seed0 = parse_arg_int(ARGS, "--seed", 42)
shock_scaling = parse_arg_symbol(ARGS, "--shock-scaling", :parameter)
shock_scale = parse_arg_float(ARGS, "--shock-scale", 0.25)
theta_sampling = parse_arg_symbol(ARGS, "--theta-sampling", :prior)
param_set = parse_arg_symbol(ARGS, "--param-set", :legacy_3params)
sep_sparse_tree = !("--no-sparse-tree" in ARGS)
rom_orders = parse_int_list(parse_arg_string(ARGS, "--rom-orders", "1,2"))
rom_mode = parse_arg_symbol(ARGS, "--rom-mode", :baseline)
output_dir_arg = parse_arg_string(ARGS, "--output-dir", "")
checkpoint_every = parse_arg_int(ARGS, "--checkpoint-every", 10)
resume = "--resume" in ARGS
timing = "--timing" in ARGS
force_obc = "--use-obc" in ARGS
force_zlb = "--use-zlb" in ARGS
force_no_obc = "--no-obc" in ARGS
cprobp_min = parse_arg_float(ARGS, "--cprobp-min", 0.5)
cprobp_max = parse_arg_float(ARGS, "--cprobp-max", 0.95)
cindp_min = parse_arg_float(ARGS, "--cindp-min", 0.01)
cindp_max = parse_arg_float(ARGS, "--cindp-max", 0.99)
curvp_min = parse_arg_float(ARGS, "--curvp-min", 25.0)
curvp_max = parse_arg_float(ARGS, "--curvp-max", 125.0)
if sum([force_obc, force_zlb, force_no_obc]) > 1
    error("Specify at most one of --use-obc, --use-zlb, or --no-obc.")
end
use_obc = force_obc || force_zlb
use_zlb = force_zlb
if force_no_obc
    use_obc = false
    use_zlb = false
end
theta_attempts_per_theta >= 1 || error("--theta-attempts-per-theta must be >= 1.")
seed_attempt_stride >= 1 || error("--seed-attempt-stride must be >= 1.")
(0.0 < retry_shock_scale_backoff <= 1.0) || error("--retry-shock-scale-backoff must be in (0, 1].")
min_total_samples >= 0 || error("--min-total-samples must be >= 0.")
rom_orders = sort(unique(rom_orders))
if any(order -> !(order in (1, 2)), rom_orders)
    error("Unsupported --rom-orders=$(rom_orders). Use 1,2 or leave empty.")
end
if !(rom_mode in (:baseline, :theta))
    error("Unknown --rom-mode=$rom_mode. Use :baseline or :theta.")
end
if !(sep_expectation_method in (:gauss_hermite, :hmc))
    error("--sep-expectation-method must be :gauss_hermite or :hmc (got $sep_expectation_method).")
end

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
sample_idx = sample_length > 0 ? (sample_start:(sample_start + sample_length - 1)) : (47:230)
T_obs = length(sample_idx)
if samples_per_theta <= 0
    samples_per_theta = T_obs
end

model_name = use_zlb ? "Smets_Wouters_2007_HLT_zlb" : (use_obc ? "Smets_Wouters_2007_HLT_obc" : "Smets_Wouters_2007_HLT")
hlt_model_file_and_symbol(model_name)  # fail fast on unsupported names
model = load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)
hlt_param_names = model.parameters
base_values = copy(model.parameter_values)

# ============================================================================
# PARAMETER SET CONFIGURATION
# ============================================================================
# Get parameter names from parameter_config.jl
theta_names = get_parameter_names(param_set)
theta_idx = indexin(theta_names, hlt_param_names)
@assert all(!isnothing, theta_idx) "Missing parameters $(theta_names[findall(isnothing, theta_idx)]) in $(model.model_name)."
theta_idx = Int.(theta_idx)
d_theta = length(theta_names)
theta_baseline = base_values[theta_idx]

# Get parameter bounds for sampling
param_bounds_dict = get_parameter_bounds(param_set)

println("=" ^ 80)
println("Parameter Set: $param_set")
println("Number of parameters: $d_theta")
println("Parameters: $theta_names")
println("=" ^ 80)
println("SEP expectation method: $sep_expectation_method")
if sep_expectation_method == :hmc
    println("HMC settings:")
    println("  - Samples: $hmc_samples (warmup: $hmc_warmup)")
    println("  - Leapfrog steps: $hmc_leapfrog_steps, step size: $hmc_step_size")
    println("  - Tempering: $hmc_use_tempering")
end
if use_subdifferential
    println("Subdifferential Newton enabled:")
    println("  - Kink tolerance: $subdiff_kink_tol")
    println("  - Alpha max iterations: $subdiff_alpha_maxit")
    println("  - Alpha tolerance: $subdiff_alpha_tol")
    println("  - Verbose: $subdiff_verbose")
end
println("=" ^ 80)

shock_names = model.exo
obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
structural_idx = findall(!, obc_mask)
irf_shocks = if irf_shocks_arg == "all"
    shock_names[structural_idx]
else
    Symbol.(split(irf_shocks_arg, ","))
end

function beta_ab_from_mu_sigma(mu::Float64, sigma::Float64)
    alpha = ((1 - mu) / sigma^2 - 1 / mu) * mu^2
    beta = alpha * (1 / mu - 1)
    return alpha, beta
end

"""
    lhs_to_bounds(lhs_sample::Matrix{Float64}, bounds_dict::Dict{Symbol, Tuple{Float64, Float64}},
                  param_names::Vector{Symbol})

Transform Latin hypercube samples from [0,1]^d to parameter bounds.

# Arguments
- `lhs_sample`: Matrix of LHS samples (d × n), values in [0, 1]
- `bounds_dict`: Dictionary mapping parameter names to (lb, ub) tuples
- `param_names`: Ordered list of parameter names

# Returns
- Matrix of samples in parameter space (d × n)
"""
function lhs_to_bounds(lhs_sample::Matrix{Float64},
                        bounds_dict::Dict{Symbol, Tuple{Float64, Float64}},
                        param_names::Vector{Symbol})
    d, n = size(lhs_sample)
    @assert d == length(param_names) "Dimension mismatch: LHS has $d dimensions, $(length(param_names)) parameters"

    transformed = similar(lhs_sample)

    for (i, name) in enumerate(param_names)
        lb, ub = bounds_dict[name]
        transformed[i, :] = lb .+ (ub - lb) .* lhs_sample[i, :]
    end

    return transformed
end

function draw_shocks(rng::AbstractRNG, model, total_periods::Int, shock_scaling::Symbol, shock_scale::Float64)
    shock_names = model.exo
    nshocks = length(shock_names)
    shocks = zeros(nshocks, total_periods)
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    if isempty(structural_idx)
        return shocks
    end
    sigmas = ones(length(structural_idx))
    if shock_scaling == :parameter
        for (i, idx) in enumerate(structural_idx)
            sigmas[i] = MacroModelling.sep_irf_shock_std(model, shock_names[idx])
        end
    end
    sigmas .*= shock_scale
    if length(structural_idx) == 1
        shocks[structural_idx[1], :] .= randn(rng, total_periods) .* sigmas[1]
    else
        shocks[structural_idx, :] .= Diagonal(sigmas) * randn(rng, length(structural_idx), total_periods)
    end
    return shocks
end

@inline function max_attempts_for_theta(theta_sampling::Symbol, theta_max_attempts::Int, theta_attempts_per_theta::Int)
    return theta_sampling == :prior ? theta_max_attempts : theta_attempts_per_theta
end

@inline function attempt_seed(seed0::Int, theta_i::Int, attempt::Int, seed_attempt_stride::Int)
    return seed0 + theta_i - 1 + (attempt - 1) * seed_attempt_stride
end

@inline function attempt_shock_scale(base_scale::Float64, attempt::Int, backoff::Float64)
    return base_scale * (backoff ^ (attempt - 1))
end

@inline function available_periods(sim, shocks)
    sim_periods = max(size(sim, 2) - 1, 0)
    shock_periods = ndims(shocks) == 2 ? size(shocks, 2) : 0
    return min(sim_periods, shock_periods)
end

@inline function retry_threshold_period(burn_in::Int, stable_min_periods::Int)
    return max(1, burn_in + max(stable_min_periods, 1))
end

function should_retry_theta(theta_sampling::Symbol, stable_prefix::Bool, retry_on_early_failure::Bool,
                           res_errorflag::Bool, failure_period::Int, T_available::Int, burn_in::Int,
                           stable_min_periods::Int, attempt::Int, max_attempts::Int)
    attempt < max_attempts || return false
    theta_sampling == :prior && return true
    retry_on_early_failure || return true
    if stable_prefix && stable_min_periods > 0 && T_available > 0 && T_available < stable_min_periods
        return true
    end
    if !res_errorflag
        return false
    end
    thresh = retry_threshold_period(burn_in, stable_min_periods)
    return failure_period <= thresh
end

rng = MersenneTwister(seed0)

state_idx = sort(unique(vcat(model.timings.past_not_future_and_mixed_idx,
                             model.timings.future_not_past_and_mixed_idx)))
state_names = model.var[state_idx]
obs_idx = indexin(observables, model.var)
@assert all(!isnothing, obs_idx) "Observable indices not found in $(model.model_name)."
obs_idx = Int.(obs_idx)

d_state = length(state_idx)
d_obs = length(obs_idx)
d_eps = length(model.exo)

output_dir = output_dir_arg == "" ?
    joinpath(@__DIR__, "..", "data", "hlt_sep_surrogate_dataset_$(Dates.format(now(), "yyyymmdd_HHMMSS"))") :
    output_dir_arg
checkpoint_path = joinpath(output_dir, "hlt_sep_surrogate_dataset_checkpoint.jls")

sample_theta = () -> zeros(0)
theta_grid = Vector{Vector{Float64}}()
cursor = 0
start_theta = 1
theta_times = Float64[]
theta_ids = Int[]
theta_seeds = Int[]
theta_attempts = Int[]
theta_success = Bool[]
theta_full_success = Bool[]
theta_stable_periods = Int[]
failure_periods = Int[]
sample_full_success = Bool[]
sep_residuals = Float64[]
X = zeros(0, 0)
Y = zeros(0, 0)
Y_rom1 = zeros(0, 0)
Y_rom2 = zeros(0, 0)

function save_checkpoint(path::String;
                         X, Y, Y_rom1, Y_rom2, theta_ids, theta_grid, theta_seeds, theta_attempts,
                         theta_success, theta_full_success, theta_stable_periods, failure_periods, cursor, theta_last,
                         sample_full_success, sep_residuals=nothing,
                         rng, theta_times, settings)
    d = Dict(
        "X" => X,
        "Y" => Y,
        "Y_rom1" => Y_rom1,
        "Y_rom2" => Y_rom2,
        "theta_ids" => theta_ids,
        "theta_grid" => theta_grid,
        "theta_seeds" => theta_seeds,
        "theta_attempts" => theta_attempts,
        "theta_success" => theta_success,
        "theta_full_success" => theta_full_success,
        "theta_stable_periods" => theta_stable_periods,
        "theta_failure_periods" => failure_periods,
        "sample_full_success" => sample_full_success,
        "cursor" => cursor,
        "theta_last" => theta_last,
        "rng" => rng,
        "theta_times" => theta_times,
        "settings" => settings,
    )
    # RISK-1b: include SEP residuals when available
    if sep_residuals !== nothing
        d["sep_residuals"] = sep_residuals
    end
    serialize(path, d)
end

settings = Dict{String, Any}()

if resume
    if output_dir_arg == ""
        error("Resume requested but --output-dir was not provided.")
    end
    if !isfile(checkpoint_path)
        error("Checkpoint not found at $checkpoint_path")
    end
    chk = deserialize(checkpoint_path)
    settings = chk["settings"]
    if haskey(settings, "use_obc") && settings["use_obc"] != use_obc
        error("Checkpoint uses use_obc=$(settings["use_obc"]); rerun with --use-obc to match.")
    end

    # Load param_set from checkpoint (with backward compatibility)
    param_set = get(settings, "param_set", :legacy_3params)
    theta_names = get(settings, "theta_names", [:cprobp, :cindp, :curvp])

    grid_points = settings["grid_points"]
    theta_samples = settings["theta_samples"]
    theta_max_attempts = settings["theta_max_attempts"]
    theta_attempts_per_theta = get(settings, "theta_attempts_per_theta", theta_attempts_per_theta)
    seed_attempt_stride = get(settings, "seed_attempt_stride", seed_attempt_stride)
    retry_on_early_failure = get(settings, "retry_on_early_failure", retry_on_early_failure)
    retry_shock_scale_backoff = get(settings, "retry_shock_scale_backoff", retry_shock_scale_backoff)
    allow_empty_dataset = get(settings, "allow_empty_dataset", allow_empty_dataset)
    min_total_samples = get(settings, "min_total_samples", min_total_samples)
    samples_per_theta = settings["samples_per_theta"]
    burn_in = settings["burn_in"]
    sep_horizon = settings["sep_horizon"]
    sep_order = settings["sep_order"]
    sep_nnodes = settings["sep_nnodes"]
    sep_maxit = settings["sep_maxit"]
    sep_tol = settings["sep_tol"]
    sep_sparse_tree = settings["sep_sparse_tree"]
    sep_linear_solver = get(settings, "sep_linear_solver", sep_linear_solver)
    sep_fallback_solver = get(settings, "sep_fallback_solver", sep_fallback_solver)
    sep_stall_iters = get(settings, "sep_stall_iters", sep_stall_iters)
    sep_stall_rel_tol = get(settings, "sep_stall_rel_tol", sep_stall_rel_tol)
    sep_stall_abs_tol = get(settings, "sep_stall_abs_tol", sep_stall_abs_tol)
    sep_line_search = get(settings, "sep_line_search", sep_line_search)
    sep_line_search_maxit = get(settings, "sep_line_search_maxit", sep_line_search_maxit)
    sep_line_search_factor = get(settings, "sep_line_search_factor", sep_line_search_factor)
    sep_line_search_min_alpha = get(settings, "sep_line_search_min_alpha", sep_line_search_min_alpha)
    sep_lm_lambda = get(settings, "sep_lm_lambda", sep_lm_lambda)
    sep_lm_lambda_scale = get(settings, "sep_lm_lambda_scale", sep_lm_lambda_scale)
    sep_lm_lambda_min = get(settings, "sep_lm_lambda_min", sep_lm_lambda_min)
    sep_lm_lambda_max = get(settings, "sep_lm_lambda_max", sep_lm_lambda_max)
    sep_shock_scale = get(settings, "sep_shock_scale", sep_shock_scale)
    sep_accept_tol = get(settings, "sep_accept_tol", sep_accept_tol)
    shock_scaling = settings["shock_scaling"]
    shock_scale = settings["shock_scale"]
    seed0 = settings["seed0"]
    theta_sampling = settings["theta_sampling"]
    cprobp_min = get(settings, "cprobp_min", cprobp_min)
    cprobp_max = get(settings, "cprobp_max", cprobp_max)
    cindp_min = get(settings, "cindp_min", cindp_min)
    cindp_max = get(settings, "cindp_max", cindp_max)
    curvp_min = get(settings, "curvp_min", curvp_min)
    curvp_max = get(settings, "curvp_max", curvp_max)
    rom_orders = get(settings, "rom_orders", rom_orders)
    rom_mode = get(settings, "rom_mode", rom_mode)
    sample_start = get(settings, "sample_start", sample_start)
    sample_length = get(settings, "sample_length", sample_length)
    irf_augment = get(settings, "irf_augment", false)
    irf_periods = get(settings, "irf_periods", irf_periods)
    irf_shocks = Symbol.(get(settings, "irf_shocks", string.(irf_shocks)))
    if !irf_sizes_override
        irf_sizes = Float64.(get(settings, "irf_sizes", irf_sizes))
    end
    irf_accept_tol = isnothing(irf_accept_tol) ? get(settings, "irf_accept_tol", irf_accept_tol) : irf_accept_tol
    stable_prefix = get(settings, "stable_prefix", stable_prefix)
    stable_min_periods = get(settings, "stable_min_periods", stable_min_periods)
    sample_idx = sample_length > 0 ? (sample_start:(sample_start + sample_length - 1)) : (47:230)
    T_obs = length(sample_idx)

    X = chk["X"]
    Y = chk["Y"]
    Y_rom1 = get(chk, "Y_rom1", zeros(0, 0))
    Y_rom2 = get(chk, "Y_rom2", zeros(0, 0))
    theta_ids = chk["theta_ids"]
    theta_grid = chk["theta_grid"]
    theta_seeds = chk["theta_seeds"]
    theta_attempts = chk["theta_attempts"]
    theta_success = chk["theta_success"]
    theta_full_success = get(chk, "theta_full_success", copy(theta_success))
    theta_stable_periods = get(chk, "theta_stable_periods", zeros(Int, length(theta_grid)))
    failure_periods = chk["theta_failure_periods"]
    sample_full_success = get(chk, "sample_full_success", falses(size(X, 2)))
    sep_residuals = get(chk, "sep_residuals", fill(NaN, size(X, 2)))
    cursor = chk["cursor"]
    theta_times = get(chk, "theta_times", zeros(Float64, length(theta_grid)))
    rng = chk["rng"]
    start_theta = chk["theta_last"] + 1
    if (1 in rom_orders) && size(Y_rom1, 2) == 0
        error("Checkpoint missing Y_rom1. Rerun without --resume for ROM residual datasets.")
    end
    if (2 in rom_orders) && size(Y_rom2, 2) == 0
        error("Checkpoint missing Y_rom2. Rerun without --resume for ROM residual datasets.")
    end
    if theta_sampling == :prior
        if param_set == :legacy_3params
            # Legacy 3-parameter prior sampling
            αp, βp = beta_ab_from_mu_sigma(0.5, 0.10)
            αi, βi = beta_ab_from_mu_sigma(0.5, 0.15)
            prior_cprobp = Distributions.truncated(Distributions.Beta(αp, βp), 0.5, 0.95)
            prior_cindp = Distributions.truncated(Distributions.Beta(αi, βi), 0.01, 0.99)
            prior_curvp = Distributions.Normal(75.0, 25.0)
            sample_theta = () -> [rand(rng, prior_cprobp), rand(rng, prior_cindp), rand(rng, prior_curvp)]
        else
            # Use parameter_config.jl priors for multi-parameter sets
            priors_dict = get_parameter_priors(param_set)
            sample_theta = () -> [rand(rng, priors_dict[name]) for name in theta_names]
        end
    end
else
    mkpath(output_dir)
    if theta_sampling == :grid
        # Grid sampling. Legacy 3-parameter mode keeps the historical CLI
        # bounds; all configured multi-parameter sets use parameter_config.jl.
        grid_axis(minv, maxv, n) = n == 1 ? [0.5 * (minv + maxv)] : collect(range(minv, maxv, length = n))
        if param_set == :legacy_3params
            cprobp_grid = grid_axis(cprobp_min, cprobp_max, grid_points)
            cindp_grid = grid_axis(cindp_min, cindp_max, grid_points)
            curvp_grid = grid_axis(curvp_min, curvp_max, grid_points)
            for cprobp in cprobp_grid, cindp in cindp_grid, curvp in curvp_grid
                push!(theta_grid, [cprobp, cindp, curvp])
            end
        else
            axes = [grid_axis(param_bounds_dict[name]..., grid_points) for name in theta_names]
            for tup in Iterators.product(axes...)
                push!(theta_grid, Float64[x for x in tup])
            end
        end

    elseif theta_sampling == :lhs
        # Latin hypercube sampling (recommended for 18+ parameters)
        using LatinHypercubeSampling

        println("Generating LHS samples...")
        println("  Parameters: $d_theta")
        println("  Samples: $theta_samples")

        # Generate LHS plan in [0, 1]^d
        lhs_plan = randomLHC(theta_samples, d_theta)  # Returns d × n matrix
        lhs_scaled = scaleLHC(lhs_plan, [(0.0, 1.0) for _ in 1:d_theta])

        # Transform to parameter bounds
        theta_matrix = lhs_to_bounds(Matrix(lhs_scaled'), param_bounds_dict, theta_names)

        # Convert to vector of vectors for compatibility
        for i in 1:theta_samples
            push!(theta_grid, theta_matrix[:, i])
        end

        println("✅ Generated $theta_samples LHS samples")

        # Define sample_theta as a safety fallback (should not be called for LHS)
        sample_theta = () -> error("sample_theta() called in LHS mode - this should not happen")

    else  # :prior sampling
        # Prior sampling (original mode)
        if param_set == :legacy_3params
            αp, βp = beta_ab_from_mu_sigma(0.5, 0.10)
            αi, βi = beta_ab_from_mu_sigma(0.5, 0.15)
            prior_cprobp = Distributions.truncated(Distributions.Beta(αp, βp), 0.5, 0.95)
            prior_cindp = Distributions.truncated(Distributions.Beta(αi, βi), 0.01, 0.99)
            prior_curvp = Distributions.Normal(75.0, 25.0)
            sample_theta = () -> [rand(rng, prior_cprobp), rand(rng, prior_cindp), rand(rng, prior_curvp)]
        else
            # Use parameter_config.jl priors
            priors_dict = get_parameter_priors(param_set)
            sample_theta = () -> [rand(rng, priors_dict[name]) for name in theta_names]
        end
        theta_grid = [fill(NaN, d_theta) for _ in 1:theta_samples]
    end

    theta_target = length(theta_grid)
    irf_aug_samples = irf_augment ? irf_periods * (1 + length(irf_shocks) * length(irf_sizes)) : 0
    total_samples = theta_target * samples_per_theta + irf_aug_samples
    X = zeros(d_state + d_eps + d_theta, total_samples)
    Y = zeros(d_obs + d_state, total_samples)
    if 1 in rom_orders
        Y_rom1 = zeros(d_obs + d_state, total_samples)
    end
    if 2 in rom_orders
        Y_rom2 = zeros(d_obs + d_state, total_samples)
    end
    theta_ids = zeros(Int, total_samples)
    theta_seeds = zeros(Int, theta_target)
    theta_attempts = zeros(Int, theta_target)
    theta_success = falses(theta_target)
    theta_full_success = falses(theta_target)
    theta_stable_periods = zeros(Int, theta_target)
    failure_periods = fill(0, theta_target)
    theta_times = zeros(Float64, theta_target)
    sample_full_success = falses(size(X, 2))
    # RISK-1b: Store per-sample SEP solver residual for quality-weighted training
    sep_residuals = fill(NaN, total_samples)

    settings = Dict(
        "model_name" => string(model.model_name),
        "use_obc" => use_obc,
        "param_set" => param_set,
        "theta_names" => theta_names,
        "grid_points" => grid_points,
        "theta_sampling" => theta_sampling,
        "theta_samples" => theta_samples,
        "theta_max_attempts" => theta_max_attempts,
        "theta_attempts_per_theta" => theta_attempts_per_theta,
        "seed_attempt_stride" => seed_attempt_stride,
        "retry_on_early_failure" => retry_on_early_failure,
        "retry_shock_scale_backoff" => retry_shock_scale_backoff,
        "samples_per_theta" => samples_per_theta,
        "allow_empty_dataset" => allow_empty_dataset,
        "min_total_samples" => min_total_samples,
        "burn_in" => burn_in,
        "sample_start" => sample_start,
        "sample_length" => sample_length,
        "sep_horizon" => sep_horizon,
        "sep_order" => sep_order,
        "sep_nnodes" => sep_nnodes,
        "sep_maxit" => sep_maxit,
        "sep_tol" => sep_tol,
        "sep_sparse_tree" => sep_sparse_tree,
        "sep_linear_solver" => sep_linear_solver,
        "sep_fallback_solver" => sep_fallback_solver,
        "sep_stall_iters" => sep_stall_iters,
        "sep_stall_rel_tol" => sep_stall_rel_tol,
        "sep_stall_abs_tol" => sep_stall_abs_tol,
        "sep_line_search" => sep_line_search,
        "sep_line_search_maxit" => sep_line_search_maxit,
        "sep_line_search_factor" => sep_line_search_factor,
        "sep_line_search_min_alpha" => sep_line_search_min_alpha,
        "sep_lm_lambda" => sep_lm_lambda,
        "sep_lm_lambda_scale" => sep_lm_lambda_scale,
        "sep_lm_lambda_min" => sep_lm_lambda_min,
        "sep_lm_lambda_max" => sep_lm_lambda_max,
        "sep_shock_scale" => sep_shock_scale,
        "sep_accept_tol" => sep_accept_tol,
        "shock_scaling" => shock_scaling,
        "shock_scale" => shock_scale,
        "seed0" => seed0,
        "rom_orders" => rom_orders,
        "rom_mode" => rom_mode,
        "irf_augment" => irf_augment,
        "irf_periods" => irf_periods,
        "irf_shocks" => string.(irf_shocks),
        "irf_sizes" => irf_sizes,
        "irf_accept_tol" => irf_accept_tol,
        "stable_prefix" => stable_prefix,
        "stable_min_periods" => stable_min_periods,
        "cprobp_min" => cprobp_min,
        "cprobp_max" => cprobp_max,
        "cindp_min" => cindp_min,
        "cindp_max" => cindp_max,
        "curvp_min" => curvp_min,
        "curvp_max" => curvp_max,
    )
end

if isnothing(irf_accept_tol)
    irf_accept_tol = sep_accept_tol
end

theta_target = length(theta_grid)
total_samples = size(X, 2)

total_start = timing ? time() : 0.0
run_start = time()

println("HLT SEP surrogate dataset generation")
println("Model: $(model.model_name) (use_obc=$use_obc)")
println("Theta sampling: $theta_sampling (theta count = $(length(theta_grid)))")
println("Sample length: $T_obs, samples_per_theta: $samples_per_theta, burn-in: $burn_in")
println("Shock scaling: $shock_scaling (scale=$(shock_scale))")
println("SEP settings: horizon=$sep_horizon order=$sep_order nnodes=$sep_nnodes tol=$sep_tol maxit=$sep_maxit sparse_tree=$sep_sparse_tree")
println("SEP solver: line_search=$sep_line_search maxit=$sep_line_search_maxit factor=$sep_line_search_factor min_alpha=$sep_line_search_min_alpha lm_lambda=$sep_lm_lambda")
if theta_sampling != :prior || stable_prefix
    println("Retry policy: attempts_per_theta=$theta_attempts_per_theta seed_stride=$seed_attempt_stride retry_on_early_failure=$retry_on_early_failure shock_backoff=$retry_shock_scale_backoff")
end
if stable_prefix
    println("Stable-prefix mode: min_periods=$stable_min_periods (empty datasets error unless --allow-empty-dataset)")
end
if !isempty(rom_orders)
    println("ROM residual outputs: orders=$(rom_orders) mode=$rom_mode")
end
if resume
    println("Resuming from checkpoint: $checkpoint_path (next theta index = $start_theta)")
end

rom_cache_baseline = Dict{Int, RomCache}()
if !isempty(rom_orders)
    for order in rom_orders
        rom_cache_baseline[order] = build_rom_cache(model, order;
                                                    params = base_values,
                                                    use_obc = use_obc)
    end
end

for theta_i in start_theta:theta_target
    theta_start = timing ? time() : 0.0
    attempt = 0
    max_attempts_this_theta = max_attempts_for_theta(theta_sampling, theta_max_attempts, theta_attempts_per_theta)
    last_T_available = 0
    while attempt < max_attempts_this_theta
        attempt += 1
        theta = (theta_sampling == :grid || theta_sampling == :lhs) ? theta_grid[theta_i] : sample_theta()

        # DEBUG: Check theta dimensions
        if length(theta) != d_theta
            @error "Theta dimension mismatch" theta_i theta_sampling length(theta) d_theta theta_grid_size=length(theta_grid)
            error("Theta has wrong dimensions: expected $d_theta, got $(length(theta))")
        end

        params = copy(base_values)
        params[theta_idx] = theta
        MacroModelling.write_parameters_input!(model, params, verbose = false)

        seed = attempt_seed(seed0, theta_i, attempt, seed_attempt_stride)
        theta_seeds[theta_i] = seed
        total_periods = T_obs + burn_in
        shock_scale_eff = attempt_shock_scale(shock_scale, attempt, retry_shock_scale_backoff)
        shocks_override = shock_scale_eff == 1.0 ? nothing :
            draw_shocks(MersenneTwister(seed), model, total_periods, shock_scaling, shock_scale_eff)

        res = nothing
        try
            res = MacroModelling.simulate_sep_extended_path(
                model;
                periods = T_obs,
                burn_in = burn_in,
                sep_horizon = sep_horizon,
                sep_order = sep_order,
                sep_nnodes = sep_nnodes,
                sep_maxit = sep_maxit,
                sep_tol = sep_tol,
                sep_sparse_tree = sep_sparse_tree,
                sep_linear_solver = sep_linear_solver,
                sep_fallback_solver = sep_fallback_solver,
                sep_stall_iters = sep_stall_iters,
                sep_stall_rel_tol = sep_stall_rel_tol,
                sep_stall_abs_tol = sep_stall_abs_tol,
                sep_line_search = sep_line_search,
                sep_line_search_maxit = sep_line_search_maxit,
                sep_line_search_factor = sep_line_search_factor,
                sep_line_search_min_alpha = sep_line_search_min_alpha,
                sep_lm_lambda = sep_lm_lambda,
            sep_lm_lambda_scale = sep_lm_lambda_scale,
            sep_lm_lambda_min = sep_lm_lambda_min,
            sep_lm_lambda_max = sep_lm_lambda_max,
            sep_shock_scale = sep_shock_scale,
            sep_accept_tol = sep_accept_tol,
                sep_expectation_method = sep_expectation_method,
                hmc_samples = hmc_samples,
                hmc_warmup = hmc_warmup,
                hmc_leapfrog_steps = hmc_leapfrog_steps,
                hmc_step_size = hmc_step_size,
                hmc_use_tempering = hmc_use_tempering,
                hmc_verbose = hmc_verbose,
                use_subdifferential = use_subdifferential,
                subdiff_kink_tol = subdiff_kink_tol,
                subdiff_alpha_maxit = subdiff_alpha_maxit,
                subdiff_alpha_tol = subdiff_alpha_tol,
                subdiff_verbose = subdiff_verbose,
                shock_scaling = shock_scaling,
                shocks = shocks_override,
                random_seed = seed,
                silent = true,
            )
        catch e
            if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                failure_periods[theta_i] = 1
                if should_retry_theta(theta_sampling, stable_prefix, retry_on_early_failure,
                                      true, 1, 0, burn_in, stable_min_periods, attempt, max_attempts_this_theta)
                    println("SEP singular for theta index $theta_i (attempt $attempt/$max_attempts_this_theta); retrying.")
                else
                    println("SEP solve failed (singular) for theta index $theta_i; skipping.")
                end
                continue
            else
                rethrow()
            end
        end

        failure_period = res.failure_period === nothing ? 0 : Int(res.failure_period)
        failure_periods[theta_i] = res.errorflag ? failure_period : 0

        if theta_sampling == :prior
            theta_grid[theta_i] = theta
        end
        theta_full_success[theta_i] = !res.errorflag
        theta_attempts[theta_i] = attempt

        sim = Array(res.simulation)
        shocks = res.shocks
        # RISK-1b: Extract per-period SEP solver residuals (if available from updated simulate_sep_extended_path)
        res_sep_errors = hasproperty(res, :sep_errors) ? res.sep_errors : nothing
        T_available = available_periods(sim, shocks)
        last_T_available = T_available

        if res.errorflag && !stable_prefix
            if should_retry_theta(theta_sampling, stable_prefix, retry_on_early_failure,
                                  true, failure_period, T_available, burn_in, stable_min_periods, attempt, max_attempts_this_theta)
                println("SEP failed for theta index $theta_i at period $failure_period (attempt $attempt/$max_attempts_this_theta); retrying.")
            end
            continue
        end

        if T_available <= 0
            if should_retry_theta(theta_sampling, stable_prefix, retry_on_early_failure,
                                  res.errorflag, failure_period, T_available, burn_in, stable_min_periods, attempt, max_attempts_this_theta)
                println("No usable SEP periods for theta index $theta_i (attempt $attempt/$max_attempts_this_theta; failure_period=$failure_period); retrying.")
            end
            continue
        end
        if stable_min_periods > 0 && T_available < stable_min_periods
            if should_retry_theta(theta_sampling, stable_prefix, retry_on_early_failure,
                                  res.errorflag, failure_period, T_available, burn_in, stable_min_periods, attempt, max_attempts_this_theta)
                println("Stable prefix too short for theta index $theta_i: $T_available < $stable_min_periods (attempt $attempt/$max_attempts_this_theta, failure_period=$failure_period); retrying.")
            end
            continue
        end

        theta_success[theta_i] = true
        theta_stable_periods[theta_i] = T_available
        if res.errorflag && stable_prefix
            println("Accepting partial SEP prefix for theta index $theta_i: stable_periods=$T_available, failure_period=$failure_period (attempt $attempt/$max_attempts_this_theta).")
        elseif attempt > 1
            println("Recovered theta index $theta_i on attempt $attempt/$max_attempts_this_theta.")
        end

        rom_cache_theta = rom_cache_baseline
        if !isempty(rom_orders) && rom_mode == :theta
            rom_cache_theta = Dict{Int, RomCache}()
            for order in rom_orders
                rom_cache_theta[order] = build_rom_cache(model, order;
                                                        params = params,
                                                        use_obc = use_obc)
            end
        end

        rom_next = Dict{Int, Matrix{Float64}}()
        if !isempty(rom_orders)
            for order in rom_orders
                cache = rom_cache_theta[order]
                rom_mat = zeros(cache.nvars, T_available)
                for t in 1:T_available
                    rom_mat[:, t] = rom_step_full(cache, sim[:, t], shocks[:, t])
                end
                rom_next[order] = rom_mat
            end
        end

        sample_idx_local = 1:T_available
        if samples_per_theta < T_available
            sample_idx_local = rand(rng, 1:T_available, samples_per_theta)
        end

        for t in sample_idx_local
            global cursor
            cursor += 1
            X[:, cursor] = vcat(sim[state_idx, t], shocks[:, t], theta)
            Y[:, cursor] = vcat(sim[obs_idx, t + 1], sim[state_idx, t + 1])
            if 1 in rom_orders
                Y_rom1[:, cursor] = vcat(rom_next[1][obs_idx, t], rom_next[1][state_idx, t])
            end
            if 2 in rom_orders
                Y_rom2[:, cursor] = vcat(rom_next[2][obs_idx, t], rom_next[2][state_idx, t])
            end
            theta_ids[cursor] = theta_i
            sample_full_success[cursor] = theta_full_success[theta_i]
            # RISK-1b: Store per-sample SEP residual (NaN if not available)
            if res_sep_errors !== nothing && t <= length(res_sep_errors)
                sep_residuals[cursor] = res_sep_errors[t]
            end
        end

        break
    end
    if timing
        theta_times[theta_i] = time() - theta_start
    end

    stop_early = false
    if !theta_success[theta_i]
        theta_attempts[theta_i] = attempt
        fp = failure_periods[theta_i]
        println("SEP failed for theta index $theta_i after $attempt attempt(s). failure_period=$fp, stable_periods=$last_T_available")
        if theta_sampling == :prior
            stop_early = true
        end
    end

    if checkpoint_every > 0 && (theta_i % checkpoint_every == 0 || stop_early)
        save_checkpoint(checkpoint_path;
                        X = X,
                        Y = Y,
                        Y_rom1 = Y_rom1,
                        Y_rom2 = Y_rom2,
                        theta_ids = theta_ids,
                        theta_grid = theta_grid,
                        theta_seeds = theta_seeds,
                        theta_attempts = theta_attempts,
                        theta_success = theta_success,
                        theta_full_success = theta_full_success,
                        theta_stable_periods = theta_stable_periods,
                        failure_periods = failure_periods,
                        sample_full_success = sample_full_success,
                        sep_residuals = sep_residuals,
                        cursor = cursor,
                        theta_last = theta_i,
                        rng = rng,
                        theta_times = theta_times,
                        settings = settings)
        println("Saved checkpoint: $checkpoint_path")
    end

    if theta_i % 10 == 0 || theta_i == length(theta_grid)
        elapsed = time() - run_start
        done = theta_i - start_theta + 1
        avg_per = done > 0 ? elapsed / done : 0.0
        remaining = max(theta_target - theta_i, 0)
        eta_sec = remaining * avg_per
        eta = @sprintf("%02d:%02d:%02d",
                       floor(Int, eta_sec / 3600),
                       floor(Int, (eta_sec % 3600) / 60),
                       floor(Int, eta_sec % 60))
        println("Processed theta $theta_i / $(length(theta_grid)) (ETA $eta)")
    end

    if stop_early
        break
    end
end

if irf_augment
    if cursor > 0 && any(theta_ids[1:cursor] .== 0)
        println("IRF augmentation already present; skipping.")
        irf_augment = false
    else
        println("Adding IRF augmentation samples (baseline theta).")
    end
end

if irf_augment
    MacroModelling.write_parameters_input!(model, base_values, verbose = false)
    irf_shock_idx = Int[]
    for shock in irf_shocks
        pos = findfirst(==(shock), shock_names)
        pos === nothing && error("IRF shock $shock not found in model.")
        if !(pos in structural_idx)
            error("IRF shock $shock is not structural (OBC shocks not supported here).")
        end
        push!(irf_shock_idx, pos)
    end

    function append_irf_samples!(sim::AbstractMatrix, shocks::AbstractMatrix)
        @assert size(sim, 2) >= irf_periods + 1
        for t in 1:irf_periods
            global cursor
            cursor += 1
            X[:, cursor] = vcat(sim[state_idx, t], shocks[:, t], theta_baseline)
            Y[:, cursor] = vcat(sim[obs_idx, t + 1], sim[state_idx, t + 1])
            if 1 in rom_orders
                Y_rom1[:, cursor] = rom_output_from_full(rom_cache_baseline[1], sim[:, t], shocks[:, t], obs_idx, state_idx)
            end
            if 2 in rom_orders
                Y_rom2[:, cursor] = rom_output_from_full(rom_cache_baseline[2], sim[:, t], shocks[:, t], obs_idx, state_idx)
            end
            theta_ids[cursor] = 0
        end
    end

    zero_shocks = zeros(length(shock_names), irf_periods)
    base_res = simulate_sep_extended_path(
        model;
        periods = irf_periods,
        burn_in = 0,
        shocks = zero_shocks,
        sep_horizon = sep_horizon,
        sep_order = sep_order,
        sep_nnodes = sep_nnodes,
        sep_maxit = sep_maxit,
        sep_tol = sep_tol,
        sep_sparse_tree = sep_sparse_tree,
        sep_linear_solver = sep_linear_solver,
        sep_fallback_solver = sep_fallback_solver,
        sep_stall_iters = sep_stall_iters,
        sep_stall_rel_tol = sep_stall_rel_tol,
        sep_stall_abs_tol = sep_stall_abs_tol,
        sep_line_search = sep_line_search,
        sep_line_search_maxit = sep_line_search_maxit,
        sep_line_search_factor = sep_line_search_factor,
        sep_line_search_min_alpha = sep_line_search_min_alpha,
        sep_lm_lambda = sep_lm_lambda,
        sep_lm_lambda_scale = sep_lm_lambda_scale,
        sep_lm_lambda_min = sep_lm_lambda_min,
        sep_lm_lambda_max = sep_lm_lambda_max,
        sep_accept_tol = irf_accept_tol,
        sep_shock_scale = sep_shock_scale,
        sep_expectation_method = sep_expectation_method,
        hmc_samples = hmc_samples,
        hmc_warmup = hmc_warmup,
        hmc_leapfrog_steps = hmc_leapfrog_steps,
        hmc_step_size = hmc_step_size,
        hmc_use_tempering = hmc_use_tempering,
        hmc_verbose = hmc_verbose,
        use_subdifferential = use_subdifferential,
        subdiff_alpha_min = subdiff_alpha_min,
        subdiff_alpha_max = subdiff_alpha_max,
        subdiff_max_alpha_trials = subdiff_max_alpha_trials,
        subdiff_clarke_epsilon = subdiff_clarke_epsilon,
        shock_scaling = shock_scaling,
        silent = true,
    )
    if base_res.errorflag
        error("IRF baseline SEP failed at period $(base_res.failure_period).")
    end
    append_irf_samples!(Array(base_res.simulation), base_res.shocks)

    for (shock_idx, shock) in zip(irf_shock_idx, irf_shocks)
        for size in irf_sizes
            shock_value = MacroModelling.sep_irf_shock_scale(model, shock, size * shock_scale;
                                                             shock_scaling = shock_scaling,
                                                             negative_shock = false)
            shocks_mat = zeros(length(shock_names), irf_periods)
            shocks_mat[shock_idx, 1] = shock_value
            res = simulate_sep_extended_path(
                model;
                periods = irf_periods,
                burn_in = 0,
                shocks = shocks_mat,
                sep_horizon = sep_horizon,
                sep_order = sep_order,
                sep_nnodes = sep_nnodes,
                sep_maxit = sep_maxit,
                sep_tol = sep_tol,
                sep_sparse_tree = sep_sparse_tree,
                sep_linear_solver = sep_linear_solver,
                sep_fallback_solver = sep_fallback_solver,
                sep_stall_iters = sep_stall_iters,
                sep_stall_rel_tol = sep_stall_rel_tol,
                sep_stall_abs_tol = sep_stall_abs_tol,
                sep_line_search = sep_line_search,
                sep_line_search_maxit = sep_line_search_maxit,
                sep_line_search_factor = sep_line_search_factor,
                sep_line_search_min_alpha = sep_line_search_min_alpha,
                sep_lm_lambda = sep_lm_lambda,
                sep_lm_lambda_scale = sep_lm_lambda_scale,
                sep_lm_lambda_min = sep_lm_lambda_min,
                sep_lm_lambda_max = sep_lm_lambda_max,
                sep_accept_tol = irf_accept_tol,
                sep_shock_scale = sep_shock_scale,
                sep_expectation_method = sep_expectation_method,
                hmc_samples = hmc_samples,
                hmc_warmup = hmc_warmup,
                hmc_leapfrog_steps = hmc_leapfrog_steps,
                hmc_step_size = hmc_step_size,
                hmc_use_tempering = hmc_use_tempering,
                hmc_verbose = hmc_verbose,
                use_subdifferential = use_subdifferential,
                subdiff_kink_tol = subdiff_kink_tol,
                subdiff_alpha_maxit = subdiff_alpha_maxit,
                subdiff_alpha_tol = subdiff_alpha_tol,
                subdiff_verbose = subdiff_verbose,
                shock_scaling = shock_scaling,
                silent = true,
            )
            if res.errorflag
                error("IRF SEP failed for $shock size=$size at period $(res.failure_period).")
            end
            append_irf_samples!(Array(res.simulation), res.shocks)
        end
    end
end

if theta_sampling == :prior
    last_ok = findlast(theta_success)
    if last_ok === nothing
        error("No successful theta draws. Consider lowering shock scale or narrowing priors.")
    end
    theta_grid = theta_grid[1:last_ok]
    theta_seeds = theta_seeds[1:last_ok]
    theta_attempts = theta_attempts[1:last_ok]
    theta_success = theta_success[1:last_ok]
    theta_full_success = theta_full_success[1:last_ok]
    theta_stable_periods = theta_stable_periods[1:last_ok]
    failure_periods = failure_periods[1:last_ok]
    theta_times = theta_times[1:last_ok]
end

X = X[:, 1:cursor]
Y = Y[:, 1:cursor]
theta_ids = theta_ids[1:cursor]
sample_full_success = sample_full_success[1:cursor]
sep_residuals = sep_residuals[1:cursor]
if 1 in rom_orders
    Y_rom1 = Y_rom1[:, 1:cursor]
end
if 2 in rom_orders
    Y_rom2 = Y_rom2[:, 1:cursor]
end

if cursor < min_total_samples && !allow_empty_dataset
    error("Dataset generation produced $cursor samples (< min-total-samples=$min_total_samples). " *
          "Consider enabling --stable-prefix, increasing --theta-attempts-per-theta, lowering --shock-scale, " *
          "raising --sep-accept-tol, or narrowing parameter bounds.")
elseif cursor < min_total_samples
    @warn "Dataset generation produced fewer samples than requested minimum" cursor min_total_samples
end

if cursor > 0
    state_stds = vec(Statistics.std(X[1:d_state, :]; dims = 2, corrected = false))
    println("State std min/max: $(minimum(state_stds)) / $(maximum(state_stds))")
else
    println("State std min/max: NaN / NaN (empty dataset)")
end

total_time_sec = timing ? (time() - total_start) : NaN
if timing
    println("Total elapsed (s): $(round(total_time_sec, digits = 2))")
end

meta = Dict(
    "model" => string(model.model_name),
    "observables" => observables,
    "sample_idx" => sample_idx,
    "state_names" => state_names,
    "state_idx" => state_idx,
    "state_definition" => "past_not_future_and_mixed + future_not_past_and_mixed",
    "obs_idx" => obs_idx,
    "theta_names" => theta_names,
    "theta_grid" => theta_grid,
    "theta_ids" => theta_ids,
    "theta_seeds" => theta_seeds,
    "theta_success" => theta_success,
    "theta_full_success" => theta_full_success,
    "theta_stable_periods" => theta_stable_periods,
    "theta_failure_periods" => failure_periods,
    "theta_attempts" => theta_attempts,
    "shock_names" => model.exo,
    "base_values" => base_values,
    "grid_points" => grid_points,
    "theta_sampling" => theta_sampling,
    "theta_samples" => theta_samples,
    "theta_max_attempts" => theta_max_attempts,
    "theta_attempts_per_theta" => theta_attempts_per_theta,
    "seed_attempt_stride" => seed_attempt_stride,
    "retry_on_early_failure" => retry_on_early_failure,
    "retry_shock_scale_backoff" => retry_shock_scale_backoff,
    "samples_per_theta" => samples_per_theta,
    "allow_empty_dataset" => allow_empty_dataset,
    "min_total_samples" => min_total_samples,
    "burn_in" => burn_in,
    "sample_start" => sample_start,
    "sample_length" => sample_length,
    "irf_augment" => irf_augment,
    "irf_periods" => irf_periods,
    "irf_shocks" => irf_shocks,
    "irf_sizes" => irf_sizes,
    "irf_accept_tol" => irf_accept_tol,
    "sep_horizon" => sep_horizon,
    "sep_order" => sep_order,
    "sep_nnodes" => sep_nnodes,
    "sep_maxit" => sep_maxit,
    "sep_tol" => sep_tol,
    "sep_sparse_tree" => sep_sparse_tree,
    "sep_linear_solver" => sep_linear_solver,
    "sep_fallback_solver" => sep_fallback_solver,
    "sep_stall_iters" => sep_stall_iters,
    "sep_stall_rel_tol" => sep_stall_rel_tol,
    "sep_stall_abs_tol" => sep_stall_abs_tol,
    "sep_line_search" => sep_line_search,
    "sep_line_search_maxit" => sep_line_search_maxit,
    "sep_line_search_factor" => sep_line_search_factor,
    "sep_line_search_min_alpha" => sep_line_search_min_alpha,
    "sep_lm_lambda" => sep_lm_lambda,
    "sep_lm_lambda_scale" => sep_lm_lambda_scale,
    "sep_lm_lambda_min" => sep_lm_lambda_min,
    "sep_lm_lambda_max" => sep_lm_lambda_max,
    "sep_shock_scale" => sep_shock_scale,
    "sep_accept_tol" => sep_accept_tol,
    "stable_prefix" => stable_prefix,
    "stable_min_periods" => stable_min_periods,
    "sep_mode" => (sep_order == 0 ? "deterministic" : "stochastic"),
    "sep_is_stochastic" => sep_order > 0,
    "shock_scaling" => shock_scaling,
    "shock_scale" => shock_scale,
    "rom_orders" => rom_orders,
    "rom_mode" => rom_mode,
    "output_order" => ["observables_t", "state_next"],
    "theta_times" => theta_times,
    "total_time_sec" => total_time_sec,
    "output_dir" => output_dir,
)

dataset_path = joinpath(output_dir, "hlt_sep_surrogate_dataset.jls")
payload = Dict("X" => X, "Y" => Y, "meta" => meta)
payload["sample_full_success"] = sample_full_success
# RISK-1b: Store SEP residuals for quality-weighted training
if any(isfinite, sep_residuals)
    payload["sep_residuals"] = sep_residuals
    n_valid = count(isfinite, sep_residuals)
    println("SEP residuals: $n_valid / $(length(sep_residuals)) samples have finite residuals")
    println("  Residual stats: median=$(round(Statistics.median(filter(isfinite, sep_residuals)), sigdigits=4)) " *
            "max=$(round(maximum(filter(isfinite, sep_residuals)), sigdigits=4))")
end
if 1 in rom_orders
    payload["Y_rom1"] = Y_rom1
end
if 2 in rom_orders
    payload["Y_rom2"] = Y_rom2
end
serialize(dataset_path, payload)

println("Saved dataset: $dataset_path")
