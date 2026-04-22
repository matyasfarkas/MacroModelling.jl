#!/usr/bin/env julia
# ============================================================================
# MONTE CARLO COVERAGE STUDY — Surrogate Inversion-Filter Estimation
# ============================================================================
#
# Frequentist coverage validation for the surrogate (inversion filter + NN)
# NUTS-HMC estimator. The DGP generates synthetic data from the first-order
# (linear) model — the test is whether the surrogate distorts inference
# relative to the true linear DGP.
#
# Design:
#   1. Draw N_rep "true" parameter vectors from the prior
#   2. For each replication:
#      a. Set model parameters to the DGP values
#      b. Simulate T_obs periods of synthetic observables (first-order solution)
#      c. Run NUTS-HMC with surrogate inversion-filter likelihood
#      d. Compute credible intervals and check coverage
#   3. Aggregate coverage rates, bias, RMSE across replications
#
# Key difference from monte_carlo_coverage.jl:
#   The likelihood is computed via the inversion filter + NN surrogate
#   instead of the Kalman filter. This measures the coverage distortion
#   introduced by the surrogate approximation.
#
# Output:
#   - Per-replication .jls checkpoints (restartable)
#   - Summary statistics and LaTeX table
#   - Saved to .local_artifacts/monte_carlo_coverage_surrogate/
#
# Usage:
#   julia --project=. scripts/monte_carlo_coverage_surrogate.jl \
#       --surrogate=.local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls \
#       --n-rep=10 --T-obs=184 --samples=200 --adapt=100 --seed=2026 \
#       [--start-rep=1] [--dgp=prior] [--max-depth=8] [--verbose] \
#       [--obs-sigma-scale=2.0] [--obs-sigma-floor=0.1] \
#       [--inv-maxit=10] [--inv-tol=1e-6] [--inv-lambda=1e-4]
# ============================================================================

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, var, quantile, median
import Distributions
using Printf, Dates
using AdvancedHMC, LogDensityProblems
using MacroModelling
using AxisKeys

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_kv_string(args, key, default)
    for arg in args
        if startswith(arg, "$key=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end
parse_kv_int(args, key, default) = parse(Int, parse_kv_string(args, key, string(default)))
parse_kv_float(args, key, default) = parse(Float64, parse_kv_string(args, key, string(default)))

# Standard MC args
N_rep        = parse_kv_int(ARGS, "--n-rep", 10)
T_obs        = parse_kv_int(ARGS, "--T-obs", 184)
n_samples    = parse_kv_int(ARGS, "--samples", 200)
n_adapt      = parse_kv_int(ARGS, "--adapt", 100)
target_accept = parse_kv_float(ARGS, "--target-accept", 0.65)
max_depth    = parse_kv_int(ARGS, "--max-depth", 8)
seed_base    = parse_kv_int(ARGS, "--seed", 2026)
fd_eps       = parse_kv_float(ARGS, "--fd-eps", 1e-5)
start_rep    = parse_kv_int(ARGS, "--start-rep", 1)
dgp_mode     = parse_kv_string(ARGS, "--dgp", "prior")   # "prior" or "baseline_perturb"
out_dir      = parse_kv_string(ARGS, "--out-dir",
                   ".local_artifacts/monte_carlo_coverage_surrogate")
verbose      = any(==("--verbose"), ARGS)
ci_level     = parse_kv_float(ARGS, "--ci-level", 0.90)   # Credible interval level

# Surrogate-specific args
surrogate_path  = parse_kv_string(ARGS, "--surrogate", "")
obs_sigma_scale = parse_kv_float(ARGS, "--obs-sigma-scale", 2.0)
obs_sigma_floor = parse_kv_float(ARGS, "--obs-sigma-floor", 0.1)
inv_maxit       = parse_kv_int(ARGS, "--inv-maxit", 10)
inv_tol         = parse_kv_float(ARGS, "--inv-tol", 1e-6)
inv_lambda      = parse_kv_float(ARGS, "--inv-lambda", 1e-4)

if surrogate_path == ""
    error("""Usage: julia monte_carlo_coverage_surrogate.jl \\
        --surrogate=<surrogate.jls> \\
        [--n-rep=10] [--T-obs=184] [--samples=200] [--adapt=100] [--seed=2026] \\
        [--start-rep=1] [--dgp=prior] [--max-depth=8] [--verbose] \\
        [--obs-sigma-scale=2.0] [--obs-sigma-floor=0.1] \\
        [--inv-maxit=10] [--inv-tol=1e-6] [--inv-lambda=1e-4]

    ERROR: --surrogate=<path> is required.""")
end

# Derived CI quantiles
alpha_lo = (1.0 - ci_level) / 2.0
alpha_hi = 1.0 - alpha_lo

println("=" ^ 78)
println("MONTE CARLO COVERAGE STUDY — SURROGATE INVERSION FILTER")
println("Started: $(now())")
println("=" ^ 78)
println("  N_rep:          $N_rep")
println("  T_obs:          $T_obs")
println("  Samples/rep:    $n_samples (+ $n_adapt warmup)")
println("  Target accept:  $target_accept")
println("  Max tree depth: $max_depth")
println("  FD epsilon:     $fd_eps")
println("  Seed base:      $seed_base")
println("  Start rep:      $start_rep")
println("  DGP mode:       $dgp_mode")
println("  CI level:       $(Int(ci_level*100))%")
println("  Output dir:     $out_dir")
println("  Surrogate:      $surrogate_path")
println("  Obs sigma scale:$obs_sigma_scale")
println("  Obs sigma floor:$obs_sigma_floor")
println("  Inv maxit:      $inv_maxit")
println("  Inv tol:        $inv_tol")
println("  Inv lambda:     $inv_lambda")

mkpath(out_dir)

# ============================================================================
# Step 1: Load Model
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 1: Loading HLT model (non-OBC, first-order)")
println("-" ^ 78)

repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)

println("  Model:        $(mm_model.model_name)")
println("  Parameters:   $(length(mm_model.parameters))")
println("  Shocks:       $(length(mm_model.exo))")

# ============================================================================
# Step 2: Setup Parameter Configuration
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 2: Setting up 18-parameter estimation configuration")
println("-" ^ 78)

include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
specs = get_phase1_18param_specs()
spec_names = [s.name for s in specs]

theta_names = Symbol[s.name for s in specs]
n_theta = length(theta_names)

# Build prior distributions and bounds
prior_dists  = Vector{Distributions.Distribution}(undef, n_theta)
prior_bounds = Vector{Tuple{Float64,Float64}}(undef, n_theta)

for (i, tname) in enumerate(theta_names)
    si = findfirst(==(tname), spec_names)
    spec = specs[si]
    if spec.prior_type == :Beta
        prior_dists[i] = Distributions.Beta(spec.prior_params.α, spec.prior_params.β)
    elseif spec.prior_type == :InvGamma
        prior_dists[i] = Distributions.InverseGamma(spec.prior_params.α, spec.prior_params.θ)
    elseif spec.prior_type == :Normal
        prior_dists[i] = Distributions.Normal(spec.prior_params.μ, spec.prior_params.σ)
    else
        prior_dists[i] = Distributions.Uniform(spec.bounds...)
    end
    prior_bounds[i] = spec.bounds
end

# Parameter index mapping
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx_any)
        missing_names = theta_names[isnothing.(idx_any)]
        error("Theta names not found in model parameters: $missing_names")
    end
    Int.(idx_any)
end

base_parameters = copy(mm_model.parameter_values)
baseline = get_phase1_18param_baseline()

println("  Estimated parameters ($n_theta):")
for (i, tname) in enumerate(theta_names)
    lb, ub = prior_bounds[i]
    bval = get(baseline, tname, NaN)
    @printf("    [%2d] %-12s  baseline=%.4f  bounds=(%.3f, %.3f)\n",
            i, tname, bval, lb, ub)
end

# Observables: standard SW07 7-variable set
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
d_obs = length(observables)
println("  Observables ($d_obs): $observables")

# ============================================================================
# Step 3: Load Surrogate & Build ROM Predictor
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 3: Loading NN surrogate + building ROM predictor")
println("-" ^ 78)

# Load surrogate bundle
surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)

println("  Surrogate:      d_in=$(frozen.d_in), d_out=$(frozen.d_out)")
println("  Activation:     $(get_activation(frozen))")
if val_rmse !== nothing
    println("  Val RMSE:       $(round.(val_rmse[1:min(7,length(val_rmse))], sigdigits=3))")
end

# Compute state/obs indices in model variable ordering
# For the MC study, state_names come from the model's endogenous variables list.
# We need all state variables that the surrogate expects.
# Extract state_names from the surrogate metadata if available,
# otherwise use the model's past_not_future_and_mixed variables.
state_names_sym = if haskey(sur_meta, "state_names") && !isempty(sur_meta["state_names"])
    Symbol.(sur_meta["state_names"])
else
    # Fallback: use model's state variables (past_not_future_and_mixed)
    mm_model.var[mm_model.timings.past_not_future_and_mixed_idx]
end

obs_idx = indexin(observables, mm_model.var)
state_idx = indexin(state_names_sym, mm_model.var)
any(isnothing, obs_idx) && error("Observable names not found in $(mm_model.model_name): $observables")
any(isnothing, state_idx) && error("State names not found in $(mm_model.model_name): $state_names_sym")

d_state = length(state_idx)
d_shock = length(mm_model.exo)

println("  State variables:  $d_state")
println("  Shock variables:  $d_shock")

# Build ROM predictor (baseline mode)
rom_predictor = RomPredictor(mm_model,
                             1,          # ROM order 1 (first-order perturbation)
                             :baseline,  # fixed baseline — no per-theta re-solve
                             false,      # no OBC for ROM1
                             Int[],      # theta_idx empty for baseline
                             base_parameters,
                             nothing,
                             nothing,
                             Int.(state_idx),
                             Int.(obs_idx))

# Initialize ROM cache
ensure_rom_cache!(rom_predictor, Float64[])
println("  ROM1 cache initialized (baseline mode)")

# Predict functions: same architecture as surrogate HMC script (FIX AD-04)
rom_full_predict = function(state::AbstractVector, shock_t::AbstractVector, θ_local::AbstractVector)
    return rom_predict(rom_predictor, state, shock_t, θ_local)
end

# Surrogate theta padding (if surrogate has different theta count than estimation)
surrogate_theta_names = get(sur_meta, "theta_names", Symbol[])
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    _sur_theta_baseline = zeros(Float64, length(surrogate_theta_names))
    _sur_theta_est_idx = zeros(Int, length(surrogate_theta_names))
    for (si, sname) in enumerate(surrogate_theta_names)
        ei = findfirst(==(sname), theta_names)
        if ei !== nothing
            _sur_theta_est_idx[si] = ei
        else
            pi = findfirst(==(sname), mm_model.parameters)
            _sur_theta_baseline[si] = pi !== nothing ? base_parameters[pi] : 0.0
        end
    end
    println("  Surrogate theta padding: $(length(surrogate_theta_names)) surrogate -> $(length(theta_names)) estimation")
    function _pad_theta(θ_local::AbstractVector)
        θ_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                θ_full[i] = θ_local[_sur_theta_est_idx[i]]
            end
        end
        return θ_full
    end
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, _pad_theta(θ_local)))[1:d_obs]
else
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, θ_local))[1:d_obs]
end

# Combined predict: ROM1 + NN correction
surrogate_step_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
    rom_full_predict,
    surrogate_residual_predict,
    state,
    shock_t,
    θ_local,
    d_obs;
    allow_full_residual = false,
)

# ROM-only predict: for shock recovery in inversion filter
rom_only_predict = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
    rom_full_predict,
    state,
    shock_t,
    θ_local,
    d_obs,
)

# Batch NN residual evaluation (BLAS-3 efficient)
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    batch_nn_residual = function(X_nn::AbstractMatrix)
        d_prefix = d_state + d_shock
        T_batch = size(X_nn, 2)
        X_padded = Matrix{eltype(X_nn)}(undef, frozen.d_in, T_batch)
        X_padded[1:d_prefix, :] .= X_nn[1:d_prefix, :]
        for t in 1:T_batch
            X_padded[(d_prefix+1):end, t] .= _pad_theta(X_nn[(d_prefix+1):end, t])
        end
        return predict_frozen_batch(frozen, X_padded)
    end
else
    batch_nn_residual = (X_nn) -> predict_frozen_batch(frozen, X_nn)
end

# Single-sample NN residual
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    single_nn_residual = function(x_nn::AbstractVector)
        d_prefix = d_state + d_shock
        x_padded = vcat(x_nn[1:d_prefix], _pad_theta(x_nn[(d_prefix+1):end]))
        return predict_frozen(frozen, x_padded)
    end
else
    single_nn_residual = (x_nn) -> predict_frozen(frozen, x_nn)
end

# Correction clamp bounds
if val_rmse !== nothing && length(val_rmse) == frozen.d_out
    nn_correction_clamp = 3.0 .* val_rmse
    println("  Correction clamp: +/-3xRMSE (obs max=$(round(maximum(nn_correction_clamp[1:d_obs]), sigdigits=3)))")
else
    nn_correction_clamp = nothing
    println("  Correction clamp: disabled (no val_rmse)")
end

println("  Predict functions built: rom_only + surrogate_step + batch_nn + single_nn")

# Quick test of predict functions
println("\n  Testing predict functions...")
t0 = time()
test_shock = zeros(Float64, d_shock)
test_theta = Float64[base_parameters[i] for i in theta_param_idx]
s0_test = zeros(Float64, d_state)  # steady-state deviations
test_obs_rom, test_state_rom = rom_only_predict(s0_test, test_shock, test_theta)
t_rom = time() - t0
println("    ROM1 predict: $(round(t_rom*1e6, digits=0))us, obs=$(round.(test_obs_rom[1:min(3,d_obs)], sigdigits=4))")

t0 = time()
test_obs_sur, test_state_sur = surrogate_step_predict(s0_test, test_shock, test_theta)
t_sur = time() - t0
println("    Surrogate predict: $(round(t_sur*1e6, digits=0))us, obs=$(round.(test_obs_sur[1:min(3,d_obs)], sigdigits=4))")

# ============================================================================
# Step 4: Build Observation Sigma and Shock Sigmas
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 4: Building observation sigma and shock sigmas")
println("-" ^ 78)

# Observation sigma: start from data standard deviations, inflate by surrogate RMSE
# For the MC study, use a default obs_sigma based on typical observable magnitudes.
# We will scale the sigma per replication based on the generated data.

# Shock sigmas: from the model's exogenous shock structure.
# Each shock has a standard deviation parameter (z_ea, z_eb, etc.).
# For the inversion filter, shock_sigmas are the prior std devs of the shocks.
# In the MC study, these are determined by the DGP theta_true for each replication.
# At baseline:
shock_param_names = Symbol.(["z_" * string(s) for s in mm_model.exo])
shock_param_idx = indexin(shock_param_names, mm_model.parameters)

# Build a function to extract shock_sigmas from a full parameter vector
function get_shock_sigmas(params::Vector{Float64})
    sigmas = zeros(d_shock)
    for (j, idx) in enumerate(shock_param_idx)
        if idx !== nothing
            sigmas[j] = abs(params[idx])
        end
    end
    return sigmas
end

# Baseline shock sigmas for testing
shock_sigmas_baseline = get_shock_sigmas(base_parameters)
println("  Shock sigmas (baseline): $(round.(shock_sigmas_baseline, sigdigits=3))")
println("  Structural shocks (sigma>0): $(count(shock_sigmas_baseline .> 0)) / $d_shock")

# Build obs_sigma base from surrogate validation RMSE
obs_sigma_base = fill(0.5, d_obs)  # default baseline
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_rmse = val_rmse[1:d_obs] .* obs_sigma_scale
    obs_sigma_base = max.(obs_sigma_base, obs_rmse)
    println("  obs_sigma mode: max(0.5, surrogate_rmse x $obs_sigma_scale)")
end
if obs_sigma_floor > 0
    obs_sigma_base = max.(obs_sigma_base, obs_sigma_floor)
    println("  obs_sigma floor: $obs_sigma_floor")
end
println("  obs_sigma (base): $(round.(obs_sigma_base, sigdigits=3))")

# ============================================================================
# Step 5: Parameter Transform Functions
# ============================================================================

# Constrained <-> unconstrained (logit for bounded parameters)
function constrained_to_unconstrained(θ::Vector{Float64})
    x = similar(θ)
    for i in 1:length(θ)
        lb, ub = prior_bounds[i]
        val = clamp(θ[i], lb + 1e-10, ub - 1e-10)
        x[i] = log((val - lb) / (ub - val))
    end
    return x
end

function unconstrained_to_constrained(x::Vector{Float64})
    θ = similar(x)
    for i in 1:length(x)
        lb, ub = prior_bounds[i]
        θ[i] = lb + (ub - lb) / (1.0 + exp(-x[i]))
    end
    return θ
end

function log_jacobian(x::Vector{Float64})
    lj = 0.0
    for i in 1:length(x)
        lb, ub = prior_bounds[i]
        s = 1.0 / (1.0 + exp(-x[i]))
        lj += log(ub - lb) + log(s) + log(1.0 - s)
    end
    return lj
end

# ============================================================================
# Step 6: DGP — Draw True Parameter Vectors
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 6: Drawing true parameter vectors (DGP mode: $dgp_mode)")
println("-" ^ 78)

function draw_true_theta_from_prior(rng::AbstractRNG)
    θ = zeros(n_theta)
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        d_trunc = Distributions.truncated(prior_dists[i], lb, ub)
        θ[i] = rand(rng, d_trunc)
    end
    return θ
end

function draw_true_theta_baseline_perturb(rng::AbstractRNG; scale=0.15)
    θ = zeros(n_theta)
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        bval = get(baseline, theta_names[i], (lb + ub) / 2)
        bval = clamp(bval, lb + 0.05*(ub-lb), ub - 0.05*(ub-lb))
        width = min(bval - lb, ub - bval) * scale
        θ[i] = clamp(bval + randn(rng) * width, lb + 1e-6, ub - 1e-6)
    end
    return θ
end

function draw_true_theta(rng::AbstractRNG, mode::AbstractString)
    if mode == "prior"
        return draw_true_theta_from_prior(rng)
    elseif mode == "baseline_perturb"
        return draw_true_theta_baseline_perturb(rng)
    else
        error("Unknown DGP mode: $mode. Use 'prior' or 'baseline_perturb'.")
    end
end

# Pre-draw all true theta vectors for reproducibility
rng_dgp = MersenneTwister(seed_base)
theta_true_all = [draw_true_theta(rng_dgp, dgp_mode) for _ in 1:N_rep]

println("  Drew $N_rep true parameter vectors.")
println("  Example (rep 1): $(round.(theta_true_all[1], digits=4))")

# ============================================================================
# Step 7: Synthetic Data Generation Function
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 7: Setting up synthetic data generation")
println("-" ^ 78)

"""
    generate_synthetic_data(model, theta_true, T; rng, burn_in=100)

Simulate T periods of observable data from the first-order solution of the model
at the given parameter values. Returns a (d_obs x T) matrix and the full param vector.

Uses MacroModelling's `simulate` function with `levels=true` to get the
observation equations evaluated at simulated paths.
"""
function generate_synthetic_data(model, θ_true::Vector{Float64}, T::Int;
                                  rng::AbstractRNG=Random.GLOBAL_RNG,
                                  burn_in::Int=100)
    # Build full parameter vector with true values inserted
    params = copy(base_parameters)
    for (j, idx) in enumerate(theta_param_idx)
        params[idx] = θ_true[j]
    end

    # Simulate T + burn_in periods at first order (levels=true for observables)
    total_periods = T + burn_in
    sim = nothing
    try
        sim = MacroModelling.simulate(model;
            parameters = params,
            periods = total_periods,
            algorithm = :first_order,
            levels = true,
            verbose = false)
    catch e
        @warn "Simulation failed at theta=$(round.(θ_true, digits=3)): $e"
        return nothing, nothing
    end

    # Extract observables after burn-in
    # sim is a 3D KeyedArray (Variables, Periods, Shocks) — use positional indexing
    # to avoid AxisKeys ambiguity with Symbol lookup across dimensions
    sim_vars = axiskeys(sim, 1)
    obs_matrix = zeros(d_obs, T)
    for (oi, obs_name) in enumerate(observables)
        vi = findfirst(==(obs_name), sim_vars)
        if vi !== nothing
            for t in 1:T
                obs_matrix[oi, t] = sim[vi, burn_in + t, 1]
            end
        else
            @warn "Observable $obs_name not found in simulation output."
            return nothing, nothing
        end
    end

    # Check for NaN / Inf
    if any(!isfinite, obs_matrix)
        @warn "Non-finite values in simulated data at theta=$(round.(θ_true, digits=3))"
        return nothing, nothing
    end

    return obs_matrix, params
end

# Test data generation at baseline
θ_test = Float64[get(baseline, tn, NaN) for tn in theta_names]
any(isnan, θ_test) && error("Missing baseline value for some parameter")
test_data, _ = generate_synthetic_data(mm_model, θ_test, 10; burn_in=20)
if test_data !== nothing
    println("  Data generation test passed. Shape: $(size(test_data))")
    println("  Sample obs means: $(round.(mean(test_data, dims=2)[:,1], digits=3))")
else
    println("  WARNING: Data generation test at baseline failed!")
end

# ============================================================================
# Step 8: Estimation Functions (Surrogate Inversion Filter + NUTS per replication)
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 8: Setting up per-replication surrogate estimation")
println("-" ^ 78)

# Log prior (constrained space)
function log_prior(θ_constrained::Vector{Float64})
    lp = 0.0
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        if θ_constrained[i] < lb || θ_constrained[i] > ub
            return -Inf
        end
        d = Distributions.truncated(prior_dists[i], lb, ub)
        lp += Distributions.logpdf(d, θ_constrained[i])
    end
    return lp
end

# Finite-difference gradient
function fd_gradient!(grad::Vector{Float64}, f::Function, x::Vector{Float64}, h::Float64)
    f0 = f(x)
    @inbounds for i in 1:length(x)
        x_old = x[i]
        x[i] = x_old + h
        fp = f(x)
        x[i] = x_old
        grad[i] = (fp - f0) / h
    end
    return f0
end

# Build starting point: blend of true theta and prior mode
function build_init_theta(θ_true::Vector{Float64}; noise_scale::Float64=0.05)
    θ_init = similar(θ_true)
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        # Start at a point near (but not exactly at) the true value,
        # slightly biased toward the prior mode to avoid starting at truth
        prior_mode = if specs[findfirst(==(theta_names[i]), spec_names)].prior_type == :Beta
            α = specs[findfirst(==(theta_names[i]), spec_names)].prior_params.α
            β = specs[findfirst(==(theta_names[i]), spec_names)].prior_params.β
            α / (α + β)
        elseif specs[findfirst(==(theta_names[i]), spec_names)].prior_type == :InvGamma
            specs[findfirst(==(theta_names[i]), spec_names)].prior_params.θ
        elseif specs[findfirst(==(theta_names[i]), spec_names)].prior_type == :Normal
            specs[findfirst(==(theta_names[i]), spec_names)].prior_params.μ
        else
            (lb + ub) / 2
        end
        # 70% prior mode + 30% true value + noise
        θ_init[i] = 0.7 * prior_mode + 0.3 * θ_true[i]
        θ_init[i] = clamp(θ_init[i], lb + 0.05*(ub-lb), ub - 0.05*(ub-lb))
    end
    return θ_init
end

# LogDensityProblems interface struct
struct MCSurrogateLogDensity
    dim::Int
    fd_eps::Float64
    logdensity_fn::Function
end

LogDensityProblems.logdensity(p::MCSurrogateLogDensity, x::AbstractVector) =
    p.logdensity_fn(Vector{Float64}(x))
LogDensityProblems.dimension(p::MCSurrogateLogDensity) = p.dim
LogDensityProblems.capabilities(::Type{MCSurrogateLogDensity}) =
    LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity_and_gradient(p::MCSurrogateLogDensity, x::AbstractVector)
    xv = Vector{Float64}(x)
    grad = zeros(p.dim)
    lp = fd_gradient!(grad, p.logdensity_fn, xv, p.fd_eps)
    return lp, grad
end

# Simple batch-means ESS estimator
function ess_batch_means(chain::Vector{Float64}; batch_size::Int=50)
    n = length(chain)
    n < 2 * batch_size && return Float64(n)
    n_batches = n ÷ batch_size
    batch_means = [mean(chain[(i-1)*batch_size+1 : i*batch_size]) for i in 1:n_batches]
    var_total = var(chain)
    var_batch = var(batch_means)
    var_batch < 1e-20 && return Float64(n)
    return n * var_total / (batch_size * var_batch)
end

"""
    build_surrogate_loglik(obs_data_matrix, obs_sigma, s0, shock_sigmas)

Build surrogate inversion-filter log-likelihood closure for given synthetic data.
Uses the globally-defined ROM + NN predict functions.
"""
function build_surrogate_loglik(obs_data_matrix::Matrix{Float64},
                                 obs_sigma::Vector{Float64},
                                 s0::Vector{Float64},
                                 shock_sigmas::Vector{Float64})
    function surrogate_ll(θ_constrained::Vector{Float64})
        # Run inversion filter: ROM1 recovers shocks, batch NN corrects obs
        ll_vec, _ = MacroModelling.inversion_loglik_per_period(
            rom_only_predict,           # ROM1 for shock recovery + state propagation
            s0,
            θ_constrained,
            obs_data_matrix,
            obs_sigma,
            shock_sigmas;
            batch_eval_residual_fn = batch_nn_residual,  # NN obs correction in batch Phase 2
            maxit  = inv_maxit,
            tol    = inv_tol,
            lambda = inv_lambda,
        )
        return sum(ll_vec)
    end
    return surrogate_ll
end

"""
    run_single_replication(rep_id, theta_true, obs_data; seed, ...)

Run a single Monte Carlo replication: build surrogate inversion-filter likelihood
for the given synthetic data, then run NUTS-HMC and return posterior summary.
"""
function run_single_replication(rep_id::Int,
                                θ_true::Vector{Float64},
                                obs_data::Matrix{Float64};
                                seed::Int=42)
    Random.seed!(seed)

    # Build full parameter vector for this DGP to extract shock sigmas
    params_dgp = copy(base_parameters)
    for (j, idx) in enumerate(theta_param_idx)
        params_dgp[idx] = θ_true[j]
    end
    shock_sigmas = get_shock_sigmas(params_dgp)

    # For the inversion filter, shock_sigmas must have structural shocks > 0
    # Replace any zero shock_sigmas with baseline values (the DGP may have set them)
    for j in 1:d_shock
        if shock_sigmas[j] <= 0.0
            shock_sigmas[j] = shock_sigmas_baseline[j]
        end
    end

    # Build observation sigma: scale surrogate RMSE and apply floor
    obs_sigma = copy(obs_sigma_base)

    # Optionally adapt obs_sigma to data scale (use empirical std of observables)
    data_std = vec(std(obs_data, dims=2))
    for i in 1:d_obs
        # Ensure obs_sigma is at least obs_sigma_floor and at least 5% of data std
        obs_sigma[i] = max(obs_sigma[i], 0.05 * data_std[i], obs_sigma_floor)
    end

    # Initial state: steady state in deviation space (zeros)
    s0 = zeros(Float64, d_state)

    # Build surrogate log-likelihood closure
    surrogate_ll = build_surrogate_loglik(obs_data, obs_sigma, s0, shock_sigmas)

    # Full log-density in unconstrained space
    function log_density_unc(x::Vector{Float64})
        θ = unconstrained_to_constrained(x)
        lp = log_prior(θ)
        !isfinite(lp) && return -Inf
        ll = surrogate_ll(θ)
        !isfinite(ll) && return -Inf
        lj = log_jacobian(x)
        return ll + lp + lj
    end

    # Initialize at a point near the prior mode (not at truth)
    θ_init = build_init_theta(θ_true)
    x_init = constrained_to_unconstrained(θ_init)

    # Verify log-density is finite at init
    ld_init = log_density_unc(x_init)
    if !isfinite(ld_init)
        @warn "Rep $rep_id: log-density at init is $ld_init, trying prior mode..."
        # Fallback: use prior mode
        for i in 1:n_theta
            lb, ub = prior_bounds[i]
            θ_init[i] = clamp(mean(prior_dists[i]), lb + 0.05*(ub-lb), ub - 0.05*(ub-lb))
        end
        x_init = constrained_to_unconstrained(θ_init)
        ld_init = log_density_unc(x_init)
        if !isfinite(ld_init)
            @warn "Rep $rep_id: FAILED -- log-density not finite even at prior mode."
            return nothing
        end
    end

    # Test gradient
    grad_test = zeros(n_theta)
    fd_gradient!(grad_test, log_density_unc, copy(x_init), fd_eps)
    if !all(isfinite, grad_test)
        @warn "Rep $rep_id: gradient contains non-finite values at init."
        return nothing
    end

    # Build AdvancedHMC sampler
    log_density_obj = MCSurrogateLogDensity(n_theta, fd_eps, log_density_unc)
    metric = UnitEuclideanMetric(n_theta)
    hamiltonian = Hamiltonian(metric, log_density_obj)

    initial_eps = try
        find_good_stepsize(hamiltonian, x_init)
    catch e
        @warn "Rep $rep_id: find_good_stepsize failed: $e. Using eps=0.01."
        0.01
    end

    integrator = Leapfrog(initial_eps)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator,
                 GeneralisedNoUTurn(max_depth=max_depth)))
    adaptor = StepSizeAdaptor(target_accept, integrator)

    # Run NUTS
    t_start = time()
    local samples_unc, stats
    try
        samples_unc, stats = sample(hamiltonian, kernel, x_init,
                                     n_samples + n_adapt, adaptor, n_adapt;
                                     progress=false, verbose=false)
    catch e
        @warn "Rep $rep_id: NUTS sampling failed: $e"
        return nothing
    end
    t_elapsed = time() - t_start

    # Transform to constrained space and discard warmup
    θ_chain = hcat([unconstrained_to_constrained(s) for s in samples_unc]...)'
    θ_post = θ_chain[(n_adapt+1):end, :]

    # Diagnostics
    n_divergent = sum(s.numerical_error for s in stats[(n_adapt+1):end])
    accept_rates = [s.acceptance_rate for s in stats[(n_adapt+1):end]]
    tree_depths = [s.tree_depth for s in stats[(n_adapt+1):end]]

    # Posterior summary
    post_mean = vec(mean(θ_post, dims=1))
    post_std  = vec(std(θ_post, dims=1))
    post_q_lo = [quantile(θ_post[:, i], alpha_lo) for i in 1:n_theta]
    post_q_hi = [quantile(θ_post[:, i], alpha_hi) for i in 1:n_theta]
    post_median = [median(θ_post[:, i]) for i in 1:n_theta]

    # Coverage: does the CI contain the true value?
    covered = [post_q_lo[i] < θ_true[i] < post_q_hi[i] for i in 1:n_theta]

    # ESS
    ess_vals = [ess_batch_means(θ_post[:, i]) for i in 1:n_theta]

    # Log-likelihood at posterior mean and true value
    ll_post_mean = surrogate_ll(post_mean)
    ll_true = surrogate_ll(θ_true)

    return Dict{String,Any}(
        "rep_id"         => rep_id,
        "theta_true"     => θ_true,
        "post_mean"      => post_mean,
        "post_std"       => post_std,
        "post_q_lo"      => post_q_lo,
        "post_q_hi"      => post_q_hi,
        "post_median"    => post_median,
        "covered"        => covered,
        "ess"            => ess_vals,
        "n_divergent"    => n_divergent,
        "mean_accept"    => mean(accept_rates),
        "mean_tree_depth"=> mean(tree_depths),
        "ll_post_mean"   => ll_post_mean,
        "ll_true"        => ll_true,
        "elapsed_s"      => t_elapsed,
        "n_samples"      => n_samples,
        "n_adapt"        => n_adapt,
        "seed"           => seed,
        "obs_sigma"      => obs_sigma,
        "shock_sigmas"   => shock_sigmas,
    )
end

# ============================================================================
# Step 9: Run All Replications
# ============================================================================

println("\n" * "=" ^ 78)
println("STEP 9: Running $N_rep Monte Carlo replications (surrogate)")
println("=" ^ 78)

# Check for existing checkpoint
checkpoint_path = joinpath(out_dir, "mc_coverage_surrogate_checkpoint.jls")
completed_results = Dict{Int,Any}()

if isfile(checkpoint_path)
    try
        loaded = deserialize(checkpoint_path)
        if loaded isa Dict
            completed_results = loaded
            println("  Loaded checkpoint with $(length(completed_results)) completed replications.")
        end
    catch e
        @warn "Could not load checkpoint: $e. Starting fresh."
    end
end

# Timing estimate
println("\n  Timing calibration...")
flush(stdout)

θ_timing = Float64[get(baseline, tn, NaN) for tn in theta_names]
test_data_timing, test_params_timing = generate_synthetic_data(mm_model, θ_timing, T_obs; burn_in=100)
if test_data_timing !== nothing
    shock_sigmas_timing = get_shock_sigmas(test_params_timing)
    for j in 1:d_shock
        if shock_sigmas_timing[j] <= 0.0
            shock_sigmas_timing[j] = shock_sigmas_baseline[j]
        end
    end
    s0_timing = zeros(Float64, d_state)
    obs_sigma_timing = copy(obs_sigma_base)

    surrogate_ll_timing = build_surrogate_loglik(test_data_timing, obs_sigma_timing, s0_timing, shock_sigmas_timing)

    t_inv0 = time()
    surrogate_ll_timing(θ_timing)
    t_inv_ms = (time() - t_inv0) * 1000

    est_grad_ms = t_inv_ms * (n_theta + 1)
    est_leapfrogs = 2^max_depth  # worst case NUTS
    est_per_draw_s = est_leapfrogs * est_grad_ms / 1000
    est_per_rep_h = (n_samples + n_adapt) * est_per_draw_s / 3600
    est_total_h = est_per_rep_h * (N_rep - length(completed_results))

    println("  Surrogate eval:     $(round(t_inv_ms, digits=1)) ms")
    println("  Est. gradient:      $(round(est_grad_ms, digits=0)) ms")
    println("  Est. per draw:      $(round(est_per_draw_s, digits=1)) s (worst case)")
    println("  Est. per rep:       $(round(est_per_rep_h, digits=2)) hours (worst case)")
    println("  Est. total:         $(round(est_total_h, digits=1)) hours ($(N_rep - length(completed_results)) remaining)")
end

# Main replication loop
t_study_start = time()
n_success = 0
n_fail = 0

for rep in start_rep:N_rep
    global n_success, n_fail, completed_results
    # Skip if already completed
    if haskey(completed_results, rep)
        println("\n  Rep $rep/$N_rep: already completed (from checkpoint), skipping.")
        n_success += 1
        continue
    end

    println("\n" * "-" ^ 78)
    println("  REPLICATION $rep / $N_rep (SURROGATE)")
    println("-" ^ 78)

    θ_true = theta_true_all[rep]
    rep_seed = seed_base * 1000 + rep

    # Generate synthetic data
    println("  Drawing true theta: $(round.(θ_true, digits=4))")
    flush(stdout)

    t_data_start = time()
    rng_sim = MersenneTwister(rep_seed)
    obs_data, _ = generate_synthetic_data(mm_model, θ_true, T_obs;
                                           rng=rng_sim, burn_in=100)

    if obs_data === nothing
        @warn "  Rep $rep: data generation failed. Skipping."
        n_fail += 1
        completed_results[rep] = Dict("rep_id" => rep, "status" => "dgp_failed",
                                       "theta_true" => θ_true)
        serialize(checkpoint_path, completed_results)
        continue
    end

    t_data_elapsed = time() - t_data_start
    println("  Data generated: $(size(obs_data)) in $(round(t_data_elapsed, digits=1))s")
    println("  Obs ranges: min=$(round(minimum(obs_data), digits=2)), max=$(round(maximum(obs_data), digits=2))")
    flush(stdout)

    # Run estimation
    println("  Running NUTS-HMC with surrogate inversion filter ($n_samples draws + $n_adapt warmup)...")
    flush(stdout)

    result = run_single_replication(rep, θ_true, obs_data; seed=rep_seed + 7)

    if result === nothing
        @warn "  Rep $rep: estimation failed."
        n_fail += 1
        completed_results[rep] = Dict("rep_id" => rep, "status" => "estimation_failed",
                                       "theta_true" => θ_true)
    else
        n_success += 1
        result["status"] = "success"
        completed_results[rep] = result

        # Print per-parameter summary
        covered = result["covered"]
        n_covered = sum(covered)
        println("\n  Rep $rep results:")
        @printf("  %-12s %8s %8s %8s %8s %8s %5s %6s\n",
                "Param", "True", "Mean", "Std", "Q_lo", "Q_hi", "In CI", "ESS")
        println("  " * "-" ^ 68)
        for i in 1:n_theta
            @printf("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f %5s %6.0f\n",
                    theta_names[i],
                    θ_true[i],
                    result["post_mean"][i],
                    result["post_std"][i],
                    result["post_q_lo"][i],
                    result["post_q_hi"][i],
                    covered[i] ? "yes" : "NO",
                    result["ess"][i])
        end
        println("  " * "-" ^ 68)
        println("  Coverage: $n_covered / $n_theta ($(round(100*n_covered/n_theta, digits=0))%)")
        println("  LL at truth: $(round(result["ll_true"], digits=1))")
        println("  LL at post mean: $(round(result["ll_post_mean"], digits=1))")
        println("  Divergences: $(result["n_divergent"])")
        println("  Elapsed: $(round(result["elapsed_s"]/60, digits=1)) min")
    end

    # Save checkpoint after every replication
    serialize(checkpoint_path, completed_results)
    println("  Checkpoint saved ($n_success success, $n_fail failed out of $(rep - start_rep + 1))")
    flush(stdout)
end

t_study_elapsed = time() - t_study_start

# ============================================================================
# Step 10: Aggregate Results
# ============================================================================

println("\n" * "=" ^ 78)
println("STEP 10: Aggregating Monte Carlo results (surrogate)")
println("=" ^ 78)

# Collect successful replications
success_results = [completed_results[k] for k in sort(collect(keys(completed_results)))
                   if get(completed_results[k], "status", "") == "success"]
N_success = length(success_results)

println("  Successful replications: $N_success / $N_rep")

if N_success < 2
    println("  ERROR: fewer than 2 successful replications. Cannot compute statistics.")
    println("  Check the log output above for failures.")
    exit(1)
end

# Collect arrays
all_true    = hcat([r["theta_true"] for r in success_results]...)'    # (N_success, n_theta)
all_mean    = hcat([r["post_mean"] for r in success_results]...)'
all_std     = hcat([r["post_std"] for r in success_results]...)'
all_covered = hcat([r["covered"] for r in success_results]...)'       # (N_success, n_theta) Bool
all_ess     = hcat([r["ess"] for r in success_results]...)'
all_diverge = [r["n_divergent"] for r in success_results]
all_accept  = [r["mean_accept"] for r in success_results]

# Per-parameter statistics
coverage_rates = vec(mean(all_covered, dims=1))
mean_bias      = vec(mean(all_mean .- all_true, dims=1))
median_bias    = [median(all_mean[:, i] .- all_true[:, i]) for i in 1:n_theta]
rmse           = [sqrt(mean((all_mean[:, i] .- all_true[:, i]).^2)) for i in 1:n_theta]
mean_post_std  = vec(mean(all_std, dims=1))
mean_ess       = vec(mean(all_ess, dims=1))

# Relative bias: bias / mean of |theta_true|
true_abs_mean  = vec(mean(abs.(all_true), dims=1))
rel_bias       = mean_bias ./ max.(true_abs_mean, 1e-6)

# Confidence interval for coverage rate (Clopper-Pearson)
function clopper_pearson_ci(k::Int, n::Int, alpha::Float64=0.05)
    if k == 0
        lo = 0.0
    else
        lo = quantile(Distributions.Beta(k, n - k + 1), alpha / 2)
    end
    if k == n
        hi = 1.0
    else
        hi = quantile(Distributions.Beta(k + 1, n - k), 1.0 - alpha / 2)
    end
    return lo, hi
end

# Print summary table
println("\n" * "=" ^ 90)
@printf("  SURROGATE COVERAGE SUMMARY  (%d%% CI, %d successful replications)\n",
        Int(ci_level*100), N_success)
println("=" ^ 90)
@printf("  %-12s %8s %8s %8s %8s %8s %8s %12s\n",
        "Parameter", "Cover", "CI_lo", "CI_hi", "Bias", "RMSE", "Post SD", "Mean ESS")
println("  " * "-" ^ 86)

for i in 1:n_theta
    k_covered = sum(all_covered[:, i])
    ci_lo, ci_hi = clopper_pearson_ci(k_covered, N_success)
    @printf("  %-12s %7.1f%% %7.1f%% %7.1f%% %+8.4f %8.4f %8.4f %8.0f\n",
            theta_names[i],
            100 * coverage_rates[i],
            100 * ci_lo,
            100 * ci_hi,
            mean_bias[i],
            rmse[i],
            mean_post_std[i],
            mean_ess[i])
end
println("  " * "-" ^ 86)
@printf("  %-12s %7.1f%%\n", "AVERAGE", 100 * mean(coverage_rates))
println("  " * "-" ^ 86)
println("  Total divergences: $(sum(all_diverge)) across $N_success reps ($(round(mean(all_diverge), digits=1)) per rep)")
println("  Mean acceptance:   $(round(mean(all_accept), digits=3))")

# ============================================================================
# Step 11: Generate LaTeX Table
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 11: Generating LaTeX table (surrogate)")
println("-" ^ 78)

latex_path = joinpath(out_dir, "mc_coverage_surrogate_table.tex")

# Nice parameter display names
param_display = Dict(
    :crhoa => raw"\rho_a", :crhob => raw"\rho_b", :crhog => raw"\rho_g",
    :crhoqs => raw"\rho_{qs}", :crhopinf => raw"\rho_\pi", :crhow => raw"\rho_w",
    :crhoms => raw"\rho_{ms}",
    :z_ea => raw"\sigma_a", :z_eb => raw"\sigma_b", :z_eg => raw"\sigma_g",
    :z_eqs => raw"\sigma_{qs}", :z_epinf => raw"\sigma_\pi", :z_ew => raw"\sigma_w",
    :z_em => raw"\sigma_{ms}",
    :cprobp => raw"\xi_p", :cindp => raw"\iota_p",
    :curvp => raw"\varepsilon_p", :cprobw => raw"\xi_w",
)

open(latex_path, "w") do io
    println(io, "% Surrogate Monte Carlo coverage study: $(Int(ci_level*100))% CI, $N_success replications, T=$T_obs, DGP=$dgp_mode")
    println(io, "% Likelihood: inversion filter + NN surrogate")
    println(io, "% Generated: $(now())")
    println(io, raw"\begin{table}[htbp]")
    println(io, raw"\centering")
    println(io, "\\caption{Surrogate Monte Carlo coverage study: $(Int(ci_level*100))\\% credible intervals ($N_success replications, \$T=$T_obs\$)}")
    println(io, raw"\label{tab:mc_coverage_surrogate}")
    println(io, raw"\begin{tabular}{lcccccc}")
    println(io, raw"\hline\hline")
    println(io, raw"Parameter & Coverage (\%) & 95\% CI & Bias & RMSE & Post.\ SD & ESS \\")
    println(io, raw"\hline")

    # Persistence parameters
    println(io, raw"\multicolumn{7}{l}{\textit{Shock persistence}} \\")
    for sym in [:crhoa, :crhob, :crhog, :crhoqs, :crhopinf, :crhow, :crhoms]
        i = findfirst(==(sym), theta_names)
        i === nothing && continue
        k_covered = sum(all_covered[:, i])
        ci_lo, ci_hi = clopper_pearson_ci(k_covered, N_success)
        dname = get(param_display, sym, string(sym))
        @printf(io, "\$%s\$ & %.1f & [%.1f, %.1f] & %+.4f & %.4f & %.4f & %.0f \\\\\n",
                dname, 100*coverage_rates[i], 100*ci_lo, 100*ci_hi,
                mean_bias[i], rmse[i], mean_post_std[i], mean_ess[i])
    end

    # Volatility parameters
    println(io, raw"\hline")
    println(io, raw"\multicolumn{7}{l}{\textit{Shock volatility}} \\")
    for sym in [:z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em]
        i = findfirst(==(sym), theta_names)
        i === nothing && continue
        k_covered = sum(all_covered[:, i])
        ci_lo, ci_hi = clopper_pearson_ci(k_covered, N_success)
        dname = get(param_display, sym, string(sym))
        @printf(io, "\$%s\$ & %.1f & [%.1f, %.1f] & %+.4f & %.4f & %.4f & %.0f \\\\\n",
                dname, 100*coverage_rates[i], 100*ci_lo, 100*ci_hi,
                mean_bias[i], rmse[i], mean_post_std[i], mean_ess[i])
    end

    # Structural parameters
    println(io, raw"\hline")
    println(io, raw"\multicolumn{7}{l}{\textit{Structural}} \\")
    for sym in [:cprobp, :cindp, :curvp, :cprobw]
        i = findfirst(==(sym), theta_names)
        i === nothing && continue
        k_covered = sum(all_covered[:, i])
        ci_lo, ci_hi = clopper_pearson_ci(k_covered, N_success)
        dname = get(param_display, sym, string(sym))
        @printf(io, "\$%s\$ & %.1f & [%.1f, %.1f] & %+.4f & %.4f & %.4f & %.0f \\\\\n",
                dname, 100*coverage_rates[i], 100*ci_lo, 100*ci_hi,
                mean_bias[i], rmse[i], mean_post_std[i], mean_ess[i])
    end

    println(io, raw"\hline")
    @printf(io, "Average & %.1f & & & & & \\\\\n", 100*mean(coverage_rates))
    println(io, raw"\hline\hline")
    println(io, raw"\end{tabular}")
    println(io, raw"\begin{minipage}{0.95\textwidth}")
    println(io, "\\footnotesize\\textit{Notes:} Surrogate coverage rates for $(Int(ci_level*100))\\% highest posterior density intervals from $N_success Monte Carlo replications. ",
            "Each replication draws a true parameter vector from the prior, simulates \$T=$T_obs\$ quarters of data from the \\emph{first-order} (linear) solution, ",
            "and estimates the model using NUTS-HMC with the surrogate inversion-filter likelihood ($n_samples post-warmup draws). ",
            "The surrogate uses a neural network correction on top of ROM1 (first-order perturbation) predictions. ",
            "``95\\% CI'' gives the Clopper--Pearson confidence interval for the coverage rate. ",
            "``Bias'' is the average posterior mean minus the true value. ",
            "``RMSE'' is the root mean squared error of the posterior mean. ",
            "``Post.\\ SD'' is the average posterior standard deviation. ",
            "``ESS'' is the effective sample size (batch means estimator). ",
            "The DGP is linear, so any coverage distortion measures the bias introduced by the surrogate approximation.")
    println(io, raw"\end{minipage}")
    println(io, raw"\end{table}")
end

println("  LaTeX table saved: $latex_path")

# ============================================================================
# Step 12: Save Final Results
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 12: Saving final results")
println("-" ^ 78)

final_results = Dict{String,Any}(
    "N_rep"            => N_rep,
    "N_success"        => N_success,
    "T_obs"            => T_obs,
    "n_samples"        => n_samples,
    "n_adapt"          => n_adapt,
    "ci_level"         => ci_level,
    "dgp_mode"         => dgp_mode,
    "seed_base"        => seed_base,
    "theta_names"      => theta_names,
    "prior_bounds"     => prior_bounds,
    "coverage_rates"   => coverage_rates,
    "mean_bias"        => mean_bias,
    "median_bias"      => median_bias,
    "rmse"             => rmse,
    "rel_bias"         => rel_bias,
    "mean_post_std"    => mean_post_std,
    "mean_ess"         => mean_ess,
    "all_true"         => all_true,
    "all_mean"         => all_mean,
    "all_std"          => all_std,
    "all_covered"      => all_covered,
    "all_diverge"      => all_diverge,
    "all_accept"       => all_accept,
    "per_rep_results"  => completed_results,
    "elapsed_total_s"  => t_study_elapsed,
    "timestamp"        => string(now()),
    # Surrogate-specific metadata
    "likelihood"       => "inversion_filter + surrogate",
    "surrogate_path"   => surrogate_path,
    "inv_maxit"        => inv_maxit,
    "inv_tol"          => inv_tol,
    "inv_lambda"       => inv_lambda,
    "obs_sigma_scale"  => obs_sigma_scale,
    "obs_sigma_floor"  => obs_sigma_floor,
    "obs_sigma_base"   => obs_sigma_base,
)

final_path = joinpath(out_dir, "mc_coverage_surrogate_results.jls")
serialize(final_path, final_results)
println("  Final results: $final_path")

# Also save a human-readable summary
summary_path = joinpath(out_dir, "mc_coverage_surrogate_summary.txt")
open(summary_path, "w") do io
    println(io, "MONTE CARLO COVERAGE STUDY — SURROGATE INVERSION FILTER — SUMMARY")
    println(io, "=" ^ 78)
    println(io, "Date:           $(now())")
    println(io, "N_rep:          $N_rep ($N_success successful)")
    println(io, "T_obs:          $T_obs")
    println(io, "Samples/rep:    $n_samples (+ $n_adapt warmup)")
    println(io, "CI level:       $(Int(ci_level*100))%")
    println(io, "DGP mode:       $dgp_mode")
    println(io, "Likelihood:     inversion filter + NN surrogate")
    println(io, "Surrogate:      $surrogate_path")
    println(io, "Inv maxit:      $inv_maxit")
    println(io, "Inv tol:        $inv_tol")
    println(io, "Inv lambda:     $inv_lambda")
    println(io, "Obs sigma scale:$obs_sigma_scale")
    println(io, "Obs sigma floor:$obs_sigma_floor")
    println(io, "Total time:     $(round(t_study_elapsed/3600, digits=1)) hours")
    println(io, "")
    @printf(io, "%-12s %8s %8s %8s %8s %8s\n",
            "Parameter", "Cover%", "Bias", "RMSE", "PostSD", "ESS")
    println(io, "-" ^ 60)
    for i in 1:n_theta
        @printf(io, "%-12s %7.1f%% %+8.4f %8.4f %8.4f %8.0f\n",
                theta_names[i],
                100*coverage_rates[i],
                mean_bias[i],
                rmse[i],
                mean_post_std[i],
                mean_ess[i])
    end
    println(io, "-" ^ 60)
    @printf(io, "%-12s %7.1f%%\n", "AVERAGE", 100*mean(coverage_rates))
    println(io, "")
    println(io, "Total divergences: $(sum(all_diverge))")
    println(io, "Mean acceptance:   $(round(mean(all_accept), digits=3))")
    println(io, "")
    println(io, "Note: DGP generates data from the linear (first-order) model.")
    println(io, "Any coverage distortion measures bias from the surrogate approximation.")
end

println("  Summary:      $summary_path")

# ============================================================================
# Done
# ============================================================================

println("\n" * "=" ^ 78)
println("SURROGATE MONTE CARLO COVERAGE STUDY COMPLETE")
println("Finished: $(now())")
println("Elapsed:  $(round(t_study_elapsed/3600, digits=2)) hours")
println("Success:  $N_success / $N_rep replications")
println("Mean coverage: $(round(100*mean(coverage_rates), digits=1))%")
println("=" ^ 78)
