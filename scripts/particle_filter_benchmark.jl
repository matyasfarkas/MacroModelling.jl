#!/usr/bin/env julia
# ============================================================================
# PARTICLE FILTER BENCHMARK — SW07-HLT DSGE Model
# ============================================================================
#
# Evaluates the bootstrap particle filter log-likelihood at the posterior
# mean from the linear HMC chain and (if available) the surrogate HMC chain,
# then compares against the Kalman filter LL.
#
# For the linear (first-order) model, the particle filter LL should converge
# to the Kalman filter LL as n_particles -> infinity. This validates:
#   1. The particle filter implementation is correct
#   2. The surrogate approximation quality (LL gap)
#
# Usage:
#   julia --project=. scripts/particle_filter_benchmark.jl \
#       [--data=<payload.jls>] [--linear-chain=<chain.jls>] \
#       [--surrogate-chain=<chain.jls>] [--out=<dir>] [--verbose]
#
# Defaults look in .local_artifacts/hlt_18param_realdata/ for standard files.
# ============================================================================

using Serialization, Random, LinearAlgebra, Printf, Dates
import Statistics: mean, std, quantile
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

repo_root = normpath(joinpath(@__DIR__, ".."))
artifacts_dir = joinpath(repo_root, ".local_artifacts", "hlt_18param_realdata")

data_path = parse_kv_string(ARGS, "--data",
    joinpath(artifacts_dir, "hlt_real_data_payload_extended_18p.jls"))
linear_chain_path = parse_kv_string(ARGS, "--linear-chain",
    joinpath(artifacts_dir, "hlt_linear_hmc_extended_18p_2000.jls"))
surrogate_chain_path = parse_kv_string(ARGS, "--surrogate-chain",
    joinpath(artifacts_dir, "hlt_surrogate_hmc_extended_18p_2000.jls"))
out_dir = parse_kv_string(ARGS, "--out",
    joinpath(repo_root, ".local_artifacts", "particle_filter_benchmark"))
verbose = any(==("--verbose"), ARGS)

println("=" ^ 76)
println("PARTICLE FILTER BENCHMARK — SW07-HLT")
println("Started: $(now())")
println("=" ^ 76)

# ============================================================================
# Step 1: Load Data Payload
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 1: Loading data payload")
println("-" ^ 76)

if !isfile(data_path)
    error("Data payload not found: $data_path\n" *
          "Run the data preparation pipeline first.")
end

payload = Serialization.deserialize(data_path)
if !(payload isa Dict)
    payload = Dict{Any,Any}(payload)
end

obs_data     = payload["obs_data"]          # (n_obs, T)
theta_names  = payload["theta_names"]       # Vector{Symbol}
observables  = payload["observables"]       # Vector{Symbol}

n_obs = size(obs_data, 1)
T_obs = size(obs_data, 2)
n_theta = length(theta_names)

println("  Data payload:   $data_path")
println("  Observables:    $observables ($n_obs)")
println("  Periods:        $T_obs")
println("  Parameters:     $theta_names ($n_theta)")

# ============================================================================
# Step 2: Load Model
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 2: Loading HLT model (non-OBC, first-order)")
println("-" ^ 76)

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)

# Parameter index mapping
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx_any)
        missing_names = theta_names[isnothing.(idx_any)]
        error("Theta names not found in model parameters: $missing_names")
    end
    Int.(idx_any)
end

base_parameters = copy(mm_model.parameter_values)

println("  Model:          $(mm_model.model_name)")
println("  Total params:   $(length(mm_model.parameters))")
println("  Estimated idx:  $theta_param_idx")

# ============================================================================
# Step 3: Load Chain Files and Extract Posterior Means
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 3: Loading posterior chains")
println("-" ^ 76)

"""
    extract_posterior_mean(chain_path, theta_names)

Load a chain payload and compute the posterior mean parameter vector.
Returns (theta_mean, ll_post_mean, found) where found indicates if the file exists.
"""
function extract_posterior_mean(chain_path::String, theta_names::Vector{Symbol})
    if !isfile(chain_path)
        println("  WARNING: Chain not found: $chain_path")
        return nothing, NaN, false
    end

    raw = Serialization.deserialize(chain_path)
    chain_payload = raw isa Dict ? Dict{Any,Any}(raw) : Dict{Any,Any}("chain" => raw)

    chain_mat = chain_payload["chain"]  # (n_samples, n_params)
    chain_theta_names = get(chain_payload, "theta_names", theta_names)

    # Compute posterior mean (excluding initial warmup if present)
    n_samples = size(chain_mat, 1)
    burn_in = min(div(n_samples, 4), 500)  # discard first 25% or 500 draws
    theta_mean = vec(mean(chain_mat[burn_in+1:end, :], dims=1))

    ll_stored = get(chain_payload, "ll_post_mean", NaN)

    println("  Loaded chain:   $chain_path")
    println("  Samples:        $n_samples (burn-in=$burn_in)")
    println("  Posterior mean:  $(round.(theta_mean, digits=4))")
    if !isnan(ll_stored)
        println("  Stored LL:      $(round(ll_stored, digits=2))")
    end

    return theta_mean, ll_stored, true
end

theta_linear, ll_linear_stored, have_linear = extract_posterior_mean(linear_chain_path, theta_names)
theta_surrogate, ll_surr_stored, have_surrogate = extract_posterior_mean(surrogate_chain_path, theta_names)

if !have_linear && !have_surrogate
    # Fall back to calibrated baseline
    println("\n  No chains found. Using calibrated baseline parameters.")
    include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
    baseline = get_phase1_18param_baseline()
    theta_linear = Float64[baseline[tn] for tn in theta_names]
    have_linear = true
    ll_linear_stored = NaN
end

# ============================================================================
# Step 4: Build Full Parameter Vectors
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 4: Preparing parameter vectors")
println("-" ^ 76)

function build_full_params(theta_est::Vector{Float64})
    params = copy(base_parameters)
    for (i, idx) in enumerate(theta_param_idx)
        params[idx] = theta_est[i]
    end
    return params
end

params_linear = have_linear ? build_full_params(theta_linear) : nothing
params_surrogate = have_surrogate ? build_full_params(theta_surrogate) : nothing

# ============================================================================
# Step 5: Kalman Filter Baseline
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 5: Kalman filter log-likelihood baseline")
println("-" ^ 76)

obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)

function kalman_loglik(params::Vector{Float64})
    ll_vec = MacroModelling.get_loglikelihood_per_period(
        mm_model,
        obs_data_ka,
        params;
        algorithm = :first_order,
        filter = :kalman,
        on_failure_loglikelihood = -1e12,
        presample_periods = 0,
        initial_covariance = :theoretical,
        verbose = false
    )
    return sum(ll_vec)
end

kalman_ll_linear = NaN
kalman_ll_surrogate = NaN

if params_linear !== nothing
    t0 = time()
    kalman_ll_linear = kalman_loglik(params_linear)
    t_kf = time() - t0
    println("  Kalman LL (linear posterior mean):    $(round(kalman_ll_linear, digits=2))  [$(round(t_kf*1000, digits=1)) ms]")
end

if params_surrogate !== nothing
    t0 = time()
    kalman_ll_surrogate = kalman_loglik(params_surrogate)
    t_kf = time() - t0
    println("  Kalman LL (surrogate posterior mean): $(round(kalman_ll_surrogate, digits=2))  [$(round(t_kf*1000, digits=1)) ms]")
end

# ============================================================================
# Step 6: Load Particle Filter Implementation
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 6: Loading bootstrap particle filter")
println("-" ^ 76)

include(joinpath(repo_root, "src", "particle_filter_bootstrap.jl"))
println("  Loaded: src/particle_filter_bootstrap.jl")

# ============================================================================
# Step 7: Run Particle Filter at Multiple Particle Counts
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 7: Running particle filter benchmark")
println("-" ^ 76)

n_particles_grid = [100, 500, 1000, 5000]
n_seeds = 5  # average over multiple seeds for Monte Carlo variance estimate

# Structure to hold results
mutable struct PFResult
    n_particles::Int
    ll_mean::Float64
    ll_std::Float64
    ess_min_mean::Float64
    n_resamples_mean::Float64
    wall_time::Float64
end

function run_pf_benchmark(params::Vector{Float64}, label::String;
                          n_particles_grid=n_particles_grid, n_seeds=n_seeds)
    println("\n  --- Particle filter: $label ---")
    results = PFResult[]

    for np in n_particles_grid
        lls = Float64[]
        ess_mins = Float64[]
        n_res = Float64[]

        t0 = time()
        for seed in 1:n_seeds
            pf_result = particle_filter_loglik(
                mm_model, obs_data, observables, params;
                n_particles = np,
                seed = seed * 1000 + 42,
                measurement_error = :auto,
                resample_scheme = :systematic,
                resample_threshold = 0.5,
                initial_covariance = :theoretical,
                verbose = false
            )
            push!(lls, pf_result.ll_total)
            push!(ess_mins, minimum(pf_result.ess_per_period))
            push!(n_res, pf_result.n_resamples)
        end
        wall_time = (time() - t0) / n_seeds

        ll_m = mean(lls)
        ll_s = n_seeds > 1 ? std(lls) : 0.0
        ess_min_m = mean(ess_mins)
        n_res_m = mean(n_res)

        push!(results, PFResult(np, ll_m, ll_s, ess_min_m, n_res_m, wall_time))

        @printf("    N=%5d  LL=%.2f +/- %.2f  ESS_min=%.0f  resamples=%.0f  time=%.2fs\n",
                np, ll_m, ll_s, ess_min_m, n_res_m, wall_time)
    end

    return results
end

results_linear = nothing
results_surrogate = nothing

if params_linear !== nothing
    results_linear = run_pf_benchmark(params_linear, "Linear posterior mean")
end

if params_surrogate !== nothing
    results_surrogate = run_pf_benchmark(params_surrogate, "Surrogate posterior mean")
end

# ============================================================================
# Step 7b: Run COPF (Conditionally-Optimal Particle Filter) Benchmark
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 7b: Running COPF benchmark (optimal proposal)")
println("-" ^ 76)

function run_copf_benchmark(params::Vector{Float64}, label::String;
                             n_particles_grid=n_particles_grid, n_seeds=n_seeds)
    println("\n  --- COPF: $label ---")
    results = PFResult[]

    for np in n_particles_grid
        lls = Float64[]
        ess_mins = Float64[]
        n_res = Float64[]

        t0 = time()
        for seed in 1:n_seeds
            pf_result = particle_filter_copf_loglik(
                mm_model, obs_data, observables, params;
                n_particles = np,
                seed = seed * 1000 + 42,
                measurement_error = :auto,
                resample_scheme = :systematic,
                resample_threshold = 0.5,
                initial_covariance = :theoretical,
                verbose = false
            )
            push!(lls, pf_result.ll_total)
            push!(ess_mins, minimum(pf_result.ess_per_period))
            push!(n_res, pf_result.n_resamples)
        end
        wall_time = (time() - t0) / n_seeds

        ll_m = mean(lls)
        ll_s = n_seeds > 1 ? std(lls) : 0.0
        ess_min_m = mean(ess_mins)
        n_res_m = mean(n_res)

        push!(results, PFResult(np, ll_m, ll_s, ess_min_m, n_res_m, wall_time))

        @printf("    N=%5d  LL=%.2f +/- %.2f  ESS_min=%.0f  resamples=%.0f  time=%.2fs\n",
                np, ll_m, ll_s, ess_min_m, n_res_m, wall_time)
    end

    return results
end

results_copf_linear = nothing
results_copf_surrogate = nothing

if params_linear !== nothing
    results_copf_linear = run_copf_benchmark(params_linear, "COPF Linear posterior mean")
end

if params_surrogate !== nothing
    results_copf_surrogate = run_copf_benchmark(params_surrogate, "COPF Surrogate posterior mean")
end

# COPF convergence diagnostic
if results_copf_linear !== nothing && !isnan(kalman_ll_linear)
    println("\n  COPF convergence to Kalman filter (linear posterior mean):")
    for r in results_copf_linear
        gap = r.ll_mean - kalman_ll_linear
        @printf("    N=%5d  COPF-KF gap = %+.2f  (%.4f%% of |KF|)\n",
                r.n_particles, gap, abs(gap / kalman_ll_linear) * 100)
    end
end

# ============================================================================
# Step 8: Print Summary Table
# ============================================================================

println("\n" * "=" ^ 76)
println("SUMMARY TABLE: Particle Filter vs Kalman Filter")
println("=" ^ 76)

# Header
@printf("\n%-10s | %-24s | %-24s | %-12s\n",
        "N_part", "PF LL (linear)", "PF LL (surrogate)", "Wall time")
println("-" ^ 76)

for i in 1:length(n_particles_grid)
    np = n_particles_grid[i]

    ll_lin_str = if results_linear !== nothing
        r = results_linear[i]
        @sprintf("%.2f +/- %.2f", r.ll_mean, r.ll_std)
    else
        "N/A"
    end

    ll_surr_str = if results_surrogate !== nothing
        r = results_surrogate[i]
        @sprintf("%.2f +/- %.2f", r.ll_mean, r.ll_std)
    else
        "N/A"
    end

    wt = if results_linear !== nothing
        @sprintf("%.2fs", results_linear[i].wall_time)
    elseif results_surrogate !== nothing
        @sprintf("%.2fs", results_surrogate[i].wall_time)
    else
        "N/A"
    end

    @printf("%-10d | %-24s | %-24s | %-12s\n", np, ll_lin_str, ll_surr_str, wt)
end

println("-" ^ 76)
@printf("%-10s | %-24s | %-24s |\n",
        "Kalman",
        isnan(kalman_ll_linear) ? "N/A" : @sprintf("%.2f (exact)", kalman_ll_linear),
        isnan(kalman_ll_surrogate) ? "N/A" : @sprintf("%.2f (exact)", kalman_ll_surrogate))
println("=" ^ 76)

# ESS summary
println("\nEffective Sample Size (minimum across periods):")
@printf("%-10s | %-20s | %-20s\n", "N_part", "ESS_min (linear)", "ESS_min (surrogate)")
println("-" ^ 56)
for i in 1:length(n_particles_grid)
    np = n_particles_grid[i]
    ess_lin = results_linear !== nothing ? @sprintf("%.0f", results_linear[i].ess_min_mean) : "N/A"
    ess_surr = results_surrogate !== nothing ? @sprintf("%.0f", results_surrogate[i].ess_min_mean) : "N/A"
    @printf("%-10d | %-20s | %-20s\n", np, ess_lin, ess_surr)
end

# Convergence diagnostic
if results_linear !== nothing && !isnan(kalman_ll_linear)
    println("\nConvergence to Kalman filter — Bootstrap (linear posterior mean):")
    for r in results_linear
        gap = r.ll_mean - kalman_ll_linear
        @printf("  N=%5d  BPF-KF gap = %+.2f  (%.1f%% of |KF|)\n",
                r.n_particles, gap, abs(gap / kalman_ll_linear) * 100)
    end
end

# COPF summary table
println("\n" * "=" ^ 76)
println("SUMMARY TABLE: COPF (Optimal Proposal) vs Kalman Filter")
println("=" ^ 76)

@printf("\n%-10s | %-24s | %-24s | %-12s\n",
        "N_part", "COPF LL (linear)", "COPF LL (surrogate)", "Wall time")
println("-" ^ 76)

for i in 1:length(n_particles_grid)
    np = n_particles_grid[i]

    ll_lin_str = if results_copf_linear !== nothing
        r = results_copf_linear[i]
        @sprintf("%.2f +/- %.2f", r.ll_mean, r.ll_std)
    else
        "N/A"
    end

    ll_surr_str = if results_copf_surrogate !== nothing
        r = results_copf_surrogate[i]
        @sprintf("%.2f +/- %.2f", r.ll_mean, r.ll_std)
    else
        "N/A"
    end

    wt = if results_copf_linear !== nothing
        @sprintf("%.2fs", results_copf_linear[i].wall_time)
    elseif results_copf_surrogate !== nothing
        @sprintf("%.2fs", results_copf_surrogate[i].wall_time)
    else
        "N/A"
    end

    @printf("%-10d | %-24s | %-24s | %-12s\n", np, ll_lin_str, ll_surr_str, wt)
end

println("-" ^ 76)
@printf("%-10s | %-24s | %-24s |\n",
        "Kalman",
        isnan(kalman_ll_linear) ? "N/A" : @sprintf("%.2f (exact)", kalman_ll_linear),
        isnan(kalman_ll_surrogate) ? "N/A" : @sprintf("%.2f (exact)", kalman_ll_surrogate))
println("=" ^ 76)

# COPF ESS summary
println("\nCOPF Effective Sample Size (minimum across periods):")
@printf("%-10s | %-20s | %-20s\n", "N_part", "ESS_min (linear)", "ESS_min (surrogate)")
println("-" ^ 56)
for i in 1:length(n_particles_grid)
    np = n_particles_grid[i]
    ess_lin = results_copf_linear !== nothing ? @sprintf("%.0f", results_copf_linear[i].ess_min_mean) : "N/A"
    ess_surr = results_copf_surrogate !== nothing ? @sprintf("%.0f", results_copf_surrogate[i].ess_min_mean) : "N/A"
    @printf("%-10d | %-20s | %-20s\n", np, ess_lin, ess_surr)
end

if results_copf_linear !== nothing && !isnan(kalman_ll_linear)
    println("\nCOPF convergence to Kalman filter (linear posterior mean):")
    for r in results_copf_linear
        gap = r.ll_mean - kalman_ll_linear
        @printf("  N=%5d  COPF-KF gap = %+.2f  (%.4f%% of |KF|)\n",
                r.n_particles, gap, abs(gap / kalman_ll_linear) * 100)
    end
end

# ============================================================================
# Step 9: Save Results
# ============================================================================

println("\n" * "-" ^ 76)
println("STEP 9: Saving results")
println("-" ^ 76)

mkpath(out_dir)

results_dict = Dict{String, Any}(
    "timestamp" => string(now()),
    "data_path" => data_path,
    "n_obs" => n_obs,
    "T_obs" => T_obs,
    "observables" => observables,
    "theta_names" => theta_names,
    "n_particles_grid" => n_particles_grid,
    "n_seeds" => n_seeds,
    "kalman_ll_linear" => kalman_ll_linear,
    "kalman_ll_surrogate" => kalman_ll_surrogate,
)

if have_linear
    results_dict["theta_linear"] = theta_linear
    results_dict["results_linear"] = [(
        n_particles = r.n_particles,
        ll_mean = r.ll_mean,
        ll_std = r.ll_std,
        ess_min_mean = r.ess_min_mean,
        n_resamples_mean = r.n_resamples_mean,
        wall_time = r.wall_time
    ) for r in results_linear]
end

if have_surrogate
    results_dict["theta_surrogate"] = theta_surrogate
    results_dict["results_surrogate"] = [(
        n_particles = r.n_particles,
        ll_mean = r.ll_mean,
        ll_std = r.ll_std,
        ess_min_mean = r.ess_min_mean,
        n_resamples_mean = r.n_resamples_mean,
        wall_time = r.wall_time
    ) for r in results_surrogate]
end

# Save COPF results
if results_copf_linear !== nothing
    results_dict["results_copf_linear"] = [(
        n_particles = r.n_particles,
        ll_mean = r.ll_mean,
        ll_std = r.ll_std,
        ess_min_mean = r.ess_min_mean,
        n_resamples_mean = r.n_resamples_mean,
        wall_time = r.wall_time
    ) for r in results_copf_linear]
end
if results_copf_surrogate !== nothing
    results_dict["results_copf_surrogate"] = [(
        n_particles = r.n_particles,
        ll_mean = r.ll_mean,
        ll_std = r.ll_std,
        ess_min_mean = r.ess_min_mean,
        n_resamples_mean = r.n_resamples_mean,
        wall_time = r.wall_time
    ) for r in results_copf_surrogate]
end

out_path = joinpath(out_dir, "particle_filter_benchmark_results.jls")
Serialization.serialize(out_path, results_dict)
println("  Saved: $out_path")

# Also save a human-readable text report
report_path = joinpath(out_dir, "particle_filter_benchmark_report.txt")
open(report_path, "w") do io
    println(io, "Particle Filter Benchmark Report")
    println(io, "================================")
    println(io, "Date: $(now())")
    println(io, "Data: $data_path")
    println(io, "Periods: $T_obs, Observables: $n_obs")
    println(io, "")
    println(io, "Kalman LL (linear):    $(round(kalman_ll_linear, digits=4))")
    println(io, "Kalman LL (surrogate): $(round(kalman_ll_surrogate, digits=4))")
    println(io, "")
    println(io, "Particle Filter Results (averaged over $n_seeds seeds):")
    println(io, "")

    @printf(io, "%-10s  %-18s  %-18s  %-10s  %-10s\n",
            "N_part", "LL_mean(lin)", "LL_mean(surr)", "ESS_min", "Time(s)")
    println(io, "-" ^ 70)

    for i in 1:length(n_particles_grid)
        np = n_particles_grid[i]
        ll_lin = results_linear !== nothing ? @sprintf("%.2f +/- %.2f", results_linear[i].ll_mean, results_linear[i].ll_std) : "N/A"
        ll_surr = results_surrogate !== nothing ? @sprintf("%.2f +/- %.2f", results_surrogate[i].ll_mean, results_surrogate[i].ll_std) : "N/A"
        ess_val = results_linear !== nothing ? results_linear[i].ess_min_mean : (results_surrogate !== nothing ? results_surrogate[i].ess_min_mean : NaN)
        wt = results_linear !== nothing ? results_linear[i].wall_time : (results_surrogate !== nothing ? results_surrogate[i].wall_time : NaN)
        @printf(io, "%-10d  %-18s  %-18s  %-10.0f  %-10.2f\n", np, ll_lin, ll_surr, ess_val, wt)
    end

    if results_linear !== nothing && !isnan(kalman_ll_linear)
        println(io, "")
        println(io, "Bootstrap PF convergence to Kalman (linear):")
        for r in results_linear
            gap = r.ll_mean - kalman_ll_linear
            @printf(io, "  N=%5d  gap = %+.2f (%.2f%%)\n",
                    r.n_particles, gap, abs(gap / kalman_ll_linear) * 100)
        end
    end

    # COPF results
    println(io, "")
    println(io, "COPF (Conditionally-Optimal) Results:")
    println(io, "")
    @printf(io, "%-10s  %-18s  %-18s  %-10s  %-10s\n",
            "N_part", "LL_mean(lin)", "LL_mean(surr)", "ESS_min", "Time(s)")
    println(io, "-" ^ 70)

    for i in 1:length(n_particles_grid)
        np = n_particles_grid[i]
        ll_lin = results_copf_linear !== nothing ? @sprintf("%.2f +/- %.2f", results_copf_linear[i].ll_mean, results_copf_linear[i].ll_std) : "N/A"
        ll_surr = results_copf_surrogate !== nothing ? @sprintf("%.2f +/- %.2f", results_copf_surrogate[i].ll_mean, results_copf_surrogate[i].ll_std) : "N/A"
        ess_val = results_copf_linear !== nothing ? results_copf_linear[i].ess_min_mean : (results_copf_surrogate !== nothing ? results_copf_surrogate[i].ess_min_mean : NaN)
        wt = results_copf_linear !== nothing ? results_copf_linear[i].wall_time : (results_copf_surrogate !== nothing ? results_copf_surrogate[i].wall_time : NaN)
        @printf(io, "%-10d  %-18s  %-18s  %-10.0f  %-10.2f\n", np, ll_lin, ll_surr, ess_val, wt)
    end

    if results_copf_linear !== nothing && !isnan(kalman_ll_linear)
        println(io, "")
        println(io, "COPF convergence to Kalman (linear):")
        for r in results_copf_linear
            gap = r.ll_mean - kalman_ll_linear
            @printf(io, "  N=%5d  gap = %+.2f (%.4f%%)\n",
                    r.n_particles, gap, abs(gap / kalman_ll_linear) * 100)
        end
    end
end
println("  Saved: $report_path")

println("\n" * "=" ^ 76)
println("BENCHMARK COMPLETE: $(now())")
println("=" ^ 76)
