#!/usr/bin/env julia
#
# HLT Model Estimation via Kalman Filter + MH Sampling
#
# This script estimates the Smets-Wouters (2007) HLT model using
# the standard linear Kalman filter for likelihood evaluation and
# Metropolis-Hastings random walk for posterior sampling.
#
# This serves as:
# 1. A benchmark for the surrogate-based nonlinear estimation
# 2. A standalone real-data estimation result for the paper
#
# Usage:
#   julia --project=. scripts/hlt_kalman_estimation.jl [options]
#
# Options:
#   --data=<path>          Path to real-data payload (.jls)
#   --out=<path>           Output chain path
#   --n-samples=<int>      Samples per chain (default: 5000)
#   --n-chains=<int>       Number of chains (default: 1)
#   --burn-in=<int>        Burn-in samples (default: 1000)
#   --param-set=<name>     Parameter set: phase1_18params (default) or phase1_18params_narrow
#   --seed=<int>           Random seed (default: 42)
#   --verbose              Print per-iteration diagnostics
#

using Serialization
using LinearAlgebra
using Random
using Printf
import Distributions
using MacroModelling
using AxisKeys

import Statistics: mean, std, median, quantile

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

# ============================================================================
# CLI argument parsing
# ============================================================================

function parse_arg_string(args, flag, default)
    for a in args
        if startswith(a, "$flag=")
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_arg_int(args, flag, default)
    s = parse_arg_string(args, flag, "")
    return s == "" ? default : parse(Int, s)
end

function parse_arg_float(args, flag, default)
    s = parse_arg_string(args, flag, "")
    return s == "" ? default : parse(Float64, s)
end

const DATA_PATH = parse_arg_string(ARGS, "--data",
    joinpath(@__DIR__, "..", ".local_artifacts", "hlt_18param_realdata", "hlt_real_data_payload.jls"))
const OUT_PATH = parse_arg_string(ARGS, "--out",
    joinpath(@__DIR__, "..", ".local_artifacts", "hlt_18param_realdata", "hlt_kalman_mh_chain.jls"))
const N_SAMPLES = parse_arg_int(ARGS, "--n-samples", 5000)
const N_CHAINS = parse_arg_int(ARGS, "--n-chains", 1)
const BURN_IN = parse_arg_int(ARGS, "--burn-in", 1000)
const PARAM_SET = Symbol(parse_arg_string(ARGS, "--param-set", "phase1_18params"))
const SEED = parse_arg_int(ARGS, "--seed", 42)
const VERBOSE = "--verbose" in ARGS

# ============================================================================
# Load data and model
# ============================================================================

println("="^80)
println("HLT KALMAN FILTER ESTIMATION")
println("="^80)

println("\nLoading data from: $DATA_PATH")
payload = deserialize(DATA_PATH)
obs_data = Float64.(payload["obs_data"])
observables = payload["observables"]
theta_names_payload = payload["theta_names"]
theta_baseline = Float64.(payload["theta_baseline"])

d_obs, T = size(obs_data)
println("  Observables: $observables ($d_obs × $T)")

println("\nLoading HLT model...")
mm_model = load_hlt_model(normpath(joinpath(@__DIR__, "..")), "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
obs_ka = KeyedArray(obs_data; Variable=observables, Time=1:T)

# ============================================================================
# Parameter setup
# ============================================================================

theta_names = Symbol.(theta_names_payload)
n_theta = length(theta_names)

param_specs = get_parameter_specs(PARAM_SET)
spec_names = [s.name for s in param_specs]

prior_dists = Vector{Distributions.Distribution}(undef, n_theta)
prior_bounds = Vector{Tuple{Float64,Float64}}(undef, n_theta)
mh_rw_scales = Vector{Float64}(undef, n_theta)
theta_init = Vector{Float64}(undef, n_theta)
baseline = get_phase1_18param_baseline()

for (i, tname) in enumerate(theta_names)
    si = findfirst(==(tname), spec_names)
    if si !== nothing
        spec = param_specs[si]
        if spec.prior_type == :Beta
            prior_dists[i] = Distributions.Beta(spec.prior_params.α, spec.prior_params.β)
        elseif spec.prior_type == :InvGamma
            prior_dists[i] = Distributions.InverseGamma(spec.prior_params.α, spec.prior_params.θ)
        elseif spec.prior_type == :Normal
            prior_dists[i] = Distributions.Normal(spec.prior_params.μ, spec.prior_params.σ)
        elseif spec.prior_type == :Uniform
            prior_dists[i] = Distributions.Uniform(spec.bounds...)
        end
        prior_bounds[i] = spec.bounds
        mh_rw_scales[i] = hasfield(typeof(spec), :mh_scale) ? spec.mh_scale : 0.01
        init_val = get(baseline, tname, Distributions.mean(prior_dists[i]))
        # Clamp to bounds
        lb, ub = prior_bounds[i]
        theta_init[i] = clamp(init_val, lb + 1e-4, ub - 1e-4)
    else
        bv = get(baseline, tname, theta_baseline[i])
        theta_init[i] = bv
        prior_dists[i] = Distributions.Normal(bv, abs(bv) * 0.5 + 0.01)
        prior_bounds[i] = (bv * 0.1, bv * 5.0)
        mh_rw_scales[i] = abs(bv) * 0.02 + 0.001
    end
end

# Adaptive MH step sizes: start with prior std scaled down
for i in 1:n_theta
    try
        prior_std = Distributions.std(prior_dists[i])
        lb, ub = prior_bounds[i]
        range_width = ub - lb
        # Scale for 18-dim: 2.38/sqrt(d) ≈ 0.56 of the marginal proposal std
        # Start small and let adaptation grow
        mh_rw_scales[i] = min(prior_std * 0.05, range_width * 0.01)
    catch
        mh_rw_scales[i] = 0.005
    end
end

println("\nParameter setup ($n_theta parameters, set=$PARAM_SET):")
for i in 1:n_theta
    @printf("  %-12s  init=%.4f  bounds=(%.4f, %.4f)  rw_scale=%.4f  prior=%s\n",
            theta_names[i], theta_init[i], prior_bounds[i]..., mh_rw_scales[i],
            string(typeof(prior_dists[i]).name.name))
end

# ============================================================================
# Log-likelihood and log-prior functions
# ============================================================================

function logprior(theta_vec::Vector{Float64})
    lp = 0.0
    for i in eachindex(theta_vec)
        lb, ub = prior_bounds[i]
        if theta_vec[i] < lb || theta_vec[i] > ub
            return -Inf
        end
        d = Distributions.truncated(prior_dists[i], lb, ub)
        lp += Distributions.logpdf(d, theta_vec[i])
    end
    return lp
end

function loglikelihood(theta_vec::Vector{Float64})
    ll_vec = MacroModelling.linear_model_loglik_per_period(
        mm_model, obs_ka, theta_vec, theta_names;
        model_parameter_names=mm_model.parameters,
        algorithm=:first_order,
        filter=:kalman,
        on_failure_loglikelihood=-1e12,
        verbose=false
    )
    # Check for solver failure
    if any(ll_vec .== -1e12)
        return -1e12
    end
    return sum(ll_vec)
end

function logposterior(theta_vec::Vector{Float64})
    lp = logprior(theta_vec)
    if !isfinite(lp)
        return -Inf
    end
    ll = loglikelihood(theta_vec)
    if !isfinite(ll)
        return -Inf
    end
    return lp + ll
end

# ============================================================================
# Verify at initial point
# ============================================================================

println("\n--- Evaluating at initial point ---")
t0 = time()
ll_init = loglikelihood(theta_init)
lp_init = logprior(theta_init)
elapsed_init = time() - t0
println("  Log-likelihood: $ll_init")
println("  Log-prior:      $lp_init")
println("  Log-posterior:   $(ll_init + lp_init)")
println("  Eval time:       $(round(elapsed_init, digits=3))s")

if !isfinite(ll_init)
    error("Initial point has non-finite log-likelihood. Check parameter values.")
end

# ============================================================================
# MH Random Walk Sampler
# ============================================================================

function run_mh_chain(theta_start::Vector{Float64},
                      n_samples::Int,
                      burn_in::Int,
                      rw_scales::Vector{Float64},
                      rng::AbstractRNG;
                      verbose::Bool=false,
                      adapt_interval::Int=100,
                      target_accept::Float64=0.234)

    n_theta = length(theta_start)
    total = n_samples + burn_in

    # Storage
    chain = Matrix{Float64}(undef, n_theta, n_samples)
    ll_chain = Vector{Float64}(undef, n_samples)
    lp_chain = Vector{Float64}(undef, n_samples)

    # Current state
    theta_curr = copy(theta_start)
    lpost_curr = logposterior(theta_curr)
    ll_curr = loglikelihood(theta_curr)

    # Adaptive step sizes
    scales = copy(rw_scales)
    accept_count = 0
    total_count = 0
    window_accept = 0
    window_count = 0

    println("\nRunning MH chain: $total iterations ($burn_in burn-in + $n_samples samples)")
    t_start = time()

    for iter in 1:total
        # Propose
        theta_prop = theta_curr .+ scales .* randn(rng, n_theta)

        # Evaluate
        lpost_prop = logposterior(theta_prop)

        # Accept/reject
        log_alpha = lpost_prop - lpost_curr
        accepted = false
        if log_alpha >= 0 || log(rand(rng)) < log_alpha
            theta_curr = theta_prop
            lpost_curr = lpost_prop
            ll_curr = loglikelihood(theta_curr)
            accepted = true
            accept_count += 1
            window_accept += 1
        end
        total_count += 1
        window_count += 1

        # Store post-burn-in
        if iter > burn_in
            idx = iter - burn_in
            chain[:, idx] = theta_curr
            ll_chain[idx] = ll_curr
            lp_chain[idx] = lpost_curr - ll_curr  # log-prior
        end

        # Adaptive scaling during burn-in
        if iter <= burn_in && window_count >= adapt_interval
            window_rate = window_accept / window_count
            for j in 1:n_theta
                if window_rate < target_accept - 0.05
                    scales[j] *= 0.8  # shrink step
                elseif window_rate > target_accept + 0.05
                    scales[j] *= 1.2  # grow step
                end
                # Clamp to reasonable range
                lb, ub = prior_bounds[j]
                max_scale = (ub - lb) * 0.5
                scales[j] = clamp(scales[j], 1e-8, max_scale)
            end
            if verbose
                @printf("  iter %5d: accept_rate=%.3f (window), scales_mean=%.6f\n",
                        iter, window_rate, mean(scales))
            end
            window_accept = 0
            window_count = 0
        end

        # Progress
        if iter % 500 == 0
            elapsed = time() - t_start
            rate = iter / elapsed
            eta = (total - iter) / rate
            overall_rate = accept_count / total_count
            @printf("  [%5d/%5d] accept=%.3f  ll=%.1f  rate=%.1f iter/s  ETA=%.0fs\n",
                    iter, total, overall_rate, ll_curr, rate, eta)
        end
    end

    elapsed = time() - t_start
    overall_accept = accept_count / total_count
    println(@sprintf("\nChain complete: %.1fs, accept=%.3f, final_ll=%.1f",
                     elapsed, overall_accept, ll_curr))

    return (chain=chain, ll=ll_chain, lp=lp_chain,
            accept_rate=overall_accept, scales=scales,
            elapsed=elapsed)
end

# ============================================================================
# Run chains
# ============================================================================

Random.seed!(SEED)
results = []

for c in 1:N_CHAINS
    println("\n" * "="^60)
    println("CHAIN $c / $N_CHAINS")
    println("="^60)

    # Perturb initial point slightly for each chain
    rng = MersenneTwister(SEED + c)
    theta_start = copy(theta_init)
    if c > 1
        for i in 1:n_theta
            lb, ub = prior_bounds[i]
            theta_start[i] += mh_rw_scales[i] * 2 * randn(rng)
            theta_start[i] = clamp(theta_start[i], lb, ub)
        end
    end

    result = run_mh_chain(theta_start, N_SAMPLES, BURN_IN, mh_rw_scales, rng;
                          verbose=VERBOSE)
    push!(results, result)
end

# ============================================================================
# Summary statistics
# ============================================================================

println("\n" * "="^80)
println("POSTERIOR SUMMARY")
println("="^80)

# Combine chains
all_chains = hcat([r.chain for r in results]...)
all_ll = vcat([r.ll for r in results]...)

post_mean = vec(mean(all_chains, dims=2))
post_std = vec(std(all_chains, dims=2))
post_q05 = [quantile(all_chains[i, :], 0.05) for i in 1:n_theta]
post_q95 = [quantile(all_chains[i, :], 0.95) for i in 1:n_theta]

@printf("\n%-12s  %10s  %10s  %10s  %10s  %10s\n",
        "Parameter", "Mean", "Std", "Q5", "Q95", "Init")
println("-"^70)
for i in 1:n_theta
    @printf("%-12s  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n",
            theta_names[i], post_mean[i], post_std[i],
            post_q05[i], post_q95[i], theta_init[i])
end

println("\nLog-likelihood summary:")
@printf("  Mean:   %.2f\n", mean(all_ll))
@printf("  Median: %.2f\n", median(all_ll))
@printf("  Max:    %.2f\n", maximum(all_ll))
@printf("  Min:    %.2f\n", minimum(all_ll))

for (c, r) in enumerate(results)
    @printf("\nChain %d: accept=%.3f, elapsed=%.1fs, mean_ll=%.1f\n",
            c, r.accept_rate, r.elapsed, mean(r.ll))
end

# ============================================================================
# Save results
# ============================================================================

output = Dict{String,Any}(
    "chain" => all_chains,
    "ll_chain" => all_ll,
    "theta_names" => theta_names,
    "n_theta" => n_theta,
    "n_samples" => N_SAMPLES,
    "n_chains" => N_CHAINS,
    "burn_in" => BURN_IN,
    "seed" => SEED,
    "param_set" => String(PARAM_SET),
    "post_mean" => post_mean,
    "post_std" => post_std,
    "post_q05" => post_q05,
    "post_q95" => post_q95,
    "ll_mean" => mean(all_ll),
    "ll_max" => maximum(all_ll),
    "accept_rates" => [r.accept_rate for r in results],
    "elapsed_seconds" => [r.elapsed for r in results],
    "theta_init" => theta_init,
    "prior_bounds" => prior_bounds,
    "data_path" => DATA_PATH,
    "method" => "kalman_mh",
    "model" => "Smets_Wouters_2007_HLT",
)

# Also save summary separately
summary = Dict{String,Any}(
    "post_mean_theta" => post_mean,
    "post_std_theta" => post_std,
    "post_q05_theta" => post_q05,
    "post_q95_theta" => post_q95,
    "theta_names" => theta_names,
    "loglik_post_mean" => mean(all_ll),
    "loglik_post_max" => maximum(all_ll),
    "accept_rates" => [r.accept_rate for r in results],
    "n_samples" => N_SAMPLES,
    "n_chains" => N_CHAINS,
    "method" => "kalman_mh",
    "param_set" => String(PARAM_SET),
)

serialize(OUT_PATH, output)
summary_path = replace(OUT_PATH, ".jls" => "_summary.jls")
serialize(summary_path, summary)

println("\nSaved chain to: $OUT_PATH")
println("Saved summary to: $summary_path")
println("\n" * "="^80)
println("ESTIMATION COMPLETE")
println("="^80)
