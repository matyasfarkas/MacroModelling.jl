#!/usr/bin/env julia
# ============================================================================
# LINEAR HMC BASELINE — AdvancedHMC.jl (NUTS + Leapfrog + Finite-Diff)
# ============================================================================
#
# Standalone 18-parameter linear Kalman filter estimation using AdvancedHMC.jl
# directly. Bypasses Turing.jl for full control over the sampler and easier
# debugging.
#
# Key design choices:
#   1. Kalman filter likelihood via MacroModelling.get_loglikelihood_per_period
#   2. Finite-difference gradients (19 evals × ~5ms = ~100ms per gradient)
#      — avoids ForwardDiff Dual numbers through the entire Kalman filter
#   3. Stan-style NUTS with diagonal mass matrix adaptation
#   4. Parameter transforms: unconstrained ↔ constrained via logit/log
#
# Usage:
#   julia --project=. scripts/run_linear_hmc_advancedhmc.jl \
#       --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls \
#       --out=.local_artifacts/hlt_18param_realdata/hlt_linear_hmc_advhmc.jls \
#       --samples=500 --adapt=200 --seed=42
# ============================================================================

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, var, quantile
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

data_path    = parse_kv_string(ARGS, "--data", "")
out_path     = parse_kv_string(ARGS, "--out", "hlt_linear_hmc_advhmc.jls")
n_samples    = parse_kv_int(ARGS, "--samples", 500)
n_adapt      = parse_kv_int(ARGS, "--adapt", 200)
target_accept = parse_kv_float(ARGS, "--target-accept", 0.65)
max_depth    = parse_kv_int(ARGS, "--max-depth", 8)
seed         = parse_kv_int(ARGS, "--seed", 42)
fd_eps       = parse_kv_float(ARGS, "--fd-eps", 1e-5)
verbose      = any(==("--verbose"), ARGS)

# Optional: override fixed MA coefficients (default 0.0 = HLT baseline)
override_cmap  = parse_kv_float(ARGS, "--cmap", NaN)
override_cmaw  = parse_kv_float(ARGS, "--cmaw", NaN)

# Optional warm-start: init θ at the posterior mean of a previous chain
init_from_path = parse_kv_string(ARGS, "--init-from", "")

if data_path == ""
    error("Usage: julia run_linear_hmc_advancedhmc.jl --data=<payload.jls> [--out=...] [--samples=500] [--adapt=200]")
end

Random.seed!(seed)

println("=" ^ 72)
println("LINEAR HMC BASELINE — AdvancedHMC.jl")
println("Started: $(now())")
println("=" ^ 72)
println("  Data:           $data_path")
println("  Output:         $out_path")
println("  Samples:        $n_samples")
println("  Adapt:          $n_adapt")
println("  Target accept:  $target_accept")
println("  Max tree depth: $max_depth")
println("  FD epsilon:     $fd_eps")
println("  Seed:           $seed")

# ============================================================================
# Step 1: Load Data Payload
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 1: Loading data payload")
println("-" ^ 72)

payload = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data     = payload["obs_data"]          # (d_obs, T)
obs_sigma    = payload["obs_sigma"]         # (d_obs,)
theta_true   = get(payload, "theta_true", nothing)
theta_names  = payload["theta_names"]       # Vector{Symbol}
observables  = payload["observables"]       # Vector{Symbol}

d_obs = size(obs_data, 1)
T_obs = size(obs_data, 2)
n_theta = length(theta_names)

println("  Observables:    $observables ($d_obs)")
println("  Periods:        $T_obs")
println("  Parameters:     $theta_names ($n_theta)")
if theta_true !== nothing
    println("  Theta true:     $(round.(theta_true, digits=4))")
else
    println("  Theta true:     (real data — no true values)")
end

# ============================================================================
# Step 2: Load Model (non-OBC linear)
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 2: Loading HLT model (non-OBC, first-order)")
println("-" ^ 72)

repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)

# Build KeyedArray for Kalman filter
obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)

# Parameter index mapping
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx_any)
        missing_names = theta_names[isnothing.(idx_any)]
        error("Theta names not found in model parameters: $missing_names")
    end
    Int.(idx_any)
end

base_parameters = copy(mm_model.parameter_values)

# Override fixed MA coefficients if requested
if !isnan(override_cmap)
    cmap_idx = findfirst(==(:cmap), mm_model.parameters)
    if cmap_idx !== nothing
        base_parameters[cmap_idx] = override_cmap
        println("  Override cmap:  $(override_cmap) (was $(mm_model.parameter_values[cmap_idx]))")
    end
end
if !isnan(override_cmaw)
    cmaw_idx = findfirst(==(:cmaw), mm_model.parameters)
    if cmaw_idx !== nothing
        base_parameters[cmaw_idx] = override_cmaw
        println("  Override cmaw:  $(override_cmaw) (was $(mm_model.parameter_values[cmaw_idx]))")
    end
end

println("  Model:          $(mm_model.model_name)")
println("  Total params:   $(length(mm_model.parameters))")
println("  Estimated idx:  $theta_param_idx")

# ============================================================================
# Step 3: Setup Priors and Transforms
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 3: Setting up priors and parameter transforms")
println("-" ^ 72)

include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
specs = get_phase1_18param_specs()
spec_names = [s.name for s in specs]

# Build prior distributions and bounds aligned to theta_names ordering
prior_dists  = Vector{Distribution}(undef, n_theta)
prior_bounds = Vector{Tuple{Float64,Float64}}(undef, n_theta)

for (i, tname) in enumerate(theta_names)
    si = findfirst(==(tname), spec_names)
    if si === nothing
        error("No prior specification found for parameter: $tname")
    end
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

# Constrained → unconstrained transforms
# Beta/bounded: logit transform   x_unc = log((x - lb) / (ub - x))
# InvGamma:     log transform     x_unc = log(x - lb)
# Normal:       logit on bounds   x_unc = log((x - lb) / (ub - x))

function constrained_to_unconstrained(θ::Vector{Float64})
    x = similar(θ)
    for i in 1:length(θ)
        lb, ub = prior_bounds[i]
        # Clamp to interior of bounds
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

# Log-Jacobian of the transform (for density correction)
function log_jacobian(x::Vector{Float64})
    lj = 0.0
    for i in 1:length(x)
        lb, ub = prior_bounds[i]
        s = 1.0 / (1.0 + exp(-x[i]))
        # d(θ)/d(x) = (ub - lb) * s * (1 - s)
        lj += log(ub - lb) + log(s) + log(1.0 - s)
    end
    return lj
end

# ============================================================================
# Step 4: Log-Density Function (Kalman + Prior)
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 4: Building log-density function")
println("-" ^ 72)

# Kalman filter log-likelihood for a given θ (constrained)
function kalman_loglik(θ_constrained::Vector{Float64})
    ll_vec = MacroModelling.linear_model_loglik_per_period(
        mm_model,
        obs_data_ka,
        θ_constrained,
        theta_names;
        model_parameter_names = mm_model.parameters,
        base_parameters       = base_parameters,
        theta_idx             = theta_param_idx,
        algorithm             = :first_order,
        filter                = :kalman,
        on_failure_loglikelihood = -1e12,
        presample_periods     = 0,
        initial_covariance    = :theoretical,
        verbose               = false,
        theta_label           = "Theta",
    )
    return sum(ll_vec)
end

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

# Full log-density in UNCONSTRAINED space (what HMC samples)
function log_density_unconstrained(x::Vector{Float64})
    θ = unconstrained_to_constrained(x)
    lp = log_prior(θ)
    if !isfinite(lp)
        return -Inf
    end
    ll = kalman_loglik(θ)
    if !isfinite(ll)
        return -Inf
    end
    lj = log_jacobian(x)
    return ll + lp + lj
end

# Finite-difference gradient
function fd_gradient!(∇f::Vector{Float64}, f::Function, x::Vector{Float64}, h::Float64)
    f0 = f(x)
    @inbounds for i in 1:length(x)
        x_old = x[i]
        x[i] = x_old + h
        fp = f(x)
        x[i] = x_old
        ∇f[i] = (fp - f0) / h
    end
    return f0
end

# Initialize at prior mode (well inside bounds, not at calibrated boundary)
# Calibrated baseline has crhoa=0.9977, crhopinf=0.0, etc. at bounds — bad for logit
baseline = get_phase1_18param_baseline()
θ_calib = Float64[get(baseline, tname, NaN) for tname in theta_names]
any(isnan, θ_calib) && error("Missing baseline value for some parameters")

# Build init: warm-start from --init-from posterior mean, else blended calib/prior.
θ_init = similar(θ_calib)
if init_from_path != ""
    println("  Loading initialization from: $init_from_path")
    init_chain = deserialize(init_from_path)
    θ_src = if haskey(init_chain, "theta_post_mean")
        Float64.(init_chain["theta_post_mean"])
    elseif haskey(init_chain, "chain") && init_chain["chain"] isa AbstractMatrix
        Float64.(vec(mean(init_chain["chain"], dims=1)))
    else
        error("Cannot extract init from $init_from_path: no theta_post_mean or chain key")
    end
    length(θ_src) == n_theta || error("init-from chain has $(length(θ_src)) θ, expected $n_theta")
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        θ_init[i] = clamp(θ_src[i], lb + 0.01*(ub-lb), ub - 0.01*(ub-lb))
    end
    println("  Init source: posterior mean from $(init_from_path)")
else
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        # Prior mode (approximate: use mean for Beta, mode for InvGamma)
        prior_mode = if specs[findfirst(==(theta_names[i]), spec_names)].prior_type == :Beta
            α = specs[findfirst(==(theta_names[i]), spec_names)].prior_params.α
            β = specs[findfirst(==(theta_names[i]), spec_names)].prior_params.β
            α / (α + β)  # prior mean
        elseif specs[findfirst(==(theta_names[i]), spec_names)].prior_type == :InvGamma
            specs[findfirst(==(theta_names[i]), spec_names)].prior_params.θ  # scale ≈ mode
        elseif specs[findfirst(==(theta_names[i]), spec_names)].prior_type == :Normal
            specs[findfirst(==(theta_names[i]), spec_names)].prior_params.μ
        else
            (lb + ub) / 2
        end
        # Clamp calibrated value into interior of bounds
        calib_clamped = clamp(θ_calib[i], lb + 0.05*(ub-lb), ub - 0.05*(ub-lb))
        # Average of calibrated (clamped) and prior mode
        θ_init[i] = 0.5 * calib_clamped + 0.5 * prior_mode
        θ_init[i] = clamp(θ_init[i], lb + 0.05*(ub-lb), ub - 0.05*(ub-lb))
    end
end

println("  Initial values (blended calib/prior):")
for i in 1:n_theta
    @printf("    %-12s  calib=%.4f  init=%.4f  bounds=(%.2f, %.2f)\n",
            theta_names[i], θ_calib[i], θ_init[i], prior_bounds[i]...)
end

println("\n  Testing Kalman likelihood at init point...")
t0 = time()
ll_init = kalman_loglik(θ_init)
t_kalman = time() - t0
println("  Kalman LL at init: $(round(ll_init, digits=2)) ($(round(t_kalman*1000, digits=1)) ms)")

# Also test at calibrated baseline for reference
ll_baseline = kalman_loglik(clamp.(θ_calib, [b[1]+1e-6 for b in prior_bounds], [b[2]-1e-6 for b in prior_bounds]))
println("  Kalman LL at calib baseline: $(round(ll_baseline, digits=2))")

lp_init = log_prior(θ_init)
println("  Log prior at init: $(round(lp_init, digits=2))")

x_init = constrained_to_unconstrained(θ_init)
lj_init = log_jacobian(x_init)
println("  Log Jacobian at init: $(round(lj_init, digits=2))")

ld_init = log_density_unconstrained(x_init)
println("  Total log-density: $(round(ld_init, digits=2))")

# Test gradient
println("\n  Testing finite-difference gradient...")
∇f = zeros(n_theta)
t0 = time()
fd_gradient!(∇f, log_density_unconstrained, copy(x_init), fd_eps)
t_grad = time() - t0
println("  Gradient norm: $(round(norm(∇f), digits=4)) ($(round(t_grad*1000, digits=1)) ms)")
println("  Gradient:      $(round.(∇f, digits=3))")
all(isfinite, ∇f) || error("Gradient contains non-finite values!")

# Estimate per-eval cost
cost_per_eval_ms = t_kalman * 1000
cost_per_grad_ms = t_grad * 1000
est_leapfrog_per_draw = 2^max_depth  # worst case for NUTS
est_time_per_draw_s = est_leapfrog_per_draw * cost_per_grad_ms / 1000
est_total_hours = (n_samples + n_adapt) * est_time_per_draw_s / 3600

println("\n  Timing estimates:")
println("  Kalman eval:        $(round(cost_per_eval_ms, digits=1)) ms")
println("  Gradient (FD, $n_theta evals): $(round(cost_per_grad_ms, digits=1)) ms")
println("  Est. per NUTS draw: $(round(est_time_per_draw_s, digits=1)) s (worst case, depth=$max_depth)")
println("  Est. total:         $(round(est_total_hours, digits=1)) hours (worst case)")

# ============================================================================
# Step 5: AdvancedHMC Setup
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 5: Setting up AdvancedHMC NUTS sampler")
println("-" ^ 72)

# LogDensityProblems interface
struct KalmanLogDensity
    dim::Int
    fd_eps::Float64
end

LogDensityProblems.logdensity(p::KalmanLogDensity, x::AbstractVector) =
    log_density_unconstrained(Vector{Float64}(x))

LogDensityProblems.dimension(p::KalmanLogDensity) = p.dim

LogDensityProblems.capabilities(::Type{KalmanLogDensity}) =
    LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::KalmanLogDensity, x::AbstractVector)
    xv = Vector{Float64}(x)
    ∇f = zeros(p.dim)
    lp = fd_gradient!(∇f, log_density_unconstrained, xv, p.fd_eps)
    return lp, ∇f
end

log_density_obj = KalmanLogDensity(n_theta, fd_eps)

# Unit Euclidean metric — step-size-only adaptation to avoid mass matrix blow-up.
# The logit transform provides baseline scaling; the step size adaptor tunes from there.
metric = UnitEuclideanMetric(n_theta)
println("  Mass matrix: unit Euclidean (step-size-only adaptation)")

# Build sampler
hamiltonian = Hamiltonian(metric, log_density_obj)

println("\n  Finding initial step size...")
initial_ϵ = find_good_stepsize(hamiltonian, x_init)
println("  Initial step size: $(round(initial_ϵ, sigdigits=3))")

integrator = Leapfrog(initial_ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=max_depth)))
# Step-size-only adaptation (no mass matrix adaptation)
adaptor = StepSizeAdaptor(target_accept, integrator)

# ============================================================================
# Step 6: Run NUTS
# ============================================================================

println("\n" * "=" ^ 72)
println("RUNNING NUTS SAMPLER ($n_samples draws + $n_adapt warmup)")
println("=" ^ 72)

t_start = time()
samples_unc, stats = sample(hamiltonian, kernel, x_init, n_samples + n_adapt, adaptor, n_adapt;
                             progress=true, verbose=verbose)
t_elapsed = time() - t_start

println("\n  Sampling complete! Elapsed: $(round(t_elapsed/60, digits=1)) minutes")
println("  Draws/sec: $(round((n_samples + n_adapt) / t_elapsed, digits=2))")

# ============================================================================
# Step 7: Post-processing
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 7: Post-processing")
println("-" ^ 72)

# Transform back to constrained space
θ_chain = hcat([unconstrained_to_constrained(s) for s in samples_unc]...)'  # (n_total, n_theta)
θ_post = θ_chain[(n_adapt+1):end, :]  # discard warmup

# Diagnostics
acceptance_rates = [s.acceptance_rate for s in stats[(n_adapt+1):end]]
step_sizes = [s.step_size for s in stats[(n_adapt+1):end]]
tree_depths = [s.tree_depth for s in stats[(n_adapt+1):end]]
n_divergent = sum(s.numerical_error for s in stats[(n_adapt+1):end])

println("\n  NUTS Diagnostics:")
println("  Mean acceptance rate: $(round(mean(acceptance_rates), digits=3))")
println("  Final step size:     $(round(mean(step_sizes[end-min(50,length(step_sizes)-1):end]), sigdigits=3))")
println("  Mean tree depth:     $(round(mean(tree_depths), digits=1))")
println("  Max tree depth:      $(maximum(tree_depths))")
println("  Divergences:         $n_divergent / $n_samples")

# Compute log-likelihood at posterior mean
θ_post_mean = vec(mean(θ_post, dims=1))
ll_post_mean = kalman_loglik(θ_post_mean)
lp_post_mean = log_prior(θ_post_mean)

println("\n  Posterior mean log-likelihood: $(round(ll_post_mean, digits=2))")
println("  Posterior mean log-prior:     $(round(lp_post_mean, digits=2))")

# Parameter summary
has_true = theta_true !== nothing && length(theta_true) == n_theta
if has_true
    println("\n  " * "-" ^ 68)
    @printf("  %-12s %8s %8s %8s %8s %8s %8s\n",
            "Parameter", "True", "Mean", "Std", "Q2.5", "Q97.5", "Cover")
    println("  " * "-" ^ 68)
else
    println("\n  " * "-" ^ 60)
    @printf("  %-12s %8s %8s %8s %8s %8s\n",
            "Parameter", "Calib", "Mean", "Std", "Q2.5", "Q97.5")
    println("  " * "-" ^ 60)
end

coverage_flags = Bool[]
for i in 1:n_theta
    post_mean = mean(θ_post[:, i])
    post_std = std(θ_post[:, i])
    q025 = quantile(θ_post[:, i], 0.025)
    q975 = quantile(θ_post[:, i], 0.975)

    if has_true
        true_val = theta_true[i]
        covered = q025 < true_val < q975
        push!(coverage_flags, covered)
        @printf("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f %5s\n",
                theta_names[i], true_val, post_mean, post_std, q025, q975,
                covered ? "yes" : "NO")
    else
        @printf("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f\n",
                theta_names[i], θ_init[i], post_mean, post_std, q025, q975)
    end
end
if has_true
    coverage_count = count(identity, coverage_flags)
    println("  " * "-" ^ 68)
    println("  Coverage: $coverage_count / $n_theta ($(round(100*coverage_count/n_theta, digits=0))%)")
else
    println("  " * "-" ^ 60)
end

# Simple ESS estimate (batch means)
function ess_batch_means(chain::Vector{Float64}; batch_size::Int=50)
    n = length(chain)
    if n < 2 * batch_size
        return Float64(n)  # can't estimate well
    end
    n_batches = n ÷ batch_size
    batch_means = [mean(chain[(i-1)*batch_size+1 : i*batch_size]) for i in 1:n_batches]
    var_total = var(chain)
    var_batch = var(batch_means)
    if var_batch < 1e-20
        return Float64(n)
    end
    return n * var_total / (batch_size * var_batch)
end

println("\n  Effective Sample Size (ESS):")
for i in 1:n_theta
    ess = ess_batch_means(θ_post[:, i])
    @printf("    %-12s  ESS = %.0f  (%.1f%%)\n", theta_names[i], ess, 100*ess/n_samples)
end

# ============================================================================
# Step 8: Save Results
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 8: Saving results")
println("-" ^ 72)

results = Dict{String,Any}(
    "chain"              => θ_post,
    "chain_full"         => θ_chain,
    "samples_unc"        => samples_unc,
    "stats"              => stats,
    "theta_names"        => theta_names,
    "theta_true"         => theta_true,
    "theta_post_mean"    => θ_post_mean,
    "theta_init"         => θ_init,
    "prior_bounds"       => prior_bounds,
    "n_samples"          => n_samples,
    "n_adapt"            => n_adapt,
    "target_accept"      => target_accept,
    "max_depth"          => max_depth,
    "seed"               => seed,
    "fd_eps"             => fd_eps,
    "elapsed_seconds"    => t_elapsed,
    "ll_post_mean"       => ll_post_mean,
    "lp_post_mean"       => lp_post_mean,
    "n_divergent"        => n_divergent,
    "acceptance_rates"   => acceptance_rates,
    "step_sizes"         => step_sizes,
    "tree_depths"        => tree_depths,
    "observables"        => observables,
    "T_obs"              => T_obs,
    "model_name"         => "Smets_Wouters_2007_HLT",
    "sampler"            => "NUTS (AdvancedHMC.jl)",
    "gradient"           => "finite_differences",
    "timestamp"          => string(now()),
)

mkpath(dirname(out_path))
serialize(out_path, results)
println("  Saved: $out_path")

println("\n" * "=" ^ 72)
println("LINEAR HMC BASELINE COMPLETE")
println("Finished: $(now())")
println("Elapsed:  $(round(t_elapsed/60, digits=1)) minutes")
println("=" ^ 72)
