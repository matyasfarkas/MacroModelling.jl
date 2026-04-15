"""
HLT Model Estimation with Simple Regime-Switching (Real Implementation)

This script estimates the Smets-Wouters 2007 HLT model using a simplified
regime-switching approach that computes TOTAL likelihoods (not per-period)
for ROM and FOM, then uses parameter-dependent mixing.

Sample period: 1960Q1-2027Q2 (244 observations)
Estimated parameters: cprobp, cindp, curvp (price Phillips curve)
"""

using MacroModelling
using Zygote
import Turing, Pigeons
import Turing: NUTS, sample
import ADTypes: AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
using Dates
using Serialization

# Load the HLT model
include(joinpath(@__DIR__, "../models/Smets_Wouters_2007_HLT_obc.jl"))

println("="^80)
println("HLT Simple Regime-Switching Estimation")
println("Sample: 1960Q1-2027Q2 (244 observations)")
println("Date: ", Dates.now())
println("="^80)

# ============================================================================
# Data Loading
# ============================================================================

println("\n[1/7] Loading U.S. macro data...")

dat = CSV.read(joinpath(@__DIR__, "data", "usmodel_update.csv"), DataFrame)
data_full = KeyedArray(Array(dat)', Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

observables_csv = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
sample_idx = 47:290  # 1960Q1-2027Q2
data = data_full(observables_csv, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data = rekey(data, :Variable => observables)

println("  ✓ Loaded $(length(sample_idx)) observations")

# ============================================================================
# Parameter Setup
# ============================================================================

println("\n[2/7] Setting up parameters...")

hlt_param_names = string.(Smets_Wouters_2007_HLT_obc.parameters)
base_values = copy(Smets_Wouters_2007_HLT_obc.parameter_values)

# Estimate only price Phillips curve parameters
est_names = ["cprobp", "cindp", "curvp"]
est_idx = map(name -> findfirst(==(name), hlt_param_names), est_names)
@assert all(!isnothing, est_idx) "Missing parameters in model"
est_idx = Int.(est_idx)

init_params = base_values[est_idx]
println("  ✓ Estimating: ", join(est_names, ", "))
println("  ✓ Initial: cprobp=$(init_params[1]), cindp=$(init_params[2]), curvp=$(init_params[3])")

# Priors (from HLT 2016)
dists = [
    Beta(0.5, 0.10, 0.5, 0.95, μσ = true),   # cprobp
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),  # cindp
    Normal(75.0, 25.0, 1.0, 150.0)           # curvp
]

# ============================================================================
# Simple Regime-Switching Likelihood
# ============================================================================

println("\n[3/7] Defining simple regime-switching likelihood...")

"""
Simple Turing model with regime-switching using TOTAL likelihoods.

Computes ROM (first-order Kalman) and FOM (second-order inversion) total
log-likelihoods, then mixes them using a parameter-dependent weight.
"""
Turing.@model function SW07_HLT_simple_switching(
    data,
    m,
    observables,
    base_values,
    est_idx
)
    # Sample parameters from priors
    all_params ~ Turing.arraydist(dists)
    cprobp, cindp, curvp = all_params

    # Skip likelihood in prior sampling
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        # Build full parameter vector
        params = map(eachindex(base_values)) do i
            if i == est_idx[1]
                cprobp
            elseif i == est_idx[2]
                cindp
            elseif i == est_idx[3]
                curvp
            else
                base_values[i]
            end
        end

        # Compute ROM (first-order Kalman) TOTAL log-likelihood
        ll_rom = get_loglikelihood(
            m,
            data(observables),
            params,
            presample_periods = 4,
            initial_covariance = :diagonal,
            algorithm = :first_order,
            filter = :kalman,
        )

        # Compute FOM (second-order inversion) TOTAL log-likelihood
        # Note: This may be unstable for some parameter values
        ll_fom = try
            get_loglikelihood(
                m,
                data(observables),
                params,
                presample_periods = 4,
                initial_covariance = :diagonal,
                algorithm = :pruned_second_order,
                filter = :inversion,
            )
        catch e
            # If inversion fails (common with second-order), use ROM likelihood with penalty
            println("  Warning: FOM likelihood failed, using ROM with penalty")
            ll_rom - 100.0
        end

        # Simple parameter-dependent mixing weight
        # Use more FOM when curvp is large (more nonlinearity)
        # and when cprobp is low (less price stickiness → more ZLB binding)
        fom_weight = 0.25 + 0.5 * (1.0 - cprobp) + 0.25 * min(curvp / 100.0, 1.0)
        fom_weight = clamp(fom_weight, 0.0, 1.0)

        # Mix likelihoods using log-sum-exp trick for numerical stability
        # log(w*exp(ll_fom) + (1-w)*exp(ll_rom))
        if fom_weight > 0.999
            llh = ll_fom
        elseif fom_weight < 0.001
            llh = ll_rom
        else
            # Log-sum-exp: log(a*exp(x) + b*exp(y)) = max(x,y) + log(a*exp(x-max) + b*exp(y-max))
            max_ll = max(ll_fom, ll_rom)
            llh = max_ll + log(fom_weight * exp(ll_fom - max_ll) + (1.0 - fom_weight) * exp(ll_rom - max_ll))
        end

        Turing.@addlogprob! llh
    end
end

println("  ✓ Simple regime-switching model defined")
println("    - ROM: First-order Kalman filter (TOTAL likelihood)")
println("    - FOM: Second-order inversion filter (TOTAL likelihood)")
println("    - Mixing: Parameter-dependent weight (curvp, cprobp)")

# ============================================================================
# Mode Finding
# ============================================================================

println("\n[4/7] Finding posterior mode...")

function find_mode(; verbose::Bool = true)
    Random.seed!(42)

    loglik = SW07_HLT_simple_switching(
        data,
        Smets_Wouters_2007_HLT_obc,
        observables,
        base_values,
        est_idx,
    )

    verbose && println("  Running optimization...")

    mode = Turing.maximum_a_posteriori(
        loglik,
        Optim.NelderMead(),
        initial_params = init_params
    )

    if verbose
        println("  ✓ Mode found:")
        println("    cprobp = $(round(mode.values[1], digits=4))")
        println("    cindp  = $(round(mode.values[2], digits=4))")
        println("    curvp  = $(round(mode.values[3], digits=2))")
        println("    Log-posterior = $(round(mode.lp, digits=2))")
    end

    return mode
end

mode_result = find_mode()

# ============================================================================
# MCMC Sampling
# ============================================================================

println("\n[5/7] Running MCMC sampling...")

function run_mcmc(mode_init; n_samples::Int = 2000, verbose::Bool = true)
    Random.seed!(123)

    loglik = SW07_HLT_simple_switching(
        data,
        Smets_Wouters_2007_HLT_obc,
        observables,
        base_values,
        est_idx,
    )

    verbose && println("\n  Sampling $(n_samples) draws with simple regime-switching...")
    verbose && println("  Using NUTS with AutoZygote differentiation...")

    start_time = time()
    samps = Turing.sample(
        loglik,
        NUTS(adtype = AutoZygote()),
        n_samples,
        progress = true,
        initial_params = mode_init.values
    )
    elapsed = time() - start_time

    if verbose
        println("\n  ✓ Sampling complete ($(round(elapsed, digits=1))s)")
        println("\n  Posterior means:")
        means = mean(samps).nt.mean
        println("    cprobp = $(round(means[1], digits=4))")
        println("    cindp  = $(round(means[2], digits=4))")
        println("    curvp  = $(round(means[3], digits=2))")

        println("\n  MCMC Diagnostics:")
        summary_stats = summarystats(samps)
        println(summary_stats)
    end

    return samps, elapsed
end

samples, elapsed = run_mcmc(mode_result, n_samples = 2000)

# ============================================================================
# Save Results
# ============================================================================

println("\n[6/7] Saving results...")

results_dir = joinpath(@__DIR__, "..", ".local_artifacts", "hlt_estimation_switching_simple_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(results_dir)

# Save chain
chain_path = joinpath(results_dir, "mcmc_chain.jls")
serialize(chain_path, samples)

# Save summary
summary_path = joinpath(results_dir, "estimation_summary.jls")
summary = Dict(
    "model" => "Smets_Wouters_2007_HLT_obc",
    "sample_period" => "1960Q1-2027Q2",
    "n_observations" => length(sample_idx),
    "sample_idx" => sample_idx,
    "estimated_params" => est_names,
    "mode" => Dict(zip(est_names, mode_result.values)),
    "mode_logposterior" => mode_result.lp,
    "posterior_means" => Dict(zip(est_names, mean(samples).nt.mean)),
    "n_samples" => 2000,
    "elapsed_seconds" => elapsed,
    "filter" => "simple_regime_switching (parameter-dependent mixing of ROM and FOM total likelihoods)",
    "timestamp" => string(now()),
)
serialize(summary_path, summary)

# Save CSV
csv_path = joinpath(results_dir, "posterior_estimates.csv")
posterior_df = DataFrame(samples)
CSV.write(csv_path, posterior_df)

println("  ✓ Results saved to: $(results_dir)")
println("    - MCMC chain: mcmc_chain.jls")
println("    - Summary: estimation_summary.jls")
println("    - CSV: posterior_estimates.csv")

println("\n" * "="^80)
println("Simple Regime-Switching Estimation Complete!")
println("="^80)
println("\nResults directory: $(results_dir)")
println("="^80)
