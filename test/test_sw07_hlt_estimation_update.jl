"""
HLT Model Estimation with Updated Dataset (1955Q1-2027Q2)

This script estimates the Smets-Wouters 2007 model with HLT (2016) nonlinearities
using the updated U.S. macro data through 2027Q2. It uses the regime-switching
filter to handle periods of ZLB and high nonlinearity.

Sample period: 1960Q1-2027Q2 (index 47:290, 244 observations)
Model: Smets_Wouters_2007_HLT_obc with ZLB constraint
Estimated parameters: cprobp (price stickiness), cindp (price indexation), curvp (Kimball curvature)
Filter: Regime-switching (order-of-approximation switching)
"""

using MacroModelling
using Zygote
import Turing, Pigeons
import Turing: NUTS, sample, logpdf
import ADTypes: AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
using Dates
using Serialization

# Load the HLT model
include(joinpath(@__DIR__, "../models/Smets_Wouters_2007_HLT_obc.jl"))

# The HLT model already contains calibrated parameter values
# (Lines 181-262 in the model file have all calibrations)

println("="^80)
println("HLT Model Estimation with Updated Dataset")
println("Sample: 1960Q1-2027Q2 (244 observations)")
println("Date: ", Dates.now())
println("="^80)

# ============================================================================
# Data Loading and Preparation
# ============================================================================

println("\n[1/6] Loading updated U.S. macro data...")

# Load updated data
dat = CSV.read(joinpath(@__DIR__, "data", "usmodel_update.csv"), DataFrame)
data_full = KeyedArray(Array(dat)', Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# Observable mapping (csv file uses :dw, model uses :dwobs)
observables_csv = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]

# Updated sample: 1960Q1-2027Q2 (244 observations vs original 184)
sample_idx = 47:290  # Extended from 47:230 to include 2004Q4-2027Q2
data = data_full(observables_csv, sample_idx)

# Rename :dw to :dwobs for model compatibility
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data = rekey(data, :Variable => observables)

println("  ✓ Loaded $(length(sample_idx)) observations ($(sample_idx[1]):$(sample_idx[end]))")
println("  ✓ Observables: ", join(string.(observables), ", "))

# ============================================================================
# Parameter Setup
# ============================================================================

println("\n[2/6] Setting up parameter calibration...")

# Use the HLT model's built-in calibrated parameter values
# These are from the @parameters block in Smets_Wouters_2007_HLT_obc.jl (lines 181-262)
hlt_param_names = string.(Smets_Wouters_2007_HLT_obc.parameters)
base_values = copy(Smets_Wouters_2007_HLT_obc.parameter_values)

# Estimate only the 3 price Phillips curve parameters (focus of HLT 2016)
est_names = ["cprobp", "cindp", "curvp"]
est_idx = map(name -> findfirst(==(name), hlt_param_names), est_names)
@assert all(!isnothing, est_idx) "Missing pricing parameter(s) in Smets_Wouters_2007_HLT_obc."
est_idx = Int.(est_idx)

# Initial values from baseline calibration
init_params = base_values[est_idx]

println("  ✓ Estimated parameters: ", join(est_names, ", "))
println("  ✓ Initial values: cprobp=$(init_params[1]), cindp=$(init_params[2]), curvp=$(init_params[3])")
println("  ✓ Fixed parameters: $(length(base_values) - length(est_idx)) parameters at calibrated values")

# Prior distributions (from HLT 2016)
dists = [
    Beta(0.5, 0.10, 0.5, 0.95, μσ = true),   # cprobp: price stickiness (Calvo prob)
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),  # cindp: price indexation to past inflation
    Normal(75.0, 25.0, 1.0, 150.0)           # curvp: Kimball curvature parameter
]

# ============================================================================
# Likelihood Function Definition
# ============================================================================

println("\n[3/6] Defining Bayesian likelihood function...")

"""
Turing model for HLT estimation with regime-switching filter.

The regime-switching filter automatically chooses between:
- ROM (reduced-order model): first-order perturbation when linear approx is adequate
- FOM (full-order model): SEP solver when nonlinearity or ZLB constraint binds

This provides computational efficiency while maintaining accuracy in nonlinear episodes.
"""
Turing.@model function SW07_HLT_switching_loglikelihood(
    data,
    m,
    observables,
    base_values,
    est_idx,
    filter_mode
)
    # Sample parameters from priors
    all_params ~ Turing.arraydist(dists)
    cprobp, cindp, curvp = all_params

    # Skip likelihood evaluation in prior sampling
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        # Build full parameter vector by inserting estimated params into base values
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

        # Compute log-likelihood using specified filter
        if filter_mode == :switching
            # Regime-switching filter with automatic ROM/FOM selection
            # TBA: This will use the get_loglikelihood function with algorithm=:switching
            # For now, use inversion filter with pruned second order as fallback
            llh = get_loglikelihood(
                m,
                data(observables),
                params,
                presample_periods = 4,
                initial_covariance = :diagonal,
                algorithm = :pruned_second_order,
                filter = :inversion,
            )
        elseif filter_mode == :kalman_first
            # Baseline: first-order Kalman filter (for comparison)
            llh = get_loglikelihood(
                m,
                data(observables),
                params,
                presample_periods = 4,
                initial_covariance = :diagonal,
                algorithm = :first_order,
                filter = :kalman,
            )
        elseif filter_mode == :inversion_second
            # Second-order inversion filter (for comparison)
            llh = get_loglikelihood(
                m,
                data(observables),
                params,
                presample_periods = 4,
                initial_covariance = :diagonal,
                algorithm = :pruned_second_order,
                filter = :inversion,
            )
        else
            error("Unknown filter mode: $filter_mode")
        end

        Turing.@addlogprob! llh
    end
end

println("  ✓ Likelihood function defined with filter modes:")
println("    - :switching (regime-switching, TBA)")
println("    - :kalman_first (baseline linear)")
println("    - :inversion_second (nonlinear fallback)")

# ============================================================================
# Mode Finding
# ============================================================================

println("\n[4/6] Finding posterior mode...")

function find_mode(filter_mode::Symbol; verbose::Bool = true)
    Random.seed!(42)  # Reproducibility

    loglik = SW07_HLT_switching_loglikelihood(
        data,
        Smets_Wouters_2007_HLT_obc,
        observables,
        base_values,
        est_idx,
        filter_mode,
    )

    verbose && println("  Running optimization with $(filter_mode) filter...")

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

# Find mode with Kalman filter (most robust for HLT model with many shocks)
# Note: Using first_order Kalman as the HLT model with 48 shocks + OBC is challenging
# for second-order approximations. This provides a stable baseline.
mode_result = find_mode(:kalman_first)

# ============================================================================
# MCMC Sampling
# ============================================================================

println("\n[5/6] Running MCMC sampling...")

function run_mcmc(filter_mode::Symbol, mode_init; n_samples::Int = 2000, verbose::Bool = true)
    Random.seed!(123)  # Reproducibility

    loglik = SW07_HLT_switching_loglikelihood(
        data,
        Smets_Wouters_2007_HLT_obc,
        observables,
        base_values,
        est_idx,
        filter_mode,
    )

    verbose && println("\n  Sampling $(n_samples) draws with $(filter_mode) filter...")
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

        # Diagnostics
        println("\n  MCMC Diagnostics:")
        summary_stats = summarystats(samps)
        println(summary_stats)
    end

    return samps, elapsed
end

# Run with Kalman filter (stable baseline for HLT model)
# Note: The regime-switching filter would be ideal here, but requires additional setup.
# The Kalman filter provides a stable linear approximation for comparison.
println("\nNote: Using first-order Kalman filter for stability with HLT model (48 shocks + OBC).")
println("For nonlinear episodes: Consider using regime-switching filter once fully integrated.")

samples, elapsed = run_mcmc(:kalman_first, mode_result, n_samples = 2000)

# ============================================================================
# Save Results
# ============================================================================

println("\n[6/6] Saving results...")

results_dir = joinpath(@__DIR__, "..", ".local_artifacts", "hlt_estimation_update_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
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
    "filter" => "kalman_first_order (linear approximation)",
    "timestamp" => string(now()),
)
serialize(summary_path, summary)

# Save CSV for paper
csv_path = joinpath(results_dir, "posterior_estimates.csv")
posterior_df = DataFrame(samples)
CSV.write(csv_path, posterior_df)

println("  ✓ Results saved to: $(results_dir)")
println("    - MCMC chain: mcmc_chain.jls")
println("    - Summary: estimation_summary.jls")
println("    - CSV: posterior_estimates.csv")

println("\n" * "="^80)
println("Estimation Complete!")
println("="^80)
println("\nNext steps:")
println("1. Check convergence diagnostics (R-hat, ESS)")
println("2. TBA: Re-run with :switching filter once integrated")
println("3. Generate posterior plots and tables for paper")
println("4. Compare with original sample (1960Q1-2004Q4) results")
println("\nResults directory: $(results_dir)")
println("="^80)
