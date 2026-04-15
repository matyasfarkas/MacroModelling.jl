"""
HLT Turing Model Factory
========================

Dynamic generation of Turing models for surrogate NN-based MCMC estimation.

Supports variable parameter counts via metaprogramming, enabling:
- Legacy 3-parameter estimation (cprobp, cindp, curvp)
- Phase 1: 18-parameter estimation (shocks + structural)
- Future: 28+ parameter estimation

Author: Claude Code
Date: January 2026
"""

using Turing
using Distributions
using MacroModelling

include("parameter_config.jl")

# ============================================================================
# Core Turing Model Factory
# ============================================================================

"""
    create_hlt_surrogate_model(param_set::Symbol, frozen_mlp, data;
                                 use_cached_calibration::Bool=false,
                                 calibrator=nothing)

Create a Turing model for HLT surrogate estimation.

# Arguments
- `param_set::Symbol`: Parameter set identifier (`:legacy_3params` or `:phase1_18params`)
- `frozen_mlp`: Trained FrozenMLP surrogate network
- `data`: Observable data (KeyedArray, 2D: Variables × Time)
- `use_cached_calibration::Bool`: Enable cached gate calibration (default: false)
- `calibrator`: CachedGateCalibrator instance (required if use_cached_calibration=true)

# Returns
- Turing model ready for sampling

# Example
```julia
# Load trained surrogate
frozen_mlp = load("hlt_surrogate_18param.jld2", "frozen_mlp")

# Create model
turing_model = create_hlt_surrogate_model(:phase1_18params, frozen_mlp, data)

# Sample
chain = sample(turing_model, NUTS(), 2000)
```
"""
function create_hlt_surrogate_model(param_set::Symbol,
                                     frozen_mlp,
                                     data;
                                     use_cached_calibration::Bool=false,
                                     calibrator=nothing)

    # Get parameter specifications
    specs = get_parameter_specs(param_set)
    n_params = length(specs)

    # Validate
    if use_cached_calibration && calibrator === nothing
        error("Cached calibration enabled but no calibrator provided")
    end

    # Build Turing model dynamically
    @model function hlt_surrogate_dynamic(obs_data)
        # ====================================================================
        # Prior Sampling
        # ====================================================================
        θ = Vector{Float64}(undef, n_params)

        for (i, spec) in enumerate(specs)
            if spec.prior_type == :Beta
                # Bounded Beta distribution
                lb, ub = spec.bounds
                α, β = spec.prior_params.α, spec.prior_params.β

                # Use MacroModelling's bounded Beta if available
                try
                    θ[i] ~ MacroModelling.Beta(α, β, lower=lb, upper=ub)
                catch
                    # Fallback: transform standard Beta to bounds
                    θ_raw ~ Beta(α, β)
                    θ[i] = lb + (ub - lb) * θ_raw
                end

            elseif spec.prior_type == :Normal
                μ, σ = spec.prior_params.μ, spec.prior_params.σ
                lb, ub = spec.bounds

                # Truncated normal
                θ[i] ~ truncated(Normal(μ, σ), lb, ub)

            elseif spec.prior_type == :InvGamma
                α, θ_ig = spec.prior_params.α, spec.prior_params.θ
                lb, ub = spec.bounds

                # Truncated inverse gamma
                θ[i] ~ truncated(InverseGamma(α, θ_ig), lb, ub)

            elseif spec.prior_type == :Uniform
                lb, ub = spec.bounds
                θ[i] ~ Uniform(lb, ub)

            else
                error("Unsupported prior type: $(spec.prior_type)")
            end
        end

        # ====================================================================
        # Likelihood Computation
        # ====================================================================

        # Compute log-likelihood using NN surrogate
        ll = compute_surrogate_loglikelihood(frozen_mlp, obs_data, θ)

        # Add to model log probability
        Turing.@addlogprob! ll

        return θ
    end

    return hlt_surrogate_dynamic(data)
end

# ============================================================================
# Likelihood Computation
# ============================================================================

"""
    compute_surrogate_loglikelihood(frozen_mlp, data, θ::Vector{Float64})

Compute log-likelihood using the trained NN surrogate.

# Arguments
- `frozen_mlp`: FrozenMLP surrogate network
- `data`: Observable data (KeyedArray, 2D: Variables × Time)
- `θ`: Parameter vector

# Returns
- `ll::Float64`: Log-likelihood value

# Details
This function:
1. Constructs NN inputs (state, shocks, parameters) for each time period
2. Runs forward pass through frozen MLP
3. Computes Gaussian likelihood from predictions vs observations
"""
function compute_surrogate_loglikelihood(frozen_mlp, data, θ::Vector{Float64})
    # Extract observables
    T = size(data, 2)  # Time periods
    n_obs = size(data, 1)  # Number of observables

    # Initialize log-likelihood
    ll = 0.0

    # TODO: This is a placeholder implementation
    # Actual implementation should:
    # 1. Extract state variables from data
    # 2. For each time t:
    #    - Construct input: [state_{t-1}; shocks_t; θ]
    #    - Predict: y_pred_t = frozen_mlp(input_t)
    #    - Compute: ll += logpdf(Normal(y_pred_t, σ_obs), y_actual_t)
    # 3. Sum over all time periods and observables

    # For now, return a simple placeholder
    # This will be replaced with actual likelihood computation
    # based on the existing hlt_sep_surrogate_synthetic_estimation.jl logic

    @warn "compute_surrogate_loglikelihood is using placeholder implementation"

    return ll
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    create_legacy_3param_model(frozen_mlp, data)

Convenience function for legacy 3-parameter model.

Equivalent to:
```julia
create_hlt_surrogate_model(:legacy_3params, frozen_mlp, data)
```
"""
function create_legacy_3param_model(frozen_mlp, data)
    return create_hlt_surrogate_model(:legacy_3params, frozen_mlp, data)
end

"""
    create_phase1_18param_model(frozen_mlp, data)

Convenience function for Phase 1 18-parameter model.

Equivalent to:
```julia
create_hlt_surrogate_model(:phase1_18params, frozen_mlp, data)
```
"""
function create_phase1_18param_model(frozen_mlp, data)
    return create_hlt_surrogate_model(:phase1_18params, frozen_mlp, data)
end

"""
    sample_hlt_model(param_set::Symbol, frozen_mlp, data;
                     n_samples::Int=2000,
                     n_chains::Int=4,
                     sampler=NUTS(0.65),
                     progress::Bool=true)

High-level interface for sampling HLT surrogate model.

# Arguments
- `param_set::Symbol`: Parameter set (`:legacy_3params` or `:phase1_18params`)
- `frozen_mlp`: Trained surrogate network
- `data`: Observable data
- `n_samples::Int`: Number of MCMC samples per chain (default: 2000)
- `n_chains::Int`: Number of parallel chains (default: 4)
- `sampler`: Turing sampler (default: NUTS with 65% acceptance)
- `progress::Bool`: Show progress bar (default: true)

# Returns
- `MCMCChains.Chains`: Posterior samples

# Example
```julia
chain = sample_hlt_model(
    :phase1_18params,
    frozen_mlp,
    data,
    n_samples = 2000,
    n_chains = 4
)

# Check convergence
using MCMCChains
rhat_vals = rhat(chain)
println("Max R-hat: ", maximum(rhat_vals))
```
"""
function sample_hlt_model(param_set::Symbol,
                          frozen_mlp,
                          data;
                          n_samples::Int=2000,
                          n_chains::Int=4,
                          sampler=NUTS(0.65),
                          progress::Bool=true)

    # Create model
    model = create_hlt_surrogate_model(param_set, frozen_mlp, data)

    # Sample
    println("=" ^ 80)
    println("HLT Surrogate Model Sampling")
    println("=" ^ 80)
    println("Parameter set: $param_set")
    println("Number of parameters: $(length(get_parameter_specs(param_set)))")
    println("Samples per chain: $n_samples")
    println("Number of chains: $n_chains")
    println("Sampler: $sampler")
    println("=" ^ 80)
    println()

    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains; progress=progress)

    println()
    println("=" ^ 80)
    println("Sampling Complete")
    println("=" ^ 80)

    return chain
end

# ============================================================================
# Parameter Extraction and Utilities
# ============================================================================

"""
    extract_parameter_dict(chain, param_set::Symbol; summary_stat::Symbol=:mean)

Extract parameter estimates as a dictionary.

# Arguments
- `chain`: MCMC chains from Turing
- `param_set::Symbol`: Parameter set used
- `summary_stat::Symbol`: Which statistic to extract (`:mean`, `:median`, `:mode`)

# Returns
- `Dict{Symbol, Float64}`: Map from parameter name to estimated value

# Example
```julia
param_estimates = extract_parameter_dict(chain, :phase1_18params)
println("Estimated ρ_a: ", param_estimates[:ρ_a])
```
"""
function extract_parameter_dict(chain, param_set::Symbol; summary_stat::Symbol=:mean)
    specs = get_parameter_specs(param_set)
    param_dict = Dict{Symbol, Float64}()

    for spec in specs
        param_name = spec.name

        if summary_stat == :mean
            param_dict[param_name] = mean(chain[param_name])
        elseif summary_stat == :median
            param_dict[param_name] = median(chain[param_name])
        elseif summary_stat == :mode
            # Use KDE mode approximation
            param_dict[param_name] = mode(chain[param_name])
        else
            error("Unknown summary statistic: $summary_stat")
        end
    end

    return param_dict
end

"""
    print_estimation_summary(chain, param_set::Symbol; true_params=nothing)

Print formatted estimation results.

# Arguments
- `chain`: MCMC chains
- `param_set::Symbol`: Parameter set
- `true_params`: Optional dictionary of true parameter values (for synthetic validation)

# Example
```julia
print_estimation_summary(chain, :phase1_18params, true_params=trabandt_posteriors)
```
"""
function print_estimation_summary(chain, param_set::Symbol; true_params=nothing)
    specs = get_parameter_specs(param_set)

    println("=" ^ 80)
    println("HLT Surrogate Estimation Results")
    println("=" ^ 80)
    println()

    println(@sprintf("%-12s %12s %12s %12s %12s",
                     "Parameter", "Mean", "Std", "2.5%", "97.5%"))
    println("-" ^ 80)

    for spec in specs
        param_name = spec.name
        param_samples = Array(chain[param_name])

        μ = mean(param_samples)
        σ = std(param_samples)
        q025 = quantile(param_samples[:], 0.025)
        q975 = quantile(param_samples[:], 0.975)

        println(@sprintf("%-12s %12.4f %12.4f %12.4f %12.4f",
                         string(param_name), μ, σ, q025, q975))

        # Show error if true params provided
        if true_params !== nothing && haskey(true_params, param_name)
            true_val = true_params[param_name]
            error_pct = 100 * abs(μ - true_val) / true_val
            println(@sprintf("             [True: %.4f, Error: %.2f%%]", true_val, error_pct))
        end
    end

    println("=" ^ 80)

    # Convergence diagnostics
    println()
    println("Convergence Diagnostics:")
    println("-" ^ 80)

    using MCMCChains
    rhat_vals = rhat(chain)
    ess_vals = ess(chain)

    println(@sprintf("Max R-hat: %.4f (target: < 1.01)", maximum(rhat_vals)))
    println(@sprintf("Min ESS: %.0f (target: > 400)", minimum(ess_vals)))

    println("=" ^ 80)
end

# ============================================================================
# Exports
# ============================================================================

export create_hlt_surrogate_model
export create_legacy_3param_model, create_phase1_18param_model
export sample_hlt_model
export extract_parameter_dict, print_estimation_summary
export compute_surrogate_loglikelihood
