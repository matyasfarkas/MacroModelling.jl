"""
HLT Model Parameter Configuration
==================================

Defines parameter sets for surrogate NN estimation.
Based on Trabandt et al. (2023) "Understanding Post-COVID Inflation Dynamics"

Author: Claude Code
Date: January 2026
"""

using Distributions

# ============================================================================
# Parameter Set Definitions
# ============================================================================

"""
    ParameterSpec

Specification for a single parameter.

Fields:
- `name::Symbol`: Parameter name (e.g., :ρ_a)
- `prior_type::Symbol`: Distribution type (:Beta, :Normal, :InvGamma, :Uniform)
- `prior_params::NamedTuple`: Distribution parameters
- `bounds::Tuple{Float64, Float64}`: (lower, upper) bounds for sampling
- `description::String`: Human-readable description
"""
struct ParameterSpec
    name::Symbol
    prior_type::Symbol
    prior_params::NamedTuple
    bounds::Tuple{Float64, Float64}
    description::String
end

# ============================================================================
# Phase 1: 18 Parameters (Shocks + Key Structural)
# ============================================================================

"""
    get_phase1_18param_specs()

Returns parameter specifications for Phase 1 estimation (18 parameters).

Includes:
- 7 shock persistence parameters (ρ)
- 7 shock volatility parameters (σ)
- 4 structural parameters (cprobp, cindp, curvp, ξ_w)

Based on Trabandt et al. (2023) Table A.2.
"""
function get_phase1_18param_specs()
    specs = ParameterSpec[]

    # -----------------------------------------------------------------------
    # Shock Persistence Parameters (7)
    # Prior: Beta(0.5, 0.2) → mean ≈ 0.71, constrained to (0.01, 0.99)
    # -----------------------------------------------------------------------

    push!(specs, ParameterSpec(
        :crhoa, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "TFP shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhob, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "Risk premium shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhog, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "Government spending shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhoqs, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "Investment shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhopinf, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "Price markup shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhow, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "Wage markup shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhoms, :Beta,
        (α=0.5, β=0.2),
        (0.01, 0.99),
        "Monetary policy shock persistence"
    ))

    # -----------------------------------------------------------------------
    # Shock Volatility Parameters (7)
    # Prior: InverseGamma(2, 0.1) → mean = 0.1, std ≈ 0.14
    # Bounds: (0.001, 1.0) to avoid numerical issues
    # -----------------------------------------------------------------------

    push!(specs, ParameterSpec(
        :z_ea, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "TFP shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_eb, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "Risk premium shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_eg, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "Government spending shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_eqs, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "Investment shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_epinf, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "Price markup shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_ew, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "Wage markup shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_em, :InvGamma,
        (α=2.0, θ=0.1),
        (0.001, 5.0),
        "Monetary policy shock volatility"
    ))

    # -----------------------------------------------------------------------
    # Structural Parameters (4)
    # From Trabandt Table A.2 and current 3-param estimation
    # -----------------------------------------------------------------------

    push!(specs, ParameterSpec(
        :cprobp, :Beta,
        (α=0.5, β=0.1),
        (0.5, 0.95),
        "Calvo price stickiness (ξ_p)"
    ))

    push!(specs, ParameterSpec(
        :cindp, :Beta,
        (α=0.5, β=0.15),
        (0.01, 0.99),
        "Price indexation (ι_p)"
    ))

    push!(specs, ParameterSpec(
        :curvp, :Normal,
        (μ=64.5, σ=25.0),  # Trabandt posterior mode = 64.5
        (2.0, 150.0),       # Lower bound 2.0 to include HLT baseline=10.0
        "Kimball price curvature (ε_p)"
    ))

    push!(specs, ParameterSpec(
        :cprobw, :Beta,  # Note: using cprobw to match MacroModelling.jl naming
        (α=0.5, β=0.1),
        (0.5, 0.95),
        "Calvo wage stickiness (ξ_w)"
    ))

    return specs
end

# ============================================================================
# Phase 1 NARROW: 18 Parameters with Tight Priors (for SEP dataset generation)
# ============================================================================

"""
    get_phase1_18param_narrow_specs()

Returns NARROW parameter specifications for Phase 1 (18 parameters).
Uses tight priors centered on HLT baseline calibration ±10-20%.

This is recommended for SEP dataset generation to ensure numerical stability.
For final Bayesian estimation, use get_phase1_18param_specs() with wider priors.

Baseline values from Smets_Wouters_2007_HLT_obc.jl:
- Persistence: crhoa=0.9977, crhob=0.5799, crhog=0.9957, etc.
- Volatilities: z_ea=0.4618, z_eb=1.8513, etc.
- Structural: cprobp=0.6, cindp=0.47, curvp=10, cprobw=0.8087
"""
function get_phase1_18param_narrow_specs()
    specs = ParameterSpec[]

    # -----------------------------------------------------------------------
    # Shock Persistence Parameters (7)
    # Narrow priors: baseline ± 10-15% (keeping in [0.01, 0.99])
    # Using Normal priors truncated to reasonable ranges
    # -----------------------------------------------------------------------

    # crhoa baseline = 0.9977, allow 0.85-0.99
    push!(specs, ParameterSpec(
        :crhoa, :Normal,
        (μ=0.95, σ=0.05),
        (0.85, 0.995),
        "TFP shock persistence"
    ))

    # crhob baseline = 0.5799, allow 0.45-0.75
    push!(specs, ParameterSpec(
        :crhob, :Normal,
        (μ=0.60, σ=0.10),
        (0.45, 0.75),
        "Risk premium shock persistence"
    ))

    # crhog baseline = 0.9957, allow 0.85-0.99
    push!(specs, ParameterSpec(
        :crhog, :Normal,
        (μ=0.95, σ=0.05),
        (0.85, 0.995),
        "Government spending shock persistence"
    ))

    # crhoqs baseline = 0.7165, allow 0.60-0.85
    push!(specs, ParameterSpec(
        :crhoqs, :Normal,
        (μ=0.72, σ=0.08),
        (0.60, 0.85),
        "Investment shock persistence"
    ))

    # crhopinf baseline = 0.0 (NO PERSISTENCE in HLT model), allow 0.0-0.30
    push!(specs, ParameterSpec(
        :crhopinf, :Normal,
        (μ=0.10, σ=0.08),
        (0.0, 0.30),
        "Price markup shock persistence"
    ))

    # crhow baseline = 0.0 (NO PERSISTENCE in HLT model), allow 0.0-0.30
    push!(specs, ParameterSpec(
        :crhow, :Normal,
        (μ=0.10, σ=0.08),
        (0.0, 0.30),
        "Wage markup shock persistence"
    ))

    # crhoms baseline = 0.0 (NO PERSISTENCE in HLT model), allow 0.0-0.30
    push!(specs, ParameterSpec(
        :crhoms, :Normal,
        (μ=0.10, σ=0.08),
        (0.0, 0.30),
        "Monetary policy shock persistence"
    ))

    # -----------------------------------------------------------------------
    # Shock Volatility Parameters (7)
    # Narrow priors: baseline × [0.5, 2.0]
    # Using Normal priors (truncated to positive) centered on baseline
    # -----------------------------------------------------------------------

    # z_ea baseline = 0.4618, allow 0.20-0.80
    push!(specs, ParameterSpec(
        :z_ea, :Normal,
        (μ=0.46, σ=0.15),
        (0.20, 0.80),
        "TFP shock volatility"
    ))

    # z_eb baseline = 1.8513, allow 1.0-3.0
    push!(specs, ParameterSpec(
        :z_eb, :Normal,
        (μ=1.85, σ=0.50),
        (1.0, 3.0),
        "Risk premium shock volatility"
    ))

    # z_eg baseline = 0.6090, allow 0.30-1.00
    push!(specs, ParameterSpec(
        :z_eg, :Normal,
        (μ=0.61, σ=0.18),
        (0.30, 1.00),
        "Government spending shock volatility"
    ))

    # z_eqs baseline = 0.6017, allow 0.30-1.00
    push!(specs, ParameterSpec(
        :z_eqs, :Normal,
        (μ=0.60, σ=0.18),
        (0.30, 1.00),
        "Investment shock volatility"
    ))

    # z_epinf baseline = 0.1455, allow 0.05-0.30
    push!(specs, ParameterSpec(
        :z_epinf, :Normal,
        (μ=0.15, σ=0.06),
        (0.05, 0.30),
        "Price markup shock volatility"
    ))

    # z_ew baseline = 0.2089, allow 0.10-0.40
    push!(specs, ParameterSpec(
        :z_ew, :Normal,
        (μ=0.21, σ=0.08),
        (0.10, 0.40),
        "Wage markup shock volatility"
    ))

    # z_em baseline = 0.2397, allow 0.10-0.40
    push!(specs, ParameterSpec(
        :z_em, :Normal,
        (μ=0.24, σ=0.08),
        (0.10, 0.40),
        "Monetary policy shock volatility"
    ))

    # -----------------------------------------------------------------------
    # Structural Parameters (4)
    # Narrow priors centered on baseline calibration
    # -----------------------------------------------------------------------

    # cprobp baseline = 0.6, allow 0.50-0.75
    push!(specs, ParameterSpec(
        :cprobp, :Normal,
        (μ=0.60, σ=0.08),
        (0.50, 0.75),
        "Calvo price stickiness (ξ_p)"
    ))

    # cindp baseline = 0.47, allow 0.30-0.65
    push!(specs, ParameterSpec(
        :cindp, :Normal,
        (μ=0.47, σ=0.10),
        (0.30, 0.65),
        "Price indexation (ι_p)"
    ))

    # curvp baseline = 10, allow 5-20
    push!(specs, ParameterSpec(
        :curvp, :Normal,
        (μ=10.0, σ=4.0),
        (5.0, 20.0),
        "Kimball price curvature (ε_p)"
    ))

    # cprobw baseline = 0.8087, allow 0.70-0.90
    push!(specs, ParameterSpec(
        :cprobw, :Normal,
        (μ=0.81, σ=0.06),
        (0.70, 0.90),
        "Calvo wage stickiness (ξ_w)"
    ))

    return specs
end

# ============================================================================
# Legacy: 3 Parameters (Original)
# ============================================================================

"""
    get_legacy_3param_specs()

Returns parameter specifications for legacy 3-parameter estimation.

Includes:
- cprobp (Calvo price stickiness)
- cindp (Price indexation)
- curvp (Kimball curvature)

Maintained for backward compatibility.
"""
function get_legacy_3param_specs()
    specs = ParameterSpec[]

    push!(specs, ParameterSpec(
        :cprobp, :Beta,
        (α=0.5, β=0.1),
        (0.5, 0.95),
        "Calvo price stickiness (ξ_p)"
    ))

    push!(specs, ParameterSpec(
        :cindp, :Beta,
        (α=0.5, β=0.15),
        (0.01, 0.99),
        "Price indexation (ι_p)"
    ))

    push!(specs, ParameterSpec(
        :curvp, :Normal,
        (μ=75.0, σ=25.0),
        (20.0, 150.0),
        "Kimball price curvature (ε_p)"
    ))

    return specs
end

# ============================================================================
# Econometrica Bridge: Reduced Investment Block
# ============================================================================

"""
    get_investment_4p_specs()

Returns the reduced SW07-HLT investment-channel bridge block used for the
medium-scale direct SEP validation between the Galí package and the full
18-parameter HLT application.
"""
function get_investment_4p_specs()
    specs = ParameterSpec[]

    push!(specs, ParameterSpec(
        :crhob, :Normal,
        (μ=0.5799, σ=0.06),
        (0.45, 0.75),
        "Risk premium shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhoqs, :Normal,
        (μ=0.7165, σ=0.06),
        (0.60, 0.85),
        "Investment-specific shock persistence"
    ))

    push!(specs, ParameterSpec(
        :z_eb, :Normal,
        (μ=1.8513, σ=0.25),
        (1.20, 2.50),
        "Risk premium shock volatility"
    ))

    push!(specs, ParameterSpec(
        :z_eqs, :Normal,
        (μ=0.6017, σ=0.12),
        (0.35, 0.90),
        "Investment-specific shock volatility"
    ))

    return specs
end

"""
    get_investment_4p_supported_specs()

Returns the reduced investment bridge block restricted to the finite direct-SEP
support mapped by the 2026-05-27 bridge smoke runs. The only trimmed dimension
is risk-premium shock volatility: the 3-point support map solved all tested
cells at `z_eb <= 1.85` and failed all tested cells at `z_eb = 2.50`.
"""
function get_investment_4p_supported_specs()
    specs = ParameterSpec[]

    push!(specs, ParameterSpec(
        :crhob, :Normal,
        (μ=0.5799, σ=0.06),
        (0.45, 0.75),
        "Risk premium shock persistence"
    ))

    push!(specs, ParameterSpec(
        :crhoqs, :Normal,
        (μ=0.7165, σ=0.06),
        (0.60, 0.85),
        "Investment-specific shock persistence"
    ))

    push!(specs, ParameterSpec(
        :z_eb, :Normal,
        (μ=1.8513, σ=0.20),
        (1.20, 1.85),
        "Risk premium shock volatility, trimmed to mapped SEP support"
    ))

    push!(specs, ParameterSpec(
        :z_eqs, :Normal,
        (μ=0.6017, σ=0.12),
        (0.35, 0.90),
        "Investment-specific shock volatility"
    ))

    return specs
end

"""
    get_investment_curvature_5p_specs()

Adds investment adjustment-cost curvature to the bridge block for stress tests.
"""
function get_investment_curvature_5p_specs()
    specs = ParameterSpec[
        ParameterSpec(
            :csadjcost, :Normal,
            (μ=6.0144, σ=1.0),
            (4.0, 8.5),
            "Investment adjustment-cost curvature"
        ),
    ]
    append!(specs, get_investment_4p_specs())
    return specs
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_parameter_specs(set::Symbol)

Get parameter specifications for a given parameter set.

# Arguments
- `set::Symbol`: Parameter set identifier
  - `:legacy_3params` - Original 3 parameters
  - `:phase1_18params` - Phase 1: 18 parameters (wide priors for estimation)
  - `:phase1_18params_narrow` - Phase 1: 18 parameters (narrow priors for SEP dataset)
  - `:investment_4p` - Reduced investment/risk-premium bridge block
  - `:investment_4p_supported` - Bridge block trimmed to mapped direct-SEP support
  - `:investment_curvature_5p` - Bridge block plus adjustment-cost curvature

# Returns
- `Vector{ParameterSpec}`: Parameter specifications

# Example
```julia
specs = get_parameter_specs(:phase1_18params_narrow)
println("Estimating \$(length(specs)) parameters")
```
"""
function get_parameter_specs(set::Symbol)
    if set == :legacy_3params
        return get_legacy_3param_specs()
    elseif set == :phase1_18params
        return get_phase1_18param_specs()
    elseif set == :phase1_18params_narrow
        return get_phase1_18param_narrow_specs()
    elseif set == :investment_4p
        return get_investment_4p_specs()
    elseif set == :investment_4p_supported
        return get_investment_4p_supported_specs()
    elseif set == :investment_curvature_5p
        return get_investment_curvature_5p_specs()
    else
        error("Unknown parameter set: $set. Valid options: :legacy_3params, :phase1_18params, :phase1_18params_narrow, :investment_4p, :investment_4p_supported, :investment_curvature_5p")
    end
end

"""
    get_parameter_names(set::Symbol)

Get parameter names for a given parameter set.

# Example
```julia
names = get_parameter_names(:phase1_18params)
# Returns: [:ρ_a, :ρ_b, ..., :cprobw]
```
"""
function get_parameter_names(set::Symbol)
    specs = get_parameter_specs(set)
    return [spec.name for spec in specs]
end

"""
    get_parameter_bounds(set::Symbol)

Get parameter bounds as a dictionary.

# Returns
- `Dict{Symbol, Tuple{Float64, Float64}}`: Map from parameter name to (lower, upper) bounds

# Example
```julia
bounds = get_parameter_bounds(:phase1_18params)
lb, ub = bounds[:ρ_a]  # (0.01, 0.99)
```
"""
function get_parameter_bounds(set::Symbol)
    specs = get_parameter_specs(set)
    return Dict(spec.name => spec.bounds for spec in specs)
end

"""
    get_parameter_priors(set::Symbol)

Get parameter priors as distributions.

# Returns
- `Dict{Symbol, Distribution}`: Map from parameter name to prior distribution

# Example
```julia
priors = get_parameter_priors(:phase1_18params)
ρ_a_prior = priors[:ρ_a]  # Beta distribution
```
"""
function get_parameter_priors(set::Symbol)
    specs = get_parameter_specs(set)
    priors = Dict{Symbol, Distribution}()

    for spec in specs
        if spec.prior_type == :Beta
            # Bounded Beta distribution
            priors[spec.name] = Distributions.Beta(spec.prior_params.α, spec.prior_params.β)
        elseif spec.prior_type == :Normal
            priors[spec.name] = Distributions.Normal(spec.prior_params.μ, spec.prior_params.σ)
        elseif spec.prior_type == :InvGamma
            priors[spec.name] = Distributions.InverseGamma(spec.prior_params.α, spec.prior_params.θ)
        elseif spec.prior_type == :Uniform
            lb, ub = spec.bounds
            priors[spec.name] = Distributions.Uniform(lb, ub)
        else
            error("Unknown prior type: $(spec.prior_type)")
        end
    end

    return priors
end

"""
    print_parameter_summary(set::Symbol)

Print a formatted summary of parameter specifications.

# Example
```julia
print_parameter_summary(:phase1_18params)
```
"""
function print_parameter_summary(set::Symbol)
    specs = get_parameter_specs(set)

    println("=" ^ 80)
    println("Parameter Set: $set")
    println("Number of parameters: $(length(specs))")
    println("=" ^ 80)
    println()

    for (i, spec) in enumerate(specs)
        println("[$i] $(spec.name)")
        println("    Description: $(spec.description)")
        println("    Prior: $(spec.prior_type)$(spec.prior_params)")
        println("    Bounds: $(spec.bounds)")
        println()
    end

    println("=" ^ 80)
end

"""
    get_phase1_18param_baseline()

Returns calibrated baseline values for the 18 Phase 1 parameters.
These are the values used in Smets_Wouters_2007_HLT_obc.jl.

# Returns
- `Dict{Symbol, Float64}`: Map from parameter name to baseline value

# Example
```julia
baseline = get_phase1_18param_baseline()
crhoa_baseline = baseline[:crhoa]  # 0.9977
```
"""
function get_phase1_18param_baseline()
    return Dict{Symbol, Float64}(
        :csadjcost => 6.0144,

        # Shock Persistence (7) - CORRECTED from actual HLT model
        :crhoa => 0.9977,
        :crhob => 0.5799,
        :crhog => 0.9957,  # was 0.9930
        :crhoqs => 0.7165,  # was 0.9928 - CRITICAL FIX
        :crhopinf => 0.0,  # was 0.8910 - CRITICAL FIX
        :crhow => 0.0,  # was 0.9688 - CRITICAL FIX
        :crhoms => 0.0,  # was 0.9000 - CRITICAL FIX

        # Shock Volatility (7)
        :z_ea => 0.4618,
        :z_eb => 1.8513,
        :z_eg => 0.6090,
        :z_eqs => 0.6017,
        :z_epinf => 0.1455,
        :z_ew => 0.2089,
        :z_em => 0.2397,

        # Structural (4)
        :cprobp => 0.6,
        :cindp => 0.47,
        :curvp => 10.0,
        :cprobw => 0.8087
    )
end

# ============================================================================
# Exports
# ============================================================================

export ParameterSpec
export get_parameter_specs, get_parameter_names, get_parameter_bounds, get_parameter_priors
export print_parameter_summary, get_phase1_18param_baseline
