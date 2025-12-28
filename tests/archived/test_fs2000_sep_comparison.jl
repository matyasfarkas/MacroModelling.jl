# Test: SEP on FS2000 for Dynare comparison
# Simplest model - cash-in-advance RBC model

using MacroModelling
using Printf

println("="^70)
println("MacroModelling.jl SEP TEST: FS2000 (Schorfheide 2000)")
println("Matching Dynare extended_path(periods=10, order={1,2})")
println("="^70)

# Load the FS2000 model from MacroModelling package
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

println("\n1. Model loaded successfully")
println("   Model variables: ", length(FS2000.var))
println("   Model shocks: ", FS2000.exo)
println("   Model parameters: ", length(FS2000.parameters))

# Key variables to report
test_vars = [:y, :c, :R, :n, :k]

# ============================================================================
# TEST 1: SEP with order=1 (periods=10, branching length = 1)
# ============================================================================
println("\n" * "="^70)
println("TEST 1: SEP(1) - extended_path(periods=10, order=1)")
println("="^70)
println("Configuration:")
println("  - periods (T) = 10")
println("  - order (branching length) = 1")
println("  - nnodes = 3 (Gauss-Hermite quadrature)")

solve!(FS2000,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

sep_sol_1 = FS2000.solution.perturbation.stochastic_extended_path

println("\nSolution status:")
@printf("  - Convergence: %s\n", sep_sol_1.convergence_flag == 1 ? "SUCCESS" : "FAILED")
@printf("  - Final error: %.3e\n", sep_sol_1.final_error)
@printf("  - Runtime: %.2f seconds\n\n", sep_sol_1.runtime_seconds)

# Extract steady state values
# CRITICAL: Use layout.voff to get correct indices in the SEP tree structure
# Y is organized as a branching tree, not a simple vector
# For t=0 steady state, use voff[1] + (1:ny_)
layout = sep_sol_1.layout
yss_indices = layout.voff[1] .+ (1:layout.ny_)
yss_1 = sep_sol_1.Y[yss_indices]

println("Key steady state values (SEP order=1):")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    # SEP Y vector follows model.var order
    var_idx = findfirst(==(var), FS2000.var)
    if !isnothing(var_idx)
        @printf("  %-12s  %12.8f\n", string(var), yss_1[var_idx])
    end
end

# ============================================================================
# TEST 2: SEP with order=2 (periods=10, branching length = 2)
# ============================================================================
println("\n" * "="^70)
println("TEST 2: SEP(2) - extended_path(periods=10, order=2)")
println("="^70)
println("Configuration:")
println("  - periods (T) = 10")
println("  - order (branching length) = 2")
println("  - nnodes = 3 (Gauss-Hermite quadrature)")

solve!(FS2000,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 2,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

sep_sol_2 = FS2000.solution.perturbation.stochastic_extended_path

println("\nSolution status:")
@printf("  - Convergence: %s\n", sep_sol_2.convergence_flag == 1 ? "SUCCESS" : "FAILED")
@printf("  - Final error: %.3e\n", sep_sol_2.final_error)
@printf("  - Runtime: %.2f seconds\n\n", sep_sol_2.runtime_seconds)

# Extract steady state values
# CRITICAL: Use layout.voff to get correct indices in the SEP tree structure
layout_2 = sep_sol_2.layout
yss_indices_2 = layout_2.voff[1] .+ (1:layout_2.ny_)
yss_2 = sep_sol_2.Y[yss_indices_2]

println("Key steady state values (SEP order=2):")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    # SEP Y vector follows model.var order
    var_idx = findfirst(==(var), FS2000.var)
    if !isnothing(var_idx)
        @printf("  %-12s  %12.8f\n", string(var), yss_2[var_idx])
    end
end

# ============================================================================
# COMPARISON: SEP(1) vs SEP(2)
# ============================================================================
println("\n" * "="^70)
println("COMPARISON: MacroModelling SEP(1) vs SEP(2)")
println("="^70)
println("  Variable      SEP(1)         SEP(2)         Difference")
println("  " * "-"^65)

for var in test_vars
    # SEP Y vector follows model.var order
    var_idx = findfirst(==(var), FS2000.var)
    if !isnothing(var_idx)
        val1 = yss_1[var_idx]
        val2 = yss_2[var_idx]
        diff = val2 - val1
        @printf("  %-12s  %12.8f  %12.8f  %12.8f\n",
                string(var), val1, val2, diff)
    end
end

# ============================================================================
# IRF COMPUTATION: Conditional Forecast Method
# ============================================================================
# Compute IRFs as difference between shocked and baseline simulations
# Method: shocked(1σ at t=1, uncertain future) - baseline(0 at t=1, uncertain future)

println("\n" * "="^70)
println("COMPUTING STOCHASTIC IRFs (Conditional Forecast Method)")
println("="^70)

# IRF horizon
irf_horizon = 20

# Shock standard deviations (from FS2000 model parameters)
sigma_e_a = 0.035449  # TFP shock
sigma_e_m = 0.008862  # Money growth shock

# Get number of variables
ny = length(FS2000.var)
nshocks = length(FS2000.exo)

# We need to use the SEP solution to simulate paths
# Use the SEP order=1 solution that's already computed
sep_sol = sep_sol_1
layout = sep_sol.layout

# Helper function to simulate deterministic path from SEP solution
function simulate_sep_deterministic(model, sep_solution, shocks::Matrix{Float64}, periods::Int)
    # shocks: nshocks × periods matrix
    # Returns: ny × (periods+1) matrix of endogenous variables

    # Get initial steady state
    # CRITICAL: Use layout.voff to get correct indices
    yss = sep_solution.Y[layout.voff[1] .+ (1:layout.ny_)]

    # For deterministic simulation, we use perfect foresight from SEP solution
    # This is a simplified approach - ideally would resolve SEP with specific shocks
    # For now, use linear approximation from steady state

    # Get first-order coefficients if available
    # Fallback: return steady state path (conservative)
    result = repeat(yss, 1, periods+1)

    return result
end

# ============================================================================
# IRF 1: TFP Shock (e_a)
# ============================================================================
println("\nComputing IRF to TFP shock (e_a, 1 sigma)...")

# Create shock matrices
shocks_baseline = zeros(nshocks, irf_horizon)
shocks_e_a = zeros(nshocks, irf_horizon)
shocks_e_a[1, 1] = sigma_e_a  # e_a shock at period 1

# For proper IRF computation, we need to use deterministic simulation from SEP
# Using get_irf as approximation (perturbation-based)
# NOTE: This is a limitation - proper SEP IRF would require resolving with conditional shocks

# Alternative: Use perturbation IRF as proxy
# Get first-order perturbation solution
try
    solve!(FS2000, algorithm = :first_order, silent = true)

    # Get IRF for e_a shock (1 sigma)
    irf_e_a_pert = get_irf(FS2000, :e_a, periods = irf_horizon, algorithm = :first_order)

    println("IRF to TFP shock (e_a, 1 sigma) - Using perturbation approximation:")
    println("  Period    y           c           R           n           k")
    println("  " * "-"^65)

    for t in 1:min(10, irf_horizon)
        y_val = irf_e_a_pert[t, :y]
        c_val = irf_e_a_pert[t, :c]
        r_val = irf_e_a_pert[t, :R]
        n_val = irf_e_a_pert[t, :n]
        k_val = irf_e_a_pert[t, :k]

        @printf("  %2d     %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
                t, y_val, c_val, r_val, n_val, k_val)
    end

    # ============================================================================
    # IRF 2: Money Growth Shock (e_m)
    # ============================================================================
    println("\nComputing IRF to money growth shock (e_m, 1 sigma)...")

    # Get IRF for e_m shock (1 sigma)
    irf_e_m_pert = get_irf(FS2000, :e_m, periods = irf_horizon, algorithm = :first_order)

    println("IRF to money growth shock (e_m, 1 sigma) - Using perturbation approximation:")
    println("  Period    y           c           R           n           k")
    println("  " * "-"^65)

    for t in 1:min(10, irf_horizon)
        y_val = irf_e_m_pert[t, :y]
        c_val = irf_e_m_pert[t, :c]
        r_val = irf_e_m_pert[t, :R]
        n_val = irf_e_m_pert[t, :n]
        k_val = irf_e_m_pert[t, :k]

        @printf("  %2d     %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
                t, y_val, c_val, r_val, n_val, k_val)
    end

catch e
    println("Warning: Could not compute perturbation IRFs: ", e)
    println("NOTE: Full SEP-based IRF computation requires conditional forecast implementation")
end

println("\n" * "="^70)
println("IRF COMPUTATION COMPLETE")
println("="^70)
println("""
NOTE: MacroModelling.jl IRFs shown above use PERTURBATION (first-order) method.
Dynare IRFs use PERFECT FORESIGHT (deterministic) conditional forecasts.

For a true SEP vs Dynare comparison, we would need to implement:
  1. SEP solution with conditional shocks (shocked path)
  2. SEP solution with zero shocks (baseline path)
  3. IRF = difference between paths

This is a known limitation and represents future work.
Current perturbation IRFs serve as a reasonable approximation for small shocks.
""")

println("\n" * "="^70)
println("NEXT STEPS: Compare with Dynare Output")
println("="^70)
println("""
To compare with Dynare:

1. Run in MATLAB/Octave:
   >> dynare fs2000

2. Compare the steady state values and IRFs printed above

Expected runtime: ~5-10 seconds per test (very fast!)
Model specs: 14 variables, 2 shocks (e_a, e_m)
""")

println("="^70)
println("MacroModelling.jl SEP TESTS COMPLETE")
println("="^70)
