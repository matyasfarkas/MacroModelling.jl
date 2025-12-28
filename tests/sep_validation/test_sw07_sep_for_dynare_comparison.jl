# Test: Run SW07_HLT SEP in MacroModelling.jl for Dynare comparison
# This matches the extended_path() tests in Smets_Wouters_2007_HLT.mod

using MacroModelling
using Printf

println("="^70)
println("MacroModelling.jl SEP TESTS: SW07_HLT")
println("Matching Dynare extended_path(periods=10, order={1,2})")
println("="^70)

# Load the SW07_HLT model
include("../../models/Smets_Wouters_2007_HLT.jl")

println("\n1. Model loaded successfully")
println("   Model variables: ", length(Smets_Wouters_2007_HLT.var))
println("   Model shocks: ", Smets_Wouters_2007_HLT.exo)
println("   Model parameters: ", length(Smets_Wouters_2007_HLT.parameters))

# Key variables to report (matching Dynare output)
test_vars = [:y, :c, :inve, :pinf, :lab]

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

solve!(Smets_Wouters_2007_HLT,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

sep_sol_1 = Smets_Wouters_2007_HLT.solution.perturbation.stochastic_extended_path

println("\nSolution status:")
println("  - Convergence: ", sep_sol_1.convergence_flag ? "✓" : "✗")
println("  - Final error: ", sep_sol_1.final_error)
println("  - Runtime: ", sep_sol_1.runtime_seconds, " seconds")

# Extract steady state values
layout = sep_sol_1.layout
yss_indices = 1:layout.ny_
yss_1 = sep_sol_1.Y[yss_indices]

println("\nKey steady state values (SEP order=1):")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    var_idx = findfirst(==(var), Smets_Wouters_2007_HLT.var)
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

solve!(Smets_Wouters_2007_HLT,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 2,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

sep_sol_2 = Smets_Wouters_2007_HLT.solution.perturbation.stochastic_extended_path

println("\nSolution status:")
println("  - Convergence: ", sep_sol_2.convergence_flag ? "✓" : "✗")
println("  - Final error: ", sep_sol_2.final_error)
println("  - Runtime: ", sep_sol_2.runtime_seconds, " seconds")

# Extract steady state values
layout_2 = sep_sol_2.layout
yss_indices_2 = 1:layout_2.ny_
yss_2 = sep_sol_2.Y[yss_indices_2]

println("\nKey steady state values (SEP order=2):")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    var_idx = findfirst(==(var), Smets_Wouters_2007_HLT.var)
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
    var_idx = findfirst(==(var), Smets_Wouters_2007_HLT.var)
    if !isnothing(var_idx)
        val1 = yss_1[var_idx]
        val2 = yss_2[var_idx]
        diff = val2 - val1
        @printf("  %-12s  %12.8f  %12.8f  %12.8f\n",
                string(var), val1, val2, diff)
    end
end

# ============================================================================
# INSTRUCTIONS FOR DYNARE COMPARISON
# ============================================================================
println("\n" * "="^70)
println("NEXT STEPS: Dynare Validation")
println("="^70)
println("""
To compare with Dynare:

1. Run the exported .mod file in MATLAB/Octave:
   >> dynare Smets_Wouters_2007_HLT.mod

2. Compare the output:
   - Look for "Key steady state values (SEP order=1)" in Dynare output
   - Look for "Key steady state values (SEP order=2)" in Dynare output
   - Compare with the values above

3. Expected results:
   - Values should match closely if implementations are consistent
   - Small differences may arise from:
     * Numerical precision
     * Quadrature method (GH vs Unscented)
     * Newton solver tolerances
     * Initial guess strategies

4. Large differences would indicate:
   - Shock scaling issues
   - Different interpretation of "order" parameter
   - Fundamental algorithmic differences

File location: ./Smets_Wouters_2007_HLT.mod
""")

println("="^70)
println("MacroModelling.jl SEP TESTS COMPLETE")
println("="^70)
