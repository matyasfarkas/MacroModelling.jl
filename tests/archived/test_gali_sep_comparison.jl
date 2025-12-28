# Test: SEP on Gali_2015_chapter_3_nonlinear for Dynare comparison
# Small model - fast execution for validation

using MacroModelling
using Printf

println("="^70)
println("MacroModelling.jl SEP TEST: Gali_2015_chapter_3_nonlinear")
println("Matching Dynare extended_path(periods=10, order={1,2})")
println("="^70)

# Load the Gali model
include("models/Gali_2015_chapter_3_nonlinear.jl")

println("\n1. Model loaded successfully")
println("   Model variables: ", length(Gali_2015_chapter_3_nonlinear.var))
println("   Model shocks: ", Gali_2015_chapter_3_nonlinear.exo)
println("   Model parameters: ", length(Gali_2015_chapter_3_nonlinear.parameters))

# Key variables to report (matching typical macro variables)
test_vars = [:Y, :C, :Pi, :R, :N]

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

solve!(Gali_2015_chapter_3_nonlinear,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

sep_sol_1 = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path

println("\nSolution status:")
@printf("  - Convergence: %s\n", sep_sol_1.convergence_flag == 1 ? "SUCCESS" : "FAILED")
@printf("  - Final error: %.3e\n", sep_sol_1.final_error)
@printf("  - Runtime: %.2f seconds\n\n", sep_sol_1.runtime_seconds)

# Extract steady state values
layout = sep_sol_1.layout
yss_indices = 1:layout.ny_
yss_1 = sep_sol_1.Y[yss_indices]

println("Key steady state values (SEP order=1):")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    var_idx = findfirst(==(var), Gali_2015_chapter_3_nonlinear.var)
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

solve!(Gali_2015_chapter_3_nonlinear,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 2,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

sep_sol_2 = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path

println("\nSolution status:")
@printf("  - Convergence: %s\n", sep_sol_2.convergence_flag == 1 ? "SUCCESS" : "FAILED")
@printf("  - Final error: %.3e\n", sep_sol_2.final_error)
@printf("  - Runtime: %.2f seconds\n\n", sep_sol_2.runtime_seconds)

# Extract steady state values
layout_2 = sep_sol_2.layout
yss_indices_2 = 1:layout_2.ny_
yss_2 = sep_sol_2.Y[yss_indices_2]

println("Key steady state values (SEP order=2):")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    var_idx = findfirst(==(var), Gali_2015_chapter_3_nonlinear.var)
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
    var_idx = findfirst(==(var), Gali_2015_chapter_3_nonlinear.var)
    if !isnothing(var_idx)
        val1 = yss_1[var_idx]
        val2 = yss_2[var_idx]
        diff = val2 - val1
        @printf("  %-12s  %12.8f  %12.8f  %12.8f\n",
                string(var), val1, val2, diff)
    end
end

println("\n" * "="^70)
println("NEXT STEPS: Compare with Dynare Output")
println("="^70)
println("""
To compare with Dynare:

1. Run in MATLAB/Octave:
   >> dynare Gali_2015_chapter_3_nonlinear.mod

2. Compare the steady state values printed above

Expected runtime: ~1-2 minutes per test (much faster than SW07!)
""")

println("="^70)
println("MacroModelling.jl SEP TESTS COMPLETE")
println("="^70)
