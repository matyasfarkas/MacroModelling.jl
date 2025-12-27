# Test: Replicate Dynare's rs.mod SEP tests
# Compare MacroModelling SEP vs Dynare extended_path()

using MacroModelling
using Printf

println("="^70)
println("REPLICATING DYNARE SEP TESTS: rs.mod")
println("="^70)

# Load the translated model
include("/tmp/rstrue_clean.jl")

# Model instance already created by include()
println("\n1. Model loaded successfully")

println("   Model variables: ", length(rstrue_clean.var))
println("   Model shocks: ", rstrue_clean.exo)
println("   Model parameters: ", length(rstrue_clean.parameters))

# Dynare Test 1: extended_path(periods=10, order=1)
println("\n2. Running SEP(1): extended_path(periods=10, order=1)...")
println("   (T=10, Lbr=1, nnodes=3)")

solve!(rstrue_clean,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

println("   ✓ SEP(1) solved")

# Extract steady state and solution
sep_sol_1 = rs_clean.solution.perturbation.stochastic_extended_path
println("\n3. SEP(1) Solution summary:")
println("   Newton iterations: ", sep_sol_1.iterations)
println("   Final residual: ", sep_sol_1.residual)
println("   Solution vector size: ", length(sep_sol_1.Y))

# Show steady state for key variables
test_vars = [:V, :lC, :lY, :lpi, :Int]
println("\n4. Steady state values (SEP order 1):")
println("   Variable      Value")
println("   " * "-"^30)

layout = sep_sol_1.layout
yss_indices = 1:layout.ny_
yss = sep_sol_1.Y[yss_indices]

for var in test_vars
    var_idx = findfirst(==(var), rstrue_clean.var)
    if !isnothing(var_idx)
        @printf("   %-12s  %12.8f\n", string(var), yss[var_idx])
    end
end

# Dynare Test 2: extended_path(periods=10, order=2)
println("\n5. Running SEP(2): extended_path(periods=10, order=2)...")
println("   (T=10, Lbr=2, nnodes=3)")

solve!(rstrue_clean,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 2,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)

println("   ✓ SEP(2) solved")

# Extract second solution
sep_sol_2 = rs_clean.solution.perturbation.stochastic_extended_path
println("\n6. SEP(2) Solution summary:")
println("   Newton iterations: ", sep_sol_2.iterations)
println("   Final residual: ", sep_sol_2.residual)
println("   Solution vector size: ", length(sep_sol_2.Y))

# Show steady state for comparison
println("\n7. Steady state values (SEP order 2):")
println("   Variable      Value")
println("   " * "-"^30)

layout_2 = sep_sol_2.layout
yss_indices_2 = 1:layout_2.ny_
yss_2 = sep_sol_2.Y[yss_indices_2]

for var in test_vars
    var_idx = findfirst(==(var), rstrue_clean.var)
    if !isnothing(var_idx)
        @printf("   %-12s  %12.8f\n", string(var), yss_2[var_idx])
    end
end

# Compare the two solutions
println("\n8. COMPARISON: SEP(1) vs SEP(2)")
println("   Variable      SEP(1)         SEP(2)         Difference")
println("   " * "-"^65)

for var in test_vars
    var_idx = findfirst(==(var), rstrue_clean.var)
    if !isnothing(var_idx)
        val1 = yss[var_idx]
        val2 = yss_2[var_idx]
        diff = val2 - val1
        @printf("   %-12s  %12.8f  %12.8f  %12.8f\n",
                string(var), val1, val2, diff)
    end
end

println("\n" * "="^70)
println("INTERPRETATION:")
println("="^70)
println("""
Dynare's extended_path with order=1 means branching for 1 period (Lbr=1).
With order=2, branching continues for 2 periods (Lbr=2).

Higher order should capture more nonlinear effects from future uncertainty.
Differences between SEP(1) and SEP(2) indicate the importance of extended
branching horizons.

Next step: Compare these results with Dynare's output if available.
""")

println("="^70)
println("TEST COMPLETE")
println("="^70)
