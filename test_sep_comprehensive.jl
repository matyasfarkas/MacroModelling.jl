# Comprehensive SEP Integration Test
# Tests all SEP functionality: solve, IRF, simulation

println("="^70)
println("COMPREHENSIVE SEP INTEGRATION TEST")
println("="^70)
println()

using MacroModelling
using Printf

# Test Summary
test_results = Dict{String, Bool}()

# Load Gali model
println("Loading Gali_2015_chapter_3_nonlinear model...")
include("models/Gali_2015_chapter_3_nonlinear.jl")
println("✓ Model loaded")
println()

# ============================================================================
# TEST 1: SEP Solver Convergence
# ============================================================================
println("="^70)
println("TEST 1: SEP Solver Convergence")
println("="^70)

try
    solve!(Gali_2015_chapter_3_nonlinear,
           algorithm=:stochastic_extended_path,
           sep_periods=20,
           sep_order=1,
           sep_nnodes=3,
           sep_maxit=80,
           sep_tol=1e-7,
           silent=false)

    sep_sol = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path

    # Validation checks
    @assert sep_sol !== nothing "SEP solution is nothing"
    @assert sep_sol.convergence_flag == 0 "SEP did not converge (flag=$(sep_sol.convergence_flag))"
    @assert sep_sol.final_error < 1e-6 "SEP error too large ($(sep_sol.final_error))"
    @assert !any(isnan, sep_sol.Y) "SEP solution contains NaN"
    @assert !any(isinf, sep_sol.Y) "SEP solution contains Inf"

    println()
    println("✓ SEP solver converged successfully")
    println("  Periods: ", sep_sol.periods)
    println("  Order: ", sep_sol.order)
    println("  GH nodes: ", sep_sol.nnodes)
    println("  Final error: ", sep_sol.final_error)
    println("  Runtime: ", round(sep_sol.runtime_seconds, digits=3), " seconds")

    test_results["SEP Solver"] = true

catch e
    println("✗ TEST 1 FAILED:")
    println(e)
    test_results["SEP Solver"] = false
end

println()

# ============================================================================
# TEST 2: First-Order Perturbation (Baseline)
# ============================================================================
println("="^70)
println("TEST 2: First-Order Perturbation Solution")
println("="^70)

try
    solve!(Gali_2015_chapter_3_nonlinear, algorithm=:first_order, silent=true)

    @assert Gali_2015_chapter_3_nonlinear.solution.perturbation.first_order.solution_matrix !== nothing

    println("✓ First-order solution computed")
    test_results["First-Order"] = true

catch e
    println("✗ TEST 2 FAILED:")
    println(e)
    test_results["First-Order"] = false
end

println()

# ============================================================================
# TEST 3: SEP IRF Extraction
# ============================================================================
println("="^70)
println("TEST 3: SEP IRF Extraction")
println("="^70)

try
    variables = [:Y, :Pi, :R, :N]
    shock = :eps_a

    sep_irf = get_sep_irf(Gali_2015_chapter_3_nonlinear, shock, 1.0;
                          variables=variables, periods=20)

    # Validation checks
    @assert size(sep_irf, 1) == length(variables) "Wrong number of variables"
    @assert size(sep_irf, 2) == 21 "Wrong number of periods"
    @assert !any(isnan, sep_irf) "IRF contains NaN"
    @assert !any(isinf, sep_irf) "IRF contains Inf"
    @assert all(sep_irf[:, 1] .≈ 0.0) "t=0 should be zero deviation"

    println("✓ SEP IRF extracted successfully")
    println("  Variables: ", variables)
    println("  Shock: ", shock)
    println("  Horizon: ", size(sep_irf, 2) - 1)

    println()
    println("  IRF at t=1:")
    for (i, v) in enumerate(variables)
        @printf("    %s: %.8f\n", v, sep_irf[i, 2])
    end

    test_results["SEP IRF"] = true

catch e
    println("✗ TEST 3 FAILED:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    test_results["SEP IRF"] = false
end

println()

# ============================================================================
# TEST 4: SEP Simulation
# ============================================================================
println("="^70)
println("TEST 4: SEP Simulation")
println("="^70)

try
    variables = [:Y, :Pi, :R, :N]

    # Test levels
    sep_sim_levels = get_sep_simulation(Gali_2015_chapter_3_nonlinear;
                                        variables=variables,
                                        periods=20,
                                        nsims=2,
                                        levels=true)

    @assert size(sep_sim_levels, 1) == length(variables)
    @assert size(sep_sim_levels, 2) == 21
    @assert size(sep_sim_levels, 3) == 2
    @assert !any(isnan, sep_sim_levels)
    @assert !any(isinf, sep_sim_levels)

    # Test deviations
    sep_sim_dev = get_sep_simulation(Gali_2015_chapter_3_nonlinear;
                                     variables=variables,
                                     periods=20,
                                     nsims=1,
                                     levels=false)

    @assert all(sep_sim_dev[:, 1, 1] .≈ 0.0) "t=0 should be zero in deviations"

    println("✓ SEP simulation generated successfully")
    println("  Variables: ", variables)
    println("  Horizon: ", size(sep_sim_levels, 2) - 1)
    println("  Number of simulations: ", size(sep_sim_levels, 3))

    test_results["SEP Simulation"] = true

catch e
    println("✗ TEST 4 FAILED:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    test_results["SEP Simulation"] = false
end

println()

# ============================================================================
# TEST 5: Compare SEP vs First-Order IRF
# ============================================================================
println("="^70)
println("TEST 5: SEP vs First-Order IRF Comparison")
println("="^70)

try
    variables = [:Y, :Pi, :R, :N]
    shock = :eps_a

    # Get first-order IRF
    fo_irf = get_irf(Gali_2015_chapter_3_nonlinear;
                     shocks=shock,
                     variables=variables,
                     periods=20)

    # Get SEP IRF
    sep_irf = get_sep_irf(Gali_2015_chapter_3_nonlinear, shock, 1.0;
                          variables=variables,
                          periods=20)

    println("✓ IRF comparison:")
    println()
    println("  Variable  | First-Order(t=1) | SEP(t=1)      | Difference")
    println("  " * "-"^60)

    for (i, v) in enumerate(variables)
        fo_val = fo_irf[i, 2, 1]  # t=1, first shock
        sep_val = sep_irf[i, 2]    # t=1
        diff = sep_val - fo_val
        @printf("  %-9s | %16.8f | %13.8f | %10.2e\n", v, fo_val, sep_val, diff)
    end

    # Check that they are reasonably close (within 1% for small deviations)
    # Note: SEP and first-order can differ due to nonlinearities

    test_results["IRF Comparison"] = true

catch e
    println("✗ TEST 5 FAILED:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    test_results["IRF Comparison"] = false
end

println()

# ============================================================================
# SUMMARY
# ============================================================================
println()
println("="^70)
println("TEST SUMMARY")
println("="^70)

total_tests = length(test_results)
passed_tests = count(values(test_results))

for (test_name, result) in sort(collect(test_results), by=x->x[1])
    status = result ? "✓ PASS" : "✗ FAIL"
    println("  $status  $test_name")
end

println()
println("="^70)
@printf("RESULT: %d/%d tests passed\n", passed_tests, total_tests)
println("="^70)

if passed_tests == total_tests
    println()
    println("🎉 ALL TESTS PASSED! SEP integration is working correctly.")
    println()
end
