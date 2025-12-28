# Test SEP Integration with Gali_2015_chapter_3_nonlinear Model
# Phase 1: Basic SEP solve and validation

println("="^70)
println("PHASE 1: SEP Integration Test - Gali Model")
println("="^70)
println()

using MacroModelling

# Load Gali model
println("Loading Gali_2015_chapter_3_nonlinear model...")
include("models/Gali_2015_chapter_3_nonlinear.jl")
println("✓ Model loaded")
println()

# Display model info
println("Model characteristics:")
println("  Variables: ", length(Gali_2015_chapter_3_nonlinear.var))
println("  Shocks: ", length(Gali_2015_chapter_3_nonlinear.exo))
println("  Parameters: ", length(Gali_2015_chapter_3_nonlinear.parameters))
println()

# Test 1: Solve with SEP (small problem for quick test)
println("-"^70)
println("Test 1: SEP solve with periods=10, order=1, nnodes=3")
println("-"^70)

try
    solve!(Gali_2015_chapter_3_nonlinear,
           algorithm=:stochastic_extended_path,
           sep_periods=10,
           sep_order=1,
           sep_nnodes=3)

    # Check solution
    sep_sol = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path

    println()
    println("✓ SEP Solution Summary:")
    println("  Periods (T): ", sep_sol.periods)
    println("  Branching order (L): ", sep_sol.order)
    println("  GH nodes: ", sep_sol.nnodes)
    println("  Convergence flag: ", sep_sol.convergence_flag,
            sep_sol.convergence_flag == 0 ? " (SUCCESS)" : " (FAILED)")
    println("  Final error: ", sep_sol.final_error)
    println("  Runtime: ", round(sep_sol.runtime_seconds, digits=3), " seconds")
    println("  Solution vector size: ", length(sep_sol.Y))
    println()

    # Validate solution
    if sep_sol.convergence_flag == 0
        println("✓ TEST 1 PASSED: SEP converged successfully")
    else
        println("✗ TEST 1 FAILED: SEP did not converge")
    end

    if sep_sol.final_error < 1e-7
        println("✓ TEST 2 PASSED: Final error below tolerance")
    else
        println("✗ TEST 2 FAILED: Final error too high: ", sep_sol.final_error)
    end

    if !any(isnan, sep_sol.Y) && !any(isinf, sep_sol.Y)
        println("✓ TEST 3 PASSED: Solution contains no NaN/Inf values")
    else
        println("✗ TEST 3 FAILED: Solution contains NaN or Inf values")
    end

catch e
    println("✗ ERROR during SEP solve:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println()
println("="^70)
println("Phase 1 Testing Complete")
println("="^70)
