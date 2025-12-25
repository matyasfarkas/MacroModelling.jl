# Test SEP Integration - Phase 2: IRF Extraction
# Compare SEP IRF with first-order perturbation IRF

println("="^70)
println("PHASE 2: SEP IRF Extraction Test - Gali Model")
println("="^70)
println()

using MacroModelling
using Printf

# Load Gali model
println("Loading Gali_2015_chapter_3_nonlinear model...")
include("models/Gali_2015_chapter_3_nonlinear.jl")
println("✓ Model loaded")
println()

# Solve with both algorithms
println("-"^70)
println("Test 1: Solve with first-order perturbation")
println("-"^70)

solve!(Gali_2015_chapter_3_nonlinear, algorithm=:first_order)
println("✓ First-order solution computed")
println()

println("-"^70)
println("Test 2: Solve with SEP")
println("-"^70)

solve!(Gali_2015_chapter_3_nonlinear,
       algorithm=:stochastic_extended_path,
       sep_periods=20,  # Longer horizon for better IRF
       sep_order=1,
       sep_nnodes=3,
       silent=false)

sep_sol = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path
println("✓ SEP solution computed (T=$(sep_sol.periods))")
println()

# Extract IRF using SEP
println("-"^70)
println("Test 3: Extract SEP IRF")
println("-"^70)

try
    # Test the new get_sep_irf function
    variables = [:Y, :Pi, :R, :N]  # Key variables
    shock = :eps_a  # Technology shock

    sep_irf = get_sep_irf(Gali_2015_chapter_3_nonlinear, shock, 1.0;
                          variables=variables, periods=20)

    println("✓ SEP IRF extracted successfully")
    println("  Variables: ", variables)
    println("  Shock: ", shock)
    println("  Periods: ", size(sep_irf, 2))
    println()

    # Display first few periods
    println("SEP IRF (first 5 periods):")
    for (i, v) in enumerate(variables)
        print("  $v: ")
        for t in 1:min(5, size(sep_irf, 2))
            @printf("%.6f  ", sep_irf[i, t])
        end
        println()
    end
    println()

    println("✓ TEST 3 PASSED: SEP IRF extraction works")

catch e
    println("✗ TEST 3 FAILED:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println()
println("="^70)
println("Phase 2 Testing Complete")
println("="^70)
