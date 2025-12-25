# Test SEP Integration - Phase 3: Simulation
# Test SEP simulation function

println("="^70)
println("PHASE 3: SEP Simulation Test - Gali Model")
println("="^70)
println()

using MacroModelling
using Printf

# Load Gali model
println("Loading Gali_2015_chapter_3_nonlinear model...")
include("models/Gali_2015_chapter_3_nonlinear.jl")
println("✓ Model loaded")
println()

# Solve with SEP
println("-"^70)
println("Test 1: Solve with SEP")
println("-"^70)

solve!(Gali_2015_chapter_3_nonlinear,
       algorithm=:stochastic_extended_path,
       sep_periods=20,
       sep_order=1,
       sep_nnodes=3,
       silent=false)

sep_sol = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path
println("✓ SEP solution computed (T=$(sep_sol.periods))")
println()

# Test SEP simulation
println("-"^70)
println("Test 2: Generate SEP simulation")
println("-"^70)

try
    # Test the new get_sep_simulation function
    variables = [:Y, :Pi, :R, :N]  # Key variables
    nsims = 3  # Generate 3 simulation paths

    sep_sim = get_sep_simulation(Gali_2015_chapter_3_nonlinear;
                                 variables=variables,
                                 periods=20,
                                 nsims=nsims,
                                 levels=true)

    println("✓ SEP simulation generated successfully")
    println("  Variables: ", variables)
    println("  Periods: ", size(sep_sim, 2))
    println("  Simulations: ", nsims)
    println()

    # Display first few periods of first simulation
    println("SEP Simulation #1 (first 5 periods, levels):")
    for (i, v) in enumerate(variables)
        print("  $v: ")
        for t in 1:min(5, size(sep_sim, 2))
            @printf("%.6f  ", sep_sim[i, t, 1])
        end
        println()
    end
    println()

    # Test simulation in deviations
    sep_sim_dev = get_sep_simulation(Gali_2015_chapter_3_nonlinear;
                                     variables=variables,
                                     periods=20,
                                     nsims=1,
                                     levels=false)

    println("SEP Simulation in deviations (first 5 periods):")
    for (i, v) in enumerate(variables)
        print("  $v: ")
        for t in 1:min(5, size(sep_sim_dev, 2))
            @printf("%.6f  ", sep_sim_dev[i, t, 1])
        end
        println()
    end
    println()

    println("✓ TEST 2 PASSED: SEP simulation works")

catch e
    println("✗ TEST 2 FAILED:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println()
println("="^70)
println("Phase 3 Testing Complete")
println("="^70)
