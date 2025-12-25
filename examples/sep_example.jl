# Stochastic Extended Path (SEP) - Complete Working Example
# This example demonstrates SEP usage with the Gali (2015) New Keynesian model

using MacroModelling
using Printf

println("="^70)
println("SEP Example: Gali (2015) New Keynesian Model")
println("="^70)
println()

# Load the model
println("Loading Gali_2015_chapter_3_nonlinear model...")
include("../models/Gali_2015_chapter_3_nonlinear.jl")
println("✓ Model loaded successfully")
println()

# =============================================================================
# PART 1: Solve with Different Methods
# =============================================================================
println("="^70)
println("PART 1: Comparing Solution Methods")
println("="^70)
println()

# Solve with first-order perturbation (for comparison)
println("1.1 First-Order Perturbation Solution")
println("-"^70)
@time solve!(Gali_2015_chapter_3_nonlinear, algorithm=:first_order, silent=true)
println("✓ First-order solution computed")
println()

# Solve with SEP
println("1.2 Stochastic Extended Path Solution")
println("-"^70)
@time solve!(Gali_2015_chapter_3_nonlinear,
             algorithm=:stochastic_extended_path,
             sep_periods=20,     # 20-period horizon
             sep_order=1,        # First-order branching
             sep_nnodes=3,       # 3 GH nodes per shock
             sep_maxit=80,       # Max iterations
             sep_tol=1e-7,       # Tolerance
             silent=false)

# Check convergence
sep_sol = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path
println()
println("SEP Solution Summary:")
println("  Convergence flag: ", sep_sol.convergence_flag,
        " (", sep_sol.convergence_flag == 0 ? "SUCCESS" : "FAILED", ")")
println("  Final error: ", sep_sol.final_error)
println("  Runtime: ", round(sep_sol.runtime_seconds, digits=3), " seconds")
println("  Horizon (T): ", sep_sol.periods)
println("  Order (L): ", sep_sol.order)
println("  GH nodes: ", sep_sol.nnodes)
println()

# =============================================================================
# PART 2: Impulse Response Functions
# =============================================================================
println("="^70)
println("PART 2: Impulse Response Functions")
println("="^70)
println()

# Select variables and shock
variables = [:Y, :Pi, :R, :N]  # Output, Inflation, Interest Rate, Labor
shock = :eps_a                  # Technology shock

println("2.1 Technology Shock IRF (First-Order)")
println("-"^70)
irf_fo = get_irf(Gali_2015_chapter_3_nonlinear;
                 shocks=shock,
                 variables=variables,
                 periods=20)
println("✓ First-order IRF computed")
println()

println("2.2 Technology Shock IRF (SEP)")
println("-"^70)
irf_sep = get_sep_irf(Gali_2015_chapter_3_nonlinear,
                      shock, 1.0;  # 1 std deviation shock
                      variables=variables,
                      periods=20)
println("✓ SEP IRF computed")
println()

println("2.3 Comparing Impact Responses (t=1)")
println("-"^70)
println("Variable | First-Order  | SEP          | Abs. Diff")
println("-"^70)
for (i, v) in enumerate(variables)
    fo_val = irf_fo[i, 2, 1]      # t=1 (period 2 in the array)
    sep_val = irf_sep[i, 2]        # t=1 (period 2 in the array)
    diff = abs(sep_val - fo_val)
    @printf("%-8s | %12.8f | %12.8f | %10.2e\n", v, fo_val, sep_val, diff)
end
println()

println("2.4 SEP IRF - First 10 Periods")
println("-"^70)
println("Period ", join([@sprintf("%8s", v) for v in variables], " "))
println("-"^70)
for t in 1:min(10, size(irf_sep, 2))
    @printf("%6d ", t-1)
    for i in 1:length(variables)
        @printf("%8.4f ", irf_sep[i, t])
    end
    println()
end
println()

# =============================================================================
# PART 3: Stochastic Simulations
# =============================================================================
println("="^70)
println("PART 3: Stochastic Simulations")
println("="^70)
println()

println("3.1 Generate Simulation Paths (Levels)")
println("-"^70)
sim_levels = get_sep_simulation(Gali_2015_chapter_3_nonlinear;
                                variables=variables,
                                periods=20,
                                nsims=3,
                                levels=true)
println("✓ Generated ", size(sim_levels, 3), " simulation paths")
println("  Variables: ", variables)
println("  Periods: ", size(sim_levels, 2) - 1)
println()

println("Simulation #1 - First 10 periods (levels):")
println("-"^70)
println("Period ", join([@sprintf("%8s", v) for v in variables], " "))
println("-"^70)
for t in 1:min(10, size(sim_levels, 2))
    @printf("%6d ", t-1)
    for i in 1:length(variables)
        @printf("%8.5f ", sim_levels[i, t, 1])
    end
    println()
end
println()

println("3.2 Generate Simulation Paths (Deviations from SS)")
println("-"^70)
sim_dev = get_sep_simulation(Gali_2015_chapter_3_nonlinear;
                             variables=variables,
                             periods=20,
                             nsims=1,
                             levels=false)

println("Simulation #1 - First 10 periods (deviations):")
println("-"^70)
println("Period ", join([@sprintf("%8s", v) for v in variables], " "))
println("-"^70)
for t in 1:min(10, size(sim_dev, 2))
    @printf("%6d ", t-1)
    for i in 1:length(variables)
        @printf("%8.5f ", sim_dev[i, t, 1])
    end
    println()
end
println()

# =============================================================================
# PART 4: Performance Comparison
# =============================================================================
println("="^70)
println("PART 4: Performance Comparison")
println("="^70)
println()

println("Testing different SEP configurations...")
println()

configs = [
    (periods=10, order=1, nnodes=3, name="Conservative (fast)"),
    (periods=20, order=1, nnodes=3, name="Balanced"),
    (periods=30, order=1, nnodes=3, name="Accurate (slow)"),
]

for config in configs
    println("Config: ", config.name)
    println("  T=$(config.periods), L=$(config.order), nodes=$(config.nnodes)")

    time_start = time()
    solve!(Gali_2015_chapter_3_nonlinear,
           algorithm=:stochastic_extended_path,
           sep_periods=config.periods,
           sep_order=config.order,
           sep_nnodes=config.nnodes,
           silent=true)
    runtime = time() - time_start

    sep = Gali_2015_chapter_3_nonlinear.solution.perturbation.stochastic_extended_path
    println("  Runtime: ", round(runtime, digits=3), " sec")
    println("  Error: ", sep.final_error)
    println("  Converged: ", sep.convergence_flag == 0 ? "Yes" : "No")
    println()
end

# =============================================================================
# Summary
# =============================================================================
println("="^70)
println("EXAMPLE COMPLETE")
println("="^70)
println()
println("Summary:")
println("  ✓ Solved model with both first-order and SEP methods")
println("  ✓ Extracted and compared impulse response functions")
println("  ✓ Generated stochastic simulation paths")
println("  ✓ Tested different SEP configurations")
println()
println("The SEP method provides globally accurate solutions for DSGE models,")
println("capturing nonlinear dynamics that local approximations may miss.")
println()
println("="^70)
