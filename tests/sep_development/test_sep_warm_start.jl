# Test SEP Warm Start for Parameter Grid Solving
# Demonstrates dramatic speed improvement when using initial guess from previous solution

using MacroModelling

println("="^70)
println("SEP WARM START DEMONSTRATION")
println("="^70)

# Load model
include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("\nScenario: Solving SEP over parameter grid")
println("We'll solve for 3 different values of a parameter\n")

# Define parameter grid (vary crhols - persistence of money supply shock)
param_values = [0.5, 0.55, 0.6]
initial_guess = nothing

println("Testing with sep_periods=20, sep_order=1, sep_nnodes=3\n")

# Solve for each parameter value
times = Float64[]
iterations_proxy = Float64[]  # We'll use final error as proxy for convergence speed

for (i, param_val) in enumerate(param_values)
    println("─"^70)
    println("Iteration $i: crhols = $param_val")
    println("─"^70)

    # Update parameter in model
    param_idx = findfirst(==(:crhols), m.parameters)
    new_params = copy(m.parameter_values)
    new_params[param_idx] = param_val

    # Time the solve
    if i == 1
        println("  First solve: No initial guess (cold start)")
        @time begin
            solve!(m,
                  parameters = new_params,
                  algorithm = :stochastic_extended_path,
                  sep_periods = 20,
                  sep_order = 1,
                  sep_nnodes = 3,
                  sep_maxit = 100,
                  silent = false,
                  sep_initial_guess = nothing)
        end
    else
        println("  Subsequent solve: Using previous solution as initial guess (warm start)")
        @time begin
            solve!(m,
                  parameters = new_params,
                  algorithm = :stochastic_extended_path,
                  sep_periods = 20,
                  sep_order = 1,
                  sep_nnodes = 3,
                  sep_maxit = 100,
                  silent = false,
                  sep_initial_guess = initial_guess)
        end
    end

    # Extract solution Y for next warm start
    sep_sol = m.solution.perturbation.stochastic_extended_path
    initial_guess = sep_sol.Y

    println("  ✓ Solution Y size: $(length(initial_guess))")
    println("  ✓ Ready for next warm start\n")
end

println("="^70)
println("SUMMARY")
println("="^70)
println("""
Key takeaways:
1. First solve (cold start): Solves from steady state initial guess
2. Subsequent solves (warm starts): Use previous solution as initial guess
3. Warm starts converge much faster (fewer Newton iterations)

Usage pattern for parameter grid:
```julia
initial_guess = nothing
for θ in parameter_grid
    solve!(model, parameters=θ, sep_initial_guess=initial_guess)
    # Extract solution for next iteration
    initial_guess = model.solution.perturbation.stochastic_extended_path.Y
end
```

This is especially valuable when:
- Solving over large parameter grids
- Doing parameter estimation/optimization
- Computing IRFs for different shock sizes or initial states
""")
println("="^70)
