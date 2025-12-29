# Stochastic Extended Path (SEP) - Global Nonlinear Solutions

## Overview

The Stochastic Extended Path (SEP) method is a global nonlinear solution algorithm for DSGE models. Unlike perturbation methods (first-order, second-order, etc.) which provide local approximations around the steady state, SEP computes globally accurate solutions by constructing a tree of possible future paths under uncertainty.

### When to Use SEP

- **Strong nonlinearities**: Models where second or third-order approximations may not be sufficient
- **Far from steady state**: Analysis of dynamics far from equilibrium
- **Occasionally binding constraints**: Models with inequality constraints (use with OBC features)
- **Policy experiments**: Evaluation of large policy changes or structural breaks

### Advantages
- Globally accurate (not limited to neighborhood of steady state)
- Handles strong nonlinearities
- No approximation error in the model equations (only in expectation integration)

### Limitations
- Computationally more expensive than perturbation methods
- Current implementation follows expected/mean paths for IRFs and simulations
- Tree size grows exponentially with branching order

## Basic Usage

### 1. Solve a Model with SEP

```julia
using MacroModelling

# Load your model
@model MyModel begin
    # ... your model equations ...
end

# Solve with SEP
solve!(MyModel,
       algorithm = :stochastic_extended_path,
       sep_periods = 20,    # Horizon (T)
       sep_order = 1,       # Branching order (L)
       sep_nnodes = 3)      # Gauss-Hermite nodes per shock
```

### 2. SEP Parameters

- `sep_periods::Int = 20`: Planning horizon T. The model solves for T periods into the future.
- `sep_order::Int = 1`: Branching order L. Number of shocks that can be non-zero simultaneously.
  - `L=1`: Only one shock at a time (recommended starting point)
  - `L=2`: Pairwise shock interactions
  - Higher L increases accuracy but computational cost grows as `(nshocks * nnodes)^L`
- `sep_nnodes::Int = 3`: Number of Gauss-Hermite quadrature nodes per shock dimension
  - More nodes = better approximation of expectations
  - Typical values: 3, 5, or 7
- `sep_maxit::Int = 80`: Maximum Newton iterations
- `sep_tol::Float64 = 1e-7`: Convergence tolerance
- `silent::Bool = false`: Suppress solver output

### 3. Check Convergence

```julia
# Access the SEP solution
sep_sol = MyModel.solution.perturbation.stochastic_extended_path

# Check convergence
if sep_sol.convergence_flag == 0
    println("✓ SEP converged successfully")
    println("  Final error: ", sep_sol.final_error)
    println("  Runtime: ", sep_sol.runtime_seconds, " seconds")
else
    @warn "SEP did not converge (flag: $(sep_sol.convergence_flag))"
end
```

Convergence flags:
- `0`: Success
- `1`: Maximum iterations reached
- `2`: Domain error (NaN/Inf encountered)

## Impulse Response Functions

### Extract SEP IRF

```julia
# Compute IRF for a technology shock
irf = get_sep_irf(MyModel,
                  :eps_z,           # Shock name
                  1.0;              # Shock size (shock units)
                  variables = [:c, :k, :y],  # Variables to plot
                  periods = 40)     # IRF horizon

# IRF is returned as deviations from steady state
# Access values: irf[variable_index, period]
```

`get_sep_irf` interprets `shock_size` in shock units (consistent with `get_irf`).
If you store shock standard deviations in parameters named `z_<shock>` and want
Dynare-style scaling, pass `shock_scaling = :parameter`.

### Example: Technology Shock IRF

```julia
using MacroModelling

# Gali (2015) New Keynesian model
include("models/Gali_2015_chapter_3_nonlinear.jl")

# Solve with SEP
solve!(Gali_2015_chapter_3_nonlinear,
       algorithm = :stochastic_extended_path,
       sep_periods = 20,
       sep_order = 1,
       sep_nnodes = 3)

# Extract IRF for technology shock
variables = [:Y, :Pi, :R, :N]
irf = get_sep_irf(Gali_2015_chapter_3_nonlinear,
                  :eps_a, 1.0;
                  variables = variables,
                  periods = 20)

# Display impact effects
println("Technology shock - Impact on variables:")
for (i, v) in enumerate(variables)
    println("  $v: ", irf[i, 2])  # Period 2 = t+1 (first period after shock)
end
```

### Compare SEP vs First-Order IRF

```julia
# Solve with both methods
solve!(MyModel, algorithm = :first_order)
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 20, sep_order = 1, sep_nnodes = 3)

# Get IRFs
shock = :eps_z
vars = [:c, :k, :y]

irf_fo = get_irf(MyModel; shocks = shock, variables = vars, periods = 20)
irf_sep = get_sep_irf(MyModel, shock, 1.0; variables = vars, periods = 20)

# Compare impact responses
println("Shock: $shock")
println("Variable | First-Order | SEP        | Difference")
println("-" ^ 50)
for (i, v) in enumerate(vars)
    fo_val = irf_fo[i, 2, 1]
    sep_val = irf_sep[i, 2]
    diff = sep_val - fo_val
    @printf("%-8s | %11.6f | %10.6f | %10.2e\n", v, fo_val, sep_val, diff)
end
```

## Stochastic Simulations

### Generate Simulation Paths

```julia
# Generate stochastic simulations
sim = get_sep_simulation(MyModel;
                         variables = [:c, :k, :y],
                         periods = 100,
                         nsims = 10,      # Number of simulations
                         levels = true)   # true = levels, false = deviations

# Access simulations: sim[variable, period, simulation_number]
# Example: consumption in period 50 of simulation 3
c_50_3 = sim[1, 50, 3]
```

### Example: Plotting Simulation Paths

```julia
using MacroModelling
using Plots

# Solve model
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 40, sep_order = 1, sep_nnodes = 3)

# Generate simulations
nsims = 20
sim = get_sep_simulation(MyModel;
                         variables = [:Y],
                         periods = 40,
                         nsims = nsims,
                         levels = false)  # Deviations from SS

# Plot all paths
plot(0:40, sim[1, :, 1], label = "Sim 1",
     xlabel = "Period", ylabel = "Output deviation",
     title = "Stochastic Simulation Paths")
for s in 2:nsims
    plot!(0:40, sim[1, :, s], label = "Sim $s")
end
```

**Note**: The current SEP simulation implementation follows the expected/mean path through the stochastic tree. For fully stochastic simulations with random shock realizations, perturbation methods are recommended as they provide explicit policy functions.

## Performance Tips

### 1. Start with Conservative Settings

```julia
# Conservative (fast, less accurate)
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 10,    # Short horizon
       sep_order = 1,       # Only single shocks
       sep_nnodes = 3)      # Few nodes
```

### 2. Increase Accuracy Gradually

```julia
# More accurate (slower)
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 30,    # Longer horizon
       sep_order = 1,       # Still L=1
       sep_nnodes = 5)      # More nodes

# High accuracy (much slower)
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 40,
       sep_order = 2,       # Shock interactions
       sep_nnodes = 7)
```

### 3. Monitor Convergence

```julia
# If convergence is slow or fails:

# Option 1: Increase max iterations
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_maxit = 150)     # Default is 80

# Option 2: Relax tolerance slightly
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_tol = 1e-6)      # Default is 1e-7

# Option 3: Use better initial guess (solve first-order first)
solve!(MyModel, algorithm = :first_order)
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 20, sep_order = 1, sep_nnodes = 3)
```

## Complete Example: Gali (2015) Model

```julia
using MacroModelling

# Define Gali (2015) New Keynesian model
@model Gali_2015 begin
    # ... (model equations here)
end

# Compare solution methods
println("="^70)
println("Comparing First-Order and SEP Solutions")
println("="^70)

# 1. Solve with first-order perturbation
println("\n1. First-Order Perturbation...")
solve!(Gali_2015, algorithm = :first_order, silent = true)
println("✓ First-order solution computed")

# 2. Solve with SEP
println("\n2. Stochastic Extended Path...")
solve!(Gali_2015,
       algorithm = :stochastic_extended_path,
       sep_periods = 20,
       sep_order = 1,
       sep_nnodes = 3,
       silent = false)

sep_sol = Gali_2015.solution.perturbation.stochastic_extended_path
println("✓ SEP solution computed")
println("  Convergence: ", sep_sol.convergence_flag == 0 ? "SUCCESS" : "FAILED")
println("  Final error: ", sep_sol.final_error)
println("  Runtime: ", round(sep_sol.runtime_seconds, digits=3), " sec")

# 3. Compare IRFs
println("\n3. Technology Shock IRFs...")
shock = :eps_a
vars = [:Y, :Pi, :R, :N]

irf_fo = get_irf(Gali_2015; shocks = shock, variables = vars, periods = 20)
irf_sep = get_sep_irf(Gali_2015, shock, 1.0; variables = vars, periods = 20)

println("\nImpact responses (t=1):")
println("Variable | First-Order | SEP        | Difference")
println("-"^55)
for (i, v) in enumerate(vars)
    fo_val = irf_fo[i, 2, 1]
    sep_val = irf_sep[i, 2]
    @printf("%-8s | %11.8f | %10.8f | %10.2e\n",
            v, fo_val, sep_val, abs(sep_val - fo_val))
end

# 4. Generate simulations
println("\n4. Stochastic Simulations...")
sim = get_sep_simulation(Gali_2015;
                         variables = vars,
                         periods = 20,
                         nsims = 5,
                         levels = true)

println("✓ Generated ", size(sim, 3), " simulation paths")
println("  Variables: ", vars)
println("  Horizon: ", size(sim, 2) - 1, " periods")

println("\n" * "="^70)
println("Analysis Complete!")
println("="^70)
```

## Advanced Usage

### Accessing the SEP Solution Structure

```julia
sep_sol = MyModel.solution.perturbation.stochastic_extended_path

# Solution fields
sep_sol.Y                    # Full solution vector
sep_sol.layout               # Tree structure (SEPLayout)
sep_sol.periods              # Horizon T
sep_sol.order                # Branching order L
sep_sol.nnodes               # GH nodes per shock
sep_sol.convergence_flag     # 0 = success
sep_sol.final_error          # Maximum residual
sep_sol.runtime_seconds      # Solve time
```

### Adjusting for Model Size

Small models (< 10 variables):
```julia
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 40, sep_order = 1, sep_nnodes = 5)
```

Medium models (10-30 variables):
```julia
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 20, sep_order = 1, sep_nnodes = 3)
```

Large models (> 30 variables):
```julia
solve!(MyModel, algorithm = :stochastic_extended_path,
       sep_periods = 10, sep_order = 1, sep_nnodes = 3)
```

## Troubleshooting

### Convergence Issues

**Problem**: SEP fails to converge

**Solutions**:
1. Solve first-order perturbation first (provides better initialization)
2. Reduce `sep_periods` (shorter horizon is easier to solve)
3. Increase `sep_maxit` (allow more iterations)
4. Check model for equilibrium existence
5. Verify steady state is correctly computed

### Slow Convergence

**Problem**: SEP takes many iterations

**Solutions**:
1. The adaptive damping should help automatically
2. Start with fewer GH nodes (`sep_nnodes = 3`)
3. Use `sep_order = 1` (higher orders are much slower)
4. Reduce horizon (`sep_periods = 10-15`)

### Memory Issues

**Problem**: Out of memory errors

**Solution**: The SEP tree size is approximately:
```
nvars × T × (nshocks × nnodes)^L
```

Reduce by:
- Decreasing `sep_periods` (T)
- Using `sep_order = 1` (L)
- Reducing `sep_nnodes`

## References

- Fair, R. C., & Taylor, J. B. (1983). Solution and maximum likelihood estimation of dynamic nonlinear rational expectations models. *Econometrica*, 1169-1185.
- Adjemian, S., & Juillard, M. (2013). Stochastic extended path approach. In *Handbook of Computational Economics* (Vol. 3, pp. 35-71).
- Judd, K. L., Maliar, L., & Maliar, S. (2011). Numerically stable and accurate stochastic simulation approaches for solving dynamic economic models. *Quantitative Economics*, 2(2), 173-210.

## See Also

- [Occasionally Binding Constraints](obc.md) - Combine SEP with inequality constraints
- [RBC Tutorial](../tutorials/rbc.md) - Basic model solution
- [Calibration](../tutorials/calibration.md) - Parameter estimation with SEP
