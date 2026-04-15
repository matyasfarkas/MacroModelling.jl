# Stochastic Extended Path (SEP) Method for OBC Models

## Overview

The Stochastic Extended Path (SEP) method is a powerful nonlinear solution technique for models with occasionally binding constraints (OBC). MacroModelling.jl implements a robust SEP solver with advanced features including subdifferential Newton for handling singular Jacobians at constraint kinks.

## When to Use SEP

Use the SEP method when:
- Your model has occasionally binding constraints (e.g., zero lower bound on interest rates)
- You need accurate nonlinear solutions beyond first-order perturbation
- You want to simulate stochastic paths with OBC enforcement
- You need impulse response functions with active constraints

## Basic Usage

### Simple Stochastic Simulation

```julia
using MacroModelling

# Load a model with OBC (e.g., Gali 2015 with ZLB)
include("models/Gali_2015_chapter_3_obc.jl")
m = Gali_2015_chapter_3_obc

# Run stochastic simulation with SEP
result = simulate_sep_extended_path(
    m;
    periods = 100,              # Number of periods to simulate
    shocks = :simulate,         # Random shocks from model distribution
    silent = false              # Show convergence info
)

# Access results
Y_sim = result.simulation      # Simulated paths (KeyedArray)
shocks_used = result.shocks    # Actual shocks applied
```

### Impulse Response Functions

```julia
# Single shock IRF
irf_result = simulate_sep_extended_path(
    m;
    periods = 40,
    shocks = zeros(length(m.exo), 40),  # Zero shocks (deterministic IRF from NSSS)
    shock_names = [:eps_z],             # Which shock to use
    shock_size = 0.01,                  # Shock magnitude (in std units)
    shock_period = 1                    # When to apply shock
)

# Custom shock path
custom_shocks = zeros(length(m.exo), 40)
custom_shocks[1, 5] = 0.02    # eps_z shock in period 5
custom_shocks[2, 15] = -0.01  # eps_a shock in period 15

irf_custom = simulate_sep_extended_path(
    m;
    periods = 40,
    shocks = custom_shocks
)
```

### Conditional Forecasts

```julia
# Forecast conditional on observed shocks
observed_shocks = randn(length(m.exo), 10)  # First 10 periods observed

forecast = simulate_sep_extended_path(
    m;
    periods = 10,
    shocks = observed_shocks,
    y0 = my_initial_state  # Optional: specify initial state
)
```

## Advanced Options

### SEP Solver Parameters

The SEP solver offers fine-grained control over the nonlinear solution process:

```julia
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate,

    # Convergence control
    sep_maxit = 500,           # Max iterations per period (default: 500)
    sep_tol = 1e-8,            # Residual tolerance (default: 1e-8)

    # Levenberg-Marquardt regularization
    lm_lambda = 1e-4,          # Initial LM damping (default: 1e-4)
    lm_lambda_max = 1e10,      # Max LM damping (default: 1e10)
    lm_lambda_scale = 2.0,     # LM scaling factor (default: 2.0)

    # Subdifferential Newton (for hard constraints)
    use_subdifferential = false,     # Enable subdifferential Newton
    subdiff_kink_tol = 1e-6,        # Kink detection tolerance
    subdiff_alpha_maxit = 20,       # α optimization iterations
    subdiff_alpha_tol = 1e-3,       # α convergence tolerance
    subdiff_verbose = false,        # Print subdiff diagnostics

    # Output control
    silent = false             # Show/hide convergence messages
)
```

### Subdifferential Newton for Hard Constraints

For models with hard (non-smooth) constraints like `max(0, R - R_bar)`, the standard Newton method can encounter singular Jacobians at constraint kinks. MacroModelling.jl includes a subdifferential Newton method to handle these cases:

```julia
# Example: Smets-Wouters with hard ZLB
include("models/Smets_Wouters_2007_HLT_obc.jl")
m_hard = Smets_Wouters_2007_HLT_obc

# Standard SEP (may fail at kinks)
result_standard = simulate_sep_extended_path(
    m_hard;
    periods = 50,
    shocks = :simulate,
    use_subdifferential = false  # Default
)

# With subdifferential Newton (robust at kinks)
result_robust = simulate_sep_extended_path(
    m_hard;
    periods = 50,
    shocks = :simulate,
    use_subdifferential = true,   # Enable subdifferential method
    subdiff_verbose = true        # See when it activates
)
```

The subdifferential Newton method:
- Automatically activates when singular Jacobians are detected
- Uses Clarke subdifferential to compute convex combinations of active/inactive Jacobians
- Finds optimal mixing parameter α ∈ [0,1] via golden section search
- Adds zero performance overhead when constraints are not binding
- Works seamlessly across linear and nonlinear models

## Solver Algorithm Details

The SEP solver implements a multi-stage fallback hierarchy:

1. **Standard Newton**: `Δy = -J \ R`
2. **Levenberg-Marquardt**: `Δy = -(J'J + λI) \ (J'R)` if Newton fails
3. **Subdifferential Newton** (if enabled): Convex combination of Jacobians at kinks
4. **Adaptive damping**: Increase λ if LM fails

This ensures robust convergence across a wide range of models and shock realizations.

## Examples

### Example 1: Zero Lower Bound Analysis

```julia
using MacroModelling
using StatsPlots  # For plotting

# Load model
include("models/Gali_2015_chapter_3_obc.jl")
m = Gali_2015_chapter_3_obc

# Large negative demand shock (drives to ZLB)
shocks = zeros(length(m.exo), 40)
shocks[findfirst(s -> s == :eps_z, m.exo), 1] = -3.0  # Large negative shock

# Simulate with ZLB
result_zlb = simulate_sep_extended_path(
    m;
    periods = 40,
    shocks = shocks
)

# Simulate without ZLB (for comparison)
result_no_zlb = simulate_sep_extended_path(
    m;
    periods = 40,
    shocks = shocks,
    ignore_obc = true  # Ignore occasionally binding constraint
)

# Plot comparison
plot_irf(m, shocks = shocks)  # With ZLB
plot_irf(m, shocks = shocks, ignore_obc = true)  # Without ZLB
```

### Example 2: Simulated Moments

```julia
# Long stochastic simulation
sim = simulate_sep_extended_path(
    m;
    periods = 1000,
    shocks = :simulate
)

# Compute moments
using Statistics
mean_output = mean(sim.simulation(:Y, :, :))
std_output = std(sim.simulation(:Y, :, :))
mean_rate = mean(sim.simulation(:R, :, :))

println("Mean output: $mean_output")
println("Std output: $std_output")
println("Mean interest rate: $mean_rate")

# Compare to linear model moments
println("\nLinear model moments:")
println("Mean output: $(get_mean(m)(:Y))")
println("Std output: $(get_std(m)(:Y))")
```

### Example 3: Forecast Error Variance Decomposition

```julia
# Multiple shock simulations
n_sims = 100
horizon = 40

output_paths = zeros(n_sims, horizon)

for i in 1:n_sims
    result = simulate_sep_extended_path(
        m;
        periods = horizon,
        shocks = :simulate
    )
    output_paths[i, :] = vec(result.simulation(:Y, :, 1))
end

# Variance over time
forecast_variance = [var(output_paths[:, t]) for t in 1:horizon]

# Plot uncertainty bands
using StatsPlots
plot(1:horizon, mean(output_paths, dims=1)[:],
     ribbon = 2 .* sqrt.(forecast_variance),
     label = "Mean ± 2σ",
     xlabel = "Periods",
     ylabel = "Output",
     title = "Forecast Uncertainty with ZLB")
```

## Performance Tips

### 1. Adjust Iteration Limits

For complex models or large shocks, increase iteration limits:

```julia
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate,
    sep_maxit = 1000  # Increase from default 500
)
```

### 2. Warm Starting

Use previous period's solution as initial guess (done automatically).

### 3. Parallel Simulation

For multiple independent simulations:

```julia
using Distributed
addprocs(4)  # Add 4 worker processes

@everywhere using MacroModelling
@everywhere include("models/Gali_2015_chapter_3_obc.jl")
@everywhere m = Gali_2015_chapter_3_obc

# Parallel simulations
results = pmap(1:100) do i
    simulate_sep_extended_path(m; periods = 50, shocks = :simulate, silent = true)
end
```

### 4. Reducing Tolerance for Speed

For quick exploratory analysis:

```julia
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate,
    sep_tol = 1e-6,  # Looser tolerance (faster)
    sep_maxit = 200   # Fewer iterations
)
```

## Troubleshooting

### Issue: "SEP did not converge"

**Symptom**: Warning message showing high residual after max iterations.

**Solutions**:
1. Increase `sep_maxit` (try 1000 or 2000)
2. Reduce shock magnitude
3. Enable subdifferential Newton: `use_subdifferential = true`
4. Adjust LM parameters: `lm_lambda = 1e-3`

### Issue: Singular Jacobian at Kink

**Symptom**: `SingularException` or very slow convergence near constraints.

**Solution**: Enable subdifferential Newton:
```julia
result = simulate_sep_extended_path(m;
    use_subdifferential = true,
    subdiff_verbose = true  # See diagnostics
)
```

### Issue: Slow Convergence

**Symptoms**: Many iterations per period.

**Solutions**:
1. Check model steady state is correct: `SS(m)`
2. Verify constraint is not binding in NSSS
3. Reduce `sep_tol` if high precision not needed
4. Consider smaller time steps (shorter forecast horizon per solve)

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `periods` | Int | (required) | Number of periods to simulate |
| `shocks` | Array/Symbol | `:simulate` | Shock matrix or `:simulate` for random |
| `y0` | Vector | NSSS | Initial state vector |
| `sep_maxit` | Int | 500 | Max iterations per period |
| `sep_tol` | Float64 | 1e-8 | Residual convergence tolerance |
| `lm_lambda` | Float64 | 1e-4 | Initial LM regularization |
| `lm_lambda_max` | Float64 | 1e10 | Maximum LM damping |
| `lm_lambda_scale` | Float64 | 2.0 | LM scaling factor |
| `use_subdifferential` | Bool | `false` | Enable subdifferential Newton |
| `subdiff_kink_tol` | Float64 | 1e-6 | Kink detection tolerance |
| `subdiff_alpha_maxit` | Int | 20 | α optimization max iterations |
| `subdiff_alpha_tol` | Float64 | 1e-3 | α convergence tolerance |
| `subdiff_verbose` | Bool | `false` | Print subdiff diagnostics |
| `silent` | Bool | `false` | Suppress convergence messages |
| `ignore_obc` | Bool | `false` | Ignore OBC (use linear solution) |

## See Also

- [Occasionally Binding Constraints](obc.md) - General OBC modeling guide
- [IRF Computation](../tutorials/irfs.md) - Impulse response functions
- [Stochastic Simulation](../tutorials/simulation.md) - General simulation guide
- [API Reference](../api.md) - Complete function documentation

## References

- Adjemian, S., & Juillard, M. (2013). "Stochastic Extended Path" Working Paper
- Guerrieri, L., & Iacoviello, M. (2015). "OccBin: A toolkit for solving dynamic models with occasionally binding constraints easily"
- Clarke, F. H. (1990). "Optimization and Nonsmooth Analysis"
