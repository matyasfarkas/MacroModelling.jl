# MacroModelling.jl Quick Reference

## Basic Model Operations

### Load and Solve a Model

```julia
using MacroModelling

# Load model
include("models/Smets_Wouters_2003.jl")
m = Smets_Wouters_2003

# Get steady state
SS(m)

# Get solution (first-order perturbation)
get_solution(m)
```

### Impulse Response Functions

```julia
# Default IRF (all shocks, standard deviations from steady state)
plot_irf(m)

# Specific shock
plot_irf(m, shocks = :eps_z)

# Custom shock size
plot_irf(m, shocks = :eps_z, shock_size = 0.02)

# Negative shock
plot_irf(m, shocks = :eps_z, negative_shock = true)

# Multiple periods
plot_irf(m, periods = 60)
```

### Stochastic Simulation

```julia
# Standard simulation
plot_simulations(m)

# Custom periods
plot_simulations(m, periods = 200)

# Get data (not plots)
sim = get_simulations(m, periods = 100)
```

### Model Moments

```julia
# Theoretical moments (linear models)
get_mean(m)              # Mean
get_std(m)               # Standard deviation
get_variance(m)          # Variance
get_covariance(m)        # Covariance
get_correlation(m)       # Correlation
get_autocorrelation(m)   # Autocorrelation

# Specific variable
get_mean(m)(:Y)
get_std(m)(:Y, :C)
```

## Occasionally Binding Constraints (OBC)

### Define OBC Model

```julia
@model MyModel begin
    # Regular equations...
    Y[0] = C[0] + I[0]

    # ZLB constraint using max()
    R[0] = max(R̄, β^(-1) * Pi[0]^φπ * Y[0]^φy)
end

@parameters MyModel begin
    R̄ = 1.0  # Zero lower bound
    β = 0.99
    φπ = 1.5
    φy = 0.125

    # Ensure NSSS has R > R̄ for perturbation solution
    R > 1.00001
end
```

### Simulate OBC Models

```julia
# With OBC enforcement
plot_simulations(m)
plot_irf(m, shocks = :eps_z)

# Without OBC (linear solution)
plot_simulations(m, ignore_obc = true)
plot_irf(m, shocks = :eps_z, ignore_obc = true)
```

## Stochastic Extended Path (SEP) - Nonlinear OBC Solution

### Basic SEP Simulation

```julia
# Stochastic simulation with OBC
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate  # Random shocks from model distribution
)

# Access results
Y_sim = result.simulation(:Y, :, :)  # Output time series
shocks_used = result.shocks          # Actual shocks applied
```

### SEP Impulse Response Functions

```julia
# IRF from steady state
irf = simulate_sep_extended_path(
    m;
    periods = 40,
    shocks = zeros(length(m.exo), 40),  # Start from zero
    shock_names = [:eps_z],              # Shock to apply
    shock_size = 0.01,                   # Magnitude (std units)
    shock_period = 1                     # When to shock
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

### SEP with Subdifferential Newton (Hard Constraints)

For models with hard constraints like `max(0, R - R_bar)`:

```julia
# Standard SEP (may fail at kinks)
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate,
    use_subdifferential = false  # Default
)

# Robust SEP (handles singular Jacobians at kinks)
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate,
    use_subdifferential = true,   # Enable subdifferential Newton
    subdiff_verbose = true        # Show when it activates
)
```

### Common SEP Parameters

```julia
result = simulate_sep_extended_path(
    m;
    periods = 100,                # Number of periods
    shocks = :simulate,           # :simulate or matrix
    y0 = my_initial_state,       # Initial state (optional)

    # Convergence
    sep_maxit = 500,             # Max iterations per period
    sep_tol = 1e-8,              # Residual tolerance

    # Levenberg-Marquardt
    lm_lambda = 1e-4,            # LM regularization
    lm_lambda_max = 1e10,        # Max LM damping
    lm_lambda_scale = 2.0,       # LM scaling

    # Subdifferential Newton (for hard constraints)
    use_subdifferential = false,  # Enable/disable
    subdiff_kink_tol = 1e-6,     # Kink detection
    subdiff_alpha_maxit = 20,    # α optimization iterations
    subdiff_alpha_tol = 1e-3,    # α tolerance
    subdiff_verbose = false,     # Diagnostics

    # Output
    silent = false               # Show convergence info
)
```

## Estimation (Basic)

### Kalman Filter

```julia
using DataFrames, CSV

# Load data
data = CSV.read("mydata.csv", DataFrame)

# Run Kalman filter
filter_result = get_kalman_filter_estimate(
    m,
    data,
    data_variables = [:Y, :C, :I],  # Observables
    parameters = [:α, :β],           # Parameters to estimate
    parameter_bounds = Dict(
        :α => (0.0, 1.0),
        :β => (0.0, 1.0)
    )
)
```

## Parameter Manipulation

### Change Parameters

```julia
# Single parameter
plot_irf(m, parameters = :β => 0.95)

# Multiple parameters
plot_irf(m, parameters = Dict(:β => 0.95, :α => 0.30))

# In simulations
plot_simulations(m, parameters = :β => 0.95)
```

## Useful Functions

### Model Information

```julia
# Variable names
m.var

# Parameter names
m.parameters

# Parameter values
m.parameter_values

# Exogenous shocks
m.exo

# Number of variables
length(m.var)
```

### Extract Specific Results

```julia
# From IRF
irf = get_irf(m, shocks = :eps_z)
output_irf = irf(:Y, :, :)  # Y response to eps_z

# From simulation
sim = get_simulations(m, periods = 100)
output_sim = sim(:Y, :, :)  # Y time series
```

## Troubleshooting

### No Solution Found

```julia
# Check steady state
SS(m)

# Check if NSSS satisfies constraint
SS(m)(:R)  # Should be > R̄ for ZLB model

# Add constraint in @parameters
R > 1.00001  # Forces R above ZLB in NSSS
```

### SEP Convergence Issues

```julia
# Increase iterations
simulate_sep_extended_path(m, sep_maxit = 1000)

# Reduce tolerance
simulate_sep_extended_path(m, sep_tol = 1e-6)

# Enable subdifferential Newton
simulate_sep_extended_path(m, use_subdifferential = true)

# Adjust LM regularization
simulate_sep_extended_path(m, lm_lambda = 1e-3)
```

### Singular Jacobian at Kink

```julia
# Use subdifferential Newton
result = simulate_sep_extended_path(
    m;
    use_subdifferential = true,
    subdiff_verbose = true  # See diagnostics
)
```

## Example Workflows

### Complete OBC Analysis

```julia
using MacroModelling, StatsPlots

# 1. Load model
include("models/Gali_2015_chapter_3_obc.jl")
m = Gali_2015_chapter_3_obc

# 2. Check steady state
println("NSSS: ", SS(m)(:R))

# 3. IRF with ZLB
plot_irf(m, shocks = :eps_z, negative_shock = true)

# 4. Compare with/without ZLB
plot_irf(m, shocks = :eps_z, negative_shock = true)  # With ZLB
plot_irf(m, shocks = :eps_z, negative_shock = true, ignore_obc = true)  # Without

# 5. Stochastic simulation
sim = simulate_sep_extended_path(m, periods = 200, shocks = :simulate)

# 6. Compute moments
using Statistics
println("Mean output: ", mean(sim.simulation(:Y, :, :)))
println("Std output: ", std(sim.simulation(:Y, :, :)))
```

### Robust SEP with Hard Constraints

```julia
using MacroModelling

# 1. Load hard OBC model
include("models/Smets_Wouters_2007_HLT_obc.jl")
m = Smets_Wouters_2007_HLT_obc

# 2. Run robust SEP
result = simulate_sep_extended_path(
    m;
    periods = 100,
    shocks = :simulate,
    use_subdifferential = true,   # Robust to kinks
    subdiff_verbose = true,       # Show diagnostics
    sep_maxit = 500               # Adequate iterations
)

# 3. Check success
if !result.errorflag
    println("✓ Simulation successful")
    println("Mean R: ", mean(result.simulation(:R, :, :)))
else
    println("⚠ Simulation hit iteration limit")
end
```

## Key References

- `SS(model)` - Get steady state
- `get_solution(model)` - Get first-order solution
- `plot_irf(model, ...)` - Plot IRFs
- `plot_simulations(model, ...)` - Plot stochastic simulations
- `get_irf(model, ...)` - Get IRF data
- `get_simulations(model, ...)` - Get simulation data
- `simulate_sep_extended_path(model, ...)` - Nonlinear OBC solution
- `get_kalman_filter_estimate(model, data, ...)` - Estimate parameters

## Additional Documentation

- Full SEP guide: `docs/src/how-to/stochastic_extended_path.md`
- OBC guide: `docs/src/how-to/obc.md`
- API reference: `docs/src/api.md`
- Tutorials: `docs/src/tutorials/`
