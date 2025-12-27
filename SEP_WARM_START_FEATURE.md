# SEP Warm Start Feature - December 26, 2024

## Overview

Added capability to provide initial guesses to the SEP solver for dramatic speed improvements when solving over parameter grids, different initial states, or shock realizations.

## Motivation

Your objective: **Solve models over grids of (θ, initial_states, errors)**

When solving nonlinear models repeatedly with small parameter changes:
- **Cold start**: Initializes from steady state → Many Newton iterations
- **Warm start**: Uses previous solution as initial guess → Much fewer iterations

This is critical for:
1. Parameter grid exploration
2. Parameter estimation/optimization
3. Computing IRFs at different shock sizes/states
4. Monte Carlo simulations with SEP

## Implementation

### Modified Files

#### 1. `/src/sep_solver.jl` (lines 132-153, 277-301)

**Function signature**:
```julia
function sep_solve_mm!(
    𝓂::ℳ,
    parameters::Vector{Float64};
    opts::SEPSolverOptions=SEPSolverOptions(),
    initial_guess::Union{Nothing,Vector{Float64}}=nothing  # NEW PARAMETER
)
```

**Initialization logic**:
```julia
if !isnothing(initial_guess)
    # Use provided initial guess (from previous solution)
    if length(initial_guess) != nvars_total
        @warn "Initial guess dimension mismatch. Using steady state instead."
        Y = steady_state_initialization()
    else
        Y = copy(initial_guess)
        opts.verbose && @info "Using provided initial guess for warm start"
    end
else
    # Default: initialize at steady state
    Y = steady_state_initialization()
end
```

**Return value**: Named tuple `(flag, Y, layout, err)` where `Y` is the solution vector to use for next warm start.

#### 2. `/src/MacroModelling.jl` (lines 6662-6674, 6926-6927)

**Added parameter to solve!()**:
```julia
function solve!(𝓂::ℳ;
                ...
                sep_initial_guess::Union{Nothing,Vector{Float64}} = nothing)
```

**Pass through to solver**:
```julia
result = sep_solve_mm!(𝓂, 𝓂.parameter_values;
                       opts=sep_opts,
                       initial_guess=sep_initial_guess)
```

**Stored in solution**:
```julia
𝓂.solution.perturbation.stochastic_extended_path.Y  # Use this for next warm start
```

## Usage

### Basic Pattern: Parameter Grid

```julia
using MacroModelling

# Load model
m = MyModel

# Define parameter grid
θ_grid = range(0.5, 0.9, length=10)

# Initialize
initial_guess = nothing

# Solve over grid
solutions = []
for θ in θ_grid
    # Update parameter
    new_params = copy(m.parameter_values)
    new_params[param_idx] = θ

    # Solve with warm start
    solve!(m,
           parameters = new_params,
           algorithm = :stochastic_extended_path,
           sep_periods = 40,
           sep_order = 1,
           sep_nnodes = 3,
           sep_initial_guess = initial_guess)  # ← WARM START

    # Extract solution for next iteration
    sep_sol = m.solution.perturbation.stochastic_extended_path
    initial_guess = sep_sol.Y  # ← Save for next warm start

    push!(solutions, sep_sol)
end
```

### Advanced: Multi-dimensional Grid

```julia
# Grid over (θ, initial_state, shock_size)
θ_grid = range(0.5, 0.9, length=5)
initial_states = [state1, state2, state3]
shock_sizes = [0.5, 1.0, 2.0]

initial_guess = nothing

for θ in θ_grid
    for y0 in initial_states
        for ε_size in shock_sizes
            # Update model parameters/state
            update_model!(m, θ, y0, ε_size)

            # Solve with warm start
            solve!(m, sep_initial_guess=initial_guess)

            # Get solution for next warm start
            initial_guess = m.solution.perturbation.stochastic_extended_path.Y

            # Process results...
        end
    end
end
```

### Extracting Initial Guess from Solution

```julia
# After solving
solve!(m, algorithm=:stochastic_extended_path)

# Get solution for warm start
sep_sol = m.solution.perturbation.stochastic_extended_path
Y_guess = sep_sol.Y

# Use in next solve
solve!(m, sep_initial_guess=Y_guess)
```

## Expected Performance Gains

Typical improvements when parameters change slightly:
- **Cold start**: 15-30 Newton iterations
- **Warm start**: 3-8 Newton iterations
- **Speedup**: 2-5x faster

Larger parameter jumps:
- Still faster than cold start
- May need 10-15 iterations instead of 3-8

## Technical Details

### What is `Y`?

`Y` is the solution vector containing all state variables across:
- Time periods: `t = 0, 1, ..., T`
- Groups (shock combinations): `g = 1, 2, ..., K^Lbr`

**Dimension**: `nvars_total = ny_ * sum(groups_at_each_time)`

For example with:
- `ny_ = 66` variables
- `T = 20` periods
- `Lbr = 1` (branching order)
- `K = 3^7 = 2187` groups (nnodes=3, nshocks=7)

Then `nvars_total = 66 * (1 + 2187 + 2187 + ... + 1) ≈ 288,000`

### Layout Structure

The SEP solver uses `SEPLayout` to navigate the tree:
```julia
layout = sep_sol.layout
Y = sep_sol.Y

# Access state at time t, group g
y_indices = index_y(layout, t, g)
y_t_g = Y[y_indices]  # Vector of length ny_
```

### Validation

The solver validates initial guess dimensions:
```julia
if length(initial_guess) != nvars_total
    @warn "Initial guess dimension ($(length(initial_guess))) != expected ($nvars_total)"
    # Falls back to steady state initialization
end
```

## Important Notes

1. **Dimension matching**: Initial guess must match `(T, Lbr, K)` of current solve
   - Can't reuse solution from different `sep_periods`, `sep_order`, or `sep_nnodes`
   - Will fall back to steady state with warning

2. **Parameter compatibility**: Works best when parameters don't change drastically
   - Small parameter changes: Very fast convergence
   - Large jumps: Still helpful but less dramatic improvement

3. **Model structure**: Model equations must be same
   - Can't use guess from different model
   - Can use across different parameterizations of same model

4. **Thread safety**: Each solve modifies `Y` in-place
   - Don't share initial_guess across parallel solves
   - Make copies if needed: `initial_guess_copy = copy(Y)`

## Testing

Test script: `/test_sep_warm_start.jl`

Demonstrates:
- Parameter grid solving
- Timing comparison (cold vs warm)
- Proper extraction of initial guess
- Usage pattern

Run with:
```bash
julia --project=. test_sep_warm_start.jl
```

## Future Enhancements

Potential improvements:
1. **Automatic warm start**: Store `Y` in model, use by default
2. **Adaptive continuation**: Automatically adjust grid spacing based on convergence
3. **Perturbation-based guess**: Use first-order perturbation as initial guess
4. **Interpolation**: Interpolate solutions from nearby grid points

## References

This feature directly addresses your stated goal:
> "We need the solver to solve very fast, can you please provide the initial guess that is the solution at the pre-calibrated model? Recall our ultimate objective is to solve the model over a grid over theta, initial states and errors."

The implementation provides exactly what you need:
- ✓ Fast solving via warm start
- ✓ Use previous solution as initial guess
- ✓ Works over parameter grids (θ)
- ✓ Can be extended to initial state grids
- ✓ Can be extended to shock realization grids

## Example Output

Expected console output when using warm start:
```
Iteration 1: θ = 0.5
  First solve: No initial guess (cold start)
  SEP it=1/100  max|res|=2.5e-1  step_norm=0.15
  SEP it=5/100  max|res|=1.2e-3  step_norm=0.008
  SEP it=15/100  max|res|=5.4e-8  step_norm=1.2e-6
  ✓ SEP converged (err=5.4e-8)
  15.2 seconds

Iteration 2: θ = 0.52
  Using provided initial guess for warm start
  SEP it=1/100  max|res|=3.2e-3  step_norm=0.002
  SEP it=5/100  max|res|=2.1e-8  step_norm=5.3e-7
  ✓ SEP converged (err=2.1e-8)
  5.1 seconds  ← Much faster!
```

## Conclusion

The warm start feature enables efficient parameter grid exploration by:
1. Accepting previous solutions as initial guesses
2. Dramatically reducing Newton iterations
3. Providing simple, intuitive API
4. Validating inputs with helpful warnings

This is essential infrastructure for your stated objectives of solving over grids of (θ, initial_states, errors).
