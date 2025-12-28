# SEP Deterministic Shock Sequence Implementation Plan

**Date**: December 27, 2024
**Goal**: Extend MacroModelling.jl SEP solver to support deterministic shock sequences for IRF validation against Dynare

## Current Status

### ✅ Completed

1. **Sparse tree implementation** - Adjemian-Juillard fishbone algorithm working
2. **Dynare benchmark data** - Saved to CSV with pdss() transformation:
   - `RBC_irf_pos3.csv` - Positive +3σ shock
   - `RBC_irf_neg3.csv` - Negative -3σ shock
3. **Validation framework** - `test_rbc_sparse_tree_irf_validation.jl` ready

### ❌ Blocker

Current SEP solver uses stochastic shocks via Gauss-Hermite quadrature. It computes **expected paths** under uncertainty, not paths with **specific shock sequences**.

**What we need**: Ability to solve for a path given deterministic shocks: `ε = [ε₁, ε₂, ..., εT]`

## Implementation Plan

### Phase 1: Add Deterministic Shock Parameter

**File**: `src/sep_solver.jl`

**Changes to `SEPSolverOptions`**:
```julia
struct SEPSolverOptions
    periods::Int
    order::Int
    nnodes::Int
    maxit::Int
    tol::Float64
    verbose::Bool
    shock_scale::Float64
    sparse_tree::Bool
    deterministic_shocks::Union{Matrix{Float64}, Nothing}  # NEW: T×dε matrix or nothing

    function SEPSolverOptions(;
        periods=20,
        order=1,
        nnodes=3,
        maxit=80,
        tol=1e-7,
        verbose=true,
        shock_scale=1.0,
        sparse_tree=false,
        deterministic_shocks=nothing  # NEW
    )
        # Validation
        if !isnothing(deterministic_shocks)
            @assert size(deterministic_shocks, 1) == periods "Shock sequence must have $periods rows"
        end
        new(periods, order, nnodes, maxit, tol, verbose, shock_scale, sparse_tree, deterministic_shocks)
    end
end
```

### Phase 2: Modify Main Solver Logic

**File**: `src/sep_solver.jl`, function `stochastic_extended_path()`

**Key modification**: When `deterministic_shocks` is provided, skip GH quadrature and use deterministic shocks directly.

```julia
function stochastic_extended_path(...)
    opts = SEPSolverOptions(...)

    if !isnothing(opts.deterministic_shocks)
        # DETERMINISTIC PATH MODE
        return solve_deterministic_path(model, opts, initial_state)
    else
        # STOCHASTIC MODE (current implementation)
        return solve_stochastic_path(model, opts, initial_state)
    end
end
```

### Phase 3: Implement Deterministic Path Solver

**New function** in `src/sep_solver.jl`:

```julia
function solve_deterministic_path(
    model::ℳ,
    opts::SEPSolverOptions,
    initial_state::Union{Vector{Float64}, Nothing}
)
    # Extract model components
    T = opts.periods
    dε = model.timings.nExo
    ny = length(model.var)

    # No branching in deterministic mode - single path
    # Layout: G = [1, 1, 1, ..., 1]  (one group at each time)
    layout = create_deterministic_layout(T, ny, dε)

    # Initialize solution vector
    Y = zeros(ny * (T + 1))

    # Set initial state
    if !isnothing(initial_state)
        Y[1:ny] = initial_state
    else
        Y[1:ny] = model.solution.non_stochastic_steady_state
    end

    # Newton solver for deterministic path
    # Stack all equilibrium conditions for t=0, 1, ..., T
    # Using the specific shock sequence from opts.deterministic_shocks

    for iter in 1:opts.maxit
        F = zeros(ny * T)  # Residuals
        J = spzeros(ny * T, ny * (T + 1))  # Jacobian

        # Assemble equations for each period
        for t in 1:T
            # Get shock at this period
            ε_t = opts.deterministic_shocks[t, :]

            # Get states: y_{t-1}, y_t, y_{t+1}
            y_tm1 = Y[(t-1)*ny .+ (1:ny)]
            y_t = Y[t*ny .+ (1:ny)]
            y_tp1 = (t < T) ? Y[(t+1)*ny .+ (1:ny)] : Y[T*ny .+ (1:ny)]  # Terminal = stay at final state

            # Evaluate model equations
            # F[t] = model_equations(y_tm1, y_t, y_tp1, ε_t) = 0
            # J[t, :] = jacobian w.r.t [y_{t-1}, y_t, y_{t+1}]

            eq_idx = (t-1)*ny .+ (1:ny)
            F[eq_idx] = evaluate_dynamic_equations(model, y_tm1, y_t, y_tp1, ε_t)
            J[eq_idx, (t-1)*ny .+ (1:ny)] = jacobian_wrt_ytm1(model, y_tm1, y_t, y_tp1, ε_t)
            J[eq_idx, t*ny .+ (1:ny)] = jacobian_wrt_yt(model, y_tm1, y_t, y_tp1, ε_t)
            if t < T
                J[eq_idx, (t+1)*ny .+ (1:ny)] = jacobian_wrt_ytp1(model, y_tm1, y_t, y_tp1, ε_t)
            end
        end

        # Check convergence
        res_norm = norm(F, Inf)
        if opts.verbose && (iter == 1 || iter % 10 == 0)
            @printf("  Iter %3d: ||F|| = %.3e\n", iter, res_norm)
        end

        if res_norm < opts.tol
            if opts.verbose
                println("  Converged in $iter iterations")
            end
            break
        end

        # Newton step
        ΔY = J \ (-F)
        Y[ny+1:end] += ΔY  # Update all periods (keep period 0 fixed)

        if iter == opts.maxit
            @warn "Deterministic path solver did not converge"
        end
    end

    # Package results
    return SEPSolution(layout, Y, Dict())
end
```

### Phase 4: Update Main API

**File**: `src/MacroModelling.jl`, function `solve!()`

Add parameter:
```julia
function solve!(...)
    # Existing parameters
    sep_periods::Int = 20,
    sep_order::Int = 1,
    sep_nnodes::Int = 3,
    sep_maxit::Int = 80,
    sep_tol::Float64 = 1e-7,
    sep_sparse_tree::Bool = false,
    sep_initial_guess::Union{Vector{Float64}, Nothing} = nothing,
    sep_deterministic_shocks::Union{Matrix{Float64}, Nothing} = nothing,  # NEW
    ...
)
```

Pass through to SEP solver:
```julia
if algorithm == :stochastic_extended_path
    opts = SEPSolverOptions(
        periods=sep_periods,
        order=sep_order,
        nnodes=sep_nnodes,
        maxit=sep_maxit,
        tol=sep_tol,
        sparse_tree=sep_sparse_tree,
        deterministic_shocks=sep_deterministic_shocks  # NEW
    )
    ...
end
```

### Phase 5: Update Validation Script

**File**: `test_rbc_sparse_tree_irf_validation.jl`

```julia
# Create shock sequence: +3σ at t=1, zero elsewhere
shock_sequence = zeros(total_periods, 1)  # 1 shock (epsilon)
shock_sequence[1, 1] = shock_magnitude_pos  # +3.0

# Solve with deterministic shocks
solve!(RBC_Dynare,
       algorithm=:stochastic_extended_path,
       sep_periods=total_periods,
       sep_order=maxorder,
       sep_nnodes=3,
       sep_sparse_tree=true,
       sep_deterministic_shocks=shock_sequence)  # NEW parameter
```

## Implementation Complexity

**Estimated effort**: 4-6 hours

**Key challenges**:
1. Extracting model equation evaluation from existing SEP code
2. Building Jacobian efficiently for stacked system
3. Testing with various shock sequences
4. Handling terminal conditions correctly

## Alternative: Quick Validation

If full implementation is too complex right now, we can:

1. **Validate stochastic steady state** instead of IRFs
2. **Compare sparse vs full tree** at same configuration
3. **Document** that IRF requires deterministic shock extension

This would still validate the sparse tree implementation, just not against Dynare IRFs.

## Recommendation

**Proceed with full implementation** if you need exact Dynare IRF comparison. The infrastructure is sound, just needs the deterministic shock mode added.

The key insight: **SEP with deterministic shocks = perfect foresight solver** for a specific shock sequence. This is exactly what Dynare's `extended_path` does when you give it an `innovations` sequence.

## Files to Modify

1. `src/sep_solver.jl` (main changes)
2. `src/MacroModelling.jl` (API parameter pass-through)
3. `test_rbc_sparse_tree_irf_validation.jl` (use new parameter)

## Expected Outcome

After implementation, you'll be able to:

```julia
# Replicate Dynare's extended_path exactly
shocks = [3.0; zeros(59)]  # +3σ at t=1
solve!(model, algorithm=:stochastic_extended_path,
       sep_deterministic_shocks=shocks,
       sep_sparse_tree=true)

# Extract path and compare with Dynare benchmark
```

This will enable direct IRF validation and demonstrate that MacroModelling.jl's sparse tree gives identical results to Dynare's fishbone algorithm.
