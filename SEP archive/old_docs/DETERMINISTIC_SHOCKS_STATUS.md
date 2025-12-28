# Deterministic Shock Sequence Implementation - Status

**Date**: December 27, 2024
**Goal**: Enable MacroModelling.jl SEP solver to accept deterministic shock sequences for IRF validation against Dynare

## ✅ Completed (Phases 1-2)

### 1. Modified `SEPSolverOptions` struct (`src/sep_solver.jl:9-38`)
```julia
struct SEPSolverOptions
    # ... existing fields ...
    deterministic_shocks::Union{Matrix{Float64}, Nothing}  # NEW: T×dε matrix

    function SEPSolverOptions(;
        # ... existing parameters ...
        deterministic_shocks=nothing  # NEW
    )
        # Validation
        if !isnothing(deterministic_shocks)
            @assert size(deterministic_shocks, 1) == periods
            @assert size(deterministic_shocks, 2) >= 1
        end
        new(..., deterministic_shocks)
    end
end
```

### 2. Updated `solve!()` API (`src/MacroModelling.jl:6662-6927`)
```julia
function solve!(𝓂::ℳ;
    # ... existing parameters ...
    sep_deterministic_shocks::Union{Nothing,Matrix{Float64}} = nothing)  # NEW

    # ... in SEP block ...
    sep_opts = SEPSolverOptions(
        # ... existing ...
        deterministic_shocks = sep_deterministic_shocks  # NEW
    )
```

### 3. Updated validation script (`test_rbc_sparse_tree_irf_validation.jl:82-94`)
```julia
shock_sequence_pos = zeros(total_periods, 1)  # T×1 matrix
shock_sequence_pos[1, 1] = shock_magnitude_pos * sigma_epsilon  # +3σ

solve!(RBC_Dynare,
       algorithm=:stochastic_extended_path,
       sep_periods=total_periods,
       sep_order=maxorder,
       sep_nnodes=3,
       sep_sparse_tree=true,
       sep_deterministic_shocks=shock_sequence_pos)  # NEW
```

**Compilation Test**: ✅ MacroModelling.jl compiles successfully

## ✅ Additional Completed Work

### Phase 3-4: Deterministic Path Solver (COMPLETED)

**File**: `src/sep_solver.jl:330-513`

Implemented `solve_deterministic_path` function with:
- Perfect foresight solver for deterministic shock sequences
- Newton method with sparse Jacobian (tridiagonal block structure)
- First-order approximation using existing Jacobian blocks [∇₊, ∇₀, ∇₋, ∇ₑ]
- Adaptive damping for robustness
- Deterministic layout (no branching tree)

**Branching logic added**: `src/sep_solver.jl:398-402`
```julia
if !isnothing(opts.deterministic_shocks)
    opts.verbose && @info "Deterministic mode detected - using perfect foresight solver"
    return solve_deterministic_path(𝓂, parameters, opts, initial_guess, yss, SS_and_pars)
end
```

## 🔧 Remaining Issues

### Issue 1: Newton Solver Divergence

**Status**: Solver runs but diverges (residual exploding: ~10¹²)

**Root cause**: Jacobian indexing issue
- Current code includes y₀ (initial condition) in column indices
- But y₀ is fixed, so Jacobian should only act on [y₁, ..., yT]
- Jacobian dimensions: (ny_×T) × (ny_×(T+1)) - WRONG! Should be (ny_×T) × (ny_×T)

**Fix needed**:
1. Remove y₀ from solution vector for Newton step
2. OR: Adjust Jacobian column indices to exclude y₀
3. Need to map equation indices correctly

### Issue 2: Fixed Variable Error

**Status**: FIXED ✓
- Moved `R` declaration outside loop (line 412)
- Added `err` initialization (line 413)

## 🔧 Remaining Work

### Phase 3: Modify `sep_solve_mm!` to branch on deterministic mode

**File**: `src/sep_solver.jl:347`

Add branching logic at the beginning of `sep_solve_mm!`:

```julia
function sep_solve_mm!(𝓂::ℳ, parameters::Vector{Float64};
                        opts::SEPSolverOptions=SEPSolverOptions(),
                        initial_guess::Union{Nothing,Vector{Float64}}=nothing)

    # NEW: Detect deterministic mode
    if !isnothing(opts.deterministic_shocks)
        # DETERMINISTIC PATH MODE
        return solve_deterministic_path(𝓂, parameters, opts, initial_guess)
    else
        # STOCHASTIC MODE (existing implementation continues below)
        # ... existing code ...
    end
end
```

### Phase 4: Implement `solve_deterministic_path` function

**New function** in `src/sep_solver.jl` (add before `sep_solve_mm!`):

Key requirements:
- No branching tree - single deterministic path
- Solve stacked nonlinear system for all periods: F(Y) = 0
- Y = [y₀, y₁, ..., yT] where each yₜ has length ny_
- Shock sequence from `opts.deterministic_shocks`
- Use Newton solver with sparse Jacobian
- Terminal condition: yT = yss (return to steady state)

Pseudo-code structure:
```julia
function solve_deterministic_path(
    𝓂::ℳ,
    parameters::Vector{Float64},
    opts::SEPSolverOptions,
    initial_guess::Union{Nothing,Vector{Float64}}
)
    # 1. Extract dimensions
    ny_ = length(𝓂.var)
    dε = length(𝓂.exo)
    T = opts.periods

    # 2. Get steady state and Jacobian
    yss = ...  # steady state vector
    ∇₁ = ...   # Jacobian from model

    # 3. Initialize solution vector Y
    Y = zeros(ny_ * (T+1))
    Y[1:ny_] = yss  # Initial condition

    # 4. Newton iterations
    for iter in 1:opts.maxit
        # Build stacked residual F and Jacobian J
        F = zeros(ny_ * T)
        J = spzeros(ny_ * T, ny_ * (T+1))

        for t = 1:T
            # Get shock at period t
            ε_t = opts.deterministic_shocks[t, :]

            # Get states
            y_tm1 = Y[(t-1)*ny_ .+ (1:ny_)]
            y_t = Y[t*ny_ .+ (1:ny_)]
            y_tp1 = (t < T) ? Y[(t+1)*ny_ .+ (1:ny_)] : yss  # Terminal = SS

            # Evaluate equilibrium conditions
            eq_idx = (t-1)*ny_ .+ (1:ny_)
            F[eq_idx] = evaluate_model_equations(y_tm1, y_t, y_tp1, ε_t)

            # Fill Jacobian blocks
            J[eq_idx, (t-1)*ny_ .+ (1:ny_)] = ∂F/∂y_{t-1}
            J[eq_idx, t*ny_ .+ (1:ny_)] = ∂F/∂y_t
            if t < T
                J[eq_idx, (t+1)*ny_ .+ (1:ny_)] = ∂F/∂y_{t+1}
            end
        end

        # Check convergence
        if norm(F, Inf) < opts.tol
            break
        end

        # Newton step
        ΔY = J \ (-F)
        Y[ny_+1:end] += ΔY  # Update all periods except initial condition
    end

    # 5. Package results
    layout = create_deterministic_layout(T, ny_, dε)
    return (Y=Y, layout=layout, flag=0, err=...)
end
```

### Phase 5: Helper function for equation evaluation

Need to extract/use existing model evaluation from MacroModelling.jl:
- How to evaluate `f(y_{t-1}, y_t, y_{t+1}, ε_t) = 0`
- How to compute Jacobian blocks

**Note**: This is complex because it requires interfacing with MacroModelling's internal equation representation.

## Implementation Strategy

Given the complexity, I recommend:

1. **Start with a simplified test**: Create a minimal deterministic path solver that uses the existing first-order solution matrices
2. **Test compilation and integration** before full Newton implementation
3. **Leverage existing code**: Look for how `𝓂.SS_solve_func` or similar functions evaluate equilibrium conditions
4. **Reference**: The existing SEP solver in `src/sep_solver.jl` lines 400+ shows how to evaluate model equations

## Next Action

Continue reading `sep_solve_mm!` to understand how it evaluates model equations, then adapt that logic for the deterministic case.

**Key file locations**:
- Main solver: `src/sep_solver.jl:347+`
- Equation evaluation: Need to find in `sep_solver.jl` (around lines 500-600)
- Model structure: `𝓂` object has methods for evaluation
