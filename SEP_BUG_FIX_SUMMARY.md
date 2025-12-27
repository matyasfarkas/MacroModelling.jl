# SEP IRF Bug Fix - December 25, 2024

## Problem
SEP IRFs returned machine-epsilon values (~1e-11) instead of meaningful responses.
All shock realizations (2187 different groups) converged to identical steady-state values.

## Root Cause: Incorrect Shock Indexing

In `src/sep_solver.jl` lines 337-361, the expectation loop used:

```julia
for (kidx, cg) in enumerate(cgs)
    ε_curr = view(X, :, kidx)  # ❌ WRONG!
    wk = W[kidx]              # ❌ WRONG!
```

**Problem**: `kidx` is the ENUMERATION index (1, 2, 3, ...), not the actual group index!

- For t=1 with parent at t=0: `cgs = 1:2187`, so `kidx` goes 1,2,...,2187 (accidentally correct)
- For t≥1 with parent at t≥1: `cgs = g:g` (single element), so `kidx` = 1 always!
  **This meant ALL groups used X[:,1] (the first shock combination) instead of their specific shock!**

## The Fix

Changed to use the CHILD GROUP index `cg` instead of enumeration index `kidx`:

```julia
for (kidx, cg) in enumerate(cgs)
    ε_curr = view(X, :, cg)  # ✅ CORRECT!
    wk = W[cg]              # ✅ CORRECT!
```

**Modified lines**:
- Line 350: `ε_curr = view(X, :, cg)` (was `kidx`)
- Line 361: `wk = W[cg]` (was `kidx`)

## Results

### Before Fix:
```
Group 1 (no shock): deviation = 1.73e-11
Group 1175 (epinf shock): deviation = 1.84e-11
Difference: 1.08e-12 (numerical noise)
```

### After Fix:
```
Group 1 (no shock): deviation = -1.20e-5
Group 1175 (epinf shock): deviation = -9.72e-5
Difference: 8.53e-5 (10,000x larger!)
```

## Additional Fixes Applied

1. **Shock covariance extraction** (`sep_solver.jl` lines 191-224)
   - Was: `Σ = 0.01×I` (hardcoded tiny values)
   - Now: Extracts from model parameters `z_shock` (e.g., σ_epinf = 0.1455)

2. **Group indexing for IRF** (`sep_irf.jl` lines 47-83)
   - Fixed tensor product calculation: `idx = Σ (node-1) * nnodes^(d-1)`
   - Uses base-nnodes arithmetic to find correct group for specific shock

3. **Field access** (`sep_irf.jl` line 68)
   - Get nshocks from `layout.dε` instead of non-existent `sep_sol.Σ_ε`

4. **IRF shock size scaling** (`sep_irf.jl` lines 85-100, 122, 182, 188)
   - Problem: GH nodes at ±√3σ for nnodes=3, but users request 1.0σ shocks
   - Fix: Calculate `scale_factor = requested_size / actual_node_std`
   - Apply scaling when converting to deviations: `irf[i, :] .*= scale_factor`
   - Ensures IRF magnitude matches requested shock size, not GH node location

## Files Modified

- `src/sep_solver.jl`: Main bug fix + shock covariance extraction + diagnostics
- `src/sep_irf.jl`: IRF extraction with correct group indexing

## Testing

Created diagnostic tests:
- `test_sep_group_indexing.jl`: Verifies group indexing calculation
- `test_gh_nodes.jl`: Verifies GH node transformation
- `test_sep_convergence.jl`: Tests with increased iterations
- `test_shock_cov.jl`: Verifies shock covariance extraction

## Impact

This fix enables SEP to properly compute IRFs for DSGE models in MacroModelling.jl.
The SEP algorithm now correctly differentiates between different shock realizations
and produces meaningful impulse responses that can be compared with perturbation methods.
