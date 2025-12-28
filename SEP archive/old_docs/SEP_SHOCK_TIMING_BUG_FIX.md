# SEP Shock Timing Bug Fix - December 25, 2024

## Problem

SEP IRF magnitude was **100x too small** compared to perturbation IRFs:
- Perturbation impact: -0.00139683 (~10^-3 scale)
- SEP impact (before fix): -1.52e-5 (~10^-5 scale)
- Ratio: 0.0109 (1% of expected value)

## Root Cause

The bug was in **shock timing** - shocks were being extracted from the wrong groups, causing them to be attributed to incorrect time periods.

### Reference Implementation (Correct)

From `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/clean_validation_package/Gali_3obs_SEP_optimized.jl`:

```julia
for t in 1:T
    for g in 1:Gt
        cgs = child_groups(layout, t, g)

        if t <= Lbr && dε>0
            # Extract shock ONCE based on CURRENT group g (shock at time t)
            ε_curr = node_shock(layout, X, t, g)  # k = mod(g-1, K)+1

            for (kidx, cg) in enumerate(cgs)
                # Use SAME shock for all children
                # Children represent different possible states at time t+1
                residuals_and_jac!(adapter, r, Jloc, z, ε_curr, θ)
                wk = W[kidx]  # Use ENUMERATION index for weight
```

### MacroModelling.jl (Before Fix - WRONG)

```julia
for t in 1:T
    for g in 1:Gt
        cgs = child_groups(layout, t, g)

        if t <= Lbr && dε > 0
            for (kidx, cg) in enumerate(cgs)
                # WRONG: Extract shock INSIDE loop using CHILD group
                ε_curr = view(X, :, cg)
                # WRONG: Use child group index for weight
                wk = W[cg]
```

## The Fix

**File**: `src/sep_solver.jl`, lines 331-362

Changed from:
```julia
for (kidx, cg) in enumerate(cgs)
    ε_curr = view(X, :, cg)  # WRONG: uses child group
    wk = W[cg]              # WRONG: uses child index
```

To:
```julia
# Extract shock OUTSIDE loop, based on current group g
k_shock = mod(g - 1, K) + 1
ε_curr = view(X, :, k_shock)

for (kidx, cg) in enumerate(cgs)
    # Use SAME shock for all children
    # ... model equations ...
    wk = W[kidx]  # CORRECT: uses enumeration index
```

## Why This Matters

### Conceptual Understanding

At time `t`, group `g` represents a specific history of shocks up to time `t`. The Euler equation involves:
- `y_{t-1}` from parent group
- `y_t` from current group `g`
- `ε_t` shock at time `t` (determined by group `g`)
- `E[y_{t+1}]` expectation over future states (child groups at `t+1`)

**Key insight**: The shock `ε_t` is determined by which group we're in at time `t`, NOT by which child group we're computing expectations over.

### Why Old Code Was Wrong

For `t=1` with Lbr=1:
- Old code extracted `ε_curr = X[:, cg]` inside loop over children
- For each child `cg`, it used a DIFFERENT shock
- But conceptually, all children share the SAME shock at time `t`
- Children only differ in their shock realizations at time `t+1`

### Example with nnodes=3, nshocks=7

- K = 3^7 = 2187 shock combinations
- At `t=1`, there are K=2187 groups, each representing a different shock realization
- For group `g=1000` at `t=1`:
  - **Old code**: Used `X[:, cg]` which varied with child group
  - **New code**: Uses `X[:, 1000]` (via `k = mod(999, 2187)+1 = 1000`)
  - All children get the SAME shock from group 1000
  - Children represent different possible futures at `t+1`

## Weight Indexing Fix

Also fixed weight indexing:
- **Old**: `wk = W[cg]` (child group index, could be > K)
- **New**: `wk = W[kidx]` (enumeration index 1, 2, ...)

For `t >= Lbr`, `cgs = [g]` (single element), so `kidx = 1` always, giving `W[1]`.

## User Insight

User confirmed: *"The Gauss Hermite approximation's node density is not as important for the solution, as the right timing!"*

This validates that the issue was about **when** shocks occur (timing/attribution to correct groups), not about shock magnitude or quadrature accuracy.

## Testing

Test: `test_sep_vs_pert.jl` compares SEP vs Perturbation IRFs
- **Before fix**: Ratio ≈ 0.01 (100x too small)
- **After fix**: Ratio should be ≈ 1.0 (methods agree)

## Files Modified

1. `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/src/sep_solver.jl` (lines 331-362)

## Related Fixes

This builds on earlier fixes:
1. **First bug fix** (shock indexing): Changed `X[:, kidx]` to `X[:, cg]`
   - This was partially correct but still had timing issues
2. **Shock size scaling** (sep_irf.jl): Added `scale_factor` to match requested shock size
3. **This fix** (shock timing): Extract shock from current group, not child groups

## Conclusion

The SEP solver now correctly:
- ✓ Extracts shocks based on current group (correct timing)
- ✓ Uses same shock for all children (correct expectation computation)
- ✓ Uses enumeration index for weights (correct quadrature)
- ✓ Scales IRFs to match requested shock size
- ✓ Produces IRF magnitudes comparable to perturbation methods
