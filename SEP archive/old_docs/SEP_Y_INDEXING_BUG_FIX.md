# SEP Y Vector Indexing Bug - FIXED

**Date**: December 27, 2024
**Status**: ✅ RESOLVED

## Summary

Fixed a critical bug in how test files extracted steady state values from the SEP solution Y vector. The bug was NOT in the SEP solver itself, but in the test file's misunderstanding of the Y vector structure.

## The Bug

### What Appeared to Be Wrong

When comparing MacroModelling.jl SEP results to Dynare, the test reported:
```
MacroModelling:  y=0.31672915, c=2.71856217, R=0.00000000
Dynare:          y=1.35587677, c=1.00699667, R=1.00725076
```

This looked like MacroModelling was:
1. Missing variable `y` entirely
2. Returning scrambled/shifted values
3. Reporting impossible values (R=0)

### Root Cause

The SEP Y vector is structured as a **branching tree**, not a simple flat vector.

**Structure**:
- Y has length 1639 (not 18!)
- Contains variables for all time periods and GH quadrature nodes
- Uses `layout.voff` (variable offsets) to navigate the tree
- For t=0 steady state: variables are at indices `voff[1] + (1:ny_)` = **2:19**, not 1:18

**What the test was doing (WRONG)**:
```julia
yss = sep_sol.Y[1:layout.ny_]  # Gets indices 1:18
```

This extracted:
- Y[1] = 0.0 (unused, voff starts at 1)
- Y[2] = P = 0.993...
- Y[3] = Pᴸ⁽¹⁾ = 0.0 (auxiliary)
- ...
- Y[18] = n = 0.3167... ← This got reported as "y"!

**What it should do (CORRECT)**:
```julia
yss = sep_sol.Y[layout.voff[1] .+ (1:layout.ny_)]  # Gets indices 2:19
```

This correctly extracts:
- Y[2] = P = 0.993...
- Y[3] = Pᴸ⁽¹⁾ = 0.0
- ...
- Y[19] = y = 1.3558... ← CORRECT!

## The Fix

### Files Modified

**test_fs2000_sep_comparison.jl** - Three locations:

1. **Line 54** - SEP order=1 extraction:
```julia
# WRONG:
yss_indices = 1:layout.ny_

# FIXED:
yss_indices = layout.voff[1] .+ (1:layout.ny_)
```

2. **Line 97** - SEP order=2 extraction:
```julia
# WRONG:
yss_indices_2 = 1:layout_2.ny_

# FIXED:
yss_indices_2 = layout_2.voff[1] .+ (1:layout_2.ny_)
```

3. **Line 165** - IRF helper function:
```julia
# WRONG:
yss = sep_solution.Y[1:layout.ny_]

# FIXED:
yss = sep_solution.Y[layout.voff[1] .+ (1:layout.ny_)]
```

Also updated comments from "SEP uses timings.var order" to "SEP Y vector follows model.var order" (both are equivalent, but var is clearer).

## Verification

### Before Fix
```
  y               0.31672915  ← WRONG (actually n!)
  c               2.71856217  ← WRONG (actually W!)
  R               0.00000000  ← WRONG (auxiliary variable!)
```

### After Fix
```
  y               1.35587677  ← ✓ Matches Dynare!
  c               1.00699667  ← ✓ Matches Dynare!
  R               1.00725076  ← ✓ Matches Dynare!
  n               0.31672915  ← ✓ Matches Dynare!
  k              18.98219172  ← ✓ Matches Dynare!
```

Perfect match with Dynare extended_path results!

## Key Learnings

1. **SEP Y vector is a tree structure**, not a flat vector
   - Length = `sum(layout.G .* layout.ny_)` where G is groups at each time
   - FS2000: 1639 total elements for T=10, Lbr=1, K=3, ny_=18

2. **Always use layout.voff for navigation**
   - `voff[t+1]` gives starting index for time t
   - For t=0: `voff[1] = 1`, so variables are at `1 + (1:18)` = **2:19**

3. **The SEP solver is correct**
   - Initialization builds `yss` vector correctly (verified via trace_yss_construction.jl)
   - All 18 variables including y are present
   - Steady state values match deterministic SS exactly

4. **Variable ordering is model.var**
   - Both model.var and timings.var have same order for FS2000
   - Y vector follows this order: [P, Pᴸ⁽¹⁾, R, W, c, cᴸ⁽¹⁾, d, dA, e, gp_obs, gy_obs, k, l, log_gp_obs, log_gy_obs, m, n, y]

## Diagnostic Files Created

These files helped identify and fix the bug:

1. **diagnose_Y_indexing.jl** - CRITICAL: Proved the fix
   - Compared `Y[1:18]` vs `Y[voff[1]+(1:18)]`
   - Showed `Y[19] = 1.3558...` matches deterministic SS for y

2. **trace_yss_construction.jl** - Proved initialization correct
   - Replicated SEP solver's yss building logic
   - Confirmed y is at position 18 in yss with correct value

3. **find_sep_variable_mapping.jl** - Reverse-engineered mapping
   - Matched Y values to SS variable names
   - Showed the off-by-one indexing issue

4. **diagnose_variable_ordering.jl** - Initial discovery
   - First spotted that values didn't match expected variables

## Impact on Other Code

### No Changes Needed in SEP Solver

The SEP solver (`src/sep_solver.jl`) is working correctly:
- Lines 169-179: yss initialization is correct
- Line 47: `index_y` function properly computes tree indices
- All internal navigation uses voff correctly

### Potential Issues in Other Files

Any code that extracts values from SEP solution.Y should use:
```julia
layout = solution.perturbation.stochastic_extended_path.layout
Y = solution.perturbation.stochastic_extended_path.Y

# For steady state (t=0):
yss = Y[layout.voff[1] .+ (1:layout.ny_)]

# For arbitrary time t and group g:
y_t_g = Y[layout.voff[t+1] + (g-1)*layout.ny_ .+ (1:layout.ny_)]
```

**Files to check**:
- Any test files using SEP
- Documentation examples
- Tutorial notebooks

## Next Steps

1. ✅ Fix applied to test_fs2000_sep_comparison.jl
2. ✅ Verified against Dynare results
3. ⏳ Check other test files for same issue
4. ⏳ Add documentation about Y vector structure
5. ⏳ Consider adding helper function `get_sep_steady_state(solution)` to avoid this error in future

## References

- SEP solver: `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/src/sep_solver.jl`
- Fixed test: `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/test_fs2000_sep_comparison.jl`
- Dynare comparison: `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/fs2000.mod`
- Diagnostic script: `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/diagnose_Y_indexing.jl`
