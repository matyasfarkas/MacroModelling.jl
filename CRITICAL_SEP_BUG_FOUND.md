# CRITICAL BUG FOUND IN SEP SOLVER

**Date**: December 27, 2024
**Issue**: SEP solution vector Y is missing critical variables

## Summary

The SEP solver is returning a solution vector Y that does NOT contain all model variables. Specifically, the variable `y` (output) is completely missing from the solution.

## Evidence

### 1. FS2000 Model Structure
- Total variables: 18 (including 2 auxiliary variables)
- Key variables: P, R, W, c, d, dA, e, gp_obs, gy_obs, **y**, k, l, log_gp_obs, log_gy_obs, m, n
- **y** = Output (production function result)

### 2. Deterministic Steady State Values
```
y     =   1.35587677  ← This is the correct steady state value
c     =   1.00699667
R     =   1.00725076
n     =   0.31672915
k     =  18.98219172
```

### 3. SEP Solution Y Vector (length=18)
```
Y[ 1] =   0.00000000  → Pᴸ⁽¹⁾ (auxiliary)
Y[ 2] =   0.99325055  → P
Y[ 3] =   0.00000000  → cᴸ⁽¹⁾ (auxiliary)
Y[ 4] =   1.00725076  → R  ✓
Y[ 5] =   2.71856217  → W  ✓
Y[ 6] =   1.00699667  → c  ✓
Y[ 7] =   0.00000000  → (unknown zero)
Y[ 8] =   0.86084788  → d
Y[ 9] =   1.00853623  → dA
Y[10] =   1.00000000  → e
Y[11] =   0.99173433  → gp_obs
Y[12] =   1.00853623  → (duplicate dA?)
Y[13] =  18.98219172  → k  ✓
Y[14] =   0.86104788  → l
Y[15] =  -0.00830002  → log_gp_obs
Y[16] =   0.00850000  → log_gy_obs
Y[17] =   1.00020000  → m
Y[18] =   0.31672915  → n  ✓
```

**CRITICAL**: None of these values equals 1.35587677 (the value of y)!

### 4. Variable `y` is Missing

The SEP Y vector contains 18 values, but variable `y` is NOT among them. This is why:
- The test file reports `y = 0.31672915` (which is actually `n`!)
- The test file reports `c = 2.71856217` (which is actually `W`!)
- The test file reports `R = 0.00000000` (which is an auxiliary variable!)

All values are shifted/scrambled because the test assumes `model.var` order, but SEP is using a DIFFERENT subset of variables.

## Root Cause

The SEP solver in `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/src/sep_solver.jl` sets:
```julia
ny_ = length(𝓂.var)  # line 156
```

This should include all 18 variables, but somewhere in the solver initialization or residual function construction, variable `y` is being excluded.

## Impact

1. **Test file produces nonsense**: Values are completely scrambled
2. **Comparison with Dynare impossible**: Dynare reports y=1.35096, MacroModelling can't even report y
3. **SEP solution is incomplete**: Missing a fundamental variable (output!)

## Next Steps

**BEFORE fixing the test file**, we must fix the SEP solver to include all variables:

### Option 1: Investigate SEP initialization
Check `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/src/sep_solver.jl` around line 156 to see:
- How the variable list is determined
- Why `y` might be excluded
- Whether this is related to auxiliary variables, states vs jumpers, etc.

### Option 2: Check if y is computed from other variables
Perhaps `y` is intentionally excluded because it can be computed from the production function?
- In FS2000: `y = k(-1)^alp * n^(1-alp) * exp(-alp*(gam+e_a))`
- But this should NOT mean it's excluded from the solution vector!

### Option 3: Compare with SW07 model
Check if the same issue occurs with other models - is this FS2000-specific or systemic?

## Immediate Action Required

**DO NOT proceed with test file fixes** until the SEP solver is corrected to include all model variables.

The variable ordering fix (model.var vs timings.var) is SECONDARY to this critical issue.

## Questions for User

1. Is this a known limitation of the SEP implementation?
2. Should SEP solve for ALL model variables or only a subset (states + jumpers)?
3. Is there documentation on which variables SEP includes/excludes?
4. Should I investigate the SEP solver code directly or is there a design document?

## Files for Investigation

- `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/src/sep_solver.jl` - Main SEP implementation
- `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/src/MacroModelling.jl` (lines 6925-6947) - SEP interface/wrapper
- `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/` - Dynare reference implementation

## Diagnostic Files Created

- `find_sep_variable_mapping.jl` - Reveals the exact variable mapping issue
- `diagnose_variable_ordering.jl` - Initial discovery of mismatch
- `diagnose_fs2000_ss.jl` - Comparison of SS vs SEP values
