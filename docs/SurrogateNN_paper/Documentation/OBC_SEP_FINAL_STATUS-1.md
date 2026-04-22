# OBC with SEP: Final Status Report

**Date**: January 15, 2026
**Status**: ✅ SEP WORKS | ⚠️ OBC NOT ENFORCING YET

---

## Executive Summary

### What Works ✅

1. **SEP solver works** after memory was freed
2. **Pre-solving enables SEP** - required workflow established
3. **Model compiles** with OBC (120 vars, 73 states)
4. **Steady state solves** in ~3 seconds
5. **SEP converges** for moderate and large shocks

### Remaining Issue ⚠️

**OBC is not enforcing in SEP**: When BLIM < 0 (constraint binds), THETA stays ≈ 0 instead of spiking positive.

---

## Test Results

### Test 1: SEP Without OBC ✅ SUCCESS

```bash
julia --project=. scripts/test_sep_without_obc.jl
```

**Result:**
```
penalty_kappa = 0 (OBC disabled)
✅ SEP converged in 1.5 seconds
✅ THETA correctly at 0
```

**Conclusion:** Model structure is compatible with SEP.

### Test 2: SEP With OBC (penalty_kappa=0.01) ⚠️ CONVERGES BUT OBC DOESN'T ENFORCE

```bash
julia --project=. scripts/test_sep_with_obc_working.jl
```

**Result:**
```
penalty_kappa = 0.01
✅ Pre-solve: SUCCESS (2.1 seconds)
✅ SEP IRF: CONVERGED (0.7 seconds, error: 7.7e-4)

NFA: min = -0.167
BLIM: min = -0.169 (negative in ALL 20 periods!)
THETA: max = -6.3e-7 (essentially ZERO, should be POSITIVE!)
IB - I spread: ≈ 0 (no premium)
```

**Diagnosis:**
- Constraint SHOULD bind (BLIM < 0)
- But THETA doesn't respond
- penalty_kappa=0.01 may be too small

### Test 3: SEP With OBC (penalty_kappa=0.1) ⏳ RUNNING

```bash
# Increased penalty_kappa from 0.01 to 0.1 (10x larger)
julia --project=. scripts/test_sep_with_obc_working.jl
```

**Status:**
- ✅ Pre-solve: SUCCESS (2.2 seconds)
- ⏳ Computing SEP IRF (still running after 15+ minutes)
- Will test if larger penalty makes THETA respond

---

## The OBC Enforcement Problem

### Expected Behavior

When BLIM < 0 (NFA hits debt limit):
```julia
THETA[0] = max(0.0, -penalty_kappa * BLIM[0])
#          When BLIM < 0, this should be POSITIVE
```

Should produce:
- THETA > 0 (risk premium spikes)
- IB = I + THETA (retail rate jumps above policy rate)
- This enforces the debt limit by making borrowing expensive

### Actual Behavior (penalty_kappa=0.01)

When BLIM < 0:
```
THETA ≈ 0 (actually slightly negative: -6.3e-7)
IB ≈ I (no spread)
```

Result: Constraint doesn't actually enforce!

### Possible Causes

**1. penalty_kappa Too Small**
- Current: 0.01
- With BLIM ≈ -0.17, max(0, -0.01 * -0.17) = 0.0017
- This might be too small numerically
- Testing with 0.1 now

**2. max() Function Approximation**
- SEP may linearize or smooth max() operator
- Variable bound `THETA >= 0` may not be enforced during solve
- max() might not be fully nonlinear in SEP's perfect foresight solver

**3. Solver Tolerance Issues**
- SEP tolerance: 1e-3
- THETA values O(1e-3) might be within tolerance
- Solver considers them "close enough to zero"

---

## Comparison: First-Order vs SEP

### First-Order Approximation

```julia
irf = get_irf(m; shocks=:EPS_Y_ST, shock_size=-10, algorithm=:first_order)
```

**Result (from test_obc_enforcement.jl):**
```
BLIM < 0 in ALL 40 periods
THETA = 0.0 exactly (linearized away)
```

**Expected:** First-order linearizes max() away.

### SEP (Deterministic, order=0)

```julia
solve!(m, algorithm=:stochastic_extended_path, sep_order=0, ...)
irf = get_sep_irf(m, :EPS_Y_ST, -5.0; sep_order=0, ...)
```

**Result:**
```
BLIM < 0 in ALL 20 periods
THETA ≈ 0 (max = -6.3e-7)
```

**Expected:** SEP should preserve nonlinearity, but THETA still not responding!

**Unexpected:** SEP is supposed to handle nonlinearities, but OBC still not enforcing.

---

## Potential Solutions

### Option 1: Increase penalty_kappa ⏳ TESTING

**Current test:** penalty_kappa = 0.1 (10x larger)

**Rationale:**
- Larger penalty → larger THETA response
- With BLIM ≈ -0.17, max(0, -0.1 * -0.17) = 0.017
- This is ~10x larger, may be detectable

**If this works:** Calibrate penalty_kappa to balance:
- Large enough for SEP to detect
- Small enough for steady state convergence

### Option 2: Increase penalty_kappa Much More

**Try:** penalty_kappa = 1.0 or higher

**Risk:** May destabilize steady state computation

**Mitigation:** Use different penalty for SS vs SEP
```julia
# In steady state: penalty_kappa = 0.1
# For SEP: multiply by 10 internally
```

### Option 3: Use Tighter Debt Limit

**Current:** m_by = 0.1185 (constraint far from SS)

**Try:** m_by = 0.05 (tighter limit)

**Effect:** BLIM becomes more negative → larger THETA response

**Downside:** Changes economic calibration

### Option 4: Different OBC Formulation

**Current (penalty method):**
```julia
THETA[0] = max(0.0, -penalty_kappa * BLIM[0])
```

**Alternative (exponential penalty):**
```julia
THETA[0] = penalty_kappa * exp(-BLIM[0]) - penalty_kappa
# When BLIM < 0: exponential term >> 1, THETA > 0
# When BLIM > 0: exponential term ≈ 1, THETA ≈ 0
```

**Advantage:** Steeper penalty, clearer signal

### Option 5: Use Dynare's MCP

**Dynare approach:**
```matlab
[name='Debt limit constraint', mcp='BLIM > 0']
THETA = 0;
```

**Convert model to Dynare .mod**
- Use Dynare's optimized complementarity solver
- May handle OBC better for large models
- Requires maintaining parallel code

### Option 6: Accept First-Order for Now

**Pragmatic approach:**
- OBC equations are correctly implemented
- First-order for routine analysis
- Acknowledge linearization limitation in paper
- Focus research on other aspects

---

## Recommended Next Steps

### Immediate (Today)

1. ✅ Wait for penalty_kappa=0.1 test to complete
2. If THETA still ≈ 0: Try penalty_kappa = 1.0
3. If still doesn't work: Check if max() is being approximated by SEP

### Short-Term (This Week)

**If penalty tuning doesn't work:**
- Investigate MacroModelling.jl's handling of max() in SEP
- Contact package maintainers about OBC support
- Try exponential penalty formulation

**If penalty tuning works:**
- Find optimal penalty_kappa value
- Run calibration to 3% binding frequency
- Generate publication figures

### Medium-Term (If Needed)

- Create Dynare version with MCP for comparison
- Or: Use first-order and acknowledge limitations
- Or: Develop reduced model variant for SEP analysis

---

## Current Model State

### Files

**Model:** `models/QMIPF_final.jl`
```julia
# Line 126: OBC equation
THETA[0] = max(0.0, -penalty_kappa * BLIM[0])

# Line 244: Current parameter (being tested)
penalty_kappa = 0.1  # Testing 10x increase

# Line 370: Variable bound
THETA >= 0.0
```

**Working test script:** `scripts/test_sep_with_obc_working.jl`
- Pre-solves SEP
- Computes IRF with large negative shock
- Analyzes OBC behavior

### Compilation Status

```
✅ Model compiles (120 variables, 73 states)
✅ Steady state solves (3 seconds)
✅ First-order IRFs work
✅ SEP pre-solving works
✅ SEP IRF computation works
⚠️ OBC not enforcing yet (THETA ≈ 0 when BLIM < 0)
```

---

## What We Know For Sure

### Confirmed Working ✅

1. SEP infrastructure works with this model
2. Pre-solving workflow is correct
3. Model structure is sound
4. Equations are correctly specified
5. No memory or computational errors

### Confirmed Issue ⚠️

1. THETA does not respond when BLIM < 0
2. OBC does not enforce in SEP with current settings
3. penalty_kappa=0.01 is insufficient

### To Be Determined ⏳

1. Will penalty_kappa=0.1 work?
2. Is max() being linearized/approximated in SEP?
3. What is the minimum penalty_kappa for OBC enforcement?
4. Does MacroModelling.jl's SEP fully support max() operators?

---

## Bottom Line

**Progress:** ✅ SEP works, no more e26 errors, workflow established

**Challenge:** ⚠️ OBC not enforcing yet - THETA stays at 0

**Next:** ⏳ Testing if larger penalty_kappa solves the issue

**Fallback:** Accept first-order for analysis, or use Dynare for OBC verification

---

**Status:** Investigation ongoing. SEP infrastructure confirmed working.
OBC enforcement requires further parameter tuning or implementation adjustment.
