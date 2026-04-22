# Proper OBC Implementation for MacroModelling.jl

**Date**: January 19, 2026
**Issue**: Current penalty method doesn't enforce OBC in SEP
**Solution**: Use MacroModelling.jl's native OBC system or alternative formulations

---

## Current Problem

### What We Have (Penalty Method)

```julia
# Line 119: Distance from debt limit
BLIM[0] = NFA[0] + m * Y[0]

# Line 126: Risk premium (penalty method)
THETA[0] = max(0.0, -penalty_kappa * BLIM[0])

# Line 129: Retail rate
IB[0] = I[0] + THETA[0]
```

### Why It Fails

1. **First-order**: max() gets linearized → THETA ≈ 0 always
2. **SEP**: max() not enforced as complementarity → THETA oscillates at numerical precision (~1e-6)
3. **Calibration finds**: 59% "binding" but it's just numerical noise, not true enforcement

### Dynare's Approach (MCP)

```matlab
IB = I + THETA;                                   // retail rate
[name='Debt limit constraint',mcp = 'BLIM > 0']  // complementarity condition
THETA = 0;                                        // dual variable equation
BLIM = B + m*Y(+1);                              // constraint equation
```

**Complementarity conditions:**
- THETA ≥ 0 (dual variable/multiplier)
- BLIM ≥ 0 (constraint)
- THETA · BLIM = 0 (at least one must be zero)

When BLIM > 0: THETA = 0 (constraint slack)
When BLIM = 0: THETA can be > 0 (constraint binds)

---

## Solution Options for MacroModelling.jl

### Option 1: Use Native OBC System (RECOMMENDED)

**MacroModelling.jl philosophy**: Write the binding constraint directly, not the multiplier.

**Instead of defining THETA, constrain IB:**

```julia
# Remove the THETA equation entirely
# Define IB with a lower bound constraint using max()

# Distance from debt limit (when this hits 0, constraint binds)
BLIM[0] = NFA[0] + m * Y[0]

# Retail rate can't fall below policy rate by more than the debt limit penalty
# When BLIM < 0, IB must rise above I to enforce the limit
IB[0] = max(I[0], I[0] - penalty_kappa * BLIM[0])

# THETA is derived, not primary
THETA[0] = IB[0] - I[0]
```

**How it works:**
- When BLIM > 0: max picks I[0], so IB = I, THETA = 0
- When BLIM < 0: max picks I[0] - penalty_kappa*BLIM > I[0], so IB > I, THETA > 0
- MacroModelling's OBC parser will add anticipated shock sequences to enforce this

**Key parameter:** `max_obc_horizon` in @model macro (default: 40)
- Increase if constraint enforcement fails
- This determines how many periods ahead the model anticipates the constraint

**Usage:**
```julia
@model QMIPF_step9e_Real_UIP max_obc_horizon = 80 begin
    # ... equations ...
    IB[0] = max(I[0], I[0] - penalty_kappa * BLIM[0])
    # ...
end
```

### Option 2: Slack Variable Formulation

**Explicitly introduce slack variable:**

```julia
# Slack variable (measures constraint violation)
SLACK[0] = BLIM[0]

# Risk premium is positive when slack is negative
THETA[0] = max(0.0, -penalty_kappa * SLACK[0])

# Or use complementarity-style formulation
# THETA[0] * SLACK[0] = 0 (approximately)
THETA[0] = penalty_kappa * max(0.0, -SLACK[0])

# Retail rate
IB[0] = I[0] + THETA[0]

# Distance from limit
BLIM[0] = NFA[0] + m * Y[0]
```

This makes the complementarity structure more explicit.

### Option 3: Smooth Complementarity Function

**Fischer-Burmeister function** (smooth approximation of complementarity):

```julia
# Instead of: THETA = max(0, -κ*BLIM)
# Use Fischer-Burmeister: sqrt(a² + b²) - a - b ≈ 0 when a,b satisfy complementarity

# Helper variables
A[0] = penalty_kappa * BLIM[0]
B[0] = THETA[0]

# Fischer-Burmeister condition (smooth, differentiable)
0 = sqrt(A[0]^2 + B[0]^2) - A[0] - B[0]

# This enforces:
# - THETA ≥ 0
# - BLIM ≥ 0
# - THETA * BLIM ≈ 0

# Retail rate
IB[0] = I[0] + THETA[0]

# Distance from limit
BLIM[0] = NFA[0] + m * Y[0]
```

**Advantage**: Smooth and differentiable, better for nonlinear solvers.

**Disadvantage**: Requires solving an implicit equation.

### Option 4: Exponential Penalty (Sharper Signal)

**Make penalty much steeper:**

```julia
# Exponential penalty grows rapidly as BLIM becomes negative
THETA[0] = penalty_kappa * max(0.0, exp(-alpha * BLIM[0] / SS_BLIM) - 1.0)

# When BLIM >> 0: exp(negative) ≈ 0, so THETA ≈ 0
# When BLIM < 0: exp(positive) >> 1, so THETA spikes

# Parameters
penalty_kappa = 0.1  # Overall scale
alpha = 10.0         # Steepness

# Retail rate
IB[0] = I[0] + THETA[0]

# Distance from limit
BLIM[0] = NFA[0] + m * Y[0]
```

**Advantage**: Provides clearer signal, less numerical ambiguity.

### Option 5: Use Variable Bounds (If Supported)

**Some nonlinear solvers support box constraints:**

```julia
# In @parameters block
THETA >= 0.0    # Already have this
BLIM >= 0.0     # Add this

# Then use penalty method
THETA[0] = -penalty_kappa * BLIM[0]

# Solver enforces THETA ≥ 0 and BLIM ≥ 0 during solve
# This creates implicit complementarity
```

**Check**: Does SEP solver respect variable bounds during solve?

---

## Recommended Approach

### Step 1: Try Native OBC System (Option 1)

**Modify QMIPF_final.jl:**

```julia
@model QMIPF_step9e_Real_UIP max_obc_horizon = 100 begin
    # ... other equations ...

    # ========================================================================
    # Debt limit constraint (OBC) - Native MacroModelling.jl approach
    # ========================================================================

    # Distance from debt limit
    BLIM[0] = NFA[0] + m * Y[0]

    # Retail interest rate: rises above policy rate when approaching debt limit
    # max() triggers MacroModelling's OBC system (anticipated shocks)
    IB[0] = max(I[0], I[0] - penalty_kappa * BLIM[0])

    # Risk premium (derived)
    THETA[0] = IB[0] - I[0]

    # ... other equations ...
end
```

**Test with:**
```julia
# Pre-solve
solve!(m, algorithm=:stochastic_extended_path,
       sep_order=0, ignore_obc=false)  # Don't ignore OBC!

# Get IRF
irf = get_sep_irf(m, :EPS_Y_ST, -5.0;
                  sep_order=0, ignore_obc=false)
```

**Expected behavior:**
- MacroModelling adds many auxiliary variables for anticipated OBC shocks
- These shocks help the model anticipate and enforce the constraint
- Should see more variables in model (maybe 150-200 instead of 120)

### Step 2: If That Doesn't Work, Try Exponential Penalty (Option 4)

More robust numerical signal:

```julia
# Exponential penalty
THETA[0] = penalty_kappa * max(0.0, exp(-10.0 * BLIM[0]) - 1.0)
IB[0] = I[0] + THETA[0]
BLIM[0] = NFA[0] + m * Y[0]
```

### Step 3: Contact Package Maintainers

If neither works:
- Open issue on MacroModelling.jl GitHub
- Ask about proper OBC implementation for SEP
- Show that max() in direct equation definitions isn't working
- Ask if MCP-style complementarity is supported

---

## Testing Plan

### Test 1: Native OBC with First-Order

```julia
using MacroModelling
include("models/QMIPF_final_obc_native.jl")
m = QMIPF_step9e_Real_UIP

# Test with first-order
irf = get_irf(m; shocks=:EPS_Y_ST, shock_size=-10,
              algorithm=:first_order, ignore_obc=false)

# Check if THETA responds
theta_idx = findfirst(==(Symbol("THETA")), m.var)
theta_vals = irf[theta_idx, :, 1]

println("THETA range: ", extrema(theta_vals))
println("Max THETA: ", maximum(theta_vals))
```

**Expected with native OBC:** THETA should spike to meaningful values (> 0.001)

### Test 2: Native OBC with SEP

```julia
# Pre-solve
solve!(m, algorithm=:stochastic_extended_path,
       sep_order=0, ignore_obc=false)

# Large shock
irf = get_sep_irf(m, :EPS_Y_ST, -5.0;
                  sep_order=0, ignore_obc=false)

# Check enforcement
theta_vals = irf[theta_idx, 2:end]
blim_vals = irf[blim_idx, 2:end]

println("BLIM < 0 periods: ", sum(blim_vals .< 0))
println("THETA > 0 periods: ", sum(theta_vals .> 1e-4))  # Use threshold
```

**Expected:** When BLIM < 0, THETA should be > 0 by a meaningful amount.

---

## Key Differences: Penalty vs Native OBC

| Aspect | Penalty Method (Current) | Native OBC (Option 1) |
|--------|-------------------------|---------------------|
| **Implementation** | `THETA = max(0, -κ*BLIM)` | `IB = max(I, I - κ*BLIM)` |
| **Approach** | Define multiplier directly | Constrain endogenous variable |
| **First-order** | Linearizes max() away | Uses anticipated shocks |
| **SEP** | max() not enforced | Should use anticipated shocks |
| **Variables added** | None | Many (OBC shock sequences) |
| **Horizon parameter** | N/A | max_obc_horizon needed |
| **Enforcement** | ❌ Fails | ✅ Should work |

---

## Why This Matters

The penalty method approximates MCP but doesn't have the right mathematical properties for nonlinear solvers. The native OBC system is designed specifically for this:

1. **Anticipation**: Model looks ahead and adjusts behavior before constraint binds
2. **Smooth enforcement**: Uses sequence of anticipated shocks rather than discrete max()
3. **Solver compatibility**: Designed to work with MacroModelling's solution methods

---

## Implementation Steps

1. ✅ Understand the issue (penalty method doesn't enforce in SEP)
2. ⏭️ Implement native OBC approach (Option 1)
3. ⏭️ Test with first-order and SEP
4. ⏭️ If successful, run calibration to target 3% binding
5. ⏭️ If unsuccessful, try exponential penalty (Option 4)
6. ⏭️ Contact maintainers if needed

---

**Bottom line**: We've been fighting the framework instead of using it. MacroModelling.jl has a built-in OBC system - we just need to use it correctly by constraining IB directly rather than defining THETA as max().
