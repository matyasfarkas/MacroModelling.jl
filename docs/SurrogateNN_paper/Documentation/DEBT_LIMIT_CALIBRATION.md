# Debt Limit Calibration Report

## Executive Summary

This report documents the implementation and calibration of an occasionally binding debt limit constraint in the QMIPF model.

## Problem Statement

### Initial Issue
The model `/Volumes/MacMini/matyasfarkas/Documents/MacroModelling_local/models/QMIPF_final.jl` originally contained a debt limit constraint using the non-differentiable `max()` function:

```julia
THETA[0] = max(0.0, -penalty_kappa * BLIM[0] / SS_Y)
```

This caused the steady-state solver to fail with the error:
```
Could not find non-stochastic steady state
```

## Solution: Smooth Approximation

### Implementation
Replaced the `max()` function with a smooth, differentiable approximation:

```julia
# Smooth approximation: max(0, x) ≈ 0.5 * (x + sqrt(x^2 + ε^2))
THETA[0] = 0.5 * penalty_kappa * (-BLIM[0] / SS_Y + sqrt((BLIM[0] / SS_Y)^2 + smooth_param^2))
```

This formula:
- Is continuously differentiable everywhere
- Approximates `max(0, -penalty_kappa * BLIM[0] / SS_Y)` when `smooth_param` is small
- Allows the steady-state solver to converge successfully

### Parameters Added
```julia
penalty_kappa = 10000.0  # Penalty parameter for complementarity (very large)
smooth_param = 1e-8      # Smoothing parameter (extremely small for very sharp approximation)
SS_THETA | 0.0 >= 0.0   # No risk premium in steady state (with bounds)
```

## Steady State Results

### Steady State Successfully Found
- **SS_Y** = 3.715166 (quarterly output)
- **SS_NFA** = -0.377735 (net foreign assets, negative = debtor position)
- **SS_BLIM** = 3.337432 (distance from debt limit, positive = away from limit)
- **SS_THETA** ≈ 0.000000 (no risk premium in steady state, as expected)
- **SS_IB** = SS_I = 1.013730 (interest rates equal when no premium)

The steady state is consistent with economic intuition:
- Country is a net debtor (NFA < 0)
- But well away from debt limit (BLIM > 0)
- No risk premium applied (THETA ≈ 0)

## Challenge: Linearization vs. Nonlinearity

### Critical Finding
When using **first-order (linearized) approximation** for stochastic simulations:

```julia
sim = simulate(m, periods = 10000, algorithm = :first_order)
```

The constraint does **NOT bind properly**:
- Even when BLIM < 0 (violating the constraint), THETA remains ≈ 0
- This occurs because the linearization is taken around the steady state where THETA = 0
- The derivative of the smooth max function is zero at the steady state
- Therefore, in the linearized model, THETA stays near zero for all values of BLIM

### Evidence
Debug output shows:
```
Period  172: BLIM =  -3.373307, THETA =  -0.000000
Period  173: BLIM =  -1.055738, THETA =  -0.000000
Period  174: BLIM =  -4.405977, THETA =  -0.000000
...
```

Despite BLIM being negative (violating the constraint), THETA does not increase to enforce it.

## Implication for Calibration

### Why Standard Calibration Won't Work
The target "debt limit binds in 3% of periods" **cannot be meaningfully calibrated** using first-order approximation methods because:

1. **The constraint doesn't actually bind** in the linearized model
2. THETA remains effectively zero regardless of BLIM
3. The nonlinearity is essential to the mechanism but is lost in linearization

### What We Observed
Testing different values of `m_by`:

| m_by Value | SS_BLIM | BLIM < 0 Frequency | THETA > 0 Frequency |
|------------|---------|-------------------|---------------------|
| 0.1185     | 1.376   | 48.53%            | 56.59%             |
| 0.15       | 1.851   | 44.70%            | 19.76%             |
| 0.25       | 3.337   | 50.94%            | 0.00%              |
| 0.3849     | 5.342   | 51.87%            | 0.00%              |
| 0.5147     | 7.271   | 47.73%            | 0.00%              |

**Observation**: With `m_by ≥ 0.25`, THETA never becomes positive, even though BLIM is negative nearly half the time. This confirms that the linearized model fails to enforce the constraint.

## Recommended Next Steps

### Option 1: Use Nonlinear Solution Methods
For OBCs (Occasionally Binding Constraints), use methods that preserve nonlinearities:

1. **Stochastic Extended Path (SEP)**: Already implemented in MacroModelling.jl
   - See: `/Volumes/MacMini/matyasfarkas/Documents/MacroModelling_local/src/sep_solver.jl`
   - Use: `solve_stochastic_extended_path(m, ...)`

2. **Piecewise Linear Methods**: Switch between regimes
   - When BLIM > 0: use linear model without constraint
   - When BLIM ≤ 0: use linear model with constraint binding

3. **Higher-Order Perturbation**: Use second or third-order approximation
   - Captures nonlinearities in policy functions
   - May still struggle with sharp kinks from constraints

### Option 2: Reformulate the Constraint
Alternative approaches that may work better with linearization:

1. **Smooth Penalty Always Active**: Use `penalty_kappa * max(0, -BLIM)^2` instead of step function
2. **State-Dependent Risk Premium**: Make risk premium depend smoothly on NFA position
3. **Expectational Effects**: Include expected future violations in current premium

### Option 3: Accept Linearization Limitations
If linearized analysis is preferred:

1. Interpret binding frequency as "periods when BLIM would violate without the constraint"
2. Calibrate `m_by` based on desired BLIM < 0 frequency (currently ~50% with m_by = 0.25)
3. Accept that THETA doesn't actually enforce the constraint in simulations
4. Use the model for comparative statics rather than quantitative moments

## Current Model Status

### File Locations
- **Model**: `/Volumes/MacMini/matyasfarkas/Documents/MacroModelling_local/models/QMIPF_final.jl`
- **Test Script**: `/Volumes/MacMini/matyasfarkas/Documents/MacroModelling_local/scripts/test_debt_limit_calibration.jl`
- **This Report**: `/Volumes/MacMini/matyasfarkas/Documents/MacroModelling_local/Documentation/DEBT_LIMIT_CALIBRATION.md`

### Current Calibration
```julia
m_by = 0.25              # Distance from debt limit (% of quarterly GDP)
penalty_kappa = 10000.0  # Penalty parameter
smooth_param = 1e-8      # Smoothing parameter
```

### Model State
- ✅ Steady state: **SOLVED** successfully
- ✅ Smooth approximation: **IMPLEMENTED**
- ❌ Constraint enforcement in simulations: **NOT WORKING** (requires nonlinear methods)
- ❌ Calibration to 3% binding frequency: **NOT ACHIEVED** (requires nonlinear methods)

## Technical Details

### The Debt Limit Constraint Equations
```julia
# Distance from debt limit (in units of output)
BLIM[0] = NFA[0] + m * Y[0]

# where m = -SS_NFA/SS_Y + m_by*4
# m_by is the key calibration parameter

# Debt limit risk premium (smooth penalty function)
THETA[0] = 0.5 * penalty_kappa * (-BLIM[0] / SS_Y + sqrt((BLIM[0] / SS_Y)^2 + smooth_param^2))

# Retail interest rate includes debt limit premium
IB[0] = I[0] + THETA[0]
```

### Economic Interpretation
- **BLIM > 0**: Country is below its debt limit → no premium (THETA ≈ 0)
- **BLIM ≤ 0**: Country hits/exceeds debt limit → premium kicks in (THETA > 0)
- **THETA**: Acts as a "sudden stop" premium that makes borrowing expensive when limit is approached
- **m_by**: Controls how far the debt limit is from the steady state (higher = more room to borrow)

### Calibration Formula
The debt limit is set at:
```
Debt Limit = -m * Y = -(m_by * 4) * Y (quarterly)
           = -m_by * 4Y (annualized)
```

Example with `m_by = 0.25` and `SS_Y = 3.715`:
- Quarterly: -0.25 * 4 * 3.715 = -3.715 (100% of quarterly GDP)
- Annual: -0.25 * 4 * 14.86 = -14.86 (100% of annual GDP)

## Conclusion

The steady-state problem has been **successfully resolved** using a smooth approximation. However, **proper calibration of the binding frequency** requires switching from first-order linearized simulations to nonlinear solution methods such as Stochastic Extended Path (SEP).

The current model is ready for nonlinear analysis but will not produce meaningful binding statistics under linearized simulation.

---

*Report generated: 2026-01-15*
*Model: QMIPF_step9e_Real_UIP*
*Analysis method: First-order perturbation + smooth penalty approximation*
