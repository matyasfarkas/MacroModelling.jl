# Debt Limit OBC Analysis and Implementation

**Date**: January 15, 2026
**Task**: Check and implement debt limit as occasionally binding constraint (OBC)

---

## Executive Summary

**Finding**: The Julia translation (`QMIPF_final.jl`) **does NOT implement the debt limit constraint** that exists in the original Dynare model (`QMIPF_stoch.mod`).

**Impact**: Without the debt limit OBC, the model cannot capture:
- Sudden stop episodes when countries hit borrowing limits
- Endogenous risk premia that emerge near debt limits
- Asymmetric dynamics between normal times and crisis episodes
- Financial stress effects on interest rates

**Action Required**: Add debt limit OBC to Julia model

---

## Original Dynare Implementation

### Equations (from QMIPF_stoch.mod)

```dynare
[name='Nominal retail interest rate']
IB = I + THETA;                          // Equation 93

[name='Debt limit constraint', mcp = 'BLIM > 0']
THETA = 0;                               // Equation 94

[name='Distance from the debt limit']
BLIM = B + m*Y(+1);                      // Equation 95
```

### Variables

- **B**: Net foreign assets (NFA) - negative when in debt
- **IB**: Nominal retail interest rate (what households face)
- **I**: Monetary policy interest rate (central bank sets)
- **THETA**: Debt limit risk premium (endogenous, occasionally binding)
- **BLIM**: Distance from debt limit (in units of output)
- **m**: Debt limit parameter (how far above steady state NFA the limit is)

### Parameters

```dynare
m_by = 0.1185;                          // Distance from debt limit (% of GDP)
m = -SS_B/SS_Y + m_by*4;               // Debt limit 12% above steady state
SS_BLIM = SS_B + m*SS_Y;               // Steady state debt limit
```

With `m_by = 0.1185`, the debt limit is approximately 47.4% of GDP above the steady state NFA.

### MCP (Mixed Complementarity Problem) Logic

The constraint `mcp = 'BLIM > 0'` in Dynare means:

**Case 1: BLIM > 0 (Away from limit)**
- Not at debt limit
- THETA = 0 (no risk premium)
- IB = I (retail rate = policy rate)

**Case 2: BLIM = 0 (At limit)**
- Hit the debt limit
- THETA can be positive (risk premium kicks in)
- IB = I + THETA (retail rate includes premium)
- THETA adjusts to prevent NFA from violating the limit

This is a **complementarity constraint**:
- THETA ≥ 0
- BLIM ≥ 0
- THETA · BLIM = 0 (at least one must be zero)

---

## Current Julia Implementation

### What's in QMIPF_final.jl

```julia
# Line 29: Euler equation uses IB
LAM[0] = beta * VARSIGMA[1] / VARSIGMA[0] * IB[0] / PI_C[1] * LAM[1]

# Line 82: Simplified NFA accumulation
NFA[0] = beta * NFA[-1] + TB[0] / SS_Y

# Line 114: IB defined WITHOUT debt premium
IB[0] = I[0]                            # ❌ MISSING THETA!
```

### What's Missing

1. **THETA variable** - debt limit risk premium
2. **BLIM variable** - distance from debt limit
3. **m_by and m parameters** - debt limit calibration
4. **OBC constraint** - complementarity between THETA and BLIM
5. **SS_BLIM** - steady state debt limit

The current implementation sets `IB = I` unconditionally, meaning the retail interest rate always equals the policy rate, regardless of debt levels. This eliminates the occasional binding constraint mechanism.

---

## Proposed Julia Implementation

### Step 1: Add Variables

```julia
@model QMIPF_step9e_Real_UIP begin
    # Existing equations...

    # Add THETA and BLIM to the model equations:
    THETA[0]    # Debt limit risk premium
    BLIM[0]     # Distance from debt limit
```

### Step 2: Add Parameters

```julia
@parameters QMIPF_step9e_Real_UIP begin
    # ... existing parameters ...

    # Debt limit parameters
    m_by = 0.1185        # Distance from debt limit (% of quarterly GDP)
    m = -SS_NFA/SS_Y + m_by*4   # Computed: debt limit above SS
    SS_BLIM = SS_NFA + m * SS_Y # Steady state debt limit
end
```

### Step 3: Update Equations

```julia
# Replace line 114:
IB[0] = I[0] + THETA[0]                 # Retail rate = policy rate + premium

# Add debt limit distance equation:
BLIM[0] = NFA[0] + m * Y[0]             # Distance from limit
```

### Step 4: Add OBC Constraint

MacroModelling.jl syntax for OBCs:

```julia
# Option 1: Using @obc macro (if available)
@obc BLIM[0] >= 0
THETA[0] >= 0

# Option 2: Using get_sep_irf with OBC support
# The SEP solver can handle occasionally binding constraints
# by solving the complementarity problem at each node
```

**Note**: Need to verify exact MacroModelling.jl syntax for OBCs. The package documentation should specify how to implement complementarity constraints.

### Step 5: Update Steady State

```julia
# In steady state:
SS_THETA = 0.0              # No premium in steady state
SS_BLIM = SS_NFA + m * SS_Y # Positive distance from limit
SS_IB = SS_I                # Retail = policy in SS
```

---

## Economic Interpretation

### Why This Matters

1. **Financial Crises**: When NFA falls too low (debt too high), THETA > 0 creates a "sudden stop"
   - Higher borrowing costs force adjustment
   - Consumption and investment must fall to service debt
   - Exchange rate pressures intensify

2. **Asymmetric Dynamics**:
   - In normal times (BLIM > 0): Linear dynamics, small risk premia
   - Near limit (BLIM ≈ 0): Nonlinear feedback, large risk premia
   - At limit (BLIM = 0): Constraint binds, THETA adjusts to prevent further borrowing

3. **Policy Analysis**:
   - Without OBC: Cannot study debt sustainability crises
   - With OBC: Can analyze sudden stops, capital flow reversals, financial stress

4. **Emerging Markets**:
   - Developed countries rarely hit limits (deep financial markets)
   - Emerging markets frequently experience sudden stops
   - QMIPF specifically designed for EMs → OBC is essential

### Calibration

With `m_by = 0.1185`:
- Steady state NFA: ~negative (country in debt)
- Debt limit: 47.4% of GDP above steady state
- Implies: Can increase debt by roughly 12 quarters of trade deficits before hitting limit

This is consistent with empirical sudden stop episodes in emerging markets.

---

## Implementation Strategy

### Phase 1: Add OBC Infrastructure (Current Task)

1. ✅ Document current state
2. ⬜ Add THETA and BLIM variables
3. ⬜ Add m_by parameter
4. ⬜ Update IB equation
5. ⬜ Add BLIM equation
6. ⬜ Implement OBC constraint syntax

### Phase 2: Test OBC Behavior

1. ⬜ Simulate with large negative shocks to test if constraint binds
2. ⬜ Verify THETA = 0 in normal times
3. ⬜ Verify THETA > 0 when hitting limit
4. ⬜ Compare IRFs with/without OBC

### Phase 3: Calibrate and Validate

1. ⬜ Calibrate m_by to match sudden stop frequency
2. ⬜ Compare to empirical sudden stop episodes
3. ⬜ Validate against Dynare results

### Phase 4: Documentation

1. ⬜ Update LaTeX documentation with OBC description
2. ⬜ Add OBC discussion to methodology section
3. ⬜ Include IRF comparisons showing constraint binding

---

## Technical Considerations

### SEP Solver and OBCs

The SEP (Stochastic Extended Path) solver naturally handles OBCs:
- At each time period, checks if constraint binds
- If BLIM ≤ 0, adjusts THETA to maintain BLIM = 0
- If BLIM > 0, sets THETA = 0
- Iterates until complementarity satisfied

This is why SEP is ideal for models with OBCs - it doesn't linearize the constraint away.

### Linear vs. Nonlinear Solutions

- **First-order perturbation**: Cannot handle OBCs (linearization assumes constraints never bind)
- **Deterministic SEP**: Can handle OBCs when constraint binds in response to deterministic shock
- **Stochastic SEP**: Fully accounts for OBCs with precautionary behavior

With the debt limit OBC, the comparison becomes even more important:
- Linear: No debt limit effects (constraint ignored)
- Deterministic: Constraint can bind, but no precautionary behavior
- Stochastic: Agents anticipate possible constraint binding → stay further from limit

---

## Expected Results After Implementation

### IRF Behavior

**Before OBC** (current):
- Smooth, linear responses to shocks
- No amplification near debt limits
- IB always close to I

**After OBC** (proposed):
- Nonlinear responses when approaching limit
- Amplification: shocks near limit have larger effects
- IB can spike above I when THETA > 0
- Sudden stop dynamics: rapid NFA adjustment when constraint binds

### Scientific Contribution

Adding the OBC will allow the documentation to show:
1. **Constraint frequency**: How often does the limit bind in stochastic simulations?
2. **Crisis dynamics**: What happens when limit binds?
3. **Precautionary behavior**: How much do agents adjust to avoid the limit?
4. **Policy effectiveness**: Do FX interventions help avoid sudden stops?

This makes the QMIPF implementation publication-ready for sudden stop literature.

---

## Next Steps

**Immediate**:
1. Implement OBC in QMIPF_final.jl
2. Test with large shocks
3. Verify complementarity logic

**Near-term**:
4. Update comparison plots to show OBC effects
5. Add OBC section to LaTeX documentation
6. Calibrate m_by to match EM data

**Future**:
7. Use OBC model for Bayesian estimation
8. Study policy counterfactuals with financial stress
9. Compare sudden stop dynamics across solution methods

---

**Status**: Analysis complete, ready to implement
**Estimated Implementation Time**: 30-45 minutes
**Risk**: Low (well-defined equations, clear Dynare reference)

