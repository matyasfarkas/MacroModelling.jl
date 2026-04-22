# Debt Limit Calibration to 3% Binding Frequency

**Date**: January 15, 2026
**Task**: Calibrate m_by so debt limit binds in 3% of simulations
**Status**: 🔄 In Progress

---

## Objective

Calibrate the debt limit parameter `m_by` such that the occasionally binding constraint (OBC) binds in approximately **3% of periods** during a 10,000-period stochastic simulation.

---

## Model Setup

### Debt Limit Equations

```julia
# Distance from debt limit
BLIM[0] = NFA[0] + m * Y[0]

# Risk premium (using MacroModelling's max operator)
THETA[0] = max(0.0, -penalty_kappa * BLIM[0] / SS_Y)

# Retail interest rate
IB[0] = I[0] + THETA[0]
```

### Parameters

- **m_by**: Debt limit distance as % of quarterly GDP (TO BE CALIBRATED)
- **penalty_kappa = 10**: Reduced for steady state convergence
- **m = -SS_NFA/SS_Y + m_by × 4**: Derived debt limit parameter

### Constraint Binding Condition

The constraint binds when:
- BLIM ≤ 0 → NFA ≤ -m × Y
- Which triggers THETA > 0
- Which increases retail rate IB = I + THETA
- Forcing current account adjustment

---

## Steady State Solution

### Problem Fixed ✅

**Issue**: Original formulation with `max()` caused steady state solver to fail

**Solution**: MacroModelling.jl has its own `max` operator that handles steady state computation correctly. Combined with reduced `penalty_kappa = 10`, the steady state now solves successfully.

**Verification**:
```julia
# Steady state values (example)
SS_NFA = 0.0           # Balanced position
SS_Y = 3.715           # Output
SS_BLIM = m × SS_Y > 0 # Positive distance from limit
SS_THETA = 0.0         # No premium in SS
```

---

## Calibration Method

### Step 1: Measure Current Binding Frequency

Run 10,000-period stochastic simulation with current m_by = 0.1185:

```julia
sim = simulate(model, param_values; periods=10000, algorithm=:first_order)
theta_series = sim[THETA_idx, :]
binding_periods = sum(theta_series .> 1e-6)
binding_pct = 100.0 * binding_periods / 10000
```

### Step 2: Binary Search Algorithm

To find optimal m_by:

1. **Initial bounds**: m_by ∈ [0.01, 1.0]
   - Low value → tight limit → binds often
   - High value → loose limit → binds rarely

2. **Iteration**:
   - Try midpoint: m_by_mid = (m_by_low + m_by_high) / 2
   - Measure binding frequency
   - If freq > 3%: increase m_by (looser limit)
   - If freq < 3%: decrease m_by (tighter limit)

3. **Convergence**: Stop when |freq - 3%| < 0.5%

### Step 3: Verify Calibration

- Run final simulation with calibrated m_by
- Confirm binding frequency ≈ 3%
- Check that THETA behavior is economically reasonable

---

## Expected Outcomes

### Binding Frequency Interpretation

**3% binding frequency** means:
- Out of 10,000 quarterly periods (2,500 years)
- Debt limit binds in ~300 quarters (75 years)
- On average, once every 33 quarters (~8 years)
- Consistent with empirical sudden stop frequency in emerging markets

### Economic Interpretation

**Before binding** (97% of time):
- Normal times
- THETA = 0
- Standard monetary policy transmission
- NFA can fluctuate freely

**When binding** (3% of time):
- Financial stress episode
- THETA > 0 (risk premium spikes)
- IB > I (retail rate above policy rate)
- Forced current account adjustment
- Consumption compressed
- Trade balance improves

### Parameter Relationship

The relationship between m_by and binding frequency depends on:
- Shock volatilities (larger shocks → more binding)
- NFA persistence (high persistence → longer binding episodes)
- Policy response (FX intervention can prevent binding)

**Expected calibration**: m_by ≈ 0.05-0.20 for 3% binding
- If m_by too high: rarely binds (<1%)
- If m_by too low: binds frequently (>10%)

---

## Implementation Details

### File: scripts/calibrate_debt_limit_3pct.jl

The calibration script:

1. **Loads model**: `include("../models/QMIPF_final.jl")`

2. **Defines measurement function**:
   ```julia
   function measure_binding_frequency(model, m_by_value; n_periods=10000)
       # Update m_by parameter
       # Recompute m and SS_BLIM
       # Run stochastic simulation
       # Count THETA > threshold
       # Return binding percentage
   end
   ```

3. **Implements binary search**:
   ```julia
   function calibrate_m_by(model; target_pct=3.0, tolerance=0.5)
       # Binary search over m_by ∈ [0.01, 1.0]
       # Iterate up to 10 times
       # Return calibrated m_by
   end
   ```

4. **Reports results**:
   - Current binding frequency
   - Calibrated m_by value
   - Final binding frequency
   - Instructions for updating model file

---

## First-Order Approximation Considerations

### Potential Limitation

**First-order perturbation** (linearization) may not fully capture the OBC:
- Linearizes around THETA = 0
- May understate the effect of constraint binding
- Agents don't exhibit precautionary behavior

**Implications**:
- Calibrated m_by may differ from what would work with SEP
- Binding frequency may be approximate
- For research purposes, should verify with nonlinear solver

### Verification Recommended

After calibration with first-order, verify using **Stochastic Extended Path (SEP)**:

```julia
# Run SEP simulation (if computational resources allow)
sim_sep = simulate(model, param_values;
                   periods=1000,  # Shorter due to computational cost
                   algorithm=:stochastic_extended_path,
                   sep_order=1,
                   sep_nnodes=3)

# Measure binding frequency with SEP
# Should be close to first-order result
```

---

## Diagnostic Checks

### After Calibration, Verify:

1. **THETA distribution**:
   - Should be zero most of the time
   - Occasional spikes when binding
   - No negative values (constraint prevents)

2. **BLIM distribution**:
   - Usually positive
   - Occasionally hits zero (3% of time)
   - Should not go significantly negative

3. **NFA dynamics**:
   - Should be stationary
   - Mean around SS_NFA
   - Occasional large deviations (before binding)

4. **IB behavior**:
   - Usually IB ≈ I (THETA ≈ 0)
   - Spikes when constraint binds (IB > I)
   - Spikes should be rare (3% of time)

---

## Output Files

### Generated Files

1. **scripts/calibrate_debt_limit_3pct.jl**
   - Calibration script
   - Binary search algorithm
   - Measurement functions

2. **Documentation/DEBT_LIMIT_CALIBRATION_APPROACH.md** (this file)
   - Methodology documentation
   - Expected results
   - Diagnostic checks

3. **Documentation/DEBT_LIMIT_CALIBRATION_RESULTS.md** (to be created)
   - Calibration results
   - Final m_by value
   - Binding frequency statistics
   - Diagnostic plots

### Model Update

After successful calibration, update:

**File**: `models/QMIPF_final.jl`
**Line**: 244

```julia
# Before:
m_by = 0.1185  # Default value

# After:
m_by = 0.XXXX  # Calibrated to achieve 3% binding frequency
```

---

## Troubleshooting

### If Calibration Fails

**Problem**: Simulation errors or numerical issues

**Solutions**:
1. Check steady state solved correctly
2. Verify penalty_kappa = 10 (not too large)
3. Ensure shock volatilities are reasonable
4. Try shorter simulation first (1,000 periods) to debug

**Problem**: Binding frequency too sensitive to m_by

**Solutions**:
1. Widen tolerance (±1% instead of ±0.5%)
2. Average over multiple simulations
3. Use different random seeds

**Problem**: Never binds (even with low m_by)

**Solutions**:
1. Check shock volatilities (may be too small)
2. Verify THETA equation is active
3. Try nonlinear solver (SEP) instead of first-order

**Problem**: Always binds (even with high m_by)

**Solutions**:
1. Check steady state NFA (should be near zero)
2. Verify m computation is correct
3. Check for model mis-specification

---

## Next Steps After Calibration

1. **Update model file** with calibrated m_by

2. **Regenerate IRF plots** showing:
   - Normal shock (no binding)
   - Large shock (binding occurs)
   - THETA behavior

3. **Add to documentation**:
   - Calibration methodology section
   - Table showing binding frequency by shock type
   - Discussion of sudden stop dynamics

4. **Verify with SEP**:
   - Run shorter simulation with SEP
   - Compare binding frequency
   - Document any differences

5. **Policy analysis**:
   - Test FX intervention effectiveness
   - Compare outcomes with/without OBC
   - Analyze welfare implications

---

**Status**: 🔄 Calibration script running
**Expected completion**: 5-10 minutes
**Output location**: `/tmp/claude/.../tasks/b49ba02.output`

