# Gali Model: Dynare vs MacroModelling.jl SEP Comparison

## Overview

This guide provides step-by-step instructions for comparing Stochastic Extended Path (SEP) solutions between Dynare and MacroModelling.jl using the Gali_2015_chapter_3_nonlinear model.

**Why this model?**
- Small (22 variables, 3 shocks) → fast execution
- Nonlinear New Keynesian model with price dispersion
- Good test case before running large SW07_HLT model

## Files

- `Gali_2015_chapter_3_nonlinear.mod` - Dynare model file
- `models/Gali_2015_chapter_3_nonlinear.jl` - MacroModelling.jl model
- `test_gali_sep_comparison.jl` - MacroModelling test script
- `export_gali_to_dynare.jl` - Export script (for reference)

## Step 1: Run MacroModelling.jl SEP Tests

```bash
cd /Users/matyasfarkas/Documents/GitHub/MacroModelling.jl
julia --project=. test_gali_sep_comparison.jl
```

**Configuration:**
- SEP(1): `sep_periods=10, sep_order=1, sep_nnodes=3`
- SEP(2): `sep_periods=10, sep_order=2, sep_nnodes=3`
- Integration: Gauss-Hermite quadrature with 3 nodes

**Expected output:**
```
======================================================================
TEST 1: SEP(1) - extended_path(periods=10, order=1)
======================================================================
Configuration:
  - periods (T) = 10
  - order (branching length) = 1
  - nnodes = 3 (Gauss-Hermite quadrature)

Solution status:
  - Convergence: SUCCESS
  - Final error: <1e-7
  - Runtime: ~10-30 seconds

Key steady state values (SEP order=1):
  Variable      Value
  ------------------------------
  Y             0.95057982
  C             0.95057982
  Pi            1.00000000
  R             1.01010101
  N             0.93465527
```

## Step 2: Run Dynare Extended Path

**In MATLAB or Octave:**
```matlab
cd /Users/matyasfarkas/Documents/GitHub/MacroModelling.jl
dynare Gali_2015_chapter_3_nonlinear
```

**Dynare Configuration:**
- Test 1: `extended_path(periods=10, order=1)` - Branching length = 1
- Test 2: `extended_path(periods=10, order=2)` - Branching length = 2
- Solver: Trust region (solve_algo=4)
- Tolerance: 1e-5 for both f and x
- Max iterations: 500

**Expected runtime:** ~30-60 seconds per test

## Step 3: Compare Results

### Key Variables to Compare

| Variable | Description | MacroModelling | Dynare |
|----------|-------------|----------------|--------|
| Y | Output | | |
| C | Consumption | | |
| Pi | Inflation | | |
| R | Nominal interest rate | | |
| N | Labor | | |

### Expected Differences

**Sources of potential differences:**

1. **Integration method**:
   - MacroModelling: Gauss-Hermite quadrature (nnodes=3)
   - Dynare: Default quadrature (check with `options_.ep.stochastic.order`)

2. **Solver tolerance**:
   - MacroModelling: 1e-7 (default)
   - Dynare: 1e-5 (configured in .mod file)

3. **Solver algorithm**:
   - MacroModelling: Newton with line search
   - Dynare: Trust region (solve_algo=4)

4. **Initial guess**:
   - Both start from deterministic steady state

**Acceptable difference:** < 1e-4 in absolute value for key variables

## Critical Fix: Shock Variance Scaling

### The Issue

The Gali model equations include shock scaling:
```matlab
log(A(0)) = rho__a * log(A(-1)) + std_a * eps_a;  % Line 43
log(Z(0)) = rho__z * log(Z(-1)) - std_z * eps_z;  % Line 45
nu(0) = rho__nu * nu(-1) + std_nu * eps_nu;       % Line 47
```

Notice: `std_a * eps_a` - shock is **already scaled**

### Correct Configuration

When shocks are pre-scaled in model equations, the `shocks` block should declare **unit variance**:

```matlab
shocks;
var eps_a = 1;      % NOT std_a^2
var eps_nu = 1;     % NOT std_nu^2
var eps_z = 1;      % NOT std_z^2
end;
```

**Why?**
- Effective variance = (variance in shocks block) × (scaling in equation)²
- With pre-scaling: Var(std_a * eps_a) = std_a² × Var(eps_a)
- If `var eps_a = 1`, then Var(std_a * eps_a) = std_a² ✓
- If `var eps_a = std_a^2`, then Var(std_a * eps_a) = std_a⁴ ✗

## Troubleshooting

### Error: "No convergence of the (stochastic) perfect foresight solver"

**Possible causes:**
1. Shock variances too large (check scaling issue above)
2. Periods too long (try reducing from 10 to 5)
3. Solver tolerance too tight (try 1e-4 instead of 1e-5)
4. Wrong solver algorithm (try solve_algo=1 instead of 4)

**Solutions:**
```matlab
% Try relaxing tolerance
options_.ep.tolerance.f = 1e-4;
options_.ep.tolerance.x = 1e-4;

% Or reduce periods
extended_path(periods=5, order=1);
```

### Error: "Matrix is too large to convert to linear index"

This indicates the solver failed to allocate memory for the solution. Possible fixes:

1. Reduce periods: `extended_path(periods=5, order=1)`
2. Reduce branching: Start with `order=1` only
3. Check model determinacy: Run `stoch_simul(order=1)` first

### MacroModelling vs Dynare Differences > 1e-3

If results differ substantially:

1. **Check integration method**: Verify Dynare uses same quadrature
   ```matlab
   disp(options_.ep.stochastic.order);  % Should be 0 for Gauss-Hermite
   ```

2. **Check shock variances**: Verify both use same effective variances
   ```matlab
   % In Dynare after running
   disp(M_.Sigma_e);  % Shock covariance matrix
   ```

3. **Check solver convergence**: Both should converge cleanly
   ```matlab
   % Check Dynare convergence flag
   disp(oo_.deterministic_simulation.status);
   ```

## Model Parameters

**Calibration:**
```
sigma = 1.0          % CRRA utility
varphi = 5.0         % Frisch elasticity inverse
phi_p_i = 1.5        % Taylor rule inflation response
phi_y = 0.125        % Taylor rule output gap response
theta = 0.75         % Calvo parameter
rho_nu = 0.5         % Monetary shock persistence
rho_z = 0.5          % Preference shock persistence
rho_a = 0.9          % Technology shock persistence
beta = 0.99          % Discount factor
eta = 3.77           % Money demand elasticity
alpha = 0.25         % Returns to scale
epsilon = 9.0        % Elasticity of substitution
tau = 0.0            % Subsidy
std_a = 0.01         % Technology shock std
std_z = 0.05         % Preference shock std
std_nu = 0.0025      % Monetary shock std
```

## Next Steps

1. **If Gali comparison succeeds**: Run SW07_HLT comparison
   - Expected runtime: ~15-30 minutes per test
   - File: `Smets_Wouters_2007_HLT.mod`

2. **Document methodology differences**: If results differ, investigate:
   - Quadrature node placement
   - Shock transformation details
   - Solver algorithm details

3. **Validate against Dynare test suite**: Check Dynare's own SEP tests
   - Location: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/dynare EP/matlab/ep/`
   - Files: `rs.mod`, `rstrue.mod`

## References

- Gali (2015), *Monetary Policy, Inflation, and the Business Cycle*, Chapter 3
- Dynare manual: Extended Path method
- MacroModelling.jl SEP implementation: `src/sep_solver.jl`
