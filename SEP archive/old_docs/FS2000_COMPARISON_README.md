# FS2000 Model: Dynare vs MacroModelling.jl Comparison

## Why FS2000?

The **FS2000** (Schorfheide 2000) model is the **simplest possible test case** for validating SEP implementation:

### Model Specs
- **Variables**: 14 endogenous variables
- **Shocks**: 2 exogenous shocks (TFP: e_a, Money growth: e_m)
- **Type**: Cash-in-advance RBC model
- **Complexity**: Very simple - linear equations, no New Keynesian frictions
- **Dynare status**: Standard Dynare example, well-tested

### Advantages for Testing

1. **Fast**: SEP should converge in seconds (not minutes)
2. **Simple**: Linear constraints, no complex nonlinearities
3. **Reliable**: Known to work in Dynare extended_path
4. **Debuggable**: Small enough to inspect all equations

## Comparison vs Previous Models

| Model | Variables | Shocks | Complexity | Dynare Status |
|-------|-----------|--------|------------|---------------|
| **FS2000** | 14 | 2 | Very simple | ✓ WORKS |
| Gali_2015 | 22 | 3 | Moderate (NK frictions) | ✗ Failed |
| SW07_HLT | 66 | 7 | High (Kimball aggregation) | ? (Too large) |

## Files

### Dynare
- `fs2000.mod` - Already configured with SEP tests (lines 149-207)
  - Has `steady_state_model` block (lines 119-143)
  - Shock variances: e_a (0.014²), e_m (0.005²)
  - SEP options already added

### MacroModelling.jl
- Model: Loaded from MacroModelling package
  - Path: `~/.julia/packages/MacroModelling/fkrRI/models/FS2000.jl`
  - Pre-defined, no translation needed
- Test script: `test_fs2000_sep_comparison.jl`

## How to Run

### Step 1: MacroModelling.jl Test

```bash
julia --project=. test_fs2000_sep_comparison.jl
```

**Expected output:**
- SEP(1) converges in ~5-10 seconds
- SEP(2) converges in ~10-20 seconds
- Clean convergence (error < 1e-7)

**Key variables to extract:**
- `y` - Output
- `c` - Consumption
- `R` - Interest rate
- `n` - Labor
- `k` - Capital

### Step 2: Dynare Test

**In MATLAB or Octave:**
```matlab
dynare fs2000
```

**Expected output:**
- `steady` and `check` complete successfully
- Extended path test 1 (order=1) converges
- Extended path test 2 (order=2) converges
- Key steady state values printed

**Expected runtime:** ~30-60 seconds total

### Step 3: Compare Results

Compare the "Key steady state values" from both outputs:

```
MacroModelling SEP(1):
  y  = [value]
  c  = [value]
  R  = [value]
  ...

Dynare SEP order=1:
  y  = [value]
  c  = [value]
  R  = [value]
  ...
```

**Acceptable difference:** < 1e-4 (absolute value)

## Why FS2000 Should Work

### 1. Model Equations are Simple

From `fs2000.mod`:
- Euler equation (line 89)
- Budget constraint (line 99)
- Cash-in-advance constraint (line 101)
- Production function (line 107)
- No complex New Keynesian features
- No Calvo pricing complications

### 2. Dynare Has Analytical Steady State

The `steady_state_model` block (lines 119-143) provides **closed-form** steady state values:
```matlab
steady_state_model;
  dA = exp(gam);
  gst = 1/dA;
  m = exp(logmst);
  khst = ( (1-gst*bet*(1-del)) / (alp*gst^alp*bet) )^(1/(alp-1));
  ...
end;
```

This means:
- No numerical steady state solver needed
- Perfect initial guess for SEP
- Fewer convergence issues

### 3. Well-Tested in Dynare

FS2000 is a **standard Dynare example**:
- Included in Dynare distribution
- Used in documentation
- Tested extensively by Dynare team
- Known to work with extended_path

## Shock Variance Specification

**Important:** The FS2000.mod file specifies shock **standard deviations**:

```matlab
shocks;
var e_a; stderr 0.014;  % Standard deviation
var e_m; stderr 0.005;  % Standard deviation
end;
```

This is **different** from the Gali model, which had:
```matlab
shocks;
var eps_a = 1;  % Variance (unitless, scaling in equations)
end;
```

**FS2000 approach is correct** for models where shocks appear directly (not pre-scaled).

## Expected Outcomes

### If Test Succeeds (Most Likely)

✓ Both Dynare and MacroModelling converge
✓ Values match within tolerance (< 1e-4)
✓ This validates:
  - SEP algorithm implementation
  - Gauss-Hermite quadrature
  - Newton solver
  - Shock scaling methodology

**Next steps:**
- Try incrementally more complex models
- Investigate why Gali failed (likely model-specific issue, not algorithm bug)

### If Test Fails

Possible issues:

1. **Dynare convergence failure**
   - Try increasing `options_.ep.maxit`
   - Try different `solve_algo` (1, 2, 3, 4)
   - Check Dynare version compatibility

2. **MacroModelling convergence failure**
   - Check model loading
   - Verify shock variance parameters match
   - Increase `sep_maxit`

3. **Both converge, but values differ**
   - Check quadrature method match
   - Verify shock scaling is equivalent
   - Compare deterministic steady states first
   - Check solver tolerance settings

## Model Details

### Parameters (Estimated in Schorfheide 2000)

```
alp = 0.356      % Capital share
bet = 0.993      % Discount factor
gam = 0.0085     % Long-run TFP growth
mst = 1.0002     % Long-run money growth (exp(logmst))
rho = 0.129      % Money growth autocorrelation
psi = 0.65       % Labor weight in utility (phi in .mod)
del = 0.01       % Depreciation rate
z_e_a = 0.035449 % TFP shock std dev
z_e_m = 0.008862 % Money shock std dev
```

### Equilibrium Conditions

1. **Euler equation**: Intertemporal consumption choice
2. **Labor supply**: Intratemporal labor-consumption trade-off
3. **Firm optimality**: Marginal product = factor price
4. **Credit market clearing**: Loans = Money - 1 + Dividends
5. **Cash-in-advance**: P*c = m
6. **Resource constraint**: c + k = y + (1-del)*k(-1)

## References

**Original Paper:**
- Schorfheide, F. (2000). "Loss function-based evaluation of DSGE models." *Journal of Applied Econometrics*, 15(6), 645-670.

**Dynare Documentation:**
- https://archives.dynare.org/documentation/examples.html
- Example file: `fs2000.mod`

**MacroModelling.jl:**
- Model file: `~/.julia/packages/MacroModelling/fkrRI/models/FS2000.jl`
- Translated from Dynare example

## Summary

FS2000 is the **ideal starting point** for Dynare validation:
- Simple enough to debug
- Complex enough to be meaningful
- Well-tested and reliable
- Should work if SEP implementation is correct

If FS2000 works, it builds confidence for testing larger models.
If FS2000 fails, it indicates a fundamental issue to investigate.
