# Dynare SEP Validation - Current Status

**Date:** December 27, 2025
**Goal:** Validate MacroModelling.jl SEP implementation against Dynare extended_path

## Summary

We have successfully established the infrastructure for comparing SEP (Stochastic Extended Path) solutions between MacroModelling.jl and Dynare. The validation approach uses **model export** (MacroModelling → Dynare) rather than model import, which proved more reliable.

## Key Accomplishments

### 1. Fixed Critical Type System Issue ✓

**Problem:** MacroModelling's `@model` macro couldn't process Float64 literals from Dynare translations.

**Error:**
```
MethodError: Cannot `convert` an object of type Float64 to an object of type Union{Int64, Expr, Symbol}
```

**Solution:** Modified `src/MacroModelling.jl:2936`
```julia
# Before:
function simplify(ex::Expr)::Union{Expr,Symbol,Int}

# After:
function simplify(ex::Expr)::Union{Expr,Symbol,Int,Float64}
```

**Impact:** Unblocked all Dynare validation work.

### 2. Established Bidirectional Translation Workflow ✓

**Export (MacroModelling → Dynare):**
- `write_mod_file(model)` generates syntactically correct .mod files
- Preserves parameter values, equations, and initial values
- Successfully tested on SW07_HLT (66 vars, 7 shocks) and Gali (22 vars, 3 shocks)

**Files:**
- `Smets_Wouters_2007_HLT.mod` - 317 lines
- `Gali_2015_chapter_3_nonlinear.mod` - 166 lines

### 3. Created Test Infrastructure ✓

**MacroModelling Test Scripts:**
- `test_gali_sep_comparison.jl` - Gali model SEP tests
- `extract_sep_values.jl` - SW07 value extraction
- `export_sw07_to_dynare.jl` - Export script
- `export_gali_to_dynare.jl` - Export script

**Dynare Configuration:**
- Added `extended_path(periods=10, order={1,2})` commands
- Configured solver options for convergence
- Added diagnostic output for key variables

### 4. Fixed Shock Variance Scaling Issue ✓

**Problem:** Dynare extended_path was failing to converge due to incorrect shock variance specification.

**Root Cause:** Model equations include pre-scaling:
```matlab
log(A(0)) = rho__a * log(A(-1)) + std_a * eps_a;  % Already scaled!
```

**Incorrect (previous):**
```matlab
shocks;
var eps_a = std_a^2;  % WRONG: Double scaling!
end;
```

**Correct (fixed):**
```matlab
shocks;
var eps_a = 1;  % RIGHT: Unit variance, scaling in equation
end;
```

**Why:** Effective variance = (shocks block variance) × (equation scaling)²
- Correct: Var(std_a * eps_a) = std_a² × 1 = std_a² ✓
- Wrong: Var(std_a * eps_a) = std_a² × std_a² = std_a⁴ ✗

### 5. Optimized Dynare Solver Configuration ✓

Added to `Gali_2015_chapter_3_nonlinear.mod`:
```matlab
options_.ep.stochastic.order = 0;       % Gauss-Hermite quadrature
options_.ep.verbosity = 1;               % Diagnostic output
options_.ep.maxit = 500;                 % Max Newton iterations
options_.ep.tolerance.f = 1e-5;          % Function tolerance
options_.ep.tolerance.x = 1e-5;          % Variable tolerance
options_.solve_algo = 4;                 % Trust region solver
```

## Current Status

### Ready for Testing ✓

**Gali Model (Small, Fast):**
- MacroModelling: `test_gali_sep_comparison.jl` ready to run
- Dynare: `Gali_2015_chapter_3_nonlinear.mod` configured with SEP tests
- Expected runtime: ~30-60 seconds per test
- Status: **Ready for user to run Dynare test**

**SW07_HLT Model (Large, Comprehensive):**
- MacroModelling: SEP(1) already converged (797 sec, err=6.6e-8)
- Dynare: `Smets_Wouters_2007_HLT.mod` ready
- Expected runtime: ~15-30 minutes per test
- Status: **User requested NOT to run yet (memory concerns)**

### Documentation ✓

**Guides Created:**
1. `GALI_DYNARE_COMPARISON_GUIDE.md` - Step-by-step comparison instructions
2. `DYNARE_VALIDATION_WORKFLOW.md` - Bidirectional translation workflow
3. `DYNARE_VALIDATION_STATUS.md` - This file

## Next Steps

### Immediate (Waiting for User)

1. **Run Dynare test on Gali model:**
   ```matlab
   dynare Gali_2015_chapter_3_nonlinear
   ```

2. **Compare MacroModelling vs Dynare output:**
   - Key variables: Y, C, Pi, R, N
   - Expected difference: < 1e-4 (absolute)
   - Check both SEP(1) and SEP(2)

### If Gali Comparison Succeeds

1. **Run SW07_HLT comparison** (when memory available)
2. **Document any systematic differences** between implementations
3. **Validate IRF computation** from SEP solutions

### If Gali Comparison Shows Differences

**Potential sources of differences:**

1. **Integration method:**
   - MacroModelling: Gauss-Hermite with nnodes=3
   - Dynare: Check `options_.ep.stochastic.order`

2. **Shock discretization:**
   - Verify both use √3 scaling for GH nodes
   - Check effective shock variances match

3. **Solver convergence:**
   - Compare final residuals
   - Check number of Newton iterations

4. **Numerical precision:**
   - Tolerance differences (1e-7 vs 1e-5)
   - Floating point accumulation

## Technical Details

### Model: Gali_2015_chapter_3_nonlinear

**Size:**
- Variables: 22 (endogenous)
- Shocks: 3 (eps_a, eps_nu, eps_z)
- Parameters: 16

**Shocks:**
- Technology: σ_a = 0.01
- Preference: σ_z = 0.05
- Monetary policy: σ_nu = 0.0025

**Key Features:**
- New Keynesian model with Calvo pricing
- Price dispersion dynamics
- Taylor rule monetary policy
- Non-linear Phillips curve

### SEP Configuration

**Common Settings:**
- Periods (T): 10
- Branching length: 1 (order=1), 2 (order=2)
- Integration: Gauss-Hermite quadrature

**MacroModelling:**
```julia
solve!(model,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = false)
```

**Dynare:**
```matlab
extended_path(periods=10, order=1);
```

## Known Issues

### Resolved ✓

1. **Float64 type error** - Fixed in `src/MacroModelling.jl:2936`
2. **Shock variance scaling** - Fixed in all .mod files
3. **Dynare solver configuration** - Optimized for convergence

### Monitoring

1. **Dynare convergence** - May still fail if:
   - Model is too nonlinear
   - Shocks are too large
   - Initial guess is poor
   - Solver hits iteration limit

2. **Memory constraints** - SW07_HLT uses 90% memory in MATLAB
   - Solution: Test Gali first, then SW07 when resources available

## References

**Files:**
- `src/MacroModelling.jl` - Core package with Float64 fix
- `src/sep_solver.jl` - SEP implementation
- `models/Gali_2015_chapter_3_nonlinear.jl` - Julia model
- `Gali_2015_chapter_3_nonlinear.mod` - Dynare model
- `test_gali_sep_comparison.jl` - Test script

**Literature:**
- Gali (2015), *Monetary Policy, Inflation, and the Business Cycle*
- Fair & Taylor (1983), "Solution and Maximum Likelihood Estimation of Dynamic Nonlinear Rational Expectations Models"
- Dynare Manual, Extended Path Method
