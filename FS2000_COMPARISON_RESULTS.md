# FS2000 Dynare vs MacroModelling Comparison Results

## Dynare Results (WORKING ✓)

**Configuration:**
- Extended path periods=10
- Order 1 and 2
- Runtime: 1 minute 51 seconds total

**SEP Order=1:**
```
y  = 0.57984277
c  = 0.44341618
R  = 1.02777510
```

**SEP Order=2:**
```
y  = 0.57978829
c  = 0.44345105
R  = 1.02777510
```

**Deterministic Steady State:**
```
m          1.011
P          2.25815
c          0.447711
e          1
W          4.5959
R          1.02121
k          5.80122
d          0.849425
n          0.187216
l          0.860425
gy_obs     1.003
gp_obs     1.00797
y          0.580765
dA         1.003
```

## MacroModelling Results (NEEDS FIX ✗)

**Configuration:**
- SEP periods=10, order=1, nnodes=3
- Runtime: 1.44 seconds
- Convergence: error=8.026e-12 ✓

**SEP Order=1:**
```
y  = 0.31672915   (vs Dynare 0.58)
c  = 2.71856217   (vs Dynare 0.44)
R  = 0.00000000   (vs Dynare 1.03)  ← WRONG!
n  = 1.00020000
k  = 1.00853623
```

**SEP Order=2:** NOT RUN (showing same values as order=1)

## Problem Diagnosis

### Issue 1: Auxiliary Variables

**Root cause:** FS2000 model has leads≥2 (c(+1), P(+1)) which create auxiliary variables:
- `Pᴸ⁽¹⁾` - auxiliary for P lead
- `cᴸ⁽¹⁾` - auxiliary for c lead

**Current fix:** Initialize auxiliary variables to zero (line 177 in sep_solver.jl)
```julia
push!(yss, 0.0)  # For auxiliary variables
```

**Problem:** Zero initialization may cause solver to find wrong equilibrium!

**Better fix needed:** Compute auxiliary variable values from steady state:
- For lead aux: `Pᴸ⁽¹⁾ = P` (at steady state, P(+1) = P)
- For lag aux: use lagged steady state value

### Issue 2: SEP(2) Not Running

The test script shows identical values for SEP(1) and SEP(2), suggesting SEP(2) didn't actually run or didn't update the solution.

**Possible causes:**
1. SEP(2) solve! call failed silently
2. Model solution object not updated
3. Wrong solution extraction

### Issue 3: Values Don't Match

Even if we fix the auxiliary variable issue, need to investigate why values differ so much from Dynare:

**Hypotheses:**
1. **Auxiliary variable init**: Wrong initial guess → wrong equilibrium
2. **Shock scaling**: Different effective variances
3. **Model differences**: FS2000.jl vs fs2000.mod may have subtle differences
4. **Quadrature**: Different node placement or weights

## Next Steps

### 1. Fix Auxiliary Variable Initialization (CRITICAL)

Modify `src/sep_solver.jl` line 169-179:

```julia
# Build yss vector
yss = Float64[]
for var in 𝓂.var
    if var ∈ ss_keys
        push!(yss, Float64(SS_result(var)))
    else
        # For auxiliary variables: use steady state of parent variable
        # Pattern: var_nameᴸ⁽ⁿ⁾ → use steady state of var_name
        parent_var = # extract parent from auxiliary variable name
        if parent_var ∈ ss_keys
            push!(yss, Float64(SS_result(parent_var)))
        else
            push!(yss, 0.0)
        end
    end
end
```

###2. Debug SEP(2)

Check why SEP order=2 shows same values:
- Add diagnostic output before each solve!
- Verify solution object is updated
- Check if solve! with order=2 actually runs

### 3. Compare Deterministic Steady States

Before fixing SEP, verify basic model match:
```julia
using MacroModelling
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

# Get steady state
ss = get_steady_state(FS2000)

# Compare with Dynare
println("MacroModelling SS:")
println("  y = ", ss(:y))
println("  c = ", ss(:c))
println("  R = ", ss(:R))
```

If deterministic SS doesn't match Dynare, there's a model specification difference.

### 4. Shock Variance Check

Verify effective shock variances match:
```julia
# MacroModelling
z_e_a = 0.035449
z_e_m = 0.008862

# Dynare
var e_a; stderr 0.014;  → variance = 0.014² = 0.000196
var e_m; stderr 0.005;  → variance = 0.005² = 0.000025
```

But MacroModelling uses **parameters** z_e_a and z_e_m as std devs:
- z_e_a = 0.035449 → variance should be 0.035449²
- Dynare stderr 0.014 doesn't match!

**ACTION:** Check if FS2000.jl shock scaling matches fs2000.mod

## Summary

**Status:** Dynare working, MacroModelling not matching

**Critical fixes needed:**
1. Auxiliary variable initialization
2. SEP(2) execution bug
3. Verify model specification match
4. Check shock variance scaling

**Good news:**
- Auxiliary variable fix unblocked execution
- SEP solver converges cleanly
- Fast execution (1.4s vs 2min in Dynare)
