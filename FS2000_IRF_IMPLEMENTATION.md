# FS2000 IRF Comparison Implementation

## Summary

Added **stochastic IRF computation** to both Dynare and MacroModelling.jl test files to enable dynamic comparison beyond steady states.

## Implementation Details

### Dynare (fs2000.mod)

**Location**: Lines 209-308

**Method**: Conditional forecast using perfect foresight solver

**Algorithm**:
```matlab
% For each shock:
1. Baseline: perfect_foresight_solver with shocks = 0
2. Shocked: perfect_foresight_solver with shock(1) = 1σ
3. IRF = shocked - baseline
```

**Shocks tested**:
- `e_a` (TFP): σ = 0.035449
- `e_m` (Money growth): σ = 0.008862

**Horizon**: 20 periods

**Output**: First 10 periods of IRF for variables {y, c, R, n, k}

**Key features**:
- Uses deterministic perfect foresight
- Agents know exact shock sequence
- Nonlinear solution method
- Computationally intensive (~2-3 minutes)

### MacroModelling.jl (test_fs2000_sep_comparison.jl)

**Location**: Lines 125-249

**Method**: First-order perturbation approximation

**Algorithm**:
```julia
1. solve!(FS2000, algorithm = :first_order)
2. irf_e_a = get_irf(FS2000, :e_a, periods=20)
3. irf_e_m = get_irf(FS2000, :e_m, periods=20)
```

**Shocks tested**:
- `e_a` (TFP): σ = 0.035449
- `e_m` (Money growth): σ = 0.008862

**Horizon**: 20 periods

**Output**: First 10 periods of IRF for variables {y, c, R, n, k}

**Key features**:
- Uses linear perturbation (first-order Taylor expansion)
- Infinitesimal shock approximation
- Fast computation (~1 second)
- Approximate for nonlinear models

## Methodological Differences

| Aspect | Dynare | MacroModelling.jl |
|--------|--------|-------------------|
| **Method** | Perfect foresight | Perturbation |
| **Shock assumption** | Known deterministic path | Infinitesimal (ε→0) |
| **Solution** | Nonlinear | Linear approximation |
| **Computational cost** | High (~2-3 min) | Low (~1 sec) |
| **Accuracy** | Exact for perfect foresight | Approximate for small shocks |
| **Limitations** | Assumes agents know shocks | Ignores higher-order terms |

## Why They Differ

### 1. Perfect Foresight vs. Perturbation

**Dynare approach** (perfect foresight):
- Solves: F(y_{t-1}, y_t, y_{t+1}, ε_t) = 0 for all t
- With: ε_1 = σ, ε_t = 0 for t > 1
- Agents know exact shock path at t=0

**MacroModelling approach** (perturbation):
- Approximates: y_t ≈ ȳ + A(y_{t-1} - ȳ) + Bε_t
- Linearizes around steady state ȳ
- Valid for small ε (infinitesimal approximation)

### 2. Shock Magnitude

**Dynare**: Uses actual σ = 0.035449 for e_a
- This is **not infinitesimal** (3.5% shock)
- Nonlinear effects may be significant

**MacroModelling**: Assumes ε→0 in derivation
- Then scales linear response by σ
- Works well if y(σε) ≈ σ·y(ε)

### 3. When Results Differ

IRFs will differ significantly when:
1. **Nonlinearity**: If y = aε + bε², perturbation misses bσ²ε² term
2. **Occasionally binding constraints**: Perturbation can't handle kinks
3. **State-dependent dynamics**: Nonlinear propagation mechanisms
4. **Large shocks**: σ = 0.035 may be "large" for some models

IRFs should be similar when:
1. Model is nearly linear around steady state
2. Shocks are small (σ < 0.01 typically)
3. No binding constraints or regime switches

## How to Interpret Comparison

### If IRFs match closely (< 10% difference):
✅ **Interpretation**:
- Model is approximately linear for these shocks
- Perturbation approximation is valid
- MacroModelling SEP solution likely correct

### If IRFs differ significantly (> 50% difference):
⚠️ **Possible causes**:
1. **Nonlinearity**: Model has strong nonlinear propagation
2. **Implementation error**: Bug in one of the methods
3. **Shock scaling**: Mismatch in shock standard deviations
4. **Steady state**: Different reference points

**Next steps**:
- Verify shock sizes match exactly
- Check steady states are identical
- Test with smaller shocks (σ/10) to see if IRFs converge
- Use SEP-based IRF in MacroModelling (future work)

### If signs differ:
❌ **Red flag**: This indicates a serious problem
- Check model specification (parameters, equations)
- Verify shock enters equations the same way
- Check for transcription errors

## Future Work: SEP-Based IRF for MacroModelling

**Goal**: Match Dynare's methodology more closely

**Implementation**:
```julia
# Conditional forecast using SEP
function compute_sep_irf(model, shock_idx, shock_size, periods)
    # Baseline: SEP solution with ε = 0 at t=1
    shocks_baseline = zeros(nshocks, periods)
    path_baseline = simulate_sep_conditional(model, shocks_baseline)

    # Shocked: SEP solution with ε = σ at t=1
    shocks_shocked = zeros(nshocks, periods)
    shocks_shocked[shock_idx, 1] = shock_size
    path_shocked = simulate_sep_conditional(model, shocks_shocked)

    # IRF = difference
    return path_shocked - path_baseline
end
```

**Requirements**:
1. Implement conditional shock simulation in SEP
2. Modify SEP solver to accept exogenous shock paths
3. Resolve nonlinear system for each conditional forecast

**Benefits**:
- True nonlinear IRF (like Dynare's perfect foresight)
- Accounts for higher-order terms
- Better comparison with Dynare results

**Challenges**:
- Computational cost (2 SEP solutions per shock)
- Implementation complexity
- May require new solver features

## Files Modified

1. **fs2000.mod** (lines 209-308)
   - Added IRF computation section
   - Uses `perfect_foresight_setup` and `perfect_foresight_solver`
   - Reports first 10 periods for key variables

2. **test_fs2000_sep_comparison.jl** (lines 125-249)
   - Added IRF computation using perturbation
   - Includes warning about methodology difference
   - Reports first 10 periods matching Dynare output

3. **FS2000_PARAMETERS_UPDATED.md**
   - Added IRF comparison methodology section
   - Documented differences and interpretation guidelines

## Running the Tests

### Dynare:
```matlab
cd /Users/matyasfarkas/Documents/GitHub/MacroModelling.jl
dynare fs2000
```

Output includes:
- SEP steady states (order 1 and 2)
- IRF to e_a (TFP shock)
- IRF to e_m (money shock)

### MacroModelling.jl:
```bash
julia --project=. test_fs2000_sep_comparison.jl
```

Output includes:
- SEP steady states (order 1 and 2)
- Perturbation IRF to e_a
- Perturbation IRF to e_m
- Warning about methodology differences

## Summary

**Status**: ✅ IRF comparison implemented in both systems

**Current limitation**: Different methodologies (perfect foresight vs perturbation)

**Expected**: IRFs may differ quantitatively, should agree qualitatively

**Future work**: Implement SEP-based conditional forecast for true comparison
