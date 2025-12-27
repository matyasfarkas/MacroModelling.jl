# SEP IRF Zero-Response Bug Investigation

**Date**: December 25, 2024
**Issue**: SEP IRFs return machine-epsilon values (~1e-11) instead of meaningful responses

## Summary of Findings

### What Works ✅
1. **Shock covariance extraction**: Fixed to extract actual σ values from model parameters
   - Was: `Σ = 0.01×I` (hardcoded tiny values)
   - Now: `Σ[i,i] = (z_shock)^2` (e.g., σ_epinf = 0.1455)

2. **GH node transformation**: Correctly transforms nodes by shock covariance
   - Verified: Group 1175 has epinf = 0.252 ≈ √3 × 0.1455 ✓

3. **Group indexing for IRF**: Correctly calculates tensor product group index
   - For shock i at node n, uses base-nnodes arithmetic ✓

4. **Shock Jacobian ∇ₑ**: Non-zero and reasonable
   - Size: (66, 7) with 8 non-zero elements
   - epinf enters equation 40 with coefficient -0.0473 ✓

### The Problem ❌

Despite all components being correct, the SEP solution shows:
- Group 1 (no shock): deviation = 1.73e-11
- Group 1175 (epinf shock): deviation = 1.84e-11
- **Difference: 1.08e-12** (numerical noise!)

Even with 200 iterations and tighter tolerance, variables stay at steady state.

## Technical Details

### SEP Tree Structure (L=1, nnodes=3, nshocks=7)
- K = 3^7 = 2187 total shock combinations
- t=0: 1 group (steady state)
- t=1: 2187 groups (K^1, one branching)
- t≥2: 2187 groups (K^Lbr, no more branching)

### Residual Computation (Two Branches)

**Branch 1** (t ≤ Lbr, lines 319-389): Expectation integration WITH shocks
```julia
for (kidx, cg) in enumerate(child_groups(layout, t, g))
    ε_curr = view(X, :, kidx)
    r = ∇₊*Δy_fwd + ∇₀*Δy_cur + ∇₋*Δy_lag + ∇ₑ*ε_curr  # ← HAS SHOCKS
    r_sum += W[kidx] * r
end
```

**Branch 2** (t > Lbr, lines 390-441): Deterministic continuation WITHOUT shocks
```julia
r = ∇₊*Δy_fwd + ∇₀*Δy_cur + ∇₋*Δy_lag  # ← NO SHOCKS
```

This is CORRECT - after branching period, no new shocks arrive.

### Current Hypothesis

The SEP solver computes EXPECTED residuals at t=1:
- For group g=1 at t=0, child_groups returns 1:2187
- Loops over all K shock combinations
- Computes weighted average: r_sum = Σ W[k] * r[k]

**Question**: Is the solver finding Y such that the EXPECTED residual is zero (correct),
but this doesn't produce variation across different shock realizations?

### What to Investigate Next

1. ❓ **Check residuals for specific groups**: Are residuals for group 1175 actually zero?
2. ❓ **Verify shock application**: Print ε_curr values during solver iterations
3. ❓ **Check if Y values are being updated**: Add diagnostics to Newton step
4. ❓ **Test on simpler model**: Try 1-shock model to isolate the issue

## Files Modified

1. `src/sep_solver.jl`:
   - Lines 191-224: Extract shock covariance from model parameters
   - Lines 191-203: Added diagnostic output for ∇ₑ

2. `src/sep_irf.jl`:
   - Lines 47-83: Fixed tensor product group indexing
   - Line 68: Get nshocks from layout.dε

## Next Steps

Need to add detailed logging to understand:
- What shocks are actually being applied during Newton iterations
- Whether Y values are being updated at all
- If there's a fundamental algorithmic issue with expectation integration

The fact that even 200 iterations with perfect convergence (err<1e-7) produces
essentially zero deviations suggests a deeper structural problem, not just convergence issues.
