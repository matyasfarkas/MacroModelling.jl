# SEP Porting Step 15: RBC Validation Run (Post-Core/Harness Fixes)

## Command
`julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

## Key Results (Positive Shock)
- **SEP solve did not converge** (err ≈ 0.291 after 80 iterations).
- `∇ₑ` shows `-0.1` (matches sign from residual form).
- Shock std fallback: `z_ϵ` missing → default 1.0.
- **MM tt/ts responses are tiny** compared to Dynare:
  - Output at t=2: MM ≈ 1.54% vs Dynare ≈ 63.46%.
  - Errors in tt and ts remain large (absolute errors ≈ 60+%).
- IRF error (tt - ts) small in absolute terms (≈ 1.35%), but that reflects both tt and ts being near each other and too small.

## Key Results (Negative Shock)
- Similar issue: MM responses are small and of opposite sign relative to Dynare magnitudes.
- Output t=2: MM ≈ +1.54% vs Dynare ≈ −43.80%.

## Interpretation
- The SEP system is still not matching Dynare behavior.
- Likely causes (still unresolved):
  1. **Fishbone tree structure** (group count grows over time in Dynare; MacroModelling uses constant groups).
  2. **Side-branch shock timing** (node shock should apply only at branch time; current code applies it every period).
  3. **Stochastic SEP residuals are linearized**, while Dynare uses full nonlinear residuals.

## Next Actions
- Implement dynamic fishbone group growth and branch-time shock logic.
- Decide whether to switch stochastic SEP to nonlinear residuals for exact Dynare alignment.
- Rerun validation after each change.
