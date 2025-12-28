# SEP Porting Step 17: RBC Validation Run (After Sparse Tree Growth)

## Command
`julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

## Outcome
- **SEP solve still fails to converge** (err ≈ 0.295 after 80 iterations).
- **Responses remain far below Dynare**:
  - Output at t=2: MM ≈ 0.84% vs Dynare ≈ 63.46% (+ shock).
  - Output at t=2: MM ≈ 0.84% vs Dynare ≈ −43.80% (− shock).
- IRF absolute error remains small because both tt and ts are similarly small; levels mismatch remains large.

## Interpretation
Dynamic fishbone structure alone did not close the gap. The remaining issue is likely **nonlinear residuals** (Dynare uses full nonlinear `dynamic_resid`) and/or residual/Jacobian assembly still missing correct nonlinear terms.

## Next Candidate Fix
Implement **nonlinear residual and Jacobian evaluation** in stochastic SEP (similar to deterministic solver), then rerun validation.
