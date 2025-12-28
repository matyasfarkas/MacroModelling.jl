# SEP Porting Step 10: RBC Validation Run (Nonlinear Deterministic + Stochastic with Deterministic Shocks)

Command
- `julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

Outcome
- Test completed without runtime errors.
- Convergence succeeded for both the stochastic SEP pass (order=10 with deterministic shocks) and the funnel baseline.

Key Observations
1) Shock covariance warning
- The solver warns:
  `Shock std parameter z_ϵ not found, using default 0.01`
- For RBC, the shock standard deviation is parameter `sigma = 0.1`, not `z_ϵ`.
- This likely distorts the integration nodes and expectations in stochastic SEP.

2) Positive shock results (tt)
- tt deviations are negative where Dynare is positive (e.g., Output t=2: -5.7580 vs +63.4580).
- Indicates a sign or scaling mismatch in the stochastic SEP path with deterministic shocks.

3) ts baseline and IRF
- `ts` still diverges from Dynare, but IRF error is much smaller than before (max abs ~4.98%).
- `efficiency` columns show `Inf`/`NaN` in both outputs due to baseline level being zero (consistent with Dynare CSVs).

4) Negative shock results
- Signs are flipped compared with Dynare and magnitudes are off (relative error ~0.8–0.9 range).

Raw Error Summary (Output variable)
- Positive shock:
  - tt max abs error: 69.215973%
  - ts max abs error: 69.207973%
  - IRF max abs error: 4.980542%
- Negative shock:
  - tt max abs error: 38.041027%
  - ts max abs error: 38.055027%
  - IRF max abs error: 4.980242%

Interpretation
- Stochastic SEP is running and converging.
- Deterministic shock sequences are now applied inside the stochastic solver.
- Remaining mismatch likely driven by:
  1) Shock standard deviation mapping (`z_ϵ` missing; uses 0.01 instead of `sigma=0.1`).
  2) Potential sign convention in residuals (∇ₑ entry is -0.1).

Next Actions Suggested
1) Confirm how to map shock std dev for RBC (and generally):
   - Should we map missing `z_<shock>` to `sigma` when the model has a single shock?
   - Or allow an explicit override from the test harness?
2) Decide whether to flip sign in the residual shock term to align with Dynare’s sign convention (if needed).

Files
- No code changes in this step; this is a test run only.

