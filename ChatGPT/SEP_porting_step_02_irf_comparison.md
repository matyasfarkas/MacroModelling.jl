# SEP Porting Step 02: IRF Computation and Comparison Outputs

Goal
- Ensure the RBC validation explicitly reports IRFs as % deviations from deterministic steady state and cross-checks both `tt` and `ts` components against Dynare CSVs.

Scope
- File updated: `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`
- No solver changes.

Why
- Dynare defines IRFs as percent deviations from deterministic steady state:
  `pdss(tt) = 100 * (tt.data(1:T)/tt.data(1) - 1)`.
- The CSVs store `pdss(ts)` and `pdss(tt)` separately. The full IRF is often interpreted as `pdss(tt) - pdss(ts)`.
- The previous harness compared only `tt` to Dynare; this step adds explicit `ts` and IRF comparisons.

Changes Applied
1) Add comparison output for `ts`
- New section prints per-period comparisons for `ts` (same format as `tt`).
- Summary stats for Output variable are now shown for `ts`.

2) Add IRF = `tt - ts` diagnostics
- Compute `irf_pct = tt_pct - ts_pct`.
- Compute `dynare_irf = dynare_tt - dynare_ts`.
- Print summary stats for the Output variable.

Behavior Notes
- The percent deviation formula is unchanged (`pdss` uses first row as the steady state level).
- The comparison is still anchored to Dynare CSVs in `tests/sep_validation/SEP`.

Files Touched
- `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

