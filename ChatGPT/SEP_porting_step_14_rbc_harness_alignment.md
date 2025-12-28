# SEP Porting Step 14: RBC Harness Alignment to Dynare Settings

## Objective
Align the RBC validation harness with Dynare’s SEP configuration (horizon length, shock scaling, and funnel baseline procedure).

## Changes Applied
File: `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

1. **Horizon aligned to Dynare**
   - Added `dynare_horizon = 400` (matches `options_.ep.periods` in `rbc.mod`).
   - All `solve!` calls now use `sep_periods = dynare_horizon`.
   - IRF comparison still uses `total_periods = 60` (first 60 periods only).

2. **Shock size mapped to parameter (standard deviation)**
   - Added `shock_std_param` helper to pull `z_<shock>` if present; fallback to `1.0` (Dynare default for missing shocks block in MacroModelling context).
   - Deterministic shock sequences are now set to `shock_value * shock_std`.
   - Printed applied shock magnitude after scaling.

3. **Funnel baseline updated to Dynare horizon**
   - `funnel_baseline` now takes `sep_horizon` and always solves with horizon 400.
   - For order=0, the deterministic solve uses horizon 400 and extracts first `total_periods` values.
   - For orders >0, horizon remains 400 but only the first period is taken.

4. **Initial guess handling**
   - For order=0, `initial_guess` length updated to `sep_horizon + 1` to match deterministic solver layout.
   - For stochastic orders, `initial_guess` is omitted to avoid size mismatch warnings.

## Notes
- Shock std defaults to 1.0 unless a `z_<shock>` parameter exists. This matches Dynare’s `var shock = 1` convention and avoids double-scaling when the model equation already multiplies by `sigma`.
- This step aligns the harness inputs; it does not change the SEP tree structure itself.
