# Step 19 - Add sep_initial_state and force SEP re-solves; align RBC funnel baseline

## Goal
Match Dynare's funnel baseline and avoid SEP cache reuse when deterministic shocks or initial state changes. This is required for correct ts paths and for negative shock runs.

## Changes
- `src/sep_solver.jl`
  - Added `initial_state` support in `sep_solve_mm!` and `solve_deterministic_path`.
  - If `sep_initial_state` is provided, overwrite y0 and keep it fixed in the stochastic SEP solve.
  - Added validation for `sep_initial_state` length.
- `src/MacroModelling.jl`
  - Added `sep_initial_state` argument to `solve!`.
  - Added SEP re-solve logic: always recompute SEP when deterministic shocks or initial state are provided, or when previous SEP config differs (periods/order/nodes/sparse_tree).
- `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`
  - Pass `sep_initial_state=prev` in `funnel_baseline` for each order step (Dynare-style chaining).
  - Pass `sep_initial_state=dss` for the shocked tt solve.

## Why this change
- Dynare uses the previous-period state as the initial condition for each order step in the funnel baseline. Without `sep_initial_state`, our ts path was anchored at steady state each step.
- SEP caching ignored changes in deterministic shocks, so the negative shock path reused the positive shock solution.

## Tests
- `julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl` (output captured in `/tmp/rbc_sep_validation.log`).

## Results
Positive shock (+3 sigma):
- tt (Output) max abs error: 0.013978%, mean abs error: 0.002666%.
- ts (Output) max abs error: 0.000311%, mean abs error: 0.000034%.
- IRF max abs error: 0.014000% (relative error large because Dynare IRF near 0).

Negative shock (-3 sigma):
- tt (Output) max abs error: 0.014002%, mean abs error: 0.002577%.
- ts (Output) max abs error: 0.000498%, mean abs error: 0.000045%.
- IRF max abs error: 0.014000% (relative error large because Dynare IRF near 0).

## Notes
- The tt and ts paths now match Dynare within ~1.4e-2% absolute error for Output, and ts within ~5e-4%.
- IRF relative errors are dominated by periods where Dynare IRF is near zero; absolute errors remain small.
