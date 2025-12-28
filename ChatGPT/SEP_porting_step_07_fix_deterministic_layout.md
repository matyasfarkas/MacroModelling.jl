# SEP Porting Step 07: Fix Deterministic Layout Indexing

Goal
- Resolve bounds errors in deterministic SEP layout indexing (`layout.voff`) when accessing the final period.

Cause
- `index_y(layout, t, g)` uses `layout.voff[t+1] + (1:ny_)`.
- Deterministic layout starts `voff` at 1, so index 1 is unused.
- `Y` was allocated with length `ny*(T+1)`, which is too short for the extra padding implied by `voff`.

Change Applied
- Increased deterministic `Y` length to include one unused padding slot:
  - `nvars_total = ny_ * (T + 1) + 1`
- Updated initialization to fill `Y[2:end]` with steady state values.
- Added support for `initial_guess` of length `nvars_total - 1` by padding the front with one zero.

Files Touched
- `src/sep_solver.jl`

Next
- Re-run `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`.

