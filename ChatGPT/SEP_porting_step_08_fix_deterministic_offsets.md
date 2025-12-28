# SEP Porting Step 08: Fix Deterministic Padding Offsets

Goal
- Correct index offsets after adding padding to deterministic `Y` so the solver reads and updates the right segments.

Cause
- `Y` now has a leading unused slot (index 1). The solver still sliced `Y` as if there was no padding, causing a dimension mismatch when applying Newton updates.

Change Applied
- Updated state extraction to add a +1 offset:
  - `y_lag = view(Y, 1 .+ (t-1)*ny_ .+ (1:ny_))`
  - `y_cur = view(Y, 1 .+ t*ny_ .+ (1:ny_))`
  - `y_fwd = view(Y, 1 .+ (t+1)*ny_ .+ (1:ny_))`
- Updated Newton update slice to skip the padding and y₀:
  - `Y[(ny_+2):end] .+= α * Δ`

Files Touched
- `src/sep_solver.jl`

Next
- Re-run `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`.

