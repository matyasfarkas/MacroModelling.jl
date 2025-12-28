# SEP Porting Step 05: Fix Missing MacroTools Import

Goal
- Fix the runtime error during deterministic SEP construction: `UndefVarError: MacroTools not defined`.

Cause
- New helper `replace_symbols_local` uses `MacroTools.postwalk`, but `MacroTools` was not imported in `src/sep_solver.jl`.

Change Applied
- Added `MacroTools` to the imports at the top of `src/sep_solver.jl`:
  `using SparseArrays, LinearAlgebra, ForwardDiff, MacroTools`

Files Touched
- `src/sep_solver.jl`

Next
- Re-run `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`.

