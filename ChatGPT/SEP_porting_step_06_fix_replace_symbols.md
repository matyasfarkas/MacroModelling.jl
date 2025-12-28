# SEP Porting Step 06: Fix replace_symbols_local Method Dispatch

Goal
- Fix the runtime error when calling `replace_symbols_local` with `Dict{Symbol, Symbol}` during dynamic residual construction.

Cause
- `replace_symbols_local` accepted only `Dict{Symbol, Any}`; Julia does not dispatch `Dict{Symbol, Symbol}` to that method.

Change Applied
- Relaxed the signature to accept any value type:
  `replace_symbols_local(exprs, remap::Dict{Symbol,<:Any})`

Files Touched
- `src/sep_solver.jl`

Next
- Re-run `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`.

