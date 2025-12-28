# SEP Porting Step 09: Apply Deterministic Shock Sequences in Stochastic SEP

Goal
- Make deterministic shock sequences compatible with stochastic SEP when `sep_order > 0`.
- This is needed for Dynare’s funnel baseline (order>0) which still uses stochastic expectations with a deterministic innovation at t=1.

Change Summary
1) Deterministic shocks now apply inside the stochastic SEP residuals
- If `opts.deterministic_shocks` is provided and `opts.order > 0`, the solver no longer switches to the deterministic path.
- The deterministic shock `ε_det[t]` is added to the node shock `ε_to_child` before computing the residual.

2) Behavior for non-branching nodes
- For sparse side branches and for trunk nodes after branching, `ε_det[t]` is added to the residual’s shock term as well.

Implementation Notes
- Added `ε_det` and `ε_tmp` workspace buffers in `sep_solve_mm!`.
- For branching nodes:
  - `ε_tmp = ε_to_child + ε_det[t]`
- For sparse side branches:
  - `ε_tmp = ε_branch + ε_det[t]`
- For non-branching trunk nodes:
  - `r = ... + ∇ₑ * ε_det[t]`

Files Touched
- `src/sep_solver.jl`

Next
- Re-run `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl` to check `ts` and `IRF` against Dynare.

