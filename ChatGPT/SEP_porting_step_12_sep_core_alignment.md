# SEP Porting Step 12: Core SEP Alignment (Initial Conditions, Shock Ordering, t=1 Shock)

## Objective
Bring MacroModelling SEP core closer to Dynare’s SEP behavior without overhauling the tree structure yet. Focus on:
- Fixing initial conditions (y0 fixed)
- Ordering Gauss–Hermite nodes so zero node is first (Dynare ordering)
- Ensuring deterministic shock is used at t=1 in trunk residuals (no node shocks)

## Changes Applied
### 1) Fixed initial conditions in stochastic SEP
File: `src/sep_solver.jl`
- **Problem**: stochastic SEP Newton updates were modifying y0, but Dynare treats initial conditions as fixed.
- **Fix**: store `y0_fixed` after initialization and zero out the Newton update for y0. Reapply `y0_fixed` after each update.

### 2) Dynare-compatible node ordering (zero first)
File: `src/sep_solver.jl`
- **Problem**: Gauss–Hermite nodes were not guaranteed to have the zero node first; Dynare explicitly reorders nodes.
- **Fix**:
  - Added `reorder_1d_zero_first!` and applied it to 1D GH nodes for both full and sparse node generation.
  - Kept `reorder_nodes_zero_first!` for full-tree nodes after covariance transformation.
  - **Important**: removed post-hoc reordering in `build_sparse_shock_nodes` to avoid breaking `(h,k)` → column mapping in `get_shock_for_branch`.

### 3) Deterministic shock at t=1 for trunk nodes
File: `src/sep_solver.jl`
- **Problem**: trunk residuals were using node shocks at t=1; Dynare uses deterministic shock at period 1 and integrates nodes only for future periods.
- **Fix**:
  - In branching nodes, when `t == 1`, use deterministic shocks only (if provided); ignore node shocks for the current-period residual.
  - For `t >= 2`, use node shocks as before (Dynare assigns nodes to current-period shocks for i>1).

## Notes / Remaining Gaps
- The **fishbone tree structure** is still simplified (constant group count), whereas Dynare’s fishbone grows the number of worlds over time. This may still cause mismatches.
- Side-branch shocks are still applied at all periods for those branches; Dynare applies the node shock only at the branch period.
- Shock variance mapping (`z_<shock>` vs default) and horizon alignment are not addressed yet.

## Next Checks
- Run RBC validation to see how much the mismatch shrinks.
- If mismatch persists, implement dynamic fishbone group counts and branch-time shock logic.
