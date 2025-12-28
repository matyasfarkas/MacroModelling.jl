# SEP Porting Step 16: Sparse Tree Growth + Branch-Time Shocks

## Objective
Align MacroModelling’s fishbone (sparse tree) layout with Dynare’s growth over time and apply node shocks only at branch creation times.

## Changes Applied
File: `src/sep_solver.jl`

1. **Dynamic group counts (fishbone growth)**
   - `G[t+1]` now increases by `(K-1)` per period until `t = Lbr+1`:
     - `t=1`: 1 world (trunk)
     - `t=2`: `1+(K-1)` worlds
     - …
     - `t=Lbr+1`: `1+(K-1)*Lbr` worlds
   - This matches `get_block_world_nbr.m` for Dynare’s sparse tree world count.

2. **Branch-time mapping functions**
   - Added `branch_info_sparse(layout, g)` to map group index → `(branch_time, node_index)`.
   - Added `branch_group_index(layout, branch_time, node_index)` to map back.

3. **Sparse tree parent/child logic**
   - `parent_group_sparse`: side branches use trunk as parent only at their branch time; otherwise parent is self.
   - `child_groups_sparse`: trunk branches using node-to-branch mapping; side branches never branch.
   - Trunk node behavior:
     - `t=1`: nodes map to branches created at `t=2`.
     - `t>=2`: nodes map to branches created at `t`.

4. **Branch-time shock application**
   - Side branches now apply node shocks **only at their branch time**.
   - For later periods, side branches use deterministic shocks (if any) or zero.

## Status
- Structural alignment improved, but SEP still fails to converge (see Step 17 results).
- Next likely required change: use **nonlinear residuals/Jacobians** in stochastic SEP (Dynare uses nonlinear `dynamic_resid`).
