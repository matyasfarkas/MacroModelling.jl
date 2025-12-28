# SEP Porting Plan: Dynare -> MacroModelling.jl

Goal: Port Dynare's Extended Path (SEP) algorithm into MacroModelling.jl and validate against Dynare outputs, with RBC as the primary IRF benchmark.

This document references Dynare SEP sources in `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep` and MacroModelling.jl sources in `src/`. It also references the Dynare RBC benchmark files in `tests/sep_validation/SEP`.

-------------------------------------------------------------------------------
## 1. Dynare SEP: Exact Execution Flow (Ground Truth)

### 1.1 Entry point: `extended_path.m`
File: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/extended_path.m`

Flow:
1. `extended_path_initialization(...)`
2. `extended_path_shocks(...)`
3. For each period t=1..samplesize:
   - Update `spfm_exo_simul(2,:)` with shocks for period t.
   - Build `initialguess = [endogenousvariablespaths(:, 2:end), steady_state]` if t>2.
   - Call `extended_path_core(...)` to solve the stochastic perfect foresight problem.
4. Return `ts` as a `dseries` built from `endogenous_variables_paths`.

Key details:
- Initial condition is `initialconditions` (steady state if empty).
- The solver is re-run for each simulation period; the solver outcome at t becomes the next period's initial state.
- In the RBC Dynare script, `extended_path` is called multiple times to build `tt` and `ts`.

### 1.2 Initialization: `extended_path_initialization.m`
File: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/extended_path_initialization.m`

Key outputs in `pfm` (perfect foresight model struct):
- `pfm.periods` = ep.periods
- `pfm.positive_var_indx`, `pfm.effective_number_of_shocks`, `pfm.Sigma`, `pfm.Omega` (shock covariance and Cholesky)
- `pfm.nodes`, `pfm.weights`, `pfm.nnodes` from `setup_integration_nodes` if `ep.stochastic.order > 0`
- `pfm.block_nbr`, `pfm.world_nbr` from `get_block_world_nbr`
- `pfm.hybrid_order`, optional higher-order correction for hybrid mode

Important:
- If `ep.stochastic.order > 0`, homotopy is disabled: `options_.no_homotopy = true`.
- The integration nodes are constructed for **effective shocks** (positive variance only).

### 1.3 Shocks: `extended_path_shocks.m`
File: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/extended_path_shocks.m`

Key behavior:
- If explicit exogenous shocks are provided, they are used directly and `innovations.positive_var_indx` is derived from non-zero columns.
- Otherwise, shocks are simulated using the covariance matrix and normal draws.
- `spfm_exo_simul` is initialized to steady-state exogenous values with shape `(ep.periods+2, n_exo)`.

### 1.4 Core step: `extended_path_core.m`
File: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/extended_path_core.m`

Key behavior:
- For `order == 0`: use deterministic perfect foresight solver (`perfect_foresight_solver_core`).
- For `order > 0`: use stochastic perfect foresight solver:
  - `algo = 0`: full tree (`solve_stochastic_perfect_foresight_model_0`)
  - `algo = 1`: sparse tree (`solve_stochastic_perfect_foresight_model_1`)

### 1.5 Full tree solver: `solve_stochastic_perfect_foresight_model_0.m`
File: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/solve_stochastic_perfect_foresight_model_0.m`

Key behavior:
- Ensures a **zero node** exists and is first in node list.
- Builds stacked system dimension:
  `dimension = ny*(sum(nnodes.^(0:order-1)) + (periods-order)*world_nbr)`
- Creates index maps (`i_upd_r`, `i_upd_y`, `icA`, `i_cols_1`, `i_cols_j`, `i_cols_T`).
- Uses `ep_problem_0` for residuals/Jacobian and solves with `dynare_solve`.

### 1.6 Sparse tree solver: `solve_stochastic_perfect_foresight_model_1.m`
File: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/solve_stochastic_perfect_foresight_model_1.m`

Key behavior:
- Ensures a **zero node** exists and is first.
- Computes `block_nbr` and `world_nbr` from `get_block_world_nbr`.
- Builds stacked system size: `dimension = ny * block_nbr`.
- Uses `ep_problem_1` and `dynare_solve`.

Sparse tree mapping (from Dynare):
- World count: `world_nbr = 1 + (nnodes-1)*order` (see `get_block_world_nbr`).
- Block count (for residual stacking):
  `block_nbr = order + (nnodes-1)*(order-1)*order/2 + (periods-order)*world_nbr`.

### 1.7 Residuals/Jacobians: `ep_problem_0.m` and `ep_problem_1.m`
Files:
- `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/ep_problem_0.m`
- `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/ep_problem_1.m`

Core points:
- The system is **nonlinear** and uses `dynamic_resid` and `dynamic_g1` (full nonlinear residual and Jacobian).
- Residuals are assembled across periods and worlds.
- Expectation is approximated by **Gauss-Hermite quadrature** weights/nodes; the 0 node is used for the unshocked branch.

-------------------------------------------------------------------------------
## 2. MacroModelling.jl SEP: Current State (Relevant Files)

Key files:
- `src/sep_solver.jl`
- `src/sep_irf.jl`
- `src/MacroModelling.jl` (SEP dispatch in `solve!`)

Key current behaviors:
- SEP solver in `sep_solve_mm!` computes a **linearized** residual with `∇₊, ∇₀, ∇₋, ∇ₑ`.
- Deterministic path solver (`solve_deterministic_path`) is also **linearized**.
- Shock covariance `Σ` is inferred from parameters `z_<shock>`. If not found, defaults to `0.01^2`.
- `sep_irf.jl` provides tree extraction and simulation-based IRFs, but the tree extraction path does not update groups for `order > 1`.

-------------------------------------------------------------------------------
## 3. RBC Dynare Benchmark Reference

Benchmark files:
- Dynare model: `tests/sep_validation/SEP/rbc.mod`
- IRF outputs: `tests/sep_validation/SEP/RBC_irf_pos3.csv`, `tests/sep_validation/SEP/RBC_irf_neg3.csv`
- Helper scripts: `tests/sep_validation/SEP/pdss.m`, `tests/sep_validation/SEP/spfirf.m`

Dynare IRF workflow (from `rbc.mod`):
1. `tt = extended_path(steady_state, 80, innovations)` with `innovations(1)=+3`.
2. Build `ts` via a **funnel** of decreasing order (from `maxorder` down to 0), using repeated 1-period solves for order>0 and a final 80-period solve at order=0.
3. Compute IRFs: `pdss(tt) - pdss(ts)` and save to CSV.
4. Repeat for `innovations(1)=-3`.

-------------------------------------------------------------------------------
## 4. Key Gaps vs Dynare (Porting Targets)

### 4.1 Nonlinear residuals
Dynare solves the **full nonlinear residuals** via `dynamic_resid` and `dynamic_g1` inside `ep_problem_0/1`.
MacroModelling SEP currently uses **linearized** residuals in both stochastic and deterministic solvers.

### 4.2 Zero-node ordering and integration
Dynare ensures a zero node is present and placed first, so the no-shock path is always index 1.
MacroModelling does not explicitly enforce the zero node ordering in the GH nodes (verify for all integration methods).

### 4.3 Sparse tree topology
Dynare’s sparse tree uses specific **block and world indexing** (see `get_block_world_nbr` and `ep_problem_1`).
MacroModelling’s sparse tree uses a fishbone logic but does not match Dynare’s block counting scheme.

### 4.4 Funnel baseline (ts)
Dynare builds `ts` by decreasing order, with repeated 1-step solves, then a long order=0 solve.
MacroModelling RBC test does not implement this sequence.

### 4.5 Shock scaling and shock naming
Dynare uses `innovations(1) = 3` and model equation `efficiency = rho*efficiency(-1) + sigma*epsilon`.
MacroModelling RBC model uses `sigma * ϵ[x]`. To match Dynare, the deterministic shock must be `ϵ=3`, not `3*sigma`.

-------------------------------------------------------------------------------
## 5. Porting Plan (Precise Steps)

### Phase A: Establish exact Dynare parity for RBC (no new features)

1. **Align shock sequence with Dynare**
   - Use the CSV files in `tests/sep_validation/SEP` as the canonical benchmark.
   - Ensure `innovations(1)=±3` (not scaled by `sigma` again).
   - In MacroModelling tests, set `sep_deterministic_shocks[1] = ±3`.

2. **Fix time indexing in test extraction**
   - SEP layout uses `voff[t+1]` for time t. Extract t-th period with `layout.voff[t+1]`.
   - This must align with Dynare’s `tt.data(1,:)` (period 1).

3. **Implement Dynare’s funnel baseline (ts)**
   - For order = maxorder downto 0:
     - if order == maxorder: solve 1 period with shock value (scalar).
     - if 0 < order < maxorder: solve 1 period with shock value 0.
     - if order == 0: solve `T` periods with zero shock sequence.
   - Concatenate the period-2 outputs into `ds` just like Dynare.

4. **Match pdss**
   - Use `pdss` definition: `100*(data - data[1]) / data[1]`.
   - Ensure `data[1]` corresponds to period 1 (post-shock period).

5. **Ensure deterministic solver uses nonlinear residuals**
   - For RBC validation parity, deterministic path solver must use full nonlinear residuals (dynamic equations), not the linearized Jacobian.
   - This is required because Dynare’s `extended_path` is nonlinear even for order=0.

Deliverable: `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl` reproduces Dynare CSVs for both +3 and -3 and for both `tt` and `ts`.

### Phase B: Stochastic SEP parity (full tree and sparse tree)

6. **Implement full tree solver parity**
   - Mirror Dynare’s `solve_stochastic_perfect_foresight_model_0`:
     - Use nonlinear residuals and Jacobians (MacroModelling has symbolic Jacobians).
     - Construct the same stacked system dimensions and indexing as Dynare.
     - Enforce zero-node ordering and weight alignment.

7. **Implement sparse tree solver parity**
   - Mirror Dynare’s `solve_stochastic_perfect_foresight_model_1`:
     - Ensure `world_nbr = 1 + (nnodes-1)*order`.
     - Ensure `block_nbr` formula matches Dynare.
     - Reproduce `ep_problem_1` indexing and node assignment logic.

8. **Hybrid corrections (optional)**
   - Dynare applies hybrid corrections (`pfm.h_correction`) for higher-order integration.
   - Decide if MacroModelling will support `hybrid_order` and, if so, implement analogously.

Deliverable: MacroModelling SEP reproduces Dynare outputs for `order > 0` and both `algo=0` and `algo=1` (full and sparse).

### Phase C: Integration and API compatibility

9. **SEP API alignment**
   - Ensure `solve!(..., algorithm=:stochastic_extended_path)` can accept:
     - explicit shock sequences (Dynare’s `innovations`),
     - deterministic initial conditions, and
     - options for integration algorithm and node count.

10. **Record and surface model outputs**
    - Expose `Y` and layout in a way that is compatible with Dynare’s per-period per-world extraction.

-------------------------------------------------------------------------------
## 6. Validation Protocol (RBC Only, Precise)

### 6.1 Inputs
- Model: `models/RBC_Dynare.jl`
- Dynare reference: `tests/sep_validation/SEP/rbc.mod`
- CSV targets: `tests/sep_validation/SEP/RBC_irf_pos3.csv` and `tests/sep_validation/SEP/RBC_irf_neg3.csv`

### 6.2 Required checks
1. **Deterministic IRF parity (order=0)**
   - Build tt: `extended_path(steady_state, T, innovations)` with innovations(1)=±3.
   - Build ts: funnel from order=maxorder down to 0.
   - Compute `pdss(tt) - pdss(ts)`.
   - Compare with CSV columns.

2. **Stochastic IRF parity (order>0)**
   - Replicate with order=10 and node count 3.
   - Confirm node ordering puts zero first.
   - Ensure shock mapping to node index matches Dynare indexing.

3. **Symmetry/asymmetry**
   - Compare +3 and -3 responses for asymmetry.

4. **Numerical tolerance**
   - Set explicit max absolute and relative tolerances (e.g., `1e-8` abs, `1e-6` rel).

-------------------------------------------------------------------------------
## 7. Concrete Implementation Tasks (Actionable Checklist)

### Task 1: Deterministic solver parity
- Replace linearized residual in `solve_deterministic_path` with nonlinear residual evaluation.
- Use MacroModelling’s dynamic residual and Jacobian (equivalent to Dynare’s `dynamic_resid` and `dynamic_g1`).
- Add terminal condition at steady state, consistent with Dynare.

### Task 2: Node ordering and shock mapping
- Add a zero-node check in SEP GH nodes; reorder nodes/weights so the zero node is index 1.
- Update `sep_irf` extraction logic to follow Dynare’s node indexing (node index maps to innovation shock values).

### Task 3: Sparse tree indexing parity
- Replace current fishbone logic with a structure that matches Dynare’s `block_nbr` and `world_nbr` formulas.
- Implement index maps equivalent to `i_upd_r`, `i_upd_y`, `icA` (see `solve_stochastic_perfect_foresight_model_1.m`).

### Task 4: Funnel baseline support
- Implement a helper to build `ts` via decreasing order solves:
  - `order=maxorder`: one-period solve with shock
  - `order in (maxorder-1..1)`: one-period solve with zero shock
  - `order=0`: T-period solve with zero sequence
- Store intermediate `ds` consistently with Dynare.

### Task 5: RBC validation harness
- Update `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl` to:
  - Use local CSV files in `tests/sep_validation/SEP`.
  - Use correct shock magnitude (±3) for the RBC model.
  - Implement `ts` funnel.
  - Compare both `tt` and `ts` as Dynare does.

-------------------------------------------------------------------------------
## 8. Questions / Clarifications Needed

1. Should MacroModelling’s SEP support Dynare’s hybrid correction (`pfm.hybrid_order`) now, or is this out of scope for the initial port?
2. Do you want exact replication of Dynare’s block/world indexing for sparse tree, or is functional equivalence acceptable?
3. For deterministic order=0, should MacroModelling use the same nonlinear solver options as Dynare (`stack_solve_algo`, `solve_algo`)?
4. Is the RBC validation intended to accept only `Tensor-Gaussian-Quadrature` with 3 nodes, or should other integration algorithms be supported in the parity checks?
5. Should SEP results be stored in a Dynare-compatible structure (e.g., a `dseries`-like wrapper), or is the current `sep_solution` layout sufficient for validation?

-------------------------------------------------------------------------------
## 9. Review Log (Clarity & Precision)

Review pass 1:
- Checked cross-references to Dynare file names and key functions.
- Verified formulas for `world_nbr` and `block_nbr` are copied from `get_block_world_nbr.m`.

Review pass 2:
- Verified Dynare RBC workflow steps and the funnel sequence against `tests/sep_validation/SEP/rbc.mod`.
- Verified shock scaling statement matches `efficiency = rho*efficiency(-1) + sigma*epsilon`.

Review pass 3:
- Ensured all claims about MacroModelling SEP are tied to current file paths (`src/sep_solver.jl`, `src/sep_irf.jl`).
- Checked that each action step maps to a Dynare source function.

