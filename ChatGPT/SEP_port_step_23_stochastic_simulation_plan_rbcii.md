# SEP Porting Step 23: Stochastic Simulation Plan for Dynare `rbcii.mod`

## Objective
Replicate Dynare’s stochastic extended path simulations for the RBC-II model (`rbcii.mod`) in MacroModelling. The goal is to reproduce the Dynare `ts0..ts5` paths (orders 0–5), especially the Investment series over periods 120–400.

## Source References
- Dynare model: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/ep-mj-30-years-master/models/rbcii/rbcii.mod`
- Dynare SEP implementation: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep`
  - `extended_path.m`
  - `extended_path_core.m`
  - `extended_path_initialization.m`
  - `extended_path_shocks.m`
  - `setup_integration_nodes.m`
  - `solve_stochastic_perfect_foresight_model_1.m`

## Dynare Behavior Summary (rbcii.mod)
- **Simulation length:** `samplesize = 400`.
- **Shock:** `epsilon`, VAR(1) process `efficiency = rho*efficiency(-1) + sigma*epsilon` with `sigma=0.007`.
- **Integration nodes:** `options_.ep.stochastic.quadrature.nodes = 3` and `IntegrationAlgorithm = 'Tensor-Gaussian-Quadrature'`.
- **SEP order:** runs `options_.ep.stochastic.order = 0..5`.
- **Sparse tree:** `options_.ep.stochastic.algo = 1` (sparse / fishbone).
- **Random seed:** `set_dynare_seed(0)` before each run.
- **Complementarity constraint:**
  - `LagrangeMultiplier = 0 ⟂ Investment > @{ZLB}*0.241741953339345` (with `ZLB = 0.85`).
- **Output to match:** `ts0..ts5`, plotted as `Investment` for periods 120–400.

## Gaps in MacroModelling (Current State)
- `simulate_sep` uses a **heuristic**: it does not re-solve SEP from the current state each period.
- For RBC-II, Dynare solves a **fresh SEP** at each period with a warm start (`extended_path_core`), then takes the first-period solution.
- Complementarity constraints are expressed via Dynare’s MCP solver (`solve_algo=11`). MacroModelling uses OBC shocks, so mapping needs to be explicit.

## Implementation Plan (MacroModelling)
### 1) Add RBC-II model (MacroModelling version)
- Create a MacroModelling model matching `rbcii.mod` equations and parameters:
  - Variables: `Capital`, `Output`, `Labour`, `Consumption`, `Efficiency`, `efficiency`, `Investment`, `LagrangeMultiplier`.
  - Shock: `epsilon`.
  - Parameters: `beta, theta, tau, alpha, psi, delta, Effstar, rho, sigma`.
- **OBC mapping:** encode the complementarity constraint in MacroModelling’s OBC form (likely via `max/min` or explicit OBC shock constraints).
- Confirm steady state matches Dynare’s `steady_state_model`.

### 2) Create a true extended-path simulation loop
- Implement a new function (or extend `simulate_sep`) that mirrors Dynare’s `extended_path`:
  - Inputs: `samplesize`, `sep_horizon`, `sep_order`, `sep_nnodes`, `sep_sparse_tree`, `shocks`, `random_seed`, `warm_start=true`.
  - At each period `t`:
    1. Build `sep_deterministic_shocks` with the realized shock at `t=1`, zeros thereafter (Dynare uses the shock in the first period of the SPFM).
    2. Solve `solve!(algorithm=:stochastic_extended_path, ...)` from the current state.
    3. Use the **previous SEP solution** as `sep_initial_guess` for warm-start (like Dynare’s `initialguess = [previous_path, steady_state]`).
    4. Extract `y_{t+1}` (first-period solution) and advance the simulation.
- For `order=0`, use deterministic perfect-foresight solver (same approach, but no stochastic branching).

### 3) Shock generation alignment
- Dynare uses `set_dynare_seed(0)` and draws shocks from `N(0, I)` (then scaled by `sigma` inside the model equation).
- **MacroModelling choice:**
  - Option A: Use `Random.seed!(0)` and `randn` to generate shocks, then compare statistically.
  - Option B (exact match): Export shocks from Dynare (`exogenousvariables`) and feed them into MacroModelling as an explicit shock matrix.
- Recommendation: Start with Option B for exact replication and only then check if Option A yields a close match.

### 4) SEP configuration mapping
- Dynare settings → MacroModelling:
  - `options_.ep.stochastic.order = k` → `sep_order = k`
  - `options_.ep.stochastic.quadrature.nodes = 3` → `sep_nnodes = 3`
  - `options_.ep.stochastic.algo = 1` → `sep_sparse_tree = true`
  - `options_.ep.periods` → `sep_horizon`
- Confirm node/weight ordering (Dynare reorders GH nodes so the zero node is first). MacroModelling already does this for full-tree GH nodes; confirm for sparse tree.

### 5) Validation workflow
- Run MacroModelling simulations for `order = 0..5` with `samplesize = 400`.
- Extract `Investment` series and compare to Dynare’s `ts0..ts5`:
  - Primary check: overlay `Investment` for periods `120:400`.
  - Quantitative check: `max(abs(diff))`, `RMSE` by order.
- Persist comparison plots and metrics in a `ChatGPT/SEP_port_step_XX_*.md` doc.

## Open Questions
1. Should we require **exact shock sequence parity** with Dynare (export shocks), or is statistical similarity acceptable?
2. How should we encode the **complementarity condition** in MacroModelling’s OBC system for `Investment > 0.85*0.241741953339345`?
3. Is the target to match Dynare **exactly**, or to be **functionally equivalent** (same qualitative dynamics)?

## Next Actions (Pending your confirmation)
- Confirm the preferred shock alignment approach (exact vs statistical).
- Confirm how you want the ZLB/complementarity condition handled in MacroModelling.
- Proceed to implement the extended-path simulation loop and RBC-II model in MacroModelling.
