# SEP Porting Step 11: Dynare SEP Alignment Review (RBC)

## Scope
Review Dynare SEP (fishbone/sparse tree) behavior for RBC to identify mismatches with MacroModelling SEP.
Sources:
- Dynare EP implementation: `/Applications/Dynare/7-2025-12-17-2028-arm64/matlab/ep/` (`extended_path.m`, `extended_path_core.m`, `solve_stochastic_perfect_foresight_model_1.m`, `ep_problem_1.m`, `setup_integration_nodes.m`, `setup_stochastic_perfect_foresight_model_solver.m`)
- Dynare RBC script: `tests/sep_validation/SEP/rbc.mod`

## Dynare behavior (observed)
1. **Shock timing in SEP residuals (ep_problem_1)**
   - For period `i=1` (current period), the shock used in residuals is `innovation(2,:)` from `exo_simul` (deterministic shock sequence).
   - For periods `i>1` within the stochastic horizon, trunk world integrates over nodes by setting `innovation(i+1,:) = nodes(k,:)` *per node*.
   - New side-branch worlds at period `i` set the period-`i` shock to a fixed node; existing side branches use `innovation(i+1,:)` from `exo_simul` (typically zero).

2. **Initial conditions are fixed**
   - `pfm.y0 = endo_simul(:,1)` and is treated as given; SEP solves for future states only.

3. **Horizon length**
   - RBC script sets `options_.ep.periods = 400` and uses `extended_path` with `sample_size = 80`.
   - Each period’s SEP solve uses horizon 400; output is the first step of each solve.
   - Funnel baseline (`ts`) uses the same horizon but only takes one step for each order (except order=0, which simulates 80 periods).

4. **Shock variance / scaling**
   - `shocks; var epsilon = 1; end;` sets `M_.Sigma_e = 1` (std dev = 1 for epsilon).
   - Model equation scales shock: `efficiency = rho*efficiency(-1) + sigma*epsilon;` so shock scale is embedded in the model.
   - Quadrature nodes are based on `M_.Sigma_e` (std 1), not `sigma`.

5. **IRF definition**
   - IRFs are percent deviations from period-1 value: `(data(1:T)./data(1)-1)*100`.
   - `RBC_irf_pos3.csv` stores `[ts, tt]` in this PDSS format.

## Current MacroModelling SEP mismatches (likely)
1. **Initial state not fixed** in stochastic SEP solver (y0 updated by Newton step).
2. **Shock timing**: current SEP uses node shocks for trunk residuals at all branching periods; Dynare uses deterministic shock at period 1 and nodes only for i>1.
3. **Horizon mismatch**: current RBC validation uses `sep_periods = 60`, but Dynare uses 400 (and only reports first 60).
4. **Shock variance default**: MacroModelling uses `z_<shock>` parameter; when missing, defaults to 0.01 (not Dynare’s `1`).

## Open questions
- Should MacroModelling treat shocks as unit-variance (Dynare-style) by default when `z_<shock>` is missing, or should it attempt a parameter fallback (e.g., `sigma`) without double-scaling?
- Should deterministic shocks always override nodes at period 1 only (Dynare behavior) or be applied to all periods with nodes added (current behavior)?

## Next steps (planned)
- Fix initial-condition handling, shock timing in branching nodes, and horizon alignment.
- Decide and implement shock variance mapping consistent with Dynare (avoid double-scaling).
- Rerun RBC validation after each fix.
