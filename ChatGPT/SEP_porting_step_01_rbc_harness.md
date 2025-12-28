# SEP Porting Step 01: RBC Validation Harness Alignment

Goal
- Align the RBC validation harness with Dynare reference files and the Dynare workflow in `tests/sep_validation/SEP/rbc.mod`.
- Keep the solver behavior unchanged for now (deterministic SEP remains linearized), but make the test workflow and inputs consistent.

Scope
- File updated: `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`
- No solver code changes in this step.

Changes Applied
1) Use local Dynare reference CSVs
- Old paths were external: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SEP/...`
- New paths use repo-local files:
  - `tests/sep_validation/SEP/RBC_irf_pos3.csv`
  - `tests/sep_validation/SEP/RBC_irf_neg3.csv`

2) Use robust model include path
- `include("models/RBC_Dynare.jl")` replaced with
  `include(joinpath(@__DIR__, "..", "..", "models", "RBC_Dynare.jl"))`

3) Correct shock scaling for RBC (Dynare parity)
- Dynare uses `innovations(1) = ±3` and the model equation multiplies by `sigma`.
- Test now sets the deterministic shock to `±3` directly (no extra `* sigma` scaling).

4) Fix time indexing to match Dynare’s `ts.data(1,:)`
- Dynare’s `extended_path` outputs include the initial condition as row 1.
- Extraction now uses `t = t_idx - 1` and `layout.voff[t+1]` so row 1 is `t=0` (steady state), row 2 is the first response, matching Dynare.

5) Implement Dynare’s funnel baseline (`ts`)
- Added a `funnel_baseline(...)` helper that reproduces the decreasing-order sequence:
  - `order = maxorder`: 1-period solve with shock
  - `order in (maxorder-1 .. 1)`: 1-period solve with zero shock
  - `order = 0`: `T`-period solve with zero shocks
- The baseline path is assembled exactly as Dynare does, by appending the period-2 values for order>0 and all periods for order=0.

6) Add a reusable per-shock runner
- Added `run_shock_case(...)` to compute `tt`, build `ts`, print comparisons, and return percent-deviation paths.
- Both positive and negative shock cases now run in one script.

Behavior Notes
- The SEP solver still uses linearized residuals in deterministic mode. This means numerical parity with Dynare (which solves the nonlinear system) may not be achieved yet, but the validation harness now matches the Dynare workflow and data format.

Follow-up Needed
- Implement nonlinear residual evaluation in deterministic SEP (order=0 parity).
- Compare IRF = `pdss(tt) - pdss(ts)` against Dynare CSVs (both +3 and -3).

Files Touched
- `tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

