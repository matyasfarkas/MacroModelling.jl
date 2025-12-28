# SEP Porting Step 04: Nonlinear Residuals in Deterministic SEP

Goal
- Switch deterministic SEP (`solve_deterministic_path`) from linearized residuals to full nonlinear residuals and Jacobians, matching Dynare’s `extended_path` behavior for order=0.

Scope
- File updated: `src/sep_solver.jl`
- No changes to output structure (`sep_solution` unchanged).

What Changed
1) Added a dynamic-residual builder
- New helper `build_dynamic_residual_jacobian(𝓂)` builds:
  - residual function for `𝓂.dyn_equations`
  - Jacobian function `∂res/∂vars`
  - variable ordering `vars_raw` (future, present, past, shocks)
  - parameter/SS ordering `parameters_and_SS`
- Uses `Symbolics.build_function` to generate in-place residual/Jacobian evaluators.
- Includes calibration equation replacements, matching the existing symbolic pipeline.

2) Added parameter/SS value assembler
- New helper `build_parameters_and_ss_values(...)` maps:
  - model parameters from `parameters`
  - calibration equation parameters from `SS_result` (if available)
  - steady state variables (symbols ending with `₍ₛₛ₎`) from `yss`

3) Deterministic SEP now evaluates nonlinear residuals
- The linearized residual `∇₊*Δy_fwd + ∇₀*Δy_cur + ∇₋*Δy_lag + ∇ₑ*ε` is removed.
- For each period `t`, the solver now:
  - Builds `dyn_values` in the same variable order as `vars_raw`.
  - Calls `dyn_resid_func(resid_buffer, params_and_ss, dyn_values)`.
  - Calls `dyn_jac_func(jac_buffer, params_and_ss, dyn_values)`.
  - Assembles the stacked Jacobian from these nonlinear derivatives.

4) Deterministic SEP call signature updated
- `solve_deterministic_path` now receives `SS_result` (KeyedArray) to map calibration parameters.
- `sep_solve_mm!` now passes `SS_result` along when deterministic shocks are provided.

Notes
- This approach mirrors Dynare’s use of `dynamic_resid` and `dynamic_g1` by generating equivalent functions from `𝓂.dyn_equations`.
- For calibration equation parameters missing from `SS_result`, values default to `0.0` (conservative fallback).

Files Touched
- `src/sep_solver.jl`

Follow-up
- Run RBC validation to see if tt/ts/IRF move closer to Dynare once nonlinear residuals are in place.
- If needed, tighten parameter/SS mapping for models with heavy calibration-equation use.

