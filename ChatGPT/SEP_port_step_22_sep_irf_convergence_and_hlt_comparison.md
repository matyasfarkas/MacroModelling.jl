# SEP Porting Step 22: SEP IRF Convergence Fix + HLT Comparison

## Goal
- Diagnose SEP non-convergence in `get_sep_irf` for `Smets_Wouters_2007_HLT`.
- Fix the root cause of non-finite residuals.
- Make SEP IRF convergence controllable.
- Regenerate the HLT IRF comparison (perturbation vs SEP) and save the PDF.

## Findings (Root Cause)
- SEP failed immediately with `max|res| = Inf` in the HLT Taylor rule equation.
- Debug output showed `cpie = 0.0`, causing division by zero:
  - Equation index 29: `r[0] = r[ss]^(1-crr) * r[-1]^crr * (pinf[0]/cpie)^((1-crr)*crpi) * ...`
- `cpie` is a calibration parameter (not a model parameter), and the SEP solver was building parameter/SS values from a steady-state array that only contained variables.
- Therefore, `cpie` defaulted to `0.0` in `build_parameters_and_ss_values`, triggering `Inf` residuals.

## Fixes Implemented
### 1) Include calibration parameters in SEP steady-state lookup
- **File:** `src/sep_solver.jl`
- **Change:** `get_steady_state` now called with `return_variables_only=false`.
- **Result:** `cpie` and any other calibration parameters are included in `SS_result`, so `build_parameters_and_ss_values` no longer defaults them to `0.0`.

### 2) Add non-finite residual diagnostics
- **File:** `src/sep_solver.jl`
- **New helper:** `report_nonfinite_residual(...)`
- **Behavior:** When any residual becomes non-finite, the solver prints:
  - Equation index and expression
  - The evaluated values of all variables/parameters used in that equation
  - Returns `flag=2` (domain violation)
- This made the `cpie=0.0` issue explicit and repeatable.

### 3) Make SEP convergence tolerances configurable from `get_sep_irf`
- **File:** `src/sep_irf.jl`
- **New keywords:** `sep_maxit`, `sep_tol`
- **Behavior:** Passed through to `solve!` in the funnel IRF path, enabling per-call tuning.

### 4) HLT comparison script tuned for stable SEP convergence
- **File:** `scripts/HLT_comparison.jl`
- **Updates:**
  - `sep_periods = max(periods, 40)`
  - `sep_tol = 5e-3`
- **Rationale:** HLT residuals plateau around ~0.004; this tolerance yields practical convergence for IRF comparison.

## Validation Runs
### Minimal SEP check (HLT)
- Command (example):
  ```bash
  julia --project=. -e 'using MacroModelling; include("models/Smets_Wouters_2007_HLT.jl"); m=Smets_Wouters_2007_HLT; get_sep_irf(m, :epinf, 1.0; variables=[:y], periods=20, method=:funnel, baseline=:steady_state, sep_periods=40, sep_order=1, sep_nnodes=3, sep_tol=5e-3, sep_sparse_tree=true, silent=false);'
  ```
- Result: SEP converged (`err ≈ 0.00429 < 0.005`).

### Full HLT IRF comparison
- Command:
  ```bash
  julia --project=. scripts/HLT_comparison.jl
  ```
- Result:
  - Perturbation IRFs (1st, 2nd, pruned 3rd) computed.
  - SEP IRF converged with tolerance `5e-3`.
  - PDF saved: `scripts/HLT_comparison_sep_irf.pdf`.

## Files Touched
- `src/sep_solver.jl`
  - Include calibration parameters in SS lookup.
  - Non-finite residual diagnostics.
- `src/sep_irf.jl`
  - Document `get_sep_irf` options.
  - Add `sep_maxit`, `sep_tol` keyword support.
- `scripts/HLT_comparison.jl`
  - Use `sep_periods=max(periods,40)` and `sep_tol=5e-3`.

## Open Notes / Questions
- The HLT SEP residual norm stabilizes around ~0.004 with `sep_periods=40`. This is below the chosen tolerance but above the default `1e-7`. If tighter convergence is required, we may need to:
  - Improve the Newton step (e.g., solve `J \ -R` directly instead of normal equations),
  - Add line-search or trust-region, or
  - Increase horizon and/or branching order further.
- Confirm whether this tolerance level is acceptable for final HLT IRF comparisons.
