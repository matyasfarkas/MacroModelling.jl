# SEP port step 35: HLT SEP IRF baseline fix and comparison refresh

## Goal
- Resolve HLT SEP vs perturbation IRF scaling/sign mismatch by aligning SEP baseline to MacroModelling IRF conventions.
- Regenerate the HLT comparison PDF with consistent scaling and period alignment.

## Diagnosis
- `get_irf` returns deviations from an unshocked baseline path over periods 1..T.
- `get_sep_irf(...; baseline=:steady_state)` subtracts the initial steady state only, so the SEP path includes a shock-size-independent jump caused by the SEP path drifting away from the perturbation SSS.
- This produced an apparent scaling/sign discrepancy (especially for inflation) and made IRFs insensitive to `shock_size`.
- `get_sep_irf` includes period 0, while `get_irf` starts at period 1, so the SEP series needed to drop the initial element for alignment.

## Changes
- Added a SEP IRF baseline option that subtracts the zero-shock SEP path with the same SEP settings.
  - New `baseline=:zero_shock` in `get_sep_irf_funnel` and docs.
  - This isolates the shock effect and aligns the SEP IRFs with MacroModelling’s `get_irf` baseline.
- Updated the HLT comparison script to use the new baseline and correct scaling:
  - `method=:funnel`, `baseline=:zero_shock`, `shock_scaling=:none`.
  - Plot inflation using `:pinfobs` (percent points) to avoid sign/scale confusion.
  - Drop SEP period 0 so the SEP series aligns with `get_irf` periods 1..T.
  - Removed the unused pruned third-order plot.

## Files changed
- `src/sep_irf.jl`
  - Added `baseline=:zero_shock` behavior and updated docstrings.
- `scripts/HLT_comparison.jl`
  - Switched to `:pinfobs`, `baseline=:zero_shock`, `shock_scaling=:none`.
  - Aligned SEP periods with perturbation periods (drop t=0).

## Verification
- Ran `julia --project=. scripts/HLT_comparison.jl`.
- SEP solves converged with `sep_tol=5e-3` (reported final errors ~4e-3 and ~3e-3).
- Output PDF: `scripts/HLT_comparison_sep_irf.pdf`.

## Result
- SEP IRFs now scale with the shock size and match perturbation IRFs for inflation (via `pinfobs`) and other variables.
- The sign mismatch disappears once the SEP baseline is defined as shocked minus unshocked SEP path.

## Questions / follow-ups
- Do you want `baseline=:zero_shock` to become the default for `get_sep_irf` when `method=:funnel`?
- Should I tighten SEP convergence targets beyond `sep_tol=5e-3` for HLT, or keep this tolerance for speed?
- Confirm whether you prefer plotting `:pinf` or `:pinfobs` in the comparison figure.
