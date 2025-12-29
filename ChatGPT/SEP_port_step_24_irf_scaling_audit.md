# SEP Porting Step 24: IRF Scaling Audit (plot_irf vs SEP)

Date: 2025-01-05

## Goal
Confirm MacroModelling IRFs are on a comparable scale and fix SEP IRF shock scaling if it diverges from `get_irf`/`plot_irf` semantics.

## What I checked
- `src/plotting.jl` (`plot_irf`):
  - Uses `get_relevant_steady_states` and `SSS_delta`.
  - Default `initial_state` is `-SSS_delta` (so states are deviations from SSS).
  - `irf(..., level=zeros)` returns deviations, then `SSS_delta` is added to get deviations from SSS.
  - Plotting adds `reference_steady_state` (SSS for higher-order) to show levels, and optionally a %Δ dual axis.
  - **Conclusion**: `plot_irf` uses absolute deviations from the stochastic steady state (SSS), not percent deviations.

- `src/get_functions.jl` (`get_irf`):
  - Uses `SSS_delta` for `levels=false` (default) and returns deviations from SSS.
  - **Conclusion**: `get_irf` and `plot_irf` are aligned on absolute deviations from SSS.

- `src/sep_irf.jl` (`get_sep_irf` / `get_sep_irf_funnel`):
  - `baseline=:steady_state` returns deviations from `initial_state` (SSS if default).
  - **But** shock size was multiplied by `z_<shock>` (via `sep_irf_shock_std`).
  - In models like Smets_Wouters_2007_HLT, the standard deviation `z_<shock>` is *already in the equations*, so this **double-scales** the shock and makes SEP IRFs appear off-scale vs `get_irf`.

## Change made
**File:** `src/sep_irf.jl`

- Added a helper to control shock scaling explicitly:
  - `sep_irf_shock_scale(...; shock_scaling=:none|:parameter)`
- Added `shock_scaling` keyword to:
  - `get_sep_irf`
  - `get_sep_irf_funnel`
  - `get_sep_irf_tree`
- Default is now `shock_scaling=:none`, meaning:
  - `shock_size` is interpreted in **shock units**, consistent with `get_irf` and `plot_irf`.
  - Use `shock_scaling=:parameter` to multiply by `z_<shock>` (Dynare-style stderr scaling).

### Why this is correct
- MacroModelling’s perturbation IRFs apply `shock_size` directly to the shock variable.
- If the model already includes `z_<shock>` in the equation, multiplying again gives the wrong magnitude.
- The new option keeps compatibility for cases where `z_<shock>` is *not* in the equations.

## Files touched
- `src/sep_irf.jl`
  - Added `sep_irf_shock_scale`
  - Added `shock_scaling` keyword to SEP IRF functions
  - Updated docs to clarify shock scaling semantics

## Next verification (pending)
- Re-run `scripts/HLT_comparison.jl` with new default scaling to regenerate the SEP vs perturbation PDF.
- Confirm SEP impact responses are now in the same units as `get_irf` (absolute deviation from SSS).

## Open questions
- Should we also update `simulate_sep` to honor `shock_scaling` for stochastic burn-in (affects `method=:simulation`)?
  - I can do this in the upcoming simulation loop step if you agree.
