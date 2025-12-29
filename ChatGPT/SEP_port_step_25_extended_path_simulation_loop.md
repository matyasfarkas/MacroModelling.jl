# SEP Porting Step 25: Extended-Path Simulation Loop (SEP)

Date: 2025-01-05

## Goal
Implement a Dynare-style extended-path simulation loop that **re-solves SEP each period** using the current state and the realized shock, while keeping the existing fast (approximate) `simulate_sep` intact.

## Key implementation
**File:** `src/sep_simulation.jl`

### New function
`simulate_sep_extended_path(𝓂; ...)` added with:
- Per-period SEP solve using:
  - `sep_initial_state = y_t`
  - `sep_deterministic_shocks` with the realized shock in row 1, zeros thereafter
- Uses the SEP solution’s `t=1` group-1 state as `y_{t+1}`
- Warm starts via `solve!`’s built-in reuse of the previous SEP solution

### Shock handling
- Accepts a shock matrix (`nshocks × periods+burn_in`).
- If the matrix only includes **non-OBC shocks**, it is padded with zero rows for OBC shocks.
- If no shock matrix is provided, generates N(0,1) shocks and optionally scales by `z_<shock>` when `shock_scaling=:parameter`.

### Return value
A NamedTuple:
- `simulation`: KeyedArray (Variables × Time)
- `shocks`: shock matrix used
- `errorflag`: true if SEP failed in some period
- `failure_period`: first failure period (or `nothing`)

### Doc update
The existing `simulate_sep` docstring now explicitly states it is **approximate** (tree-mapping) and points to the extended-path version for accuracy.

## Files touched
- `src/sep_simulation.jl`
  - Added `simulate_sep_extended_path`
  - Documented `simulate_sep` as approximate
- `src/MacroModelling.jl`
  - Exported `simulate_sep_extended_path`

## Quick sanity check (local)
- `simulate_sep_extended_path` works on `RBC_Dynare` with short horizons.

## Notes
- This does not change existing `simulate_sep` behavior or tests.
- For exact Dynare shock parity, pass the shock sequence from Dynare (`oo_.exo_simul`); I will wire a comparison script once you confirm the preferred source format (CSV vs .mat).
