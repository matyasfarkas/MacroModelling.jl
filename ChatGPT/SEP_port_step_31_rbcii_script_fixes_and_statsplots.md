# SEP porting step 31: RBCII comparison script fixes + StatsPlots dependency

## Goal
Stabilize the RBCII comparison script execution and ensure plotting dependencies are present.

## Changes made
1) Added StatsPlots as a direct dependency
- `Project.toml`: added `StatsPlots` to `[deps]` so plotting works in scripts.
- `Manifest.toml`: updated to include `StatsPlots` and its dependencies (`Plots`, `GR`, etc.).

2) Fixed Julia parsing issues
- Removed escaped quotes inside string interpolation (e.g. `join(ds.names, \", \")` -> `join(ds.names, ", ")`, `join(orders, \",\")` -> `join(orders, ",")`).

3) Eliminated global scope warnings
- Wrapped the script logic in `function main()` and called `main()` at the end.
- This resolves ambiguity around `shocks_ref` and improves performance.

4) Added missing import for statistics
- Added `using Statistics` to provide `mean` for RMSE calculation.

## Known issue (non-blocking)
- MacroModelling declares `StatsPlotsExt` but the extension source file is missing, so Julia prints an extension load error each run. This does not block the script or plots, but it is noisy.

## Files touched
- `Project.toml`
- `Manifest.toml`
- `scripts/rbcii_sep_simulation_comparison.jl`

## Next step (optional)
Implement the missing `StatsPlotsExt` extension file to silence the extension load error if desired.
