# SEP porting step 32: StatsPlotsExt extension implementation

## Goal
Provide the missing `StatsPlotsExt` extension module so MacroModelling can load plotting utilities without the extension load error.

## Change
Added the extension file:
- `ext/StatsPlotsExt.jl`

Contents:
- Imports `StatsPlots` and `MacroModelling`.
- Injects `StatsPlots` into `MacroModelling` with `Base.eval`.
- Includes `src/plotting.jl` into the `MacroModelling` module so the plotting methods attach to the existing stub functions.

## Rationale
MacroModelling declares `StatsPlotsExt` in `Project.toml` but the extension file was missing, so Julia raised an error at load time. This file completes the extension and makes plotting methods available when `StatsPlots` is installed.

## Next step
Re-run the RBCII comparison script to confirm the extension load error is gone.
