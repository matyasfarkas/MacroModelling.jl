# SEP porting step 33: StatsPlotsExt runtime loading fix

## Goal
Make the StatsPlots extension load cleanly without precompile errors or constant redefinition failures.

## Changes
1) Deferred extension wiring to runtime
- `ext/StatsPlotsExt.jl` now loads plotting code in `__init__()` to avoid precompile-time `eval` errors.

2) Converted constant assignments to function wrappers
File: `src/plotting.jl`
- `gr_backend = StatsPlots.gr` -> `gr_backend()` wrapper.
- `plotlyjs_backend = StatsPlots.plotlyjs` -> `plotlyjs_backend()` wrapper.
- `plot_IRF`, `plot_irfs`, `plot_fevd`, `plot_forecast_error_variance_decomposition` now use wrapper functions instead of assignments, avoiding redefinition of constants.

3) Fixed DocStringExtensions placeholders
- Replaced `$MODEL`, `$DATA`, etc. with the correct `...®` constants (`$MODEL®`, `$DATA®`, `$PARAMETERS®`, etc.) so docstring interpolation works when plotting code is loaded.

## Verification
```
julia --project=. -e 'using MacroModelling, StatsPlots; println("StatsPlotsExt loaded")'
```
- Result: `StatsPlotsExt loaded` with no extension load errors.

## Files touched
- `ext/StatsPlotsExt.jl`
- `src/plotting.jl`
