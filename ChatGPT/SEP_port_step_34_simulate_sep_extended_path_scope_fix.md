# SEP porting step 34: qualify simulate_sep_extended_path in RBCII script

## Goal
Resolve `UndefVarError: simulate_sep_extended_path not defined in Main` when running the RBCII comparison script from a REPL session without `using MacroModelling` in the active scope.

## Change
File: `scripts/rbcii_sep_simulation_comparison.jl`
- Replaced the unqualified call with a module-qualified one:
  - `simulate_sep_extended_path(...)` -> `MacroModelling.simulate_sep_extended_path(...)`

## Rationale
When the script is included but the `using MacroModelling` statement is not executed in the current REPL scope (or `main()` is invoked manually later), the unqualified name can be missing. Qualifying the call makes it robust in any scope.

## Verification
Re-run the script:
```
julia --project=. scripts/rbcii_sep_simulation_comparison.jl
```
