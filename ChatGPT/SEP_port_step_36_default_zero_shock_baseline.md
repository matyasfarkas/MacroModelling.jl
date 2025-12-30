# SEP port step 36: Default SEP IRF baseline set to zero_shock

## Goal
- Make the SEP IRF default baseline use `:zero_shock` for funnel-based IRFs.

## Change
- `get_sep_irf_funnel` default `baseline` updated to `:zero_shock`.
- `get_sep_irf` default `baseline` updated to `:zero_shock` in both doc signature and function signature.

## Files changed
- `src/sep_irf.jl`

## Notes
- This makes SEP IRFs comparable to `get_irf` out of the box by subtracting the unshocked SEP path.
- No tolerance or solver settings were changed.
