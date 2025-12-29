# SEP porting step 29: MAT-based importer and RBCII paper-style plots

## Goal
Use Dynare `.mat` dseries data to recover the shock sequence, run MacroModelling SEP simulations with matching shocks, and generate paper-style Investment plots and comparison figures for the RBCII model.

## Changes made
### 1) Reworked the RBCII validation script to read Dynare `.mat`
File: `scripts/rbcii_sep_simulation_comparison.jl`
- Replaced CSV ingestion with MAT.jl dseries loader (`DATA__`, `NAMES__`).
- Added `DynareDSeries` helper with:
  - `load_dynare_dseries(path)`
  - `get_series(ds, name)`
- Implemented `implied_shocks_from_efficiency(eff; rho, sigma)`:
  - Recovers shocks via `epsilon_t = (eff_t - rho*eff_{t-1})/sigma`.
  - Uses `t=0` as the first observation in the Dynare series.

### 2) Paper-default configuration
- Orders: `0, 1, 2, 5`.
- ZLB: `0.85` (from the model parameter, unchanged).
- Shock size: `sigma = 0.007`.
- SEP horizon: `200` (matches `SEP_PERIODS` default in Dynare).
- SEP nodes: `3`.
- SEP tolerance: `1e-5`.
- Sparse tree: `true`.
- Periods: `220` (paper figure uses 120:220).
- Hybrid: `0` only (MacroModelling does not implement Dynare’s hybrid correction).

### 3) Validation outputs
The script now produces two PDFs:
- `scripts/rbcii_sep_simulation_paper_defaults.pdf`
  - Two stacked panels:
    - Dynare Investment paths for orders 0/1/2/5 (paper-style color gradient).
    - MacroModelling Investment paths with the same shock sequence.
- `scripts/rbcii_sep_simulation_comparison_orders.pdf`
  - 2x2 grid of Dynare vs MacroModelling Investment comparisons by order.

The script also prints RMSE and max absolute differences for each order (excluding t=0).

## Notes / constraints
- Dynare does not ship order-3 `.mat` files; the paper uses orders 0/1/2/5 for Euler-error figures.
- Hybrid corrections (`hybrid=4`) are not implemented in MacroModelling; the comparison is limited to hybrid=0 outputs.
- The `.mat` files in `tests/sep_validation/sep_simulation_data/accuracy-sc/` contain 10,000 periods; the script uses the first 220 to match the paper figure window.

## Verification
Not executed here because SEP extended-path simulation is computationally heavy (220 periods × horizon 200 × orders 0/1/2/5). Run the script when ready:
```
julia --project=. scripts/rbcii_sep_simulation_comparison.jl
```

## Next step
Run the script and inspect the PDFs for order-by-order alignment; if discrepancies persist, investigate shock alignment, initial state, or SEP horizon settings.
