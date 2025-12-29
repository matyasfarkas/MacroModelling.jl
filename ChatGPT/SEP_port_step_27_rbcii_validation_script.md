# SEP Porting Step 27: RBC-II Dynare Comparison Script

Date: 2025-01-05

## Goal
Provide a reproducible script to compare MacroModelling’s SEP extended-path simulation against Dynare outputs for RBC-II, including a PDF plot and basic error metrics.

## Script added
**File:** `scripts/rbcii_sep_simulation_comparison.jl`

### What it does
- Loads `RBCII_Dynare` model.
- Reads **Dynare shock sequence** from CSV (`oo_.exo_simul` export).
- Reads **Dynare series** (e.g., `Investment`) from CSV (`ts*.data` export).
- Runs `simulate_sep_extended_path` with:
  - matching number of periods
  - `sep_order`, `sep_periods`, `sep_nnodes` configurable
  - `sep_tol` defaulted to `1e-5`
- Compares `Investment` series:
  - RMSE
  - Max absolute difference
- Generates a PDF plot with:
  - Dynare vs MacroModelling series
  - steady-state and floor lines

### Required inputs
Update these at the top of the script:
- `dynare_shocks_path`
- `dynare_series_path`

And pick the column:
- `dynare_series_name` (if header exists)
- or `dynare_series_index`

### Output
- PDF: `scripts/rbcii_sep_simulation_comparison_order_<order>.pdf`

## Notes
- The script assumes Dynare CSVs use **rows = periods**. It transposes shocks to match MacroModelling’s `nshocks × periods` layout.
- If Dynare CSV exports are not available, we can optionally add a MAT.jl-based loader for `.mat` files (pending your preference).
