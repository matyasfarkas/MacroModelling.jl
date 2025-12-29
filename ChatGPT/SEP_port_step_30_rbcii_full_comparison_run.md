# SEP porting step 30: full 220-period RBCII comparison run

## Goal
Run the full 220-period SEP comparison (paper defaults) using the Dynare `.mat` dseries shocks and generate the paper-style and comparison PDFs.

## Run command
```
julia --project=. scripts/rbcii_sep_simulation_comparison.jl
```

## Runtime notes
- Total runtime: ~7.2 minutes for orders 0/1/2/5 (SEP horizon 200, 220 periods).
- MacroModelling reported warnings:
  - `StatsPlotsExt` extension missing (MacroModelling declares the extension but no source file exists). This did **not** block the run.
  - `Shock std parameter z_epsilon not found, using default 1.0` repeated during SEP solve. This is expected for RBCII because the model uses `sigma` and the script injects shocks directly (no `z_epsilon` parameter).

## Results
Comparison metrics (Investment, t>=1):
- order=0: RMSE = `1.0397130107014274e-5`, max abs diff = `2.4704917546192195e-5`, n = 221
- order=1: RMSE = `8.068514692263525e-8`, max abs diff = `2.431752662013231e-7`, n = 221
- order=2: RMSE = `1.04704756131707e-7`, max abs diff = `3.1910223055597875e-7`, n = 221
- order=5: RMSE = `2.143141233825034e-7`, max abs diff = `6.22173665504322e-7`, n = 221

## Outputs
- Paper-style plot: `scripts/rbcii_sep_simulation_paper_defaults.pdf`
- Order-by-order comparison: `scripts/rbcii_sep_simulation_comparison_orders.pdf`

## Follow-ups (optional)
- Implement the missing `StatsPlotsExt` extension file to silence the extension load error.
- If needed, store a text summary of the metrics alongside the PDFs for traceability.
