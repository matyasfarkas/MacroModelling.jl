# SEP port step 20: add RBC SEP IRF comparison to HLT_comparison

Goal
- Extend `scripts/HLT_comparison.jl` to compute the RBC SEP IRF (tt/ts funnel baseline) and compare against Dynare benchmarks, then save the IRF comparison plots as PDF.

Scope of changes
- Updated `scripts/HLT_comparison.jl` to append an RBC SEP IRF comparison section (Dynare CSV load, SEP tt/ts funnel baseline, IRF calculation, plotting, PDF output).
- Added a guard for the HLT plotting block to avoid a hard failure when `plot_irf` is unavailable in the current environment.

Implementation details (step-by-step)
1) Load Dynare benchmark CSVs
   - Read `tests/sep_validation/SEP/RBC_irf_pos3.csv` and `tests/sep_validation/SEP/RBC_irf_neg3.csv` via `CSV.read` and split into tt/ts blocks.
2) Configure RBC SEP settings
   - Set `maxorder = 10`, `dynare_horizon = 400`, `total_periods = 60` to mirror Dynare `options_.ep`.
   - Derive the shock scale from `z_epsilon` if present, otherwise use default 1.0 (Dynare shocks block convention).
3) Build SEP tt/ts paths with funnel baseline
   - `sep_irf_paths` solves the SEP tt path at order 10 with deterministic shocks.
   - `funnel_baseline` iterates order 10 -> 0 to build the ts path with deterministic continuation and `sep_initial_state` updates.
4) Convert to percent deviations and compute IRFs
   - Apply `pdss = 100 * (path / path[1] - 1)` to tt and ts paths.
   - IRF is `tt_pct - ts_pct` and compared to Dynare `tt - ts` from CSV.
5) Plot and save PDFs
   - Plot MacroModelling vs Dynare for each variable and save:
     - `scripts/HLT_comparison_rbc_sep_irf_pos3.pdf`
     - `scripts/HLT_comparison_rbc_sep_irf_neg3.pdf`

Notes on the HLT section
- `plot_irf` is currently unavailable in this environment because `StatsPlotsExt` is not installed (weak dependency extension). The HLT section is now wrapped in a `try/catch` so the RBC comparison still runs and produces PDFs.

Run log (most recent)
- Command: `julia --project=. scripts/HLT_comparison.jl`
- Result: RBC SEP IRF comparison succeeded; PDFs created.
- Warning: `plot_irf` unavailable; HLT plot skipped due to missing `StatsPlotsExt` extension.

Outputs
- `scripts/HLT_comparison_rbc_sep_irf_pos3.pdf`
- `scripts/HLT_comparison_rbc_sep_irf_neg3.pdf`

Questions / follow-ups
- Do you want me to fix the `StatsPlotsExt` setup so the HLT plot section works again in this environment?
- Should the RBC IRF PDFs be moved to a dedicated output folder (e.g., `plots/` or `ChatGPT/`)?
