# Scripts overview

This folder contains runnable entry points. For the curated replication
workflow, start with `REPLICATION.md` at the repository root and
`scripts/replication_smoke.sh`.

## HLT SEP surrogate + regime switching
- Shock convention: HLT and Galí model files multiply unit structural shocks by
  volatility parameters inside the equations. New HLT/Galí SEP runs should use
  `shock_scaling=:none` and the default `sep_shock_scaling=:none`; use
  `:parameter` only to reproduce historical pre-2026-06-17 artifacts.
- `scripts/hlt_sep_surrogate_dataset_generate.jl` — build SEP training data (supports stable-prefix sampling).
- `scripts/hlt_sep_surrogate_train.jl` — train surrogate (ROM1 residual + observables).
- `scripts/hlt_sep_surrogate_synthetic_data.jl` — generate synthetic SEP data with a high‑volatility window.
- `scripts/run_hlt_unitshock_rebuild.sh` — corrected unit-shock full-scale wrapper: base/ZLB dataset generation, combined obs-only ROM1-residual surrogate training, matched full gate+NN HMC, and matched linear+gate HMC.
- `scripts/run_hlt_unitshock_mvp.sh` — smaller corrected unit-shock MVP wrapper used for the current dynamic HLT HMC smoke evidence.
- `scripts/hlt_regime_switching_illustration.jl` — ROM1 vs ROM1+delta error/IRF plots.
- `scripts/hlt_sep_surrogate_gate_calibration.jl` — gate calibration using ROM vs surrogate errors.
- `scripts/hlt_sep_surrogate_synthetic_estimation.jl` — synthetic estimation (regime switching).
- `scripts/hlt_sep_surrogate_chain_report.jl` — posterior report (LaTeX).
- `scripts/hlt_sep_surrogate_validate_hlt3.jl` — reproducible HLT 3-parameter smoke/benchmark harness (writes run manifest + summary).
- `scripts/hlt_sep_surrogate_fom_benchmark.jl` — direct-SEP FOM benchmark utility for chain/synthetic outputs (records failures if direct FOM eval is unavailable).
- `scripts/hlt_direct_sep_surrogate_posterior_validation.jl` — bounded HLT direct-SEP/MH smoke against the surrogate validation bundle.
- `scripts/hlt_reduced_bridge_validation.jl` — Econometrica validation bridge for a reduced SW07-HLT investment block; the smoke stage now runs finite ROM1/inversion and direct SEP/inversion checks before any heavier bridge grid or HMC run.
- `scripts/decompose_ll_gap.jl` — extended-sample LL decomposition and linear+gate ablation.
- `scripts/mode_sensitivity_report.jl` — chain-level mode sensitivity report.
- `scripts/oos_forecast_evaluate.jl` and `scripts/oos_gate_recalibration.jl` — quiet-sample OOS and gate recalibration diagnostics.
- `scripts/sep_sensitivity_study.jl` — SEP tolerance/node sensitivity study; the checked paper artifact is the bounded 40-period pilot, and the completed full 265-period rerun is documented in `docs/review/SEP_265_RUN_STATUS.md`.
- `scripts/kimball_curvature_sensitivity.jl` — fixed-calibration high-Kimball stress diagnostic for the HLT/JME neighborhood.
- `scripts/hlt_counterfactual_decomposition.jl` — mechanism ablation engine for investment, pricing, policy, and local +/-10 percent posterior-region sensitivity designs.
- `scripts/run_hlt_posterior_region_ablation_queued.sh` — queued wrapper for the 40-draw posterior-region local ablation used in the paper provenance ledger.
- `scripts/hlt_posterior_mean_curvature_surface.jl` — SEP-minus-ROM1 forecast-error surfaces for visualizing local curvature around a chosen HLT parameter center; supports multiplier grids and density-scaled standard-deviation grids. The manuscript uses locked 13-by-13 standard-deviation exports for the SW07--HLT baseline and the HLT high-Kimball stress center.

## SEP / Dynare comparisons
- `scripts/HLT_comparison.jl` — HLT IRF comparison (perturbation vs SEP).
- `scripts/RBCII_comparison.jl` — RBC-II SEP comparison to Dynare.
- `scripts/rbcii_sep_simulation_comparison.jl` — SEP extended‑path simulation vs Dynare.
- `scripts/rbcii_filter_diagnostic.jl` — KF vs inversion filter diagnostic on RBC‑II synthetic data.
- `scripts/rbcii_surrogate_filter_diagnostic.jl` — RBC‑II surrogate diagnostic (KF vs inversion vs surrogate inversion).

## Galí OBC validation
- `scripts/gali_obc_stochastic_comparison_plot.jl` — corrected adverse `eps_z` same-shock OBC versus linear stress path.
- `scripts/gali_obc_actual_floor_residual_grid_validation.jl` — one-parameter known-shock actual-floor residual-grid validation for `std_z`.
- `scripts/gali_obc_actual_floor_inversion_grid_validation.jl` — one-parameter ROM1-inversion actual-floor residual-grid validation for `std_z`, with optional matched one-parameter HMC smoke via `--hmc-draws`.
- `scripts/gali_obc_actual_floor_twoparam_inversion_grid_validation.jl` — two-parameter ROM1-inversion actual-floor residual-grid and matched HMC validation for `std_z` plus a configurable second shock scale; the clean current design uses `std_a` and post-ELB identification blocks.
- `scripts/gali_validation_package.jl` — consolidated Galí hard-ELB validation package; audits the curated stress-path, residual-grid, inversion-grid, HMC, `std_nu` identification-probe, and direct-SEP smoke artifacts and writes a package-level report/manifest.

## Diagnostics
- `scripts/diagnostics/HLT_sep_residual_audit.jl` — residual hotspot audit for SEP convergence.
- `scripts/diagnostics/` — other validation utilities (Y indexing, steady state checks).

## Misc utilities
- `scripts/compute_irf_from_csv.jl`, `scripts/extract_sep_values.jl`, `scripts/find_sep_variable_mapping.jl` — analysis helpers.
- `scripts/export_*` — Dynare export utilities.

For the HLT surrogate pipeline, see `scripts/hlt_surrogate/README.md`.
