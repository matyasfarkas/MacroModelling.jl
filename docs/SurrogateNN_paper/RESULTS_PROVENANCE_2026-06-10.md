# Paper Result Provenance

**Paper:** Investment Adjustment Costs and the Nonlinear Posterior of Smets-Wouters
**Date:** 2026-06-10

This note records the local artifacts behind the current paper draft. Heavy
artifacts under `.local_artifacts/` are intentionally ignored by Git. The
committed replication package records the scripts, run ids, and headline
numbers needed to regenerate and audit the results.

## Core Likelihood And Posterior Results

- Likelihood decomposition:
  `.local_artifacts/ll_decomposition/LL_DECOMPOSITION_SUMMARY.md`
- Headline arithmetic: nonlinear surrogate benchmark improves the objective by
  35.92 nats relative to the linear benchmark; the symmetric decomposition is
  80.9 percent criterion change and 19.1 percent parameter relocation.
- Main chains:
  `.local_artifacts/hlt_18param_realdata/`
- Mode-sensitivity reporting:
  `.local_artifacts/mode_sensitivity/`

## HLT Bridge Validation

- Supported four-parameter bridge sweep:
  `.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid10_full_sweep_20260527/`
- Support result: 10,000 direct-SEP cells solved on the supported grid.
- Posterior-grid comparison:
  `.local_artifacts/hlt_reduced_bridge_validation/posterior_grid_compare_investment4p_supported_grid10_full_sweep_20260527/SUMMARY.md`
- Prediction result: raw ROM1 residual prediction RMSE is 0.143574; surrogate
  residual prediction RMSE is 0.000749395.
- Log-posterior surface result: ROM1 surface RMSE is 88.0719; surrogate
  surface RMSE is 0.0534526.

## Corrected Unit-Shock HLT Dynamic MVP

- Artifact root:
  `.local_artifacts/hlt_18param_validation_unitshock_mvp_20260620/`
- Summary:
  `.local_artifacts/hlt_18param_validation_unitshock_mvp_20260620/HMC_SMOKE_COMPARISON_SUMMARY.md`
- Real-data payload:
  `.local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p_unitshock_nonobc_20260620.jls`
- Surrogate bundle:
  `.local_artifacts/hlt_18param_validation_unitshock_mvp_20260620/hlt_sep_surrogate_trained_with_zlb_unitshock_mvp.jls`
- Dataset status: base non-OBC dataset has 1,664 samples; ZLB-binding dataset
  has 1,536 samples; combined MVP dataset has 3,200 samples.
- Surrogate validation RMSE by observable:
  `[0.0973, 0.1196, 0.3601, 0.0889, 0.0450, 0.0372, 0.0331]`, versus ROM1
  baseline RMSE `[0.8413, 1.8637, 4.8532, 1.1507, 0.4952, 0.2940, 0.4477]`.
- HMC status: matched 100-warmup/100-draw full gate+NN and linear+gate chains
  both complete with zero post-warmup divergences. Full gate+NN mean LL is
  `-1382.805`; linear+gate mean LL is `-1382.877`.
- Fixed-theta cross-evaluation: at the full gate+NN posterior mean, full
  gate+NN LL is `-1382.805307` and linear+gate LL is `-1382.799759`; at the
  linear+gate posterior mean, full gate+NN LL is `-1382.882119` and
  linear+gate LL is `-1382.876532`.
- Interpretation: the corrected unit-shock MVP validates end-to-end dynamic
  real-data execution of the gate/inversion/HMC stack. It does not support a
  large incremental real-data likelihood gain from the NN residual correction
  in this MVP; full gate+NN and linear+gate are effectively indistinguishable
  at the evaluated posterior means.

## Gali Hard-ELB Validation Package

- Package report:
  `.local_artifacts/gali_validation_package/gali_validation_package_20260521/VALIDATION_PACKAGE_REPORT.md`
- Interpretation: the maintained small-model pipeline passes the hard-ELB
  residual, inversion, grid, and two-parameter HMC checks used as a transparent
  validation bridge. It is not presented as a completed three-parameter
  direct-SEP HMC certificate.

## SEP Sensitivity

- Bounded paper pilot:
  `.local_artifacts/sep_sensitivity/SEP_SENSITIVITY_SUMMARY.md`
- Full 265-period status:
  `docs/review/SEP_265_RUN_STATUS.md`
- Interpretation: the full-window rerun converges in all cells, with mild
  quadrature-node sensitivity. It supports the tolerance-stability caveat but
  is not an unconditional clean pass at every quadrature setting.

## Nonlinearity-Source Evidence

### Baseline Equation-Block Decomposition

- Script family:
  `scripts/hlt_counterfactual_decomposition.jl`,
  `scripts/kimball_curvature_sensitivity.jl`
- Paper table: investment/capital block accounts for 68.9 percent of the
  SEP--ROM1 equation-block gap; price Phillips curve accounts for 0.2 percent.

### HLT High-Kimball Neighborhood

- Artifact:
  `.local_artifacts/kimball_curvature_sensitivity/hlt_reported_full_mapped_maxit400_20260606/KIMBALL_CURVATURE_SENSITIVITY_SUMMARY.md`
- Design: fixed-calibration stress check mapping the Harding-Linde-Trabandt
  high-curvature neighborhood into the maintained code where direct analogues
  exist.
- Headline result: investment/capital share is 76.5 percent; price Phillips
  share is 2.0 percent.
- Interpretation: nonlinear Phillips-curve curvature is available at selected
  high-curvature calibrations, but the investment/capital block remains the
  dominant local residual in this mapped neighborhood.

### Posterior-Region Local Ablation

- Artifact:
  `.local_artifacts/counterfactual_decomposition/hlt_posterior_region_local10_40draws_scales012505_queued_20260609/COUNTERFACTUAL_ABLATION_SUMMARY.md`
- Wrapper:
  `scripts/run_hlt_posterior_region_ablation_queued.sh`
- Design: local +/-10 percent perturbations of `csadjcost`, `czcap`, `curvp`,
  and `curvw`; 40 representative posterior draws from
  `.local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_pooled_8000_20260609.jls`;
  shock scales 0.1, 0.25, and 0.5; four simulated periods after two burn-in
  periods.
- Solver status: 100 percent SEP path coverage for every variant and shock
  scale; all accepted on the primary attempt.
- Headline dominance:
  - Scale 0.1: Kimball largest in 2/40 draws (5.0 percent); real-side largest
    in 38/40 draws (95.0 percent).
  - Scale 0.25: Kimball largest in 0/40 draws; real-side largest in 40/40
    draws.
  - Scale 0.5: Kimball largest in 0/40 draws; real-side largest in 40/40
    draws.
- Median absolute local log-MSE elasticities:
  - Scale 0.1: `csadjcost` 0.306, `czcap` 0.209, `curvp` 0.013, `curvw` 0.016.
  - Scale 0.25: `csadjcost` 0.308, `czcap` 0.235, `curvp` 0.011, `curvw` 0.013.
  - Scale 0.5: `csadjcost` 0.302, `czcap` 0.235, `curvp` 0.011, `curvw` 0.019.
- Interpretation: at the estimated posterior region, the largest local
  nonlinearity is real-side investment curvature, not nonlinear
  Phillips-curve curvature.

### Density-Scaled Curvature Surfaces

- Artifacts:
  - `.local_artifacts/hlt_posterior_mean_curvature_surface/sw07_hlt_baseline_surface_13x13_std1_shock01_1path_20260611/CURVATURE_SURFACE_SUMMARY.md`
  - `.local_artifacts/hlt_posterior_mean_curvature_surface/sw07_hlt_baseline_surface_13x13_std2_bounded_shock01_1path_20260611/CURVATURE_SURFACE_SUMMARY.md`
  - `.local_artifacts/hlt_posterior_mean_curvature_surface/hlt_highkimball_surface_13x13_std1_shock01_1path_20260611/CURVATURE_SURFACE_SUMMARY.md`
  - `.local_artifacts/hlt_posterior_mean_curvature_surface/hlt_highkimball_surface_13x13_std2_shock01_1path_20260611/CURVATURE_SURFACE_SUMMARY.md`
- Script:
  `scripts/hlt_posterior_mean_curvature_surface.jl`
- Manuscript figures:
  - `docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_sw07_hlt_baseline_13x13_std1_shock01.png`
  - `docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_sw07_hlt_baseline_13x13_std2_shock01.png`
  - `docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_highkimball_13x13_std1_shock01.png`
  - `docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_highkimball_13x13_std2_shock01.png`
- Regeneration command for figures only:
  `julia --project=. scripts/hlt_posterior_mean_curvature_surface.jl --plot-only=true --run-id=<run_id> --paper-figure-stem=<figure_stem>`
- Metric: RMSE of one-step observable forecast errors `SEP - ROM1`, evaluated
  at the same SEP state, same shocks, and same parameters.
- Design: 13-by-13 standard-deviation grids around either the maintained
  SW07--HLT baseline calibration or the HLT high-Kimball stress center; common
  shock seeds across all grid cells; shock scale 0.1; one shock path per cell;
  two simulated periods after one burn-in period.
- Range choice: axes are in local standard deviations. Estimated parameters use
  posterior/prior scales where available. Fixed calibration parameters use
  documented local calibration-scale proxies. The baseline `curvp` +/-2 grid is
  truncated at the documented density support `[5,20]`; the high-Kimball
  `curvp` +/-2 grid is bounded to `[2,150]`.
- Solver status: all high-Kimball cells are accepted. The baseline Kimball
  surface has one isolated ROM1-construction failure; it is retained in the raw
  CSV and filled only for plotting.
- Headline result:
  - SW07--HLT baseline, +/-1 s.d.: real-side range 11.4 vs. Kimball range 1.4,
    in RMSE times 1,000.
  - SW07--HLT baseline, +/-2 s.d.: real-side range 23.5 vs. Kimball range 2.6.
  - HLT high-Kimball, +/-1 s.d.: real-side range 37.0 vs. Kimball range 54.7.
  - HLT high-Kimball, +/-2 s.d.: real-side range 53.9 vs. Kimball range 210.6.
- Interpretation: around the baseline/posterior region, real-side investment
  curvature dominates. At the intentionally high-Kimball stress center,
  price/wage Kimball curvature can dominate the local short-run forecast-error
  surface. This is a caveat on global claims and a useful validation that the
  diagnostic can detect Kimball nonlinearity when the calibration puts the model
  there.

## Compilation And Smoke Tests

The current replication entry point is `REPLICATION.md`. Fast checks are run
with:

```bash
RUN_PAPER=0 bash scripts/replication_smoke.sh
bash docs/SurrogateNN_paper/compile.sh
```

The full empirical package is intentionally staged. Direct full posterior
validation remains expensive and should be reported as conditional on the
documented support, bridge validation, high-likelihood posterior basin, and
gate diagnostics.
