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
  80.9 percent nonlinear correction and 19.1 percent gate/inversion design.
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
