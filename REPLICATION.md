# Replication Guide

This document is the canonical entry point for reproducing the current paper,
Investment Adjustment Costs and the Nonlinear Posterior of Smets-Wouters.

The repository contains both research history and the replication package. Use
the curated commands below unless you are intentionally extending the project.

## Package Scope

The replication package has three tiers:

1. Fast verification: API tests, validation harness smoke tests, and paper
   compilation. This should run on a laptop.
2. Cached-paper replication: rebuild the manuscript from checked-in figure and
   table snapshots plus the local Julia environment.
3. Heavy empirical replication: regenerate SEP datasets, train the surrogate,
   rerun HMC chains, and regenerate local artifacts. This is a multi-hour to
   multi-day workflow depending on hardware and chain lengths.

The repository intentionally does not version `.local_artifacts/`, large `.jls`
chains, or timestamped SEP datasets. Heavy scripts write those files locally.
Small TeX table snapshots needed for paper compilation live under
`docs/SurrogateNN_paper/generated/`.

## Environment

Recommended Julia version: 1.12.x. The top-level `Project.toml` and
`Manifest.toml` define the replication environment.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

If you need a fresh artifact cache, start from a clean shell in the repository
root and let scripts write into `.local_artifacts/`.

## Fast Verification

Run the curated smoke suite:

```bash
bash scripts/replication_smoke.sh
```

The smoke suite runs:

- `test/test_regime_switching_api.jl`
- `test/test_hlt_validation_harness.jl`
- `test/test_hlt_acceptance_smoke.jl`
- `test/test_sep_inversion_filter_likelihood.jl`
- `docs/SurrogateNN_paper/compile.sh`

To skip the paper build:

```bash
RUN_PAPER=0 bash scripts/replication_smoke.sh
```

## Build The Paper

```bash
bash docs/SurrogateNN_paper/compile.sh
```

The build uses checked-in figures under `docs/SurrogateNN_paper/figures/` and
checked-in table snapshots under `docs/SurrogateNN_paper/generated/`. Auxiliary
LaTeX outputs and the generated PDF are ignored by Git.

## Main Empirical Workflows

The commands below are the reproducible entry points for regenerating the main
classes of results. They are intentionally explicit about output paths so that
artifacts do not overwrite the checked-in package state.

### 1. Generate SEP Surrogate Dataset

Small smoke-scale example:

```bash
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --param-set=phase1_18params_narrow \
  --theta-samples=4 \
  --samples-per-theta=4 \
  --sep-horizon=4 \
  --sep-order=1 \
  --sep-nnodes=3 \
  --output-dir=.local_artifacts/replication_smoke/dataset
```

Production runs should increase `--theta-samples`, `--samples-per-theta`, and
`--sep-horizon` according to the paper design. The generated dataset is written
as:

```text
.local_artifacts/.../hlt_sep_surrogate_dataset.jls
```

### 2. Train The Surrogate

```bash
julia --project=. scripts/hlt_sep_surrogate_train.jl \
  .local_artifacts/replication_smoke/dataset/hlt_sep_surrogate_dataset.jls \
  --rom-residual=1 \
  --obs-only \
  --out=.local_artifacts/replication_smoke/hlt_sep_surrogate_trained.jls
```

### 3. Run Linear HMC Baseline

```bash
julia --project=. scripts/run_linear_hmc_advancedhmc.jl \
  --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls \
  --out=.local_artifacts/replication/linear_hmc.jls \
  --samples=500 \
  --adapt=200 \
  --seed=42
```

### 4. Run Surrogate HMC

```bash
julia --project=. scripts/run_surrogate_hmc_advancedhmc.jl \
  --surrogate=.local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls \
  --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls \
  --gate-calibration=.local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p.jls \
  --out=.local_artifacts/replication/surrogate_hmc.jls \
  --samples=500 \
  --adapt=200 \
  --seed=42
```

For the linear+gate ablation, add:

```bash
--disable-nn-correction=true
```

### 5. Likelihood Decomposition And Gate Ablation

```bash
julia --project=. scripts/decompose_ll_gap.jl \
  --out-dir=.local_artifacts/ll_decomposition
```

### 6. Mode Sensitivity

```bash
julia --project=. scripts/mode_sensitivity_report.jl \
  --out-dir=.local_artifacts/mode_sensitivity
```

### 7. Quiet-Sample OOS Diagnostics

```bash
julia --project=. scripts/oos_forecast_evaluate.jl \
  --tag=quiet_1994 \
  --t-pre=144 \
  --t-end=196 \
  --window-label=Early-quiet \
  --window-start=145 \
  --window-end=160 \
  --linear-chain=.local_artifacts/oos_forecast/hlt_linear_hmc_quiet_1994_200.jls \
  --surrogate-chain=.local_artifacts/oos_forecast/hlt_surrogate_hmc_quiet_1994_150.jls \
  --gate-calibration=.local_artifacts/oos_forecast/gate_calibration_quiet_1994_18p.jls \
  --out-dir=.local_artifacts/oos_forecast

julia --project=. scripts/oos_gate_recalibration.jl \
  --out-dir=.local_artifacts/oos_gate_recalibration \
  --oos-out-dir=.local_artifacts/oos_forecast

julia --project=. scripts/run_surrogate_hmc_advancedhmc.jl \
  --surrogate=.local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls \
  --data=.local_artifacts/oos_forecast/hlt_real_data_payload_quiet_1994_18p.jls \
  --gate-calibration=.local_artifacts/oos_gate_recalibration/gate_q95_pad.jls \
  --out=.local_artifacts/oos_forecast/hlt_surrogate_hmc_quiet_1994_q95_pad_150.jls \
  --samples=150 \
  --adapt=100 \
  --seed=42 \
  --init-from=.local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls \
  --gate-mode=soft

julia --project=. scripts/oos_forecast_evaluate.jl \
  --tag=quiet_1994_q95_reestimated_fullgate \
  --t-pre=144 \
  --t-end=196 \
  --window-label=Early-quiet \
  --window-start=145 \
  --window-end=160 \
  --linear-chain=.local_artifacts/oos_forecast/hlt_linear_hmc_quiet_1994_200.jls \
  --surrogate-chain=.local_artifacts/oos_forecast/hlt_surrogate_hmc_quiet_1994_q95_pad_150.jls \
  --gate-calibration=.local_artifacts/oos_gate_recalibration/gate_q95_pad.jls \
  --gate-k-pre=4 \
  --gate-k-post=8 \
  --gate-min-len=4 \
  --out-dir=.local_artifacts/oos_forecast
```

### 8. SEP Sensitivity

Completed bounded pilot currently used by the paper:

```bash
julia --project=. scripts/sep_sensitivity_study.jl \
  --out-dir=.local_artifacts/sep_sensitivity
```

Completed full 265-period production rerun:

```bash
julia --project=. scripts/sep_sensitivity_study.jl \
  --n-draws=10 \
  --cap-draws=5 \
  --periods=265 \
  --sep-horizon=40 \
  --sep-maxit=200 \
  --out-dir=.local_artifacts/sep_sensitivity/full_265_20260521
```

Inspect the completed artifact with:

```bash
tail -n 120 .local_artifacts/sep_sensitivity/logs/full_265_20260521.log
sed -n '1,220p' .local_artifacts/sep_sensitivity/full_265_20260521/SEP_SENSITIVITY_SUMMARY.md
```

The status/provenance note is
`docs/review/SEP_265_RUN_STATUS.md`. The full-window run completed with all
cells converged, but `K=3` differs from the `K=5` reference by RMSE `1.08e-02`,
just above the pre-specified `1e-2` screen. Treat it as evidence of tolerance
stability with mild quadrature-node sensitivity, not as an unconditional clean
pass.

### 9. Galí Hard-ELB Validation Package

```bash
julia --project=. scripts/gali_validation_package.jl \
  --run-id=gali_validation_package_20260521
```

The package-level report is written under
`.local_artifacts/gali_validation_package/<run-id>/`. The current report audits
the hard-ELB stress path, ROM1-residual grid, ROM1-inversion grid,
one-parameter HMC smoke, two-parameter `std_z,std_a` HMC validation,
`std_nu` identification probes, and the direct-SEP feasibility smoke. It is a
positive validation of the maintained two-parameter ROM1-residual/inversion/HMC
pipeline, not a completed three-parameter direct-SEP HMC certificate.

### 10. HLT Direct-SEP Smoke

```bash
julia --project=. scripts/hlt_direct_sep_surrogate_posterior_validation.jl \
  --samples=1 \
  --burnin=0 \
  --proposal-scales=0.00001,0.00001,0.0001
```

This is a provenance smoke test, not a submission-grade posterior comparison.
It verifies that the direct SEP inversion likelihood now returns a finite
exact-determinant value under the bounded HLT three-parameter harness.

### 11. Reduced HLT Validation Bridge

Econometrica development starts from a reduced SW07-HLT investment-block bridge:

```bash
julia --project=. scripts/hlt_reduced_bridge_validation.jl \
  --stage=design \
  --dry-run=true \
  --run-id=econometrica_bridge_design
```

Executable finite-support smoke:

```bash
julia --project=. scripts/hlt_reduced_bridge_validation.jl \
  --stage=smoke \
  --run-id=econometrica_bridge_smoke \
  --periods=1 \
  --grid-axis=1 \
  --direct-eval-points=1 \
  --sep-horizon=2 \
  --sep-maxit=20 \
  --observables=dy,dinve,robs
```

The default bridge block is `crhob`, `crhoqs`, `z_eb`, and `z_eqs`, with
observables `dy`, `dinve`, `labobs`, `pinfobs`, and `robs`. The design stage
writes the manifest and acceptance gates. The smoke stage builds a tiny
steady-state HLT panel and evaluates ROM1/inversion plus direct SEP/inversion
on a local grid before any surrogate training or HMC bridge run is launched.
The executable test path is gated behind
`RUN_HLT_BRIDGE_EXEC_SMOKE=1 julia --project=. test/test_hlt_reduced_bridge_validation.jl`.
The same reduced block is available to dataset-generation and estimation
scripts as `--param-set=investment_4p`; the curvature stress block is
`--param-set=investment_curvature_5p`. After support mapping, the finite-support
bridge block is available as `--param-set=investment_4p_supported`, which keeps
the same four parameters but trims `z_eb` to `[1.20, 1.85]`.

Tiny bridge dataset smoke:

```bash
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --param-set=investment_4p \
  --theta-sampling=grid \
  --grid=1 \
  --samples-per-theta=1 \
  --burn-in=1 \
  --sample-length=1 \
  --sample-start=47 \
  --rom-orders=1 \
  --sep-horizon=2 \
  --sep-maxit=20 \
  --sep-accept-tol=1e-2 \
  --shock-scale=0.05 \
  --use-obc \
  --output-dir=.local_artifacts/hlt_reduced_bridge_validation/investment4p_dataset_grid_smoke
```

Tiny bridge training smoke:

```bash
julia --project=. scripts/hlt_sep_surrogate_train.jl \
  .local_artifacts/hlt_reduced_bridge_validation/investment4p_dataset_grid_smoke/hlt_sep_surrogate_dataset.jls \
  --rom-residual=1 \
  --obs-only \
  --epochs=3 \
  --hidden=16 \
  --hidden2=8 \
  --out=.local_artifacts/hlt_reduced_bridge_validation/investment4p_dataset_grid_smoke/hlt_sep_surrogate_trained_smoke.jls
```

Bridge support expansion smoke:

```bash
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --param-set=investment_4p \
  --theta-sampling=grid \
  --grid=2 \
  --samples-per-theta=1 \
  --burn-in=1 \
  --sample-length=1 \
  --sample-start=47 \
  --rom-orders=1 \
  --sep-horizon=2 \
  --sep-maxit=30 \
  --sep-accept-tol=1e-2 \
  --shock-scale=0.05 \
  --use-obc \
  --theta-attempts-per-theta=2 \
  --retry-on-early-failure=true \
  --output-dir=.local_artifacts/hlt_reduced_bridge_validation/investment4p_dataset_grid2
```

The 2026-05-27 run produced finite SEP residuals for 8 of 16 grid cells
(median `2.243e-8`, maximum `8.87e-7`). Treat this as a support-mapping check:
the full rectangular bridge support includes high-stress corners that should be
trimmed or reached by adaptive continuation before any posterior comparison.

Summarize the numerical support map:

```bash
julia --project=. scripts/hlt_bridge_support_report.jl \
  .local_artifacts/hlt_reduced_bridge_validation/investment4p_dataset_grid2/hlt_sep_surrogate_dataset.jls
```

Finite-support bridge run after trimming:

```bash
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --param-set=investment_4p_supported \
  --theta-sampling=grid \
  --grid=4 \
  --samples-per-theta=1 \
  --burn-in=1 \
  --sample-length=1 \
  --sample-start=47 \
  --rom-orders=1 \
  --sep-horizon=2 \
  --sep-maxit=30 \
  --sep-accept-tol=1e-2 \
  --shock-scale=0.05 \
  --use-obc \
  --theta-attempts-per-theta=2 \
  --retry-on-early-failure=true \
  --output-dir=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4
```

The 2026-05-27 finite-support run over
`.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4_20260527/`
solved all 256 grid cells. SEP residuals were finite in all cells, with
min/median/max `2.384e-12 / 1.517e-9 / 9.745e-6`. The corresponding obs-only
ROM1-residual surrogate training smoke improved validation RMSE relative to
ROM1 in all seven output dimensions by roughly 89--93 percent.

## Artifact Map

| Result class | Script | Default artifact directory |
|---|---|---|
| Paper build | `docs/SurrogateNN_paper/compile.sh` | `docs/SurrogateNN_paper/` |
| Surrogate dataset | `scripts/hlt_sep_surrogate_dataset_generate.jl` | `data/` or `--output-dir` |
| Surrogate training | `scripts/hlt_sep_surrogate_train.jl` | `--out` path |
| HLT bridge support report | `scripts/hlt_bridge_support_report.jl` | beside input dataset |
| Linear HMC | `scripts/run_linear_hmc_advancedhmc.jl` | `--out` path |
| Surrogate HMC | `scripts/run_surrogate_hmc_advancedhmc.jl` | `--out` path |
| LL decomposition | `scripts/decompose_ll_gap.jl` | `.local_artifacts/ll_decomposition/` |
| Mode sensitivity | `scripts/mode_sensitivity_report.jl` | `.local_artifacts/mode_sensitivity/` |
| OOS forecast | `scripts/oos_forecast_evaluate.jl` | `.local_artifacts/oos_forecast/` |
| Gate recalibration | `scripts/oos_gate_recalibration.jl` | `.local_artifacts/oos_gate_recalibration/` |
| SEP sensitivity | `scripts/sep_sensitivity_study.jl` | `.local_artifacts/sep_sensitivity/` |
| Galí hard-ELB validation package | `scripts/gali_validation_package.jl` | `.local_artifacts/gali_validation_package/` |
| Direct SEP smoke | `scripts/hlt_direct_sep_surrogate_posterior_validation.jl` | `.local_artifacts/hlt_direct_sep_surrogate_validation/` |
| Reduced HLT validation bridge | `scripts/hlt_reduced_bridge_validation.jl` | `.local_artifacts/hlt_reduced_bridge_validation/` |

## Known Heavy-Run Caveats

- The Galí validation package passes for the maintained hard-ELB,
  two-parameter ROM1-residual/inversion/HMC design. The original
  three-parameter direct-SEP HMC comparison remains a scaling target; current
  `std_nu` probes indicate weak local identification in the short Galí design,
  not a surrogate approximation failure.
- The full 265-period SEP sensitivity rerun completed under
  `.local_artifacts/sep_sensitivity/full_265_20260521/`. All cells converged,
  and `accept_tol` is stable within each fixed `K`; however, moving from `K=5`
  to `K=3` gives RMSE `1.08e-02`, just above the original `1e-2` screen.
- The quiet-sample OOS results identify a gate-calibration issue. The original
  hard gate is active throughout the quiet holdout. The package
  documents that pilot, a fixed-posterior q95 padded-gate diagnostic, and a
  150-draw q95 re-estimation pass. The re-estimated q95 chain cuts the full
  aggregate RMSE ratio from 2.40 to 1.09, but ROM1 still wins the full quiet
  window.
- `.local_artifacts/` may contain local cached chains on the author's machine,
  but a clean checkout should regenerate them with the commands above.
