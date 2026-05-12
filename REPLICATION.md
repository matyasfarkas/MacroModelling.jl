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

```bash
julia --project=. scripts/sep_sensitivity_study.jl \
  --out-dir=.local_artifacts/sep_sensitivity
```

### 9. HLT Direct-SEP Smoke

```bash
julia --project=. scripts/hlt_direct_sep_surrogate_posterior_validation.jl \
  --samples=1 \
  --burnin=0 \
  --proposal-scales=0.00001,0.00001,0.0001
```

This is a provenance smoke test, not a submission-grade posterior comparison.
It verifies that the direct SEP inversion likelihood now returns a finite
exact-determinant value under the bounded HLT three-parameter harness.

## Artifact Map

| Result class | Script | Default artifact directory |
|---|---|---|
| Paper build | `docs/SurrogateNN_paper/compile.sh` | `docs/SurrogateNN_paper/` |
| Surrogate dataset | `scripts/hlt_sep_surrogate_dataset_generate.jl` | `data/` or `--output-dir` |
| Surrogate training | `scripts/hlt_sep_surrogate_train.jl` | `--out` path |
| Linear HMC | `scripts/run_linear_hmc_advancedhmc.jl` | `--out` path |
| Surrogate HMC | `scripts/run_surrogate_hmc_advancedhmc.jl` | `--out` path |
| LL decomposition | `scripts/decompose_ll_gap.jl` | `.local_artifacts/ll_decomposition/` |
| Mode sensitivity | `scripts/mode_sensitivity_report.jl` | `.local_artifacts/mode_sensitivity/` |
| OOS forecast | `scripts/oos_forecast_evaluate.jl` | `.local_artifacts/oos_forecast/` |
| Gate recalibration | `scripts/oos_gate_recalibration.jl` | `.local_artifacts/oos_gate_recalibration/` |
| SEP sensitivity | `scripts/sep_sensitivity_study.jl` | `.local_artifacts/sep_sensitivity/` |
| Direct SEP smoke | `scripts/hlt_direct_sep_surrogate_posterior_validation.jl` | `.local_artifacts/hlt_direct_sep_surrogate_validation/` |

## Known Heavy-Run Caveats

- The Galí direct SEP-HMC versus surrogate-HMC posterior comparison is still a
  pending contribution. Current direct likelihood smokes are finite, but the
  matched HMC artifact is not yet part of the package.
- The quiet-sample OOS results identify a gate-calibration issue. The original
  hard gate is active throughout the quiet holdout. The package
  documents that pilot, a fixed-posterior q95 padded-gate diagnostic, and a
  150-draw q95 re-estimation pass. The re-estimated q95 chain cuts the full
  aggregate RMSE ratio from 2.40 to 1.09, but ROM1 still wins the full quiet
  window.
- `.local_artifacts/` may contain local cached chains on the author's machine,
  but a clean checkout should regenerate them with the commands above.
