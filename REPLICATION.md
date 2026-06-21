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

On a fresh Mac clone:

```bash
git clone <repository-url>
cd <repository-directory>
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
RUN_PAPER=0 bash scripts/replication_smoke.sh
```

If Julia is not on `PATH`, either install it with `juliaup` or pass an explicit
binary path to the long-running wrappers:

```bash
JULIA_BIN=/path/to/julia bash scripts/run_hlt_unitshock_rebuild.sh base
```

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

### Current Corrected Unit-Shock HLT Pipeline

The current full-scale target is the corrected unit-shock HLT pipeline. In the
HLT and Galí model files, shock standard deviations enter the equations, so
SEP shock nodes and inverted shocks should be unit structural innovations:
`--shock-scaling=none` for dataset generation and the default
`sep_inv_shock_scaling=:none` for inversion likelihoods.

The wrapper below is the production entry point. It writes one log per stage
under `$RUN_ROOT/logs/`, checkpoints dataset generation after each theta point,
and keeps the full gate+NN and linear+gate HMC runs exactly matched except for
`--disable-nn-correction`.

```bash
export RUN_ROOT=.local_artifacts/hlt_18param_validation_unitshock_full_$(date +%Y%m%d)

screen -dmS hlt_unit_base bash scripts/run_hlt_unitshock_rebuild.sh base
screen -dmS hlt_unit_zlb  bash scripts/run_hlt_unitshock_rebuild.sh zlb

# Monitor:
tail -f "$RUN_ROOT/logs/base_dataset.log"
tail -f "$RUN_ROOT/logs/zlb_dataset.log"

# After both datasets complete:
bash scripts/run_hlt_unitshock_rebuild.sh combine-train

# Required before full HMC:
bash scripts/run_hlt_unitshock_rebuild.sh hmc-smoke-full
bash scripts/run_hlt_unitshock_rebuild.sh hmc-smoke-lineargate

# Full matched estimation, one chain at a time:
HMC_SEED=42 FULL_ADAPT=500 FULL_SAMPLES=1000 \
  screen -dmS hlt_unit_fullnn bash scripts/run_hlt_unitshock_rebuild.sh hmc-full

HMC_SEED=42 FULL_ADAPT=500 FULL_SAMPLES=1000 \
  screen -dmS hlt_unit_lineargate bash scripts/run_hlt_unitshock_rebuild.sh hmc-lineargate
```

Default inputs for the HMC stages are:

```text
DATA_PATH=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p_unitshock_nonobc_20260620.jls
GATE_PATH=.local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p_unitshock_20260619.jls
INIT_FROM=.local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls
SURROGATE_PATH=$RUN_ROOT/hlt_sep_surrogate_trained_with_zlb_unitshock.jls
```

Acceptance checks before reporting the full run are: completed base and ZLB
datasets with documented solver coverage, validation RMSE below the ROM1
baseline in every observable, finite log posterior and finite finite-difference
gradient at the initialization point, zero or negligible post-warmup
divergences, and a fixed-theta comparison of full gate+NN versus linear+gate at
the two posterior means.

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

### 11. HLT Nonlinearity-Source Ablations

Fixed high-Kimball neighborhood diagnostic:

```bash
julia --project=. scripts/kimball_curvature_sensitivity.jl \
  --no-chain-theta \
  --curvp-values=77.3 \
  --cprobp-values=0.667 \
  --cfc-values=1.2 \
  --param-overrides=calfa=0.2,cg=0.18,chabb=0.67,clandaw=1.1,constebeta=0.3,constepinf=0.7,crdy=0.0,crpi=1.5,crr=0.73,cry=0.125,csadjcost=4.89,csigl=2.0,ctou=0.025,ctrend=0.4,curvw=8.31,czcap=0.431818 \
  --out=.local_artifacts/kimball_curvature_sensitivity/hlt_reported_full_mapped_maxit400_20260606
```

Posterior-region local ablation:

```bash
JULIA_BIN=julia bash scripts/run_hlt_posterior_region_ablation_queued.sh
```

The posterior-region run writes:

```text
.local_artifacts/counterfactual_decomposition/
  hlt_posterior_region_local10_40draws_scales012505_queued_20260609/
```

The completed paper artifact uses 40 representative posterior draws and shock
scales 0.1, 0.25, and 0.5. SEP coverage is 100 percent for all variants and
scales. Kimball curvature is the largest local sensitivity in 5, 0, and 0
percent of draws across those scales; investment-adjustment and utilization
curvature are largest in the remaining 95, 100, and 100 percent. This is the
posterior-region evidence behind the paper's claim that the dominant local
nonlinearity is real-side investment curvature rather than nonlinear
Phillips-curve curvature.

### 12. HLT Density-Scaled Curvature Surfaces

This visual diagnostic plots the RMSE of one-step observable forecast errors
`SEP - ROM1` over parameter surfaces around a chosen HLT parameter center. The
paper surfaces compare real-side curvature (`csadjcost`, `czcap`) against
Kimball curvature (`curvp`, `curvw`) using common shock paths in every grid
cell. The current manuscript uses density-scaled 13-by-13 grids at +/-1 and
+/-2 local standard deviations around two centers: the maintained SW07--HLT
baseline calibration and the HLT high-Kimball stress point.

```bash
julia --project=. scripts/hlt_posterior_mean_curvature_surface.jl \
  --plot-only=true \
  --run-id=sw07_hlt_baseline_surface_13x13_std2_bounded_shock01_1path_20260611 \
  --paper-figure-stem=fig_hlt_curvature_surface_sw07_hlt_baseline_13x13_std2_shock01
```

The diagnostic writes CSV/JLS payloads, contour plots, 3D surfaces, and
center cross-sections under:

```text
.local_artifacts/hlt_posterior_mean_curvature_surface/<run-id>/
```

The locked manuscript figure set is:

```text
docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_sw07_hlt_baseline_13x13_std1_shock01.png
docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_sw07_hlt_baseline_13x13_std2_shock01.png
docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_highkimball_13x13_std1_shock01.png
docs/SurrogateNN_paper/figures/fig_hlt_curvature_surface_highkimball_13x13_std2_shock01.png
```

The baseline charts show real-side dominance over +/-1 and +/-2 local
standard deviations. The high-Kimball stress charts show that price/wage
Kimball curvature can dominate when the center is deliberately moved to the
HLT high-curvature neighborhood. This is why the manuscript states the
investment-channel conclusion as a posterior-region finding, not a global
theorem.

To regenerate only the figures from the saved surface payload:

```bash
julia --project=. scripts/hlt_posterior_mean_curvature_surface.jl \
  --plot-only=true \
  --run-id=<run-id> \
  --paper-figure-stem=<figure-stem>
```

## Paper Result Provenance

The lightweight provenance ledger for the current paper draft is:

```text
docs/SurrogateNN_paper/RESULTS_PROVENANCE_2026-06-10.md
```

That file records the run ids, artifact paths, and headline numbers used by
the manuscript. Heavy `.jls`, `.log`, and timestamped `.local_artifacts`
outputs remain local by design.

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

One-period posterior-grid comparison on the supported bridge:

```bash
julia --project=. scripts/hlt_bridge_posterior_grid_compare.jl \
  --dataset=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4_20260527/hlt_sep_surrogate_dataset.jls \
  --surrogate=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4_20260527/hlt_sep_surrogate_trained_supported_grid4.jls \
  --out-dir=.local_artifacts/hlt_reduced_bridge_validation/posterior_grid_compare_supported_grid4_sigma100_noise025_20260527 \
  --param-set=investment_4p_supported \
  --truth-mode=validation-nearest-center \
  --obs-sigma-scale=1.0 \
  --dgp-noise-scale=0.25
```

The noisy 2026-05-27 comparison passes: the held-out truth point is in the
validation split, the surrogate posterior intervals overlap direct SEP for all
four bridge parameters, prediction RMSE falls from `0.1427` under ROM1 to
`0.00564` under the surrogate, and log-posterior surface RMSE falls from
`86.48` under ROM1 to `0.36` under the surrogate. This is a finite-support
known-feature grid comparison, not yet the full inversion-filter HMC bridge.

Dense supported-grid robustness sweep:

```bash
julia --project=. scripts/hlt_bridge_robustness_sweep.jl \
  --stage=full \
  --run-id=investment4p_supported_grid10_full_sweep_20260527 \
  --param-set=investment_4p_supported \
  --grid=10 \
  --epochs=600 \
  --obs-sigma-scale=1.0 \
  --dgp-noise-scale=0.25
```

The sweep wrapper runs dataset generation, support reporting, obs-only
ROM1-residual training, and the posterior-grid comparison in sequence. It can
also be launched with `--stage=postprocess --dataset-dir=<existing dataset dir>`
to wait for an already-running dataset generator before training and comparing.
The 2026-05-27/28 grid-10 sweep completed all 10,000 direct-SEP cells and passed
the same posterior-grid comparison. SEP residual min/median/max are
`2.700e-13 / 1.571e-9 / 9.996e-6`. Prediction RMSE against direct SEP fell from
`0.143574` under ROM1 to `0.000749395` under the surrogate, and log-posterior
surface RMSE fell from `88.0719` to `0.0534526`. This is the paper's maintained
finite-support HLT bridge artifact. It remains a one-period known-feature
comparison, not the full inversion-filter HMC bridge.

Multi-period inversion bridge scaffold and bounded executable stress test:

```bash
julia --project=. scripts/hlt_bridge_inversion_filter_compare.jl \
  --dataset=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4_20260527/hlt_sep_surrogate_dataset.jls \
  --surrogate=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4_20260527/hlt_sep_surrogate_trained_supported_grid4.jls \
  --out-dir=.local_artifacts/hlt_reduced_bridge_validation/inversion_bridge_grid4_design_20260527 \
  --param-set=investment_4p_supported \
  --periods=8 \
  --direct-eval-points=25 \
  --sep-horizon=4 \
  --sep-maxit=80 \
  --dry-run=true
```

This scaffold freezes the next validation step: a short multi-period
inversion-filter bridge using a held-out validation truth point. The current
mode verifies dataset/surrogate compatibility, split provenance, observation
scaling, and artifact schema.

The maintained dynamic HLT bridge uses the completed grid-10 artifact and a
direct-SEP generator-path panel. This mode mirrors the HLT estimator's online
architecture more closely: ROM1 recovers shocks and propagates states, while the
surrogate supplies an observation-only residual correction in the likelihood
evaluation. The direct SEP objective uses the same recovered shocks under a
common measurement-error likelihood. The fixed one-step NN residual is retained
as a failure diagnostic; the maintained passing run uses a path-calibrated ridge
residual trained on direct SEP residuals along dynamic ROM1 inversion paths.

```bash
julia --project=. scripts/hlt_bridge_inversion_filter_compare.jl \
  --dataset=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid10_full_sweep_20260527/hlt_sep_surrogate_dataset.jls \
  --surrogate=.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid10_full_sweep_20260527/hlt_sep_surrogate_trained_investment4p_supported_grid10_full_sweep_20260527.jls \
  --out-dir=.local_artifacts/hlt_reduced_bridge_validation/full_dynamic_hlt_profile_dynamic_ridge_floor01_20260602 \
  --param-set=investment_4p_supported \
  --periods=8 \
  --direct-eval-points=10 \
  --panel-mode=direct-sep-rollout \
  --direct-panel-solver=generator-path \
  --direct-objective=common-measurement-error \
  --surrogate-objective=dynamic-ridge-residual \
  --dynamic-calibration-train-points=5 \
  --dynamic-ridge-lambda=1e-4 \
  --dynamic-feature-mode=state-shock-theta-time \
  --obs-sigma-floor=0.1 \
  --sep-horizon=2 \
  --sep-maxit=120 \
  --inversion-maxit=10 \
  --profile-direct-repeats=1 \
  --profile-fast-repeats=20 \
  --dry-run=false
```

Three corrected 2026-06-02 dynamic diagnostics are recorded. The four-period
smoke
`.local_artifacts/hlt_reduced_bridge_validation/full_dynamic_hlt_profile_generator_smoke_fixed_20260602/`
passes on three nearby anchors: all direct-SEP objectives are finite, local MAP
agreement holds, surrogate intervals overlap direct SEP, and centered surface
RMSE falls from `1.190` under ROM1 to `0.861` under the residual surrogate. The
fixed one-step NN eight-period run
`.local_artifacts/hlt_reduced_bridge_validation/full_dynamic_hlt_profile_generator_moderate_20260602/`
is finite at all direct-SEP anchors but fails the dynamic surface criterion:
centered surface RMSE is `1.872` for the surrogate versus `1.395` for ROM1, with
a large mean objective offset. The path-calibrated dynamic ridge run shown above
supersedes that failure for the reduced bridge. It uses five calibration anchors
and five held-out anchors, reaches held-out residual RMSE `0.011818`, reduces
held-out centered surface RMSE from `1.437` under ROM1 to `0.703` under the
dynamic residual, matches the direct local MAP, and has interval overlap for all
four bridge parameters. Direct SEP takes a median `34.63` seconds per
eight-period candidate (`4.33` seconds per period), while the dynamic surrogate
takes a median about `0.001` seconds per candidate after 20-repeat averaging.
The projected direct cost for 10 anchors and 265 periods is about `3.19` hours.
This result validates the reduced dynamic bridge; a full 18-parameter
direct-SEP HMC comparison remains a scaling target.

## Artifact Map

| Result class | Script | Default artifact directory |
|---|---|---|
| Paper build | `docs/SurrogateNN_paper/compile.sh` | `docs/SurrogateNN_paper/` |
| Surrogate dataset | `scripts/hlt_sep_surrogate_dataset_generate.jl` | `data/` or `--output-dir` |
| Surrogate training | `scripts/hlt_sep_surrogate_train.jl` | `--out` path |
| HLT bridge support report | `scripts/hlt_bridge_support_report.jl` | beside input dataset |
| HLT bridge posterior grid | `scripts/hlt_bridge_posterior_grid_compare.jl` | `--out-dir` |
| HLT bridge robustness sweep | `scripts/hlt_bridge_robustness_sweep.jl` | `.local_artifacts/hlt_reduced_bridge_validation/` |
| HLT multi-period inversion bridge | `scripts/hlt_bridge_inversion_filter_compare.jl` | `--out-dir` |
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
