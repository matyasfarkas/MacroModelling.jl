# Econometrica Validation Bridge And Theory Cleanup

Date: 2026-05-27

## Target

The Econometrica version should be a methods paper first and an SW07-HLT
application second. The central claim should be:

> ROM1-residual neural surrogates can deliver auditable Bayesian estimation for
> nonlinear DSGE models with occasionally binding constraints, provided their
> approximation error is validated against direct nonlinear benchmarks.

The SW07-HLT posterior gap is then the main economic application, not the sole
identification of the method.

## Validation Bridge

The current Galí hard-ELB package is a useful small-model validation. It is not
enough for a high-probability Econometrica submission because referees can
argue that the medium-scale HLT application is still unvalidated. The bridge
should be a reduced SW07-HLT benchmark close to the paper's investment channel.

Default bridge block:

- `crhob`: risk-premium persistence
- `crhoqs`: investment-specific shock persistence
- `z_eb`: risk-premium shock scale
- `z_eqs`: investment-specific shock scale

Optional stress block:

- add `csadjcost` to verify that the residual surrogate remains accurate when
  adjustment-cost curvature itself moves.

Default observables:

- `dy`, `dinve`, `labobs`, `pinfobs`, `robs`

The bridge should compare three objects on identical synthetic data and priors:

1. ROM1/inversion objective.
2. Direct SEP objective on a reduced local grid.
3. ROM1-residual surrogate objective on the same recovered shocks and scaling.

The direct benchmark can be a local grid before it becomes HMC. A grid is easier
to audit, gives a visible posterior surface, and avoids wasting weeks on a
direct NUTS run before finite objective support is proven.

## Acceptance Criteria

- Direct SEP grid anchors are finite or explicitly classified with recovery
  diagnostics.
- Direct and surrogate reduced-block posterior surfaces agree in posterior
  means, interval overlap, and ranking of high-density cells.
- Surrogate residual RRMSE is below `0.005` on bridge observables and below
  `0.01` on all audited observables.
- ROM1 differs from direct nonlinear in the direction predicted by the
  investment-channel decomposition.
- Matched HMC is launched only after the grid pass. HMC success requires
  overlapping 90 percent intervals for every bridge parameter and posterior mean
  differences below two combined MCSE units.

## Implementation Steps

1. Freeze the bridge design manifest with
   `scripts/hlt_reduced_bridge_validation.jl --stage=design`.
2. Wire `--stage=smoke` to build a tiny steady-state HLT panel and evaluate
   finite ROM1/inversion and direct SEP/inversion likelihoods over a tiny local
   grid.
3. Add the reduced investment block to the HLT synthetic-estimation prior logic.
4. Train a bridge residual surrogate over the reduced block, not the full
   18-parameter Sobol space.
5. Produce a local direct/surrogate/ROM1 posterior grid table and contour plot.
6. Only after the grid passes, launch a short matched HMC check.

## Current Status

- Implemented the executable bridge smoke stage.
- Corrected the bridge observables to the maintained HLT model names:
  `dy`, `dinve`, `labobs`, `pinfobs`, and `robs`.
- Added `investment_4p` and `investment_curvature_5p` to the shared HLT
  parameter configuration so existing dataset-generation and estimation scripts
  can target the bridge block directly.
- Verified the minimal finite-support smoke on 2026-05-27 with one period, one
  grid cell, and observables `dy,dinve,robs`: ROM1/inversion log likelihood
  `-3.459`, direct SEP/inversion log likelihood `-2.793`, status `pass`.
- Generalized deterministic grid sampling in
  `hlt_sep_surrogate_dataset_generate.jl` to configured multi-parameter sets and
  verified a one-theta `investment_4p` dataset smoke. The resulting single
  sample had finite SEP residual `4.411e-7`.
- Verified the next smoke hop by training a tiny ROM1-residual obs-only bridge
  surrogate on that dataset. This is a metadata/pipeline check, not an accuracy
  result because the dataset has one sample.
- Ran the next deterministic support check on a 2-point-per-axis
  `investment_4p` grid. The run generated finite SEP residuals for 8 of 16
  grid cells, with median residual `2.243e-8` and maximum residual `8.87e-7`.
  The failing cells are the high-stress corners, so the bridge should proceed
  with an adaptive or trimmed support rather than treating the full rectangle
  as numerically feasible.
- Trained a bridge-specific ROM1-residual obs-only surrogate on the feasible
  8-sample grid payload. This confirms the training path and residual target
  work on the current HLT bridge data. It is still only a smoke result: the
  validation RMSE improves on the ROM baseline in all seven reported output
  dimensions, but the dataset is too small for an accuracy claim.
- Added `scripts/hlt_bridge_support_report.jl` to convert dataset metadata into
  a reproducible finite-support report. On the 2-point grid, the support report
  confirms that failures are concentrated at the high `z_eb=2.5` edge, while
  the tested `crhob`, `crhoqs`, and `z_eqs` ranges remain represented among
  successful cells.
- Completed the 3-point support map: 54 of 81 cells solved, all finite residuals
  among successful cells, median residual `3.450e-9`, maximum residual
  `9.771e-6`. The map confirms the support boundary: all cells with
  `z_eb=1.20` or `1.85` solved; all cells with `z_eb=2.50` failed at period 1.
- Trained a bridge-specific ROM1-residual obs-only surrogate on the 54-sample
  3-point feasible grid. The validation RMSE improves on the ROM1 baseline in
  all seven output dimensions by roughly 34--43 percent. This is a meaningful
  reduced-support training smoke, but it is still too small to be the final
  posterior-validation artifact.
- Added `investment_4p_supported`, which keeps the same four bridge parameters
  but trims `z_eb` to `[1.20, 1.85]`, the mapped direct-SEP support from the
  3-point run.
- Re-ran the reduced bridge test suite after adding the supported block:
  `26/26` tests passed.
- Launched the first finite-support expansion run over
  `investment_4p_supported`, using a 4-point grid, one period, one sample per
  theta, ROM1 residual outputs, and the same SEP smoke settings. Its artifact
  directory is
  `.local_artifacts/hlt_reduced_bridge_validation/investment4p_supported_grid4_20260527/`.
- Completed that finite-support expansion run: all 256 cells solved, with
  finite SEP residuals in every cell. The residual min/median/max were
  `2.384e-12 / 1.517e-9 / 9.745e-6`.
- Trained the obs-only ROM1-residual surrogate on the 256-cell supported grid.
  Validation RMSE improved relative to ROM1 in all seven output dimensions by
  roughly 89--93 percent.
- Added and ran `scripts/hlt_bridge_posterior_grid_compare.jl`, a one-period
  known-feature posterior-grid comparison over the finite HLT bridge support.
  In the noisy comparison with `obs-sigma-scale=1.0` and
  `dgp-noise-scale=0.25`, the truth point is held out by the surrogate training
  split, surrogate posterior intervals overlap direct SEP for all four bridge
  parameters, and ROM1 intervals also overlap only under this looser noise
  calibration.
- The same comparison strongly favors the surrogate over ROM1 as a likelihood
  surface approximation: prediction RMSE against direct SEP falls from `0.1427`
  to `0.00564`, and log-posterior surface RMSE falls from `86.48` to `0.36`.
- Added `scripts/hlt_bridge_robustness_sweep.jl` to run a supported-grid
  robustness pipeline end-to-end: dataset generation, support report, obs-only
  ROM1-residual training, and posterior-grid comparison. The grid-10 supported
  sweep completed all 10,000 direct-SEP cells and passed the same known-feature
  posterior-grid comparison. SEP residual min/median/max are
  `2.700e-13 / 1.571e-9 / 9.996e-6`; prediction RMSE fell from `0.143574`
  under ROM1 to `0.000749395` under the surrogate; and log-posterior surface
  RMSE fell from `88.0719` to `0.0534526`.
- Added and executed `scripts/hlt_bridge_inversion_filter_compare.jl`, the
  scaffold for the next bridge step. The bounded executable stress test
  evaluated five nearby direct-SEP inversion anchors without numerical failure,
  but it did not pass the multi-period shape-matching criterion: after removing
  the mean objective offset, centered surface RMSE was `1.2067` for the
  surrogate and `0.5169` for ROM1, with no local MAP agreement. This should be
  treated as a useful plumbing/stress-test result only, because the panel is
  assembled from held-out one-step bridge observations rather than a coherent
  synthetic time-series DGP.

## Theory Cleanup

The current appendix has the right ingredients but needs to be rewritten around
one theorem and one validation proposition.

Main theorem shape:

- Define the direct nonlinear likelihood `ell_T(theta)`.
- Define the ROM1-residual surrogate likelihood `hat_ell_T(theta)`.
- Assume compact parameter space, identification, geometric mixing, local
  smoothness, independent measurement error, high-probability training support,
  mean-square residual error, and score perturbation control.
- Prove that if the observation-space residual error satisfies
  `delta_T = o(T^{-1/2})`, then the surrogate posterior and direct nonlinear
  posterior are asymptotically equivalent in total variation.

Separate propositions:

- Deterministic pathwise log-likelihood perturbation is first order in the
  sup-norm error.
- Expected surrogate-induced log-likelihood bias is second order in
  mean-square error under measurement-error orthogonality.
- Posterior mean and credible-set perturbations follow from score control and
  local likelihood curvature.

## Writing Cuts

For an Econometrica draft, move the following out of the main text:

- long HLT implementation history,
- stale validation attempts,
- repeated caveats that can be consolidated into one identification paragraph,
- most OOS forecast detail,
- solver debugging narrative.

Keep in the main text:

- residual-learning algorithm,
- one clean theorem,
- Galí validation package,
- reduced HLT bridge,
- SW07-HLT economic application.
