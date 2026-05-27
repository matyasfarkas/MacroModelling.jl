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
- Next bridge step: expand the direct grid and train a bridge-specific
  ROM1-residual surrogate over the reduced investment block.

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
