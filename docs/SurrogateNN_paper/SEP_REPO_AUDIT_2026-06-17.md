# SEP Repository Audit

Date: 2026-06-17

## Bottom Line

The core SEP Newton solver is functioning on the tested paths. The audit did
not find evidence that the solver residual/convergence flag itself is
misreported. It did find a serious shock-unit convention bug in the HLT-facing
pipeline: HLT and Galí model equations already multiply unit structural shocks
by their volatility parameters, but several SEP code paths and scripts treated
the shock variables as if they were already scaled by those same parameters.

This has now been corrected in the maintained code. Existing HLT artifacts
created before this correction must not be used as final paper evidence unless
they are regenerated under the unit-shock convention.

## Critical Findings

1. **SEP expectation nodes used the wrong shock covariance for HLT/Galí-style
   model files.**

   The old `sep_solve_mm!` path built the Gauss-Hermite/HMC expectation
   covariance from parameters named `z_<shock>`. In HLT, however, equations have
   terms such as `z_em / 100 * em[x]`; in Galí, equations use terms such as
   `std_a * eps_a[x]`. The shock variables supplied to SEP should therefore be
   unit structural innovations by default. Scaling expectation nodes by `z_*`
   double-counted the volatility parameters.

   Fix: `SEPSolverOptions` now has `shock_scaling=:none` by default, and
   `_sep_shock_covariance` returns an identity covariance for structural shocks
   unless `shock_scaling=:parameter` is requested explicitly.

2. **The built-in SEP inversion likelihood used the same wrong scaling.**

   The old inversion filter constructed shock prior scales from `z_<shock>` and
   transformed the observation-shock Jacobian accordingly. That was inconsistent
   with HLT/Galí equations in which shock standard deviations already enter the
   model equations.

   Fix: `_sep_inv_shock_sigmas(...; shock_scaling=:none)` now defaults to unit
   structural shocks. The inversion diagnostics record
   `sep_inv_shock_scaling`, and `get_loglikelihood` exposes
   `sep_inv_shock_scaling` for explicit overrides.

3. **Several HLT empirical scripts and artifacts used the old convention.**

   Script defaults and replay paths have been changed to `shock_scaling=:none`
   for HLT dataset generation, real-data payload construction, bridge
   comparison, shock-scale decomposition, curvature sensitivity, and
   counterfactual ablation scripts.

   Artifact scan result: 74 existing `.local_artifacts` HLT/bridge/counterfactual
   `.jls` files still record `shock_scaling = parameter`. These are pre-fix
   artifacts and should be treated as historical diagnostics only.

## Convergence Semantics

The solver itself remains strict:

- `convergence_flag == 0` means the SEP residual met `sep_tol`.
- `simulate_sep_extended_path` can accept a flagged path only when all retained
  residuals are below the explicit `sep_accept_tol`.
- Production scripts often use looser acceptance thresholds, for example
  `sep_accept_tol = 0.35` in counterfactual decomposition. That is not a silent
  solver bug, but paper tables must report residual quantiles and distinguish
  strict convergence from acceptance-by-threshold.

## Verification Run After Fix

All of the following tests passed after the unit-shock correction:

- `julia --project=. test/test_sep_inversion_filter_likelihood.jl`: 32/32
- `julia --project=. test/test_sep_solver.jl`: 57/57
- `julia --project=. test/test_sep_stochastic.jl`: 28/28
- `julia --project=. test/test_sep_integration.jl`: 29/29
- `julia --project=. test/test_hlt_obc_sep.jl`: 4/4
- `julia --project=. test/test_sep_hmc_gh_convergence.jl`: 7/7
- `julia --project=. test/test_hmc_sep.jl`: 10/10

The HLT hard-OBC test now logs identity covariance for the seven structural
shocks in the stochastic SEP expectation tree, confirming that the maintained
HLT path uses unit structural innovations.

## Artifact Status

Galí validation artifacts that explicitly record unit shocks remain conceptually
consistent with the corrected convention. HLT artifacts generated under
`shock_scaling = parameter` do not. In particular, old bridge validation,
real-data payload, curvature, and ablation artifacts should be rerun before
being cited in the paper or replication package.

## Required Next Steps

1. Regenerate the HLT real-data payloads and SEP surrogate datasets with
   `shock_scaling=:none`.
2. Rerun the HLT bridge validation under the corrected SEP expectation
   covariance.
3. Rerun the investment-channel ablation/curvature evidence under the corrected
   convention.
4. For every regenerated table or figure, record `shock_scaling`, `sep_shock_scale`,
   `sep_tol`, `sep_accept_tol`, residual quantiles, and solver coverage in the
   artifact manifest.
5. Do not claim the existing HLT bridge/ablation artifacts as final evidence
   until the reruns replace the 74 pre-fix parameter-scaled artifacts.

## Honest Assessment

The concern was justified. The SEP solver can solve the tested models, but the
HLT empirical pipeline before this audit mixed shock units in a way that can
change nonlinear curvature, likelihood values, surrogate residuals, and
mechanism decompositions. The code now uses the correct default convention for
the repo's HLT and Galí model files, but the affected HLT empirical results need
to be regenerated before submission.
