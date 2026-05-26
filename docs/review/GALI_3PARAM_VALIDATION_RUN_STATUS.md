# Galí Three-Parameter Validation Run Status

Date: 2026-05-21

Worker scope: three-parameter direct-SEP versus surrogate validation artifacts under `.local_artifacts/gali_direct_sep_surrogate_hmc/`.

## Runner

Script: `scripts/gali_direct_sep_surrogate_hmc_validation.jl`

Maintained model: `models/Gali_2015_chapter_3_obc.jl`

Full-stage defaults:

- Parameters: `std_a`, `std_z`, `std_nu`
- Observables: `log_y`, `pi_ann`, `i_ann`
- Periods: 80
- Surrogate training samples: 4000
- Chains: 4 per objective
- Warmup: 500 per chain
- Draws: 1000 per chain
- Direct objective: `measurement_error`
- HMC objectives: `both`

## Timing Evidence

Existing smoke artifact:

`.local_artifacts/gali_direct_sep_surrogate_hmc/direct_sep_full_pipeline_smoke_20260518/`

That run used 4 periods, 1 chain, 4 warmup draws, 6 post-warmup draws, and max tree depth 1. The serialized chain reports:

- Direct elapsed time: 237.289 seconds for 10 total HMC iterations
- Surrogate elapsed time: 0.042 seconds for 10 total HMC iterations

This implies about 23.7 seconds per direct HMC iteration at only 4 periods and tree depth 1. The full stage uses 80 periods, 6000 direct HMC iterations, and max tree depth 7, so the direct run is expected to be a multi-week scaling run on this machine. This is not a submission-ready validation unless it completes and passes the existing criteria.

## Launched Command

Run id: `gali_3param_direct_sep_full_20260521`

Screen name: `gali_3param_full_20260521`

Log path:

`.local_artifacts/gali_direct_sep_surrogate_hmc/logs/gali_3param_direct_sep_full_20260521.log`

Artifact directory:

`.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_direct_sep_full_20260521/`

Command:

```sh
julia --project=. scripts/gali_direct_sep_surrogate_hmc_validation.jl \
  --stage=full \
  --run-id=gali_3param_direct_sep_full_20260521 \
  --out-dir=.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_direct_sep_full_20260521 \
  --hmc-objectives=both \
  --direct-objective=measurement_error
```

## Interpretation

This run preserves the full three-parameter validation criteria. If it fails the full-stage surrogate RRMSE gate, finite-gradient checks, or HMC diagnostics, that failure should be reported as a scaling/validation failure rather than papered over. If it remains in progress for days, the existing two-parameter hard-ELB validation remains the completed methodological validation, and this run remains the full three-parameter scaling target.

## 2026-05-21 Update

The default full-stage run failed before surrogate training or HMC. The initial 80-period direct SEP DGP generation returned `errorflag=true` after a non-convergence warning at `sep_maxit=100`:

```text
SEP did not converge in 100 iterations (err=0.0014959744755236315)
ERROR: LoadError: SEP simulation failed for DGP.
```

This is a strict full-stage DGP-generation failure, not a posterior comparison result.

I launched a second strict retry that keeps the full-stage validation design but increases direct SEP iterations from 100 to 200. This does not loosen the acceptance tolerance or posterior validation criteria; it only gives the direct solver more iterations.

Retry run id: `gali_3param_direct_sep_full_maxit200_20260521`

Retry screen name: `gali_3param_full_maxit200_20260521`

Retry log path:

`.local_artifacts/gali_direct_sep_surrogate_hmc/logs/gali_3param_direct_sep_full_maxit200_20260521.log`

Retry artifact directory:

`.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_direct_sep_full_maxit200_20260521/`

Retry command:

```sh
julia --project=. scripts/gali_direct_sep_surrogate_hmc_validation.jl \
  --stage=full \
  --run-id=gali_3param_direct_sep_full_maxit200_20260521 \
  --out-dir=.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_direct_sep_full_maxit200_20260521 \
  --hmc-objectives=both \
  --direct-objective=measurement_error \
  --sep-maxit=200
```

## 2026-05-21 Follow-Up

The `sep_maxit=200` retry also failed before surrogate training or HMC. The
80-period direct SEP DGP generation returned `errorflag=true` after a
non-convergence warning:

```text
SEP did not converge in 200 iterations (err=0.0004462571204779664)
ERROR: LoadError: SEP simulation failed for DGP.
```

This is again a full-stage DGP-generation failure, not a posterior comparison
result.

I launched a third strict retry that keeps the full-stage validation design and
acceptance tolerance intact but increases the direct SEP iteration budget to
1000.

Retry run id: `gali_3param_direct_sep_full_maxit1000_20260521`

Retry screen name: `gali_3param_full_maxit1000_20260521`

Retry log path:

`.local_artifacts/gali_direct_sep_surrogate_hmc/logs/gali_3param_direct_sep_full_maxit1000_20260521.log`

Retry artifact directory:

`.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_direct_sep_full_maxit1000_20260521/`

Retry command:

```sh
julia --project=. scripts/gali_direct_sep_surrogate_hmc_validation.jl \
  --stage=full \
  --run-id=gali_3param_direct_sep_full_maxit1000_20260521 \
  --out-dir=.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_direct_sep_full_maxit1000_20260521 \
  --hmc-objectives=both \
  --direct-objective=measurement_error \
  --sep-maxit=1000
```

## 2026-05-21 Outcome

The `sep_maxit=1000` retry also failed before surrogate training or HMC. The
80-period direct SEP DGP generation returned `errorflag=true` after a
non-convergence warning:

```text
SEP did not converge in 1000 iterations (err=0.00018473898682592332)
ERROR: LoadError: SEP simulation failed for DGP.
```

No `SUMMARY.md` or posterior comparison table was produced. The failure is a
strict DGP-generation failure in the direct SEP simulator for the full
three-parameter, 80-period validation design, not evidence of surrogate-HMC
posterior mismatch.

The completed paper-grade validation evidence remains the hard-ELB
two-parameter package. A full three-parameter validation will require changing
the validation design, for example by reducing the DGP horizon, changing the
shock draw, or adding a continuation/fallback strategy for direct SEP DGP
generation. Such changes should be reported as a new validation design rather
than as completion of the original strict full-stage run.

## 2026-05-23 Solver Recovery and DGP Redesign

Implemented direct SEP recovery in `src/sep_simulation.jl`: failed stochastic
SEP solves now attempt failed-iterate continuation, deterministic
perfect-foresight initialization, lifted perfect-foresight paths, zero-variance
stochastic-tree initialization, and shock-scale continuation back to the target
stochastic tree. The runner now calls this recovery path with QR first and
normal-equations fallback.

The recovery path fixes the original first-period random-shock failure, but it
does not make the original 80-period random full DGP reliable. A later random
state still stalls despite recovery and `sep_maxit=1000`; the residual remains
around `0.00536` after the cold solve and the continuation path worsens as the
shock scale rises. This remains a random-DGP feasibility failure, not posterior
evidence.

The hard monetary-policy shock DGP is also unsuitable for direct stochastic SEP:
one-period `eps_nu` amplitudes from `-2` through `-14` all failed under the
strict full-stage solve, and the old hard block produced residuals above the
acceptance threshold. The controlled DGP was therefore redesigned to use an
adverse `eps_z` ELB block, moderate separated pulses for `eps_a` and `eps_nu`,
and a deterministic startup perturbation. The startup perturbation is needed
because a zero first-period shock stalls the full stochastic tree at the
steady-state lower-bound kink before the later ELB block is reached.

The remaining binding solver constraint is the SEP horizon. With three
Gauss-Hermite nodes:

- Horizon 16 fails for the redesigned controlled DGP, even with
  `sep_maxit=1000`, with the first residual stuck near `0.00213` and recovery
  improving only to about `0.00142`.
- Horizon 10 fails after one good period.
- Horizon 8 fails after ten good periods.
- Horizon 6 solves the 80-period controlled ELB DGP end to end with no recovery
  steps and residuals around `1e-9`.

The runner defaults were updated so pilot and full stages use
`sep_horizon=6`. This is a new, explicit validation design: it preserves the
three-node stochastic tree and the 80-period controlled ELB DGP, but no longer
claims to validate the unstable horizon-16 full-stage design.

Confirmed DGP-only full-stage check after the change:

```text
full horizon=6
DGP periods=80
obs size=(3, 80)
recovery steps=0
shock design=controlled_elb
sep errorflag=false
obs sigma=[0.001, 0.001, 0.001]
first obs=[-0.07694205936068878, -0.034922483354014024, -1.789928983017518e-8]
```

Focused runner smoke test passed:

```text
Test Summary:                                      | Pass  Total   Time
Gali direct SEP vs surrogate HMC validation runner |   23     23  53.9s
```

Optional finite-likelihood smoke test also passed with
`RUN_GALI_LIKELIHOOD_SMOKE=1`:

```text
direct at theta_true log posterior at theta_true: -7.810458; finite gradient=true
direct at prior_center log posterior at theta_true: -16.780012; finite gradient=true
surrogate at theta_true log posterior at theta_true: -7.810458; finite gradient=true
surrogate at prior_center log posterior at theta_true: -7.578586; finite gradient=true
Test Summary:                                      | Pass  Total     Time
Gali direct SEP vs surrogate HMC validation runner |   26     26  1m41.2s
```

## 2026-05-25 Pilot Likelihood Gate

The pilot likelihood gate was retried after the horizon-6 DGP fix. The first
40-period pilot likelihood run with the default 600 local-path training samples
was stopped manually because the sampler spent excessive time on broad
off-path perturbations that repeatedly triggered direct SEP recovery. This was
an inefficient training-design problem, not a DGP failure.

The local-path sampler was tightened around the relevant ROM-filter/DGP
neighborhood:

- `draw_local_theta` now draws 95% of proposals near `THETA_TRUE` with log sd
  `0.025`, with a small broader component near the baseline.
- Local-path states now use the ROM-filter path 95% of the time.
- State perturbations were reduced from `0.01` to `0.002` times the empirical
  state scale.
- Structural shock perturbations were reduced from `0.10` or full standard
  normal draws to `0.02` with rare `0.05` perturbations.

The direct ROM-inversion measurement-error objective was also separated from
strict DGP-generation acceptance. DGP generation remains strict and solves at
residuals around `1e-9`, while one-step direct SEP predictions inside the
filter use `DIRECT_PREDICT_ACCEPT_TOL = 1e-2` and report the accepted residual
in diagnostics. This is needed because ROM-inversion filter states near the
prior center can generate one-step residuals between roughly `0.005` and
`0.02`, even though the same direct objective is finite at the synthetic truth.

HMC initialization was changed from the prior center to the synthetic truth for
both direct and surrogate objectives. The direct objective is therefore gated at
the HMC initialization rather than at the prior center, which is not a reliable
point for the direct SEP filter in this controlled ELB design.

A 64-sample pilot likelihood check with these changes passed the finite
likelihood/gradient smoke gate:

```text
ROM-filter-path RRMSE on observable scale: 0.00041286, 0.0066479, 0.0030284
direct at hmc_init log posterior: -1241.047788; finite gradient=true
surrogate at theta_true log posterior: -1419.342655; finite gradient=true
surrogate at prior_center log posterior: -1210.979152; finite gradient=true
Likelihood smoke complete. No NUTS chains were launched.
```

The 64-sample surrogate is not yet paper-grade because the ROM-filter
inflation RRMSE is above the `0.005` gate. I launched a 256-sample pilot
likelihood run to test whether the local-path fit tightens enough before
launching HMC.

Run id: `gali_3param_pilot_likelihood_h6_t256_20260525`

Screen name: `gali_pilot_likelihood_t256_20260525`

Log path:

`.local_artifacts/gali_direct_sep_surrogate_hmc/logs/gali_3param_pilot_likelihood_h6_t256_20260525.log`

Artifact directory:

`.local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_pilot_likelihood_h6_t256_20260525/`

The 256-sample likelihood gate completed. It passed the finite direct and
surrogate likelihood/gradient checks, but it still did not meet the paper-grade
surrogate-accuracy gate:

```text
ROM-filter-path RRMSE on observable scale: 0.00090086, 0.0095635, 0.004066
direct at hmc_init log posterior: -1241.047788; finite gradient=true
surrogate at theta_true log posterior: -1427.829281; finite gradient=true
surrogate at prior_center log posterior: -2194.389844; finite gradient=true
```

The failure is concentrated in inflation on the exact ROM-filter path. Generic
local-path sample count did not improve that diagnostic, so the next retry
increases the repeated weight on the exact ROM-filter anchor residuals and adds
explicit diagnostics for valid versus failed direct SEP anchor predictions.

The anchor-weighted retry passed the surrogate likelihood gate:

```text
Run id: gali_3param_pilot_likelihood_h6_anchor12_t128_h192_20260525
ROM-filter-path RRMSE on observable scale: 9.6703e-05, 0.0022609, 0.0010793
ROM-filter-path valid direct SEP predictions: 40
ROM-filter-path failed direct SEP predictions: 0
ROM-filter-path max direct SEP prediction error: 0.0013009
direct at hmc_init log posterior: -1241.047788; finite gradient=true
surrogate at theta_true log posterior: -1418.882890; finite gradient=true
surrogate at prior_center log posterior: -1358.650477; finite gradient=true
```

This is the first three-parameter Galí validation artifact that satisfies the
finite direct/surrogate likelihood gate and the `0.005` ROM-filter-path
surrogate-accuracy gate. A tiny `T=40` NUTS smoke run for both direct and
surrogate objectives was launched next:

```text
Run id: gali_3param_hmc_smoke_t40_anchor12_20260525
Screen name: gali_3param_hmc_smoke_t40_20260525
Warmup/draws/chains: 1 / 2 / 1
Log path: .local_artifacts/gali_direct_sep_surrogate_hmc/logs/gali_3param_hmc_smoke_t40_anchor12_20260525.log
Artifact directory: .local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_hmc_smoke_t40_anchor12_20260525/
```

The tiny HMC smoke completed. It confirms that direct SEP-HMC can run at
`T=40`, but it is too expensive for the originally proposed moderate pilot
without changing the direct-gradient strategy:

```text
Direct NUTS: warmup/draws/chains = 1 / 2 / 1
Direct elapsed: 1361.42s
Direct post-warmup numerical errors: 0
Direct mean acceptance: 1
Surrogate elapsed: 0.23s
Surrogate post-warmup numerical errors: 2
Surrogate mean acceptance: 0
```

The posterior comparison from this smoke artifact is not substantively
interpretable because there are only two post-warmup draws and the surrogate
chain had numerical errors. The useful information is operational: direct HMC
with finite-difference gradients is feasible only as a tiny reference run at
`T=40`; the surrogate HMC needs a smaller step-size retry before a longer
surrogate pilot is scheduled.

The surrogate-only HMC tuning retry with `hmc_step_size=1e-5` succeeded:

```text
Run id: gali_3param_surrogate_hmc_tune_step1e5_20260525
Warmup/draws/chains: 20 / 40 / 1
Surrogate elapsed: 2.55s
Surrogate post-warmup numerical errors: 0
Surrogate mean acceptance: 0.8817
Surrogate posterior mean: [0.0117975, 0.0423574, 0.00271335]
Surrogate 90% intervals:
  std_a:  [0.0117842, 0.01181]
  std_z:  [0.0422724, 0.042439]
  std_nu: [0.00270974, 0.00271607]
```

This establishes stable surrogate HMC on the same `T=40` controlled-ELB DGP.
The remaining validation question is no longer finite likelihood or surrogate
training; it is how much direct SEP-HMC evidence is computationally sensible to
collect given the 22.7-minute runtime for a 1/2 direct smoke chain.

Because the direct finite-difference HMC runtime is prohibitive, the next
scheduled validation artifact is a surrogate HMC pilot with direct SEP
log-posterior audits on selected posterior draws:

```text
Run id: gali_3param_surrogate_pilot_direct_audit_20260525
Screen name: gali_3param_surrogate_pilot_audit_20260525
Surrogate warmup/draws/chains: 100 / 200 / 1
Direct audit draws: theta_true, prior_center, and 8 surrogate posterior draws
Log path: .local_artifacts/gali_direct_sep_surrogate_hmc/logs/gali_3param_surrogate_pilot_direct_audit_20260525.log
Artifact directory: .local_artifacts/gali_direct_sep_surrogate_hmc/gali_3param_surrogate_pilot_direct_audit_20260525/
```

The surrogate pilot with direct audit completed:

```text
Surrogate warmup/draws/chains: 100 / 200 / 1
Surrogate elapsed: 48.7s
Surrogate post-warmup numerical errors: 0
Surrogate mean acceptance: 0.8979
Surrogate posterior mean: [0.0117969, 0.0423855, 0.00271356]
Surrogate 90% intervals:
  std_a:  [0.0117736, 0.011818]
  std_z:  [0.0422949, 0.042483]
  std_nu: [0.00271034, 0.0027168]
Direct audit elapsed: 447.9s for 10 points
```

The audit is informative but not a clean validation. The direct likelihood
also prefers the surrogate posterior region over `theta_true`: across the
eight audited posterior draws, direct log posteriors are 46 to 97 nats above
the direct log posterior at `theta_true`. However, the direct and surrogate
log-posterior rankings across those eight draws are weakly aligned
(`correlation = 0.117`), and the surrogate posterior intervals do not cover the
true values in this finite sample. This is not evidence that the methodology is
invalid, because direct and surrogate both move away from the synthetic truth
under the chosen measurement-error likelihood, but it is not strong enough for
the paper as an unbiased-recovery validation table.

An exploratory retry with ROM-filter anchors at seven nearby theta values was
also run:

```text
Run id: gali_3param_likelihood_thetaanchors_t128_h192_20260525
ROM-filter-path RRMSE on observable scale: 0.00035621, 0.0051052, 0.0023359
direct at hmc_init log posterior: -1241.047788; finite gradient=true
surrogate at theta_true log posterior: -1399.817417; finite gradient=true
surrogate at prior_center log posterior: -933.984288; finite gradient=true
```

This theta-anchor design was rejected as the new default: it just misses the
inflation RRMSE gate and makes the surrogate objective implausibly favorable at
the prior center. The runner was reverted to the exact ROM-filter anchor design
that produced the valid path-accuracy artifact.

## 2026-05-25 Profiled Direct Objective Check

The active validation artifacts above use the ROM-inversion
measurement-error objective: shocks are recovered with the ROM1 inversion
filter and then evaluated under direct SEP or the surrogate. This keeps the
direct and surrogate objectives comparable, but it is not a fully profiled
direct SEP measurement-error objective.

The runner also contains `--direct-objective=profiled_measurement_error`,
which re-optimizes shocks against direct SEP using finite-difference SEP
Jacobians. A tiny four-period smoke check passed:

```text
Run id: gali_3param_profiled_direct_smoke_t4_20260525
direct at hmc_init log posterior: 60.394447; finite gradient=true
surrogate at theta_true log posterior: -15.060720; finite gradient=true
surrogate at prior_center log posterior: -10.780088; finite gradient=true
```

This confirms the profiled direct objective is numerically possible at very
small scale. It is far more expensive than the ROM-inversion direct audit,
because each log-posterior evaluation embeds direct SEP finite-difference shock
profiling.

The profiled direct check was then scaled to ten periods with direct-only
finite-gradient gating:

```text
Run id: gali_3param_profiled_direct_smoke_t10_20260525
direct at hmc_init log posterior: 150.338815; finite gradient=true
mean residual norm: 0.0153613
mean shock norm: 0.490949
max inversion iteration: 5
jacobian ranks: 3 in all ten periods
```

This is a stronger confirmation that the direct SEP profiled
measurement-error objective is numerically coherent on the controlled-ELB DGP.
It is still not a practical HMC target in the current finite-difference form:
the ten-period finite-gradient gate took several minutes, so a 40-period NUTS
chain would be prohibitive without analytic/AD derivatives or a much cheaper
direct SEP Jacobian.
