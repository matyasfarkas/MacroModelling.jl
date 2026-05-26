# Galí ELB ROM1-Residual Validation Plan

This validation isolates the question raised in the paper review: can a ROM1-residual surrogate recover unbiased parameter estimates when the nonlinear benchmark contains a hard occasionally binding constraint?

The maintained actual-floor stress-path runner is:

```bash
julia --project=. scripts/gali_obc_stochastic_comparison_plot.jl
```

The maintained local residual-grid validation runner is:

```bash
julia --project=. scripts/gali_obc_actual_floor_residual_grid_validation.jl
```

The maintained one-parameter inversion-grid validation runner is:

```bash
julia --project=. scripts/gali_obc_actual_floor_inversion_grid_validation.jl
```

The maintained two-parameter inversion-grid validation runner is:

```bash
julia --project=. scripts/gali_obc_actual_floor_twoparam_inversion_grid_validation.jl \
  --second-std-name=std_a \
  --second-shock-name=eps_a \
  --second-baseline=0.01 \
  --second-true=0.01 \
  --z-ident-period=12 \
  --z-ident-span=6 \
  --z-ident-shock=1.0 \
  --z-ident-pattern=alternating \
  --policy-shock-period=18 \
  --policy-shock-span=6 \
  --policy-shock=1.0 \
  --policy-shock-pattern=alternating
```

The consolidated package-level audit runner is:

```bash
julia --project=. scripts/gali_validation_package.jl
```

It checks the curated stress-path, residual-grid, inversion-grid, one-parameter HMC, two-parameter grid, extended two-parameter HMC, `std_nu` identification-probe, and direct-SEP smoke artifacts, then writes a Markdown report and TOML manifest under `.local_artifacts/gali_validation_package/`.

## Design

- Model: `models/Gali_2015_chapter_3_obc.jl`.
- Hard nonlinearity: the max operator in the Taylor-rule lower bound for `R`.
- Observables: `log_y`, `pi_ann`, and `i_ann`.
- First parameter block: `std_z` only. The current two-parameter extension estimates `std_z` and `std_a`; `std_nu` is retained as a diagnostic stress case because the profiled inversion objective pushes it to the local upper bound in these short samples.
- DGP: first-order OBC simulator with an adverse `eps_z[6:10]` block, no random background shocks, and the same standardized shocks for the OBC and linearized paths. The two-parameter run adds alternating post-ELB `eps_z[12:17]` and `eps_a[18:23]` blocks to identify `std_z` and `std_a` away from the binding window.
- Sign convention: a negative monetary-policy shock is expansionary in this Galí file, so the ELB recession diagnostic uses a positive `eps_z` shock instead.
- Surrogate target: observable residual `y_OBC - y_ROM1`, not a direct-level map.
- First posterior objective: known-state, known-shock measurement-error grid. This intentionally isolates the nonlinear transition approximation before adding inversion-filter shock recovery.
- Second posterior objective: linear ROM1 inversion recovers shocks from the OBC observations; the same recovered shocks are evaluated under direct OBC and under ROM1 plus the interpolated OBC-minus-ROM1 residual. This isolates the inversion-filter likelihood layer before adding HMC.

## Stages

1. `stress_plot`: clean 24-period adverse `eps_z` path. Output and inflation fall during the forced ELB window and the OBC policy rate stays at the floor for six periods.
2. `grid_smoke`: one-parameter local `std_z` grid with period-wise interpolation of the OBC-minus-linear residual path.
3. `inversion_grid`: one-parameter local `std_z` grid with ROM1 shock recovery and common-shock direct OBC versus residual-surrogate evaluation.
4. `hmc_smoke`: one-parameter direct OBC versus residual-surrogate HMC on the inversion-grid objective, using identical HMC seeds and a shallow NUTS setting.
5. `grid_dense_diagnostic`: denser local grids for sensitivity. These expose isolated MacroModelling OBC solver pathologies and are not paper-ready by default.
6. `two_parameter_grid`: two-parameter `std_z` and `std_a` ROM1-inversion grid with the same actual-floor path and separate alternating identification blocks.
7. `two_parameter_hmc`: matched direct-OBC and ROM1-residual surrogate HMC on the two-parameter profiled inversion objective.
8. Only after the one- and two-parameter stages pass, expand to the requested three-parameter matched HMC sampler.

## Acceptance Criteria

A stage passes only if:

- The actual OBC policy rate is at the floor for multiple periods and output/inflation move in the recessionary direction.
- Direct OBC and ROM1-residual surrogate grids have enough finite support.
- Direct and surrogate posterior means differ by less than the configured fraction of the direct posterior standard deviation.
- Direct and surrogate 90 percent intervals overlap.
- The true `std_z` lies in both 90 percent intervals.
- DGP, training-grid, holdout-midpoint, and posterior-grid solver warning counts are zero for paper-ready clean status.

The smoke stage is a local transition-validation exercise, not a matched direct SEP-HMC versus surrogate-HMC posterior comparison. The paper-stage run must replace known shocks with inversion-filter shocks and then run matched direct/surrogate posterior sampling.
The inversion-grid stage now covers the first half of that paper-stage requirement: the ROM1 inversion layer is active and the residual surrogate is compared with a direct OBC evaluation under common recovered shocks. The one-parameter HMC smoke then verifies that the same objective can be sampled by matched direct/surrogate NUTS without numerical errors. It still does not replace the requested three-parameter matched direct/surrogate HMC exercise.

The two-parameter grid and HMC stages are stronger transition/inversion diagnostics but still not the missing matched three-parameter direct-SEP HMC run. They confirm that the ROM1-residual surrogate can reproduce the direct OBC profiled objective when a second scale parameter is locally identified. A parallel `std_z`/`std_nu` diagnostic did not pass: direct and surrogate objectives agreed, but both pushed `std_nu` to the local upper grid boundary. That failure is treated as an objective-identification warning for the policy-shock scale, not as surrogate approximation error.

## Completed Follow-Up Runs

On 2026-05-18--2026-05-19, the clean two-parameter `std_z`/`std_a` matched-HMC design completed an extended run:

```text
.local_artifacts/gali_actual_floor_twoparam_inversion_grid/twoparam_inversion_hmc_balanced_extended_20260518/SUMMARY.md
```

The run uses the same 24-period actual-floor path, post-ELB alternating identification blocks, and direct/surrogate comparison as the acceptance run, but increases the sampler length to four chains with 500 warmup draws and 1,000 post-warmup draws per chain. The grid/training stage is clean: zero solver warnings, holdout-midpoint RMSE approximately zero, actual-floor periods `9 / 24`, and finite direct/surrogate support at `36 / 36` grid points. The matched HMC posterior comparison passes the planned criteria: direct and surrogate 90 percent intervals overlap for both parameters, both true values are covered, posterior-mean differences are `-0.4543` combined MCSE units for `std_z` and `-0.46` for `std_a`, surrogate post-warmup numerical errors are zero, and the generated overall validation flag is true. The direct sampler records six numerical-error proposals out of 4,000 post-warmup draws, so the strict zero-issue flag is false while the negligible-issue diagnostic is true. The run log is:

```text
.local_artifacts/gali_actual_floor_twoparam_inversion_grid/logs/twoparam_inversion_hmc_balanced_extended_20260518.log
```

Before scheduling the extended HMC run, a cheap `std_z`/`std_nu` amplitude sweep was rerun with `eps_nu` amplitudes 0.25, 0.5, 1.0, and 2.0. All four grid probes were numerically clean, with zero solver warnings and finite direct/surrogate support at all 36 grid points, but all pushed `std_nu` to the upper local grid edge and failed true-value coverage. This confirms that the `std_nu` issue is an inversion-objective identification problem in this short ELB design rather than a surrogate approximation problem, so no `std_z`/`std_nu` HMC run is scheduled until the DGP is redesigned.

## Current Clean Artifacts

Stress-path figure and summary:

```text
.local_artifacts/gali_elb_stochastic/gali_obc_eps_z_same_shocks_actualfloor_span5_shock0p8_bg0p0.png
.local_artifacts/gali_elb_stochastic/gali_obc_eps_z_same_shocks_actualfloor_span5_shock0p8_bg0p0_summary.md
```

The forced window has OBC output in `[-5.887, -2.389]`, inflation in `[-6.932, -3.035]`, and the policy rate fixed at `0.020` annualized percent.

Local residual-grid validation:

```text
.local_artifacts/gali_actual_floor_residual_grid/actual_floor_grid_stdz_interp_default_serial_20260518/SUMMARY.md
```

It has zero solver warnings, actual-floor periods `6 / 24`, direct mean `0.0499795`, surrogate mean `0.0499795`, overlapping 90 percent intervals, and true-value coverage for both objectives.

Inversion-grid validation:

```text
.local_artifacts/gali_actual_floor_inversion_grid/actual_floor_inversion_grid_default_20260518/SUMMARY.md
```

It has zero solver warnings, actual-floor periods `6 / 24`, finite direct and surrogate support at `18 / 18` grid points, direct mean `0.0500531`, surrogate mean `0.0500531`, mean difference `7.074e-09` direct posterior standard deviations, overlapping 90 percent intervals, and true-value coverage for both objectives.

One-parameter HMC smoke:

```text
.local_artifacts/gali_actual_floor_inversion_grid/actual_floor_inversion_hmc_tuned_commonseed_20260518/SUMMARY.md
```

It uses the 12-period smoke path, the same linear ROM1 inversion architecture, one chain per objective, 10 warmup draws, 30 post-warmup draws, identical HMC seeds, and shallow NUTS (`max_depth=1`). Both chains finish with zero numerical errors. Direct mean is `0.049654`, surrogate mean is `0.049716`, the mean difference is `0.2067` combined MCSE units, intervals overlap, and the true value is covered by both 90 percent intervals.

Two-parameter inversion-grid validation:

```text
.local_artifacts/gali_actual_floor_twoparam_inversion_grid/twoparam_inversion_grid_stda_balanced_T24_train3_probe_20260518/SUMMARY.md
```

It estimates `std_z` and `std_a` on a 24-period actual-floor path with post-ELB alternating identification blocks. The run has zero solver warnings, actual-floor periods `9 / 24`, finite direct and surrogate support at `36 / 36` grid points, direct and surrogate means of `0.0491347` for `std_z` and `0.0098216` for `std_a`, overlapping 90 percent intervals, and true-value coverage for both objectives.

Two-parameter matched HMC validation:

```text
.local_artifacts/gali_actual_floor_twoparam_inversion_grid/twoparam_inversion_hmc_balanced_extended_20260518/SUMMARY.md
```

This run uses four chains per objective, 500 warmup draws, and 1,000 post-warmup draws per chain. The grid/training stage is clean: zero solver warnings, holdout-midpoint RMSE approximately zero, actual-floor periods `9 / 24`, and finite direct/surrogate support at `36 / 36` grid points. The matched HMC posterior comparison passes the planned criteria: direct and surrogate 90 percent intervals overlap for both parameters, both true values are covered, posterior-mean differences are `-0.4543` combined MCSE units for `std_z` and `-0.46` for `std_a`, surrogate post-warmup numerical errors are zero, and the generated overall validation flag is true. The direct sampler records six numerical-error proposals out of 4,000 post-warmup draws, so the strict zero-issue flag is false while the negligible-issue diagnostic is true.
