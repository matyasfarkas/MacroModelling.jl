# HLT Pipeline Verification

Date: 2026-05-26

This note records the current-runtime verification pass for the HLT/Smets-Wouters surrogate pipeline.

## What Was Verified

- Fast API and harness checks pass:
  - `julia --project=. test/test_regime_switching_api.jl`
  - `julia --project=. test/test_hlt_validation_harness.jl`
  - `julia --project=. test/test_hlt_acceptance_smoke.jl`
- HLT model payload checks pass:
  - `julia --project=. test/test_hlt_obc_sep.jl`
  - `julia --project=. test/test_hlt_real_data_payload.jl`
- A fresh current-runtime HLT quick-smoke pipeline completes end to end:
  - Run directory: `.local_artifacts/hlt_pipeline_verification_20260525/hlt3_quick_smoke_tuned_run`
  - Dataset generation: ok, 8 finite SEP residual samples, median residual about `2.4e-10`, max residual about `9.1e-7`.
  - Surrogate training: ok, ROM1-residual MLP trained and serialized.
  - Synthetic data generation: ok, 8 requested periods generated on first attempt.
  - Gate calibration: ok, gate share `0.25`.
  - Switching estimation: ok, 1 chain with 5 draws.
  - Acceptance smoke: ok, truth-shock direct SEP RMSE `0.0` versus ROM1 RMSE `0.4436` on the fit window.

## Fixes Made

- `scripts/hlt_surrogate/hlt_sep_surrogate_cli_utils.jl` now treats bare boolean flags such as `--dry-run`, `--quick-smoke`, and `--skip-fom` as `true`.
- `test/test_hlt_validation_harness.jl` adds coverage for bare boolean flags.
- `scripts/hlt_sep_surrogate_validate_hlt3.jl` quick-smoke SEP settings now match the maintained HLT OBC SEP smoke settings more closely:
  - `sep_horizon=12`
  - `sep_maxit=100`
  - `sep_tol=1e-6`
  - `sep_shock_scale=0.5`
  - `sep_accept_tol=0.25`
  - lower structural shock scales for the tiny smoke run.

## Direct SEP Versus Surrogate Probe

The fresh current-runtime artifacts remove the stale surrogate deserialization fallback.

- Probe directory: `.local_artifacts/hlt_pipeline_verification_20260525/direct_sep_surrogate_tuned_probe`
- Summary: `.local_artifacts/hlt_pipeline_verification_20260525/direct_sep_surrogate_tuned_probe/HLT_DIRECT_SEP_SURROGATE_VALIDATION_SUMMARY.md`
- Status: `smoke_only_production_run_needed`
- Surrogate source: `surrogate_mh_same_window`
- Direct SEP elapsed time: about 69 seconds for one one-period gated-window MH draw.

This is not a production posterior validation. It verifies that direct SEP and the fresh surrogate objective can both be invoked on the same current-runtime HLT artifact window, but the comparison uses only one draw and should not be cited as evidence of posterior equivalence.

## Important Limitations

- The old default HLT validation artifact at `.local_artifacts/hlt_validation_runs/hlt3_20260302_153555` is stale under the current Julia runtime. Its surrogate bundle fails deserialization with `MethodError: Cannot convert Int64 to String`, forcing fallback to an existing surrogate chain payload.
- The first current quick-smoke attempt failed under the previous quick-smoke settings because random OBC SEP dataset draws repeatedly hit the 80-iteration cap. The tuned settings above are required for a reliable smoke.
- The fresh quick-smoke chain is deliberately tiny and numerically uninformative. It is a wiring and solver-regression check, not empirical evidence.
- The HLT direct SEP comparison remains too expensive for a meaningful posterior validation at this scale. A submission-grade HLT posterior comparison would require a larger run design and a runtime budget measured in many hours or days.
- The paper should continue to rely on the Galí direct-SEP validation for posterior-equivalence evidence, and should describe HLT direct-SEP checks as smoke/provenance unless a larger HLT run is completed.

## Current Assessment

The maintained HLT pipeline is operational after the CLI and quick-smoke setting fixes. The following pieces are now verified on current code: OBC SEP solve, real-data payload construction, surrogate dataset generation, ROM1-residual surrogate training, synthetic OBC data generation, gate calibration, switching estimation, acceptance smoke, and a one-draw direct SEP versus surrogate objective probe on fresh artifacts.

The HLT pipeline is not yet verified as a full empirical posterior validation. The remaining blocker is scale: the direct SEP HLT objective is expensive, and the existing direct comparison is smoke-level only.
