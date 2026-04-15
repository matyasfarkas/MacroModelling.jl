# HLT 3-Parameter Validation Runbook

This runbook standardizes the first HLT switching-estimator validation milestone:

- Operational estimator: `ROM + NN surrogate` switching
- FOM benchmark: direct `SEP` likelihood evaluation (outside the operational estimator)
- Parameter scope: `legacy_3params` (`cprobp`, `cindp`, `curvp`)

## Prerequisites

- Repository root: `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl`
- Julia environment instantiated for this project
- Sufficient local disk space for `.local_artifacts/`

## Outputs

The harness writes a timestamped run folder under:

- `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/.local_artifacts/hlt_validation_runs/`

Each run contains:

- `manifests/run_manifest.toml`
- `manifests/RUN_SUMMARY.md`
- Dataset, surrogate, synthetic, gate calibration, estimation chain outputs
- Optional FOM benchmark payload + summary

## Smoke Run (default milestone smoke test)

```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=smoke \
  --use-obc=true \
  --samples=150 \
  --chains=1
```

This uses deterministic output paths inside one run directory and fixed defaults for:

- `legacy_3params` dataset generation
- ROM1 residual surrogate training
- HLT synthetic OBC data with injected volatility window
- Gate calibration
- Hard-gate switching estimation
- FOM benchmark utility (allow-fail mode)

## Quick Smoke With Tuned Direct SEP FOM (experimental, slower)

Use this when you want the quick-smoke harness to attempt the tuned bounded stochastic-SEP direct FOM benchmark that has produced finite HLT/OBC values on gated subsets.

```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=smoke \
  --quick-smoke=true \
  --quick-smoke-fom-preset=direct_sep_gated_smoke_order1_tuned \
  --require-direct-fom-ok=true \
  --use-obc=true
```

Notes:

- This is slower than the default quick-smoke FOM step.
- It uses a bounded `gated_block` direct-SEP benchmark with `context` periods and tuned stochastic-SEP settings.
- It is intended for direct-FOM smoke validation, not full milestone acceptance evidence.
- `--require-direct-fom-ok=true` makes the harness fail if the FOM payload has no successful direct SEP result (deterministic recovery-ladder success counts).

## Direct SEP Floor Recovery (deterministic ladder)

The FOM benchmark supports a deterministic SEP floor-recovery ladder:

- `--recovery-ladder=true`

This is designed for recurring HLT/OBC direct `SEP + inversion` floor failures (`on_failure_loglikelihood`) and records:

- base attempt failure classification (for example `invalid_logabsdet`)
- SEP inversion diagnostics (failing period / iteration / SEP error when available)
- fixed recovery rung sequence and the rung that succeeded (if any)

Use this to avoid ad-hoc tuning and to produce reproducible failure-to-recovery artifacts.

## Acceptance Smoke (switching + recovery + FOM-vs-ROM1 fit)

Use the acceptance smoke after a successful strict quick-smoke run (or any completed HLT synthetic estimation run directory).

This script checks:

- switching actually occurs (`0 < gate_share < 1`)
- gate overlaps the injected volatility window
- posterior mean recovers the 3 synthetic truth parameters within smoke tolerances
- direct SEP has better fit than ROM1 on a short HLT microcase built from the same shock pattern around the injected volatility window

```bash
julia --project=. scripts/hlt_sep_surrogate_acceptance_smoke.jl \
  .local_artifacts/hlt_validation_runs/hlt3_YYYYMMDD_HHMMSS
```

Notes:

- The fit comparison uses a short direct-SEP microcase (`SEP`-generated observations) centered on the synthetic volatility episode for speed and robustness.
- The ROM1 comparator is simulated with `MacroModelling.get_irf(...; algorithm=:first_order)` on the non-OBC HLT model as a linear baseline.
- The optional inversion-benchmark comparison panel is disabled by default; enable it with `--run-inversion-benchmark-panel=true` for additional diagnostics.

## Benchmark Run (heavier, primary evidence)

```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=benchmark \
  --use-obc=true \
  --samples=1000 \
  --chains=4
```

## Fast Benchmark Reuse Modes (for iterative validation)

If dataset/surrogate/synthetic artifacts already exist in a prior run directory, skip rebuild:

```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=benchmark \
  --benchmark-profile=bounded \
  --benchmark-skip-build=true \
  --run-dir=.local_artifacts/hlt_validation_runs/hlt3_YYYYMMDD_HHMMSS \
  --samples=30 \
  --chains=1 \
  --require-direct-fom-ok=true \
  --use-obc=true
```

If you only want to re-check direct SEP FOM (no new switching estimation), reuse the existing chain too:

```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=benchmark \
  --benchmark-profile=bounded \
  --benchmark-skip-build=true \
  --benchmark-skip-estimation=true \
  --run-dir=.local_artifacts/hlt_validation_runs/hlt3_YYYYMMDD_HHMMSS \
  --require-direct-fom-ok=true \
  --use-obc=true
```

Input requirements for reuse:

- `dataset/hlt_sep_surrogate_trained.jls`
- `synthetic/hlt_sep_synth_data.jls`
- `synthetic/gate_calibration.jls`
- `synthetic/hlt_sep_surrogate_estimation_chain.jls` (required only with `--benchmark-skip-estimation=true`)

## Dry Run (generate manifest without execution)

```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=smoke \
  --dry-run=true
```

Use this to verify commands and output locations before launching long runs.

## Direct FOM Benchmark Re-run (standalone)

```bash
julia --project=. scripts/hlt_sep_surrogate_fom_benchmark.jl \
  <chain.jls> <hlt_sep_synth_data.jls> \
  --filter=inversion \
  --recovery-ladder=true \
  --allow-fail=true
```

Notes:

- The FOM script attempts direct `SEP` likelihood evaluation via `get_loglikelihood(..., algorithm=:stochastic_extended_path)`.
- If direct SEP likelihood evaluation fails in the current environment/model configuration, the script records the failure and still emits a benchmark payload/summary when `--allow-fail=true`.

### Tuned Bounded Direct SEP Re-run (gated block, stochastic SEP)

This preset reproduces the tuned bounded stochastic-SEP smoke configuration used to obtain finite HLT/OBC direct FOM values on a gated nonlinear episode.

```bash
julia --project=. scripts/hlt_sep_surrogate_fom_benchmark.jl \
  <chain.jls> <hlt_sep_synth_data.jls> \
  --benchmark-preset=direct_sep_gated_smoke_order1_tuned \
  --period-selection=gated_block \
  --gated-block=first \
  --use-obc \
  --recovery-ladder=true \
  --allow-fail=true
```

Optional overrides for additional probes:

- `--gated-block=last`
- `--max-periods=2`
- `--context-periods=1`

## Recommended Validation Sequence

1. Run `test/test_hlt_obc_sep.jl` (hard and smooth modes) to confirm SEP prerequisites.
2. Run a harness dry-run and inspect the generated manifest.
3. Run smoke validation.
4. Inspect `RUN_SUMMARY.md`, chain payload, and FOM summary.
5. Promote to benchmark run only after smoke succeeds.
