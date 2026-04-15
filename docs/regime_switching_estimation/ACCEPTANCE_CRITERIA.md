# HLT Switching Estimator Acceptance Criteria (3-Parameter Milestone)

## Scope

This document defines the first acceptance gate for the consolidated HLT switching-estimator pipeline:

- Model: HLT (`Smets_Wouters_2007_HLT` / `..._obc`)
- Estimator: switching `ROM + NN surrogate`
- FOM benchmark: direct `SEP`
- Parameter scope: `legacy_3params`

## Consolidation Acceptance

1. Provenance audit exists and is reproducible:
   - `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/consolidation/PROVENANCE_MAP.md`
   - `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/consolidation/MIGRATION_DECISION_MATRIX.md`
2. Staging bundles are externalized from the repo root and indexed:
   - `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/docs/consolidation/STAGING_EXTERNALIZATION_MANIFEST.md`
3. Local artifact policy is active (`.local_artifacts/` is gitignored).

## Functional Acceptance (Smoke)

1. HLT validation harness dry-run completes and writes a manifest.
2. Smoke run completes end-to-end without manual intervention.
3. Switching estimation writes a chain payload with finite regime and linear log-likelihood summaries.
4. Gate calibration produces a nontrivial nonlinear regime share and a saved `gate_calibration.jls`.

## Accuracy / Recovery Acceptance (Primary KPI)

Default thresholds for the 3-parameter milestone:

- `|cprobp error| <= 0.10`
- `|cindp error| <= 0.15`
- `|curvp error| <= 20`

Required evidence:

1. Posterior summaries improve over the prior/baseline center for all 3 parameters.
2. Truth lies inside the reported credible interval for at least `2/3` parameters.

## Speedup Acceptance (Primary KPI)

1. Switching estimator benchmark runtime is at least `2x` faster than the direct SEP/FOM benchmark protocol on the documented theta evaluation workload.
2. Benchmark summary must include:
   - wall-clock runtime(s)
   - algorithm/filter choices
   - workload definition (theta panel)

## FOM Fidelity Acceptance

1. Switching-vs-FOM log-likelihood differences are reported for each evaluated theta point.
2. If direct SEP FOM evaluation fails in the current environment, the failure must be recorded and the milestone remains **infrastructure-complete but not FOM-validated**.
3. Full milestone validation requires at least one successful direct FOM comparison on:
   - `true theta`
   - `post_mean theta` (if available)
4. Bounded `gated_block` direct-SEP benchmarks are acceptable as interim evidence when full-sample direct SEP is runtime-prohibitive, provided the benchmark payload documents:
   - selected/evaluation/context periods
   - effective presample periods
   - SEP + inversion settings (including any tuned smoke preset)
5. If a direct SEP floor event occurs (`on_failure_loglikelihood`), the benchmark artifact must record a failure classification (when available) and any deterministic recovery rung used.
6. For strict smoke validation, the harness may be run with `--require-direct-fom-ok=true`, which requires at least one successful direct SEP FOM result in the FOM payload.
7. The optional acceptance smoke script (`scripts/hlt_sep_surrogate_acceptance_smoke.jl`) is a compact evidence check that requires:
   - nondegenerate switching with volatility-window overlap
   - posterior-mean recovery of the 3 synthetic truth parameters within smoke tolerances
   - direct SEP better fit than a ROM1 linear baseline on the short HLT volatility-window microcase (`SEP`-generated observations vs `ROM1` simulated baseline)

## Reproducibility Acceptance

1. Re-running the harness with the same seed and config reproduces deterministic artifacts (manifests, command lines, paths).
2. Deterministic components (gate calibration on fixed inputs, API unit tests) reproduce exactly.
3. MCMC summary differences, if any, are documented with seed/chain settings.
