# Full 265-Period SEP Sensitivity Run Status

_Worker 3 status note. Last updated 2026-05-22 00:55 EDT._

## Purpose

The paper currently reports a bounded 40-period SEP solver-sensitivity pilot.
This production job reruns the existing `scripts/sep_sensitivity_study.jl`
design over the full 265-period extended sample, preserving the intended
posterior-draw stratification, tolerance/quadrature grid, SEP horizon, and
solver iteration settings.

## Exact Command

```bash
julia --project=. scripts/sep_sensitivity_study.jl \
  --n-draws=10 \
  --cap-draws=5 \
  --periods=265 \
  --sep-horizon=40 \
  --sep-maxit=200 \
  --out-dir=.local_artifacts/sep_sensitivity/full_265_20260521
```

Detached launch command:

```bash
screen -dmS sep265_20260521 /bin/zsh -lc 'cd <repo-root>; julia --project=. scripts/sep_sensitivity_study.jl --n-draws=10 --cap-draws=5 --periods=265 --sep-horizon=40 --sep-maxit=200 --out-dir=.local_artifacts/sep_sensitivity/full_265_20260521 > .local_artifacts/sep_sensitivity/logs/full_265_20260521.log 2>&1'
```

## Paths

- Screen session: `sep265_20260521`
- Log: `.local_artifacts/sep_sensitivity/logs/full_265_20260521.log`
- Artifact directory: `.local_artifacts/sep_sensitivity/full_265_20260521/`
- Expected summary on completion: `.local_artifacts/sep_sensitivity/full_265_20260521/SEP_SENSITIVITY_SUMMARY.md`
- Expected table on completion: `.local_artifacts/sep_sensitivity/full_265_20260521/sep_sensitivity_table.tex`
- Expected serialized payload on completion: `.local_artifacts/sep_sensitivity/full_265_20260521/sep_sensitivity_results.jls`

## Current Status

Completed on 2026-05-21 at 23:57 EDT. The `sep265_20260521` screen session has
exited and the expected summary, table, and serialized payload were written.
The run used the script's existing slow-run safety mechanism: the first
full-window cell took 1036.7 seconds, so the effective draw count was capped
from 10 to 5.

## Completed Results

- Effective draws: 5 posterior draws stratified by Mahalanobis distance.
- Window: 265 periods.
- Total wall clock: 40,331.5 seconds (about 11.2 hours).
- Convergence: all 6 tolerance/quadrature cells converged on all 5 draws.
- Mean periods solved: 265.0 in every cell.
- Mean max Euler residuals: about `9.7e-08` to `9.9e-08`.
- Acceptance tolerance: changing `accept_tol` within a fixed `K` did not change
  predictions at reported precision.
- Quadrature nodes: moving from `K=5` to `K=3` produced overall observation
  RMSE `1.08e-02` against the `(accept_tol=0.01, K=5)` reference, just above
  the script's `1e-2` stability screen.
- Largest per-variable RMSE at `K=3`: `labobs = 2.49e-02`; inflation is
  `pinfobs = 1.08e-02`.

The generated headline is therefore:

```text
SEP decomposition shows sensitivity to the (accept_tol, K) grid — see table for details.
```

This is not a failure of SEP convergence or tolerance stability. It is a mild
full-window quadrature-node sensitivity: `K=3` is just outside the pre-specified
screen relative to `K=5`, while all `K=5` tolerance cells are identical at
reported precision.

## Monitoring

```bash
screen -ls
tail -n 120 .local_artifacts/sep_sensitivity/logs/full_265_20260521.log
find .local_artifacts/sep_sensitivity/full_265_20260521 -maxdepth 1 -type f -print
```

## Interpretation Rules

- The full 265-period run can now be cited as completed, but it should not be
  described as a clean pass under the original `1e-2` RMSE screen because the
  `K=3` cells report RMSE `1.08e-02` versus the `K=5` reference.
- The paper should distinguish tolerance stability from quadrature sensitivity:
  `accept_tol ∈ {0.01, 0.1, 0.35}` is stable within a fixed `K`, while lowering
  quadrature nodes from 5 to 3 causes a small full-window prediction difference.
- The 40-period pilot remains useful context, but the full-window artifact
  supersedes the previous pending-status caveat.

## Risks

- The generated summary's final "Files" section uses the script's generic
  `.local_artifacts/sep_sensitivity/...` labels even though the actual full-run
  files live under `.local_artifacts/sep_sensitivity/full_265_20260521/`.
- The effective draw count is 5 because the script's cap was triggered. This
  should be reported transparently.
