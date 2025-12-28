# Step 18 - Drop t=1 lag Jacobian columns (Dynare alignment)

## Goal
Align the stochastic SEP Jacobian with Dynare's `ep_problem_1` behavior at the first period by excluding derivatives w.r.t. predetermined (t=0) variables. This should stabilize Newton and match Dynare's fixed initial conditions.

## Changes
- `src/sep_solver.jl`: For stochastic SEP, skip `:past` Jacobian contributions when `t == 1` for both branching and non-branching nodes. This mirrors Dynare's `i == 1` block where predetermined variables are not included in the Jacobian.

## Why this change
Dynare fixes initial conditions in SEP and excludes the lagged block from the Jacobian in the first period (`ep_problem_1.m`). Our solver kept lag columns, then manually zeroed the y0 update, which distorted Newton steps and prevented convergence.

## Tests
- `julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl`

## Results (post-change, before initial-state fix)
- SEP now converges (residual ~1e-9) and the tt path matches Dynare closely.
- Positive shock tt max abs error: ~0.013978% (Output).
- Baseline ts path still diverged after t=2 (max abs error for Output ~55.757662%).
- Negative shock run reused the cached positive solution because `solve!` did not re-run SEP when only deterministic shocks changed.

## Follow-ups identified
- Add a `sep_initial_state` option so the funnel baseline can pass the previous period state like Dynare.
- Force SEP re-solve when deterministic shocks or initial state changes to avoid cache reuse.
