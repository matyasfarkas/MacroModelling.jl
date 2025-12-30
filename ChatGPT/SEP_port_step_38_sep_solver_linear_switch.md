# SEP port step 38: SEP linear solver fallback + stiffness explanation

## What “stiff nonlinear block” means here
In this context, “stiff” means a subset of equations reacts extremely strongly to small changes in certain variables, while other equations react much more gently. This leads to:
- an ill‑conditioned Jacobian (some rows have much larger curvature than others),
- Newton steps that must be heavily damped to avoid violating domain constraints (e.g., raising a negative term to a non‑integer power),
- residual norms that plateau even as step sizes shrink.

In `Smets_Wouters_2007_HLT`, the utilization block (`afunc`, `afuncD`, `rk`) and the wage markup block (via `wdot` inside `xi`) are especially stiff because they include exponential and power terms. Under stochastic expectations, a small step can push `curvW + wdot` negative, which makes `((curvW + wdot)^(1+csigl))` undefined in real arithmetic. The solver dampens steps, but the residual floor remains.

## Change: optional linear solver switch on stall
Added an **optional fallback** to switch the Newton step from normal equations to a QR-based least squares solve when residuals stall.

### New SEP options
Available via `solve!(...; sep_*)` and propagated through `get_sep_irf(...; sep_*)`:
- `sep_linear_solver` (`:normal_equations` or `:qr`) — primary solver for the Newton step
- `sep_fallback_solver` (same symbols or `nothing`) — optional fallback when stalling
- `sep_stall_iters` — iterations without improvement before switching
- `sep_stall_rel_tol` / `sep_stall_abs_tol` — improvement thresholds

### Safety guard for fallback
When `active_solver == :qr`, the Newton step uses a **more conservative damping cap** (`α ≤ 0.2`) to reduce the risk of stepping into non‑finite regions (e.g., negative bases for fractional powers).

## Files changed
- `src/sep_solver.jl`
  - Added stall tracking and optional solver switch.
  - Added QR fallback and damping cap when active.
- `src/MacroModelling.jl`
  - Added new `sep_*` keywords to `solve!` and pass-through to `SEPSolverOptions`.
- `src/sep_irf.jl`
  - Added new `sep_*` keywords to `get_sep_irf` and `get_sep_irf_funnel`.
- `scripts/HLT_comparison.jl`
  - Restored higher precision (`sep_tol=5e-5`, `sep_maxit=1000`), no fallback enabled by default.

## Usage example
```
solve!(m;
       algorithm=:stochastic_extended_path,
       sep_linear_solver=:normal_equations,
       sep_fallback_solver=:qr,
       sep_stall_iters=25,
       sep_stall_rel_tol=1e-4,
       sep_stall_abs_tol=1e-10)
```

## Verification
- `julia --project=. scripts/HLT_comparison.jl` (high precision, no fallback)
  - PDF generated: `scripts/HLT_comparison_sep_irf.pdf`.
  - SEP still stalls around the residual floor (~0.0035), consistent with earlier diagnosis.

## Notes
- The fallback is **optional**; it is not enabled by default to avoid domain violations in stiff blocks.
- If you want to try QR fallback on HLT, I recommend a conservative `sep_stall_iters` and keeping the damping cap in place.
