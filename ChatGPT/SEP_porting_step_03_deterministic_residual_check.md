# SEP Porting Step 03: Deterministic Residuals Check (No Code Changes)

Goal
- Verify whether deterministic SEP in MacroModelling.jl currently evaluates nonlinear residuals (as Dynare does) or uses linearized residuals.

What I checked
- Searched `src/sep_solver.jl` for any use of `dynamic_resid`, `dynamic_g1`, or nonlinear residual calls.
- Reviewed the deterministic path solver: `solve_deterministic_path` in `src/sep_solver.jl`.
- Reviewed the main stochastic SEP loop in `sep_solve_mm!` to see if post-horizon (t > Lbr) becomes nonlinear.

Findings
1) Deterministic SEP is linearized
- `solve_deterministic_path` constructs residuals as:
  `r = ∇₊*Δy_fwd + ∇₀*Δy_cur + ∇₋*Δy_lag + ∇ₑ*ε_t`
- This is first-order (linearized) and does **not** call a nonlinear residual evaluator.
- No `dynamic_resid` or `dynamic_g1` calls exist in `src/sep_solver.jl`.

2) Stochastic SEP remains linearized at all horizons
- In `sep_solve_mm!`, the residual is always computed from `∇₊, ∇₀, ∇₋, ∇ₑ`.
- The “after look-ahead horizon” branch (`t > Lbr`) only drops shock integration; it does **not** switch to nonlinear equations.

Conclusion
- The current deterministic SEP path in MacroModelling.jl does **not** solve nonlinear residuals.
- The current stochastic SEP path also stays linearized throughout.
- This differs from Dynare’s `extended_path`, which solves nonlinear residuals even when order = 0.

Questions (Need your guidance)
1) Do you want deterministic SEP updated to evaluate nonlinear residuals (Dynare parity), or keep the linearized solver for now?
2) If updating to nonlinear, should we reuse MacroModelling’s internal dynamic residual/Jacobian pipeline, or call into the existing perfect-foresight solver machinery?

Files Reviewed
- `src/sep_solver.jl`

