# SEP Porting Step 13: Shock Std Mapping (Dynare-Style Fallback)

## Objective
Ensure shock scaling matches Dynare when `z_<shock>` parameters are missing, and remove the hard-coded `0.01` fallback that was distorting RBC SEP IRFs.

## Changes Applied
### 1) SEP solver shock covariance fallback
File: `src/sep_solver.jl`
- **Before**: missing `z_<shock>` parameter defaulted to std = 0.01.
- **After**: missing `z_<shock>` parameter defaults to std = 1.0 (variance = 1.0), consistent with Dynare’s `shocks; var shock = 1; end;` behavior.
- **Warning updated** to reflect default 1.0.

### 2) SEP simulation fallback
File: `src/sep_simulation.jl`
- **Before**: missing `z_<shock>` left Σ at zero for that shock (no variation).
- **After**: missing `z_<shock>` defaults to std = 1.0 with warning.

## Rationale
- Dynare RBC uses `var epsilon = 1` and scales the shock in the model equation via `sigma`.
- MacroModelling models often encode shock scale inside the equation, so `z_<shock>` may be absent.
- Defaulting to unit variance is safer and aligns with Dynare’s convention when explicit shock std parameters are missing.

## Remaining Question
- If we want to explicitly map missing `z_<shock>` to a **different** parameter (e.g., `sigma`), we need a rule to avoid double-scaling when the equation already includes that parameter. This is left as a follow-up decision once IRF replication is checked.
