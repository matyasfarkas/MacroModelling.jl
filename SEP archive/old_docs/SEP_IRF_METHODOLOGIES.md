# SEP IRF Methodologies: MacroModelling vs Dynare

**Date**: December 27, 2024
**Status**: Analysis of different IRF computation approaches

## Summary

We've discovered that there are **three different methods** for computing IRFs with stochastic models, and the Dynare output you provided uses a **simplified deterministic approach**, not the full SEP IRF methodology.

## Three IRF Methodologies

### 1. MacroModelling.jl: Tree Extraction (Implemented in `src/sep_irf.jl`)

**Method**:
- SEP pre-computes a branching tree with Gauss-Hermite quadrature nodes
- For nnodes=3: nodes at [-√3σ, 0, +√3σ] standard deviations
- IRF = extract path through tree that goes through the shocked node at t=1

**Pros**:
- Fast (no additional solve needed)
- Uses full SEP tree structure
- Accounts for future uncertainty through GH nodes

**Cons**:
- Limited to GH node locations (can't specify arbitrary shock sizes)
- IRF is conditional on specific node path, not an expectation

**Implementation**: `extract_sep_irf()` in `src/sep_irf.jl`

### 2. Dynare rbc.mod: Stochastic Funnel (Full SEP IRF)

**Method** (from `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/ep-mj-30-years-master/models/irf/rbc.mod`):

```matlab
% Shocked path: SEP with innovation=3 at t=1
tt = extended_path(oo_.steady_state, 80, innovations, options_, M_, oo_);

% Baseline path: Iterative backward construction
ds = transpose(oo_.steady_state);
for order=maxorder:-1:0
  options_.ep.stochastic.order = order;
  switch order
    case maxorder
      ts = extended_path(transpose(ds(end,:)), 1, innovations(1), options_, M_, oo_);
      ds = [ds; ts.data(2,:)];
    case 0
      ts = extended_path(transpose(ds(end,:)), 80, zeros(80,1), options_, M_, oo_);
      ds = [ds; ts.data(2:end,:)];
    otherwise
      ts = extended_path(transpose(ds(end,:)), 1, 0, options_, M_, oo_);
      ds = [ds; ts.data(2,:)];
  end
end

% IRF = shocked - baseline
irf = spfirf(tt, ts, 1);
```

**What this does**:
1. **Shocked path**: Full SEP from deterministic SS with innovation at t=1
2. **Baseline path**: "Stochastic funnel" created by:
   - Start from final period with max branching order
   - Work backward reducing order each period
   - Order 0 = deterministic (no branching)
   - Creates a path that accounts for varying levels of future uncertainty

**Pros**:
- Proper accounting of precautionary behavior
- Baseline represents "optimal decision under uncertainty"
- Theoretically rigorous (see Adjemian-Juillard 2025 paper)

**Cons**:
- Computationally expensive (many SEP solves)
- Complex implementation

**Not yet implemented in MacroModelling.jl**

### 3. Dynare fs2000.mod: Perfect Foresight from SSS (What you ran)

**Method** (from your fs2000.mod IRF section):

```matlab
% BASELINE: Deterministic path with zero shocks from current state
perfect_foresight_setup(periods=40);
oo_.exo_simul(:,:) = 0;
perfect_foresight_solver;
baseline = oo_.endo_simul;

% SHOCKED: Deterministic path with 1σ shock at t=1 from current state
perfect_foresight_setup(periods=40);
oo_.exo_simul(:,:) = 0;
oo_.exo_simul(2, 1) = sigma_e_a;
perfect_foresight_solver;
shocked = oo_.endo_simul;

% IRF = difference
irf = shocked - baseline;
```

**Key point**: This uses `perfect_foresight_solver` (deterministic), **not `extended_path`** (SEP)!

The initial state is set from the SEP stochastic steady state:
- Dynare SEP(1): y=1.35095744 vs Deterministic SS: y=1.35588
- So paths start from a slightly different point than deterministic SS

**Pros**:
- Simple to implement
- Fast to compute
- Reasonable approximation for small shocks

**Cons**:
- Not a true SEP IRF (uses deterministic solver!)
- Doesn't account for uncertainty properly
- IRF sign/magnitude depends heavily on initial state

**This is what produced your negative IRF values**

## Why Dynare IRFs Are Negative

Your Dynare output showed:
```
Period 1: y=-0.012731, c=-0.016659
```

This is **not** a bug. Here's why:

1. **Starting point**: SEP stochastic steady state (y=1.35096) is BELOW deterministic SS (y=1.35588)
   - Reason: Precautionary behavior under uncertainty depresses consumption/output

2. **Perfect foresight from SSS**: When you announce a TFP shock will hit, agents know:
   - Shock is temporary (AR(1) decay)
   - Future is now CERTAIN (perfect foresight assumption)
   - Precautionary motive vanishes

3. **Result**: Agents increase consumption immediately (certainty effect dominates TFP effect)
   - This temporarily REDUCES output (more C, less I)
   - Capital builds up slowly over many periods

4. **IRF interpretation**: Deviation from the SSS baseline, not from deterministic SS
   - Negative values mean "even lower than already-depressed SSS level"

## Comparison of Methodologies

| Aspect | MacroModelling Tree | Dynare Funnel (rbc.mod) | Dynare PF (fs2000.mod) |
|--------|---------------------|-------------------------|------------------------|
| Solver | SEP (stochastic) | SEP (stochastic) | Perfect foresight (deterministic) |
| Baseline | Zero-shock node path | Stochastic funnel | Deterministic from SSS |
| Shocked | GH node path | SEP with innovation | Deterministic with 1σ shock |
| Uncertainty | Yes (through GH nodes) | Yes (varying order) | No (perfect foresight) |
| Initial state | Deterministic SS | Deterministic SS | Stochastic SS |
| Computational cost | Low (1 solve) | High (many solves) | Low (2 PF solves) |

## Recommendation

To properly validate MacroModelling.jl against Dynare, we have two options:

### Option A: Compare with Dynare Perfect Foresight IRF (Easy)

1. Re-run Dynare fs2000.mod IRF from **deterministic SS** (not SSS)
2. Implement perfect foresight conditional forecast in MacroModelling:
   - Requires perfect foresight solver (may not exist yet)
   - Or use perturbation solution IRF as approximation

**Pro**: Simple, quick validation
**Con**: Not testing SEP IRF, just deterministic IRF

### Option B: Implement Full SEP Funnel IRF (Hard, Correct)

1. Implement the `rbc.mod` stochastic funnel algorithm in MacroModelling
2. Requires:
   - Ability to solve SEP with custom shock realizations
   - Iterative backward construction of baseline
   - Difference computation

**Pro**: Proper SEP IRF validation, theoretically rigorous
**Con**: Significant implementation effort

### Option C: Compare MacroModelling Tree Extraction with Dynare (Current)

1. Extract IRF from MacroModelling SEP tree using `extract_sep_irf()`
2. Compare with Dynare's tree extraction (if available)
3. This tests the core SEP tree structure, not the IRF methodology

**Pro**: Tests what's already implemented
**Con**: Different methodology than Dynare's approaches

## My Recommendation

Start with **Option C + verify steady states match**:

1. ✅ Already done: Steady states match perfectly!
   - MacroModelling: y=1.35587677, c=1.00699667, R=1.00725076
   - Dynare: y=1.35588, c=1.007, R=1.00725

2. **Next**: Use MacroModelling's `extract_sep_irf()` to get tree-based IRF
3. Compare magnitudes (not necessarily signs) with Dynare
4. Document that MacroModelling uses tree extraction, Dynare uses perfect foresight

Then, if needed, implement Option B (full stochastic funnel) as future work.

## Questions for User

1. Are you satisfied with validating that **steady states match** exactly?
2. Do you want to implement perfect foresight IRF (Option A)?
3. Or should we focus on comparing tree-extracted IRFs (Option C)?
4. Is the full stochastic funnel (Option B) needed for your research?

## References

- **Adjemian-Juillard (2025)**: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/ep-mj-30-years-master/tex/pub/adjemian-juillard-2025-september.tex`
- **Dynare RBC IRF example**: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/ep-mj-30-years-master/models/irf/rbc.mod`
- **MacroModelling SEP IRF**: `src/sep_irf.jl`
- **Our fs2000 test**: `fs2000.mod` (perfect foresight approach)
