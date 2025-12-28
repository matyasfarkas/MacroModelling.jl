# Stochastic Funnel IRF Validation - Status Report

**Date**: December 27, 2024
**Goal**: Validate MacroModelling.jl sparse tree SEP IRF computation against Dynare's stochastic funnel methodology

## Background

The user wants to replicate Dynare's IRF methodology from `rbc.mod` which uses a "stochastic funnel" approach:

1. **Shocked path (tt)**: Standard SEP with shock=-3 at t=1, order=10, 80 periods
2. **Baseline path (ts)**: Iterative construction with decreasing branching order:
   - Start at deterministic steady state
   - t=1: Solve order=10, shock=-3, 1 period → extract state
   - t=2: Solve order=9, shock=0, 1 period → extract state
   - ...
   - t=10: Solve order=1, shock=0, 1 period → extract state
   - t=11+: Solve order=0 (deterministic EP), shock=0, 80 periods

**IRF Computation** (from `spfirf.m` and `pdss.m`):
```matlab
pdss(data) = 100*(data-data(1))/data(1)  % Percentage deviation from initial value
IRF = pdss(tt) - pdss(ts)
```

Since both paths start at deterministic SS: `tt(1) = ts(1) = dss`, we have:
- `pdss(tt) = 100*(tt - dss)/dss`
- `pdss(ts) = 100*(ts - dss)/dss`
- `IRF = pdss(tt) - pdss(ts)`

**Example IRF values from benchmark** (period 1 after shock=-3σ):
- Output IRF: +12.63% (tt falls less than ts due to funnel uncertainty)
- Labour IRF: +21.09%
- Consumption IRF: -15.83% (tt falls more than ts)

## Progress

### ✅ Completed

1. **Sparse tree implementation validated**
   - `test_sparse_tree_rbc_validation.jl` confirms sparse vs full tree match perfectly
   - Maximum absolute difference: 0.00e+00
   - Steady states match Dynare exactly (Capital=17.77020407, etc.)

2. **Dynare fishbone algorithm fixed**
   - Changed `options_.ep.stochastic.algo = 3` to `algo = 1` in rbc.mod
   - Dynare now runs with Adjemian-Juillard fishbone method successfully

3. **Benchmark data collected**
   - Dynare IRF data saved to `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SEP/RBC_irf.csv`
   - Contains both `ts` and `tt` paths (91 periods each, 7 variables)
   - Verified benchmark starts at deterministic steady state

4. **Diagnostic framework created**
   - `test_stochastic_funnel_irf.jl` loads benchmark data
   - Loads RBC_Dynare model
   - Ready to implement funnel algorithm

### 🚧 In Progress

**Challenge**: MacroModelling.jl SEP solver API limitations

The current `solve!()` function doesn't support:
1. **Varying initial conditions**: Can't start from a specific state vector
2. **Short horizon solving**: Can't solve for just 1 period, extract state, then continue
3. **Dynamic branching order**: Can't easily vary `sep_order` between calls

**Current issue**: CSV parsing error
- Benchmark CSV has mixed String15/Missing types from headers
- Need to skip header rows properly

### ❌ Blocking Issues

**API Gap**: To implement stochastic funnel, we need:

```julia
# Desired API
current_state = dss  # Start at deterministic SS

for order in 10:-1:1
    # Solve for 1 period with specific initial state
    solve!(model, algorithm=:stochastic_extended_path,
           sep_periods=1,
           sep_order=order,
           sep_initial_state=current_state,  # NOT IMPLEMENTED
           sep_shocks=[shock_value])

    # Extract end state from solve
    current_state = extract_state_at_period(model, 1)  # NOT IMPLEMENTED
end

# Then deterministic continuation
solve!(model, algorithm=:stochastic_extended_path,
       sep_periods=80,
       sep_order=0,
       sep_initial_state=current_state)
```

**What's missing**:
1. `sep_initial_state` parameter in `solve!()`
2. Function to extract state vector at specific time from SEP solution
3. Ability to chain SEP solves with varying parameters

## Implementation Options

### Option A: Extend SEP Solver API (Recommended)

**Modify `src/MacroModelling.jl` solve!() function**:
- Add `sep_initial_state::Union{Vector{Float64}, Nothing} = nothing` parameter
- Pass through to `stochastic_extended_path()` function

**Modify `src/sep_solver.jl`**:
- Accept initial state parameter
- If provided, use it instead of deterministic steady state
- Add helper function to extract state at specific time:
  ```julia
  function extract_sep_state(model, period::Int)
      sep = model.solution.perturbation.stochastic_extended_path
      layout = sep.layout
      voff_t = layout.voff[period]
      return sep.Y[voff_t .+ (1:layout.ny_)]
  end
  ```

**Pros**:
- Clean, maintainable API
- Matches intended use pattern
- Reusable for future work (parameter grids, etc.)

**Cons**:
- Requires modifying core solver
- Need to test backward compatibility

### Option B: Direct Access to sep_solver.jl Internals

Call `stochastic_extended_path()` directly in a loop, manually managing:
- Y vectors
- Shock sequences
- Initial conditions

**Pros**:
- No API changes needed
- Works immediately

**Cons**:
- Fragile (depends on internal structure)
- Not reusable
- Hard to maintain

### Option C: Simple Validation Without Funnel

Just compare:
- MacroModelling.jl: Standard SEP IRF (shock at t=1, same order throughout)
- Dynare `tt` path: Shocked path from benchmark

**Pros**:
- Can validate core SEP algorithm immediately
- No API changes

**Cons**:
- Doesn't validate the funnel methodology
- Misses the user's specific request

## Recommendation

**Implement Option A** in two phases:

### Phase 1: Quick Validation (Now)
- Fix CSV parsing in `test_stochastic_funnel_irf.jl`
- Extract `tt` path from benchmark
- Solve MacroModelling.jl SEP with order=10, shock=-3 at t=1, 80 periods
- Compare MacroModelling `tt` vs Dynare `tt`
- **This validates the core SEP solver against Dynare**

### Phase 2: Full Funnel Implementation (Next)
- Extend `solve!()` API with `sep_initial_state` parameter
- Add `extract_sep_state()` helper function
- Implement iterative funnel construction
- Compare funnel paths MacroModelling vs Dynare
- **This validates the complete methodology**

## Files Created

1. `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/test_stochastic_funnel_irf.jl`
   - Diagnostic framework (needs CSV fix)
2. `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/SEP_IRF_METHODOLOGIES.md`
   - Documents three IRF approaches
3. `/Users/matyasfarkas/Documents/GitHub/MacroModelling.jl/test_sparse_tree_rbc_validation.jl`
   - ✅ Validates sparse vs full tree (PASSING)

## Next Immediate Steps

1. **Fix CSV parsing** (10 min)
   ```julia
   df = CSV.read(benchmark_path, DataFrame,
                 header=2,  # Skip first 2 rows
                 types=Dict(i => Float64 for i in 1:17))
   ```

2. **Simple tt-path validation** (30 min)
   - Solve MacroModelling SEP
   - Extract path
   - Compare with Dynare `tt`
   - Expected: Should match closely (both use same algorithm)

3. **Document findings** (15 min)
   - If paths match: ✅ Core SEP validated
   - If paths differ: Investigate (shock scaling? integration?)

4. **Decide on funnel implementation** (User decision)
   - If Option A: Implement API extensions
   - If Option C: Consider funnel validation complete

## Expected Outcomes

**If core SEP matches Dynare tt-path**:
- Validates shock scaling is correct
- Validates Gauss-Hermite quadrature matches Dynare
- Validates Newton solver behavior
- **Builds confidence in sparse tree implementation**

**If paths differ**:
- Investigate shock magnitude application
- Check integration algorithm (GH vs Unscented?)
- Check initial conditions (DSS vs SSS?)

## Key Insight from User

> "Let me clarify, Dynare does start at the DSS. It iterates forward by shortening the look ahead order."

This confirms:
- Both `tt` and `ts` start at **deterministic steady state**
- The funnel reduces order from 10→0 over first 11 periods
- After t=11, uses deterministic EP (order=0) for remaining periods
- **IRF = difference between two complete solution paths**

## Context

This validation is critical because:
1. Sparse tree reduces nodes from 2,187 (full tree, H=7) to 15 (sparse)
2. Must ensure sparse approximation matches full tree
3. Must ensure MacroModelling matches Dynare reference
4. SW07 model will rely on this for 7-shock analysis
