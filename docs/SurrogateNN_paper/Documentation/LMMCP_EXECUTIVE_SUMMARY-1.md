# LMMCP Integration: Executive Summary

**Date**: January 19, 2026
**Question**: How difficult would it be to port LMMCP to MacroModelling?

---

## Answer: **3-4 weeks implementation, FEASIBLE**

### The Problem

MacroModelling.jl's SEP solver uses standard Newton method, which **does not enforce complementarity constraints**. This is why:
- THETA stays at ~1e-6 when BLIM < 0 (should spike)
- Your calibration showed 59% "binding" is just numerical noise
- Dynare works because it uses **LMMCP** - a specialized MCP solver

### The Solution

**Port LMMCP from Dynare to Julia**

**Why LMMCP?**
- ✅ **Proven**: Works in Dynare for your exact use case
- ✅ **Self-contained**: 626 lines of MATLAB, no dependencies
- ✅ **Licensed**: Unlimited use permission from original authors
- ✅ **Well-documented**: Published algorithm with user guide

**What it does differently**:
- Explicitly enforces: `THETA ≥ 0`, `BLIM ≥ 0`, `THETA · BLIM = 0`
- Uses Fischer-Burmeister NCP function for complementarity
- Levenberg-Marquardt with trust region and watchdog strategy
- Designed specifically for occasionally binding constraints

### Implementation Plan

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Week 1**: Port LMMCP core | 5 days | Julia LMMCP module working |
| **Week 2**: Test & validate | 5 days | Match Dynare on test cases |
| **Week 3**: SEP integration | 5 days | QMIPF solving with MCP |
| **Week 4**: Documentation | 5 days | Production ready |

**Total effort**: 120 hours (3-4 weeks)

### What You Get

**After Phase 1** (deterministic mode only):

```julia
# Specify bounds
var_bounds = (
    lb = fill(-Inf, length(m.var)),
    ub = fill(Inf, length(m.var))
)
var_bounds.lb[theta_idx] = 0.0  # THETA ≥ 0
var_bounds.lb[blim_idx] = 0.0   # BLIM ≥ 0

# Solve with MCP enforcement
solve!(m,
       algorithm=:stochastic_extended_path,
       sep_order=0,          # Deterministic mode
       sep_use_mcp=true,     # Enable MCP solver
       sep_var_bounds=var_bounds)

# Result: THETA spikes when BLIM hits 0 ✅
```

**Limitations accepted**:
- Only works with `sep_order=0` (deterministic/perfect foresight mode)
- Stochastic SEP (order > 0) requires additional work (Phase 2, optional)

### Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|------------|
| LMMCP translation bugs | Medium | Extensive testing against Dynare |
| Numerical instability | Medium | Parameter tuning, use Dynare defaults |
| QMIPF still doesn't work | Low | Fallback to PATHSolver.jl (+2 weeks) |
| Takes longer than 4 weeks | Medium | Phased approach with decision points |

**Overall risk**: **20-30% chance of failure** → fallback available

### Alternatives Considered

| Option | Time | Likely to Work | Why Not Recommended |
|--------|------|----------------|---------------------|
| **Port LMMCP** | 3-4 weeks | ✅ High | **RECOMMENDED** |
| PATHSolver.jl | 2-3 weeks | ✅ High | Licensing, binary deps |
| Complementarity.jl | 1-2 weeks | ⚠️ Medium | Less mature, may be slow |
| Keep first-order | 0 weeks | ✅ Works | ❌ You rejected this |
| Use Dynare | 1-2 weeks | ✅ Works | Parallel codebase, slow for large models |

### Decision Points

**Decision Point 1** (End of Week 2):
- ✅ LMMCP validated → Continue to integration
- ❌ Numerical issues → Switch to PATHSolver.jl

**Decision Point 2** (End of Week 3):
- ✅ QMIPF THETA spikes → Continue to production
- ❌ Still not enforcing → Debug or alternative

### Cost-Benefit

**Costs**:
- 3-4 weeks development time
- 20-30% risk of technical failure
- Ongoing maintenance (~2-5 hours/month)

**Benefits**:
- ✅ **Research unblocked**: QMIPF works with proper OBC
- ✅ **Nonlinear solution**: SEP with complementarity enforcement
- ✅ **Calibration possible**: Can target 3% binding frequency
- ✅ **Publication quality**: Results comparable to Dynare
- ✅ **Future research**: Enables broader class of constrained models
- ✅ **Community contribution**: First MCP solver in MacroModelling.jl

**ROI**: **High** - if your research requires nonlinear OBC enforcement

### Recommendation

**PROCEED with LMMCP porting** if:
1. ✅ You can commit 3-4 weeks full-time (or 6-8 weeks part-time)
2. ✅ Your research critically depends on proper OBC enforcement
3. ✅ Deterministic SEP (order=0) is acceptable for Phase 1
4. ✅ You're comfortable with 20-30% risk of technical failure

**ALTERNATIVE: Quick test first** (1 week):
1. Try Complementarity.jl (1 week implementation)
2. If it works → Save 2-3 weeks
3. If not → Fall back to LMMCP port (original plan)

### What I Need From You

**To proceed**, please confirm:

1. **Research priority**: Is proper OBC enforcement critical for your research goals?
2. **Time commitment**: Can you dedicate 3-4 weeks to this?
3. **Scope acceptance**: Is deterministic SEP (order=0) sufficient for Phase 1?
4. **Risk tolerance**: Accept 20-30% chance of needing fallback plan?

### Quick Start Option

**If you want to try immediately** (before full LMMCP port):

**Option A**: Test with Complementarity.jl (1 week)
- Lower barrier to entry
- May solve your problem faster
- If fails, still have LMMCP option

**Option B**: Implement LMMCP port (3-4 weeks)
- More likely to work
- Better performance
- Proven algorithm

---

## Bottom Line

**Question**: "How difficult would it be to port LMMCP to MacroModelling?"

**Answer**: **3-4 weeks, technically feasible, recommended to proceed**

**Key insight**: This is the proper way to solve your problem. You were right to want SEP to work - accepting first-order was the wrong direction. LMMCP integration gives you what Dynare has: proper complementarity enforcement in nonlinear perfect foresight solver.

---

**Next Action**: Your decision - proceed with LMMCP port, try Complementarity.jl first, or discuss alternative approaches?

