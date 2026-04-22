# OBC Investigation: Current Status

**Date**: January 19, 2026
**Status**: ✅ **ROOT CAUSE IDENTIFIED** → 🔄 **SOLUTION IN PROGRESS**

---

## Executive Summary

### What We Discovered

After extensive investigation:

1. ✅ **OBC implementation is correct** (matches Dynare specification)
2. ✅ **Model compiles and runs** (180 vars, 105 OBC auxiliaries)
3. ✅ **SEP infrastructure works** (no more e26 errors)
4. ✅ **Root cause identified**: SEP lacks MCP solver for complementarity enforcement
5. ✅ **Path forward is clear**: Port LMMCP from Dynare to MacroModelling.jl

### User's Research Goal (Clarified)

**User wants**: Global nonlinear solution with proper OBC enforcement in SEP
**User does NOT want**: First-order linearization (loses nonlinearity)

**This changes everything** - we need to make SEP work, not recommend alternatives.

---

## Technical Findings

### Why SEP Doesn't Enforce OBC

**MacroModelling's SEP**:
- Uses standard Newton/Levenberg-Marquardt solver
- Treats `F(X) = 0` as unconstrained system
- Box constraints (`THETA ≥ 0`, `BLIM ≥ 0`) not enforced during solve
- `max()` operators may be smoothed or linearized

**Dynare's perfect_foresight**:
- Uses **LMMCP** (Levenberg-Marquardt Mixed Complementarity Problem)
- Explicitly enforces complementarity: `THETA · BLIM = 0`
- Box constraints active throughout solve
- Fischer-Burmeister NCP function handles complementarity

**This is a fundamental solver difference**, not a parameter tuning issue.

### Evidence

1. **Penalty method tests** (January 15):
   - penalty_kappa = 0.01 → THETA ≈ 1e-6 (should be O(0.01))
   - penalty_kappa = 0.1 → THETA ≈ 1e-6 (no change!)
   - penalty_kappa = 1.0 → Still ~1e-6
   - **Conclusion**: Increasing penalty has no effect

2. **User's calibration** (January 16-18, documented in ChatGP_QMIPF):
   - 10,000 period simulation
   - "59% binding" but THETA range: -3.35e-6 to 5.16e-6
   - BLIM always positive: 7.11 to 7.14
   - **Conclusion**: "numerical oscillations around zero rather than true binding"

3. **Native OBC system** (January 19):
   - Successfully activated (180 vars, 105 OBC auxiliaries)
   - Designed for perturbation methods, not perfect foresight
   - **Conclusion**: Wrong tool for SEP

### What Doesn't Work

❌ **First-order approximation** - User rejected (wants nonlinear solution)
❌ **Penalty method** - Tested extensively, doesn't enforce
❌ **Native OBC system** - For perturbation methods, not SEP
❌ **Parameter tuning** - Not a tuning issue, it's a solver limitation
❌ **Smooth penalties** - Already tried, approximates but doesn't enforce

---

## Solution: Port LMMCP to MacroModelling.jl

### Feasibility Assessment

**Full analysis**: See `LMMCP_PORTING_FEASIBILITY_ASSESSMENT.md`
**Quick summary**: See `LMMCP_EXECUTIVE_SUMMARY.md`

**Bottom line**: **3-4 weeks implementation, technically feasible**

### What LMMCP Provides

✅ **Explicit complementarity**: `THETA ≥ 0, BLIM ≥ 0, THETA · BLIM = 0`
✅ **Box constraint enforcement**: Bounds active during solve
✅ **Proven algorithm**: Works in Dynare for this exact problem
✅ **Self-contained**: No external dependencies
✅ **Licensed**: Unlimited use permission

### Implementation Plan

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Week 1**: Port LMMCP core | 5 days | Julia LMMCP module |
| **Week 2**: Test & validate | 5 days | Match Dynare results |
| **Week 3**: SEP integration | 5 days | QMIPF with MCP |
| **Week 4**: Documentation | 5 days | Production ready |

**Total**: 120 hours (3-4 weeks full-time)

### Decision Points

**End of Week 2**: LMMCP validated against Dynare?
- ✅ Yes → Continue to integration
- ❌ No → Switch to PATHSolver.jl (fallback)

**End of Week 3**: QMIPF THETA spikes when BLIM < 0?
- ✅ Yes → Continue to production
- ❌ No → Debug or alternative

### Risk Assessment

**Technical risk**: 20-30% chance of failure
**Mitigation**: Fallback to PATHSolver.jl or Complementarity.jl

**Risks**:
- MATLAB→Julia translation bugs
- Numerical instability
- Integration issues with MacroModelling
- Performance problems

**All have mitigations** detailed in feasibility assessment.

---

## Alternative Approaches Considered

### 1. Use Dynare for OBC analysis

**Status**: ✅ Viable fallback
**Effort**: 1-2 weeks to port QMIPF to Dynare .mod
**Pros**: Proven to work, mature
**Cons**: Parallel codebase, slower, different ecosystem
**Verdict**: Keep as Plan B if LMMCP fails

### 2. PATHSolver.jl

**Status**: ✅ Strong alternative
**Effort**: 2-3 weeks integration
**Pros**: Industry standard, fast, robust
**Cons**: Binary dependencies, PATH license required
**Verdict**: Primary fallback if LMMCP fails

### 3. Complementarity.jl

**Status**: ⚠️ Quick test option
**Effort**: 1 week
**Pros**: Pure Julia, easy to try
**Cons**: Less mature, may be slow
**Verdict**: Could try this first (1 week) before LMMCP

### 4. First-order approximation

**Status**: ❌ User rejected
**Reason**: Loses nonlinearity, defeats purpose of SEP
**Verdict**: Not acceptable for user's research goals

---

## Files and Documentation

### Investigation Documents (Completed)

1. **SEP_OBC_DIAGNOSIS.md** - Initial technical analysis
2. **SEP_SUCCESS_SUMMARY.md** - Memory fix success report
3. **OBC_SEP_FINAL_STATUS.md** - Status before native OBC attempt
4. **OBC_PROPER_IMPLEMENTATION.md** - Native OBC approach documentation
5. **DYNARE_MCP_VS_MACROMODELLING.md** - Detailed solver comparison
6. **FINAL_OBC_INVESTIGATION_COMPLETE.md** - Investigation summary (superseded)

### New Documents (This Session)

7. **LMMCP_PORTING_FEASIBILITY_ASSESSMENT.md** - Comprehensive 9-part analysis
   - Part 1: LMMCP code analysis (626 lines MATLAB)
   - Part 2: MacroModelling SEP architecture
   - Part 3: Implementation strategy (phased approach)
   - Part 4: Technical challenges
   - Part 5: Alternative approaches
   - Part 6: Recommendations
   - Part 7: Implementation timeline (4 weeks)
   - Part 8: Cost-benefit analysis
   - Part 9: Final recommendation

8. **LMMCP_EXECUTIVE_SUMMARY.md** - Quick decision guide (2 pages)

9. **OBC_INVESTIGATION_STATUS.md** - This document

### Model Files

- **models/QMIPF_final.jl** - QMIPF with native OBC (180 vars)
  - Line 22: `max_obc_horizon = 100`
  - Lines 119-129: OBC implementation (native approach)
  - Line 244: `penalty_kappa = 1.0`

### Test Scripts

- **scripts/test_native_obc.jl** - Native OBC system test
- **scripts/HLT_comparison.jl** - SEP working examples
- **scripts/test_sep_with_obc_working.jl** - Penalty method tests

### User's Calibration Work

- **ChatGP_QMIPF/** - 14 markdown files documenting:
  - Two-stage calibration (deterministic → stochastic)
  - Long-run verification (10,000 periods)
  - Key finding: "numerical noise" not true binding

---

## Current Status of Components

| Component | Status | Notes |
|-----------|--------|-------|
| **Model Implementation** | ✅ Complete | OBC equations correct, matches Dynare |
| **Steady State** | ✅ Working | Solves in 3 seconds |
| **First-Order** | ✅ Working | Linearizes OBC (expected, user rejected) |
| **SEP Solver** | ✅ Working | Infrastructure solid, no MCP support |
| **OBC Enforcement (SEP)** | ❌ Not Working | Need MCP solver |
| **OBC Enforcement (Dynare)** | ✅ Works | LMMCP available |
| **LMMCP Port** | 🔄 Not Started | Ready to implement (3-4 weeks) |

---

## Next Steps

### Immediate (Awaiting User Decision)

**User must decide**:

1. **Proceed with LMMCP port?** (3-4 weeks)
   - High confidence of success
   - Proper solution to the problem
   - Significant time investment

2. **Try Complementarity.jl first?** (1 week)
   - Quick test of pure Julia MCP solver
   - Lower risk, faster results
   - May not work, but worth trying

3. **Use Dynare as fallback?** (1-2 weeks)
   - Guaranteed to work
   - Parallel codebase maintenance
   - Acceptable if LMMCP proves too difficult

4. **Alternative research approach?**
   - Discuss if OBC enforcement is truly critical
   - Consider approximations that might be acceptable

### If User Approves LMMCP Port

**Week 1**:
- Create `src/lmmcp.jl` module structure
- Translate core LMMCP algorithm (626 lines MATLAB → ~800 lines Julia)
- Phase I preprocessor + Phase II main loop
- Phi and DPhi subfunctions

**Week 2**:
- Unit tests for simple MCP problems
- Validate against Dynare test cases
- Numerical tuning and debugging
- **Decision point**: Continue or switch to PATHSolver.jl?

**Week 3**:
- Create `solve_deterministic_path_mcp` function
- Integrate with MacroModelling SEP
- Test with QMIPF model
- **Decision point**: Does THETA spike properly?

**Week 4**:
- Validation and robustness testing
- Documentation and user guide
- Example scripts
- Production release

---

## Key Learnings

### 1. Solver Architecture Matters

Not all "nonlinear solvers" are the same:
- **Standard Newton**: General-purpose, no complementarity
- **MCP solvers** (LMMCP, PATH): Specialized for complementarity
- **Can't approximate one with the other** - need the right tool

### 2. First-Order Is Not Always Acceptable

For research requiring:
- Global nonlinear dynamics
- Occasionally binding constraints
- Large shocks far from steady state

→ First-order linearization loses critical information

### 3. Listen to User's Research Goals

**Initial approach**: Concluded first-order was appropriate, investigation complete
**User feedback**: "I think you got it wrong! I want SEP to work"
**Lesson**: Always verify understanding of research requirements

### 4. Package Differences Are Fundamental

Dynare and MacroModelling.jl have different design philosophies:
- **Dynare**: Specialized tools for specific problems (MCP for OBC)
- **MacroModelling**: General perturbation methods with OBC approximation

Neither is "better" - they target different use cases.

---

## Conclusion

### What We Know

1. ✅ **Problem diagnosed**: SEP lacks MCP solver
2. ✅ **Solution identified**: Port LMMCP from Dynare
3. ✅ **Feasibility assessed**: 3-4 weeks, 70-80% success probability
4. ✅ **Alternatives documented**: PATHSolver.jl, Complementarity.jl, Dynare
5. ✅ **User's goal clarified**: Nonlinear SEP with OBC enforcement

### What's Needed

**Decision from user**:
- Commit to LMMCP port (3-4 weeks)?
- Try Complementarity.jl first (1 week)?
- Use Dynare as primary tool (1-2 weeks)?
- Discuss alternative research approaches?

### Confidence Level

**Technical feasibility**: 70-80% (LMMCP port will work)
**Time estimate**: 3-4 weeks ± 1 week
**Research impact**: High (enables proper nonlinear OBC analysis)

---

**Status**: ✅ Investigation complete, awaiting user decision on implementation path

**Recommended next action**: Review executive summary, decide on LMMCP port vs. quick Complementarity.jl test

