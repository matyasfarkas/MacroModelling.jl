# Paper Revision Plan for Top Field Journal Submission
**Target**: Quantitative Economics or Review of Economic Dynamics
**Timeline**: 8-10 weeks
**Created**: March 4, 2026

---

## Executive Summary

**Goal**: Transform the current draft (incremental contribution, synthetic-only) into a top field journal paper with:
- Complete validation (3-parameter + 18-parameter)
- Real-data application (US data estimation)
- Transparent benchmarking (particle filter, Kalman filter)
- Full replication package

**Current Status**: Draft complete, but 185 issues identified (45 critical)

**Strategy**: Three-phase approach
1. **Phase 1 (Weeks 1-2)**: Fix critical issues, add missing tables/figures
2. **Phase 2 (Weeks 3-6)**: Run computational validation (18-param, robustness, benchmarks)
3. **Phase 3 (Weeks 7-8)**: Real-data application
4. **Phase 4 (Weeks 9-10)**: Polish, replication package, final submission

---

## Phase 1: Critical Fixes (Weeks 1-2)

### Week 1: Paper Text Fixes

**Priority 1: Critical Errors** ✓ Quick wins

1. **Fix 48-shock error** (line 503)
   - Current: "48-shock Smets-Wouters (2007) model"
   - Fix: "3-shock Smets-Wouters (2007) model"
   - Time: 2 minutes

2. **Remove git commit references** (line 946)
   - Current: "commit b224f74"
   - Fix: "See Online Appendix"
   - Time: 5 minutes

3. **Define all abbreviations** at first use
   - DSGE, ZLB, ROM, FOM, HMC, MCMC, SEP
   - Time: 30 minutes

4. **Standardize notation**
   - ROM(k) for reduced-order model of order k
   - Bold θ for parameter vector
   - εₜ ∈ ℝⁿᵉ for shocks
   - Time: 1 hour

**Priority 2: Add Mathematical Content**

5. **Define likelihood function** (Section 4.4)
   ```latex
   \begin{equation}
   \mathcal{L}(\theta, \varepsilon | y) = \prod_{t=1}^T p(y_t | y_{1:t-1}, \theta, \varepsilon_t)
   \end{equation}
   ```
   Time: 2 hours

6. **Write ZLB constraint** (Section 3)
   ```latex
   \begin{equation}
   i_t = \max(0, i_t^*)
   \end{equation}
   ```
   Time: 30 minutes

7. **Add gate decision rule** (Section 4.3)
   ```latex
   \begin{equation}
   \text{Solver}(s_t, \theta) = \begin{cases}
   f_{\text{SEP}}(s_t, \theta) & \text{if } G(s_t, \theta) > \tau \\
   f_{\text{ROM}}(s_t, \theta) & \text{otherwise}
   \end{cases}
   \end{equation}
   ```
   Time: 1 hour

8. **Add surrogate loss function** (Section 4.2)
   Time: 30 minutes

9. **Add SEP algorithm pseudocode** (Section 4.1)
   Time: 2 hours

10. **Add HMC algorithm pseudocode** (Section 4.4)
    Time: 2 hours

**Total Week 1**: ~10 hours

### Week 2: Tables and Figures from Existing Results

**Use Phase 0 validation results** (640 samples from `.local_artifacts/robustness_20260303_134123/`)

11. **Create Table 1: Parameter Recovery**
    - Extract from Phase 0 checkpoint files
    - Columns: Parameter | True | Estimated | 95% CI | |Error|
    - Time: 4 hours (data extraction + LaTeX formatting)

12. **Create Table 2: MCMC Diagnostics**
    - R-hat, ESS, acceptance rate per parameter
    - Time: 2 hours

13. **Create Table A1: Prior Specification**
    - Document priors used in validation
    - Time: 1 hour

14. **Create Figures 4-6: Posterior Diagnostics**
    - Trace plots, posterior densities, autocorrelation
    - Time: 4 hours (plotting + formatting)

15. **Create Figure 7: Shock Recovery**
    - True vs estimated shocks over time
    - Time: 3 hours

16. **Create Figure 2: Motivating Example**
    - ZLB IRF showing linear vs nonlinear divergence
    - Time: 3 hours

**Total Week 2**: ~17 hours

**Phase 1 Total**: ~27 hours (1-2 weeks at 15-20 hrs/week)

---

## Phase 2: Computational Validation (Weeks 3-6)

### Week 3: 18-Parameter Dataset Generation

**Goal**: Generate training dataset with all 18 structural parameters

17. **Configure 18-parameter validation**
    - Parameter set: sw07_18params (all structural parameters)
    - Grid: prior sampling, N=50 theta values
    - Samples per theta: 160
    - Total samples: 8,000
    - Time: Setup 2 hours + Runtime 48-72 hours

18. **Monitor and checkpoint**
    - Use checkpoint-every=5 for safety
    - Time: Monitoring 2 hours

**Compute Resources**: 3 days on single machine or 1 day on 3 parallel machines

### Week 4: 18-Parameter Training and Validation

19. **Train 18-parameter surrogate**
    - Larger network: hidden=512, hidden2=256
    - More epochs: 800
    - Time: Setup 1 hour + Training 2-3 hours

20. **Generate synthetic data (18-param)**
    - Sample length: 240
    - Time: 30 minutes

21. **Run 18-parameter estimation**
    - Samples: 3,000 per chain × 4 chains = 12,000
    - Time: 6-8 hours

22. **Create Table 8: 18-Parameter Recovery**
    - All 18 parameters with recovery statistics
    - Time: 4 hours

23. **Write Section 7.5: 18-Parameter Validation**
    - Time: 4 hours

**Week 4 Total**: ~15 hours + 10 hours compute

### Week 5: Robustness Checks

**Goal**: Demonstrate stability across initial conditions

24. **Generate synthetic data with 3 different priors**
    - Steady-state prior (baseline - already done)
    - Displaced prior (+2σ displacement)
    - Crisis prior (-2σ displacement)
    - Time: 3× 30 minutes = 1.5 hours

25. **Run 3 estimation exercises**
    - Same estimator, different initial states
    - Time: 3× 2 hours = 6 hours compute

26. **Create Table 6: Initial Condition Robustness**
    - Compare posterior modes across 3 priors
    - Show max difference <8%
    - Time: 3 hours

27. **Write Section 7.3: Robustness to Initial Conditions**
    - Time: 3 hours

**Week 5 Total**: ~8 hours + 6 hours compute

### Week 6: Benchmarking

**Goal**: Transparent comparison to standard methods

28. **Implement particle filter benchmark** (if feasible)
    - Particle MCMC with exact SEP likelihood
    - OR: Document why infeasible + cite literature
    - Time: 8 hours (implementation) OR 2 hours (infeasibility explanation)

29. **Run Kalman filter benchmark** (easier)
    - Standard Kalman filter with first-order ROM
    - Same data, same priors
    - Time: 2 hours

30. **Create Table 7: Estimator Comparison**
    - Columns: Method | Parameter Recovery RMSE | Computational Time | Notes
    - Rows: ROM-Kalman, ROM-HMC, Switched, Full-NL
    - Time: 4 hours

31. **Create Table 5: Computational Cost Breakdown**
    - Components: SEP solve time, Surrogate eval time, Filter time, Total
    - Speedup factors vs baselines
    - Time: 3 hours

32. **Write Section 7.4: Computational Comparison**
    - Time: 4 hours

**Week 6 Total**: ~21 hours + some compute

**Phase 2 Total**: ~44 hours + 90 hours compute (can parallelize)

---

## Phase 3: Real-Data Application (Weeks 7-8)

**Goal**: Demonstrate method works on actual US data

### Week 7: Real-Data Infrastructure

33. **Obtain US macro data**
    - FRED: GDP, consumption, investment, hours, wages, inflation, interest rate
    - Quarterly data: 1984:Q1 - 2019:Q4 (pre-COVID)
    - Time: 4 hours (data collection + cleaning)

34. **Implement measurement equations**
    - Map model variables to observables
    - Handle log transformations, trends
    - Time: 6 hours

35. **Implement Kalman filter for real data**
    - Need filter for actual data (can't use filter-free without shocks)
    - Inversion filter or standard Kalman
    - Time: 8 hours

36. **Test on short sample**
    - Validate likelihood computation
    - Check numerical stability
    - Time: 4 hours

**Week 7 Total**: ~22 hours

### Week 8: Real-Data Estimation

37. **Run real-data estimation** (main exercise)
    - 3 parameters initially (cprobp, cindp, curvp)
    - Fix other parameters at calibrated values
    - Samples: 2,000 × 4 chains
    - Time: 4-6 hours compute

38. **Run sensitivity checks**
    - Different prior specifications
    - Different sample periods (Great Recession vs full sample)
    - Time: 2× 4 hours = 8 hours compute

39. **Create Table 9: Real-Data Estimation Results**
    - Parameter estimates, standard errors, priors
    - Time: 4 hours

40. **Create Figure 8: Real-Data Fit**
    - Data vs model-implied observables
    - Time: 4 hours

41. **Create Figure 9: Filtered Shocks**
    - Shock estimates over time
    - Compare to historical narrative (Great Recession, etc.)
    - Time: 4 hours

42. **Write Section 8: Application to US Data**
    - Discuss parameter estimates
    - Economic interpretation
    - Model fit assessment
    - Time: 8 hours

**Week 8 Total**: ~20 hours + 16 hours compute

**Phase 3 Total**: ~42 hours + 16 hours compute

---

## Phase 4: Polish and Submission (Weeks 9-10)

### Week 9: Polish Paper

43. **Address all minor issues** from pre-submission review
    - 45 minor style/grammar issues
    - Hyphenation, passive voice, paragraph length
    - Time: 8 hours

44. **Write related literature section enhancements**
    - Position contribution more clearly
    - Add comparison table (this method vs alternatives)
    - Time: 4 hours

45. **Revise abstract and introduction**
    - Align claims with evidence
    - Add real-data results to abstract
    - Time: 3 hours

46. **Revise conclusion**
    - Summarize 18-parameter + real-data results
    - Discuss limitations honestly
    - Time: 2 hours

47. **Complete all appendices**
    - Full model equations
    - Algorithm details
    - Additional robustness checks
    - Time: 6 hours

48. **Final LaTeX compilation**
    - Fix all cross-references
    - Ensure all tables/figures compile
    - Check bibliography
    - Time: 3 hours

**Week 9 Total**: ~26 hours

### Week 10: Replication Package

49. **Create replication scripts**
    - Master script that runs everything
    - Documented parameters
    - Time: 8 hours

50. **Test replication package**
    - Fresh environment test
    - Time: 4 hours

51. **Write README and documentation**
    - Installation instructions
    - Computational requirements
    - Expected runtime
    - Time: 4 hours

52. **Archive code and data**
    - Zenodo deposit
    - Get DOI
    - Time: 2 hours

53. **Final proofreading**
    - Read entire paper fresh
    - Check for inconsistencies
    - Time: 6 hours

54. **Prepare submission materials**
    - Cover letter
    - Highlights
    - Conflict of interest statement
    - Time: 3 hours

**Week 10 Total**: ~27 hours

**Phase 4 Total**: ~53 hours

---

## Timeline Summary

| Phase | Weeks | Human Hours | Compute Hours | Key Deliverables |
|-------|-------|-------------|---------------|------------------|
| Phase 1 | 1-2 | 27 | 0 | Critical fixes, tables 1-2, figures 4-7 |
| Phase 2 | 3-6 | 44 | 90 | 18-param validation, robustness, benchmarks |
| Phase 3 | 7-8 | 42 | 16 | Real-data application |
| Phase 4 | 9-10 | 53 | 0 | Polish, replication package |
| **Total** | **10 weeks** | **166 hours** | **106 hours** | **Top-journal-ready paper** |

**At 20 hours/week**: 8-9 weeks of focused work
**At 15 hours/week**: 11-12 weeks

**Compute**: Most can run overnight or in parallel

---

## Risk Mitigation

### Critical Risks

**Risk 1**: Real-data estimation fails (non-convergence, identification issues)
- **Mitigation**: Start with 3 parameters, fix rest at calibration
- **Fallback**: Estimate on simulated data with measurement error (quasi-real)

**Risk 2**: 18-parameter validation has low success rate
- **Mitigation**: Use optimal HMC settings from Phase 0
- **Fallback**: Reduce to 12 "key" parameters, document limitation

**Risk 3**: Particle filter benchmark infeasible (too slow)
- **Mitigation**: Document computational cost theoretically
- **Fallback**: Cite literature benchmarks, acknowledge limitation

**Risk 4**: Timeline overrun
- **Mitigation**: Phase 2-3 can partially parallelize
- **Fallback**: Submit to JEDC (accepts synthetic-only if thorough)

### Minor Risks

- Data availability issues → FRED is public, low risk
- Computational bottlenecks → Can rent cloud compute if needed
- Convergence issues → Have robust HMC settings validated

---

## Success Criteria

### Minimum Viable Product (for submission)

✓ All critical errors fixed (48-shock, git commits, etc.)
✓ All promised tables/figures present
✓ 18-parameter validation OR honest limitation discussion
✓ Real-data application OR quasi-real simulation with measurement error
✓ At least one benchmark comparison (Kalman filter minimum)
✓ Full replication package with DOI

### Ideal Submission

✓ All of MVP +
✓ 18-parameter full validation successful
✓ Real US data estimation
✓ Particle filter benchmark (even if limited)
✓ Robustness checks across multiple dimensions
✓ No limitations that referees can easily criticize

---

## Target Journal Decision Tree

### Quantitative Economics
**Pros**: Top field journal, values replication, computational methods
**Cons**: High bar, requires full replication package
**Requirements**: MVP + Ideal (all checkboxes)
**Submission**: If all validation + real-data successful

### Review of Economic Dynamics
**Pros**: Top field, publishes methodological papers
**Cons**: Prefers applied contributions
**Requirements**: MVP + real-data application
**Submission**: If real-data works but 18-param partial

### Journal of Economic Dynamics and Control
**Pros**: Accepts thorough synthetic validation
**Cons**: Lower ranking than QE/RED
**Requirements**: MVP minimum
**Submission**: If real-data or 18-param challenging

---

## Immediate Next Steps (Today)

1. **Start Phase 1, Week 1** ✓ Already in progress
   - Fix critical errors (2 hours)
   - Add mathematical definitions (6 hours)
   - Total: 8 hours (1 day of focused work)

2. **Extract Phase 0 results** for Table 1-2
   - Load checkpoint files from `.local_artifacts/robustness_20260303_134123/`
   - Compute statistics
   - Total: 4 hours

3. **Commit validation harness updates**
   - Preserve the 100% success configuration
   - Total: 30 minutes

**Today's Goal**: Complete all critical text fixes + extract Phase 0 results for tables

---

## Long-Term Success Metrics

**Paper Acceptance**: Quantitative Economics or Review of Economic Dynamics
**Citations (5 years)**: >50 (strong for computational methods paper)
**Code Reuse**: Publicly available package with active users
**Policy Impact**: Adoption by central bank modeling teams

---

**Let's make this a top field journal paper!**

*Created: March 4, 2026*
*Target completion: May 4, 2026 (10 weeks)*
*First submission: May 2026*
