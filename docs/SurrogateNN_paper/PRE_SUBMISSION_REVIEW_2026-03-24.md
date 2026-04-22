# Pre-Submission Referee Report

**Paper**: Global Estimation of Nonlinear DSGE Models with Neural Network Surrogates
**Authors**: Matyas Farkas (IMF)
**Date**: 2026-03-24
**Review Standard**: Econometrica

---

## Overall Assessment

The paper develops a multi-fidelity estimator for nonlinear DSGE models that combines SEP solution, neural surrogate approximation, and regime-switching likelihood evaluation. Its principal strength is the nonlinearity decomposition showing that 69% of the FOM-ROM1 gap concentrates in the investment/capital block. The single most critical issue is that the 18-parameter MCMC chains have not converged (R-hat > 1.1 for 11/18 parameters), making the reported posterior shifts preliminary.

**Preliminary Recommendation**: Revise before submitting — the methodology is sound but the empirical results need converged chains and stronger caveats.

---

## Fixes Applied (2026-03-24)

### Critical Issues Fixed
1. **301 vs 184 quarters**: Changed "13 out of 301 post-burn-in quarters" to "13 out of 184 quarters" (2 locations)
2. **Q4:2007 outside sample**: Changed to 2001Q1 (sample ends 2004Q4)
3. **T=100 vs T=200**: Standardized to T=200 throughout, updated arithmetic (625 augmented dimensions, not 325)
4. **K=5 vs K=3 Gauss-Hermite**: Added clarification that K=5 for 3-shock model, K=3 for 7-shock HLT model
5. **Neural network parameter count**: Fixed 49,434 to correct value 49,562
6. **KL divergence bound**: Added justification for the additive decomposition (likelihood factorizes conditionally on state trajectory; tightness condition noted)
7. **Sigma_eta vs Sigma_y notation**: Standardized measurement error covariance to \Sigmay throughout
8. **Alpha reuse (Newton damping vs learning rate)**: Changed Adam learning rate from alpha to eta

### Major Issues Fixed
9. **"acceptance smoke" jargon**: Replaced with "validation check"
10. **RMSE to 11 decimal places**: Rounded to 3 significant figures
11. **Causal overclaiming**: Softened language in abstract, introduction, and results sections (4 locations)
12. **Missing ZLB caveat**: Added explicit "Scope limitations" paragraph noting ZLB never binds at training scale
13. **Missing convergence caveats**: Expanded Known Limitations section with convergence, ZLB, and approximation error paragraphs
14. **ROM1-conditioning inconsistency**: Added paragraph explaining the subtle inconsistency of recovering shocks under ROM1 but evaluating likelihood under the surrogate
15. **Appendix Table 12 parameter mismatch**: Added note clarifying this uses a different model than the main-text SW07 estimation
16. **Unreferenced appendix tables**: Added cross-references to tab:sample_length_sensitivity, tab:architecture_comparison, and tab:hyperparameter_grid from the main text

### Structural Additions
17. **Algorithm box for complete estimation**: Added Algorithm 2 summarizing offline and online stages
18. **Multi-fidelity KL bound justification**: Added explanation of why the per-period decomposition holds (conditional likelihood factorization)

---

## Remaining Issues (Not Yet Fixed)

### Critical
- 7 missing IRF figure PDFs (background Julia process still computing)
- Appendix Table 12 needs actual SW07 parameter recovery data (marked as "in preparation")
- MCMC chains need extension to 500K+ draws for convergence

### Major
- Particle filter benchmark timing data (projected, not actual — script needs API fixes)
- Section 7.4 investment-dominance connection to estimation gains could be strengthened with formal mechanism test
- Missing formal proof for Proposition 5.1 (Fisher information ordering)

### Minor
- Some appendix tables have only 4 entries in hyperparameter grid (says "12 architectures")
- Training procedure appendix uses eta for learning rate while main text now also uses eta (consistent)
- Number formatting not fully standardized (percentages vs "percent")

---

## Agent 6 Summary (Contribution Evaluation)

**Rating**: Incremental for Econometrica; Significant for Quantitative Economics or JEDC
**Recommendation**: Desk reject at Econometrica in current form

**Key concerns**:
1. No converged full-order posterior benchmark to validate surrogate accuracy
2. The nonlinearity decomposition is a measurement, not a mechanism identification
3. Paper conflates "where linear models fail" with "where nonlinear estimation helps" — the ZLB never binds in the training data
4. 18-parameter results are preliminary (convergence failure)

**Path to journal**: The methodology framework is sound and would be a strong contribution to JEDC or Quantitative Economics. For Econometrica, the paper would need: (a) converged chains with 500K+ draws, (b) a full-order SEP posterior as ground truth, (c) a formal proposition with proof for the Fisher information result, and (d) sample extension through 2008/COVID to test where nonlinearity matters most.

---

## Priority Action Items

**CRITICAL** (must fix before submission):
1. Extend MCMC chains to convergence (500K draws, ~8 hours for RS chain)
2. Complete IRF figure generation for the 7 missing figures
3. Generate full-order SEP posterior as benchmark (87 days on cluster — plan for parallel execution)

**MAJOR** (should fix):
4. Add ZLB-binding training data (shock_scale=0.4)
5. Extend sample to include 2008 crisis and/or COVID
6. Replace appendix 18-parameter table with actual SW07 parameter recovery
7. Formal proof for Proposition 5.1 or downgrade to conjecture

**MINOR** (polish):
8. Standardize number formatting throughout
9. Add confidence intervals to the amplification ratio figure
10. Clean up hyperparameter grid table to show all 12 configurations
