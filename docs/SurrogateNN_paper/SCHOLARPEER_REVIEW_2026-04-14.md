# ScholarPeer Review Report

**Paper**: The Invisible Nonlinearity: Investment Adjustment Costs and Structural Bias in Linear DSGE Estimation
**Authors**: Matyas Farkas (International Monetary Fund)
**Date**: 2026-04-14
**Target Venue**: Econometrica (Top-5 Economics)
**Cutoff Date**: 2026-04-14

---

## Final Review

### Summary

This paper makes two distinct contributions. First, it documents a new analytical finding: the investment adjustment cost function $S(I_t/I_{t-1})$ in the canonical Smets-Wouters DSGE model satisfies $S(1) = S'(1) = 0$, rendering it invisible to first-order perturbation. An equation-block decomposition shows this invisible channel generates 69% of the nonlinear-linear solution gap, while the effective lower bound on interest rates -- the standard motivation for nonlinear estimation -- generates less than 11%, and the price Phillips curve generates 0.2%. Second, it develops a multi-fidelity neural-network estimator that learns the residual correction between the full nonlinear (stochastic extended path) and linear solutions, achieving 2,450x speedup and enabling 1,000 NUTS-HMC draws in 2.4 hours with zero divergent transitions. Applying this to US data (1959--2025), the paper finds that linear estimation compensates for the invisible nonlinearity by doubling risk-premium volatility, collapsing wage markup persistence by 80%, and raising Kimball curvature 2.7-fold -- all in directions predicted by the invisibility theory.

### Strengths

- **Genuinely novel analytical finding.** The perturbation invisibility property ($S(1) = S'(1) = 0$) and its quantitative dominance (69% of the FOM-ROM gap) are new. No prior paper decomposes the full-order vs. reduced-order model gap by equation block in a medium-scale DSGE. The closest work, Aruoba, Bocola, and Schorfheide (2017, JEDC), assesses aggregate nonlinearities without isolating the investment adjustment cost channel. This is a clean, important finding that should survive any methodological revision of the paper.

- **Clean identification strategy.** The Gali (2015) comparison -- same ELB, same Kimball, no investment, near-zero nonlinearity -- provides a compelling difference-in-differences argument. The counterfactual experiments (removing $S(\cdot)$, linearizing $a(z)$) reinforce the identification. These are deterministic, model-based measurements that do not depend on MCMC convergence or surrogate quality.

- **Particle filter benchmark is informative.** Documenting ESS = 1 at all particle counts (100--5,000) for the 66-variable model, producing a 21-million-nat gap from the exact Kalman value, is a useful contribution. This confirms the well-known but rarely quantified weight degeneracy in medium-scale DSGE settings (Snyder et al. 2008, Herbst and Schorfheide 2019).

- **Multi-fidelity residual learning architecture.** Learning $\Delta g = g^{\text{FOM}} - g^{\text{ROM1}}$ rather than the full transition is a smart design choice. The correction is two orders of magnitude smaller than the full transition, concentrated in the investment block, and the architecture gracefully falls back to ROM1 when the correction is zero. The achieved RRMSE of 0.0012 is impressive.

- **Formal consistency theory.** The BvM theorem for the surrogate posterior, adapted from the misspecified-likelihood framework of Kleijn and van der Vaart (2012), is properly executed. The quartic dependence of coverage distortion on RRMSE ($\text{RRMSE}^4 \times T \approx 10^{-10}$) provides a strong theoretical guarantee.

- **COVID shock reinterpretation.** The finding that linear estimation requires $56\sigma$ risk-premium shocks during COVID (probability $< 10^{-300}$) provides a structural explanation for what the literature has treated as a statistical anomaly (cf. Lenza and Primiceri 2022, Ferroni et al. 2024).

- **Self-contained paper.** Full model specification, algorithm details, hyperparameter search, and code availability make the paper reproducible.

### Weaknesses

#### Critical

1. **Missing comparison with Janssens and McCrary (2025).** This is the single most important omission. Janssens and McCrary develop a local linearization method that computes time-varying linear approximations at the forecasted state, yielding a time-varying state-space system evaluated via the Kalman filter. They report 100,000 posterior draws in under 8 minutes -- roughly 18x faster than the paper's 2.4 hours for 1,000 draws -- without neural network training, without a filter-free formulation, and without a surrogate. Their method captures nonlinear dynamics (including the quadratic investment adjustment cost) through time-varying coefficients. The paper's central methodological claim -- that the proposed multi-fidelity NN architecture is necessary for feasible nonlinear estimation -- is directly challenged by this competing approach. The paper must either demonstrate that the local linearization misses important global features of the investment-channel nonlinearity (e.g., the $S' \times p^k \times \xi_{t+1}/\xi_t$ interaction), or acknowledge this as a viable alternative. *[Baseline Scout, item 1.1]*

2. **Kase, Melosi, and Rottner (2025) not cited.** This BIS Working Paper is the closest methodological competitor: neural network surrogates for nonlinear DSGE estimation. They approximate the full likelihood (not the residual correction) and use particle filters (not filter-free HMC), but the paper must discuss how the multi-fidelity residual-correction architecture compares to the full-likelihood approximation approach. Not citing the most closely related NN-for-DSGE paper signals potential unfamiliarity with the frontier. *[Baseline Scout, item 1.2]*

3. **No full-model Monte Carlo coverage study.** The 3-parameter Gali validation is necessary but insufficient. The formal consistency theorem predicts coverage distortion below 0.5 percentage points, but this is never verified computationally for the full 18-parameter SW07-HLT model. For Econometrica, a Monte Carlo study with known DGP is near-mandatory for a methods paper. Even 50--100 replications would suffice. The paper honestly acknowledges this as future work, but this will not satisfy referees. *[Baseline Scout, item 2.1]*

4. **Severe multimodality unresolved.** A second surrogate chain (seed 123456) converges to a region 12,700 nats below the primary chain. All posterior inference in Sections 6--7 is conditional on a single mode that may not be globally representative. With only one chain finding the "good" mode, the reader has no guarantee that (a) it is the global mode, or (b) the posterior mass is concentrated there. The paper acknowledges this honestly but does not apply parallel tempering or multiple restarts. Brault (2024, Bank of Canada) provides exactly this tool for DSGE posteriors. *[Technical Soundness Q4]*

#### Major

5. **Particle filter comparison is a strawman.** Only the bootstrap particle filter -- the weakest variant -- is tested. The tempered particle filter (Herbst and Schorfheide 2019) and the conditionally optimal particle filter (Aruoba et al. 2021, RED) are cited but never implemented. The claim that "the filter-free approach is not merely convenient but structurally necessary" rests entirely on bootstrap PF results. If the tempered PF also collapses for the 66-variable model (plausible), reporting that result would strengthen the paper. If it does not collapse, the motivation for the filter-free approach weakens. *[Baseline Scout, item 1.5--1.6]*

6. **No out-of-sample forecasting comparison.** The paper makes strong claims about structural bias but provides no predictive evaluation. A rolling-window or expanding-window exercise comparing linear vs. regime-switching surrogate forecasts on GDP growth, inflation, and investment growth would directly test whether the nonlinear model's parameter shifts translate into improved forecasts. *[Baseline Scout, item 2.2]*

7. **Regime-switching gate assigns only 10% to nonlinear regime.** If 90% of periods use the linear Kalman filter, the surrogate correction is active only in a small fraction of the sample. The 18-nat log-likelihood gap (roughly 0.1 nat per period) is modest. The paper should discuss whether the gate threshold is optimally calibrated and what happens under alternative threshold choices.

8. **Causal language overstates evidence.** The paper frames posterior shifts as "predicted by the invisibility theory" and "confirmed," but the shifts could also reflect surrogate approximation error, posterior multimodality (the acknowledged 12,700-nat gap), or the regime-switching architecture itself. The paper has appropriately qualified some claims (e.g., "directionally consistent"), but the abstract and conclusion retain language like "confirms" that should be softened to "consistent with." *[Technical Soundness Q5]*

#### Minor

9. **OccBin comparison deferred.** Running OccBin on the same data would demonstrate that the investment-channel posterior shifts do NOT appear under piecewise-linear estimation, sharpening the identification. The logical argument is sound, but empirical demonstration would be more convincing.

10. **Sensitivity to $S''(1)$ calibration missing.** The 69% finding depends on $S'' = 6.01$ being calibrated, not estimated. A sensitivity analysis over the plausible range of $S''$ (say 2--10) would test robustness. The paper notes this as future work.

11. **ROM2 comparison omitted.** Second-order perturbation captures $S''(1)$ at steady state. A ROM2 comparison on the model without OBC would test whether the SEP-based approach adds value beyond what second-order perturbation already provides for the investment channel. The paper argues ROM2 cannot handle OBC, which is valid, but the decomposition results hold even at shock scales where the ZLB does not bind.

12. **Writing quality is high but the paper is long (83 pages).** The appendices are thorough but could be condensed. The Farkas-Tatar (2020) comparison appendix (Appendix H) adds 3 pages of limited value to the main narrative.

### Suggestions

1. **Add Janssens-McCrary comparison.** Either implement their local linearization on the SW07-HLT model (their code is available) and compare posterior estimates, or provide a formal argument for why time-varying local linearization cannot capture the $S' \times p^k \times \xi/\xi$ cubic cross-terms that SEP computes.

2. **Cite and discuss Kase, Melosi, Rottner (2025).** Position the multi-fidelity residual correction relative to their full-likelihood surrogate approach. Argue why learning the correction is preferable (smaller function, better conditioning, graceful fallback).

3. **Run at minimum a limited Monte Carlo.** Even 20--50 replications at a known DGP with the full 18-parameter model would provide empirical evidence on coverage. The 3-parameter result is suggestive but not sufficient.

4. **Test the tempered particle filter.** If it also produces ESS = 1 for the 66-variable model, report it -- this would strengthen the paper's argument. If it works, acknowledge the alternative.

5. **Address multimodality.** Run 4--8 surrogate chains from different initializations, or implement parallel tempering (Brault 2024). If only one chain finds the good mode, that is informative about the difficulty of the problem, but the paper should not base all inference on a single chain.

6. **Tighten causal language.** Replace "confirms" with "is consistent with" in the abstract and conclusion. The posterior shifts are directionally consistent with the invisibility theory, but causal attribution requires ruling out the surrogate error channel definitively.

7. **Add sensitivity to $S''(1)$.** Recompute the decomposition at $S'' \in \{2, 4, 6, 8, 10\}$ and report how the investment share varies. If the share is robust, this significantly strengthens the finding.

### Questions to the Authors

1. Janssens and McCrary (2025) achieve 100,000 posterior draws in 8 minutes via time-varying local linearization. Have you compared your surrogate's posterior estimates against this method? If not, what features of the investment-channel nonlinearity do you believe local linearization would miss?

2. The second surrogate chain converges 12,700 nats below the primary chain. What is your interpretation of this gap? Is the primary chain's mode the global mode, or could a better initialization find a superior region? Have you considered parallel tempering?

3. The regime-switching gate assigns only 10% of periods to the nonlinear regime. How sensitive are the posterior shifts (Table 5) to the gate threshold $\tau$? What happens if you force 100% of periods through the surrogate?

4. You test only the bootstrap particle filter and find ESS = 1. Have you tried the tempered particle filter (Herbst and Schorfheide 2019) or the conditionally optimal PF (Aruoba et al. 2021)? If these also degenerate, it would substantially strengthen your motivation for the filter-free approach.

5. The 18-nat log-likelihood gap between linear ($-1,027$) and surrogate ($-1,045$) has the "wrong" sign -- the linear model has higher likelihood. You interpret this as the surrogate "slightly overfitting to training-set parameter regions." Could this instead indicate that the surrogate's nonlinear correction is not well-calibrated at the posterior mode? What is the decomposition of the 18-nat gap into surrogate error vs. genuine nonlinear effects?

6. How does the 69% investment share change if $S''(1)$ is varied from 2 to 10? Is the dominance result robust to the calibration of the adjustment cost curvature?

7. Kase, Melosi, and Rottner (2025, BIS WP 1241) also use neural network surrogates for nonlinear DSGE estimation but approximate the full likelihood rather than the residual correction. What are the theoretical and practical advantages of the multi-fidelity residual approach over the full-likelihood approach?

### Decision Score

| Dimension | Score (1--10) | Justification |
|-----------|:---:|---|
| **Soundness** | 6 | The decomposition finding (Sections 2--5) is technically sound and model-independent. The estimation framework (Sections 6--7) has significant gaps: unresolved multimodality, no full-model Monte Carlo, strawman particle filter comparison. Formal theory is correct but not empirically verified for the full model. |
| **Significance/Novelty** | 8 | The invisibility theorem and investment-dominance finding are genuinely new and important. The multi-fidelity residual architecture is novel in the econometrics context. The COVID shock reinterpretation adds economic substance. These contributions would survive even a major methodological revision. |
| **Presentation** | 7 | Well-written, dense prose appropriate for Econometrica. Self-contained with thorough appendices. But 83 pages is long, and some appendices could be condensed. The paper would benefit from splitting the clean analytical contribution (decomposition) from the estimation methodology. |
| **Overall** | 6 | A paper with an 8-quality finding wrapped in a 6-quality estimation framework. The decomposition result is Econometrica-worthy; the estimation pipeline needs significant additional validation (Monte Carlo, multimodality resolution, competing method comparison) before the posterior inference can be trusted. |

### Preliminary Recommendation

**Weak Reject** (Revise and Resubmit encouraged)

The paper contains a genuinely important analytical finding -- the invisibility of investment adjustment costs to linearization and its quantitative dominance over the ELB -- that is novel, cleanly identified, and policy-relevant. This finding alone could support a strong publication at a top-5 journal. However, the estimation methodology has critical gaps that must be addressed: (1) comparison with Janssens-McCrary's competing method, (2) full-model Monte Carlo coverage verification, (3) multimodality resolution via multiple chains or parallel tempering, and (4) testing beyond the bootstrap particle filter. The decomposition sections (2--5) are publication-ready; the estimation sections (6--7) need a revision round. I would strongly encourage resubmission after addressing the critical weaknesses.

---

## Appendix A: Structured Summary (Agent 1)

### 1. Problem Statement & Motivation

The paper addresses a fundamental misallocation of attention in the nonlinear DSGE estimation literature. The standard motivation for nonlinear estimation is the zero lower bound (ZLB/ELB) on nominal interest rates. The paper argues this is the wrong place to look: the investment adjustment cost function $S(I_t/I_{t-1})$ satisfies $S(1) = S'(1) = 0$, making it entirely invisible to linearization while generating 69 percent of the gap between nonlinear and linear solutions in the canonical Smets-Wouters model.

Current limitations identified:
- First-order perturbation discards the *entire* investment adjustment cost mechanism
- Second-order perturbation captures $S''(1)$ at steady state but degrades with distance
- Bootstrap particle filters suffer complete weight degeneracy (ESS=1) in medium-scale models
- Piecewise-linear methods (OccBin) handle the ZLB but cannot capture the quadratic investment-channel nonlinearity

### 2. Methodology

**Perturbation Invisibility Property:** Any function satisfying $f(\bar{x}) = f'(\bar{x}) = 0$ at steady state contributes zero to first-order perturbation. The investment adjustment cost $S(x) = (\phi/2)(\gamma x - \gamma)^2$ satisfies both conditions at $x = 1$.

**Equation-Block Decomposition:** The nonlinearity measure $\Delta_j = \|g_\theta^{\text{FOM}}(s_j, \epsilon_j) - g_\theta^{\text{ROM1}}(s_j, \epsilon_j)\|_2$ is decomposed by assigning state variables to economic blocks and computing each block's share of total $\sum\|\Delta_j\|_2^2$, across 22,080 state-shock-parameter combinations.

**Multi-Fidelity Neural Network Surrogate:** The NN learns $\Delta g = g^{\text{SEP}} - g^{\text{ROM1}}$ (the correction, not the full transition). Architecture: input $\mathbb{R}^{51}$ (26 states + 7 shocks + 18 params), hidden [256, 128] with SiLU, output $\mathbb{R}^{26}$. RRMSE = 0.0012.

**Filter-Free HMC:** NUTS samples $\theta$ via AdvancedHMC.jl; an inversion filter recovers shocks analytically from observations at each $\theta$ proposal. A regime-switching gate selects per-period whether Kalman or inversion-filter likelihood applies.

**Formal Consistency:** Under $\delta_T = o(T^{-1/2})$, the surrogate posterior concentrates at $\theta_0$, contracts at $\sqrt{T}$ rate, and satisfies BvM. Coverage distortion scales as $\text{RRMSE}^4 \times T \approx 10^{-10}$.

### 3. Key Contributions

1. Invisibility theorem + investment dominance (69% of FOM-ROM1 gap)
2. Gali difference-in-differences identification (same ELB, same Kimball, no investment = no nonlinearity)
3. Multi-fidelity neural network surrogate (residual learning, not full transition)
4. Filter-free HMC framework avoiding particle filter degeneracy
5. Empirical posterior comparison showing parameter shifts predicted by invisibility theory

### 4. Main Results & Experiments

- **Equation-block decomposition:** Investment/capital = 68.9%, Euler/consumption = 13.5%, wage Phillips curve = 10.8%, price Phillips curve = 0.2%
- **3-parameter validation:** Recovery within 5%, true inside 90% CIs, $\hat{R} < 1.01$
- **18-parameter real-data estimation:** $\sigma_b$ doubles (0.12 to 0.28), $\rho_w$ collapses (0.37 to 0.08), $\varepsilon_p$ rises 2.7x (39 to 104)
- **Extended sample (2025Q1):** 30-nat LL improvement ($-1,787$ vs $-1,818$), all 7 shock volatilities fall under nonlinear model, $\sigma_b$ falls 49%
- **Particle filter:** ESS = 1 at all counts, LL = $-21.2 \times 10^6$ vs exact $-1,818$
- **B3 SEP validation:** ROM1-SEP RMSE = 1.04 avg across 20 posterior draws
- **Runtime:** 2.4 hours for 1,000 NUTS draws vs 43 days projected for PF+SEP

---

## Appendix B: Domain Narrative (Agent 3)

### a) Domain History

The estimation of medium-scale DSGE models has rested, for nearly two decades, on a remarkably stable technological stack: first-order perturbation, the Kalman filter, and random-walk Metropolis-Hastings. Smets and Wouters (2007) cemented this pipeline as the workhorse of central-bank macroeconometrics. Nonlinearity was acknowledged but rarely confronted -- Fernandez-Villaverde and Rubio-Ramirez (2007) showed that particle filters could handle nonlinear state spaces, yet the computational burden confined their use to small models.

Two parallel developments shifted the frontier after 2020. First, OccBin-style piecewise-linear methods (Guerrieri and Iacoviello 2015; Boehl 2022) made occasionally-binding constraints tractable for medium-scale models, though estimation still relied on adapted Kalman or ensemble filters that suppress higher-order nonlinearities. Second, machine learning entered the field: Maliar, Maliar, and Winant (2021) used deep learning to solve dynamic models; Kase, Melosi, and Rottner (2022) trained neural surrogates for HANK estimation; Childers et al. (2022) showed filter-free HMC could bypass filtering for perturbation-based models.

As of early 2026, the field sits at an impasse: nonlinear solution methods exist, efficient samplers exist, and neural approximation methods exist, but no one has fused all three into a coherent estimation framework for the canonical medium-scale DSGE.

### b) Open Problems

1. **Bias characterization:** No paper has decomposed which model equations generate the linear-nonlinear likelihood gap.
2. **Scalable nonlinear estimation:** Particle filters degenerate; piecewise-linear methods discard smooth nonlinearities.
3. **Extreme-shock interpretation:** COVID-era multi-tens-of-sigma shocks under linear models are unexplained structurally.
4. **Multi-fidelity correction for econometrics:** The engineering multi-fidelity literature has not been adapted to structural estimation with formal consistency guarantees.

### c) Significance Criteria for Econometrica

A contribution qualifies as significant for Econometrica if it: (i) resolves a methodological bottleneck the profession has identified but not overcome; (ii) delivers a substantive economic finding obtainable only under the new paradigm; (iii) establishes formal statistical properties (consistency, coverage). A paper checking all three boxes occupies genuinely new territory.

---

## Appendix C: Missing Baselines & Datasets (Agent 4)

### Missing Baselines

| # | Method | Reference | Severity | Reason |
|---|--------|-----------|----------|--------|
| 1 | Local linearization | Janssens & McCrary (2025), SSRN 5282668 | **CRITICAL** | 100k draws in 8 min via time-varying Kalman. Directly challenges need for NN surrogate. Not cited. |
| 2 | NN surrogate likelihood | Kase, Melosi & Rottner (2025), BIS WP 1241 | **CRITICAL** | Closest NN-for-DSGE competitor. Not cited. |
| 3 | Ensemble Kalman + piecewise-linear | Boehl & Strobel (2024), JEDC 158 | **MAJOR** | Published SW07+ZLB estimation through COVID. Not compared. |
| 4 | Differentiable HMC | Childers et al. (2022), NBER WP 30573 | **MAJOR** | Canonical filter-free HMC for DSGE. May be incorrectly cited. |
| 5 | Conditionally optimal PF | Aruoba et al. (2021), RED 41 | **MAJOR** | Only bootstrap PF tested = strawman. |
| 6 | Tempered PF (implementation) | Herbst & Schorfheide (2019) | **MAJOR** | Cited but never run. |
| 7 | OccBin estimation | Guerrieri & Iacoviello (2015), Boehl (2022) | **MAJOR** (ack.) | Deferred to future work but would sharpen identification. |
| 8 | ROM2 without OBC | Andreasen, FV, RR (2018) | MODERATE | Would test if $S''(1)$ capture is sufficient. |
| 9 | Parallel tempering | Brault (2024), BoC SWP 2024-13 | MODERATE | Paper names PT as next step; Brault provides it. |
| 10 | Mixture density networks | Naubert (2025), BoC SWP 2025-3 | MINOR | Adequately cited. |

### Missing Experiments

| # | Experiment | Severity | Reason |
|---|-----------|----------|--------|
| 1 | Full 18-param Monte Carlo coverage | **CRITICAL** | Only 3-param validation done. Coverage unverified for full model. |
| 2 | Out-of-sample forecasting | **MAJOR** | No predictive evaluation despite structural bias claims. |
| 3 | Multiple surrogate chains | **MAJOR** | 1 of 2 chains finds good mode (12,700-nat gap). |
| 4 | Formal Bayes factor | MODERATE | LL gaps without marginal likelihood comparison. |
| 5 | $S''(1)$ sensitivity | MODERATE | Core 69% finding depends on calibrated $S'' = 6.01$. |
| 6 | Cross-country (Euro area/Japan) | MINOR | US-only analysis. |

---

## Appendix D: Interrogation Log (Q&A Engine)

### Aspect: Novelty & Significance

**Q1:** What is the novelty and significance of decomposing the FOM-ROM gap in DSGE models by equation block to identify investment adjustment costs as the dominant nonlinearity source?

**A1:** This is a genuinely novel contribution. No prior paper in the DSGE literature decomposes the gap between nonlinear and linear solutions by equation block. Aruoba, Bocola, and Schorfheide (2017, JEDC) assess aggregate DSGE nonlinearities using predictive checks but do not isolate the investment adjustment cost channel or quantify its share relative to the ELB or Phillips curve. The $S(1) = S'(1) = 0$ property is analytically well-known (Christiano, Eichenbaum, Evans 2005; Smets and Wouters 2007), but its estimation consequences -- that 69% of the nonlinear-linear gap traces to this invisible function -- have never been documented. The Gali comparison (same ELB, same Kimball, no investment, near-zero nonlinearity) provides clean identification that would be difficult to challenge. **Novelty: High. Significance: High** -- this finding redirects the motivation for nonlinear DSGE estimation from the ELB to the investment channel.

**Q2:** What is the novelty of using a multi-fidelity neural network that learns the residual correction (FOM minus ROM1) rather than the full transition or likelihood for DSGE estimation?

**A2:** Multi-fidelity residual correction is well-established in engineering (Peherstorfer, Willcox, Gunzburger 2018, SIAM Review; Torzoni et al. 2024, Proc. Royal Soc. A), but its application to structural econometric estimation is new. In the DSGE literature, Kase, Melosi, and Rottner (2025) approximate the full particle-filter likelihood with a neural network; Azinovic, Gaegauf, and Scheidegger (2022) learn the full equilibrium; Naubert (2025) learns the full conditional distribution. The paper's approach of learning only the *correction* $\Delta g = g^{\text{FOM}} - g^{\text{ROM1}}$ -- a smaller, smoother function -- is distinct and draws from a different intellectual tradition (multi-fidelity methods in computational science). **Novelty: Medium-High** (novel in economics, adaptation of known engineering technique). **Significance: Medium** -- the practical speedup is large (2,450x), but Janssens and McCrary (2025) achieve comparable or greater speedup via local linearization without neural networks.

**Q3:** How novel is the filter-free HMC approach for nonlinear DSGE models, and how does it compare to existing alternatives?

**A3:** Childers, Fernandez-Villaverde, Perla, Rackauckas, and Wu (2022, NBER WP 30573) introduced filter-free HMC for DSGE estimation but applied it only to perturbation-based (linear/second-order) solutions. Naubert (2025) extends filter-free estimation to nonlinear models via mixture density networks. The paper's contribution is combining filter-free HMC with the stochastic extended path solution via a neural surrogate, achieving zero divergent transitions for a 66-variable model. This is an incremental but useful extension of the Childers et al. framework. **Novelty: Medium** (extends existing paradigm to a new solution method). **Significance: Medium** -- the zero-divergence result is noteworthy but the approach is not compared against the Janssens-McCrary alternative that achieves faster results.

**Q4:** What is the novelty of interpreting implausible COVID-era shock magnitudes (56 sigma) as evidence of linearization bias rather than extreme events?

**A4:** This interpretation is new. The existing literature treats COVID-era shocks as statistical anomalies: Lenza and Primiceri (2022, JAE) scale up the covariance matrix; Ferroni, Fisher, and Melosi (2024, JME) introduce a dedicated "Covid shock." The NY Fed DSGE model post-COVID assessment (Del Negro et al. 2024) documents poor forecasting performance without providing a structural explanation. No prior paper connects these implausible shock magnitudes to the specific mechanism of investment-cost invisibility under linearization. **Novelty: High. Significance: Medium-High** -- this reinterpretation has important implications for how central banks handle crisis-period data.

**Q5:** How significant are the posterior parameter shifts (sigma_b doubles, rho_w collapses) relative to existing evidence on linear vs. nonlinear DSGE estimation?

**A5:** Prior work on linear vs. nonlinear posterior comparison in medium-scale DSGE is very limited. Gust et al. (2017, AER) estimate a simplified nonlinear NK model with the ZLB but do not compare the full SW07 posterior. Boehl and Strobel (2024) compare piecewise-linear posteriors but cannot capture the smooth IAC nonlinearity. The paper's documentation of specific parameter shifts predicted by the invisibility theory is novel and adds economic content. However, the finding is conditional on a single posterior mode (multimodality acknowledged but unresolved) and on the surrogate's accuracy at the posterior. **Novelty: Medium-High. Significance: Medium** -- important if the estimation methodology is validated, but the unresolved multimodality and missing Monte Carlo study weaken confidence.

### Aspect: Technical Soundness

**Q6:** Is the decomposition result (investment = 69%) computed at representative parameter values, or could it be an artifact of the training grid?

**A6:** The decomposition is computed across 22,080 state-shock-parameter combinations drawn from a Sobol-sampled parameter grid. The paper reports that the investment share is stable across shock scales 0.1--1.2 and rises to 88--92% at posterior parameter values. The Gali comparison and counterfactual experiments (removing $S$, linearizing $a(z)$) provide independent confirmation. The result appears robust to the parameter grid. However, the paper does not report sensitivity to $S''(1)$ calibration (fixed at 6.01), which controls the magnitude of the invisible nonlinearity. A sensitivity analysis over plausible $S''$ values would strengthen the result.

**Q7:** Does testing only the bootstrap particle filter -- the weakest PF variant -- constitute a strawman argument for the filter-free approach?

**A7:** Yes, partially. The bootstrap PF is known to perform poorly in high dimensions (Snyder et al. 2008). The tempered PF (Herbst and Schorfheide 2019) and the conditionally optimal PF (Aruoba et al. 2021) are specifically designed to mitigate weight degeneracy. The paper cites both but implements neither. For a 66-variable model, all PF variants may indeed degenerate, but this must be demonstrated rather than assumed. If the tempered PF also produces ESS = 1, that result would *strengthen* the paper's argument. The current presentation risks appearing as if the authors chose the weakest baseline to make their approach look better.

**Q8:** How reliable is the posterior inference given that only one of two surrogate chains finds a reasonable mode?

**A8:** This is a serious concern. The second chain converges to a region 12,700 nats below the primary, which the paper describes as "severe multimodality." Standard MCMC diagnostics ($\hat{R}$, ESS) are meaningless when chains find different modes. The paper reports within-chain diagnostics (ESS $\geq$ 473, zero divergences) but these do not address across-mode coverage. All posterior comparisons in Sections 6--7 are conditional on the primary mode. Without parallel tempering or multiple successful chains, the reader cannot assess whether the posterior shifts are mode-specific artifacts. The paper's honesty in disclosing this is commendable, but the inference remains fragile.

**Q9:** Is the formal consistency theorem (BvM for the surrogate posterior) empirically verified, or does it rest entirely on theoretical arguments?

**A9:** The theorem is correct but empirically unverified for the full model. The 3-parameter Gali validation shows parameter recovery within 5% with truth inside all 90% CIs, which is consistent with the theory. The RRMSE-based scaling ($\text{RRMSE}^4 \times T \approx 10^{-10}$) provides a strong theoretical bound. The B3 SEP posterior validation (20 draws, RMSE = 1.04) confirms that the ROM1-SEP gap is real but does not verify coverage. The gap between theoretical guarantee and empirical verification is the missing Monte Carlo study: generate data from the 18-parameter model at known $\theta_0$, estimate with the surrogate, and check whether true parameters are inside credible intervals. Without this, the formal theory is a promissory note.

**Q10:** Could the 18-nat log-likelihood gap partly reflect surrogate error rather than genuine nonlinearity, and how is this decomposed?

**A10:** The paper acknowledges this ambiguity: "The 18-nat gap reflects two sources: surrogate approximation error (bounded by the RRMSE) and genuine nonlinear effects (the object of interest)." The B3 SEP validation helps: the ROM1-SEP RMSE of 1.04 across 20 posterior draws confirms that the nonlinear correction is economically material. However, the 18-nat gap has the "wrong" sign (linear LL is higher), which the paper interprets as the surrogate "slightly overfitting to training-set parameter regions." This interpretation is plausible but not verified. A clean decomposition would require computing the exact nonlinear LL at the posterior mode via SEP -- a computationally intensive but feasible exercise for a single parameter vector.

---

## Appendix E: Literature Review

### Key References (48 papers organized by sub-field)

**A. Canonical DSGE & Investment Adjustment Costs:**
Smets & Wouters (2007, AER) -- canonical medium-scale DSGE; Christiano, Eichenbaum, Evans (2005, JPE) -- originated IAC specification; Justiniano, Primiceri, Tambalotti (2010, JME) -- investment shocks dominant; Ascari et al. (2024, JAE) -- investment Euler equation identification; Christiano, Motto, Rostagno (2014, AER) -- risk shocks.

**B. Nonlinear DSGE Estimation:**
Fernandez-Villaverde & Rubio-Ramirez (2007, RES) -- particle filter for DSGE; Gust et al. (2017, AER) -- nonlinear NK with ZLB; Herbst & Schorfheide (2019, JoE) -- tempered PF; Cuba-Borda et al. (2019, JAE) -- COPF for OBC; Guerrieri & Iacoviello (2015, JME) -- OccBin; Boehl (2022, JEDC) -- fast OBC; Boehl & Strobel (2024, JEDC) -- EnKF + piecewise-linear; Holden (2023, REStat) -- OBC existence/uniqueness; Aruoba et al. (2021, RED) -- COPF for OBC.

**C. Higher-Order Perturbation:**
Schmitt-Grohe & Uribe (2004, JEDC) -- 2nd-order; Andreasen, FV, RR (2018, RES) -- pruned 3rd-order; Aruoba, Bocola, Schorfheide (2017, JEDC) -- assessing nonlinearities; Andreasen & Kronborg (2022, QE) -- extended perturbation.

**D. Stochastic Extended Path:**
Fair & Taylor (1983, Econometrica) -- original EP; Adjemian & Juillard (2025, JEDC) -- stochastic EP.

**E. Filter-Free MCMC:**
Hoffman & Gelman (2014, JMLR) -- NUTS; Farkas & Tatar (2020) -- HMC for DSGE; Childers et al. (2022, NBER) -- differentiable state-space + HMC.

**F. Neural Networks for Economics:**
Maliar, Maliar, Winant (2021, JME) -- deep learning for dynamic models; Azinovic et al. (2022, IER) -- deep equilibrium nets; Kase, Melosi, Rottner (2025, BIS) -- NN surrogate for HANK; Naubert (2025, BoC) -- mixture density networks; Janssens & McCrary (2025, SSRN) -- local linearization; FV, Nuno, Perla (2024) -- deep learning survey.

**G. Multi-Fidelity Methods:**
Peherstorfer, Willcox, Gunzburger (2018, SIAM Rev) -- multi-fidelity survey; Duffin et al. (2024) -- neural surrogate HMC.

**H. COVID & Extreme Shocks:**
Lenza & Primiceri (2022, JAE) -- scaling covariance; Ferroni et al. (2024, JME) -- Covid shock.

**I. Bayesian Asymptotics:**
Kleijn & van der Vaart (2012, EJS) -- BvM under misspecification; Mueller (2013, Econometrica) -- risk under misspecification.

---

## Agent Call Summary

| Agent | Role | Search-Enabled | Depends On | Status |
|-------|------|----------------|------------|--------|
| 1. Summary Agent | Internal Compression | No | Paper | Complete |
| 2. Literature Review & Expansion | Dynamic Context Creation | Yes | Paper | Complete (48 papers) |
| 3. Historian Agent | External Compression | No | Agent 2 | Complete |
| 4. Baseline Scout Agent | Integrity Checking | Yes | Agent 2 | Complete |
| 5a. Novelty Q Generator | Question Generation | No | Agents 1--4 | Complete (inline) |
| 5b. Novelty A Generator (x5) | Search-Verified Answers | Yes | Agent 5a | Complete (inline + web) |
| 6a. Soundness Q Generator | Question Generation | No | Agents 1--4 | Complete (inline) |
| 6b. Soundness A Generator (x5) | Paper-Based Answers | No | Agent 6a | Complete (inline) |
| 7. Review Generator | Guidelines-Driven Synthesis | No | All above | Complete (inline) |

**Total agent calls**: 4 (subagents) + 6 web searches + inline Q&A and synthesis
**Note**: Agents 5--7 were executed inline due to rate limits on subagent spawning. All Phase 2 outputs were available in context.
