# Pre-Submission Referee Report

**Paper**: Global Estimation of Nonlinear DSGE Models with Neural Network Surrogates
**Authors**: Mátyás Farkas (International Monetary Fund)
**Date**: 2026-03-31
**Review Standard**: Leading Field Journal (top-field)

---

## Overall Assessment

The paper develops a multi-fidelity estimator for nonlinear DSGE models that combines a neural network residual correction with a distance-based switching rule and filter-free HMC sampling, and documents that the investment/capital block accounts for 69 percent of the linearization gap in a Smets--Wouters model. The principal strength is the investment-dominance decomposition---a clean, deterministic measurement independent of MCMC convergence---which is genuinely novel and surprising. The single most critical issue is that the two Kalman baseline tables (Tables 7 and 8) report fundamentally different posterior values under the same "Kalman" label, making the linear-vs-nonlinear comparison---the paper's central empirical result---internally contradictory.

**Preliminary Recommendation**: Revise before submitting

---

## 1. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. **Lines 248, 926, 1291, 1745 | Inconsistent HLT expansion** -- "Holden--Lindé--Trabandt" (lines 248, 1291) vs. "Herbst--Linde--Trabandt" (lines 926, 1745). These are different people. Fix: identify the correct authors and use one expansion consistently. Also fix the missing accent on "Lindé" at line 1745.

2. **Lines 1567--1569 | Duplicated paragraph** -- The inversion-filter mode paragraph is repeated nearly verbatim (sampling dimension, FD gradients, 19 evaluations, ~1.1s gradient cost, tree depth 5.5, ~45 leapfrog steps, 50s/draw, 2.4 hours total). Delete one copy.

3. **Line 185 | Subject-verb disagreement** -- `\citet{boehl2022monetary} estimates` should be `estimate` (citation resolves to an author name taking plural verb).

4. **Line 888 | "the data prefers"** -- "Data" is plural in formal academic writing. Change to "the data prefer."

5. **Line 328 | "equity-premium shocks"** -- The SW07 model has no equity-premium shock. Replace with "investment-specific technology" or "risk-premium" to match the model's actual shock terminology.

6. **Line 890 | Misleading MH comparison** -- The sentence claims NUTS compares favorably to MH, then gives ESS from an HMC chain as the benchmark. The comparison is actually NUTS-vs-HMC. Revise.

7. **Line 853 | Grammar** -- "All $\hat{R} < 1.01$ indicates" should be "All $\hat{R} < 1.01$, indicating" (subject-verb disagreement).

### Minor Issues

1. Line 364: "are worth noting" → filler phrase; replace with "Two further design choices merit discussion."
2. Line 494 vs 505: Nonlinearity measure switches between $\Delta_j$ and $\delta_j$; pick one.
3. Line 505: "$R^2 = 0.960$" vs line 494: "0.96" -- use consistent precision.
4. Line 1478: "300-600" should use en-dash: "300--600."
5. Line 1006: "every single period" → "every period" (less emphatic).
6. Line 1719: "[512, 256]" called "deeper" but it is wider, not deeper.
7. Line 938: "An important caveat remains" → "A caveat remains."
8. Lines 1067-1070: Percentile ranges use hyphens ("0-25th") instead of en-dashes ("0--25th").

### Style Patterns to Fix Throughout

- **Number ranges**: Search for digit-hyphen-digit patterns outside math mode and replace hyphens with en-dashes globally.
- **Orphaned labels**: `\label{subsubsec:flexible_init}` (line 337) and `\label{subsubsec:hmc_sep}` (line 343) are bare labels outside any section heading, never referenced. Remove them.
- First person ("I"), percentage formatting ("69 percent"), em-dash usage, and absence of filler words are all clean and consistent.

---

## 2. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. **[Table 8 "Kalman" column] ↔ [Table 7]** | The "Kalman" column in Table 8 (`tab:18param_switching_posterior`) reports fundamentally different values from Table 7 (`tab:18param_posterior`): $\rho_b$ = 0.762 vs. 0.702; $\sigma_b$ = 0.122 vs. 0.142; $\varepsilon_p$ = 39.3 vs. 77.9; $\xi_p$ = 0.740 vs. 0.924. The abstract and introduction use Table 8 values ("$\sigma_b$: 0.12 to 0.28", "$\varepsilon_p$: 39 to 104") while Table 7 shows a completely different Kalman posterior. These appear to be from different chains/seeds without any acknowledgment. | Severity: **CRITICAL**

2. **[Line 977] ↔ [Table 9 / Table 7 notes]** | Convergence diagnostics prose: "ESS exceeds 473 for all parameters in the **linear** chain and 501 in the **regime-switching** chain." Tables report the opposite: Kalman ESS ≥ 501, RS ESS ≥ 473. The ESS values are swapped between chains. | Severity: **CRITICAL**

3. **[Figure 13 caption, line 771] ↔ [sample period]** | Caption: "The financial crisis of 2007--2009 produces the peak rolling RMSE." But the data sample is 1959Q1--2004Q4, ending three years before the crisis. | Severity: **CRITICAL**

4. **[Lines 248, 1291] ↔ [Lines 926, 1745]** | HLT = "Holden--Lindé--Trabandt" vs. "Herbst--Linde--Trabandt." Different people. | Severity: **CRITICAL**

5. **[Abstract, line 137] ↔ [Table 8]** | Abstract says Kimball curvature "rises roughly threefold." Actual ratio: 104.2/39.3 = 2.65×. Introduction and results correctly say "2.7-fold." | Severity: **CRITICAL**

### Cross-Reference Errors

1. **Missing HLT citation** -- "Holden--Lindé--Trabandt" referenced throughout but no bib entry exists for this work. Only `holden2016computation` (Holden alone) appears.
2. **Generated tables orphaned** -- Four generated .tex files (`table_hlt_realdata_posterior`, etc.) with labels are never `\input`ed or referenced.
3. All other `\ref` / `\label` pairs verified correct.

### Terminology Drift

1. **"FOM" vs "SEP"** -- Used somewhat interchangeably for the nonlinear solution. The compound "SEP FOM" in figure captions adds confusion. Recommend: define FOM = SEP consistently.
2. **$\delta_j$ vs $\Delta_j$** -- Nonlinearity measure uses both symbols. Harmonize.
3. **Activation function** -- Section 4.2 says "SiLU" without qualification; Appendix clarifies 3-param uses tanh, 18-param uses SiLU. Qualify in main text.
4. **Gali model variable count** -- Line 246 says "23 variables" but NN architecture (line 1453) uses 22 state inputs/outputs. Clarify.

### Minor Inconsistencies

1. Table 1 quintile counts (5 × 4,600 = 23,000) vs. stated dataset size (22,080). Inconsistent.
2. Extended sample $\varepsilon_p$ baseline "from 39" matches Table 8 but not Table 7 (77.9).
3. "$\varepsilon_p$ rises eightfold from 10 to 78" -- ratio is 7.79×, approximately but imprecisely "eightfold."
4. Bib keys misleading: `fernandezvillaverde2021estimating` has year=2015; `gust2012effects` has year=2017.
5. Appendix C.2 duplicated paragraph (lines 1567 vs 1569).

---

## 3. Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)

1. **[Abstract, line 137]** | "I attribute this to the interaction of convex investment adjustment costs..." | The decomposition shows 69% *loads onto* the investment block, but this is variance decomposition, not causal attribution. | Fix: "I attribute" → "This is consistent with."

2. **[Introduction, line 161]** | "When a negative risk premium shock depresses q, the marginal adjustment cost for future recovery rises---a compounding effect that linear approximation replaces with a constant coefficient." | Mechanism described as established fact without counterfactual test (e.g., shutting down adjustment cost). | Fix: "The model's equations imply that..."

3. **[Section 7.1, line 555]** | "The likely source is the interaction of three nonlinear functions..." | Entire mechanism paragraph treats conjecture as decomposition finding. No channel has been individually shut down. | Fix: Begin with "I conjecture that..." and add "Testing this requires shutting down individual nonlinear functions."

4. **[Section 7.1, line 557]** | "The distinction has policy implications---central bank communication can anchor learning-driven variation but cannot alter the curvature of adjustment cost functions." | Policy claim with zero supporting evidence. | Fix: Delete or weaken to "If confirmed, this distinction would have policy implications."

5. **[Section 7.4, line 938]** | "The economic interpretation is that the linear model...compensates by inflating exogenous shock processes." | Presented as "the" interpretation; alternatives include surrogate bias, different modes, inversion-filter wedge. | Fix: "One interpretation is that..." + "Alternative explanations include..."

6. **[Section 7.4, line 936]** | "risk premium volatility $\sigma_b$ doubles...the nonlinear model assigns a larger role" | Causal claim appears before the caveat acknowledging surrogate error. | Fix: Move caveat immediately after the claim.

7. **[Section 7.5, line 1006]** | "I therefore regard the particle filter as structurally unsuited" | Based on single evaluation with one PF variant. | Fix: "I regard the standard bootstrap particle filter as structurally unsuited; whether tempered or guided variants can overcome degeneracy remains open."

8. **[Conclusion, line 1176]** | Repeats the substitution interpretation as settled, without maintaining caveats from the body.

### Generalization Issues

1. Title says "DSGE Models" (plural, general) but evidence comes from two specific models. Investment dominance could be calibration- or model-specific. Fix: Conclusion should note "Whether investment dominance extends to other model classes remains an open question."

2. The 18-parameter estimation uses 1959Q1--2004Q4, excluding both the 2008 crisis and COVID-19---precisely the episodes where nonlinearity should matter most. Should be flagged more prominently in abstract and conclusion.

3. Line 801: "As a negative control" -- Gali differs from SW07 in many ways beyond lacking investment, so not a clean negative control. Fix: "As a comparison model that lacks the investment channel."

### Missing Caveats

1. **Model misspecification** -- The decomposition tells us where the *model's* nonlinearity concentrates, not the *economy's*. Different functional forms could reverse the finding.
2. **Single chain** -- Posterior comparisons rest on one chain per model with known multimodality.
3. **Inversion filter inconsistency** -- Shocks recovered under ROM1 but likelihood evaluated under surrogate. Bias direction not characterized.
4. **Measurement error calibration** -- Calibrated, not estimated. No sensitivity analysis for 18-parameter case.
5. **Finite-difference gradient accuracy** -- Step size not reported. FD noise could mask gradient pathologies.
6. **Unit mass matrix** -- Poor preconditioner for parameters spanning different scales ($\sigma \sim 0.1$ vs $\varepsilon_p \sim 100$).
7. **Training data coverage** -- Not verified that posterior mean lies within convex hull of training parameter draws.

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors

1. **Line 429-432 | LL bias bound arithmetic** -- With stated values ($T=184$, $\sigma_y \approx 0.5$, RMSE $\approx 1.2$, $\|H\|=1$), the formula yields $184/(2 \times 0.25) \times 1.44 \approx 530$ nats, not the claimed 442. Verify inputs.

2. **Line 401 | Missing normalizing constant** -- Gaussian log-likelihood omits $\frac{n_y}{2}\log(2\pi)$. Add constant or note "up to an additive constant."

3. **Line 1266 | Inflation notation in linearized system** -- Mixes hat-variables ($\hat{R}_t$) with level variables ($\pi_t$). Under zero SS inflation these coincide, but should be stated explicitly.

4. **Line 1397 vs 311 | Newton iteration index** -- Main text uses index $j$ with per-iteration $\alpha^{(j)}$; appendix uses $i$ with constant $\alpha$. Unify.

5. **Lines 1586-1588 | Gelman-Rubin formula** -- $n$ not defined; $B$ convention ambiguous without specifying reference.

### Notation Inconsistencies

1. **$\kappa$** -- Softplus sharpness (eq. 6, line 324) AND Phillips curve slope (eq. A.5, line 1227). Completely different. Rename one.

2. **$\delta$** -- Used for FOUR quantities: capital depreciation, nonlinearity measure, per-period KL contribution, and NUTS acceptance target. Rename at least two.

3. **$\Delta_j$ vs $\delta_j$** -- Same nonlinearity measure denoted by two symbols. $\Delta_j$ is a vector at line 277 but a scalar norm at line 492. Unify.

4. **$\Delta y$** -- Training residual (line 358) AND output growth (throughout results). Serious clash. Rename training residual.

5. **$\sigma$** -- CRRA parameter, shock SDs, SiLU sigmoid, shock scale notation ("$k\sigma$"). Write "sigmoid" explicitly in SiLU definition.

6. **$H$** -- Introduced as nonlinear function $H(s_t)$ (line 220), then used as matrix $Hs_t$ (line 401). Declare linearity from start or use consistent notation.

7. **$\Sigma_y$ vs $\Sigma_\eta$** -- Same measurement covariance denoted differently at line 220 vs appendix line 1548.

### Undefined Notation

1. $w$ / $w^*$ (network weights): introduced without formal definition.
2. $\theta_{\text{NN}}$ (line 1471): network parameters, but earlier called $w$. Unify.
3. $p_2$ (line 295): used in KL divergence without definition.
4. $\mathcal{T}^*_B$ (line 384): used without formal definition.
5. RRMSE: used throughout but never given a displayed formula.
6. $\bar{s}(\theta)$: steady state, defined only inline.
7. $\Sigma_\eps$: line 1352 implies general covariance; line 205 says identity.

### LaTeX Math Formatting

1. Line 717: $\text{IRF}^{\text{SEP}}_h(k\sigma)$ -- inconsistent subscript vs function argument across the paper.
2. Line 1130: Square brackets for vector, inconsistent with paper's parenthesis convention.
3. Line 1403: $\mathbb{R}^{(T \times n_s)}$ -- unusual parentheses in exponent. Use $\mathbb{R}^{Tn_s}$.
4. Line 1461: SiLU $= x \cdot \sigma(x)$ reuses $\sigma$ for sigmoid. Write $\text{sigmoid}(x)$.

---

## 5. Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

| Table | Missing Element |
|-------|----------------|
| Table 1 (`tab:error_by_distance`) | RMSE units not specified (level differences? percentage points?). N per quintile (4,600) inconsistent with stated total (22,080). |
| Table 3 (`tab:param_nonlinearity`) | No significance levels or standard errors on correlations. Sparse notes. |
| Table 4 (`tab:per_obs_r2`) | Unclear whether "all params" column is joint regression or sum of sub-columns. |
| Table 5 (`tab:param_region_rmse`) | N per cell not reported. Train vs test not stated. RMSE units missing. |
| Table 6 (`tab:crisis_vs_normal`) | Three observables have "---" for variance share without explanation. |
| Table 9 (`tab:regime_performance`) | N observations, model, and "fit region" definition missing. |
| **Table 11** (`tab:18param_switching_posterior`) | **Kalman column values differ from Table 10 without acknowledgment. Major inconsistency.** |
| Table 14 (`tab:surrogate_regional_accuracy`) | Model and dataset not specified. RRMSE not defined in note. |
| Table 18 (`tab:18param_full`) | Misleading title: says "18-Parameter Validation" but is on a "simplified NK model." Rename. |

### Figures with Missing or Incomplete Notes

| Figure | Missing Element |
|--------|----------------|
| Figure 1 | Data source: model name, parameter values, shock scale not in caption. |
| Figure 5-7 (IRFs) | Model and calibration not stated in caption. |
| Figure 8-9 (Correction/asymmetry surfaces) | Model specification and axis labels should be verified. |
| **Figure 13** (`fig:rolling_rmse`) | **FACTUAL ERROR: Caption references "financial crisis of 2007--2009" but sample ends 2004Q4. Must fix.** |

### Cross-Reference Issues

- 4 generated table files never `\input`ed (orphaned).
- ~20 figure files in `figures/` directory never referenced by `\includegraphics`.
- 2 orphaned `\label`s (`subsubsec:flexible_init`, `subsubsec:hmc_sep`) never referenced.
- `\label{par:results_18param_switching}` on a `\paragraph` may not produce useful `\ref` number.

### Formatting Inconsistencies

- Table note spacing: most use `\smallskip`, Table 6 uses `\vspace{0.5em}`.
- Percentage formatting mixed: "43.1\%" in body vs "68.9" in "Share (\%)" column.
- CI notation inconsistent: "90\% CI" vs "Q2.5/Q97.5" vs "RS 90\% CI" across tables.
- RMSE decimal places vary: 3 places (Table 1) vs 5 places (Table 14).
- Figure widths vary without clear pattern (0.75 to 1.0 `\textwidth`).
- Caption style mixed: title-case noun phrases vs sentence-case.

---

## 6. Contribution & Referee Assessment

### Part 1 — Central Contribution

**Claim**: The paper develops a multi-fidelity estimator for nonlinear DSGE models using NN residual learning + distance-based switching + filter-free HMC, and finds that the investment/capital block accounts for 69% of the linearization gap.

**Rating: Incremental-to-Significant** (leaning Incremental in current state)

The decomposition result is genuinely novel and surprising. The methodology combines known ingredients in a new configuration. However, neither contribution is fully developed: the decomposition is model/calibration-specific without robustness tests, and the estimation method lacks gold-standard validation. The paper honestly acknowledges most limitations, but acknowledgment does not substitute for resolution.

### Part 2 — Identification and Credibility

The decomposition is a deterministic measurement (strong). The posterior-shift result is weaker: linear and nonlinear chains use different likelihood constructions, so differences confound genuine nonlinear effects with approximation artifacts. The nonlinear model has a worse LL (-1,045 vs -1,027), making the interpretation of posterior shifts ambiguous. Single chains with known multimodality are insufficient to characterize either posterior.

**What a seminar skeptic would say**: "You are comparing posteriors from two different approximate models and attributing the difference to 'nonlinearity.' But the nonlinear model fits worse. How do I know the shifts aren't just surrogate bias? Show me the direct SEP posterior."

### Part 3 — Analyses: Required and Suggested

**Required:**

1. **Direct SEP posterior comparison** (at least partial, e.g., fixing 12 params, estimating 6 via direct SEP). The paper states this is "computationally feasible" but does not do it.

2. **Monte Carlo validation with SW07-HLT model** (not just Gali). The 3-param validation uses a model with no investment---the wrong model to validate the main finding.

3. **Multiple chains** for regime-switching estimation (at least 2-4, different seeds). Known multimodality with a single chain is insufficient.

4. **Robustness of 69% decomposition to calibration** ($S''$, $c_z$, $h$). If halving $S''$ halves the investment share, the finding is calibration-specific.

**Suggested:**

1. Second-order perturbation benchmark (ROM2 posterior comparison).
2. Decomposition at binding shock scales (shock_scale=0.4 where ZLB binds).
3. Out-of-sample forecast comparison (train 1959-1994, forecast 1995-2004).
4. Marginal likelihood or DIC comparison.
5. Extended sample estimation with retrained surrogate.

### Part 4 — Literature Positioning

**Missing citations:**
- Fernández-Villaverde and Rubio-Ramírez (2007, RES): foundational nonlinear estimation with PF
- Aruoba, Cuba-Borda, and Schorfheide (2018, QE): Markov-switching ZLB estimation
- Richter and Throckmorton (2015, 2016): global ZLB estimation
- **Justiniano, Primiceri, and Tambalotti (2010, 2011)**: investment-specific shocks as main business cycle driver --- directly relevant to the investment-dominance finding and entirely missing

**Over-citing**: Silver et al. (2016, AlphaGo) and Rezende and Mohamed (2015, normalizing flows) are not used or closely related.

**Framing**: The paper leads with methodology but the decomposition finding is more interesting and novel. Consider reframing: "I ask where linearization error matters most and develop a method to answer this question."

### Part 5 — Journal Fit and Recommendation

**Best targets**: Journal of Econometrics (computational methods), Quantitative Economics (method + empirical finding), JEDC (computational macro --- achievable in current state).

**Recommendation**: **Revise before sending to referees.** Main requirements: (1) reconcile the two Kalman tables, (2) multiple RS chains, (3) Monte Carlo validation with SW07-HLT, (4) calibration robustness of 69% finding, (5) cite Justiniano-Primiceri-Tambalotti.

### Part 6 — Questions to the Authors

1. **The nonlinear model has a worse log-likelihood (-1,045 vs -1,027). How should readers interpret posterior shifts from a model that fits worse? Is it possible the shifts are driven by surrogate bias rather than genuine nonlinear effects?**

2. **The 3-parameter validation uses the Gali model, which has no investment. Your main finding is about investment dynamics. Why not validate with a model containing the mechanism you claim is important?**

3. **How sensitive is the 69% share to the calibrated adjustment cost parameter $S''$ and utilization elasticity $c_z$?** If halving $S''$ halves the investment share, the finding is an artifact of calibration.

4. **The inversion filter recovers shocks under ROM1 but evaluates the likelihood under the surrogate. Can you bound the resulting posterior bias?**

5. **Two linear chains find different modes (34-nat gap). Yet the linear-nonlinear comparison uses a single chain from each. How do you know the RS chain found a different posterior rather than a different mode?**

6. **The bootstrap PF benchmark uses very tight measurement error ($\sigma_y$ = 0.07-0.27). Have you checked whether ESS collapse is driven by atypically tight measurement error rather than a structural property?**

7. **Kimball curvature $\varepsilon_p$ correlates negatively with nonlinearity, yet the nonlinear posterior pushes $\varepsilon_p$ up threefold. Does the estimator converge to a region where its own nonlinear correction is unnecessary?**

---

## Priority Action Items

**CRITICAL** (must fix — these could cause desk rejection or major referee objections):

1. **Reconcile the two Kalman baseline tables** (Tables 7 and 8 report different posteriors under the same "Kalman" label). This makes the paper's central empirical comparison internally contradictory. [Agent 2, Agent 5]

2. **Fix the swapped ESS values** in the convergence diagnostics paragraph (line 977 swaps linear and RS chain ESS). [Agent 2]

3. **Fix Figure 13 caption** referencing the "financial crisis of 2007--2009" when the sample ends in 2004Q4. [Agent 2, Agent 5]

4. **Fix the HLT acronym** -- identify the correct authors (Holden vs Herbst) and use one expansion consistently; add proper bib entry. [Agent 1, Agent 2]

5. **Fix the abstract's "threefold"** claim -- actual ratio is 2.65×; introduction correctly says 2.7-fold. [Agent 2]

6. **Delete the duplicated appendix paragraph** (lines 1567 vs 1569). [Agent 1]

7. **Resolve $\kappa$ notation clash** -- softplus sharpness and Phillips curve slope use the same symbol. [Agent 4]

**MAJOR** (should fix — will likely be raised by referees):

8. **Weaken causal language** throughout Sections 7.1, 7.4, and the conclusion. The decomposition shows correlation, not causation; mechanisms are conjectures, not findings. Delete the unsupported central bank communication policy claim. [Agent 3]

9. **Run multiple RS chains** (2-4 with different seeds). Single chain with known multimodality is insufficient for posterior characterization. [Agent 6]

10. **Add calibration robustness for the 69% finding** -- vary $S''$, $c_z$, $h$ across the literature range. [Agent 6]

11. **Validate on SW07-HLT** (not just Gali). The 3-param validation uses a model lacking investment, which is the main finding's core mechanism. [Agent 6]

12. **Fix the log-likelihood bias bound arithmetic** (line 429-432) -- computed value is ~530 nats, not 442. [Agent 4]

13. **Resolve $\delta$ notation clash** -- four different meanings (depreciation, nonlinearity measure, KL contribution, NUTS target). [Agent 4]

14. **Rename Table 18** from "18-Parameter Validation" to something indicating it uses a simplified NK model, not SW07-HLT. [Agent 5]

15. **Cite Justiniano, Primiceri, and Tambalotti (2010, 2011)** on investment shocks as business cycle drivers -- directly relevant to the investment-dominance finding and entirely absent. [Agent 6]

16. **Add missing caveats**: inversion-filter inconsistency direction, FD gradient accuracy, unit mass matrix limitations, training data coverage of posterior support. [Agent 3]

**MINOR** (polish — improves paper quality):

17. Fix subject-verb disagreements (lines 185, 853, 888). [Agent 1]
18. Fix "equity-premium shocks" terminology (line 328). [Agent 1]
19. Define RRMSE with a displayed formula before first use. [Agent 4]
20. Standardize CI notation across tables (90% CI vs Q2.5/Q97.5). [Agent 5]
21. Standardize percentage formatting in tables (% in cells vs column headers). [Agent 5]
22. Remove orphaned labels and unreferenced figure files. [Agent 5]
23. Add normalizing constant or "up to additive constant" to LL formula (line 401). [Agent 4]
24. Replace number-range hyphens with en-dashes throughout. [Agent 1]
25. Add table notes for Tables 3, 5, 9, 14 (missing N, units, significance). [Agent 5]

---

## Issue Counts by Category

| Agent | Critical | Major | Minor | Total |
|-------|----------|-------|-------|-------|
| 1. Spelling, Grammar & Style | 7 | 0 | 8+2 patterns | 17 |
| 2. Internal Consistency | 5 | 0 | 9 | 14 |
| 3. Unsupported Claims | 11 | 4 | 12 | 27 |
| 4. Mathematics & Notation | 5 | 7+7 undefined | 4 | 23 |
| 5. Tables, Figures & Documentation | 2 | 9 tables + 4 figures | 6 formatting | 21 |
| 6. Contribution Evaluation | — | 4 required + 5 suggested | 7 questions | 16 |
| **Total** | **30** | **40** | **48** | **118** |
