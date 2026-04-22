# Pre-Submission Referee Report

**Paper**: Global Estimation of Nonlinear DSGE Models with Neural Network Surrogates
**Authors**: Mátyás Farkas (IMF)
**Date**: 2026-03-23
**Review Standard**: Econometrica

---

## Overall Assessment

The paper proposes a practical estimator for nonlinear DSGE models that combines stochastic extended path (SEP) solution, a parameter-conditional neural network surrogate, and a regime-switching likelihood gate. Its most compelling empirical finding is that 69% of the nonlinearity gap between the full-order model and first-order perturbation concentrates in the investment/capital block, not the pricing block or the ZLB. The principal strength is the architecture's clear logic and the nonlinearity decomposition's potential to redirect the literature's focus. The single most critical issue is that the 18-parameter headline results are drawn from non-converged MCMC chains ($\hat{R}$ up to 3.62), which renders the posterior comparisons in the abstract and introduction unreliable.

**Preliminary Recommendation**: Substantial revision required

---

## 1. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. **Line 213** | "Second-order and higher-order perturbation adds curvature" → "Second-order and higher-order perturbation **methods add** curvature" | Subject-verb disagreement.

2. **Line 1048** | "1954Q3--2032Q1, 311 quarters" | The date range is inconsistent with the 1959Q1--2004Q4 estimation sample stated elsewhere and extends into the future. Must reconcile.

3. **Line 1055** | "HLT model" introduced without definition. | The paper variously calls the 18-parameter model "Smets--Wouters," "SW07," and "HLT." Define "HLT" at first use or replace consistently.

4. **Lines 163 vs. 1191--1196** | "HMC" described as the online sampler, but the 18-parameter exercise uses random-walk MH, not HMC/NUTS. | The paper should clarify which sampler is used in which exercise.

5. **Line 1064** | "the second-order correction is exactly quadratic in the state deviation, so scaling the shock by factor $k$ scales both the ROM1 response and the ROM2--ROM1 gap by $k$, leaving the ratio unchanged." | Mathematically incorrect: if ROM1 response scales as $k$ and ROM2--ROM1 gap scales as $k^2$, the ratio scales as $k$, not 1. Verify or correct.

6. **Lines 1348--1350** | Seed sensitivity table reports $\xi_p$, $\phi_\pi$, $\phi_y$ but the 3-parameter validation estimates $\sigma_A$, $\sigma_\mu$, $\sigma_R$. | Wrong parameters in the table.

7. **Line 510** | "Typically 200--500 epochs to convergence" | Sentence fragment. Add subject.

8. **Line 243** | "Farkas (2020) adapt...demonstrate" → "adapts...demonstrates" | Single-author subject-verb agreement.

9. **Line 545** | Missing bracket/delimiter in the inversion filter argmin expression. | Fix LaTeX.

10. **Line 1837** | $\sigma_\mu$ relative error of $-6.7\%$ exceeds the paper's own 5% acceptance criterion (C1). | Flag or explain.

### Minor Issues

1. Inconsistent use of accent on "Galí" — missing in plain text at lines 852 and 1053.
2. "About 45 minutes" (line 183) vs. "50 minutes" (abstract) for different exercises — clarify.
3. "4--6$\times$ speedup" should spell out or use proper $\times$ in math mode.
4. Line 482: $120 \times 180 = 21{,}600$, not the claimed ~43,000. Arithmetic error or unexplained discrepancy.
5. Line 627: "planned for the final version" — internal note, inappropriate for submission.
6. Line 726: Jargon ("maximum-rigor stage," "artifact-backed smoke evidence") will confuse journal readers.
7. Line 1123: "acceptance smoke" and "truth-shock microcase" are not standard terminology.
8. Table 7: RMSE values reported to 11 decimal places ("0.65473304686"). Round to 3--4 digits.
9. Lines 135 and 159: "This paper makes..." construction — use "I" as subject per style guidelines.

### Style Patterns to Fix Throughout

1. **"Planned for the final version" (5 instances)**: Remove or consolidate into a single "ongoing extensions" paragraph.
2. **Internal jargon**: Replace "acceptance smoke," "truth-shock microcase," "artifact-backed smoke evidence," "strict-smoke runtime" with standard academic language.
3. **Inconsistent model names**: Standardize to one name (e.g., "Smets--Wouters (2007)") and define "HLT" at first use.
4. **Inconsistent sampler terminology**: State clearly that the 3-parameter exercise uses NUTS-HMC and the 18-parameter exercise uses random-walk MH.
5. **"Percent" vs. "%"**: Use "percent" in running prose, "%" only in tables and parentheticals.
6. **Hyphenation**: Always hyphenate "first-order," "second-order," "higher-order" when used as adjectives.
7. **Editorializing language**: Remove "stark," "striking," "most interesting" — let findings speak for themselves.

---

## 2. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. **Training sample counts**: Abstract says "22,000"; Section 7.1 text says "23,000"; Table 3 notes say "22,080"; Table 2 notes say "23,000." Four different numbers for the same dataset. | Severity: **CRITICAL**

2. **Erroneous future date**: Table 1 note says "1954Q3--2032Q1, 311 quarters." The 2032Q1 date is in the future and contradicts the 1959Q1--2004Q4 sample used in estimation. | Severity: **CRITICAL**

3. **HMC vs. MH ambiguity**: Introduction and methodology describe HMC/NUTS; the 18-parameter estimation uses random-walk MH. The abstract says "50,000 Metropolis--Hastings draws" while the methodology section presents NUTS/HMC. | Severity: **CRITICAL**

4. **Training sample arithmetic**: 120 thetas × 180 periods = 21,600, not the claimed ~43,000 for the 18-parameter case. | Severity: **CRITICAL**

5. **Seed/prior sensitivity tables report wrong parameters**: Tables 12--13 test $\xi_p$, $\phi_\pi$, $\phi_y$ but the 3-parameter validation estimates $\sigma_A$, $\sigma_\mu$, $\sigma_R$. | Severity: **CRITICAL**

6. **N_sim = 10,000 then 9,000**: Line 469 claims 10,000 training samples, then line 482 corrects to 9,000 after accounting for burn-in. The initial claim is misleading. | Severity: **CRITICAL**

### Cross-Reference Errors

1. `\label{subsec:results_18param_switching}` is on a `\paragraph`, not a `\subsection`. References will resolve to parent subsection number.
2. Appendix E (`app:robustness_extended`) is never referenced from the main text — orphaned appendix.
3. Table `tab:18param_full` in Appendix E appears to report a different model specification than the 18-parameter estimation.
4. Duplicate bibliography entries: `gust2012effects` and `gust2017forward` contain identical content.
5. Eight uncited bibliography entries: betancourt2017conceptual, neal2011mcmc, robert2004monte, silver2016mastering, rezende2015variational, duarte2018machine, carroll2006method, ahnetal2018.

### Terminology Drift

1. **ROM1**: Also called "first-order perturbation," "linear approximation," "linearized model" — never formally defined.
2. **FOM**: Also called "SEP," "full-order model," "nonlinear model" — used interchangeably without explicit equivalence statement.
3. **HLT model** vs. **SW07** vs. **Smets--Wouters**: Three names for the same model, "HLT" undefined.
4. **"Filter-free"**: Applied to 3-parameter exercise (correct) but the 18-parameter exercise uses an inversion filter. Distinction is made but could confuse readers.

### Minor Inconsistencies

1. Baseline quadrature nodes: K=5 in main text (line 398) vs. K=3 in Appendix B (line 1534).
2. "12 hours" offline training (abstract) vs. "2 hours" training only (Section 4.2) vs. "~5 days" dataset generation. These describe different components but are not always distinguished.
3. HMC convergence percentages "67--80%" (line 458) not backed by any table.
4. Table 6: "50,000 draws with 20% burn-in (40,000 effective draws)" vs. text saying "50,000 posterior samples after a 10,000-iteration burn-in." Ambiguous whether total is 50k or 60k.
5. Gali model: n_s = 22 states (line 352) but Figure 6 caption says "23 variables" (line 1055).

---

## 3. Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)

1. **[Abstract]** | "Linear estimation misattributes nonlinear propagation to implausibly large shocks." | States as fact that the mechanism is misattribution. The paper shows FOM-ROM1 gaps and different posteriors, not a controlled experiment proving misattribution. | **Fix:** "presents evidence that linear estimation may misattribute..."

2. **[Abstract]** | "I show that the dominant source of this error is the financial accelerator." | "Show" implies proven. The decomposition measures approximation gaps, not estimation errors directly. | **Fix:** "I find that the dominant source of this approximation gap is the financial accelerator..."

3. **[Introduction, line 157]** | Mechanism paragraph presents the causal chain as demonstrated fact. | It is a theoretical explanation for the observed pattern. | **Fix:** "This pattern is consistent with the following mechanism:..."

4. **[Introduction, line 159]** | "nonlinear estimation is not primarily about the ZLB or price-setting nonlinearities" | Generalizes from one model (SW07) to all medium-scale DSGE models. | **Fix:** "In models with a financial accelerator channel, nonlinear estimation is primarily about..."

5. **[Section 7.1, line 989]** | "This finding contradicts the common intuition that the Kimball aggregator and the ZLB are the primary nonlinear channels." | The ZLB never binds at shock scale 0.1. Cannot "contradict" ZLB intuition without testing ZLB binding. | **Fix:** "At the shock scales examined, in which the ZLB does not bind, the financial accelerator dominates."

6. **[Section 7.2, line 1005]** | "asymmetric...consistent with the convex capital utilization cost creating asymmetric amplification" | Asymmetry could also reflect asymmetric shocks, structural breaks, or measurement error. | **Fix:** Add "though other explanations cannot be ruled out."

7. **[Section 7.4, line 1200--1202]** | "the nonlinear model accounts for low-frequency variation through level effects and nonlinear propagation" | With $\hat{R}$ up to 2.67, this interpretation is provisional at best. | **Fix:** Acknowledge explicitly that poor convergence makes this interpretation tentative.

### Generalization Issues

1. **Abstract + Introduction**: Claims about "medium-scale DSGE models" based on two specific models. Fix: Replace with model-specific claims.
2. **Line 171**: Claims architecture works "at the zero lower bound" but ZLB never binds in 18-parameter estimation. Fix: State this is designed for but not yet tested with binding ZLB in the full model.
3. **Line 1405 (Conclusion)**: Current-period shocks "negligible" finding is conditional on shock_scale=0.1. Fix: Add the conditioning statement.

### Missing Caveats

1. **Shock scale attenuation**: All 18-parameter results operate in a near-linear regime (shock_scale=0.1). This fundamental limitation deserves prominent discussion in the results section, not just a brief note in Section 8.
2. **Inversion filter conditioning bias**: ROM1-recovered shocks may be biased if the true DGP is nonlinear. Not discussed anywhere.
3. **Model misspecification**: The 68-point log-likelihood gap favoring the Kalman filter raises the possibility that nonlinear posterior shifts reflect worse fit, not better representation.
4. **Sample period ends before major crises**: Estimation excludes the 2008 financial crisis, the most important episode for testing nonlinear methods.
5. **No particle MCMC comparison**: The gold-standard benchmark is listed as "Planned" but never computed.
6. **Seed sensitivity tables report wrong parameters**: This appears to be an error, not just a missing caveat.

### Computational Claims Needing Verification

1. "Offline training costs 12 hours" not traceable to any reported measurement.
2. "A single SEP solve takes 0.8 seconds" never sourced to a benchmarking table.
3. HMC convergence rates "67--80%" not backed by any table.
4. Table A.9 (18-parameter validation): numbers suspiciously clean — unclear if actually computed.

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors

1. **Line 545**: Inversion filter objective has ambiguous notation ($Hg$ could be $H$ times $g$ or $H$ applied to $g$; scalar $\sigma_y$ denominator applied to vector numerator).
2. **Line 599 vs. line 336 vs. line 1469**: Three different Taylor rule specifications across identification section, main model, and appendix. The identification section omits interest-rate smoothing $\rho_R$. The appendix omits it entirely.
3. **Line 625--627**: SEP discretization error bound $\mathcal{O}(K^{-r/n_\varepsilon})$ conflates convergence rate with constant. Standard product-rule GH maintains $\mathcal{O}(K^{-r})$ per dimension with exponentially growing constant.
4. **Line 811 vs. Table 3**: Text says "six economic blocks" but Table 3 lists seven (including Flex-Price Block).
5. **Lines 1186**: Table notes say "20% burn-in (40,000 effective draws)" but text says "50,000 posterior samples after 10,000 burn-in" — these differ by 10,000 draws.

### Notation Inconsistencies

1. **$\varepsilon$ / $\varepsilon^{\text{surr}}$**: Structural shocks and surrogate error both use $\varepsilon$. → Use $e^{\text{surr}}$ or $\delta^{\text{surr}}$ for surrogate error.
2. **$\alpha$**: Used for both Adam learning rate (line 504) and Newton step size (line 406). → Use $\eta$ for learning rate.
3. **$w$**: Used for both network weights and quadrature weights. → Use $\omega$ for quadrature weights.
4. **$R$**: Used for interest rate ($R_t$), SEP residual ($R(Y;\theta)$), and Gelman--Rubin statistic ($\hat{R}$). → Use $\mathcal{R}$ for residual.
5. **$\Sigma_\eta$ vs. $\Sigma_y$**: Two symbols for measurement error covariance. → Standardize.
6. **$\rho_R$ vs. $\rho_r$**: Two notations for interest-rate smoothing. → Standardize.
7. **$g_\theta$ vs. $g_\theta^{\text{SEP}}$ vs. $g_\theta^{\text{FOM}}$**: Three names for the same function. → State equivalence explicitly.

### Undefined Notation

1. $\Phi$, $B$ in GIRF discussion (line 451) — state transition and shock impact matrices never defined.
2. $\Sigma_{s_0}$ (line 573) — initial state covariance never specified.
3. $\lambda$ (line 626) — "rate of mean reversion" not related to model primitives.
4. $\Sigma_{\text{surr}}$ (line 634) — surrogate error covariance never calibrated numerically.
5. $H_\theta$ (line 1068) — policy-function Hessian collides with observation matrix $H$.
6. $m_t$ (line 523) — stress metric never formally defined.
7. $\bar{\beta}$ (line 848) — appears in investment FOC without definition.
8. $\xi_t$ (line 848) — marginal utility in investment FOC collides with $\xi_p$ (Calvo probability).
9. $\lambda_f$ (line 902) — steady-state price markup not defined or calibrated.

### LaTeX Math Formatting

1. Line 344--348: Double ampersand `&&` in `align` environment creates extra column.
2. Line 1192: `\label{subsec:...}` on a `\paragraph` instead of `\subsection`.
3. Lines 838, 844, 848: Key equations (utilization cost, adjustment cost, investment FOC) are unnumbered `\[...\]` but discussed extensively — should be numbered.
4. Line 829: `\multicolumn{4}` in a 5-column table — should be `\multicolumn{5}`.
5. Line 120: `\vspace{-0.5cm}` outside document body.
6. Line 1565: $R$ residual function overloaded with interest rate notation.

---

## 5. Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

1. **Table 1 (crisis vs. normal)**: Variance share blanks for 4 of 7 rows; erroneous "1954Q3--2032Q1" date range.
2. **Table 3 (equation-block)**: Column count mismatch (5 specifiers, 4 headers); notes say "22,080" while text says "23,000."
3. **Table 6 (param region RMSE)**: No sample size or dataset specification in notes.
4. **Table 7 (regime performance)**: Raw floating-point precision (11 digits); notes reference TOML files.
5. **Table 8 (18-param posterior)**: "Init" column ambiguous — rename to "Calibration" or "Starting Value."
6. **Table 9 (posterior comparison)**: Missing Kalman 90% CI column — reader cannot assess posterior overlap.
7. **Table 10 (benchmarks)**: Two "Planned" rows with missing values; "Reference" under LL is unclear.
8. **Tables 12--13 (seed/prior sensitivity)**: Report wrong parameters (see Critical Inconsistencies).
9. **Table 14 (prior sensitivity)**: Never cited with `Table~\ref{}` in running text.
10. **Appendix Tables A1--A4**: Never formally cross-referenced in text.

### Figures with Missing or Incomplete Notes

1. **Figure 1 (nonlinearity decomposition)**: Axis labels/units not mentioned in caption.
2. **Figure 3 (nonlinearity landscape)**: Color bar range not specified.
3. **Figure 4 (ROM1 forecast errors)**: Y-axis units not stated in caption.
4. **Figure 6 (Gali vs. HLT)**: Caption says "$1$--$5\sigma$" but text says "$1\sigma$ to $20\sigma$."
5. **7 orphaned figure files**: `fig_nonlinearity_density.pdf`, `fig_r2_decomposition.pdf`, `fig_quintile_error.pdf`, `fig_nonlinearity_landscape.pdf`, `fig_per_variable_error.pdf`, `fig_improvement_by_observable.pdf`, `fig_crisis_vs_normal_rmse.pdf` exist on disk but are never included.
6. **4 generated table files**: `generated/table_hlt_realdata_*.tex` exist but are never `\input{}`-ed. Some contain degenerate data (NaN values).

### Cross-Reference Issues

1. `tab:prior_sensitivity` defined but never cited by number.
2. `tab:hyperparameter_grid`, `tab:sample_length_sensitivity`, `tab:architecture_comparison` defined in appendix but never formally referenced.
3. `\label{subsec:results_18param_switching}` on a `\paragraph` — reference resolves to parent subsection number.
4. Appendix E (`app:robustness_extended`) never referenced from main text.

### Formatting Inconsistencies

1. Table notes: mixed `\multicolumn` vs. `\smallskip\footnotesize` format.
2. Decimal precision: Table 7 uses 11 digits; other tables use 3.
3. Table placement: all use `[h]`; consider `[t]` for consistency with figures.
4. No explicit appendix table numbering convention (A1, A2, etc.).

---

## 6. Contribution & Referee Assessment

### Part 1 — Central Contribution

The paper proposes a practical estimator for nonlinear DSGE models combining SEP solution, a parameter-conditional neural surrogate, and a regime-switching gate, and claims to demonstrate that the financial accelerator is the dominant source of nonlinearity in estimated medium-scale models.

The individual components are not new: SEP (Fair 1983), neural surrogates for DSGE (Koop et al. 2022), filter-free HMC (Childers 2022, Farkas 2020). The contribution is a systems integration plus the nonlinearity decomposition finding.

**Rating:** Incremental

The paper assembles known components into a working pipeline and produces a useful nonlinearity decomposition. However, the core validation is limited to a 3-parameter exercise on a small model, the 18-parameter real-data results are non-converged ($\hat{R}$ up to 3.62), and the gold-standard comparison (particle MCMC with exact SEP) is listed as "Planned." The paper demonstrates a promising workflow but does not deliver the validated methodological contribution Econometrica requires.

### Part 2 — Identification and Credibility

Five serious threats to validity:

1. **Non-converged 18-parameter chains**: $\hat{R}$ up to 3.62 renders headline posterior comparisons unreliable.
2. **ZLB never binds in 18-parameter training data**: The OBC capability is aspirational, not demonstrated, at the 18-parameter scale.
3. **Validation and estimation models differ**: 3-parameter Gali model vs. 18-parameter SW07 model. No converged 18-parameter synthetic recovery.
4. **Ad hoc regime-switching gate**: No sensitivity analysis to gate threshold.
5. **Surrogate error not propagated into posterior uncertainty**: Per-step RRMSE of 0.0012 compounds over 184 periods.

A skeptical seminar participant would ask: "Your headline results have $\hat{R}$ of 3.62 — how can you draw inference? And the ZLB never binds — what exactly is the OBC contribution?"

### Part 3 — Analyses: Required and Suggested

**Required:**

1. **Converged 18-parameter MCMC chains** ($\hat{R} < 1.1$ for all parameters). Without this, Table 9's posterior comparison is meaningless.
2. **18-parameter synthetic data recovery with known truth** and converged chains. The 3-parameter validation on a different model does not establish method validity at scale.
3. **Training data with binding ZLB episodes**. The paper's title mentions OBCs but the ZLB never binds in the 18-parameter specification.
4. **Computed comparison with at least one alternative nonlinear estimator** (not just projected runtime). At minimum, second-order perturbation + particle filter on the 3-parameter model.
5. **Sensitivity of results to regime-switching gate threshold**.

**Suggested:**

1. Accumulated surrogate error analysis over 184-period sequential simulation.
2. Posterior predictive checks (do generated data resemble actual US data?).
3. Cross-validation of nonlinearity decomposition across model calibrations.
4. Formal analysis of inversion filter's shock recovery bias.
5. Extension of data sample through the Great Recession.

### Part 4 — Literature Positioning

**Missing citations:**
- Fernández-Villaverde, Guerrón-Quintana, and Rubio-Ramírez (2015) — nonlinear estimation with ZLB via particle filtering
- Boehl (2022) — efficient solution for OBCs
- Guerrieri and Iacoviello (2015) — OccBin, the workhorse for OBC in policy institutions
- Amisano and Tristani (2010) — nonlinear estimation with regime switching

**Framing suggestion:** The introduction tries to serve two masters: methodology and the nonlinearity decomposition. The decomposition is more novel. Lead with it more forcefully and use the methodology as the enabling technology.

### Part 5 — Journal Fit and Recommendation

**Preliminary recommendation:** Desk reject

The paper has promising elements — the nonlinearity decomposition is genuinely interesting, and the computational architecture is sensible — but it is not ready for Econometrica referees. The gap between what is promised and what is delivered is too large: non-converged chains, no binding ZLB in 18-parameter training, multiple "planned for the final version" items.

**To reach Econometrica's standard:** (1) Complete computational work: converged chains, binding ZLB, computed benchmarks. (2) Provide formal approximation-theoretic bounds on posterior distortion from surrogate error. (3) Demonstrate qualitatively different economic conclusions from nonlinear estimation. (4) Shorten substantially.

**Best alternative outlets:** *Journal of Econometrics* (with complete results), *Quantitative Economics*, or *Journal of Economic Dynamics and Control*.

### Part 6 — Questions to the Authors

1. **Table 9 reports posterior comparisons but $\hat{R}$ exceeds 3.6 for some parameters. On what basis do you interpret these posteriors as informative? Would you report these as evidence of convergence in a simulation study?**

2. **The 18-parameter training data uses shock scale 0.1, preventing the ZLB from binding. What exactly is the nonlinear surrogate capturing? Would a second-order perturbation achieve the same correction at far lower cost?**

3. **Table A.9 reports an 18-parameter synthetic recovery with all $\hat{R} < 1.01$ and neatly rounded errors. Was this exercise actually computed, or are these projected values?**

4. **The inversion filter recovers shocks using ROM1, but the surrogate evaluates nonlinear periods. Have you quantified the bias from conditioning on linearly-recovered shocks?**

5. **The nonlinearity decomposition uses shock scale 0.1. At crisis-relevant displacements, does the investment block still dominate? Does the monetary policy block's 0.1% share increase when the ZLB binds?**

6. **The regime-switching log-likelihood is 68 points below the Kalman filter's. In standard model comparison, this favors the linear model. Is it possible that the posterior shifts reflect compensating distortions rather than genuine nonlinear effects?**

7. **The training grid has only 120 points in 18 dimensions. What is the interpolation accuracy between grid points in regions of high posterior density?**

---

## Priority Action Items

**CRITICAL** (must fix — could cause desk rejection):

1. **Converge the 18-parameter MCMC chains** ($\hat{R} < 1.1$). Without this, headline results are meaningless. (Agent 6)
2. **Fix the training sample count inconsistency** (22,000 / 22,080 / 23,000 / arithmetic mismatch 120×180=21,600≠43,000). (Agent 2)
3. **Fix the erroneous future date** "1954Q3--2032Q1" in Table 1 notes. (Agent 2)
4. **Fix or replace the seed/prior sensitivity tables** that report wrong parameters ($\xi_p$, $\phi_\pi$, $\phi_y$ instead of $\sigma_A$, $\sigma_\mu$, $\sigma_R$). (Agent 2)
5. **Generate training data with binding ZLB episodes** for the 18-parameter model. The OBC claim is unsupported without this. (Agent 6)

**MAJOR** (should fix — will be raised by referees):

6. **Qualify causal language**: "misattributes" → "may misattribute"; "I show" → "I find evidence that"; "contradicts" → "qualifies." (Agent 3)
7. **Resolve HMC vs. MH ambiguity**: Clearly state which sampler is used in each exercise throughout. (Agents 1, 2)
8. **Add missing Kalman CI column** to Table 9 so readers can assess posterior overlap. (Agent 5)
9. **Fix three inconsistent Taylor rule specifications** across main text, identification section, and appendix. (Agent 4)
10. **Compute at least one alternative benchmark** (2nd-order + particle filter on 3-param model) rather than projecting runtime. (Agent 6)
11. **Discuss inversion filter conditioning bias** — ROM1-recovered shocks may be biased under nonlinear DGP. (Agent 3)
12. **Add prominent caveat about shock scale attenuation** (0.1x) and its implications for all 18-parameter results. (Agent 3)

**MINOR** (polish — improves paper quality):

13. Remove all "planned for the final version" notes (5 instances) or consolidate into conclusion.
14. Replace internal jargon ("acceptance smoke," "truth-shock microcase," etc.) with standard language.
15. Standardize notation: define ROM1/FOM at first use, resolve $\varepsilon$ overloading, fix $\alpha$/$w$/$R$ collisions.
16. Add formal cross-references for Tables 14, A1--A4 that are defined but never cited.
17. Round Table 7 values to 3--4 significant digits.
18. Add axis labels/units to figure captions (Figures 1, 3, 4, 5).
19. Clean up 8 uncited bibliography entries and 1 duplicate (gust2012effects / gust2017forward).
20. Fix `\multicolumn` width mismatches in Tables 3, 4, 7.
