# Pre-Submission Referee Report

**Paper**: Multi-Fidelity Estimation of DSGE Models with Occasionally Binding Constraints
**Author**: Matyas Farkas (IMF)
**Date**: 2026-03-25
**Review Standard**: Econometrica

---

## Overall Assessment

The paper proposes a novel multi-fidelity architecture that combines stochastic extended path (SEP) solutions with a parameter-conditional neural network surrogate and a regime-switching gate for Bayesian estimation of nonlinear DSGE models. The strongest finding -- that 69% of the nonlinearity gap in a Smets-Wouters model concentrates in the investment/capital block rather than the price Phillips curve -- is clean, descriptive, and potentially interesting for the macro literature. However, the headline 18-parameter estimation has not converged (R-hat up to 2.67, median ESS = 65), the surrogate log-likelihood bias bound (442 nats) exceeds the Kalman-RS gap it attempts to explain (68 nats), and the ZLB never binds in any reported result despite being central to the paper's framing. The abstract and methodology describe filter-free HMC as the sampler, but the 18-parameter estimation actually uses random-walk Metropolis-Hastings -- a critical inconsistency.

**Preliminary Recommendation**: Substantial revision required. In its current state, the paper would likely receive a desk reject at Econometrica; Journal of Econometrics or Quantitative Economics are more realistic targets after the required revisions are completed.

---

## 1. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. **Line 137 (Abstract)** | The abstract is a single paragraph of approximately 250 words in `\footnotesize`. This is extremely long for a journal abstract. Consider splitting into 2-3 shorter paragraphs or trimming to ~150 words.

2. **Line 157 vs. Line 685** | The abstract rounds to "69 percent" while the body uses "68.9 percent" and "68.8 percent." Standardize: either round consistently to "69 percent" in prose and use exact figures only in tables, or always use "68.9."

3. **Line 215** | Display equation on lines 211-213 ends without punctuation, but the next line starts a new sentence. Add a comma after the closing `\right]` and lowercase "The" to "the" to make the sentence flow.

4. **Line 250** | "This paper keeps the nonlinear structure at every stage." Third-person self-reference conflicts with the first-person voice used elsewhere ("I solve...", "I develop..."). Fix: "I keep the nonlinear structure at every stage."

5. **Line 331** | "regime switches" is vague and potentially confusing with the paper's own "regime-switching gate." Should be "regime changes" or "transitions to binding constraints."

6. **Line 393** | Missing comma after introductory prepositional phrase. Fix: "Under perturbation, this collapses to..."

7. **Line 469** | The general description mentions shadow-rate gap and shock magnitude as stress metric examples, but neither is actually used. The implementation uses state displacement. Fix: Clarify immediately that "In the current implementation, $m_t = \|s_t - \bar{s}(\theta)\|$."

8. **Line 497** | Element-wise division by $\sigma_y$ and $\sigma_\eps$ inside norms is not standard notation. Use Mahalanobis norm notation or clarify.

9. **Line 593** | "Schmitt-Grohe & Uribe" -- spell out "and" in running prose. This name also lacks a citation.

10. **Lines 687, 942** | "Gali (2015)" should be "Gal\'i (2015)" with the accent, and preferably as a `\citet{}` citation.

11. **Line 1068** | `\paragraph` carrying a `subsec` label prefix will confuse cross-references. Promote to `\subsection` or fix the label.

12. **Line 1172** | Data sample discussion needs explicit clarification that this is the *extended* sample (through 2025Q1), not the baseline (1959Q1-2004Q4).

13. **Line 1244** | Sentence lacks a subject; it is a dangling continuation of a display equation.

14. **Line 1310** | Circular reference: Appendix A says the SW07 model is "described in that section" (Section 7.1), while Section 7.1 points back to "Appendix~\ref{app:sw07_model}."

15. **Line 1406** | "HLT" acronym first and only defined in the appendix. Define at first use in the main text (line 264 or Section 4).

16. **Line 1440** | SW07 Taylor rule should note whether it is log-linearized, given the paper's emphasis on nonlinear estimation.

17. **Line 1687** | Symbol $\delta$ used for both acceptance rate target and nonlinearity measure ($\delta_j$). Use a different symbol to avoid overloading.

### Minor Issues

18. **Line 687** | "Financial accelerator" used loosely for investment adjustment/capital utilization costs. SW07 does not have a financial accelerator in the Bernanke-Gertler-Gilchrist sense. Fix: use "investment accelerator" or "capital accumulation channel."

19. **Line 823** | $S''$ is described as quadratic but is actually a constant (second derivative of the quadratic $S$). Fix: "the quadratic $S$ compounds with the Euler wedge through $S'$."

20. **Line 905** | 68.8% is share of transition-function gap; 53.2% is share of forecast error variance. Flag as approximate comparison.

21. **Line 942** | "75 percent" appears without clear definition of what it measures vs. the "69%" reported elsewhere.

22. **Line 1184** | Arithmetic error: at a half-life of 53 quarters, initial condition has decayed to ~0.77 after five years (20 quarters), not "negligible." Change to "approximately 15 years."

### Style Patterns to Fix Throughout

- **Inconsistent first-person voice:** "This paper keeps..." (250), "The paper links..." (169), "This paper develops..." (1283). Search for "this paper" / "the paper" and rewrite in first person.
- **Bare text citations:** Mix of bare "Smets and Wouters (2007)" and proper `\citet{}`. Replace all with natbib commands.
- **"Roughly" / "approximately" overuse:** 14+ combined instances. Vary with "about," "around," or the tilde "$\approx$."
- **Passive voice:** Several avoidable instances ("Synthetic data is generated..."). Convert to active where author is the agent.
- **Number formatting:** "7 shocks" (264), "4 performance cores" (1723) should spell out single-digit numbers in running prose.

---

## 2. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. **MH vs. HMC sampler mismatch** | The abstract, methodology (Section 4.4), and conclusion describe filter-free HMC as the sampling method and mention "50,000 Metropolis-Hastings draws." But the 18-parameter estimation (Section 7.6) uses random-walk MH with an inversion filter, not HMC. The 3-parameter validation uses HMC. This is never reconciled. | Severity: CRITICAL

2. **RRMSE threshold inconsistency** | Stated as < 0.001 (Algorithm 1 and Section 4.3), < 0.002 (Section 6.5), and < 0.003 (Table 12 note). The 18-parameter case achieves ~0.0012, which exceeds the 0.001 threshold. | Severity: CRITICAL

3. **"Tenfold reduction" arithmetic** | Text says "$\sigma_b$ revises from 1.85 to 0.12 (tenfold reduction)." Actual ratio: 1.851/0.122 = 15.2x, not tenfold. | Severity: CRITICAL

4. **Six vs. seven blocks** | Text says "six economic blocks" but Table 3 contains seven rows including a "Flex-Price Block" not in the enumerated list. | Severity: CRITICAL

5. **Seed/prior sensitivity tables** | Tables 10-11 report sensitivity for ($\xi_p, \phi_\pi, \phi_y$) but the main 3-parameter validation estimates ($\sigma_A, \sigma_\mu, \sigma_R$). These appear to be from different exercises. | Severity: CRITICAL

### Cross-Reference Errors

6. **Line 382**: `\eqref{eq:taylor_zlb}` references a nonexistent label. The actual label is `eq:taylor_zlb_full` (line 1351). Will render as "??" in compiled PDF.

7. **Lines 827-887**: Seven `\includegraphics` calls reference figure files that do not exist in `docs/paper/figures/`. Masked by `[draft]` mode on graphicx.

8. **Generated table files** (`table_hlt_realdata_posterior.tex`, etc.) are not `\input`'ed anywhere and contain only 3 parameters with zero variance from a different run.

9. **Tables `tab:sample_length_sensitivity` and `tab:prior_sensitivity`** are defined but never referenced with `\ref{}`.

### Terminology Drift

10. **"FOM" vs "SEP FOM" vs "SEP" vs "direct SEP"** -- four terms used interchangeably. Standardize to "FOM (SEP)" on first use and "FOM" thereafter.

11. **"ROM1" vs "linear" vs "first-order perturbation" vs "Kalman"** -- used interchangeably. Distinguish "ROM1" (solution method) from "Kalman filter" (inference method).

12. **"HLT model" vs "SW07 model" vs "Smets-Wouters model"** -- all three refer to the same model. Use "SW07-HLT" consistently.

### Minor Inconsistencies

13. Per-evaluation cost: 147 seconds (Section 6.3) vs. 150 seconds (Table 7). Yields 85 vs. 87 machine-days.

14. "Kimball price curvature nearly triples" -- actual ratio 2.65x, closer to "roughly 2.7x" than "nearly triples."

15. Gali negative-control shock magnitudes: "$15\sigma$" (line 687) vs. "$20\sigma$" (line 1215), with different comparison pairs (FOM-ROM1 vs. ROM1-ROM2).

16. Offline training cost: "12 hours" (abstract/intro) vs. "roughly two hours" (Section 6.5). The latter appears to describe the 3-parameter workflow.

17. "(311 quarters)" on extended sample -- 1959Q1 to 2025Q1 is 265 quarters; 311 likely refers to CSV row count.

18. Sparse tree branching horizon: $H_1 = 4$ (appendix) vs. $H_\text{branch} = 2$ (main text) for the 3-shock model.

---

## 3. Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)

1. **[Abstract, line 137]** | "attribute nonlinear propagation to implausibly large shocks" implies linear model misspecification *causes* inflated shocks. Only the 3-parameter synthetic exercise verifies this; on real data, only different posteriors are shown. Fix: "may be associated with" or qualify that on real data the pattern is consistent but not definitively identified.

2. **[Abstract, line 137]** | "consistent with capturing investment dynamics through endogenous amplification" -- no formal test distinguishes (a) genuine amplification, (b) surrogate error distorting the posterior, (c) inversion-filter conditioning effects. The 442-nat bias bound exceeds the Kalman-RS gap. Fix: "potentially consistent with ... though surrogate approximation error prevents definitive attribution."

3. **[Introduction, line 159]** | "so it compensates by inflating estimated shock magnitudes" asserts a causal mechanism. This is a hypothesis, not an established fact. Fix: "so the linear model *may* compensate."

4. **[Section 6.1, line 687]** | "confirming the financial accelerator as the dominant nonlinearity source" -- comparison of two different models differing in many dimensions beyond the investment block. Fix: "consistent with" or "supporting the interpretation that."

5. **[Section 6.1, line 813]** | "confirmed in Section [...]" -- from chains with R-hat > 1.1 for 11 of 18 parameters. Fix: "directionally *consistent with*."

6. **[Section 6.3, line 914]** | "This is exactly what a nonlinear financial accelerator predicts" -- correlation, not causation. Alternative explanations (time-varying parameters, structural breaks) not tested.

7. **[Section 6.6, line 1172]** | "absorbs its inability to represent state-dependent propagation" stated as fact. Alternative: the model itself is misspecified for COVID.

8. **[Section 6.1, line 690]** | Policy claim about learning vs. nonlinear propagation goes far beyond anything identified in the paper. Either delete or frame as theoretical distinction beyond this paper's scope.

### Generalization Issues

9. The 69% investment-block share is computed for one model, one calibration, one shock scale (0.1) where the ZLB never binds. The abstract presents this as a general finding about "medium-scale DSGE models."

10. The 1959Q1-2004Q4 sample deliberately excludes the most nonlinear episodes (2008 crisis, COVID). The abstract does not caveat this.

11. **[Section 5.2, line 542]** | Information ordering $\mathcal{I}_T^{\text{NL}} \succeq \mathcal{I}_T^{\text{lin}}$ -- the empirical evidence ($\sigma_b$ CI *widening* under nonlinear model) actually contradicts this claim. Widening is consistent with a flatter or multimodal posterior, not additional identification.

12. **[Section 6.3, line 905]** | "The connection is direct: 68.8 percent (training) versus 53.2 percent (real data)" -- these measure different things (transition function gap vs. forecast error variance). The connection is suggestive, not direct.

### Missing Caveats

13. **Model misspecification vs. nonlinearity** | Posterior differences reflect the joint effect of nonlinear correction, inversion-filter conditioning, surrogate approximation error, and differential MCMC convergence. The compound identification problem is never stated clearly.

14. **Inversion filter inconsistency** | ROM1-recovered shocks evaluated under nonlinear surrogate creates systematic inconsistency. Acknowledged in methodology but not in results section.

15. **No second-order perturbation benchmark** | ROM2+Kalman estimation would help separate "any nonlinearity matters" from "global nonlinearity matters." Conspicuous absence.

16. **Training shock scale** | 0.1 means training shocks 10x smaller than calibrated volatilities. Fraction of real-data filtered states within training convex hull not reported.

17. **MH vs. HMC inconsistency** | Abstract/methodology describe HMC; 18-parameter estimation uses MH. Reader misled about mixing properties.

18. **Measurement error calibration** | Described as calibrated but no calibration exercise reported. 18-parameter estimation does not specify measurement error values.

### Mechanism Claims Stated as Facts

19. **[Line 687]** | "The mechanism is the interaction of three nonlinear functions" -- should be "I attribute" or "the proposed mechanism is."

20. **[Line 157]** | 69% and 0.2% shares are specific to this calibration, shock scale, and block assignment. The implicit framing is that these are structural features of the model class.

### Unsupported Robustness Claims

21. Initial-condition robustness tests described in Section 5.4 are explicitly acknowledged as "not part of the current validation results."

22. Seed sensitivity test uses different parameters than the validation exercise it purports to check.

23. "The 18-parameter posterior shifts are directionally robust" (conclusion) -- directional robustness never demonstrated across restarts or prior specifications for the 18-parameter case.

### Statistical vs. Economic Significance

24. Posterior shifts reported without discussion of economic significance (impulse responses, variance decompositions, policy-relevant quantities).

25. $38\sigma$ shocks during COVID reported without showing what the linear model predicts under these magnitudes.

### Hedging Failures: Underconfident

26. The 69% decomposition is based on 22,080 deterministic comparisons (complete enumeration), not a statistical estimate. The paper should be more confident about *what the decomposition shows* while hedging *what it means*.

27. The 3-parameter validation results are very clean (recovery within 2%, R-hat < 1.003, ESS > 3,800). The paper could state more directly this is strong evidence the method works in the small-parameter case.

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors

1. **Steady-state labor supply exponent (line 1397)** | Writes $N^{ss} = [...]^{1/(1+\varphi)}$. From the FOC $N_t^\varphi = (W_t/P_t) C_t^{-\sigma}$, the correct exponent is $1/\varphi$, not $1/(1+\varphi)$.

2. **"Six blocks" vs. seven rows** | Text lists six blocks (i)-(vi); Table 3 has a seventh "Flex-Price Block" row.

3. **Broken cross-reference** | `\eqref{eq:taylor_zlb}` at line 382 -- no such label exists. Will render as "??".

4. **State dimension discrepancy** | 23 variables (line 262) vs. 22 network inputs (line 1564: $\mathbb{R}^{28}$ with "22 state + 3 shock + 3 parameter"). Clarify which variable is excluded.

5. **Pseudocode mismatch** | Appendix D.1 pseudocode samples 6 parameters but the 3-parameter validation estimates only 3. Should match or be labeled as a different exercise.

6. **Seed sensitivity table parameters** | Table 10 reports $(\xi_p, \phi_\pi, \phi_y)$ but validation estimates $(\sigma_A, \sigma_\mu, \sigma_R)$. Must reconcile.

### Notation Inconsistencies

7. **$\sigma_R$ vs. $\sigma_r$** for monetary shock standard deviation. Pick one case consistently.

8. **$\xi$ double duty** | $\xi_t$ = marginal utility of consumption in SW07; $\xi_p$, $\xi_w$ = Calvo probabilities. Same Greek letter for unrelated objects.

9. **`\eps` vs. `\varepsilon`** | Both used (~52 and ~36 occurrences). Rendered output identical but source inconsistent.

10. **$\sigma$ overloading** | Risk aversion (Gali model) and shock standard deviations ($\sigma_a$, $\sigma_b$, etc.).

11. **$\delta$ overloading** | Capital depreciation rate, per-period KL contribution, and nonlinearity measure $\delta_j$.

### Undefined Notation

12. **$\bar{\beta}$ (line 1424)** | Used in Tobin's $q$ equation, never defined. Presumably $\bar{\beta} = \beta \gamma^{-\sigma_c}$.

13. **$p_t^k$ (line 1424)** | Relationship to Tobin's $q$ not specified.

14. **$\Sigma_{s_0}$ (line 520)** | Initial state prior covariance never specified in main text (pseudocode uses $0.01 I$).

15. **$\sigma_y$ in eq. (11) (line 556)** | Scalar vs. matrix ambiguity with $\Sigma_y$.

16. **$\bar{s}(\theta)$ (line 302)** | Steady-state function used repeatedly but never formally defined.

17. **$\varepsilon_w$ (line 1436)** | Kimball wage curvature mentioned but never appears in any table or among calibrated values.

18. **$\|H\|$ in eq. (11) (line 556)** | Norm type not specified (operator? Frobenius?).

### Equation Numbering

19. Six numbered equations never referenced: `eq:policy`, `eq:likelihood`, `eq:transition_map`, `eq:augmented_posterior`, `eq:loglik`, `eq:nonlinearity_measure`. Consider unnumbering.

20. Duplicate labels on same subsection (lines 401-402): `\label{subsec:surrogate}` and `\label{subsec:surrogate_training}`.

21. `\label{subsec:results_18param_switching}` on a `\paragraph` (line 1068) -- labeling convention misleading.

### LaTeX Math Formatting

22. **Table 3 column count** | `{lcccc}` declares 5 columns but only 4 are used. Phantom empty column.

23. **$L^2$ vs. $\ell^2$** | Table 3 column header says "$L^2$ Share" but measures squared $\ell^2$ norm, not $L^2$ function-space norm.

24. **`\vspace{-0.5cm}` outside document body (line 122)** | Between `\author` and `\date` in preamble; may not render as intended.

---

## 5. Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

| Table | Issue | Suggested Fix |
|---|---|---|
| `tab:equation_block` | Column spec `{lcccc}` has phantom 5th column; only 4 data columns | Change to `{lccc}` |
| `tab:equation_block` | L2 shares sum to 100.1% | Add note: "Shares may not sum to 100 due to rounding" |
| `tab:equation_block` | N=22,080 only in notes | Add sample size to column header or visible row |
| `tab:error_by_distance` | 5 x 4,600 = 23,000 vs. stated 22,080 | Clarify quintile sizes (~4,416 per quintile) |
| `tab:crisis_vs_normal` | Notes say T=184 but body discusses 301 post-burn-in quarters | Reconcile sample sizes |
| `tab:crisis_vs_normal` | Variance share missing for 4 of 7 observables | Report all or add note explaining suppression |
| `tab:regime_performance` | RMSE of exactly 0.0 for direct SEP; ROM1 values with 11 decimal places | Round to 3-4 sig figs; add note why SEP RMSE is zero |
| `tab:18param_switching_posterior` | No convergence diagnostic column despite R-hat up to 2.67 | Add R-hat or ESS columns |
| `tab:benchmarks` | "Projected" rows have no confidence bounds | Clarify assumptions behind projections |
| `tab:seed_sensitivity` | Notes inside `tabular` as rows; inconsistent with other tables | Standardize to `\smallskip\footnotesize` after `\end{tabular}` |
| `tab:prior_sensitivity` | Not cross-referenced in text | Add `Table~\ref{tab:prior_sensitivity}` |
| `tab:hyperparameter_grid` | Not cross-referenced in text | Add reference |
| `tab:sample_length_sensitivity` | Not cross-referenced in text | Add reference |
| `tab:architecture_comparison` | Not cross-referenced in text | Add reference |
| `tab:18param_full` | Title says "18-Parameter" but only 16 rows | Add 2 missing parameters or correct title |
| `tab:mcmc_diagnostics` | "1,000 adaptation draws" (notes) vs. "2,000 draws across 4 chains" (text) | Clarify per-chain vs. total |

### Missing Figure Files (7 files)

The following figure files referenced by `\includegraphics` do not exist in `docs/paper/figures/`:
- `fig_sep_vs_rom1_eqs_investment.pdf`
- `fig_sep_vs_rom1_eb_investment.pdf`
- `fig_sep_vs_rom1_epinf_investment.pdf`
- `fig_nonlinear_correction_surface_eqs.pdf`
- `fig_irf_asymmetry_eqs.pdf`
- `fig_sep_all_vars_eqs_4sigma.pdf`
- `fig_nonlinear_amplification_ratio.pdf`

Currently masked by `[draft]` mode on graphicx. Must be generated before submission.

### Figure Caption Issues

| Figure | Issue |
|---|---|
| `fig:nonlinearity_decomposition` | Caption says "Panel (a)" and "Panel (b)" but includes single PDF -- verify sub-panels |
| `fig:gali_vs_hlt_nonlinearity` | Caption says "$1$-$5\sigma$" but text says up to $20\sigma$ |
| `fig:rom1_forecast_errors` | Caption lacks data source dates |
| `fig:rolling_rmse` | Caption lacks window specification and data vintage |
| All IRF figures (figs 4-10) | No notes on y-axis units (percentage deviations, log-deviations, or levels?) |

### Unused Figure Files on Disk

6 PDF files in `figures/` never included: `fig_nonlinearity_density.pdf`, `fig_r2_decomposition.pdf`, `fig_quintile_error.pdf`, `fig_per_variable_error.pdf`, `fig_improvement_by_observable.pdf`, `fig_crisis_vs_normal_rmse.pdf`.

### Formatting Inconsistencies

- Table notes placement: some inside `tabular`, most outside
- "Notes:" vs. `\textit{Notes}:` inconsistent across tables
- Table placement specifiers: most `[h]`, some `[t]`; recommend `[htbp]`
- N reporting: some tables in column, some in notes, some not at all

---

## 6. Contribution & Referee Assessment

### Part 1 --- Central Contribution

**Claim in one sentence:** The paper proposes a multi-fidelity estimator that uses a neural network surrogate trained on SEP residuals relative to first-order perturbation, combined with a regime-switching gate and filter-free HMC, and documents that 69% of the nonlinearity gap in a Smets-Wouters model concentrates in the investment/capital block.

**Is the finding genuinely new?** The *architecture* combines existing ideas (SEP, neural surrogates, filter-free inference, residual learning) in a novel way. The *decomposition* finding is new and potentially interesting. The *method* is competent engineering but not a fundamental conceptual advance. Maliar, Maliar, and Winant (2021) use deep learning for nonlinear DSGE solution; Naubert (2025) uses mixture density networks for transition functions; Koop et al. (2022) approximate the likelihood with neural networks.

**Rating:** Incremental. The paper combines known techniques and produces an interesting descriptive finding, but does not deliver a new theorem, identification result, or credible empirical finding. The 18-parameter estimation has not converged, and the surrogate bias bound (442 nats) exceeds the effect it tries to explain (68 nats).

### Part 2 --- Identification and Credibility

**What would a skeptical econometrician say at a seminar?**
- "Your 18-parameter chains have not converged. R-hat = 2.67 for one parameter. These are not posterior estimates; they are snapshots from a non-converged chain."
- "Your surrogate approximation introduces a 442-nat log-likelihood bias, exceeding the 68-nat gap you attribute to nonlinearity. How do you know the posterior shifts are not surrogate artifacts?"
- "The ZLB never binds at your training scale. You call this a method for occasionally binding constraints, but you have not demonstrated it on binding constraints in the medium-scale model."
- "You recover shocks under ROM1 and evaluate the likelihood under the nonlinear surrogate. That is an inconsistency you acknowledge but do not resolve."
- "The 3-parameter validation estimates only shock standard deviations in a tiny model. That does not validate the method for 18 parameters in Smets-Wouters."

**What would make identification convincing?**
1. Converged MCMC chains with R-hat < 1.05 and ESS > 400 for all 18 parameters.
2. Direct comparison against the full-order SEP posterior (at least on a parameter subset).
3. Synthetic data exercise at the 18-parameter scale where truth is known and the method recovers it.

### Part 3 --- Required and Suggested Analyses

**Required:**

1. **Converged 18-parameter MCMC chains.** No reviewer will accept posterior comparisons from chains with R-hat up to 2.67 and median ESS = 65. Hard blocker.

2. **Full-order SEP posterior benchmark.** The 442-nat bias bound exceeds the 68-nat Kalman-RS gap. Without comparison to the exact nonlinear posterior, the posterior-shift story is unverified.

3. **Synthetic validation at 18-parameter scale on the actual SW07 model.** Table 14 uses a "simplified NK model" that "differs from the SW07 HLT parameterization." The surrogate has not been validated for parameter recovery on the model it is applied to.

4. **Shock recovery consistency.** ROM1-recovered shocks vs. surrogate-recovered shocks -- magnitude of discrepancy and effect on posterior must be quantified.

**Suggested:**

5. Estimation with binding ZLB episodes (shock scale 0.4).
6. Formal approximation theory (theorem bounding posterior distortion from surrogate RMSE).
7. Comparison against second-order perturbation with particle filtering (the "Benchmark 3" proposed but never executed).
8. Extended sample with retrained surrogate (the $38\sigma$ COVID shocks come from Kalman only).
9. Uncertainty quantification for the surrogate (deep ensemble).

### Part 4 --- Literature Positioning

**Missing citations:**
- Guerrieri and Iacoviello (2015, "OccBin") -- piecewise-linear method for occasionally binding constraints, direct competitor.
- Boehl (2022, JEDC/REStud) -- nonlinear estimation with occasionally binding constraints via shooting.
- Fernandez-Villaverde and Guerron-Quintana (2021) -- SMC for nonlinear DSGE estimation.
- Winberry (2018, QE) -- deep learning for heterogeneous agent models.
- Bibliography note (lines 469-476) says "Additional references will be added as the paper progresses." Not acceptable in a submission.

**Framing suggestion:** The paper frames itself as an econometric methods paper, but the strongest content is the descriptive decomposition. Consider leading with the economic question ("Does the investment channel or the pricing channel dominate nonlinear dynamics?") and presenting the method as a tool for answering it.

### Part 5 --- Journal Fit

**Is this a strong fit for Econometrica?** No, not in current state. Econometrica requires either clean theoretical contribution (no formal theorems here) or polished empirical/computational results (headline estimation unconverged, surrogate bias exceeds treatment effect, ZLB never binds).

**Preliminary recommendation:** Desk reject at Econometrica.

**What would it take?**
1. Converged chains (R-hat < 1.05 for all parameters).
2. FOM-SEP posterior benchmark (even partial).
3. Formal theorem on posterior distortion.
4. Results with binding ZLB.
5. Clean 18-parameter synthetic validation on the actual SW07 model.
6. Substantially tighter presentation (82 pages is too long; the 69% finding is stated at least 8 times).

**Best realistic outlet:** Journal of Econometrics or Quantitative Economics. With converged chains and the FOM benchmark, this could be a solid paper for either. The decomposition finding alone, sharpened with formal analysis, could target the Review of Economic Studies. In its current state, the paper would benefit from another 6-12 months of computation and revision.

### Part 6 --- Questions to the Authors

1. Your log-likelihood bias bound is 442 nats, exceeding the 68-nat Kalman-RS gap. How do you rule out that posterior shifts are surrogate artifacts? What share of the $\sigma_b$ doubling survives if you halve the surrogate RMSE?

2. Have you computed the magnitude of the ROM1-vs-surrogate shock recovery inconsistency on synthetic data? What is the induced posterior bias?

3. The ZLB never binds in any 18-parameter result. Would you agree the actual contribution is about smooth nonlinearity, not occasionally binding constraints?

4. At what point would you consider the 18-parameter results stable enough to publish? Have you run the 500,000-draw extension?

5. Have you verified the 69% investment share is stable at shock scale 0.4 where the ZLB binds?

6. Why was the 18-parameter synthetic validation not conducted on the actual SW07 model?

7. Have you considered running a particle filter with second-order perturbation as an intermediate benchmark?

---

## Priority Action Items

### Tier 1: Hard Blockers (must resolve before any submission)

1. **Converge the 18-parameter MCMC chains.** R-hat < 1.05 and ESS > 400 for all parameters. No journal will accept the current results.

2. **Reconcile MH vs. HMC inconsistency.** The abstract/methodology/conclusion claim HMC; the estimation uses MH. Either implement HMC for 18 parameters or correct all text.

3. **Fix the 442-nat bias problem.** Either (a) produce an FOM-SEP posterior benchmark showing the surrogate does not distort the posterior, or (b) tighten the surrogate until the bias bound is below the treatment effect.

4. **Run 18-parameter synthetic validation on the actual SW07-HLT model.** The current validation uses a different model.

5. **Generate the 7 missing figure files** referenced by `\includegraphics`.

6. **Fix broken cross-reference** `\eqref{eq:taylor_zlb}` (renders as "??").

### Tier 2: Critical Corrections (required for credibility)

7. Fix all critical internal inconsistencies: RRMSE thresholds, "tenfold" arithmetic, six-vs-seven blocks, seed sensitivity parameter mismatch.

8. Add missing variable/notation definitions ($\bar{\beta}$, $p_t^k$, $\Sigma_{s_0}$, $\sigma_y$, $\bar{s}(\theta)$, $\varepsilon_w$, $\|H\|$).

9. Correct the steady-state labor exponent error ($1/\varphi$ not $1/(1+\varphi)$).

10. Resolve causal overclaiming: replace "confirming" with "consistent with," add caveats about compound identification problem.

11. Complete the bibliography (remove "additional references will be added" note, add Guerrieri-Iacoviello, Boehl, Fernandez-Villaverde-Guerron-Quintana, Winberry).

12. Quantify the inversion-filter shock-recovery inconsistency.

### Tier 3: Important Improvements (needed for top journal)

13. Produce ZLB-binding results (shock scale 0.4) to justify the paper's OBC framing.

14. Add ROM2+particle filter benchmark.

15. Standardize terminology (FOM/SEP/ROM1/ROM2, HLT/SW07).

16. Fix all table formatting issues (column counts, N reporting, note placement, decimal precision).

17. Define HLT acronym at first use in main text.

18. Add economic significance discussion for posterior shifts (impulse responses, variance decompositions).

19. Tighten presentation: reduce repetition (69% finding stated 8+ times), target 50-60 pages.

### Tier 4: Polish (before final submission)

20. Standardize first-person voice throughout.

21. Replace bare-text citations with natbib commands.

22. Fix all minor grammar/style issues (missing commas, passive voice, number formatting).

23. Add y-axis units to all IRF figure captions.

24. Cross-reference all defined tables.

25. Remove or archive unused figure files from `figures/`.
