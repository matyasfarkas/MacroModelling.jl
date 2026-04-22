# Pre-Submission Referee Report

**Paper**: The Invisible Nonlinearity: Investment Adjustment Costs and Structural Bias in Linear DSGE Estimation
**Authors**: Mátyás Farkas (International Monetary Fund)
**Date**: 2026-04-17
**Review Standard**: Leading Field Journal (Quantitative Economics / AEJ: Macro / JME)

---

## Overall Assessment

The paper identifies a novel and potentially important finding: that the investment adjustment cost---invisible to first-order perturbation because $S(1) = S'(1) = 0$---accounts for 69 percent of the nonlinear-linear gap in the Smets-Wouters model, while the price Phillips curve accounts for 0.2 percent. The analytical contribution is clean and the computational infrastructure is impressive. The principal weakness is the validation gap: the surrogate is validated on a 3-parameter toy model but applied at 18 parameters, with no Monte Carlo coverage study at the actual estimation scale. Until this gap is closed, referees cannot distinguish genuine nonlinear effects from surrogate artifacts.

**Preliminary Recommendation**: Revise before submitting (close the validation gap and address multimodality before sending to referees)

---

## 1. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. **Line ~524 vs. ~401: Contradictory framing of $\sigma_b$ direction** | Line 401 predicts "$\sigma_b$ should rise under nonlinear estimation" while line 524 says "Restoring the nonlinear channel lets $\sigma_b$ fall back toward its structural value." Both describe the same shift (0.12 → 0.28) from different perspectives (linear→nonlinear vs. nonlinear→linear). Pick one direction and use it consistently. The direction also **reverses** between the baseline sample ($\sigma_b$ rises from 0.12 to 0.28) and the extended sample ($\sigma_b$ falls from 0.179 to 0.091). This reversal must be acknowledged and explained.

2. **Line ~510/689: "within 5 percent" claim contradicted by Table 11** | The 3-parameter validation table shows $\sigma_\mu$ with a $-6.7\%$ relative error, exceeding the claimed 5% threshold. Either qualify the claim ("within 7 percent") or explain the discrepancy.

3. **Line ~1170: "12 configurations from... five architectures, two activations, and three learning rates"** | $5 \times 2 \times 3 = 30$, not 12. Rewrite to clarify that 12 survive initial screening from 30 candidates.

### Minor Issues

1. **Line 118**: "for the insightful discussions" → "for insightful discussions" (drop article).
2. **Line 160**: "2009--2015 in the United States, 2020--2022" — add country qualifier or conjunction.
3. **Line 162**: "which has received the most attention" — needs citation or soften to "substantial attention."
4. **Line 500**: "it *explodes*" — informal for journal register; consider "increases sharply."
5. **Line 520**: "The surrogate identifies 10 percent of periods" — the *gate* identifies periods, not the surrogate.
6. **Line 569**: "ridged geometry" — uncommon; verify intended meaning (ridgelike/multimodal).
7. **Line 599**: "the transition map does not depend on observed data" — imprecise; the *network weights* do not depend on observed data.
8. **Line 605**: "state space" → "state trajectory" (the state space itself is fixed).
9. **Line 665**: "half-life of 53 quarters" — of what exactly? Clarify.
10. **Line 693**: "toward dampening" → "aimed at dampening."
11. **Line 848**: "value" → "values" (plural: $\xi_w$ and $\varepsilon_w$).
12. **Line 1046**: "bug" → "implementation error" for journal register.
13. **Line 1502**: "Julia 1.12.4" — verify version (1.11.x was latest as of early 2026).
14. **Lines 611/701**: "The data speak through the investment channel..." repeated verbatim. Vary phrasing in one instance.

### Style Patterns to Fix Throughout

1. **Percent formatting**: Mixed usage ("69 percent" in prose vs. "5%" at line 1104). Choose one convention for prose (spell out "percent") and apply uniformly.
2. **Galí citation format**: Sometimes bare "Galí (2015)," sometimes `\citet{gali2015monetary}`. Use the citation command throughout.
3. **No issues found** with: filler adverbs, passive voice, "significant" misuse, hyphenation, em-dash usage. Prose is generally clean.

---

## 2. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. **Generated table files are orphaned stale artifacts** | The files in `generated/` (table_hlt_realdata_posterior.tex, etc.) show 3 degenerate parameters, NaN diagnostics, 244 periods — none matching the paper's narrative (18 parameters, T=184 or T=265). They are not `\input`'d in the paper, so no current harm, but should be deleted to avoid confusion. | Severity: CRITICAL (repo hygiene)

2. **$\sigma_b$ direction reverses between samples without acknowledgment** | Baseline: $\sigma_b$ rises 0.12→0.28 (linear→surrogate). Extended: $\sigma_b$ falls 0.179→0.091 (linear→surrogate). Introduction conflates the two. | Severity: CRITICAL

### Cross-Reference Errors

1. **Line 1216 references "columns Lower and Upper"** but Table `prior_specs` has a single "Bounds" column. Text/table mismatch.

### Terminology Drift

1. **FOM vs. SEP**: Used interchangeably; defined at line 360 but could be more prominent.
2. **ROM1 vs. linear vs. Kalman**: Standard in the literature but occasionally conflated.
3. **$\sigma_m$ vs. $\sigma_{ms}$**: Prior table uses $\sigma_m$; all other tables use $\sigma_{ms}$. Standardize.

### Minor Inconsistencies

1. **"2,450× speedup" (line 599)**: Not derivable from stated per-evaluation costs. Add calculation.
2. **Introduction claims shifts are "not pricing or monetary policy"** but $\rho_{ms}$ shifts by −0.096 in extended sample (4th-largest absolute shift). Slightly overstated for monetary policy.
3. **Gate table shows 244 periods** in the generated file, but baseline is T=184. (Moot since generated tables are not included.)

---

## 3. Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)

1. **Abstract/line 140**: "the linear model compensates for the missing investment channel by inflating exogenous shocks" — asserts mechanism as fact. Fix: "the linear model's parameter estimates are consistent with compensation for..."
2. **Line 524**: "Restoring the nonlinear channel lets $\sigma_b$ fall back toward its structural value" — no independent measurement of the "structural value." Fix: frame as hypothesis.
3. **Line 526**: "The linear model uses persistent wage markup shocks to substitute for the persistent nonlinear propagation channel" — mechanism asserted without isolation. Fix: note rho_w bimodality first, then frame as hypothesis.
4. **Line 528**: "The linear model prefers low $\varepsilon_p$ to compensate" — same pattern. Fix: "directionally consistent with the theory but occurs in a weakly identified region."
5. **Line 609**: "The nonlinear model absorbs COVID-era volatility through endogenous amplification rather than large exogenous shocks" — strongest causal claim, but identification is limited. Fix: "achieves better fit with smaller estimated shock volatilities, a pattern consistent with endogenous amplification."
6. **Line 657**: "confirms this qualitatively" → "is consistent with this prediction."
7. **Line 168**: "Each shift follows the logic of the invisibility theory" → "Each shift is directionally consistent with..."
8. **Line 407**: "because the quadratic adjustment cost compounds with..." → "consistent with the quadratic adjustment cost compounding with..."
9. **Line 611**: "The data speak through the investment channel" → "Within this model, the data strongly identify parameters entering the investment channel..."

### Generalization Issues

1. **Line 172**: "the economy is driven more by endogenous propagation" — generalizes from one model on one country's data. Fix: scope to "Within the SW07-HLT framework..."
2. **Line 172**: "prioritizing the investment block over the ELB and Kimball pricing" — strong recommendation from single model. Fix: add "at least for models sharing the SW07 quadratic adjustment cost specification."
3. **Line 693**: "If linear models systematically overestimate..." — keep the "if" framing throughout.

### Missing Caveats

1. **Surrogate trained at shock scale 0.1; estimation recovers shocks >>0.1σ** | Should note that obs-only mode and OOD fallback limit extrapolation burden.
2. **S'' = 6.01 calibrated, not estimated** | The entire paper is about the nonlinearity this parameter governs. Estimating S'' jointly is a natural extension that should be flagged.
3. **MA terms dropped from markup shocks** | Appendix I shows 34-nat cost and changes in $\rho_w$. This caveat belongs in the main text where the $\rho_w$ shift is presented.
4. **No exact nonlinear likelihood benchmark** | The 18-nat and 30-nat gaps are surrogate vs. Kalman, not surrogate vs. true nonlinear LL. Should be acknowledged.
5. **Single posterior mode throughout** | Policy implications section should remind the reader.
6. **No out-of-sample validation** | COVID fit is in-sample. Conclusion appropriately lists this as future work, but main text reads as if fit = forecasting.

### Unsupported Robustness Claims

1. **Line 667**: "Cross-seed variation is below 1.5 percent" — this is on the 3-parameter Galí model, not the 18-parameter SW07-HLT. Clarify scoping.
2. **Line 678**: Investment shares of "89–91%" come from Mahalanobis-stratified posterior draws, not the Sobol training set (which gives 68.9%). Clarify which decomposition produces which figure.

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors

1. **Eq. (eq:ll_bias_exact), line ~1700**: The stated bound with $\|H\|_{op}^2 / 2$ and the $(2\|\cdot\| + \|H\|\|\cdot\|)$ structure does not follow from the proof, which derives the simpler $|r' \Sigma_y^{-1} e| \leq \|r\| \cdot \|e\| / \sigma_{\min}^2$. Mismatch between proposition and proof.
2. **Eq. (eq:ll_bias_total), line ~1709**: The aggregate bound introduces $n_y$ without justification. The per-period bound is scalar; summing over $T$ should give $T \cdot C \cdot \delta^2$, not $T \cdot n_y \cdot C \cdot \delta^2$.
3. **Corollary 7.8 (eq:coverage_bound)**: The bound includes $\lambda_{\max}(I(\theta_0))$ but the proof involves $\lambda_{\min}$ in the denominator. Important spectral factors are suppressed.
4. **Eq. (eq:per_period_ll), line ~1304**: Normalizing constant uses $n_y \log(2\pi)$ but should be $(n_y + |S|) \log(2\pi)$ since both observation and shock dimensions contribute.

### Notation Inconsistencies

1. **$\phi$ vs. $S''$** for investment adjustment cost curvature: `phi` in introduction (line 164), `S''` in appendix (line 832). Unify or define explicitly.
2. **$\sigma_m$ vs. $\sigma_{ms}$**: Prior table uses $\sigma_m$; all other tables use $\sigma_{ms}$.
3. **$z_t$ used for both capital utilization (line 824) and binary gate indicator (line 332)**. Use $\zeta_t$ or $g_t$ for the gate.
4. **$\beta$ used for three things**: discount factor, soft-gating sensitivity coefficients ($\beta_\varepsilon$, $\beta_y$), and FiLM bias ($\beta(\theta)$). The last overlap is most confusing.
5. **$\rho$ as backtracking factor** (Algorithm 3) vs. shock persistence $\rho_j$ throughout.
6. **$\sigma(\cdot)$ for logistic function** (line 1354) vs. $\sigma$ for standard deviations throughout. Define logistic function explicitly at first use.

### Undefined Notation

1. **$D_p$ (price dispersion)**: Mentioned in Table 1 and line 846 but never given an equation.
2. **$s_t^\pi$** in Rotemberg pricing equation (line 676): not defined in that section.
3. **$\lambda_p$** in line 676: used in $\varepsilon = \lambda_p / (\lambda_p - 1)$ without definition.
4. **`curvp`** at line 2202: code variable name leaking into prose. Replace with mathematical expression.
5. **$\bar{\gamma}$**: Used but numerical value never stated.
6. **$\sigma_y$ vs. $\Sigma_y$**: Relationship between scalar observation SDs and covariance matrix never made explicit.

### LaTeX Math Formatting

1. **Line 676**: Very long inline Rotemberg equation should be displayed.
2. **Line 1354**: $\sigma(\cdot)$ for logistic function needs explicit definition.
3. **Lines 1129–1146**: FiLM equations are numbered but never cross-referenced. Consider removing numbers.

---

## 5. Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

1. **Table `equation_block`**: Missing model name (SW07-HLT), parameter draw method, shock scale, and units for $\|\Delta\|_2$.
2. **Tables `18param_switching_posterior` and `extended_sample_posterior`**: Missing prior cross-reference, $\Delta$ sign convention not specified, no Kalman 90% CI reported.
3. **Table `benchmarks`**: "Status" column values undefined; missing sample definition.
4. **Table `sep_solver_params`**: No notes at all.
5. **Table `prior_specs`**: Text references "columns Lower and Upper" but table has single "Bounds" column; $\sigma_m \neq \sigma_{ms}$.
6. **Table `crisis_vs_normal`**: Variance share reported for only 3 of 7 observables; missing units.
7. **Table `18param_full`**: Caption says "Complete 18-Parameter Validation" but uses a simplified NK model, not SW07-HLT. Highly misleading.
8. **Tables `prior_sensitivity` and `seed_sensitivity`**: Use different parameters ($\xi_p, \phi_\pi, \phi_y$) than main 3-parameter validation ($\sigma_A, \sigma_\mu, \sigma_R$). Confusing without explicit context.

### Unreferenced Tables

1. **`tab:regime_performance`** (line 1849): Defined but never cited.
2. **`tab:computational_cost`** (line 1971): Defined but never cited.
3. **Generated tables** in `generated/`: Orphaned stale artifacts (not included in paper).

### Figures with Missing or Incomplete Notes

1. **Most figures missing**: model name, parameter configuration, and/or y-axis units.
2. **Figure `rom1_forecast_errors`**: Missing parameter values and y-axis units.
3. **Figure `gali_obc_vs_no_obc`**: Missing y-axis units and calibration specification.
4. **Figure `zlb_correction_4panel`**: Does not list which four observables are shown.
5. **Figures `zlb_correction_robs`**: Missing y-axis units (annualized %, gross rate?).

### Formatting Inconsistencies

1. Inconsistent CI reporting across posterior comparison tables (some have Kalman CI, some don't).
2. Table note spacing: some use `\vspace{0.5em}`, others `\smallskip`.
3. Units missing from column headers (e.g., "Train Time" without "(min)").
4. Crisis definition in `tab:crisis_vs_normal` uses investment-specific criterion ($|\text{innovation}_{\Delta i}| > 2\sigma$), which biases toward investment-dominated episodes.

---

## 6. Contribution & Referee Assessment

### Part 1 — Central Contribution

The paper finds that the investment adjustment cost accounts for 69% of the nonlinear-linear gap in SW07, formalizes this as "perturbation invisibility," and develops a neural network surrogate to estimate the nonlinear model via Bayesian HMC.

**Rating: Significant.** The substantive finding is genuinely surprising and, if robust, would redirect how the profession prioritizes nonlinearities. The analytical observation about perturbation invisibility is clean and general. However, the result is demonstrated for one model with one solver, and the estimation validation gap (3-parameter validation vs. 18-parameter application) is the main vulnerability. "Transformative" would require generalization to multiple models or a definitive Monte Carlo study.

### Part 2 — Identification and Credibility

The decomposition is deterministic and clean. The main threat is that the SEP solver (3-node GH quadrature, 2-period branching) is treated as "truth" without independent verification. A skeptical seminar audience would ask: (1) Why no Monte Carlo coverage at 18 parameters? (2) With multimodality this severe, how can any posterior comparison be meaningful? (3) ESS of 43 is insufficient for 18-dimensional inference. (4) The inversion filter recovers shocks under ROM1 but evaluates likelihood under the surrogate — is this inconsistency bounded? (5) How does `accept_tol = 0.35` propagate into posterior accuracy?

### Part 3 — Analyses: Required and Suggested

**Required:**
1. Monte Carlo coverage study at 18-parameter scale (or intermediate scale with investment channel present).
2. SEP solver accuracy validation against an independent nonlinear solution method.
3. Multi-chain convergence diagnostics with ≥4 chains from dispersed initializations.
4. Sensitivity to SEP solver tolerances (accept_tol: 0.01 vs. 0.35) and quadrature nodes (K=3 vs. K=5).

**Suggested:**
1. Generalization to at least one other model with investment adjustment costs (CEE 2005 or JPT 2010).
2. Out-of-sample forecasting comparison (estimate on 1959–2019, forecast 2020–2025).
3. Expand direct FOM comparison from 20 to 100+ posterior draws.
4. Robustness to gate threshold (5%, 20%, 50%, 100% nonlinear periods).
5. Second-order perturbation (ROM2) benchmark decomposition.

### Part 4 — Literature Positioning

Adequate core citations. Missing: Fernandez-Villaverde, Rubio-Ramirez, Santos (JET 2006); Richter and Throckmorton (ZLB estimation papers); Basu and Bundick (2017 AER). AlphaGo/normalizing-flows citations (Silver et al., Rezende and Mohamed) seem out of place for a macro paper.

The "ELB is the wrong place to look" framing is bold and attention-grabbing but risks alienating the ELB literature. More accurate: "In medium-scale models with investment, the investment channel dominates the ELB."

### Part 5 — Journal Fit and Recommendation

- **Best targets**: Quantitative Economics (natural home), JME, AEJ: Macro
- **Econometrica**: Stretch; would need Monte Carlo coverage, multiple models, tighter theory
- **Alternative outlets**: RED, JEDC, Computational Economics
- **Recommendation**: Revise before sending to referees

### Part 6 — Questions to the Authors

1. Your decomposition treats SEP as truth. How sensitive is the 69% investment share to quadrature order and branching horizon? Have you verified SEP accuracy against any independent global solution method?

2. You validate on 3 parameters and apply at 18. The "illustrative 18-parameter validation" uses a simplified NK model, not SW07-HLT. Why not run coverage on the actual model? If resources are constrained, how should the reader assess Tables 3–4?

3. You report a 34-nat gap between linear chains and 12,700 nats between switching chains. Do the posterior shifts hold at the alternative linear mode ($\rho_w = 0.75$)?

4. Shocks are recovered under ROM1 but likelihood is evaluated under the surrogate. Have you verified that re-recovering shocks under the nonlinear model produces similar results?

5. Is the 30-nat LL gap robust to varying the gate threshold (5%, 20%, 100% nonlinear periods)? If it increases monotonically with the nonlinear share, it may reflect flexibility rather than genuine effects.

6. The $56\sigma$ COVID shocks are computed by passing data through the Kalman smoother at the linear posterior mean. Have you conducted any out-of-sample predictive exercise?

7. Could richer linear models (time-varying parameters, stochastic volatility, financial frictions) absorb the same variation currently attributed to the nonlinear investment channel?

---

## Priority Action Items

**CRITICAL** (must fix — could cause desk rejection or major referee objections):
1. **Close the validation gap**: Monte Carlo coverage study at ≥8-parameter scale with the investment channel present (Agent 6, Required #1)
2. **Acknowledge and explain the $\sigma_b$ direction reversal** between baseline (rises) and extended sample (falls) — currently the introduction/abstract conflate the two (Agents 1, 2, 4)
3. **Weaken causal language throughout**: 9 instances of mechanism claims asserted as fact; frame as "consistent with" (Agent 3)
4. **Fix formal theory**: Proposition/proof mismatch on the LL bias bound (extra $\|H\|_{op}^2$ and $n_y$ factor); coverage bound suppresses spectral factors (Agent 4)

**MAJOR** (should fix — will likely be raised by referees):
5. **Multi-chain convergence**: Run ≥4 chains from dispersed initializations; report whether shifts hold at alternative mode (Agent 6, Required #3)
6. **SEP solver validation**: Verify accuracy against independent method; sensitivity to accept_tol and quadrature nodes (Agent 6, Required #2, #4)
7. **Missing caveats in main text**: S'' calibrated not estimated; MA terms dropped from markups affect $\rho_w$; no exact nonlinear LL benchmark (Agent 3)
8. **Notation cleanup**: $z_t$ double use; $\sigma_m$ vs. $\sigma_{ms}$; $\phi$ vs. $S''$; undefined symbols ($D_p$, $\lambda_p$, `curvp`) (Agent 4)
9. **Table documentation gaps**: Missing model names, units, prior references, and $\Delta$ sign conventions across multiple tables (Agent 5)

**MINOR** (polish — improves paper quality):
10. **Fix "within 5 percent" claim** (actual max error: 6.7%) (Agent 1)
11. **Delete orphaned generated tables** in `generated/` directory (Agent 2)
12. **Remove AlphaGo/normalizing-flows citations**; add missing references (Richter-Throckmorton, FV-RR-Santos) (Agent 6)
13. **Standardize percent formatting** and Galí citation format throughout (Agent 1)
14. **Add references to 2 unreferenced tables** (`tab:regime_performance`, `tab:computational_cost`) or remove them (Agent 5)
15. **Rename misleading Table 19 caption** ("Complete 18-Parameter Validation" uses simplified NK model, not SW07-HLT) (Agent 5)
16. **Scope generalization claims** to SW07-HLT framework rather than "the economy" (Agent 3)
