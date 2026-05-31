# Pre-Submission Referee Report

**Paper**: Investment Adjustment Costs and the Nonlinear Posterior of Smets--Wouters  
**Authors**: Mátyás Farkas  
**Date**: 2026-05-31  
**Review Standard**: Leading Field Journal  

---

## Overall Assessment

The paper has a strong and useful core: direct nonlinear solution diagnostics show that, in the HLT Smets--Wouters environment studied here, the investment and capital block accounts for most of the measured first-order approximation gap. The current draft is also much cleaner than the earlier version, with a shorter main paper and a separate technical appendix. The remaining submission risk is not prose polish; it is credibility at the empirical-posterior layer. The paper still needs to reconcile active-paper artifacts, clarify that several empirical objects are profiled criteria rather than exact Bayesian posteriors, and strengthen the dynamic validation bridge between direct SEP and the surrogate pipeline.

**Preliminary Recommendation**: Substantial revision required.

---

## 1. Spelling, Grammar & Style

## Agent 1: Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. `docs/SurrogateNN_paper/SurrogateNN_technical_appendix.tex:63` | "Matyas Farkas" -> "Mátyás Farkas" | Author name is inconsistent with the main paper.
2. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:158` | "what first-order Smets--Wouters misses" -> "what the first-order approximation to Smets--Wouters misses" | Current wording treats "first-order" as the model rather than the approximation.
3. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:170` | "four-parameter HLT bridge" -> "four-parameter Holden--Linde--Trabandt (HLT) bridge" | HLT is not defined before use in the active main paper.
4. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:227` | `\(g_\theta^{\mathrm{ROM1}}\)` -> add before use: "I denote the first-order perturbation solution by ROM1." | ROM1 appears before it is defined.
5. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:305` | "FOM--ROM1 gap" -> "SEP--ROM1 gap" or define "full-order model (FOM)" before this sentence | FOM is used before definition; the paper otherwise says SEP.
6. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:369-371` | Displayed equation ends with "," before a new sentence -> end the displayed equation with "." | The punctuation incorrectly links the equation to the next sentence.
7. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:696` | "The OOD detection mechanism" -> "The out-of-distribution detection mechanism" or "The out-of-distribution (OOD) detection mechanism" | Undefined acronym in main text.
8. `docs/SurrogateNN_paper/SurrogateNN_technical_appendix.tex:854-864`; `docs/SurrogateNN_paper/generated/sep_sensitivity_table.tex:12-21` | "five posterior draws" / "265-period" / `\(1.08\times 10^{-2}\)` vs. "three posterior draws" / "40 periods" / `6.16e-03` -> make the appendix and generated table report the same draw count, period count, and RMSE | This is a visible numerical consistency problem.

### Minor Issues

1. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:163` | "has zero level" -> "has zero value" | More idiomatic mathematical prose.
2. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:170` | "all 10,000 supported direct-SEP cells solve" -> "all 10,000 supported direct-SEP cells converge" | "Cells solve" is awkward.
3. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:291` | "At the \citet{smets2007shocks} calibration" -> "Under the calibration of \citet{smets2007shocks}" | More standard academic phrasing.
4. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:413` | "Since the inversion step profiles..." -> "Because the inversion step profiles..." | Avoids temporal ambiguity.
5. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:505` | "where the Kimball aggregator enters and which receives..." -> "the block through which the Kimball aggregator enters and the focus of much of..." | Reduces clunky relative-clause stacking.
6. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:583` | "US quarterly data" -> "U.S. quarterly data" | Inconsistent with earlier "U.S. data."
7. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:694` | "ridged geometry" -> "ridge-like posterior geometry" | More standard econometric phrasing.
8. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:719` | "$^*$Per-evaluation" -> "$^*$ Per-evaluation" | Missing space after footnote marker.
9. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:738` | "first-order caveats" -> "primary caveats" | Avoids confusion with first-order perturbation.
10. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:803` | "loses ground" -> "lowers the criterion" | Colloquial phrasing.
11. `docs/SurrogateNN_paper/SurrogateNN_paper.tex:835` | "Initial-state treatment is not used as a headline validation result here." -> "I do not use the initial-state treatment as a headline validation result." | More direct and less awkward.
12. `docs/SurrogateNN_paper/SurrogateNN_technical_appendix.tex:164` | "2-by-2 objective decomposition" -> `2$\times$2 objective decomposition` | Match main-paper notation.
13. `docs/SurrogateNN_paper/SurrogateNN_technical_appendix.tex:595` | "current-repo Gal'i OBC model" -> "current repository's Gal'i OBC model" | Avoid internal shorthand.
14. `docs/SurrogateNN_paper/SurrogateNN_technical_appendix.tex:642` | "Pass as smoke" -> "Pass (smoke only)" | Clearer table status.
15. `docs/SurrogateNN_paper/csadjcost_sensitivity_table.tex:19` | "FOM-ROM1" -> "FOM--ROM1" | Use consistent dash styling for model comparisons.
16. `docs/SurrogateNN_paper/generated/sep_sensitivity_table.tex:10-17` | "tol", "conv.", "err.", `7.97e-08`, `40.0` -> "Tolerance", "Convergence", "error", `\(7.97\times10^{-8}\)`, `40` | Table formatting is too raw for submission.
17. `docs/SurrogateNN_paper/generated/table_mode_sensitivity.tex:19` | "Rhat diagnostics" -> `\(\widehat R\) diagnostics` | Use standard notation.

### Style Patterns to Fix Throughout

- Define acronyms before first use: HLT, ROM1, FOM, OOD.
- Replace standalone "first order" with "the first-order approximation" or "the first-order solution."
- Choose one lower-bound term unless a distinction is intended: ELB vs. ZLB.
- Harmonize appendix references: use either "online appendix" or "online technical appendix" consistently.
- Standardize percentages: use "percent" in prose and `\%` in tables/math.
- Remove conversational/internal phrasing in submission prose.
- Clean generated tables before submission: title-case captions, no raw scientific `e` notation, no lowercase abbreviations in headers.

---

## 2. Internal Consistency & Cross-Reference Verification

## Agent 2: Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. `SurrogateNN_paper.tex:554,559,583` <-> `figures/fig_rom1_forecast_errors.pdf` | Text/table define the ROM1 forecast-error diagnostic as 1959Q1--2004Q4, 184 quarters, 13 crisis quarters; the active figure visibly extends through COVID/post-2020. This is a sample-period mismatch in an empirical diagnostic. | Severity: CRITICAL
2. `SurrogateNN_paper.tex:219-222,607` <-> `SurrogateNN_paper.tex:365-370`; `SurrogateNN_technical_appendix.tex:593-599` | The Gal'i negative control is said to share "ELB and Kimball" nonlinearities with HLT, but the model description/validation package only document a Calvo/Taylor-rule/lower-bound Gal'i model. If Kimball is absent, the negative control only isolates investment versus ELB/simple pricing, not investment versus Kimball. | Severity: CRITICAL
3. `SurrogateNN_technical_appendix.tex:662-666` <-> `SurrogateNN_technical_appendix.tex:710-724` | The reduced HLT bridge is described as using five observables, but the RMSE table and aggregate bridge RMSE use seven observables, adding `dc` and `dwobs`. The validation object is therefore inconsistently specified. | Severity: CRITICAL
4. `SurrogateNN_technical_appendix.tex:175-178,854-867` <-> `generated/sep_sensitivity_table.tex:12-21` | Appendix claims a full 265-period SEP sensitivity rerun over five posterior draws and reports K=3 vs. K=5 RMSE of `1.08e-02`; the table file reports three posterior draws, 40 periods, and K=3 RMSE `6.16e-03`. | Severity: CRITICAL
5. `SurrogateNN_paper.tex:592,597` <-> `figures/decomposition_vs_shock_scale.pdf` | Text/caption claim shock scales 0.1--1.5, but the active figure only plots 0.8, 1.0, 1.2, and 1.5. This overstates the plotted support for the ELB-alternative claim. | Severity: CRITICAL

### Cross-Reference Errors

1. `SurrogateNN_paper.tex:732`, Figure `fig:covid_shock_comparison` | `figures/covid_shock_comparison.pdf` | Text cites the figure after claiming risk-premium, price-markup, and monetary shocks of 56 sigma / 32 sigma / 23 sigma; the target figure shows TFP, risk premium, investment, and wage markup only.
2. `SurrogateNN_paper.tex:533`, `Section~\ref{par:results_18param_switching}` | Paragraph label at `SurrogateNN_paper.tex:641-642` | The reference says "Section" but targets an unnumbered paragraph label; LaTeX will likely resolve to the surrounding subsection, not the named paragraph.
3. `SurrogateNN_paper.tex:722`, "online appendix reports the full particle-filter table" | Active technical appendix | No particle-filter table or particle-filter section appears in `SurrogateNN_technical_appendix.tex`; the detailed table appears only in the inactive legacy appendix.
4. `SurrogateNN_paper.tex:503,531,539,541,601,607,639` | Active technical appendix | Several "online appendix" references point to state-distance quintile tables, parameter correlates, IRF shock comparisons, ZLB binding shares, Gal'i shock-scale gaps, and a full baseline posterior table that are not present in the active technical appendix.
5. `generated/table_mode_sensitivity.tex` | Active main/technical files | This table is only input behind the inactive `\ifpaperappendix`; it is not included in the short paper or active technical appendix despite being listed as an active table file.

### Terminology Drift

1. ELB / ZLB / lower bound / OBC / hard-ELB / actual-floor | Main text uses ELB, figure titles/captions use ZLB, appendix uses OBC/hard-ELB/actual-floor. | Recommended standardization: use "ELB" for the economic constraint, "ZLB/gross-one floor" only for implementation details, and "OBC" only for solver architecture.
2. FOM / SEP / direct nonlinear / full-order / nonlinear-minus-linear | The same benchmark is named several ways. | Recommended standardization: define once as "SEP direct nonlinear solution (FOM)" and then use "SEP--ROM1 gap."
3. Regime-switching surrogate / RS Surrogate / switching surrogate / ROM1+NN / gate+NN | The estimator has multiple names across tables, captions, and text. | Recommended standardization: use "regime-switching surrogate" in prose and "RS surrogate" in tables.
4. Shock naming: investment-specific / `qs` / `q^s` / `\varepsilon^{qs}` / `z_eqs` | Economic, code, and table names are mixed without a stable mapping. | Recommended standardization: use economic names in main text, with code aliases only in appendix tables.

### Minor Inconsistencies

1. `SurrogateNN_paper.tex:546` <-> `figures/fig_sep_all_vars_eqs_4sigma.pdf` | Caption says y-axis is "percentage deviation," but the plotted axes show decimal values without percent labels, creating a possible 100x magnitude ambiguity. | Severity: MINOR
2. `SurrogateNN_paper.tex:450-451` <-> `SurrogateNN_paper.tex:656` | Surrogate accuracy is stated as non-binding RRMSE 0.0012, but later as "training RMSE of 0.47" without defining the metric/units. These may be different objects, but the distinction is not visible. | Severity: MINOR
3. `SurrogateNN_paper.tex:299,293` <-> `SurrogateNN_paper.tex:624` | The text says the SW07-HLT model contains seven nonlinearity sources, but capital-utilization curvature is discussed as a nonlinearity and later quantified separately. The count/classification is unclear. | Severity: MINOR
4. `SurrogateNN_paper.tex:291,855` <-> `csadjcost_sensitivity_table.tex:13,19` | Baseline investment-adjustment curvature appears as 6.06, 6.0, and bold 6.0. This is probably rounding, but the paper should state the rounding convention once. | Severity: MINOR

---

## 3. Unsupported Claims & Identification Integrity

## Agent 3: Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)

1. Gal'i Negative Control | "The Gal'i (2015) model shares the ELB and Kimball nonlinearities with the HLT model but lacks investment---a difference-in-differences design." | This is not a true difference-in-differences design; Gal'i and HLT differ along many structural dimensions besides investment. | Fix: call it "a negative-control model comparison" unless a controlled ablation within the same HLT environment is added.
2. Quantitative Decomposition | "State distance alone explains 96 percent of the nonlinear correction... the gap is a function of where the economy sits, not which parameters generated the displacement." | Separate-feature `R^2` is not causal attribution and depends on the simulated support. | Fix: "On this training support, state distance has the highest predictive `R^2`; parameter variation plays a smaller role in this diagnostic."
3. Quantitative Decomposition | "The investment channel generates the nonlinear impulse; the discount factor transmits it economy-wide." | This states a structural mechanism more strongly than the block decomposition identifies. | Fix: "In this decomposition, the largest residual loads on investment/capital variables, with discount-factor terms carrying part of the associated propagation."
4. Quantitative Decomposition | "These patterns predict testable posterior shifts. Under nonlinear estimation, `\sigma_b` should rise..." | Posterior shifts under a different objective, gate, surrogate, and mode are not a clean test of the mechanism; the extended sample later reverses the `\sigma_b` sign. | Fix: "These patterns suggest possible posterior shifts, but the sign can depend on sample period and posterior basin."
5. Consequences for Inference / Decomposing the Likelihood Gap | "This fixed-parameter ablation... rules out the simplest concern that the 35.92-nat result is mechanically generated by the gate or inversion filter." | A two-point fixed-parameter ablation cannot rule out posterior-level gate/inversion effects; the text itself says a no-NN HMC run is still needed. | Fix: "provides evidence against the simplest fixed-parameter version of this concern."
6. Conclusion | "avoids interpreting COVID as a sequence of extreme linear shocks." | The figure notes that applying a linear smoother at the surrogate posterior can produce even larger normalized shocks; the claim needs to be tied to the nonlinear profiled objective. | Fix: "within the profiled nonlinear objective, fits the COVID period with lower estimated shock volatilities than the linear posterior mean."

### Generalization Issues

1. Abstract / Introduction | "This paper asks what first-order Smets--Wouters misses. The answer... is the investment block." | Reads as a Smets--Wouters-wide claim, but evidence is for the HLT specification, selected supports, and reported decompositions. | Fix: "In the HLT specification and audited support studied here, the largest measured gap is in the investment block."
2. Introduction | "For this class of models, the first nonlinear object to audit is the investment block, not the ELB or Kimball pricing." | "This class" is too broad without cross-model evidence. | Fix: "For SW-style models with level investment adjustment costs, this evidence makes the investment block a priority diagnostic."
3. Extended-Sample Estimation | "it generalizes to any sample period as long as the state space remains within the training support." | Validity also depends on shock recovery, gate calibration, distributional shift, and out-of-distribution behavior, not just state support. | Fix: "it can be evaluated on other samples; accuracy remains conditional on audited state-shock-parameter support and gate behavior."
4. Runtime | "The same residual surrogate serves multiple estimation runs without retraining..." | True computationally, but inferentially too broad unless each run's support is checked. | Fix: add "provided the evaluated paths remain inside the documented training/validation envelope."
5. Robustness to Pricing Specification | "a generic property of pricing nonlinearity" | Three pricing variants do not establish a generic property. | Fix: "whether it persists across several pricing specifications."
6. Technical Appendix / Claim Map | "Full 265-period SEP sensitivity run over five posterior draws..." | The listed generated table reports three posterior draws and 40 periods, not five draws and 265 periods. | Fix: reconcile the table and appendix; until then, state only the narrower diagnostic.
7. ELB as Alternative Hypothesis | "At empirically relevant scales... the ZLB contributes negligibly to the FOM--ROM1 gap." | Based on the Gal'i calibration/design; not an ELB-wide claim. | Fix: "in this Gal'i negative-control calibration and shock-scale design."

### Missing Caveats

1. Decomposition scaling | Quantitative decomposition tables and captions | Suggested text: "Block shares depend on the variable scaling, block partition, norm, and simulated support; they are descriptive decompositions of the measured FOM--ROM1 gap, not invariant causal shares."
2. SEP as benchmark | Nonlinear solution / SEP algorithm and decomposition notes | Suggested text: "SEP is the maintained nonlinear benchmark under the stated finite horizon, quadrature, softplus lower-bound approximation, and solver tolerances; it is not an exact analytical solution."
3. Mixed objective units | `generated/table_mode_sensitivity.tex` and extended-sample table notes | Suggested text: "Kalman entries are exact log likelihoods; surrogate entries are profiled inversion objectives omitting the shock-inversion determinant, so levels are criterion values rather than marginal likelihoods."
4. Forecasting terminology | Abstract/Introduction references to calm-period forecasting | Suggested text: "The OOS exercises are conditional filtering diagnostics with recovered holdout shocks, not real-time unconditional forecasts."
5. Statistical versus economic significance | Posterior-shift discussion | Suggested text: "Posterior mean shifts should be interpreted alongside mode uncertainty, ESS, and the absence of matched linear credible intervals; economic magnitude is not by itself evidence of identification."
6. Policy counterfactuals | Policy implications | Suggested text: "The evidence does not validate welfare or policy-counterfactual rankings; it motivates checking investment-channel nonlinearities before using linear policy experiments in this specification."

### Minor Language Issues

1. Abstract | "The reason is simple" | Oversimplifies a multi-equation mechanism. | Fix: "The core mechanism is..."
2. Introduction | "the cleanest negative-control result" | Too strong given cross-model differences. | Fix: "the most direct negative-control comparison reported here."
3. Quantitative Decomposition | "where the Kimball aggregator enters and which receives the most attention in the nonlinear DSGE literature" | Literature overclaim unless substantiated. | Fix: cite a review or weaken to "a common focus in nonlinear DSGE applications."
4. Robustness to Pricing Specification | "The Rotemberg variant is the informative test." | Overstates one robustness check. | Fix: "The Rotemberg variant is the most informative of the three reported tests."
5. Technical Appendix / Claim Map | "The hard-ELB pipeline works in a small model" | "Works" is too binary for validation evidence. | Fix: "The hard-ELB pipeline passes the reported small-model diagnostics."
6. Technical Appendix / How to Read the HLT Bridge | "the hardest medium-scale numerical object that is currently affordable" | Unsupported and rhetorically strong. | Fix: "a currently affordable medium-scale numerical object directly tied to the investment channel."

---

## 4. Mathematics, Equations & Notation

## Agent 4: Mathematics, Equations & Notation

### Mathematical Errors

1. Main `SurrogateNN_paper.tex:356`; Appendix `SurrogateNN_technical_appendix.tex:421-463` | The stated joint likelihood includes shock densities, but the profiled surrogate period objective includes only the Gaussian observation density. This is not the profile of the displayed likelihood. | Either add the evaluated shock-prior term at `\widehat\eps_t(\theta)` to the profiled criterion, or explicitly rename it an observation-fit criterion rather than a profiled likelihood/objective.
2. Appendix `SurrogateNN_technical_appendix.tex:465-467` | "Not the exact marginal likelihood unless determinant terms are added" is incomplete: determinant terms alone do not recover a marginal likelihood when the shock prior and profiling/Laplace approximation are omitted. | State that a Laplace approximation would require the shock prior at the optimizer plus the Hessian determinant; an exact change-of-variables likelihood requires a separately derived Jacobian.
3. Appendix `SurrogateNN_technical_appendix.tex:477-486`; Main `SurrogateNN_paper.tex:795-800` | The objective mixes Kalman predictive likelihood contributions with profiled inversion observation contributions. This hybrid is not a single likelihood unless the full joint density and state-update recursion are defined consistently. | Call it a hybrid criterion, or define a unified likelihood/quasi-likelihood for all periods under one architecture.
4. Main `SurrogateNN_paper.tex:783-803` | "Likelihood gap" and the 80.9/19.1 decomposition subtract exact Kalman likelihood values from surrogate profiled-objective values with different omitted terms/constants. | Rename to "criterion gap" and state that the decomposition is accounting-only; do not interpret percentages as likelihood decomposition unless normalizations are made common.
5. Main `SurrogateNN_paper.tex:283-289` | The perturbation invisibility proposition is too broad: it needs smoothness and entry assumptions. As written, it claims zero contribution to the solution, not merely to the linearized residual component. | Add assumptions that the equilibrium residual is smooth in `f(x_t)` with finite derivatives, and conclude "this component contributes zero to the first-order linearized residual."
6. Main `SurrogateNN_paper.tex:868-870` | The active technical appendix does not contain the referenced formal coverage-distortion diagnostic/theory, while the main text uses its formula and conclusion. | Move the theorem, assumptions, and diagnostic into the active technical appendix or remove the claim from the active main paper.

### Notation Inconsistencies

1. `x_t`, `x_j` | Used for the investment ratio `I_t/I_{t-1}` in `SurrogateNN_paper.tex:216`, for retained surrogate outputs in `SurrogateNN_technical_appendix.tex:199-203`, and as a transition input in `SurrogateNN_paper.tex:448` | Use distinct notation: e.g. `u_t=I_t/I_{t-1}`, `X_j=[s_{j,-};\eps_j;\theta_j]`, and `o_t` for retained outputs.
2. `\eps_t` | Called standardized shocks in `SurrogateNN_paper.tex:323`, but penalized as raw shocks with `\eps/\sigma_\eps(\theta)` in `SurrogateNN_technical_appendix.tex:421-431` | Choose raw shocks with covariance `\Sigma_\eps(\theta)` or standardized shocks with transition input `\sigma_\eps(\theta)e_t`.
3. `g_t` | Transition rules are denoted `g_\theta` in `SurrogateNN_paper.tex:326-330`; the gate is also `g_t` in `SurrogateNN_paper.tex:461-462` | Use `m_t` or `d_t` for the gate, matching the appendix.
4. `\Delta` | Used for vector residual `\Delta g_\theta`, observation residual `\Delta y_j`, scalar norm `\Delta_j`, and NN output `\widehat\Delta_w` | Define vector residuals, block residuals, and scalar norms separately.
5. `H` | Used as a possibly nonlinear map `H(s_t)` in the main paper, but as a matrix selecting from `x_t`/`g_\theta` in the appendix | Use `h(s_t)` for nonlinear measurement maps or define `H` as a fixed matrix and write `Hs_t`.
6. `FOM` vs. `SEP` | Main notation uses `g^{SEP}`, then `g^{FOM}` | Define `FOM \equiv SEP` once or use `SEP` throughout.

### Undefined Notation

1. `\sigma_y` | First used in the technical appendix and later in the main paper | Add `\Sigma_y=\mathrm{diag}(\sigma_y^2)` in the notation section.
2. `\sigma_\eps(\theta)` | First used in the technical appendix | Define as the vector of structural-shock standard deviations and relate it to `\Sigma_\eps(\theta)`.
3. `p_t` | First used in the technical appendix | Add the soft-gate formula, e.g. log-sum-exp mixture versus linear averaging of log criteria.
4. `Q_{\text{lin}}`, `Q_{\text{surr}}`, `Q_{\text{gate}}` | First used around the likelihood/criterion decomposition tables | Define each criterion before the tables, including whether it is exact Kalman likelihood or profiled inversion objective.
5. `\mathcal{I}(\theta_0)`, `\kappa`, `\lambda_{\min}` | First active use in the coverage-distortion statement | Define these in the active technical appendix if the coverage-distortion statement remains.
6. Rotemberg equation symbols `mc_t`, `s_t^\pi`, `\phi_R`, `\bar\pi` | Used in the pricing robustness section | Add definitions before the equation or move the full pricing specification to a displayed equation with a notation note.

### Regression Specification Issues

1. Figure/regression specification | Text reports state, parameter, and shock `R^2` values as "contributions," but the caption says each feature set is regressed separately. | State clearly that these are separate univariate/group regressions, or report a joint regression/partial-`R^2` decomposition.
2. Equation-block table | `\Delta_j` is defined as the total scalar norm, but table shares require block-specific squared gaps. | Define `\Delta_{j,b}=P_b(g^{SEP}-g^{ROM1})` and shares as `\sum_j\|\Delta_{j,b}\|^2/\sum_{j,b}\|\Delta_{j,b}\|^2`.
3. SEP sensitivity table vs. appendix | Generated table says three draws and 40 periods with RMSE `6.16e-03`; appendix says five draws, full 265-period payload, and RMSE `1.08e-02`. | Regenerate the table or revise the appendix text so draw count, period count, and RMSE match.
4. Claim map | Says the 2-by-2 objective decomposition is under common inversion architecture, but the table includes exact Kalman likelihood cells. | Change scope to "mixed exact Kalman/profilled surrogate criteria" or use the linear+gate no-NN criterion for common-architecture comparisons.
5. Mode sensitivity table | "Log criterion" mixes Kalman likelihoods and surrogate profiled objectives without a printed note. | Split into "Kalman log likelihood" and "profiled surrogate objective," or add an explicit table note.

### LaTeX Math Formatting

1. Main Rotemberg Phillips curve | The equation is a long inline expression. | Put it in an `align` environment and define symbols below it.
2. Main Gal'i Taylor rule | The rule is too long for one display line. | Break into aligned lines with the lower-bound operator separated.
3. Generated SEP sensitivity table | Scientific notation appears as `7.97e-08`. | Use math notation: `\(7.97\times 10^{-8}\)`.
4. Main text and table intervals | Credible intervals are printed as bare `[56.6, 149.6]`. | Use math mode or define CI formatting consistently in tables.
5. Main residual notation | Superscripts alternate between `\mathrm{SEP}` and `\text{FOM}`/`\text{ROM1}`. | Use macros such as `\sep`, `\rom`, and `\mathrm{FOM}` consistently.

---

## 5. Tables, Figures & Documentation

## Agent 5: Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

1. Main Table 1 (`tab:equation_block`) | Gap units/scaling and variable abbreviations are not fully self-contained | Add whether variables are raw model units, transformed observables, or standardized target units; define each block's variable abbreviations or point to a block dictionary.
2. Main Table 2 (`tab:crisis_vs_normal`) | RMSE units and `---` entries are ambiguous | State observable units used for RMSE and define whether `---` means not reported, not applicable, or excluded from the variance-share accounting.
3. Main Table 4 (`tab:benchmarks`) | ESS mixes HMC and particle-filter meanings | Separate HMC ESS/divergences from particle effective sample size, or add a note defining the meaning by row.
4. Main Tables 6-7 (`tab:ll_decomposition`, `tab:linear_gate_ablation`) | Log-criterion units and gain direction need table-only clarity | Add "all entries are nats; higher/less negative is better; gains are row differences at fixed parameter vector."
5. Main Table 8 (`tab:csadjcost_sensitivity`) | Method and units incomplete | Define FOM/ROM1/SEP, parameter-draw source, shock-scale design, horizon, and whether `Gap` is observable RMSE or transition-function gap in model units.
6. Technical Appendix Table 1 (`tab:tech_claim_map`) | No table note/provenance | Add a note that claims summarize active-paper claims, distinguish direct SEP evidence from surrogate/posterior evidence, and point to artifact/provenance records.
7. Technical Appendix Table 2 (`tab:tech_gali_package`) | No note defining validation vocabulary | Define "Pass," "actual-floor," "OBC," "matched HMC," "post-warmup draws," and "combined MCSE."
8. Technical Appendix Table 3 (`tab:tech_hlt_bridge_marginals`) | No note defining posterior grid and interval construction | Add grid support, held-out truth selection, discrete posterior weights, and 90 percent interval definition.
9. Technical Appendix Table 4 (`tab:tech_hlt_bridge_rmse_obs`) | No note defining RMSE scale and observable abbreviations | Add observable-name mapping, RMSE units/scaling, held-out sample/grid support, and improvement formula.
10. Technical Appendix Table 5 (`tab:tech_validation_criteria`) | Criteria table not marked as protocol rather than result | Add a note that these are acceptance criteria, not empirical estimates, and identify which layers have passed.
11. Generated Table (`tab:sep_sensitivity`) | Table note conflicts with technical-appendix prose | Table says three draws and 40 periods; technical appendix says five draws and full 265-period payload. Reconcile the table or prose and define `tol`, `K`, `conv.`, residual norm, and reference RMSE.
12. Generated raw table (`generated/table_mode_sensitivity.tex`) | No standalone caption, label, or note | If active, wrap it in the technical appendix with caption/label/notes; if only archival, remove it from the active table list.

### Figures with Missing or Incomplete Notes

1. Main Figure 1 (`fig:nonlinearity_decomposition`) | Caption references panels (a)/(b), but figure has no panel labels | Add visible panel labels and define RMSE scale/normalization in the caption.
2. Main Figure 2 (`fig:all_vars_eqs`) | Y-axis units appear inconsistent with caption | Caption says percentage deviation, but panels plot decimal values; either rescale axes to percent or state "log deviations/model units." Add parameter draw/calibration and shock-sign definitions.
3. Main Figure 3 (`fig:rom1_forecast_errors`) | Caption sample conflicts with plotted figure | Caption says 1959Q1--2004Q4, but figure extends through COVID/2025. Regenerate for 1959Q1--2004Q4 or update caption/table text and crisis-quarter definition.
4. Main Figure 4 (`fig:decomposition_vs_scale`) | Caption says shock scales 0.1--1.5, figure shows only 0.8--1.5 | Align x-axis/caption and add shock design, horizon, and averaging support.
5. Main Figure 5 (`fig:gali_obc_vs_no_obc`) | Averaging support and gap units missing | State model calibration, shock type/grid, horizon, and units of mean `|FOM-ROM1|` gap.
6. Main Figure 6 (`fig:covid_shock_comparison`) | TFP panel label is corrupted and dotted bands are undefined | Fix the TFP shock label and define the horizontal dotted bands.
7. Listed figure (`fig:gap_magnitude_vs_scale`) | Caption mentions binding-period investment share, but plotted figure does not show that series | Add the missing series or remove it from the caption.
8. Listed figure (`fig:gali_block_decomposition`) | Caption/visual conflict | Caption describes Gal'i model with ELB/Kimball; visual title says "no ZLB" and "observable-block decomposition." Harmonize model variant and decomposition object.
9. Listed figure (`fig:zlb_binding_simulation`) | Simulation design not fully self-contained | Add initial state/parameter source, shock path details, and clarify annualized vs. quarterly percent units.
10. Listed figures (`fig:zlb_correction_robs`, `fig:zlb_correction_4panel`) | Episode origin and units incomplete | Define simulation period origin, shock path, parameter draw/model version, and y-axis units for all panels.

### Cross-Reference Issues

1. Technical appendix tables `tab:tech_claim_map`, `tab:tech_gali_package`, `tab:tech_hlt_bridge_marginals`, `tab:tech_hlt_bridge_rmse_obs`, `tab:tech_validation_criteria` | Labeled but not explicitly referenced with `Table~\ref{...}` | Add table callouts in surrounding prose.
2. `generated/sep_sensitivity_table.tex` | Only input inside inactive `\ifpaperappendix` legacy appendix, not the active technical appendix | If active, input it near `apptech:sep_sensitivity_record`.
3. `generated/table_mode_sensitivity.tex` | Only wrapped inside inactive legacy appendix and has no standalone label | If active, move wrapper/table into the technical appendix mode-sensitivity section.
4. Listed figures `gap_magnitude_vs_shock_scale`, `gali_block_decomposition`, `fig_zlb_binding_simulation`, `zlb_correction_robs_closeup`, `zlb_correction_timeseries_4panel` | Included only in inactive legacy appendix, despite being listed as active | Move to technical appendix or remove from active-asset list.
5. `fig_rom1_forecast_errors` | Active reference points to a figure whose plotted sample contradicts caption/table sample | Regenerate or retitle before submission.

### Formatting Inconsistencies

- Table notes | Main tables use `\tablenotes`; generated tables use ad hoc minipages; technical appendix tables mostly lack notes | Define and use one table-note macro in both documents.
- Terminology | ELB, ZLB, OBC, actual-floor, hard-ELB appear interchangeably | Choose one primary term and define aliases once.
- Model labels | FOM, SEP FOM, FOM (SEP truth), direct OBC, ROM1, linear, Kalman vary across captions/legends | Standardize legend/caption nomenclature.
- Units | Figures mix model units, decimals, percent deviations, annualized percent, quarterly percent, and sigma units | Put units on every axis and repeat them in captions where panels differ.
- Caption style | Some captions are title case, others sentence case with long interpretive claims | Standardize to short descriptive caption plus separate note-style details.
- Block names | Equation-block labels differ across main and sensitivity tables | Use the same block taxonomy/order across decomposition tables and figures.

---

## 6. Contribution & Referee Assessment

## Agent 6: Contribution Evaluation

### Part 1 -- Central Contribution

The strongest contribution is not the posterior exercise; it is the computational decomposition showing that, in the HLT Smets--Wouters specification, the FOM--ROM1 gap is dominated by the investment/capital block rather than by the ELB or Kimball pricing. That is interesting and potentially publishable if framed as a disciplined audit of nonlinearities in medium-scale DSGE models.

The methodological contribution is promising but not yet top-field complete. The ROM1-residual surrogate is a useful computational device, and the 10,000-cell HLT bridge is impressive. But the bridge validates a one-period, known-feature, four-parameter, finite-support comparison. It does not validate the full 18-parameter, multi-period, shock-inversion, gated empirical posterior. The empirical contribution is weaker still: it is conditional on the HLT specification, omitted ARMA markup terms, one warm-started basin, a profiled objective, and a gate that fails in calm holdouts.

Rating: promising but not ready for external review at a leading field journal. I would desk reject in current form, with encouragement to resubmit after validation and reframing.

### Part 2 -- Identification and Credibility

The paper is unusually honest about limitations, which helps credibility. The direct SEP decomposition, shock-scale checks, Gal'i negative control, pricing-variant checks, and adjustment-cost sensitivity all point in the same direction: investment nonlinearities matter more than pricing or ELB nonlinearities in this benchmark. That part is credible.

The credibility problem is the posterior layer. The paper repeatedly calls the empirical criterion a "profiled objective," not a likelihood, because shocks are recovered rather than integrated out and the determinant/Laplace terms are missing. That is the right language, but it means the "nonlinear posterior" in the title and tables is not a posterior in the usual Bayesian model-comparison sense. The HMC evidence is also not decisive: warm-started surrogate chains find the high-objective basin, while cold starts remain around a much worse region; the table shows objectives near `-14534` for cold surrogate starts versus about `-1781` to `-1787` for warm starts. That is not just a nuisance diagnostic; it means posterior mass and basin selection are unresolved.

The ARMA markup issue is equally serious. Restoring the omitted MA markup terms improves the linear likelihood by about 34 nats, almost the same size as the extended-sample surrogate objective gain of 35.92 nats. This makes the empirical posterior interpretation too specification-dependent for the current title and contribution claim.

### Part 3 -- Analyses: Required and Suggested

**Required:**

1. Provide a full multi-period validation bridge for the HLT empirical architecture: recovered shocks, gate, state propagation, and at least a reduced but genuinely dynamic posterior comparison against direct SEP. The current one-period HLT bridge is not enough.
2. Derive and implement the missing determinant/Laplace adjustment for the inversion objective, or explicitly demote all empirical posterior language to "profiled fit criterion" throughout the paper, including the title.
3. Run a systematic mode-search protocol for the surrogate objective: multi-start, tempering, or another defensible global exploration method. Warm-started-chain evidence alone is insufficient.
4. Re-estimate or at least robustly benchmark the nonlinear exercise with restored ARMA(1,1) markup terms. The current HLT-only posterior story is too confounded by the omitted markup block.
5. Resolve the gate problem. The q95 recalibration improves the quiet-sample failure but still loses to ROM1 over the full calm holdout; a leading-field submission needs a pre-specified gate or a model-based continuous mixture with out-of-sample validation.

**Suggested:**

1. Reframe the paper around the deterministic investment-channel audit and move the empirical posterior comparison to supporting evidence.
2. Add a table separating claims by evidentiary status: direct SEP fact, surrogate-validated fact, profiled-objective result, and speculative interpretation.
3. Include a no-NN gated HMC posterior run, not only fixed-parameter no-NN evaluations.
4. Compare against second-order or pruned perturbation once the failed ROM2 pilot is repaired, since referees will ask whether a neural surrogate is necessary.
5. Rename the paper to avoid implying an exact nonlinear posterior; for example, "Investment Adjustment Costs and Nonlinear Solution Gaps in Smets--Wouters."

### Part 4 -- Literature Positioning

The paper positions itself across nonlinear DSGE solution methods, surrogate modeling, and filter-free HMC. That is sensible, but the current contribution is narrower than the literature framing suggests. It is not yet a general method for nonlinear DSGE estimation, because the full empirical posterior is not validated against direct nonlinear inference. It is also not yet a broad empirical rejection of linear Smets--Wouters estimation, because the strongest empirical comparison is HLT-specific and confounded by markup-shock specification.

The paper's best literature claim is: researchers often focus on ELB and pricing nonlinearities, but in a SW/HLT environment the investment block can dominate the nonlinear solution gap. That is a clean and useful contribution. The surrogate should be framed as an enabling computational instrument, not as the main validated econometric advance unless the missing validation is added.

### Part 5 -- Journal Fit and Recommendation

Recommendation: do not send to referees yet.

For a leading field journal, the paper is too internally conditional. The decomposition is strong enough to deserve attention, but the paper currently asks the reader to accept a chain of qualifications: finite-support bridge, profiled objective, warm-started basin, omitted ARMA markup terms, gate-sensitive forecasts, and no full 18-parameter direct-SEP posterior benchmark. A top-field referee will likely treat these as central threats, not footnotes.

Path to improvement: narrow the claim, strengthen the dynamic validation, and cleanly separate the deterministic nonlinear-solution contribution from the empirical posterior exercise. If the authors can show that the investment-block result survives restored markup dynamics and that the surrogate/inversion/gate pipeline approximates a direct nonlinear dynamic benchmark, the paper could become a serious field-journal submission.

### Part 6 -- Questions to the Authors

1. Your title refers to the "nonlinear posterior," but the empirical object is a profiled inversion objective without determinant or Laplace terms. In what precise sense is this a posterior rather than a criterion-based estimator?
2. The HLT bridge validates one-period, known-feature, four-parameter support. Why should a referee view this as sufficient evidence for the 18-parameter, multi-period, recovered-shock, gated empirical exercise?
3. Cold-started surrogate chains find an objective region near `-14534`, while warm-started chains find the high-objective region near `-1781` to `-1787`. What evidence do you have about the posterior mass of these regions, not just their local objective values?
4. Restoring the ARMA markup terms improves the linear likelihood by about 34 nats, nearly matching the 35.92-nat surrogate objective gain. Why should the empirical posterior shifts be interpreted as nonlinear-investment evidence rather than as an artifact of the restricted HLT markup specification?
5. The quiet-sample holdout remains worse than ROM1 even after q95 gate re-estimation. What is the intended operational rule for deciding when the nonlinear correction is active, and is it selected without looking at the holdout?
6. The decomposition assigns most of the FOM--ROM1 gap to investment/capital, but removing the adjustment cost increases the gap dramatically. How should readers distinguish "investment adjustment costs are the source of nonlinearity" from "investment volatility is the carrier of other nonlinearities once the adjustment-cost friction changes"?
7. What result would convince you that the empirical posterior comparison is not robust? Please state the falsification criterion before adding further robustness exercises.

---

## Priority Action Items

The following issues require attention before submission, ordered by priority. The triage follows: identification and credibility failures, missing required analyses, internal inconsistencies, tables/figures documentation, mathematical errors, then style and grammar.

**CRITICAL** (must fix -- these could cause desk rejection or major referee objections):

1. Resolve the active-paper/provenance mismatches: SEP sensitivity table vs. prose, forecast-error figure sample, shock-scale figure support, active appendix references, and Gal'i/Kimball claims.
2. Repair or demote "nonlinear posterior" language: derive the determinant/Laplace/full objective, or call the empirical object a profiled/hybrid criterion throughout title, tables, captions, and conclusion.
3. Add a dynamic HLT validation bridge for recovered shocks, gate, state propagation, and posterior/criterion approximation; the 10,000-cell one-period bridge is not enough for a top-field claim.
4. Address the ARMA markup confound by restoring or benchmarking ARMA(1,1) markup terms in the relevant linear and nonlinear comparisons.
5. Resolve mode and gate credibility: use systematic multi-start/tempering or another defensible exploration protocol, and specify a gate that passes calm-period out-of-sample diagnostics without ex post tuning.

**MAJOR** (should fix -- will likely be raised by referees):

6. Tighten mechanism and causal language: do not call the Gal'i comparison a difference-in-differences design; do not interpret separate `R^2` decompositions as causal shares.
7. Standardize notation and terminology: ELB/ZLB/OBC, SEP/FOM/ROM1, shock names, gate notation, and objective units.
8. Move missing support material into the active technical appendix or remove references to it: particle-filter table, state-distance quintiles, parameter correlates, mode-sensitivity table, coverage-distortion theorem.
9. Make tables and figures self-contained: units on every axis, clear notes, consistent sample periods, definitions of `---`, ESS, nats, and RMSE scaling.
10. Clean generated artifacts before submission: no raw `e` notation, raw abbreviations, missing labels, or undocumented table wrappers.

**MINOR** (polish -- improves paper quality):

11. Finish copy/style polish: author-name consistency, acronym definitions, punctuation around displayed equations, U.S. spelling, and less colloquial phrasing.
12. Standardize caption style, panel labels, percent notation, and table note formatting across the main paper and technical appendix.

---

## Issue Counts

**Priority-action counts**: Critical 5; Major 5; Minor 2.

**Raw agent-finding counts**: Agent 1 flagged 8 critical style issues, 17 minor style issues, and 7 recurring style patterns. Agent 2 flagged 5 critical inconsistencies, 5 cross-reference errors, 4 terminology drifts, and 4 minor inconsistencies. Agent 3 flagged 6 causal overclaims, 7 generalization issues, 6 missing caveats, and 6 minor language issues. Agent 4 flagged 6 mathematical/objective issues, 6 notation inconsistencies, 6 undefined-notation issues, 5 specification issues, and 5 math-formatting issues. Agent 5 flagged 12 table-note issues, 10 figure-note issues, 5 cross-reference issues, and 6 formatting patterns. Agent 6 identified 5 required analyses and 5 suggested analyses.
