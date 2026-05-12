# Pre-Submission Referee Report

**Paper**: Investment Adjustment Costs and the Nonlinear Posterior of Smets--Wouters  
**Authors**: Mátyás Farkas  
**Date**: 2026-05-06  
**Review Standard**: Leading Field Journal

---

## Overall Assessment

The paper has a credible and potentially publishable core: a deterministic SEP/FOM--ROM1 decomposition showing that the investment and capital block, not ELB or Kimball pricing, accounts for most of the nonlinear-linear gap in the SW07-HLT model. The main strength is that this result is structural, computationally transparent, and largely independent of the empirical posterior machinery. The single most critical issue is that the empirical posterior and forecasting story remains too conditional: warm-started mode dependence, unresolved direct posterior validation, gate fragility in quiet samples, and specification sensitivity from ARMA markup terms all prevent a leading-field submission from being referee-ready.

**Preliminary Recommendation**: Substantial revision required.

---

## 1. Spelling, Grammar & Style

## Agent 1: Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. `SurrogateNN_paper.tex:150` | "A quiet-sample forecast pilot fails under the original gate..." -> "A quiet-sample forecast exercise indicates that gate calibration remains a limitation." | The abstract is too long and reads like an internal robustness log rather than a journal abstract.
2. `SurrogateNN_paper.tex:283` | "pre-submission requirement... current direct-SEP runtime blocker" -> "Appendix~... describes the validation protocol and reports the current validation status." | Avoid internal process language in a submission draft.
3. `SurrogateNN_paper.tex:418` | "The previous six-panel investment-shock figure is omitted from this draft because one panel was empty in the generated PDF." -> Delete. | This is production-note language, not academic prose.
4. `SurrogateNN_paper.tex:513` | "current-runtime failure... restores finite exact-determinant HLT direct-SEP smoke evaluations" -> Rewrite as a concise validation-status paragraph or move to replication notes. | "smoke evaluations," "current-runtime," and implementation debugging details are not submission style.
5. `SurrogateNN_paper.tex:601` | "remain far below the Kalman benchmark in the artifacted comparison" -> "remain far below the Kalman benchmark in this comparison" | "Artifacted" is not standard academic usage.
6. `SurrogateNN_paper.tex:713,715,1961-1967,2411-2415` | "pending," "artifact," "blocker," "local artifacts," "I therefore remove it" -> Replace with neutral appendix language or remove. | These passages expose internal workflow and weaken the manuscript's professional presentation.
7. `SurrogateNN_paper.tex:1105` | "a subtle bug that I discovered caused systematic convergence failures..." -> "This check prevents failed warm starts from propagating across the parameter grid." | Avoid first-person debugging narrative.
8. `SurrogateNN_paper.tex:2281,2315,2344,2365,2382` | "payload," "Raw artifacts," file paths to `.local_artifacts` -> Remove from paper; put in replication appendix or README. | Raw file provenance is inappropriate in the manuscript body.
9. `generated/table_mode_sensitivity.tex:4-16` | "LL", "Div.", "Accept", "linear seed99", "cold seed2" -> "Log likelihood", "Divergences", "Acceptance rate", "Linear, seed 99", etc. | Generated table labels look informal and code-derived.
10. `SurrogateNN_paper.tex:150,181,183,667,760` | "log likelihood", "log-likelihood", "posterior-mean log likelihood" -> Use one form consistently, preferably "log likelihood" as a noun and "log-likelihood" only as a compound modifier. | Typographic inconsistency in a central statistic.

### Minor Issues

1. `SurrogateNN_paper.tex:126` | "comments.  All errors" -> "comments. All errors" | Remove double space.
2. `SurrogateNN_paper.tex:148,179,191,2236` | "US data" -> "U.S. data" or "United States data" consistently. | Top journals usually prefer consistent country abbreviation.
3. `SurrogateNN_paper.tex:177,503` | "dampening" -> "damping" | "Damping" is the standard term for reducing volatility.
4. `SurrogateNN_paper.tex:225` | "Q/L = 18.7\% at $1\sigma$" -> "The Q/L ratio is 18.7 percent at $1\sigma$." | More polished academic syntax.
5. `SurrogateNN_paper.tex:244` | "$\eps_t \in \R^{n_\eps} \sim \mathcal{N}(0, I_{n_\eps})$ are shocks" -> "$\eps_t \in \R^{n_\eps}$ denotes shocks with $\eps_t \sim ...$" | Current construction is grammatically awkward.
6. `SurrogateNN_paper.tex:283` | "negative-control model" -> "negative control" or "negative-control specification" | Use the compound consistently.
7. `SurrogateNN_paper.tex:340` | "50--100$\times$ smaller" -> "50 to 100 times smaller" | Avoid mixed math/text notation in prose.
8. `SurrogateNN_paper.tex:425` | "SEP FOM positive... SEP FOM negative" -> "SEP, positive shock... SEP, negative shock" | Caption labels are terse.
9. `SurrogateNN_paper.tex:735` | "All 25 theta draws converge" -> "All 25 parameter draws converge" | Avoid code variable names in prose.
10. `generated/sep_sensitivity_table.tex:5-13` | Caption is a long multi-sentence paragraph. -> Shorten caption and move details to notes. | Table captions should identify the table, not carry full methods text.
11. `csadjcost_sensitivity_table.tex:9` | "Invest", "Cons" -> "Investment", "Consumption" | Avoid nonstandard abbreviations in table headers.
12. `SurrogateNN_paper.tex:2299,2313,2335` | "labour" -> "labor" | The paper otherwise uses U.S. spelling.

### Style Patterns to Fix Throughout

- Internal provenance language: remove audit/build/debug language from the manuscript and place reproducibility details in `REPLICATION.md`.
- Overlong defensive sentences: split result sentences from limitation sentences, especially in the abstract.
- Inconsistent lower-bound terminology: define ELB/ZLB/OBC once and use each term only in its intended scope.
- Mixed U.S./British spelling: use U.S. spelling throughout.
- Code-derived table language: convert generated headers to publication labels.
- Inconsistent percentage style: use "percent" in prose and `\%` in tables.
- Informal causal verbs: replace "drives," "causes," "confirms," "collapses," and "truth" where they exceed the evidence.

---

## 2. Internal Consistency & Cross-Reference Verification

## Agent 2: Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. `SurrogateNN_paper.tex:173` <-> `SurrogateNN_paper.tex:480` | Intro says that during ELB-binding episodes investment dominates the next-largest block by a factor of eight; later text says binding-episode investment share falls to 37% at shock scale 1.5. The factor-eight claim only matches the unconditional 89--92% figure, not binding episodes. | Severity: CRITICAL
2. `SurrogateNN_paper.tex:1943` <-> `SurrogateNN_paper.tex:2095,2099,2129` | Formal-consistency discussion claims the 18-parameter Monte Carlo coverage study indicates realized coverage distortion below the worst-case bound; the cited coverage table is for the linear Kalman + NUTS estimator, not the surrogate, and average coverage is 79.8% versus nominal 90%. | Severity: CRITICAL
3. `SurrogateNN_paper.tex:300,746` <-> `SurrogateNN_paper.tex:575,715` | Posterior-impact and coverage claims use realized RRMSE 0.0012 / <0.002, while ZLB-binding validation reports RRMSE = 0.12, two orders of magnitude larger, in the lower-bound region used to motivate the extended sample. | Severity: CRITICAL
4. `SurrogateNN_paper.tex:715` <-> `SurrogateNN_paper.tex:2399-2406` | Text says surrogate accuracy remains within RRMSE < 0.002 across full parameter support, but the regional table reports tail 95th percentile RRMSE = 0.00231 while its note still says all regions satisfy <0.002. | Severity: CRITICAL

### Cross-Reference Errors

1. `SurrogateNN_paper.tex:173`, `Figure~\ref{fig:decomposition_vs_scale}` | Target: `SurrogateNN_paper.tex:476` | The figure describes average block shares across all periods and shock scales, not ELB-binding episodes.
2. `SurrogateNN_paper.tex:410`, `Table~\ref{tab:per_obs_r2}` | Target: `SurrogateNN_paper.tex:2180-2196` | Text cites this table for autocorrelation 0.81 and mean burst duration 12.5 quarters, but the table only reports per-observable R² values.
3. `SurrogateNN_paper.tex:575`, `Figures~\ref{fig:zlb_correction_robs}--\ref{fig:zlb_correction_4panel}` | Target: `SurrogateNN_paper.tex:2429-2436` | The figures show ZLB correction time series, not the stated binding-period RRMSE = 0.12 diagnostic.
4. `SurrogateNN_paper.tex:503`, `Figure~\ref{fig:decomposition_vs_scale}` | Target: `SurrogateNN_paper.tex:476` | Text uses the figure to support a counterfactual "both linearized" residual-gap claim; the figure only reports baseline shock-scale block shares.

### Terminology Drift

1. Lower-bound constraint | "zero lower bound," "effective lower bound," "ZLB," "ELB," "lower-bound," and "OBC" are used interchangeably. | Use "ELB" for the empirical U.S. episode, "ZLB" only for the model floor \(R_t \ge 1\), and "OBC" only for generic method discussion.
2. Nonlinear solution benchmark | "FOM," "SEP," "full-order," "full nonlinear," "FOM truth," and "true nonlinear likelihood." | Define once as "SEP/FOM" and use "SEP solution" for the solver object, "FOM--ROM1 gap" for decompositions.
3. Surrogate estimator | "RS Surrogate," "Switch," "switching model," "regime-switching surrogate," "nonlinear model," and "surrogate posterior." | Use "regime-switching surrogate" in prose and "RS surrogate" in tables.
4. Kimball curvature notation | \(\varepsilon_p\) dominates, but one passage uses \(\epsilon_p\). | Use \(\varepsilon_p\) throughout.
5. Labor/labour | U.S.-style "labor" in text and "labour" in forecast tables. | Use "labor" throughout.

### Minor Inconsistencies

1. `SurrogateNN_paper.tex:412,527-528` <-> `SurrogateNN_paper.tex:556,613-615` | The mechanism story around \(\sigma_b\) rising in the baseline and falling in the extended sample remains internally confusing despite partial explanation. | Severity: MINOR
2. `SurrogateNN_paper.tex:2095` <-> `SurrogateNN_paper.tex:2107,2125,2133` | Text says two parameters fall outside the Clopper--Pearson coverage check and names \(\rho_b\) and \(\iota_p\); the table appears to contain nominal 90% for \(\iota_p\), suggesting 17 of 18 rather than 16 of 18. | Severity: MINOR
3. `SurrogateNN_paper.tex:2002` <-> `generated/table_mode_sensitivity.tex:10-16` | Text describes an initial four-chain cold-start diagnostic with seeds 42, 2, 3, 4; the generated table lists cold seeds 2--4 and warm seed42, so seed42's role is inconsistent. | Severity: MINOR
4. `SurrogateNN_paper.tex:150` <-> `SurrogateNN_paper.tex:183,2445-2522` | Abstract says results should be read relative to "this nonlinear HLT benchmark," but the ARMA restoration is a linear HLT specification comparison. | Severity: MINOR

---

## 3. Unsupported Claims & Identification Integrity

## Agent 3: Unsupported Claims & Identification Integrity

### Causal Overclaiming (must address)

1. `SurrogateNN_paper.tex:177` | "Three diagnostics point away from the usual alternatives." | Diagnostics are suggestive, not exclusionary. | Fix: "Three diagnostics are consistent with the investment block being more important than the ELB and pricing in these HLT exercises."
2. `SurrogateNN_paper.tex:177` | "removing \(S(\cdot)\) from the model causes the gap to explode" | This is a structural counterfactual inside a changed model. | Fix: "In this counterfactual specification, removing \(S(\cdot)\) substantially increases the measured FOM--ROM1 gap."
3. `SurrogateNN_paper.tex:406` | "a compounding \(S' \times p^k \times \xi_{t+1}/\xi_t\) interaction" | Mechanism is inferred, not term-level isolated. | Fix: call it a likely mechanism or add a term-level ablation.
4. `SurrogateNN_paper.tex:528` | "it must rely on the exogenous risk-premium shock" | "Must rely" overstates identification. | Fix: "The posterior pattern is consistent with..."
5. `SurrogateNN_paper.tex:530` | "the linear model relying on persistent wage markup shocks where the nonlinear model uses state-dependent amplification" | This states an unobserved allocation mechanism as fact. | Fix: frame as consistent with substitution, not proof.
6. `SurrogateNN_paper.tex:535` | "indicating that the shifts reflect genuine nonlinear effects" | SEP gaps at selected draws do not prove posterior shifts are caused by nonlinear effects. | Fix: acknowledge mode, prior, gate, and surrogate channels.
7. `SurrogateNN_paper.tex:615` | "the nonlinear channel then reduces \(\sigma_b\) by absorbing COVID-era investment dynamics endogenously" | Mechanism is inferred from posterior movement. | Fix: "is consistent with..."
8. `SurrogateNN_paper.tex:681` | "attributes 29.05 nats (80.9 percent) to the model change" | The two-by-two decomposition is path-dependent and lacks uncertainty. | Fix: "A symmetric posterior-mean accounting exercise assigns..."
9. `SurrogateNN_paper.tex:681` | "supports the nonlinear channel as the main source of the likelihood gain" | Does not rule out gate/inversion, mode search, or specification effects. | Fix: "is consistent with..."
10. `SurrogateNN_paper.tex:707` | "same underlying channel" | Deterministic and posterior results are aligned but not causally linked. | Fix: "point in the same direction."
11. `SurrogateNN_paper.tex:752` | "the economy is driven more by endogenous propagation than standard estimates suggest" | Moves from one model-specific posterior comparison to a statement about the economy. | Fix: "Within this HLT specification..."
12. `SurrogateNN_paper.tex:2313` | "post-COVID under-performance reflects a gating issue rather than a model-fundamental disadvantage" | Gate diagnostic does not rule out model or filter issues. | Fix: "appears partly gate-related..."
13. `SurrogateNN_paper.tex:2522` | "confirms that the ARMA(1,1) markup structure captures genuine data variation" | A 34-nat likelihood gain may reflect flexibility or specification. | Fix: "materially improves in-sample fit and changes posterior geometry."

### Generalization Issues

1. `SurrogateNN_paper.tex:185` | "For medium-scale New Keynesian models..." | Evidence is one SW07-HLT specification and one U.S. sample. | Narrow to the model studied.
2. `SurrogateNN_paper.tex:193` | "Recent nonlinear estimation work...faces either..." | Sweeping literature claim. | Use "several recent approaches."
3. `SurrogateNN_paper.tex:225` | "The ELB contributes only above \(3\sigma\)..." | Threshold is model- and shock-design-specific. | Restrict to the Galí exercise.
4. `SurrogateNN_paper.tex:272` | "Standard DSGE estimation..." | Current DSGE estimation uses more than random-walk MH. | Use "a common benchmark."
5. `SurrogateNN_paper.tex:300` | "sampler explores the entire parameter space" | NUTS explores posterior typical sets, not the entire space. | Rewrite.
6. `SurrogateNN_paper.tex:609` | "generalizes to any sample period..." | Time transfer also depends on measurement, gate behavior, shock normalization, and OOD states. | Add conditions.
7. `SurrogateNN_paper.tex:724` | "regardless of specification" | Only three alternatives are tested. | Say "across the three pricing alternatives considered here."
8. `SurrogateNN_paper.tex:737` | "confirms that investment dominance is a structural property..." | Grid varies only \(S''(1)\) and 25 draws. | Say "supports the view..."
9. `SurrogateNN_paper.tex:1622` | "particle filtering is infeasible at any realistic computational budget" | Overgeneralizes from bootstrap/COPF tests under one design. | Narrow to tested filters and budgets.
10. `SurrogateNN_paper.tex:1949` | "consistency can be guaranteed a priori..." | Universal approximation is not a finite-sample guarantee. | Rewrite as "can in principle be reduced..."
11. `SurrogateNN_paper.tex:2514` | "robust to model specification" / "well-identified regardless..." | Based on limited comparison. | Say "stable across specifications compared in Table..."

### Missing Caveats

1. Block-share attribution | Add after Table 1 and in abstract: shares depend on grouping, normalization, shock design, and accepted SEP sample.
2. Posterior-mean likelihood comparisons | Add around the 35.92-nat discussion: this is not a marginal likelihood, Bayes factor, or full posterior model comparison.
3. Mode selection | State explicitly that posterior mass across basins is not estimated.
4. Identification diagnostics | Clarify that \(\hat R\) and ESS are sampling diagnostics, not identification diagnostics.
5. Formal consistency | State that assumptions are not verified in the multimodal finite-sample application.
6. Direct validation | Say the missing Galí direct SEP-HMC comparison means the surrogate posterior is not yet matched against a direct nonlinear posterior.
7. Forecast exercise | Describe OOS exercises as conditional one-step prediction/filtering diagnostics, not strict real-time forecast comparisons.
8. Gate recalibration | Note q95 is partly post hoc and fixed-posterior recalibration is not a robustness proof.
9. Solver sensitivity | State that SEP sensitivity covers three posterior draws and a 40-period window; full 265-period sensitivity remains pending.
10. Monte Carlo coverage | State that MC coverage uses a first-order DGP and linear estimator; it does not validate nonlinear-surrogate coverage.
11. COVID shock magnitudes | Note dependence on smoother, measurement-error calibration, pandemic data treatment, and shock normalization.
12. Input-table representativeness | Define how 25 representative posterior vectors are selected.

### Minor Language Issues

1. `SurrogateNN_paper.tex:148` | "Full Bayesian estimation" | Too strong for surrogate/gate/inversion likelihood. | Use "Surrogate-based Bayesian estimation."
2. `SurrogateNN_paper.tex:187` | "where it matters" | Presumes the gate is correct. | Use "periods classified as stressed."
3. `SurrogateNN_paper.tex:227` | "confirms every prediction" | Too strong. | Use "finds patterns consistent with..."
4. `SurrogateNN_paper.tex:486` | "difference-in-differences design" | It is a model comparison, not credible DiD. | Use "negative-control model comparison."
5. `SurrogateNN_paper.tex:503` | "irrelevant" | Too absolute. | Use "small in this gap metric."
6. `SurrogateNN_paper.tex:752` | "the opposite of current practice" | Unsupported and polemical. | Use "different from the common emphasis..."
7. `SurrogateNN_paper.tex:2004` | "The fix is straightforward" | Minimizes a serious mode-selection problem. | Use "A practical workaround is..."
8. `SurrogateNN_paper.tex:2311` | "the intended test" | Reads ex post. | Use "stress-window diagnostic."
9. `SurrogateNN_paper.tex:2436` | "FOM truth" | Loaded term. | Use "SEP benchmark."

---

## 4. Mathematics, Equations & Notation

## Agent 4: Mathematics, Equations & Notation

### Mathematical Errors

1. `SurrogateNN_paper.tex:175,213,223,895-899` | The invisibility claim is over-applied. \(S(x)\) is first-order invisible if it enters in levels, but \(S'(x)\) is not: \((S')'(1)=S''(1)\neq0\). | Revise: level adjustment cost is invisible in capital accumulation at first order, while marginal adjustment-cost terms are first-order visible; nonlinear residual comes from products/cross-terms such as \(S'_t p_t^k \xi_{t+1}/\xi_t\).
2. `SurrogateNN_paper.tex:175,213,893` | Cost-function parameterization is inconsistent. If \(S(x)=\frac{S''}{2}(\gamma x-\gamma)^2\), then \(S_{xx}(1)=S''\gamma^2\), not \(S''\). | Either write \(S(x)=\frac{S''(1)}{2}(x-1)^2\) or define \(\phi=S''(1)/\gamma^2\).
3. `SurrogateNN_paper.tex:322-324,1056-1060` | Softplus error calculation is wrong. At \(\tilde R_t-1=-0.02\) and \(\kappa_s=100\), excess over the floor is \(0.01\log(1+e^{-2})\approx0.00127\), about 12.7 bps in gross-rate units, not 0.01 bps. | Correct bound or increase \(\kappa_s\).
4. `SurrogateNN_paper.tex:274-278,1361-1366,1419-1427` | The "likelihood" after shock inversion is a penalized/profile likelihood unless the change-of-variables or Laplace determinant is included. | Add determinant/Hessian term or rename as quasi-likelihood / inversion objective.
5. `SurrogateNN_paper.tex:1690-1694` | Equation `eq:ll_bias_bound` is not a valid pathwise bound; Gaussian log-likelihood differences have a linear residual-error term. | Replace with residual-dependent expected bound or state orthogonality conditions.
6. `SurrogateNN_paper.tex:1937-1943` | RRMSE threshold is off by factor 10. Displayed formula implies one-percentage-point threshold about 0.0013, not 0.013. | Recompute threshold and revise interpretation.
7. `SurrogateNN_paper.tex:1943,2093-2095` | MC coverage study is for linear Kalman estimator, not surrogate posterior, so it cannot show realized surrogate coverage distortion. | State it validates the linear HMC pipeline only.

### Notation Inconsistencies

1. \(g_t\) | Gate indicator vs policy function \(g_\theta\). | Rename gate indicator to \(d_t\) or \(m_t\).
2. \(\varepsilon_t\) | Standard-normal shocks, structural AR(1) shock states, and raw markup innovations. | Use separate notation for standardized innovations and shock states.
3. \(\sigma_y,\Sigma_y\) | Vector standard deviations and covariance matrix. | Define \(\Sigma_y=\operatorname{diag}(\sigma_y^2)\).
4. \(H\) | Nonlinear measurement function and selection matrix. | Use \(h(s_t)\) for function or matrix notation consistently.
5. \(\Delta y_j,\hat y\) | Training residual/transition output vs observables. | Reserve \(y_t\) for observables.
6. \(S''\), \(S''(1)\), \(\phi\) | Curvature parameter varies. | Pick one convention and propagate.
7. \(\kappa\) | NKPC slope and Fisher-information condition number. | Rename condition number to \(\kappa_{\mathcal I}\).

### Undefined Notation

1. FOM / ROM1 | First used before full definition. | Define before first use.
2. Q/L ratio | First used at `SurrogateNN_paper.tex:221`. | Add formula for numerator and denominator.
3. \(\tilde w^{-6}\) | First used before defining \(\tilde w\). | Define reset relative wage or move after model specification.
4. \(\bar s(\theta)\) and \(\tau\) | Used in gate equation. | Define standardized state distance and threshold calibration before equation.
5. Table variables \(\dot w,\gamma^w_{1-3},s^w,D_p,\dot p,\gamma_{1-3},s^\pi,S^D,m\) | Appear in Table 1. | Add table notes or notation table.
6. Growth observables | Listed without measurement equations. | Add log-difference, annualized/quarterly scaling, trend growth, and percent-vs-decimal convention.

### Regression Specification Issues

1. Gate specification | Main text uses state distance; appendix uses shock magnitude and forecast error. | Choose one or state conceptual vs implemented definitions.
2. Inversion filter | Main text says shocks are recovered under the linear model; appendix says Phase 1 propagates with ROM1+NN when available. | Align algorithm and likelihood description.
3. Nonlinearity regressions | \(\Delta_j\) definition, feature regressions, and block shares need exact equations and normalization. | Add regression specification.
4. OOS forecast design | Text says forecasts use "observed shocks" in holdout. | If shocks are recovered from holdout data, call it one-step-ahead conditional prediction/filtering, not unconditional forecast.
5. `csadjcost_sensitivity_table.tex` | Main decomposition baseline is 68.9%, while sensitivity table reports 92.2% at baseline. | Add note that samples/designs differ and shares are not directly comparable.

### LaTeX Math Formatting

1. `csadjcost_sensitivity_table.tex:8-16` | Uses `\hline\hline` despite `booktabs`. | Use `\toprule`, `\midrule`, `\bottomrule`.
2. `SurrogateNN_paper.tex:351,1324` | Long optimization objective is inline. | Display and label equation.
3. `SurrogateNN_paper.tex:1425` | `\text{logsumexp}` in math mode. | Declare operator.
4. `SurrogateNN_paper.tex:98-101` | Operator declarations can be improved. | Use `\DeclareMathOperator*{\argmin}{arg\,min}` etc.
5. `SurrogateNN_paper.tex:283` | Full Taylor rule is a long inline equation. | Move to display math.
6. `SurrogateNN_paper.tex:1965-1991` | Long `\path` entries produce overfull boxes. | Break paths, use smaller font, or move artifact paths to replication notes.

---

## 5. Tables, Figures & Documentation

## Agent 5: Tables, Figures & Documentation

### Tables with Missing or Incomplete Notes

- Table 1 | Missing scaling/units for `Mean ||Delta||_2`. | Add whether gaps are raw model units, log deviations, observable units, or standardized variables.
- Table 2 | RMSE units and `---` entries undefined. | State RMSE units and define `---`.
- Table 3 | Shock-volatility and likelihood units incomplete. | Add that likelihoods are in nats and define HLT shock-volatility scaling.
- Table 5 | Same volatility-unit issue as Table 3. | Add scale note.
- Table 8 | Posterior/sample provenance incomplete. | Replace "drawn from the posterior" with exact posterior/mode/sample; spell out `Invest`, `Cons`, and define `Gap`.
- Table 11 | Prior table omits units for volatility bounds. | Add internal shock-volatility scaling.
- Table 13 | Note depends on previous table. | Repeat sample, posterior-mean evaluation points, particle counts, and seed count.
- Table 15 | Notes embedded in caption; residual "raw units" unclear. | Use short caption plus notes; define raw residual units and reference-cell interpretation.
- Table 18 | Column definitions incomplete. | Define `LL`, `Div.`, `Accept`, and the sigma columns.
- Table 22 | RMSE units and improvement formula missing. | Define target/units and `Improvement = 1 - RMSE_surr/RMSE_ROM1`.
- Table 23 | Sample and p-value basis incomplete. | Add sample size and whether p-values are unadjusted.
- Table 24 | Sample definition missing. | Add 22,080-sample/18-parameter provenance.
- Table 29 | Not self-contained. | Repeat quiet estimation and forecast windows.
- Table 30 | Window definitions missing. | Define Full, Early, and Remaining.
- Table 31 | Region construction and RRMSE definition incomplete. | State Mahalanobis-distance percentiles, test sample, and RRMSE formula.

### Figures with Missing or Incomplete Notes

- Figure 1 | Caption uses Panel (a)/(b), but PDF has no panel labels. | Add visible labels or rewrite as left/right panel; define RMSE units.
- Figure 2 | Unit inconsistency. | Caption says percentage deviation, but plotted values appear as decimals.
- Figure 3 | Sample definition missing and likely inconsistent. | Figure appears to extend through 2025 while text/table describe 1959Q1--2004Q4.
- Figure 4 | Shock-scale range mismatch. | Caption says 0.1--1.5, rendered figure shows only 0.8--1.5.
- Figure 5 | Units/provenance incomplete. | Define shock scale and gap units.
- Figure 6 | Malformed panel label in PDF. | Top-left panel renders as `TFP (epsilon box)`; fix to `epsilon_a`.
- Figure 7 | Sample/grid details incomplete. | Add shock-scale grid, draws/periods, and mean observable gap definition.
- Figure 8 | Terminology/provenance incomplete. | Standardize gap vs observable-block decomposition.
- Figure 9 | Binding threshold wording unclear. | Align caption with plotted ZLB floor.
- Figure 10 | Simulation window not defined. | Add period range, shock scenario, and units.
- Figure 11 | Units, panels, and shock scenario missing. | Add panel order, annualized/quarterly units, model, shock scale, and period range.

### Cross-Reference Issues

- Figure 4 | Intro claims "during ELB-binding episodes" but references unconditional decomposition figure. | Reference binding-episode figure or add binding panel.
- Figure 3 / Table 2 | Text/table say 1959Q1--2004Q4, rendered figure appears through 2025. | Regenerate or update notes.
- Table 6 | Label exists but no `Table~\ref{tab:ll_decomposition}` reference. | Add reference.
- Table 15 | `tab:sep_sensitivity` labeled but not referenced. | Add reference.
- Table 16 | `tab:euler_errors` labeled but not referenced. | Add reference.
- Table 17 | `tab:param_recovery` labeled but not referenced. | Add reference.
- Table 18 | `tab:mode_sensitivity_chains` labeled but not referenced. | Add reference.
- Table 28 | Quiet-sample paragraph does not reference the table. | Add `Table~\ref{tab:oos_forecast_quiet}`.

### Formatting Inconsistencies

- Table-note style varies; standardize to `\tablenotes{\textit{Notes}: ...}`.
- Caption capitalization varies; choose one style.
- ELB/ZLB/lower-bound terminology varies.
- `Labor`, `labour`, `Hours worked`, and `Labor Market` vary.
- `RS Surrogate`, `Switch`, `Surrogate`, `NN`, `FOM`, `SEP FOM`, and `FOM truth` vary.
- Many figures include large internal titles duplicating captions.
- Axis units are not consistently marked.

---

## 6. Contribution & Referee Assessment

## Agent 6: Contribution Evaluation

### Part 1 — Central Contribution

The paper's best contribution is the quantitative claim that the investment/capital block, not the ELB or Kimball pricing, drives most nonlinear-linear disagreement in the SW07-HLT model: 68.9 percent of the FOM-ROM1 gap versus 0.2 percent for the price Phillips curve. The invisibility proposition is correct but conceptually elementary: any term with zero level and first derivative at steady state is absent at first order. The real contribution is empirical/computational: showing this channel is large in a medium-scale estimated model.

Rating: promising but not yet top-field referee-ready, roughly 4/10 as submitted. The deterministic decomposition is interesting; the posterior and forecasting evidence remain too conditional for a leading field journal.

### Part 2 — Identification and Credibility

The deterministic evidence is the credible core. The equation-block decomposition, Galí negative control, counterfactual variants, and sensitivity to \(S''(1)\) all point in the same direction. The \(S''(1)\) sensitivity table is especially helpful: investment remains 88.8 to 95.1 percent of the gap over \(S''(1)\in[2,10]\), while the price block is 0.0 percent.

The posterior evidence is much weaker. The baseline 1959--2004 nonlinear likelihood is worse than the linear likelihood at posterior means (\(-1{,}045\) versus \(-1{,}027\)), and the extended-sample gain of 35.92 nats is explicitly conditional on a warm-started surrogate mode. Cold-start surrogate chains land around \(-14{,}534.5\), while warm chains land around \(-1{,}781\) to \(-1{,}787\). The paper is transparent about this, but that transparency also reveals that the current empirical claim is not yet stable enough.

The direct posterior validation is pending: the Galí direct SEP-HMC versus surrogate-HMC comparison is not available, and the HLT direct comparison is only a smoke/provenance harness. The quiet-sample forecast evidence is also damaging: the original gate gives an aggregate RMSE ratio of 2.40 versus ROM1; even after q95 re-estimation the ratio is 1.09. For a leading field paper, the estimator cannot remain this dependent on gate calibration.

### Part 3 — Analyses: Required and Suggested

**Required:**

1. Complete the matched direct SEP-HMC versus surrogate-HMC validation on a smaller model, with common DGP, priors, seeds, chain lengths, posterior means, intervals, and coverage diagnostics.
2. Run a systematic multimodal posterior exploration for the 18-parameter model, preferably multi-start plus tempering, and report whether the warm-started high-likelihood basin is globally relevant.
3. Provide a full posterior ablation with the same gate/inversion architecture but no NN correction, not just fixed-parameter likelihood evaluations.
4. Resolve the gate-design problem with a prespecified rule and show it performs acceptably in both stressed and quiet samples.
5. Expand solver sensitivity beyond the current 3-draw, 40-period pilot; the paper itself says the full 265-period sensitivity rerun remains a pre-submission task.

**Suggested:**

1. Report decomposition robustness to alternative norm choices and block definitions.
2. Compare against second- or third-order perturbation as an intermediate benchmark.
3. Re-estimate or materially stress-test the model with the ARMA(1,1) markup block restored.
4. Add a model-class external validity exercise, e.g. CEE or JPT-style investment frictions.
5. Separate the substantive economics contribution from the surrogate-method contribution more cleanly.

### Part 4 — Literature Positioning

The paper is well aware of the relevant literatures: nonlinear DSGE estimation at the ELB, particle-filter degeneracy, stochastic extended path, surrogate likelihoods, and filter-free HMC. The contrast with ELB-centered nonlinear DSGE work is potentially useful.

The positioning currently overreaches. The paper sometimes frames the result as a reversal of "current practice," while the evidence is one model, one country, one specification, one main posterior basin, and a surrogate likelihood with unresolved validation. The paper should position itself as a sharp diagnostic for investment adjustment costs in SW-type models, not yet as a general reassessment of nonlinear DSGE estimation priorities.

### Part 5 — Journal Fit and Recommendation

Recommendation: do not send to referees in its current form for a leading field journal. The paper has a real idea and a potentially publishable computational economics contribution, but the empirical credibility is not yet at the desk-send threshold.

Path to improvement: make the deterministic decomposition the anchor, finish the direct validation, show posterior robustness across modes and specifications, and demonstrate a prespecified gate that does not damage quiet-sample performance. With those pieces, the paper could become a strong field-journal submission. Without them, referees will likely focus on surrogate validation, warm-start dependence, and gate fragility rather than the investment-channel insight.

### Part 6 — Questions to the Authors

1. Your central deterministic result is a block decomposition of the FOM-ROM1 gap. How sensitive is the 68.9 percent investment share to variable scaling, the Euclidean norm, and the assignment of variables such as \(q\), \(\xi_t\), and utilization to the investment block?
2. The extended-sample likelihood gain is reported only for the warm-started surrogate basin, while cold-started surrogate chains converge to much lower likelihood values. What evidence shows that the warm-started basin is not a narrow local mode selected by initialization?
3. The required Galí direct SEP-HMC versus surrogate-HMC posterior validation is still pending. Why should readers accept the 18-parameter surrogate posterior before seeing a matched direct posterior comparison in a setting where direct SEP is feasible?
4. The original quiet-sample forecast exercise fails badly, and q95 re-estimation still leaves the switching model worse than ROM1 over the full quiet holdout. What is the prespecified gate rule you would ask applied users to adopt?
5. Restoring the ARMA(1,1) markup terms improves the linear likelihood by 34 nats, about the same scale as the extended-sample nonlinear gain. Why is the nonlinear posterior comparison more informative than a specification comparison within the linear SW family?
6. The baseline 1959--2004 nonlinear posterior-mean likelihood is below the linear likelihood. Should the empirical claim be restricted to COVID-era stress episodes rather than presented as a general posterior consequence of investment-cost invisibility?
7. The surrogate improves prediction RMSE but has larger median nonlinear Euler residuals than ROM1 in the gate-on comparison. How should readers interpret estimates from an object that is accurate as a forecasting correction but not as a globally consistent model solution?

---

## Priority Action Items

The following issues require attention before submission, ordered by priority.

**CRITICAL** (must fix -- these could cause desk rejection or major referee objections):

1. Fix the formal investment-cost claim: distinguish level invisibility of \(S(\cdot)\) from first-order visibility of \(S'(\cdot)\), and correct the \(S''(1)\) / \(\gamma^2\) parameterization.
2. Resolve the inversion "likelihood" issue: either include the correct determinant/Laplace term or rename the object as a quasi-likelihood/profile inversion objective.
3. Complete matched direct SEP-HMC versus surrogate-HMC validation in a small model before relying on the 18-parameter surrogate posterior.
4. Run a systematic mode exploration for the 18-parameter posterior; current headline evidence remains conditional on the warm-started high-likelihood basin.
5. Resolve gate fragility with a prespecified gate rule that performs acceptably in both COVID-stress and quiet-sample windows.
6. Correct the RRMSE/coverage claims: ZLB RRMSE = 0.12 is not covered by the <0.002 statements, the regional table has 0.00231 in the tail, and the MC coverage study validates the linear HMC pipeline only.
7. Correct the softplus ELB approximation arithmetic; the stated 0.01 bp error is inconsistent with \(\kappa_s=100\).
8. Expand SEP sensitivity beyond the 3-draw, 40-period pilot or explicitly demote it from pre-submission evidence.
9. Correct the ELB-binding "factor of eight" claim and its reference to the unconditional decomposition figure.
10. Remove all internal audit/debug/provenance prose from the manuscript body and move file-path provenance to `REPLICATION.md`.

**MAJOR** (should fix -- will likely be raised by referees):

11. Add robustness of the 68.9 percent decomposition share to scaling, norms, and block definitions.
12. Add a full posterior linear+gate/no-NN ablation, not only fixed-parameter likelihood evaluations.
13. Stress-test or re-estimate with the ARMA(1,1) markup block restored, given its 34-nat fit gain.
14. Reframe all posterior likelihood comparisons as posterior-mean accounting exercises, not marginal likelihoods or model selection evidence.
15. Standardize terminology: ELB/ZLB/OBC, SEP/FOM/ROM1, regime-switching surrogate/Switch, labor/labour, and \(\varepsilon_p\).
16. Add missing caveats around \(\hat R\)/ESS as sampling rather than identification diagnostics.
17. Correct or regenerate Figure 3 if it extends through 2025 while text/table describe 1959Q1--2004Q4.
18. Correct Figure 4 shock-scale range mismatch and clarify binding vs unconditional decomposition.
19. Make all major tables self-contained: units, sample windows, scaling, posterior provenance, likelihood units, and raw-vs-standardized variables.
20. Add exact measurement equations for growth observables and define annualized/quarterly scaling.
21. Clarify OOS exercises as conditional one-step prediction/filtering diagnostics if holdout shocks are recovered from holdout data.

**MINOR** (polish -- improves paper quality):

22. Shorten the abstract and separate result claims from limitation claims.
23. Use U.S. spelling and U.S. country abbreviation consistently.
24. Replace code-derived labels in generated tables with publication labels.
25. Convert `csadjcost_sensitivity_table.tex` to booktabs style.
26. Add references to currently unreferenced tables (`tab:ll_decomposition`, `tab:sep_sensitivity`, `tab:euler_errors`, `tab:param_recovery`, `tab:mode_sensitivity_chains`).
27. Remove internal plot titles or standardize them with captions.
28. Replace "FOM truth" with "SEP benchmark."
29. Break long artifact paths or move them to replication notes to remove overfull boxes.

**Issue counts in this consolidated triage**: 10 critical, 11 major, 8 minor.
