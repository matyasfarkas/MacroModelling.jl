# Pre-Submission Referee Report

**Paper**: Structural Bias from Linearization in DSGE Estimation
**Author**: Mátyás Farkas (IMF)
**Date**: 2026-05-02
**Review Standard**: **Econometrica**

---

## Overall Assessment

The paper has been polished through multiple revision cycles and now compiles cleanly at 123 pages with extensive appendices, formal consistency theory, and a substantial empirical apparatus. Its principal strength is the deterministic equation-block decomposition showing investment adjustment costs at 68.9% of the FOM-ROM1 gap with the price Phillips curve at 0.2% — that is a genuinely original and clean result. The single most critical issue blocking submission is the gap between what the paper claims (a "structural bias from linearization" generic to DSGE estimation) and what its evidence actually supports (a single-model, single-country, single-mode comparison whose 37-nat headline depends on warm-start initialization to a posterior region inaccessible to cold-start chains, has a comparable 34-nat alternative explanation in the omitted ARMA(1,1) markup terms, and whose surrogate is admitted to extrapolate during the COVID periods that drive the empirical gain). Beneath that umbrella issue sit ~150 specific items, most of them fixable in 2–4 weeks, but several requiring new compute (direct SEP-HMC posterior comparison, linear+gate ablation, pruned-ROM2 benchmark) that takes longer.

**Preliminary Recommendation**: **Substantial revision required for Econometrica; alternatively, retarget to Journal of Econometrics, RED, or AEJ:Macro** where the existing evidence base is sufficient. Agent 6 (the Econometrica AE) recommends desk-reject-for-fit on the current draft and explicitly advises retargeting JoE/JAE/RED.

---

## 1. Spelling, Grammar & Style

### Critical (numerical/factual issues that must be fixed)

1. **Line 619 vs. Table 5 (line 644)**: σ_b reported as "0.091" in body, "0.092" in table.
2. **Line 619 vs. Table 5 (line 639)**: ρ_w reported as "0.236" in body, "0.230" in table.
3. **Line 617**: "monetary persistence doubles (ρ_ms: 0.18 to 0.38)" — actually 2.11×, "more than doubles".
4. **Line 658 vs. line 619**: pooled LL reported both as −1,781 (text) and −1,782 (note). Reconcile.
5. **Line 1582 / 186 / 607**: "seven orders of magnitude below the exact Kalman value" — actual ratio 21.2×10⁶ / 1,818 ≈ 1.16×10⁴ ≈ four OOM, not seven. (Cross-corroborated by Agent 2: line 1606 says "three orders of magnitude" for the same comparison — internal contradiction.)
6. **Line 706 / 710**: Kimball curvature alternates `\varepsilon_p` and `\epsilon_p` in same paragraph.
7. **Line 1383**: "too many periods … unnecessarily increases" — subject-verb error (should be "increase").
8. **Line 708 / 2523**: verb agreement issues with `\citet{richter2016zlb} find` and `\citet{farkas2020bayesian} estimate` — verify single vs. multi-author.

### Major style patterns to fix throughout

- **`vs.\\` vs `versus`** — pick one (Econometrica style: "versus" in prose, "vs." only in tables/captions).
- **`Gal\'i` vs `Gal\'{i}`** — standardize on `Gal\'i`.
- **Number-grouping**: paper mixes `22,080` and `22{,}080`; standardize to `{,}` LaTeX thin-space throughout.
- **`labor` vs `labour`** — table at line 2314 and footnote at line 2328 use British spelling; rest is American. Standardize.
- **`Generalized` vs `Generalised`** — line 1512 British; rest American.
- **Notation collisions** (also flagged by Agent 4): `\varepsilon` (structural shock vs demand elasticity vs Kimball curvature); `\beta` (discount factor vs Beta-distribution shape); `\eta_t` (observation noise vs wage-markup innovation); `s_t` (state vector vs gate score vs markup shock).
- **`N\times` vs `N`-fold vs `N` times** — inconsistent in prose (lines 411, 439, 460, 617, 1506).
- **`completely`** appears at 178/222/691 — Econometrica style would delete or replace.
- **Em-dash**: line 945 has Unicode `—` rather than `---`.

### Minor (~110 items, full list in agent output)

Includes long sentences (lines 339, 1582, 1664, 1689, 1700), table caption capitalization, "directionally consistent" overuse, decimal-place inconsistencies, "approximately" overuse, and several minor grammar issues. Most fixable in a single editing pass.

---

## 2. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. **§8.2 narrative (lines 619–646) vs Table 5**: Five distinct numerical mismatches between body text and the tab:extended_sample_posterior values:
   - σ_a: text says 0.716→0.634; table 0.716→0.636
   - σ_b: text 0.179→0.091; table 0.179→0.092
   - σ_qs: text 0.524→0.423; table 0.524→0.408
   - ρ_b: text 0.731→0.901; table 0.731→0.899
   - ρ_w: text 0.160→0.236; table 0.160→0.230
   
   The table's Δ values check out internally; the prose appears to be the wrong source. **MUST FIX** — these are headline empirical numbers.

2. **OOM contradiction lines 1584/1606**: "seven orders of magnitude" vs "three orders of magnitude" for the same bootstrap-PF / COPF comparison. Three is correct.

3. **"Seven orders of magnitude below exact"** (lines 186, 607, 1582): see Agent 1 #5.

4. **Spurious cross-reference at line 409**: claims Table tab:per_obs_r2 contains "autocorrelation 0.81 and mean duration 12.5 quarters" — Table tab:per_obs_r2 only has per-observable R² values; the cited statistics do not appear anywhere in the paper.

5. **Caption-vs-text tension at fig:decomposition_vs_scale (line 482 vs 486)**: caption says "across full shock-scale frontier"; text says "during ZLB-binding episodes". Disambiguate.

### Cross-Reference Errors

- **Line 529 (10% gate, T=184) vs Line 1684 (15.1% gate, T=265)**: paper does not explicitly explain the transition; reader has to infer.
- All 54 in-text citation keys appear in `.bib` — no dangling cites detected.
- `gust2012effects` bib key has `year = 2017` (the AER publication); `fernandezvillaverde2021estimating` has `year = 2015`. Misleading keys but resolve correctly.
- `childers2022filter` has `arXiv:2201.12345` — placeholder-looking ID; verify.

### Terminology Drift

- **S''(1) value** reported as 6.06, 6.01, 6.0, 6.0144 across four locations.
- **`z`** notation reused for capital utilization (line 869), OOD distance (line 1208), scaled shocks (line 2525).
- **"Investment block share"** reported as 69%, 68.9%, 89–92%, 89–95% — each refers to a different decomposition; paper is loose with the umbrella term.
- **"Gate share"** — 10% (calibration), 15.1% (realized), 25% (3-param validation).

### Sample sizes consistent (T=184, T=244, T=265, T_fore=21, T=200) — no confusion. Likelihood reporting (37 nats, 35.91 nats, 18 nats) is consistent across reportings.

---

## 3. Unsupported Claims & Identification Integrity

### Causal Overclaiming

1. **Line 174**: "the investment and capital block accounts for 69 percent" coupled with "generates the nonlinear impulse" (line 407) — the mechanism is asserted as fact for what is a deterministic L2 share.
2. **Line 178**: "removing S(·) causes the gap to explode" — within-model counterfactual; defensible but ceteris paribus is uncheckable.
3. **Line 184**: "the economy may be driven more by endogenous propagation through the investment channel" — "driven" is causal language for a likelihood-ranking + posterior-shift result.
4. **Line 405**: "The mechanism involves three interacting nonlinear functions" — proposed mechanism stated as fact; the paper does not isolate the contribution of S, p^k, ξ-wedge separately.
5. **Line 534**: "$\sigma_b$ absorbs the missing amplification" — stated as fact.
6. **Line 689**: "no linear model can rationalize" — full SW07 with ARMA terms might rationalize differently.
7. **Line 736**: "the economy is driven more by endogenous propagation" — even with the conditional, "the economy is driven" is causal.
8. **Line 745**: ESS (sampler efficiency) conflated with statistical identification.

### Bias-vs-Fit Framing (the central credibility issue)

The title is "Structural Bias from Linearization in DSGE Estimation." A bias claim requires the linear posterior to diverge from truth, not merely from the surrogate. The substantive evidence supports a fit improvement (37-nat LL gap) that is interpreted as bias under the assumption that the surrogate's posterior is closer to truth. Suggested fix: title becomes "Investment Adjustment Costs and the Nonlinear Posterior of Smets–Wouters" or insert "Apparent Structural Bias" with explicit caveat.

### Missing Caveats (HIGH IMPACT)

1. **ARMA(1,1) markup confound buried in Appendix L**: restoring standard ARMA delivers 34-nat improvement — comparable to 37-nat surrogate gain. This means the linear-vs-surrogate comparison is not robust to specification of the markup process. The paper notes this in Appendix L but does not foreground in main text.
2. **Single posterior mode caveat**: Bayes factor of 10^16 is conditional on the surrogate's primary mode; multi-modal mixing across linear and surrogate modes would change this substantially. Caveat is stated once but not at headline-claim points.
3. **Linear chain bimodality**: Two linear chains find modes 34 nats apart. Posterior shifts in Tables 4-6 use one of two distinct linear modes; alternative mode would change ρ_w from 0.75 to 0.39 in linear column.
4. **Single country / sample**: No caveat about US-specific results in abstract, intro, or conclusion.
5. **Surrogate vs true nonlinear posterior**: Bayes factor is linear-vs-surrogate, not linear-vs-true-FOM.
6. **OOD behavior on COVID**: Section 7.1 acknowledges training data may not cover extreme states; OOS forecast results in Appendix M leverage exactly this regime.
7. **Gate identification (40 of 265 periods)**: paper does not address whether 15% gate-on classification represents identifying information or overfitting.
8. **Frequentist coverage rate condition**: theorem claims `δ_T = o(T^{-1/2})` satisfied "by a factor of 20×" using held-out RRMSE = 0.0012 as proxy for sup-norm bound — held-out RRMSE is average error, not supremum. Worst-case RRMSE on stressed periods is 0.12, which exceeds the threshold.
9. **csadjcost sensitivity scope**: shows decomposition is robust at varying S''(1); does not re-estimate posterior at non-baseline values. Currently presented as if estimation were re-run.
10. **14-replication MC coverage shortfall (79.8%)**: stated as "consistent with short chains" but no power calculation justifies this.

### Generalization Issues

- **Title** generalizes from one model + one sample to "DSGE Estimation" as category.
- **Line 184**: "the opposite of current practice" — empirically uncited.
- **Line 736**: "models" (plural) but only one model tested.
- **Line 708**: "regardless of specification" claim is broader than three pricing variants tested.

---

## 4. Mathematics, Equations & Notation

### Verification of Past Fixes (all confirmed in place)

✅ `n_y` removed from `eq:ll_bias_total` — confirmed at line 1839.
✅ Cor 7.8 uses `O(B_T/√T)` TV-based coverage bound — confirmed line 1890.
✅ `B_T` scales like `RRMSE²·√T` — confirmed line 1895; arithmetic checks `T·RRMSE² · √T → √T·RRMSE²`.
✅ Numerical bound 0.82 pp present and arithmetically correct (line 1937).
✅ Numerical bias bound `B_T ≤ 0.02 nats` present at line 1880; arithmetic checks.

### Mathematical Errors (require attention)

1. **Prop 7.7 proof** uses informal "AM-GM applied to bias-relevant component" with `‖r_t‖ = O(1)` "noise floor". This is loose. The correct argument is the orthogonality `E[η_t' Σ_y^{-1} e_t] = 0` (independence of measurement error and surrogate error). Replace the AM-GM step with the orthogonality argument.

2. **Theorem 7.6 Step 1** conflates pathwise and expected uniform bounds. Per-period error is `O(δ_T)` pathwise, `O(δ_T²)` only in expectation. Either say "uniform in expectation" in eq:uniform_ll_approx or make the BvM argument via Le Cam's third lemma.

3. **Cor 7.8 proof score-gradient claim** requires undeclared smoothness assumption: `‖∂(g_θ - ĝ_θ)/∂θ‖_∞ = O(δ_T)`. Add as assumption or cite.

4. **Rate condition mismatch (Remark 7.9)**: deterministic uniform bound is linear in δ_T, requiring `δ_T = o(T^{-1})`. Currently `T·δ_T ≈ 0.8` — not o(1). The bound on expected bias `T·δ_T² ≈ 0.0024` does work. Clarify which interpretation is meant.

5. **Assumption 7.4 mixing condition** in footnote — promote to formal assumption text.

### Notation Collisions (should be resolved before submission)

- `g`: transition map vs gate indicator (most serious — visually identical in eq. 346)
- `s_t`: state vector vs gate score vs markup shock
- `κ`: Phillips curve slope vs Fisher information condition number
- `R_t`: gross interest rate vs SEP residual
- `\mathcal{S}`: shock index set vs visited state-shock domain
- `\varepsilon`: structural shock vs demand elasticity vs Kimball curvature

### Equation Numbering

Unused labels (consider removing): `eq:newton_iteration`, `eq:labor_supply_full`, `eq:nkpc_full`, `eq:logit_jacobian`, `eq:inversion_objective`.

`σ_min` (informal preview eq:ll_bias_bound) vs `λ_min` (formal Prop 7.7) — equal for SPD but switching notation between sections is jarring.

### LaTeX Math Formatting

- `\text{logsumexp}` should be `\operatorname{logsumexp}`.
- `\DeclareMathOperator{\Var}{\text{Var}}` is doubly-wrapped; use `\DeclareMathOperator{\Var}{Var}`.
- Line 708 uses `E_t` instead of `\E_t`.
- No `\$<\$1` Julia-interpolation artifacts found.

---

## 5. Tables, Figures & Documentation

### Critical Visual/Caption Defects (must fix)

1. **`fig_sep_vs_rom1_eqs_investment.pdf`** — empty 6th panel ([0,1]×[0,1] blank axes). Remove or fill.
2. **`covid_shock_comparison.pdf`** — TFP panel header reads "TFP (ε⊠)" with broken Unicode character; should be `ε_a`. Risk-premium panel y-axis peaks at ~140σ but caption claims 56σ — reconcile.
3. **`fig_rom1_forecast_errors.pdf`** — caption says sample 1959Q1–2004Q4 but x-axis runs to ~2024.
4. **`gap_magnitude_vs_shock_scale.pdf`** — caption describes per-equation-block decomposition; figure shows a single aggregate gap line with ZLB binding %. Rewrite caption or replace figure.
5. **`gali_block_decomposition.pdf`** — caption says "equation-block, absolute magnitudes"; figure shows observable shares. Rewrite caption.
6. **`decomposition_vs_shock_scale.pdf`** — caption says scales 0.1–1.5; figure x-axis shows 0.8–1.5 only. Reconcile.

### Tables with Data-Quality Concerns

- **`tab:rhat_linear` (line 2030) and `tab:rhat_surrogate`** — bulk and tail ESS columns are identical for every parameter. Likely pipeline bug; investigate.
- **`tab:18param_full` (line 2393)** — "Complete 18-Parameter Validation Results" reports a parameter set unrelated to the 18-parameter SW07-HLT estimation; footnote disclaims this but TOC does not. Rename the subsection or remove.

### Tables Missing Notes Elements

- `tab:18param_switching_posterior` and `tab:extended_sample_posterior` — Kalman 90% CI omitted; only RS interval shown.
- `tab:equation_block` — units of "Mean ‖Δ‖₂" not stated.
- `tab:crisis_vs_normal` — variance share "—" entries unexplained.
- `tab:benchmarks` — `^*` gloss misleading; 100-core scaling assumption unstated.
- `tab:ll_decomposition` — gate calibration policy at each cell unstated.
- `tab:euler_errors` — "eq. 20" reference to row not interpretable.
- `tab:hyperparameter_grid` — 18 dropped configurations unmotivated.
- `tab:prior_specs` — applied-to scope unstated.
- `tab:prior_sensitivity` — "Diffuse"/"Tight" SD multipliers unstated.
- `tab:oos_forecast` — boldface gloss missing.

### Figures Missing Notes Elements

- All IRF figures — y-axis units (% vs fraction) inconsistent.
- `fig:zlb_correction_4panel` — axis units missing on all four panels.
- `fig:gali_obc_vs_no_obc` — y-axis units missing.

### Cross-Reference: All 34 tables and 12 figures are referenced at least once; no orphan elements.

### Formatting Inconsistencies

- booktabs (`\toprule`/`\midrule`) vs plain `\hline` mixed across tables — standardize.
- Caption capitalization: most Title Case, a few sentence-case.
- B/W-distinguishability: some color-only legends in `decomposition_vs_shock_scale` and `gali_block_decomposition`.

---

## 6. Contribution & Referee Assessment (Econometrica AE)

### Part 1 — Central Contribution

One-sentence summary: investment adjustment cost is "invisible" to first-order perturbation because S(·)=S'(·)=0 at steady state, accounts for ~69% of FOM-ROM1 gap in SW07-HLT, and a residual NN surrogate + filter-free HMC pipeline enables nonlinear estimation. **The "invisibility" observation is a one-line corollary of Taylor's theorem (proof is three lines).** What is novel is the *quantitative* claim that this matters at 69%, but the paper does not benchmark against pruned-ROM2, which Aruoba-FV-RR (2006) and Andreasen et al. (2018) show captures ~80% of accuracy gains. The methodological piece (FiLM + residual + gate) is engineering-incremental over Kase et al. (2025) and Naubert (2025). The formal consistency theorem recasts Kleijn–van der Vaart misspecified BvM with a sieve interpretation; nothing in the proofs is not in Chen (2007), Kleijn–van der Vaart (2012), or Müller (2013).

**Rating: Incremental.**

The empirical decomposition (69% / 0.2%) is striking and original; everything else is recasting and engineering.

### Part 2 — Identification and Credibility (the most damning section)

1. **Surrogate posterior never directly compared to SEP-based posterior.** The closest evidence (twenty-draw Section 8.4) is a point-prediction comparison at draws sitting at the surrogate mode. The BvM corollary requires uniform RRMSE convergence over the visited state space; the paper trains at shock scale 0.1 with a supplementary set at 0.4, while the COVID episode produces 56σ shocks — well outside even the supplementary distribution. The OOD detector flags only 0.4% of periods.

2. **Multimodality acknowledged but not resolved.** Linear chains 34 nats apart; cold-start surrogate chains 12,700 nats below warm-started chains. The "warm-start from best chain's posterior mean" protocol is post-hoc data mining on the likelihood surface.

3. **Euler errors flag a problem the consistency theorem doesn't address.** Surrogate's median normalized residual 1.7×10⁻³ is *worse* than ROM1's 4.7×10⁻⁴ — yet the consistency theorem requires uniform state-level convergence.

4. **The 81/19 Thore decomposition is a single-mode statement.** Without showing the linear MLE used is the global MLE, the 19% parameter relocation contribution is poorly defined.

5. **Linear+gate ablation never run.** Cannot separate the 37-nat improvement into "from the gate's regime allowance" vs "from the nonlinear correction itself".

6. **OOS forecast worse on labor (3.90×) and policy rate (6.79×) on the full window.** The 16% COVID-window gain averages across observables that exclude the variables most relevant to the ZLB story.

7. **SEP `accept_tol = 0.35`** — coverage bound RRMSE = 0.0012 is measured against this loose target.

### Part 3 — Required and Suggested Analyses

**Required (blockers for Econometrica):**

1. **Direct SEP-HMC posterior comparison on the 3-parameter Galí synthetic.** Round-trip: simulate from SEP, run SEP-HMC and surrogate-HMC, compare moments and coverage.
2. **Monte Carlo at 18-parameter scale on nonlinear-DGP data** (Appendix J currently tests only linear-on-linear).
3. **Replication on a model where SEP solution is fully feasible without an NN** (small RBC or 3-eq NK with adjustment costs solved by sparse-grid projection).
4. **Linear+gate ablation** to separate gate-allowance from nonlinear-correction.
5. **Quiet-sample OOS** (e.g., 1995–2007) to test whether the gate over-fits in-sample.

**Suggested:**

1. Pruned ROM2 benchmark in equation-block decomposition.
2. Subsample stability of 69% across 1959-79, 1980-99, 2000-25.
3. Drop ε_p from headline (pooled ESS = 90 means chain is essentially sampling prior).
4. Decomposition by gated vs ungated periods.
5. FiLM removal ablation.

### Part 4 — Literature Positioning

Missing/under-cited: **Andreasen, Fernández-Villaverde, Rubio-Ramírez (2018)** as natural pruned-ROM2 benchmark; **Justiniano, Primiceri, Tambalotti (2010, 2011)** on investment-specific shocks; **Christiano, Motto, Rostagno (2014)** on financial frictions through investment; **Kollmann (2017)** on tractable approximations near ZLB; **Bocola (2016)** on sovereign default + nonlinear estimation; **Den Haan and de Wind (2012)** on nonlinear stable perturbation.

Distinguishing from competitors (Kase et al. 2025; Naubert 2025; Childers et al. 2022) is one-sentence each — should be a paragraph each with comparison table.

**Title oversells.** Paper's own evidence shows σ_b shift direction reverses across samples; "structural bias" implies generic property of model class but only one model tested.

### Part 5 — Journal Fit and Recommendation

**Econometrica fit: weak.** The methodological contribution is engineering-grade rather than theoretically deep. The empirical contribution is striking but single-mode, single-model, single-country, with comparable ARMA-confound alternative explanation.

**Preliminary recommendation: Substantial revision required if author insists on Econometrica; recommended retarget to Journal of Econometrics, RED, or AEJ:Macro** where the existing evidence base is sufficient and the contribution lands as solidly in-scope rather than as a methodological stretch.

Concrete steps to reach Econometrica:
1. Direct SEP-HMC on Galí synthetic.
2. Resolve multimodality with tempered SMC.
3. Add second model + second country.
4. Develop genuinely new theoretical result (minimax rate, identification result, or non-trivial BvM extension).
5. Run pruned-ROM2 as natural benchmark.
6. Split into two papers — decomposition (tight, beautiful) and methodology (technical).

### Part 6 — Pointed Referee Questions

1. The BvM corollary requires `δ_T = o(T^{-1/2})` uniformly over visited states. Section 7.1 trains at shock scale 0.1 (supplementary 0.4); COVID produces 56σ shocks under linear filter. On what grounds does the formal coverage bound apply on exactly the periods that drive the 37-nat headline?
2. Table H.2 reports surrogate's median normalized Euler residual at 1.7×10⁻³, larger than ROM1's 4.7×10⁻⁴. How is this consistent with Assumption F.4's uniform-approximation requirement?
3. The 81/19 Thore decomposition is path-symmetric averaged. Report each ordering separately and show robustness to the linear MLE used. Two linear chains find modes 34 nats apart — which is θ_lin, and does 81% survive against the other?
4. The OOS forecast shows labor RMSE ratio 3.90, interest-rate ratio 6.79 on the full window; post-COVID aggregate ratio 2.48 in favor of ROM1. Why should the reader accept the COVID-window 16% gain as evidence the nonlinear model is preferred?
5. Three of 18 linear-baseline parameters have R̂ > 1.5 and ESS ≈ 29. How sensitive is the 81/19 decomposition and the 37-nat headline to using the global MLE instead of these unconverged means?
6. The 22,080 training samples are at SEP `accept_tol = 0.35`. What is surrogate RRMSE measured against `τ_acc = 0.01` SEP solutions, and does the 0.82 pp coverage bound survive when target is tightened by 35×?
7. Two linear chains 34 nats apart; cold-start surrogate 12,700 nats below warm-start. The fix — "warm-start from best chain's posterior mean" — is post-hoc data mining. Provide either (a) tempered SMC confirming warm-start region is global, or (b) honest reporting of headline sensitivity to mode selection.

---

## Priority Action Items

The list below is ordered by priority using the triage hierarchy: identification/credibility (Agent 3, Agent 6 Part 2) > required analyses (Agent 6 Part 3) > internal inconsistencies (Agent 2) > tables/figures (Agent 5) > math errors (Agent 4) > style (Agent 1).

### CRITICAL (block submission to Econometrica; some block submission to any top outlet)

1. **Foreground the ARMA(1,1) markup confound in §1, §7, and conclusion.** Currently buried in Appendix L; the 34-nat alternative explanation is comparable to the 37-nat headline. Without this surfaced, the headline is misleading.
2. **Address the multimodality and warm-start initialization concern.** Either (a) run tempered SMC to confirm warm-start region is global, or (b) explicitly report headline sensitivity to mode selection (37-nat, 56σ, 81/19 — all conditional on mode).
3. **Fix the §8.2 numerical inconsistencies (5 entries: σ_a, σ_b, σ_qs, ρ_b, ρ_w)** between body text (lines 619–621) and Table 5.
4. **Fix "seven orders of magnitude" claim** in §1 / §7.2 / Appendix G — actual ratio is 4 OOM. Internal contradiction with line 1606 ("three orders of magnitude").
5. **Reconcile causal language in §1 introduction and §7.4 policy implications** — multiple "drives", "generates", "the economy is driven" instances need hedging or evidence.
6. **Caption-figure mismatches (3 figures): `gap_magnitude_vs_shock_scale.pdf`, `gali_block_decomposition.pdf`, `decomposition_vs_shock_scale.pdf`.** Each caption describes a different quantity than the figure shows.
7. **Title and bias-vs-fit framing.** Decide: keep title and add explicit "under the assumption that surrogate is closer to truth" caveat, or change title to something like "Investment Adjustment Costs and the Nonlinear Posterior of Smets–Wouters".
8. **Identical bulk/tail ESS columns in `tab:rhat_linear` and `tab:rhat_surrogate`** — investigate pipeline bug.
9. **Direct SEP-HMC posterior comparison on 3-parameter Galí synthetic (required Econometrica analysis).** Without this, the consistency theorem has no empirical anchor.
10. **Linear+gate ablation (required Econometrica analysis).** Cannot separate gate-allowance from nonlinear-correction effects.

### MAJOR (will likely be raised by referees; fixable in 2–4 weeks)

11. **Spurious cross-reference at line 409** to `tab:per_obs_r2` (statistics not in table).
12. **Notation collisions (`g`, `s_t`, `κ`, `R_t`, `\mathcal{S}`, `\varepsilon`)** — at least clean up `s_t` (state vs gate score vs markup shock) and `\varepsilon` (shock vs elasticity vs Kimball curvature).
13. **Mathematical proof tightness**: replace AM-GM hand-wave in Prop 7.7 with explicit orthogonality argument; clarify Theorem 7.6 Step 1 expectation vs pathwise distinction; promote Assumption 7.4 mixing condition from footnote.
14. **Empty 6th panel in `fig_sep_vs_rom1_eqs_investment.pdf`** and **broken TFP symbol** in `covid_shock_comparison.pdf`. Reconcile 56σ vs 140σ in the same figure.
15. **Date-range mismatch in `fig_rom1_forecast_errors.pdf`** caption (says 1959–2004, figure runs to 2024).
16. **Pruned-ROM2 benchmark** in equation-block decomposition.
17. **Drop or hedge ε_p findings** (pooled ESS = 90 means chain is essentially sampling prior; not a confirmed posterior shift).
18. **`tab:18param_full` (Table 30)** is a different parameter set than headline 18-param SW07-HLT — strengthen disclaimer or remove.
19. **Single-mode caveat** at every headline-claim point (currently stated once).
20. **Single country/sample caveat** in abstract, intro, conclusion.
21. **Quiet-sample OOS** (e.g., 1995–2007) — required Econometrica analysis.
22. **Subsample stability of 69%** across 1959-79, 1980-99, 2000-25.
23. **Add Kalman 90% CI in `tab:18param_switching_posterior` and `tab:extended_sample_posterior`** for parity.
24. **Standardize y-axis units across all IRF figures** (% vs fraction).
25. **Add missing citations** (Andreasen-FV-RR 2018, Justiniano-Primiceri-Tambalotti, Christiano-Motto-Rostagno 2014, Kollmann 2017, Den Haan-de Wind 2012).

### MINOR (polish)

26. **Standardize**: `vs.\` vs `versus`; `Gal\'i`; `22{,}080` thin-space; `labor` (American); `Generalized` (American); `N times` vs `N\times`.
27. **Long sentences** (lines 339, 1582, 1664, 1689, 1700) — split.
28. **Subject-verb errors** at lines 1383, 708, 2523.
29. **Equation labels never referenced** (5 labels) — remove or reference.
30. **LaTeX math formatting**: `\operatorname{logsumexp}`; remove double `\text{}` wrap on `\Var`/`\Cov`; `\E_t` consistency on line 708.
31. **Booktabs/hline standardization** across all tables.
32. **B/W-distinguishability** in `decomposition_vs_shock_scale.pdf` and `gali_block_decomposition.pdf` — add markers/line styles in addition to color.
33. **`tab:hyperparameter_grid`** — explain dropped 18 configurations.
34. **`tab:prior_sensitivity`** — give SD multipliers for "Diffuse"/"Tight".
35. **Decimal-place consistency** in posterior tables.

---

**Total issues catalogued**: 11 critical, 15 major, 35+ minor. ~150 specific items across all six agent reports.

**Compute requirements for Critical fixes**: 9 of 11 are paper-only edits (a few weekends). Items 9 (SEP-HMC comparison) and 10 (linear+gate ablation) require new compute (estimated 2–4 weeks each). The author's existing infrastructure (`scripts/run_linear_hmc_advancedhmc.jl`, `scripts/run_surrogate_hmc_advancedhmc.jl`) already supports both with parameter changes; the bottleneck is wall-clock runtime not script development.
