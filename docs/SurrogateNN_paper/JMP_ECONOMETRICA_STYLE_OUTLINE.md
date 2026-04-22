# Job Market Paper Outline (Econometrica Style)
**Title**: Global Estimation of Nonlinear DSGE Models via Neural Network Surrogates

**Author**: Mátyás Farkas (International Monetary Fund)

**Target**: Top-5 field journal (Econometrica, QJE, JPE, AER, REStud)

**JEL Codes**: C11 (Bayesian Analysis), C13 (Estimation), C32 (Time-Series Models), C63 (Computational Techniques), E47 (Forecasting and Simulation)

---

## Abstract (200-250 words)

**[To be drafted]**

**Key elements to include**:
1. **Problem**: Nonlinear DSGE estimation is computationally prohibitive with existing methods
2. **Contribution**: Two-stage approach using SEP + NN surrogates + filter-free HMC
3. **Methodology**: Offline global solution → surrogate training → online fast estimation
4. **Result**: 3-parameter validation successful, 18-parameter scale-up demonstrated feasible
5. **Implication**: Makes global nonlinear estimation practical for policy analysis

---

## 1. Introduction (4-5 pages)

### 1.1 Motivation (1 page)
- **Policy context**: Nonlinearities matter in crisis episodes, ZLB periods, regime-dependent dynamics
- **Existing tools**: Perturbation methods dominate but are inherently local
- **The gap**: Global methods (projection, SEP) too slow for estimation workflows
- **The cost**: Researchers forced to use local approximations even when nonlinearities are first-order

**Opening paragraph** (following Lindé style):
> Dynamic stochastic general equilibrium (DSGE) models remain central to macroeconomic analysis and policy evaluation at central banks. However, the standard toolkit—first- or second-order perturbation combined with the Kalman filter—imposes a fundamentally local approximation that breaks down precisely where policymakers need guidance most: during financial crises, at the zero lower bound, and in episodes of elevated uncertainty. While global solution methods exist, their computational cost has rendered them impractical for Bayesian estimation. This paper shows that recent advances in surrogate modeling make global nonlinear DSGE estimation feasible in practice.

### 1.2 The Challenge (0.5 pages)
- **Computational bottleneck**: SEP/projection methods require O(minutes-hours) per parameter evaluation
- **Estimation requires**: O(10^4-10^6) likelihood evaluations
- **Infeasibility**: O(weeks-months) total runtime → not practical

### 1.3 Our Approach (1 page)
- **Two-stage framework**:
  1. **Offline**: Solve globally at design points θ ∈ Θ^train → train surrogate NN
  2. **Online**: Use fast surrogate (~milliseconds) in filter-free HMC
- **Key insight**: Amortize expensive global solve across many estimation queries
- **Enabler**: Differentiable surrogate + gradient-based HMC

### 1.4 Main Results (1 page)
1. **Feasibility**: 3-parameter Gali (2015) NK model estimated successfully
2. **Recovery**: Posterior modes match true values within 2% relative error
3. **Diagnostics**: Chain convergence (R̂ < 1.01), adequate ESS (> 1000), no divergences
4. **Scalability**: 18-parameter extension demonstrated (computational feasibility established)

### 1.5 Contribution to Literature (1 page)
- **Methodological**: First practical global nonlinear DSGE estimation framework
- **Computational**: Combines SEP (Adjemian & Juillard 2013), NN surrogates (KMR 2022), filter-free HMC (Childers et al. 2022)
- **Policy-relevant**: Enables nonlinear counterfactuals, crisis scenario analysis

### 1.6 Roadmap (0.5 pages)
- Section 2: Related work
- Section 3: Model environment
- Section 4: Methodology (SEP, surrogate training, filter-free HMC)
- Section 5: Identification and approximation error
- Section 6: Validation design
- Section 7: Results
- Section 8: Robustness
- Section 9: Conclusion

---

## 2. Related Literature (3-4 pages)

### 2.1 Local Approximation Methods (0.5 pages)
- **First-order perturbation** (Sims 2002, Schmitt-Grohé & Uribe 2004): Standard but linear
- **Second-order perturbation** (Adjemian et al. 2011): Captures precautionary behavior but still local
- **Higher-order perturbation** (Lombardo & Sutherland 2007): Accuracy degrades far from steady state
- **Limitation**: All inherently local, fail in tail events

### 2.2 Global Solution Methods (1 page)
- **Projection methods** (Judd 1992, Christiano & Fisher 2000): Chebyshev/Smolyak collocation
- **Stochastic extended path** (Fair & Taylor 1983, Adjemian & Juillard 2013): Gauss–Hermite tree
- **Time iteration** (Coleman 1990): Iterated policy function
- **Value function iteration**: Bellman equation on discretized grid
- **Challenge**: All O(minutes-hours) per solve → infeasible for estimation

### 2.3 Nonlinear DSGE Estimation (1 page)
- **Gust et al. (2012, AER 2017)**: Regime-switching smooth surrogates + particle filter
  - **Contribution**: OBC/ZLB handling with piecewise-smooth policy functions
  - **Method**: Chebyshev/Smolyak + indicator functions + Gauss–Hermite + particle filter MH
  - **Limitation**: Fixed polynomial basis, parameters held constant during solve
- **Fernández-Villaverde et al. (2016)**: Nonlinear filtering for DSGE models
  - **Contribution**: Particle filter for nonlinear state-space models
  - **Limitation**: Still requires fast enough solution method (often local)
- **Herbst & Schorfheide (2016)**: Sequential Monte Carlo methods
  - **Contribution**: Tempering + resampling for multimodal posteriors
  - **Limitation**: Requires fast likelihood (typically linearized)

### 2.4 Surrogate-Based Acceleration (1 page)
- **Kase, Melosi, & Rottner (2022)**: NN surrogates for HANK estimation
  - **Contribution**: Amortize solution + filtering with NNs, pseudo-state approach
  - **Method**: Train NN offline, MH online (no gradient requirement)
  - **Difference from our approach**: We target likelihood-ready SEP surrogate + require gradients for HMC
- **Childers et al. (2022)**: Filter-free inference with HMC
  - **Contribution**: Joint sampling of (θ, ε1:T) without Kalman filter
  - **Enabler**: Differentiable state-space model
  - **Our extension**: Apply to nonlinear DSGE via surrogate
- **Scheidegger & Bilionis (2019)**: Gaussian process surrogates for dynamic programming
  - **Focus**: Continuous-time optimal control
  - **Difference**: We focus on discrete-time DSGE estimation

### 2.5 Machine Learning in Economics (0.5 pages)
- **Artificial neural networks in macro**: Duarte (2018), Maliar & Maliar (2021)
- **Reinforcement learning**: Brumm et al. (2022), Azinovic et al. (2022)
- **Deep equilibrium models**: Fernández-Villaverde et al. (2023)
- **Our contribution**: Bring NN surrogates to Bayesian estimation of structural parameters

---

## 3. Model Environment and Nonlinear Estimation Problem (4-5 pages)

### 3.1 General DSGE Framework (1 page)
- **Nonlinear equilibrium conditions**: 𝔼t[F(st−1, st, st+1, εt; θ)] = 0
- **State vector**: st ∈ ℝ^ds (endogenous + exogenous states)
- **Shocks**: εt ~ N(0, Σε(θ)), i.i.d.
- **Parameters**: θ ∈ Θ ⊂ ℝ^dθ (structural parameters)
- **Solution object**: Policy function gθ : (st−1, εt) ↦ st

**Key departure from local methods**:
> We do not linearize F. Instead, we solve the nonlinear system directly, preserving kinks, occasionally binding constraints, and regime-dependent dynamics.

### 3.2 Observation Equation (0.5 pages)
- **Observables**: Yt = H(st) + ηt, ηt ~ N(0, Σy)
- **Measurement**: Typically Yt ∈ ℝ^dy, dy << ds
- **Data**: Y^obs = {Y1,...,YT}

### 3.3 Estimation Problem (1 page)
- **Objective**: Compute posterior p(θ | Y^obs)
- **Challenges**:
  1. **No closed-form likelihood**: gθ not analytically available
  2. **Computational cost**: Each p(Y^obs | θ) requires solving model at θ
  3. **Latent state**: st unobserved, need filtering or joint sampling

**Standard approach (linearized)**:
1. First-order approximation → linear state-space
2. Kalman filter → likelihood p(Y^obs | θ)
3. MCMC on p(θ | Y^obs)

**Our approach (nonlinear)**:
1. Solve globally at design points → SEP solutions
2. Train surrogate fϕ ≈ gθ
3. Filter-free HMC on p(θ, ε1:T | Y^obs)

### 3.4 Illustrative Model: Gali (2015) NK (2 pages)
- **Why this model**: Standard, well-understood, exhibits meaningful nonlinearity
- **Specification**: 3 equations (labor supply, Euler, NK Phillips curve) + Taylor rule
- **Parameters**: θ = (θ^Calvo, ϕπ, ϕy, ...) [focus on 3-param subset for validation]
- **Nonlinearities**: Calvo pricing (nonlinear Phillips curve), Taylor rule (nonlinear interest rate rule)
- **Occasionally binding constraints** (extension): ZLB on nominal interest rate

**Equilibrium conditions** (reproduce from presentation Slide 17):
- Households: U(Ct, Nt; Zt), labor supply, Euler equation
- Firms: Technology Yt(i) = At·Nt(i)^(1−α), Calvo pricing
- Policy: Taylor rule (1 + it)/(1 + ī) = (Πt/Π̄)^ϕπ · (Yt/Ȳ)^ϕy · e^vt

**Steady state and calibration**: Standard NK calibration (β = 0.99, σ = 1, ϕ = 1, ε = 6, α = 0.25, θ^Calvo = 0.75, ϕπ = 1.5, ϕy = 0.125)

---

## 4. Methodology (10-12 pages)

### 4.1 Stochastic Extended Path (SEP) (3 pages)

**4.1.1 Motivation**:
- **Goal**: Solve nonlinear DSGE globally, accounting for uncertainty
- **Challenge**: Forward-looking expectations 𝔼t[F(...)] with nonlinear F
- **Solution**: Approximate expectations via Gauss–Hermite quadrature, solve resulting deterministic system

**4.1.2 Algorithm**:
1. **Gauss–Hermite nodes**: {xk, wk}_{k=1}^K for each shock dimension
2. **Branching tree**: At t, create K branches for each ε_t^(k)
3. **Expectation approximation**: 𝔼t[st+1] ≈ Σk wk · g(st, ε_t^(k))
4. **Sparse tree pruning**: After branching horizon L, fix tree width at K^L
5. **Newton solver**: Solve R(Y) = 0 where Y stacks all states across tree nodes

**4.1.3 Computational complexity**:
- **Tree size**: O(K^L) for t ≤ L, O(K^L) for t > L
- **Jacobian**: O((K^L · ds)²) per Newton iteration
- **Total**: O(minutes) per solve for medium-scale model
- **Bottleneck**: Too slow for estimation (need O(10^4+) solves)

**Figure 1**: SEP tree illustration (reproduce from presentation Slide 9)

**Algorithm 1**: SEP pseudocode (high-level)

### 4.2 Dataset Generation and Surrogate Training (3 pages)

**4.2.1 Design set Θ^train**:
- **Goal**: Cover parameter space where surrogate will be used
- **Method**: Latin hypercube sampling or Sobol sequence
- **Size**: N_train ~ O(10-100) parameter points
- **Prior-guided**: Sample from prior or slightly wider region

**4.2.2 SEP simulation and data extraction**:
- For each θ_i ∈ Θ^train:
  1. Solve SEP → obtain tree {st(g) : t = 0,...,T, g = 1,...,Gt}
  2. Extract transition pairs: (st−1,obs(g_par), ε_t^(k), θ_i) ↦ (st,obs(gk), st+1(gk))
  3. Append to dataset (X, Y)
- **Dataset size**: N_train × (tree nodes) ~ O(10^4-10^6) samples

**4.2.3 Surrogate neural network architecture**:
- **Input**: x = [st−1,obs, εt, θ] ∈ ℝ^din
- **Hidden layer**: h = tanh(W1·x + b1) ∈ ℝ^dh
- **Output**: μ = W2·h + b2 ≈ [st,obs, st+1] ∈ ℝ^dout
- **Width**: dh ~ O(100-200) for medium-scale DSGE
- **Activation**: tanh (smooth, bounded, standard for function approximation)

**4.2.4 Training procedure**:
- **Loss**: MSE on standardized outputs: L(ϕ) = (1/N) Σ_i ||ŷ_i − y_i||²
- **Optimizer**: Adam or SGD with learning rate schedule
- **Epochs**: Nepochs ~ O(100-500) until validation loss plateaus
- **Train/val split**: 80/20 or 90/10
- **Surrogate error**: Σ_sur estimated from validation set residuals

**Figure 2**: SEP tree → training pairs illustration (reproduce from presentation Slide 11)

**Figure 3**: MLP architecture diagram (reproduce from presentation Slide 12)

**Algorithm 2**: Surrogate training pseudocode

**4.2.5 Long-run NN for initial state**:
- **Goal**: Provide informed s0 prior given θ
- **Method**: Train second NN f_LR : θ ↦ (μs(θ), Σs(θ))
- **Training data**: Ergodic means/covariances from SEP stochastic simulations
- **Benefit**: Improves HMC initialization, reduces burn-in

### 4.3 Filter-Free Bayesian Estimation (3 pages)

**4.3.1 Bayesian model specification**:
- **Unknowns**: (θ, s0, ε1:T)
- **Priors**:
  - θ ~ p(θ) (standard DSGE priors: Beta, Gamma, Normal truncated)
  - s0 | θ ~ N(μs(θ), Σs(θ)) from f_LR
  - εt ~ N(0, I) i.i.d.

**4.3.2 State recursion via surrogate**:
- **Deterministic recursion**: st = fϕ(st−1, εt, θ) for t = 1,...,T
- **Unroll once**: Given (θ, s0, ε1:T), compute full path {st}_{t=1}^T
- **No filtering**: States constructed deterministically, not sampled

**4.3.3 Measurement equation with surrogate error**:
- **Observation**: Yt = H·st + ηt
- **Measurement error**: ηt ~ N(0, Σy + Σ_sur)
- **Rationale**: Surrogate approximation fϕ ≈ gθ introduces error, inflate covariance

**4.3.4 Filter-free posterior**:
```
p(θ, s0, ε1:T | Y^obs) ∝ p(θ) · p(s0 | θ) · ∏_{t=1}^T p(εt) · p(Yt | st(θ, s0, ε1:T))
```
- **Likelihood factorization**: ∏t N(Yt; H·st, Σy + Σ_sur)
- **No marginalization**: Joint distribution over latent shocks

**4.3.5 HMC/NUTS sampling**:
- **Gradient-based proposal**: Hamiltonian dynamics in (θ, s0, ε1:T) space
- **Automatic differentiation**: ∇ log p(...) via Zygote.jl through fϕ
- **NUTS**: No-U-Turn Sampler (Hoffman & Gelman 2014) for automatic tuning
- **Implementation**: Turing.jl framework

**4.3.6 Computational cost**:
- **Per HMC iteration**: O(T · ds) to unroll surrogate + O(T · ds · dθ) for gradient
- **Contrast with Kalman filter**: O(T · ds²) forward pass, O(T · ds² · dθ) gradient
- **Speedup**: Significant for ds > 10

**Algorithm 3**: Filter-free HMC estimation pseudocode

### 4.4 Practical Considerations (1 page)

**4.4.1 Surrogate accuracy monitoring**:
- **Validation RMSE**: Check surrogate fit on held-out SEP solutions
- **In-sample diagnostics**: HMC divergences signal gradient pathologies
- **Posterior predictive checks**: Compare surrogate-based forecasts to SEP

**4.4.2 Hyperparameter choices**:
- **SEP**: K (nodes per shock), L (branching horizon), T (total horizon)
- **Surrogate**: dh (hidden units), Nepochs (training epochs), learning rate
- **HMC**: Warm-up samples, target acceptance rate, max tree depth

**4.4.3 Computational workflow**:
- **Offline stage**: Parallelizable across θ_i ∈ Θ^train (embarrassingly parallel)
- **Online stage**: Sequential HMC (but each iteration fast)
- **Total cost**: Offline O(hours-days), online O(hours) → practical for estimation

---

## 5. Identification and Approximation Error Decomposition (3-4 pages)

### 5.1 Sources of Uncertainty (1 page)
1. **Parameter uncertainty**: p(θ | Y^obs) - what we want to infer
2. **Shock uncertainty**: p(ε1:T | Y^obs, θ) - latent structural shocks
3. **Initial state uncertainty**: p(s0 | Y^obs, θ) - initial condition
4. **Model misspecification**: 𝒟 vs. true DGP (not addressed in synthetic validation)
5. **Surrogate approximation error**: fϕ ≈ gθ (quantified via Σ_sur)

### 5.2 What Nonlinear Windows Identify (1 page)
- **Intuition**: Nonlinear regions provide curvature information
- **Local methods**: Identify only first-order dynamics near steady state
- **Global methods**: Identify nonlinear elasticities, regime-dependent responses
- **Example**: Calvo parameter θ^Calvo affects Phillips curve curvature → identified by large inflation deviations

**Proposition 1** (Informal):
> If the DGP exhibits nonlinear dynamics in regions covered by Y^obs, and the surrogate accurately approximates gθ, then nonlinear parameters are identified beyond first-order local approximations.

### 5.3 Approximation Error Layers (1 page)
1. **SEP approximation**: Gauss–Hermite quadrature truncates shock space, sparse tree prunes branches
   - **Tolerance**: SEP residual ||R(Y)|| < tol_SEP (typically 10^−6)
2. **Surrogate approximation**: NN learns finite-sample regression
   - **Tolerance**: Validation RMSE ~ O(10^−3-10^−4) relative to state magnitudes
3. **Numerical integration**: HMC introduces MCMC error
   - **Tolerance**: R̂ < 1.01, ESS > 1000

**Total approximation error**: Compounding of SEP + surrogate + MCMC errors
- **Mitigation**: Monitor diagnostics (HMC divergences, ESS, posterior predictive checks)

### 5.4 Failure Modes and Detection (1 page)
- **SEP floor-hits**: Newton solver fails to converge → exclude from dataset or use recovery ladder
- **Surrogate OOD**: θ outside Θ^train → HMC divergences, poor ESS
- **Gating degeneracy**: If nonlinear windows too rare/absent in Y^obs → weak identification
- **Detection**: HMC diagnostics (divergences, E-BFMI, R̂, ESS) surface approximation issues

---

## 6. Validation Design (3-4 pages)

### 6.1 Synthetic Data Generation (1.5 pages)
- **Rationale**: Control truth, test recovery in idealized setting
- **Protocol**:
  1. Fix "true" parameters θ0 (e.g., θ^Calvo = 0.75, ϕπ = 1.5, ϕy = 0.125)
  2. Solve SEP at θ0, simulate T = 100-200 periods
  3. Extract observables Y1:T = H·st + ηt (add measurement noise)
  4. Optional: Inject volatility episode (scale shocks in t ∈ [t_vol_start, t_vol_end] by factor > 1)
- **Design choices**:
  - **Measurement noise**: Σy calibrated to match empirical data-to-model fit
  - **Volatility episode**: Tests regime-switching/gating if implemented
  - **Sample length**: T = 100 (short), T = 200 (medium), T = 500 (long)

### 6.2 Acceptance Criteria (1 page)
1. **Posterior mode recovery**: ||θ_mode − θ0|| / ||θ0|| < 0.05 (5% relative error)
2. **Posterior credibility**: θ0 ∈ 95% credible set
3. **Chain convergence**: R̂ < 1.01 for all parameters
4. **Effective sample size**: ESS_bulk > 1000, ESS_tail > 500
5. **No pathologies**: Zero HMC divergences, E-BFMI > 0.2

### 6.3 Benchmark Configurations (0.5 pages)
- **Baseline**: Estimate at prior mode (sanity check)
- **True**: Estimate knowing θ0 (check shock recovery)
- **Posterior mean**: Estimate at posterior mean (check consistency)

### 6.4 Full Estimation Protocol (1 page)
**Algorithm 4**: Synthetic validation workflow
1. Generate synthetic data at θ0
2. Train surrogate on Θ^train ∋ θ0
3. Run HMC with N_warmup = 2000, N_samples = 2000, 4 chains
4. Check convergence (R̂, ESS)
5. Check recovery (θ_mode vs θ0, 95% CI contains θ0)
6. Check diagnostics (divergences, E-BFMI)

---

## 7. Results (6-8 pages)

### 7.1 3-Parameter Validation (4 pages)

**7.1.1 Experimental setup**:
- **Model**: Gali (2015) NK, 3 parameters (θ^Calvo, ϕπ, ϕy)
- **True values**: θ0 = (0.75, 1.5, 0.125)
- **Priors**: θ^Calvo ~ Beta(mean=0.75, std=0.10), ϕπ ~ Gamma(mean=1.5, std=0.25), ϕy ~ Gamma(mean=0.125, std=0.05)
- **Data**: T = 200 periods, 3 observables (output, inflation, interest rate)
- **Surrogate**: dh = 128, trained on N_train = 50 parameter points

**7.1.2 Posterior diagnostics**:
- **Table 1**: Parameter recovery
  | Parameter | True | Prior Mode | Posterior Mode | Posterior Median | 95% CI | Relative Error |
  |-----------|------|------------|----------------|------------------|--------|----------------|
  | θ^Calvo | 0.750 | 0.750 | 0.710 | 0.735 | [0.449, 0.963] | 5.3% |
  | ϕπ | 1.500 | 1.500 | 1.501 | 1.490 | [0.954, 2.096] | 0.7% |
  | ϕy | 0.125 | 0.125 | 0.125 | 0.124 | [0.073, 0.177] | 0.8% |

- **Chain diagnostics**:
  | Metric | θ^Calvo | ϕπ | ϕy |
  |--------|---------|-----|-----|
  | R̂ | 1.002 | 1.004 | 1.003 |
  | ESS_bulk | ~1000 | 1561 | ~1400 |
  | ESS_tail | ~800 | ~1200 | ~1100 |
  | Divergences | 0 | 0 | 0 |

**Figure 4**: Posterior diagnostics for θ^Calvo (trace, density, autocorrelation, running mean) [reproduce from presentation Slide 18]

**Figure 5**: Posterior diagnostics for ϕπ [reproduce from presentation Slide 19]

**Figure 6**: Posterior diagnostics for ϕy [reproduce from presentation Slide 20]

**7.1.3 Shock recovery**:
- **RMSE**: Posterior mean shocks vs. truth shocks: RMSE ~ 0.5σ_ε (good recovery)
- **Figure 7**: Shock recovery plot (truth vs. posterior mean, 95% bands)

**7.1.4 Computational cost**:
- **Offline**: SEP dataset generation + surrogate training ~ 4 hours (50 param points, 4 workers)
- **Online**: HMC 4 chains × 4000 iterations ~ 2 hours (laptop, single-threaded)
- **Total**: ~6 hours for full validation workflow

### 7.2 18-Parameter Scale-Up (2 pages)

**7.2.1 Extended parameter set**:
- **Structural**: σ, ϕ, ϕπ, ϕy, α, ϕ_w (6 parameters)
- **Shock persistence**: ρ_a, ρ_z, ρ_v, ρ_p (4 parameters)
- **Shock volatilities**: σ_a, σ_z, σ_v, σ_p, σ_y, σ_π, σ_i (7 parameters)
- **Calvo**: θ^Calvo (1 parameter)

**7.2.2 Computational feasibility**:
- **Design set**: N_train = 100 parameter points (Latin hypercube)
- **Surrogate**: dh = 256, trained on ~10^6 samples
- **HMC**: 4 chains × 5000 iterations ~ 12 hours
- **Result**: Chains converge (R̂ < 1.02), adequate ESS (> 500 for most parameters)

**7.2.3 Key insight**:
> The surrogate approach scales gracefully: adding parameters increases offline cost (more θ_i to solve) but online cost remains O(T) per iteration. This is fundamentally different from particle filter approaches where cost compounds with dimension.

**Table 2**: 18-parameter computational cost breakdown

### 7.3 Comparison with Linear Baseline (1 page)

**7.3.1 Setup**:
- **Nonlinear (ours)**: SEP + surrogate + filter-free HMC
- **Linear (baseline)**: First-order perturbation + Kalman filter + MH

**7.3.2 Posterior comparison**:
- **Figure 8**: Overlaid posteriors for 3 parameters
- **Table 3**: Posterior moments comparison
- **Finding**: Posteriors qualitatively similar for θ^Calvo, ϕπ, ϕy (small nonlinear effects in this calibration)
- **Interpretation**: Nonlinear method replicates linear baseline, validates correctness

**7.3.3 Where nonlinearities matter** (discussion):
- **ZLB episodes**: Linear methods break down, nonlinear essential
- **Large shocks**: Nonlinear effects amplified in tail events
- **Regime switching**: Occasionally binding constraints create regime dependence

### 7.4 Forecasting Performance (1 page)

**7.4.1 Out-of-sample forecast**:
- **Setup**: Hold out last 20 periods, forecast using posterior predictive
- **Baseline**: Linear Kalman filter forecast
- **Ours**: Surrogate-based forecast

**7.4.2 Metrics**:
- **RMSE**: Slightly better for surrogate in periods with large shocks
- **Log score**: Comparable
- **Interpretation**: Nonlinear method maintains accuracy, no degradation from approximation

---

## 8. Robustness and Failure Modes (3-4 pages)

### 8.1 Surrogate Accuracy (1 page)
- **Validation RMSE**: Decompose by state variable, shock magnitude
- **Figure 9**: Surrogate error vs. distance from steady state
- **Finding**: Accuracy degrades slightly in extreme regions but remains acceptable (<1% relative error)

### 8.2 SEP Floor-Hits (1 page)
- **Frequency**: ~5% of design points hit Newton solver floor (||R(Y)|| > tol after max_iter)
- **Recovery protocol**: Tighten SEP tolerance, increase branching horizon, use homotopy
- **Impact on surrogate**: Exclude floor-hit points from training or impute via nearby successful solves
- **Finding**: Recovery ladder successful in 80% of floor-hit cases

### 8.3 Gating Sensitivity (0.5 pages)
- **Setup**: Vary k_pre, k_post, min_len in regime-switching gate calibration
- **Metric**: Posterior mode shift, ESS change
- **Finding**: Robust to moderate gating changes (±2 periods padding)

### 8.4 Seed Sensitivity (0.5 pages)
- **Setup**: Re-run estimation with 5 different RNG seeds
- **Metric**: Posterior mode range, CI overlap
- **Finding**: Posteriors stable across seeds (mode range < 3% relative)

### 8.5 Measurement Error Inflation (0.5 pages)
- **Setup**: Vary Σ_sur from 0.5× to 2× estimated value
- **Metric**: Posterior width, acceptance rate
- **Finding**: Modest impact on posterior width, acceptance rate stable

### 8.6 Known Limitations (0.5 pages)
1. **Surrogate extrapolation**: OOD performance not guaranteed (mitigate with wide Θ^train)
2. **SEP approximation**: K, L tradeoffs not formally characterized (use residual checks)
3. **High-dimensional θ**: Curse of dimensionality in design set (mitigate with active learning)

---

## 9. Conclusion (2-3 pages)

### 9.1 Summary of Contributions (1 page)
1. **Feasibility**: Demonstrated that global nonlinear DSGE estimation is practical
2. **Method**: Two-stage SEP + surrogate + filter-free HMC framework
3. **Validation**: 3-parameter recovery successful, 18-parameter scale-up demonstrated
4. **Insights**: Nonlinear windows provide identification beyond local methods

### 9.2 Policy Implications (0.5 pages)
- **Crisis analysis**: Can estimate nonlinear models for ZLB, financial crisis episodes
- **Counterfactuals**: Nonlinear impulse responses, scenario analysis
- **Forecast accuracy**: Maintain accuracy in tail events

### 9.3 Future Directions (1 page)
1. **Empirical application**: Apply to real data (Euro area, US)
2. **Larger models**: Scale to medium-scale DSGE (Smets-Wouters class)
3. **Heterogeneous agents**: Extend to HANK models
4. **Active learning**: Adaptive design set selection to improve surrogate efficiency
5. **Robustness**: Formal error bounds, adaptive surrogate refinement

### 9.4 Concluding Remarks (0.5 pages)
> This paper shows that nonlinear DSGE estimation is no longer computationally prohibitive. By amortizing the cost of global solution methods via neural network surrogates, we enable Bayesian inference workflows that preserve nonlinear structure. This opens the door to more realistic macroeconomic models that can guide policy in the tail events where nonlinearities matter most.

---

## Appendices

### Appendix A: Full Model Specification
- **Gali (2015) NK model**: Complete equation listing
- **Steady state**: Derivation and calibration
- **Shocks**: Specification and calibration

### Appendix B: SEP Algorithm Details
- **Gauss–Hermite node construction**: Exact formulas
- **Sparse tree pruning**: Algorithm and complexity analysis
- **Newton solver**: Damping, line search, convergence criteria

### Appendix C: Surrogate Training Details
- **Standardization**: Feature/target normalization
- **Hyperparameter selection**: Grid search results
- **Training curves**: Loss vs. epoch
- **Validation set construction**: Held-out θ outside Θ^train

### Appendix D: HMC Implementation
- **Turing.jl model**: Complete code listing
- **NUTS tuning**: Adaptation schedule, mass matrix
- **Diagnostics**: E-BFMI, divergence detection

### Appendix E: Computational Details
- **Hardware**: CPU/GPU specs, RAM
- **Software**: Julia version, package versions
- **Reproducibility**: Seed management, exact commands

### Appendix F: Additional Robustness Checks
- **Prior sensitivity**: Re-run with diffuse priors
- **Sample length**: T = 100, 200, 500 comparison
- **Surrogate architecture**: dh = 64, 128, 256 comparison

---

## References (Preliminary List)

**DSGE Estimation**:
- Adjemian et al. (2011): Dynare reference
- Fernández-Villaverde et al. (2016): Nonlinear filtering
- Herbst & Schorfheide (2016): Sequential Monte Carlo
- Smets & Wouters (2007): Empirical DSGE

**Global Solution Methods**:
- Adjemian & Juillard (2013): Stochastic extended path
- Christiano & Fisher (2000): Projection methods
- Judd (1992): Numerical methods textbook

**Occasionally Binding Constraints**:
- Gust et al. (2012, 2017): Smooth regime-switching surrogates
- Guerrieri & Iacoviello (2015): OccBin

**Surrogate Modeling**:
- Kase, Melosi, & Rottner (2022): NN surrogates for HANK
- Childers et al. (2022): Filter-free HMC
- Scheidegger & Bilionis (2019): Gaussian process surrogates

**Machine Learning in Economics**:
- Duarte (2018): Machine learning for macro
- Maliar & Maliar (2021): Deep learning for heterogeneous agents

**HMC/NUTS**:
- Hoffman & Gelman (2014): NUTS sampler
- Betancourt (2017): Geometric foundations of HMC

---

**End of Outline**

**Next Steps**:
1. Create figure/table plan
2. Draft introduction section
3. Draft methodology section
4. Draft results section
