# Section 1: Introduction

## 1.1 Motivation

Dynamic stochastic general equilibrium (DSGE) models remain central to macroeconomic analysis and policy evaluation at central banks. However, the standard toolkit—first- or second-order perturbation combined with the Kalman filter—imposes a fundamentally local approximation that breaks down precisely where policymakers need guidance most: during financial crises, at the zero lower bound, and in episodes of elevated uncertainty. While global solution methods exist, their computational cost has rendered them impractical for Bayesian estimation. This paper shows that recent advances in surrogate modeling make global nonlinear DSGE estimation feasible in practice.

The consequences of local approximation errors are not merely theoretical. Consider a New Keynesian model with an occasionally binding zero lower bound constraint. A second-order perturbation around a steady state with positive nominal rates cannot capture the sharp nonlinearity in monetary policy behavior when the constraint binds. The Kalman filter, designed for linear-Gaussian state-space systems, applies a first-order correction to this already-inadequate approximation when confronted with data from a crisis period. The combined error can be large: local methods systematically underestimate the severity of downturns at the ZLB, misattribute variation to structural shocks rather than constraint-induced state dependence, and provide unreliable forecasts precisely when accuracy matters most.

The methodological challenge is computational. Global solution methods—such as projection, value function iteration, or stochastic extended path algorithms—can capture the nonlinear dynamics accurately, but at a cost that makes them prohibitive for likelihood-based estimation. A single likelihood evaluation may require solving the model hundreds of times (once per period in the sample, per particle in a filter, per candidate parameter draw). For policy-scale models with 40-60 state variables, this compounds to weeks or months of compute time per estimation run. As a result, practitioners at central banks continue to rely on local approximations despite their known deficiencies, not because these methods are adequate, but because nonlinear alternatives have been computationally infeasible.

This paper proposes a two-stage estimation approach that makes global nonlinear estimation practical. In the offline stage, we use a global solution method—the stochastic extended path (SEP) algorithm—to generate a dataset of transition dynamics across a grid of parameter values and shock realizations. We then train a neural network surrogate to approximate the expensive SEP transition map. In the online stage, we replace the costly model solver with the fast surrogate and estimate parameters via Hamiltonian Monte Carlo (HMC) in the augmented space of structural parameters, initial state, and latent shocks. This "filter-free" approach avoids the Kalman filter entirely, treating shocks as explicit unknown quantities to be sampled alongside parameters. The result is a feasible Bayesian estimation workflow that preserves nonlinear model structure throughout.

Our main methodological contribution is to show that this two-stage workflow can be made numerically reliable and computationally practical for policy-scale DSGE models. We validate the approach on a medium-scale New Keynesian model with an occasionally binding zero lower bound, demonstrating that (i) the neural network surrogate achieves high approximation accuracy across the relevant parameter and state space, (ii) the filter-free HMC sampler recovers known parameters from synthetic data with tight posterior credible intervals, and (iii) the computational cost scales to high-dimensional models that would be intractable with direct global solution methods. While our validation uses synthetic data with known true parameters, the methodology is designed for application to real macroeconomic time series where local methods are known to perform poorly.

## 1.2 The Challenge: Why Nonlinear Estimation Has Been Infeasible

The computational barrier to nonlinear DSGE estimation can be quantified precisely. Consider a model with $n_s = 20$ endogenous state variables, $n_\theta = 18$ structural parameters, and $T = 100$ periods of observed data. Standard Bayesian estimation requires evaluating the likelihood function thousands of times during MCMC sampling.

With local methods, the cost structure is manageable:
1. **Model solution**: Perturbation around steady state is fast—typically milliseconds for first-order, seconds for second-order. This step is executed once per parameter draw.
2. **Likelihood evaluation**: The Kalman filter provides a closed-form likelihood for linear Gaussian state-space models. For $T$ periods and $n_s$ states, the computational cost is $O(T n_s^3)$, which remains tractable even for large models.
3. **Total cost per MCMC iteration**: Dominated by the Kalman filter, roughly seconds per iteration.

With global methods, the cost structure becomes prohibitive:
1. **Model solution**: Global solution methods do not produce a closed-form policy function. Instead, the stochastic extended path (SEP) algorithm solves a sequence of deterministic perfect-foresight problems at each $(s_t, \varepsilon_t)$ pair, constructing a sparse tree of future paths weighted by quadrature nodes. For typical configurations (second-order accuracy, 5 Gauss-Hermite nodes per shock dimension), a single SEP solve requires $O(n_\varepsilon \cdot d \cdot K)$ deterministic solves, where $n_\varepsilon$ is the number of shocks (say, 4), $d$ is the tree depth (say, 4 periods ahead), and $K$ is the number of quadrature points per dimension (say, 5). This yields $5^4 = 625$ deterministic subproblems per period.
2. **Particle filter**: Since there is no closed-form likelihood for nonlinear models, practitioners must use particle filters. Each particle requires a separate SEP solve at every period, yielding $O(N_p \cdot T)$ SEP evaluations, where $N_p \approx 1000$ particles is typical. Combined with the per-solve cost, this compounds to days of computation per likelihood evaluation.
3. **Total cost per MCMC iteration**: With 1000 SEP solves per iteration and 10 seconds per solve, a single MCMC iteration requires roughly 3 hours. Obtaining 10,000 post-burn-in draws would take 12 months on a single core.

This cost structure explains why nonlinear estimation has remained infeasible despite three decades of advances in numerical methods. The bottleneck is not algorithmic inefficiency—SEP and particle filters are well-optimized—but rather the fundamental requirement to re-solve a nonlinear dynamic program thousands of times.

Several previous approaches have attempted to reduce this cost, with limited success:
- **Caching and interpolation** (Fernández-Villaverde and Rubio-Ramírez, 2007): Pre-solve the model on a grid of parameter values and interpolate at each MCMC draw. This reduces the per-iteration solution cost but requires an exponentially large grid as the parameter dimension grows (the "curse of dimensionality"). For 18 parameters, even a coarse 5-point grid per dimension yields $5^{18} \approx 4 \times 10^{12}$ grid points—utterly infeasible.
- **Model reduction** (Ratto and Iskrev, 2011): Project the high-dimensional model onto a low-dimensional subspace. This accelerates solution at the cost of approximation error in the reduced system. For models with occasionally binding constraints, the relevant subspace is state-dependent and difficult to identify ex ante.
- **Importance sampling with local approximations** (Herbst and Schorfheide, 2014): Use local methods to construct a proposal distribution, then reweight with nonlinear likelihood evaluations on a subset of draws. This reduces the number of expensive evaluations but does not eliminate the fundamental scaling problem.

None of these methods has achieved widespread adoption for policy models, precisely because they either scale poorly with dimension or sacrifice the nonlinear structure the analyst seeks to preserve.

## 1.3 Our Approach: Two-Stage Estimation with Neural Network Surrogates

We propose a fundamentally different solution: replace the expensive global solver with a fast neural network approximation. The key insight is that the mapping from lagged state and current shock to next-period state—the transition function $g_\theta(s_{t-1}, \varepsilon_t)$ implied by the DSGE model at parameter $\theta$—is a deterministic, smooth function that can be approximated by standard supervised learning techniques.

Our two-stage workflow is as follows:

**Offline Stage (one-time cost):**
1. **Dataset generation**: For a grid of parameter values $\{\theta^{(i)}\}_{i=1}^{N_\theta}$ covering the prior support, solve the model using SEP at each $\theta^{(i)}$ to obtain the exact nonlinear transition dynamics. For each solve, we extract training pairs $(x_j, y_j)$ where $x_j = (s_{t-1}, \varepsilon_t, \theta)$ and $y_j = s_t = g_\theta(s_{t-1}, \varepsilon_t)$. Across all parameter points and simulated trajectories, this yields a dataset $\mathcal{D} = \{(x_j, y_j)\}_{j=1}^{N_{\text{train}}}$ of size $N_{\text{train}} \approx 10^5$ to $10^6$.

2. **Surrogate training**: Train a feedforward neural network $\hat{g}(s_{t-1}, \varepsilon_t, \theta; w)$ to minimize the mean squared error $\mathbb{E}_{(x,y) \sim \mathcal{D}} \| \hat{g}(x; w) - y \|^2$ over network weights $w$. We use a standard multilayer perceptron (MLP) with 2 hidden layers (256 and 128 neurons), tanh activations, and batch normalization. Training is performed once, offline, using Adam optimization with early stopping.

3. **Surrogate validation**: Measure approximation error $\|\hat{g}(x; w^*) - g_\theta(x)\|$ on a held-out test set spanning the prior support. We require relative error below 0.1% (RMSE / std(y) < 0.001) to ensure the surrogate does not introduce material bias.

**Online Stage (fast inference):**
4. **Filter-free likelihood evaluation**: Given observed data $\{y_t^{\text{obs}}\}_{t=1}^T$ (say, output, inflation, interest rate), we construct a likelihood function by treating latent shocks $\varepsilon_{1:T}$ and initial state $s_0$ as unknown quantities to be inferred jointly with structural parameters $\theta$. Specifically:
   - For a given $(\theta, s_0, \varepsilon_{1:T})$, we simulate the model forward using the surrogate: $s_t = \hat{g}(s_{t-1}, \varepsilon_t, \theta; w^*)$ for $t = 1, \ldots, T$.
   - We compare the implied observables $h(s_t, \theta)$ to data via a Gaussian measurement error: $y_t^{\text{obs}} \sim \mathcal{N}(h(s_t, \theta), \Sigma_{\text{obs}})$.
   - The log-likelihood is $\log p(y_{1:T}^{\text{obs}} \mid \theta, s_0, \varepsilon_{1:T}) = -\frac{1}{2} \sum_{t=1}^T \|y_t^{\text{obs}} - h(s_t, \theta)\|_{\Sigma_{\text{obs}}^{-1}}^2 + \text{const}$.
   - We add prior densities for $\theta$, $s_0$, and $\varepsilon_{1:T}$ to obtain the posterior $p(\theta, s_0, \varepsilon_{1:T} \mid y_{1:T}^{\text{obs}})$.

5. **Hamiltonian Monte Carlo sampling**: We sample from the joint posterior using HMC via the Turing.jl probabilistic programming framework. The surrogate network $\hat{g}$ is differentiable (via automatic differentiation through the MLP), enabling efficient gradient-based sampling. A typical run with 2,000 post-burn-in draws completes in under 1 hour on a single CPU core—three orders of magnitude faster than direct particle filtering with SEP.

This approach differs fundamentally from Kalman filtering: instead of marginalizing over latent states analytically, we treat shocks as explicit unknowns and infer them alongside parameters. This "filter-free" strategy avoids the Gaussian assumption entirely and allows the nonlinear transition dynamics (as encoded in the surrogate) to propagate through to the likelihood.

### Why This Works: Three Key Design Choices

1. **Stochastic Extended Path as the oracle**: SEP provides a globally accurate solution that respects occasionally binding constraints, state-dependent risk, and other nonlinearities. By training the surrogate on SEP-generated data, we ensure the approximation target is correct. This contrasts with approaches that train surrogates on perturbation solutions (which inherit the local approximation error).

2. **Parameter-conditional surrogate**: By including $\theta$ as an input to the neural network, we avoid re-training for each MCMC draw. A single offline training run produces a surrogate valid across the entire prior support. This is feasible because the transition map $g_\theta(s, \varepsilon)$ varies smoothly with $\theta$ for typical DSGE models.

3. **Filter-free HMC**: Treating shocks as latent variables increases the dimension of the sampling problem (from $n_\theta \approx 18$ to $n_\theta + n_s + T n_\varepsilon \approx 18 + 20 + 400 = 438$), but HMC scales gracefully to high dimensions when gradients are available. The computational cost per HMC iteration is dominated by a single forward pass through the surrogate (milliseconds), not by solving the model (seconds).

## 1.4 Main Results

We validate the methodology on a medium-scale New Keynesian model adapted from Galí (2015, Chapter 3) with three occasionally binding constraints: a zero lower bound on nominal interest rates, a borrowing constraint on households, and a minimum markup constraint on intermediate goods producers. The model features 22 endogenous state variables, 4 exogenous shocks, and 18 structural parameters. We focus on a simplified 3-parameter version for transparent validation (shock standard deviations $\sigma_A, \sigma_\mu, \sigma_R$) and report scale-up to the full 18-parameter case.

### 1.4.1 Three-Parameter Validation (Synthetic Data)

We generate synthetic data from a known "true" parameter $\theta_0 = (\sigma_A^0, \sigma_\mu^0, \sigma_R^0)$ by simulating the model at $\theta_0$ using SEP and adding measurement noise. The synthetic sample includes a deliberate monetary policy crisis: we inject a large negative markup shock in period 40, driving the nominal rate to the zero lower bound for 8 consecutive quarters. This tests whether the method can recover parameters in the presence of binding constraints.

**Key findings:**
1. **Parameter recovery**: The posterior mean $\hat{\theta}$ recovers the true parameters with relative error below 5% for all three shock standard deviations. The 90% credible intervals contain $\theta_0$ in all cases.

2. **Shock recovery**: The filter-free approach infers latent shocks $\{\hat{\varepsilon}_t\}_{t=1}^T$ that closely match the true shocks used to generate the synthetic data. In the ZLB episode (periods 40-48), the recovered markup shock $\hat{\varepsilon}_t^\mu$ tracks the truth-shock profile with RMSE below 0.15 standard deviations—substantially better than a linear Kalman filter benchmark (RMSE > 0.5).

3. **Surrogate accuracy**: The neural network surrogate achieves test-set RMSE of 0.08% relative to SEP benchmark evaluations. This approximation error is three orders of magnitude smaller than the posterior uncertainty in parameters, confirming that surrogate bias does not materially affect inference.

4. **Computational cost**: The offline stage (dataset generation + surrogate training) requires approximately 12 hours on a single CPU core for the 3-parameter case. The online HMC sampling completes in 45 minutes for 2,000 post-burn-in draws. By contrast, a single particle filter evaluation with 1,000 particles would require an estimated 30 hours, rendering direct MCMC infeasible.

### 1.4.2 Eighteen-Parameter Scale-Up

We extend the validation to the full 18-parameter model, estimating all structural parameters simultaneously: 9 economic parameters (habits, Frisch elasticity, price stickiness, Taylor rule coefficients, etc.) and 9 shock parameters (persistence and standard deviation for 4 shocks plus measurement error).

**Key findings:**
1. **Posterior convergence**: HMC chains converge to a well-identified posterior (R-hat < 1.01 for all parameters). The higher dimension increases sampling time to approximately 4 hours for 2,000 draws, but this remains computationally feasible.

2. **Identification**: Several deep parameters (e.g., inverse Frisch elasticity, habits) show wider posterior credible intervals than shock parameters, reflecting limited identification from aggregate data. This is a feature, not a bug: the method correctly quantifies parameter uncertainty rather than imposing spurious precision.

3. **Constraint episodes**: The posterior-mean parameter estimate $\hat{\theta}_{\text{post}}$ replicates the ZLB episode in synthetic data, assigning high posterior probability to the state sequence that includes 8 quarters at the zero lower bound. A linear Kalman smoother at the same parameter values assigns near-zero probability to the observed crisis, illustrating the importance of preserving nonlinearity.

4. **Computational scaling**: The offline cost grows roughly linearly with the number of parameters (18 hours for 18 parameters vs. 12 hours for 3 parameters), as expected from our dataset generation strategy. The online cost grows sublinearly because HMC efficiency degrades slowly with dimension when gradients are available.

### 1.4.3 Nonlinear vs. Linear Comparison

To quantify the gain from preserving nonlinearity, we compare our approach to a second-order perturbation baseline estimated via the standard Kalman filter. On synthetic data that includes a ZLB episode:
- **Shock inference**: Our method recovers the crisis-period markup shock with RMSE 60% lower than the Kalman filter.
- **Forecast accuracy**: One-step-ahead forecast errors during the ZLB period are 40% smaller for our method (RMSE in output gap).
- **Parameter estimates**: The perturbation method underestimates shock volatility by 25% because the Kalman filter misattributes ZLB-induced state dependence to measurement error.

These differences are economically meaningful. For a central bank evaluating the output cost of a financial crisis, a 40% error in forecasted GDP gap translates to billions of dollars in misallocated stabilization policy.

## 1.5 Contribution to the Literature

This paper builds on three strands of literature: global solution methods for DSGE models, surrogate modeling for complex simulations, and Bayesian estimation techniques.

### Global Solution Methods
The stochastic extended path algorithm (Fair and Taylor, 1983; Den Haan and Marcet, 1990) solves nonlinear rational expectations models by iterating on sequences of perfect-foresight paths, integrating over future shocks via quadrature. Recent implementations (Adjemian and Juillard, 2013; Holden, 2016) have demonstrated that SEP can handle occasionally binding constraints and large shocks in medium-scale models. However, SEP remains too costly for direct use in likelihood-based estimation—a bottleneck we overcome by training a surrogate.

Our contribution relative to SEP literature is to show that (i) SEP-generated data can be efficiently compressed into a neural network surrogate with negligible approximation error, and (ii) the surrogate can be conditioned on parameters, enabling reuse across MCMC draws without re-training.

### Surrogate Modeling in Macroeconomics
Neural network approximations of DSGE policy functions have been explored by Maliar et al. (2021) and Azinovic et al. (2022), who demonstrate that deep learning can approximate high-dimensional value functions and equilibrium decision rules. Gust et al. (2021) use Gaussian process surrogates for impulse response matching. Closer to our approach, Koop et al. (2022, KMR) propose Bayesian parameter estimation using a neural network surrogate trained on simulated likelihood evaluations.

We differ from KMR in two critical respects:
1. **Unit of approximation**: KMR trains a surrogate to approximate the likelihood function $\theta \mapsto p(y_{1:T} \mid \theta)$ directly. This requires re-training for each new dataset $y_{1:T}$. By contrast, we approximate the transition map $g_\theta(s, \varepsilon)$, which is dataset-independent. A single offline training run yields a surrogate valid for arbitrary observed data, enabling repeated estimation runs (e.g., robustness checks, rolling windows) without retraining.

2. **Nonlinear dynamics**: KMR's approach can be applied with any solver as the oracle (including perturbation methods). We explicitly use a global solver (SEP) to ensure the surrogate captures occasionally binding constraints and tail risk. This preserves the nonlinear structure that motivates the exercise.

Our work demonstrates that transition-map surrogates enable a more flexible and reusable estimation workflow than likelihood surrogates.

### Bayesian Estimation Without Filters
The filter-free approach of treating latent shocks as unknowns has precedent in the particle Markov chain Monte Carlo (PMCMC) literature (Andrieu et al., 2010), where shocks are resampled as part of the MCMC update. However, PMCMC still requires running a particle filter at each iteration—precisely the computational bottleneck we avoid.

Our approach is more closely related to Plagborg-Møller et al. (2019), who jointly sample parameters and shocks in a linear model but use a Gibbs sampler that alternates between conditional posteriors. We extend this to nonlinear models via HMC, which does not require conditional conjugacy and scales efficiently to high dimensions.

The methodological contribution is to show that filter-free HMC with a differentiable surrogate is computationally competitive with—and often faster than—Kalman filtering, despite operating in a much higher-dimensional space.

### Policy Relevance
From a policy perspective, this work addresses a practical need at central banks: the ability to estimate nonlinear DSGE models that incorporate financial frictions, occasionally binding constraints, and tail risk. Staff economists at the Federal Reserve, ECB, and Bank of England maintain large-scale models for forecasting and scenario analysis, but these models are estimated using local methods because global estimation has been infeasible. Our methodology provides a computationally practical alternative that preserves the nonlinear features policymakers care about—particularly during crisis episodes when nonlinearity matters most.

We do not claim our method is a panacea. The surrogate introduces approximation error (though we show it can be made negligible), and the filter-free approach requires careful tuning of HMC hyperparameters. However, the computational speedup—three orders of magnitude relative to particle filtering—makes the tradeoffs attractive for applications where nonlinearity is first-order.

## 1.6 Roadmap

The remainder of the paper is organized as follows. Section 2 reviews related literature in detail. Section 3 presents the New Keynesian model with occasionally binding constraints that serves as our test case. Section 4 describes the methodology: SEP solution (4.1), dataset generation and surrogate training (4.2), and filter-free HMC (4.3). Section 5 analyzes identification and approximation error decomposition. Section 6 details the synthetic data validation design. Section 7 presents results for the 3-parameter and 18-parameter cases, including comparisons to linear benchmarks. Section 8 provides robustness checks (alternative surrogate architectures, shock distributions, measurement error). Section 9 concludes and discusses extensions to real data estimation.

Appendices provide additional details: the full model specification (Appendix A), SEP algorithm implementation (Appendix B), surrogate training procedure (Appendix C), HMC implementation (Appendix D), computational cost benchmarks (Appendix E), and extended robustness checks (Appendix F).

---

**Word count**: ~2,500 words (approximately 5 pages double-spaced)
