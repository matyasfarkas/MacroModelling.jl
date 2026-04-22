# Presentation Claims Extracted
**Source**: Farkas - Global Estimation of Nonlinear DSGE Models with NN Surrogates.pdf
**Presentation Date**: December 2025, IMF MCM Monetary Modelling Unit
**Extractor**: Claude Code (Sonnet 4.5)

---

## Slide-by-Slide Extraction

### Slide 1: Title & Context
**Key Claim**: Global estimation of nonlinear DSGE models using neural network surrogates
**Affiliation**: International Monetary Fund
**Target Audience**: IMF MCM Monetary Modelling Unit (technical, policy-relevant)

---

### Slide 2: Motivation - Analogue Lookup Tables

**Visual**: Logarithmic slide rule + Standard normal table

**Key Claim**:
- Surrogate NNs are modern "lookup tables" for computationally expensive functions
- Analogy: Just as slide rules approximated log/exp, NNs approximate SEP transition maps
- Analogy: Just as normal tables approximated Φ⁻¹(p), NNs approximate gθ(st−1, εt)

**Speaking Point (Inferred)**:
> "Before computers, we used physical lookup tables for hard functions. Today, we use neural networks as learned lookup tables for expensive DSGE solvers."

**Paper Mapping**:
- Introduction: Motivation paragraph
- Related work: Historical context (projection methods, lookup table approximation)

---

### Slide 3: Overview

**Problem Statement**:
1. Solve a fully nonlinear DSGE model
2. Keep nonlinear structure (not log-linearization)
3. Account for uncertainty
4. Estimate structural parameters θ

**Solution (4-Step Pipeline)**:
1. **Solve** using stochastic extended path (SEP)
2. **Generate** simulated dataset from nonlinear global solution
3. **Train** surrogate neural network to approximate SEP transition map
4. **Estimate** using surrogate with filter-free Hamiltonian Monte Carlo

**Key Claim**:
- Two-stage approach: offline (SEP + NN training) + online (fast HMC estimation)
- Filter-free inference: jointly sample shocks and parameters, no Kalman filter

**Paper Mapping**:
- Introduction: Research question
- Section 2: Methodology overview (4-step pipeline)

---

### Slide 4: Notation and Objects

**Mathematical Setup**:
- State vector: st ∈ ℝ^ds (endogenous states: Ct, Nt, Πt, it, vt, At, Zt)
- Observables: Yt = Hst + ηt, ηt ~ N(0, Σy)
- Parameters: θ ∈ Θ (deep DSGE parameters)
- Shocks: εt ~ N(0, I) i.i.d.

**True Transition Map** (from SEP):
- gθ : (st−1, εt) ↦ st

**Surrogate Neural Net**:
- fϕ(st−1, εt, θ) ≈ gθ(st−1, εt)
- Fully differentiable in st−1, εt, and θ

**Key Claim**:
- Surrogate is **parameter-conditioned**: single NN works across θ ∈ Θ^train
- Differentiability enables gradient-based HMC/NUTS

**Paper Mapping**:
- Section 2: Notation subsection
- Section 3: Model environment

---

### Slide 5: Offline Stage - SEP Design and Surrogate Training

**Algorithm (Pseudocode)**:
```
Input: Θ^train, SEP solver, NN architecture fϕ

1. Global SEP solutions and dataset
   For each θ ∈ Θ^train:
     - Solve DSGE nonlinearly with SEP → gθ
     - Simulate paths, collect training pairs: (st−1, εt, θ) ↦ st = gθ(st−1, εt)

2. Train surrogate transition NN
   - Train fϕ on SEP dataset to approximate gθ
   - Estimate surrogate error covariance Σ_sur from out-of-sample residuals

3. Train long-run NN for initial state
   - Train f_LR mapping θ ↦ (μs(θ), Σs(θ))
   - Approximates stationary distribution of st under SEP

4. Fix synthetic "true" data
   - Choose benchmark θ0
   - Run SEP at θ0, extract observables Y1:T for pilot estimation
```

**Key Claims**:
- **Design set Θ^train**: Latin hypercube or Sobol grid over parameter space
- **SEP as global solver**: Handles nonlinearities, occasionally binding constraints
- **Surrogate error Σ_sur**: Quantified from out-of-sample fit, added to measurement equation
- **Long-run NN f_LR**: Enables informed initial state priors in HMC

**Paper Mapping**:
- Section 4: Offline stage methodology
- Algorithm 1: SEP dataset generation
- Algorithm 2: Surrogate training

---

### Slide 6: Online Stage - Filter-Free HMC

**Algorithm (Pseudocode)**:
```
Unknowns: θ, s0, ε1:T

1. State recursion via surrogate
   Given (θ, s0, ε1:T), construct path recursively:
   st = fϕ(st−1, εt, θ) ≈ gθ(st−1, εt), t = 1,...,T

2. Measurement model
   Yt | st ~ N(Hst, Σy + Σ_sur)

3. HMC/NUTS updates
   For j = 1 to n_sim:
     - Propose new (θ', s0', ε'1:T) via Hamiltonian dynamics
     - Compute log p(θ, s0, ε1:T | Y1:T)
     - Accept/reject with MH step
     - Store (θ^(j), s0^(j), ε1:T^(j))
```

**Key Claims**:
- **Filter-free**: Shocks ε1:T are latent unknowns, no particle/Kalman filter
- **Joint posterior**: p(θ, s0, ε1:T | Y1:T) sampled directly
- **Measurement error inflation**: Σy + Σ_sur accounts for surrogate approximation error
- **HMC advantages**: Gradient-based, efficient exploration, automatic tuning (NUTS)

**Paper Mapping**:
- Section 5: Online stage methodology
- Algorithm 3: Filter-free HMC estimation

---

### Slide 7: Comparison - Gust et al (2012)

**Context**:
- Gust et al. (2012, AER 2017): Medium-scale DSGE with occasionally binding ZLB
- Method: Regime-switching smooth surrogates + particle filter

**Gust et al. Approach**:
- Approximate regime-specific expectation functions Vl,j with Chebyshev/Smolyak polynomials
- Piecewise-linear in shocks, smooth in states
- Smolyak collocation grid in state space
- Gauss–Hermite quadrature (3 nodes/shock, 5 shocks → 243 nodes)
- Fixed-point iteration, highly parallelized
- Estimation: Particle filter + MH MCMC

**NN Surrogate Framework**:
- Smolyak grid over states **and parameters**
- Generate dataset via SEP solver (ZLB handled in simulation)
- Approximate global mapping Φ(st−1, εt; θ) with NN
- Training: supervised regression on SEP-generated data
- Estimation: gradient-based HMC, filter-free

**Key Differences**:
| Aspect | Gust et al. | NN Surrogate |
|--------|-------------|--------------|
| Function class | Fixed low-order polynomial | Learned flexible NN |
| Kink handling | Explicit regime smoothing | Data-driven kink learning |
| Domain | Fix θ during solve | Globalize jointly over (x, θ) |
| Estimation | Particle filter + repeated solves | One-time surrogate + filter-free HMC |

**Speaking Point (Inferred)**:
> "Gust et al. use Christiano–Fisher Chebyshev PEA with regime-specific smoothing. We use the same grid technology but replace the fixed polynomial basis with a learned NN surrogate, and replace collocation residuals with supervised learning on SEP simulations."

**Paper Mapping**:
- Section 6: Related work (Gust et al. comparison)
- Emphasize: Methodological similarities (grid-based), differences (NN vs polynomial, filter-free vs particle)

---

### Slide 8: Comparison - Kase–Melosi–Rottner (2022)

**Context**:
- KMR (2022): Accelerate nonlinear DSGE estimation (incl. HANK) by amortizing solution/likelihood with NNs

**KMR Approach**:
- Train NN(s) offline, fast evaluation online
- Parameters as "pseudo-states": θ directly in NN input
- Policy/equilibrium approximation: learn (states, θ) ↦ equilibrium objects
- Likelihood acceleration: NN particle filter idea
- Inference: MH-style sampling (no requirement for smooth gradients everywhere)

**NN Surrogate Framework (SEP + grids + HMC)**:
- Smolyak grid over states + parameters, Gauss–Hermite nodes for shocks
- Generate dataset via SEP (constraints handled in simulation)
- Surrogate targets conditional nonlinear mapping / likelihood-relevant objects
- Inference: HMC/NUTS (requires stable gradients)

**Key Differences**:
| Aspect | KMR | NN Surrogate (SEP) |
|--------|-----|-------------------|
| Primary surrogate target | Amortize solution and/or filter | Likelihood-ready surrogate from SEP |
| Gradient requirements | Levels accuracy sufficient | Levels + local geometry (gradients/curvature) |
| Error control | Implicit (no direct diagnostic) | Observable via HMC divergences/ESS |
| Scale/verification | Smaller benchmarks → HANK | Structured grids + SEP, medium-scale DSGE |
| "Global" comes from | Training over (s, θ) clouds | Structured grids + SEP + smooth surrogate |

**Speaking Point (Inferred)**:
> "KMR amortize solution outputs; we build a likelihood-ready surrogate from SEP. KMR work with level accuracy; we need gradients for HMC, which gives us direct diagnostics (divergences, ESS). SEP + structured grids provide systematic coverage."

**Paper Mapping**:
- Section 6: Related work (KMR comparison)
- Emphasize: Complementary approaches, different inference mechanics

---

### Slide 9: Stochastic Extended Path I

**Visual**: Branching tree diagram

**Key Concept**: SEP approximates expectations via Gauss–Hermite quadrature in shock space

**Mathematical Setup**:
- DSGE model: 𝔼t[F(st−1, st, st+1, εt; θ)] = 0
- Shocks: εt ~ N(0, Σε)
- Expectation approximation:
  ```
  𝔼t[st+1] ≈ Σ_{k=1}^K wk · gθ(st, ε_t^(k))
  ```
- Gauss–Hermite nodes: {xk, wk}_{k=1}^K

**Branching Structure**:
- At t: current state st
- At t+1: K branches st+1|ε_t^(k) for k=1,...,K
- At t+2,...: Future states s_t+...^(k)

**Key Claim**:
- SEP is a **global** method: no Taylor expansion, handles kinks/constraints
- Expectations-consistent: Solves nonlinear Euler equations with stochastic uncertainty

**Paper Mapping**:
- Section 3: SEP methodology subsection
- Figure 1: SEP tree illustration

---

### Slide 10: Stochastic Extended Path II

**Algorithm Components**:

**1. Gauss–Hermite quadrature in shock space**:
- K nodes {xk, wk}_{k=1}^K
- 𝔼[f(ε)] ≈ Σ_{k=1}^K wk · f(ε^(k))
- ε^(k) obtained by scaling xk with Σε^(1/2)

**2. Branching tree over time**:
- Horizon t = 0,...,T, branching horizon L
- For t ≤ L: each node spawns K children → Gt = K^t branches
- For t > L: width fixed at K^L, no further branching
- Node indexing: st(g), g = 1,...,Gt

**3. Nonlinear system and Newton iteration**:
- For t ≤ L, node (t,g) residual:
  ```
  rt,g = Σ_{k=1}^K wk · F(st−1(g_par), st(g), st+1(gk), ε_t^(k); θ)
  ```
- For t > L: single-node residual without expectation
- Stack all st(g) into Y, residuals rt,g into R(Y)
- Solve R(Y) = 0 with damped Newton using AD-based Jacobian J(Y)

**Key Claim**:
- **Sparse tree pruning**: After branching horizon L, tree width fixed to avoid exponential growth
- **Newton solver**: Fast convergence via automatic differentiation (ForwardDiff.jl)

**Paper Mapping**:
- Section 3: SEP algorithm details
- Algorithm 1: SEP pseudocode

---

### Slide 11: Dataset for Neural Network Surrogate

**SEP Output**:
- For each θ and SEP branch: gθ : (st−1, εt) ↦ (st,obs, st+1)
- Solution is the policy function
- For likelihood: take subvector st,obs ∈ ℝ^d_out (states mapping to observables)

**Dataset Construction (3 Steps)**:

**1. Simulation: Build SEP tree**:
- SEP solves nonlinearly along Gauss–Hermite tree
- Horizontally: time t−1, t, t+1,...
- Vertically: branches ε_t^(k)

**2. Dataset: Features and targets**:
- For each parent (t−1, g_par) and child gk at t:
  ```
  x = [st−1,obs(g_par), ε_t^(k), θ]
  y = [st,obs, st+1](gk)
  ```

**3. Neural Network Surrogate fϕ**:
- Train MLP: fϕ : x ↦ ŷ ≈ [st,obs, st+1](gk)
- Fully differentiable w.r.t. st−1,obs, εt, θ (for HMC/NUTS)
- Out-of-sample residuals → Σ_sur estimate

**Key Claim**:
- **Training data from SEP**: No analytic residuals, pure supervised learning
- **Parameter as input**: Single NN generalizes across θ ∈ Θ^train
- **Error covariance Σ_sur**: Quantified from held-out test set

**Paper Mapping**:
- Section 4: Dataset generation methodology
- Figure 2: SEP tree → training pairs illustration

---

### Slide 12: Surrogate for SEP Transition Map

**Architecture**: Multilayer perceptron (MLP)

**Input**: x ∈ ℝ^d_in
```
x = [st−1,obs^⊤, εt^⊤, θ^⊤]^⊤
```

**Output**: μ ∈ ℝ^d_out
```
μ = fϕ(x) ≈ [st,obs, st+1]
```

**Hidden Layer (tanh activation)**:
```
h = tanh(W1·x + b1)
μ = W2·h + b2
```
with weights W1 ∈ ℝ^(dh × din), W2 ∈ ℝ^(dout × dh)

**Visual**: Network diagram showing:
- Input nodes: st−1, εt, θ
- Hidden layer (tanh activation)
- Output nodes: [st,obs, st+1]
- Comparison to SEP: gθ(st−1, εt) = st

**Key Claim**:
- **Simple architecture**: Single hidden layer, tanh activation (smooth, bounded)
- **Adequate capacity**: dh ~ O(100) sufficient for medium-scale DSGE
- **Training**: Gradient descent on MSE loss

**Paper Mapping**:
- Section 4: Surrogate architecture subsection
- Figure 3: MLP architecture diagram

---

### Slide 13: Online - Synthetic Data from SEP

**Validation Protocol**:

**Synthetic Data Generation (4 Steps)**:
1. Fix "true" parameters θ0
2. Run SEP many times at θ0
3. Train surrogate NNs fϕ and f_LR
4. Extract observables path Y1:T from SEP solution

**Key Claim**:
- **Synthetic validation**: Test algorithm on controlled data where truth is known
- **Empirical application**: Replace synthetic Y1:T with actual macro time series

**Speaking Point (Inferred)**:
> "To validate the method, we generate synthetic data from SEP at a known θ0, then try to recover θ0 using the surrogate. This is a standard 'can we get the truth back?' test before applying to real data."

**Paper Mapping**:
- Section 7: Validation design subsection
- Procedure: Synthetic data generation protocol

---

### Slide 14: Online - Bayesian Model with Surrogate

**Unknowns**:
- Structural parameters θ
- Initial state s0
- Latent shocks {εt}_{t=1}^T

**Priors**:
- p(θ): Standard DSGE priors
- s0 | θ ~ N(μs(θ), Σs(θ)) with (μs, Σs) predicted by f_LR
- εt ~ N(0, I) i.i.d.

**State Recursion** (deterministic, approximated by surrogate):
```
st = gθ(st−1, ε[1:t−1]) ≈ fϕ(st−1, εt−1, θ), t = 1,...,T
```

**Measurement Equation**:
```
Yt = Hst + ηt, ηt ~ N(0, Σ_sur)
```

**Key Claim**:
- **Measurement error inflation**: Σ_sur captures surrogate approximation error
- **No filtering**: States constructed deterministically via surrogate recursion
- **Joint posterior**: p(θ, s0, ε1:T | Y1:T) sampled via HMC

**Paper Mapping**:
- Section 5: Bayesian model specification
- Equation block: Prior, state recursion, measurement equation

---

### Slide 15: Filter-Free Posterior

**Surrogate Recursion**:
```
st(θ, s0, ε1:t) := fϕ(st−1(·), εt, θ)
```

**Likelihood Factorization**:
```
p(Y1:T | θ, s0, ε1:T) = ∏_{t=1}^T N(Yt; Hst(θ, s0, ε1:t), Σ_sur)
```

**Filter-Free Target Posterior**:
```
p(θ, s0, ε1:T | Y1:T) ∝ p(θ) · p(s0 | θ) · ∏_{t=1}^T p(εt) · p(Yt | st(θ, s0, ε1:t))
```

**Key Claim**:
- **No Kalman/particle filter**: State path generated by unrolling surrogate NN
- **Latent objects sampled in HMC**: θ (parameters), s0 (initial condition), ε1:T (shocks)

**Speaking Point (Inferred)**:
> "We avoid filtering entirely. Given a proposed (θ, s0, ε1:T), we just unroll the surrogate to get the full state path, then evaluate the likelihood. This is what Childers et al. (2022) call 'filter-free' inference."

**Paper Mapping**:
- Section 5: Filter-free posterior derivation
- Emphasize: Computational advantage (no particle filter overhead)

---

### Slide 16: Sampling - NUTS/HMC

**Implementation**:
- Use Turing.jl framework
- No-U-Turn Sampler (NUTS), a variant of HMC

**Advantages**:
1. **Automatic differentiation**: AD through fϕ gives exact gradients ∇ log p(θ, s0, ε1:T | Y1:T)
2. **Filter-free** (Childers et al. 2022): Joint sampling of shocks + parameters, no Kalman/particle filter
3. **Cost per likelihood**: O(T · ds) - unroll surrogate once along time dimension, no SEP solves
4. **Surrogate error monitoring**: Σ_sur quantified, can debug misfit

**Key Claim**:
- **Gradient-based exploration**: More efficient than random-walk MH
- **Scalability**: O(T) cost per iteration vs O(T²) for Kalman filter
- **Diagnostics**: HMC divergences, ESS, R-hat provide approximation quality feedback

**Paper Mapping**:
- Section 5: HMC/NUTS implementation subsection
- Emphasize: Computational efficiency, diagnostic richness

---

### Slide 17: Nonlinear Equilibrium Conditions - Gali (2015)

**Model**: Baseline closed-economy NK model (Gali 2015, Chapter 3)

**Households**:
- Preferences: U(Ct, Nt; Zt) = Zt[Ct^(1−σ)/(1−σ) − Nt^(1+ϕ)/(1+ϕ)]
- Labor supply: Wt/Pt = UN,t/UC,t = Ct^σ · Nt^ϕ
- Euler equation: 1 = β(1 + it)·𝔼t[(Ct+1/Ct)^(−σ) · (Zt+1/Zt) · (1/Πt+1)]
- where Πt+1 ≡ Pt+1/Pt

**Firms, technology, policy**:
- Technology: Yt(i) = At·Nt(i)^(1−α), 0 < α < 1
- Demand for variety i: Yt(i) = (Pt(i)/Pt)^(−ε)·Yt, ε > 1
- Calvo pricing (aggregate price index): Pt^(1−ε) = θ·Pt−1^(1−ε) + (1−θ)·(Pt*)^(1−ε)
- Taylor rule: (1 + it)/(1 + ī) = (Πt/Π̄)^ϕπ · (Yt/Ȳ)^ϕy · e^vt, vt = ρv·vt−1 + εt^v

**Key Claim**:
- **Nonlinear form preserved**: No log-linearization
- **Standard NK model**: Widely understood, good testbed for method validation

**Paper Mapping**:
- Section 3: Model environment subsection
- Appendix A: Full model specification

---

### Slide 18: Pilot SEP+NN Estimation - Calvo Parameter θ^Calvo

**Visual**: 4-panel diagnostic plot
- Trace plot: θ^Calvo ~ 0.6-1.0 range, good mixing
- Density: Posterior centered near 0.75, mode ≈ 0.70, truth ≈ 0.75
- Autocorrelation: Decays quickly to near zero
- Running mean: Converges to truth (~0.73)

**Diagnostics Footer**:
```
posterior diagnostics combined across chains:
median = 0.735151, mode = 0.709546
95% band = [0.449369, 0.963161]
ESS = 10%?, R̂ = 1.00201
```

**Key Claim**:
- **3-parameter recovery**: Calvo parameter θ^Calvo successfully recovered
- **Chain diagnostics**: Good mixing (low autocorrelation), converged (R̂ ≈ 1)

**Paper Mapping**:
- Section 8: Results - 3-parameter validation
- Figure 4: Posterior diagnostics for θ^Calvo

---

### Slide 19: Pilot SEP+NN Estimation - Taylor Rule Slope ϕπ

**Visual**: 4-panel diagnostic plot
- Trace plot: ϕπ ~ 0.8-2.5 range, good mixing
- Density: Posterior centered near 1.5, mode ≈ 1.50, truth ≈ 1.5
- Autocorrelation: Decays quickly
- Running mean: Converges to truth (~1.49)

**Diagnostics Footer**:
```
median = 1.4899, mode = 1.5005
95% band = [0.95393, 2.0956]
ESS = 1560.54, R̂ = 1.00407
```

**Key Claim**:
- **Taylor rule slope ϕπ recovered**: Posterior mode matches truth
- **Reasonable uncertainty**: 95% band [0.95, 2.10] reflects identification

**Paper Mapping**:
- Section 8: Results - 3-parameter validation
- Figure 5: Posterior diagnostics for ϕπ

---

### Slide 20: Pilot SEP+NN Estimation - Taylor Rule Slope ϕy

**Visual**: 4-panel diagnostic plot
- Trace plot: ϕy ~ 0.05-0.20 range, good mixing
- Density: Posterior centered near 0.125, mode ≈ 0.125, truth ≈ 0.125
- Autocorrelation: Decays quickly
- Running mean: Converges to truth (~0.123)

**Diagnostics Footer**:
```
median = 0.12372, mode = 0.124567
95% band = [0.0729987, 0.176992]
ESS = 14%?, R̂ = 1.00321
```

**Key Claim**:
- **Output gap response ϕy recovered**: Posterior mode matches truth
- **All 3 parameters successfully recovered**: θ^Calvo, ϕπ, ϕy

**Paper Mapping**:
- Section 8: Results - 3-parameter validation
- Figure 6: Posterior diagnostics for ϕy

---

### Slide 21: Summary

**Key Contributions**:
1. **SEP (Stochastic Extended Path)**: Provides nonlinear, expectations-consistent solutions at design points in θ
2. **Neural Network Surrogate**: Learns transition map (st−1, εt, θ) ↦ st from global solutions
3. **Bayesian Inference with NUTS**:
   - θ as unknown structural parameters
   - εt as latent shocks
   - s0 as unknown initial condition
   - Surrogate as fast emulator of full model
4. **Result**: Tractable global estimation of nonlinear DSGE models without linearization or filtering approximations

**Speaking Point (Inferred)**:
> "We've shown that you can estimate nonlinear DSGE models globally by training a surrogate on SEP solutions, then using that surrogate in a filter-free HMC sampler. This makes global nonlinear estimation practical."

**Paper Mapping**:
- Section 9: Conclusion
- Emphasize: Feasibility result (global estimation now practical), future extensions (18-param scale-up)

---

### Slide 22: HOW TO PILOT IT IN AI (Meta-Slide)

**Visual**: Large text "HOW TO PILOT IT IN AI"

**Implied Content**: Discussion of AI-assisted development workflow

**Speaking Point (Inferred from Slide 23)**:
> "This project was developed end-to-end using AI tools. Let me show you the workflow."

**Paper Mapping**:
- Not directly relevant to paper (methodological, not meta-methodological)
- Could mention in acknowledgments: "Development accelerated by AI-assisted coding tools"

---

### Slide 23: AI-Assisted Development Workflow

**Visual**: 4-step flowchart

**Workflow**:
1. **Kick-off in ChatGPT**: High-level prompt + nonlinear Gali & SEP code
2. **ChatGPT Projects**: Branching chats into focused sub-tasks
3. **VS Code + Codex**: Tight loop on SEP and estimation code
4. **Claude Sonnet**: Cross-checking, refinement, diagnostics

**End-to-end development**: Multiple LLMs as complementary tools, from idea/model setup through code refactoring to cross-checking and experimentation

**Speaking Point (Inferred)**:
> "I used ChatGPT for high-level design, Codex in VS Code for implementation, and Claude for validation. This hybrid approach accelerated development significantly."

**Paper Mapping**:
- Not relevant to main text
- Possible acknowledgment: "AI-assisted development tools"

---

### Slide 24: How to Reproduce This AI Workflow

**Workflow Steps**:

**1. Kick-start in chat interface**:
- Write high-level prompt describing estimation goal
- Paste minimal working DSGE (e.g., Gali model) + global solver (SEP/Dynare)
- Use model to draft clean specification and baseline script

**2. Turn exploration into structured "project"**:
- Create branches for: (i) SEP solver, (ii) NN surrogate, (iii) Bayesian estimation and diagnostics
- Keep prompts, assumptions, design decisions documented in project

**3. Move code to editor + model-agnostic helper**:
- Sync Git repo into VS Code (or IDE)
- Use code-focused assistant (Codex-style) for refactoring, debugging, profiling SEP and HMC code

**Speaking Point (Inferred)**:
> "The key is to use different tools for different tasks: chat interfaces for design, code editors for implementation, cross-checking tools for validation."

**Paper Mapping**:
- Not relevant to main text
- Possible methods appendix: "Computational workflow and reproducibility"

---

## Summary of Key Claims for Paper

### Core Methodological Claims
1. **Global nonlinear DSGE estimation is feasible** using SEP + NN surrogates
2. **Two-stage approach**: Offline (SEP + training) + Online (fast HMC estimation)
3. **Filter-free inference**: Joint sampling of (θ, s0, ε1:T) without Kalman/particle filter
4. **Parameter-conditioned surrogate**: Single NN generalizes across θ ∈ Θ^train
5. **Surrogate error quantification**: Σ_sur estimated from out-of-sample residuals
6. **Gradient-based HMC**: Differentiable surrogate enables efficient exploration

### Empirical Claims
7. **3-parameter recovery validated**: θ^Calvo, ϕπ, ϕy successfully recovered from synthetic data
8. **Chain diagnostics**: Good mixing, convergence (R̂ ≈ 1), adequate ESS

### Comparison Claims
9. **Gust et al. comparison**: Similar grid technology, different function class (NN vs polynomial)
10. **KMR comparison**: Similar surrogate idea, different targets (likelihood-ready vs solution amortization)

### Computational Claims
11. **O(T · ds) cost per HMC iteration**: Linear in time horizon, no quadratic filter overhead
12. **SEP as global solver**: Handles kinks/constraints without analytic regime decomposition
13. **Smolyak + Gauss–Hermite grids**: Efficient structured coverage of (state, shock, parameter) space

---

## Paper Section Mapping (Preliminary)

1. **Introduction**: Motivation (Slide 2), Problem statement (Slide 3), Contributions (Slide 21)
2. **Related Work**: Gust et al. (Slide 7), KMR (Slide 8)
3. **Model Environment**: Notation (Slide 4), Gali NK model (Slide 17)
4. **Methodology - SEP**: SEP algorithm (Slides 9-10)
5. **Methodology - Surrogate**: Dataset generation (Slide 11), MLP architecture (Slide 12)
6. **Methodology - Estimation**: Bayesian model (Slide 14), Filter-free posterior (Slide 15), HMC (Slide 16)
7. **Validation Design**: Synthetic data (Slide 13)
8. **Results**: 3-parameter recovery (Slides 18-20)
9. **Conclusion**: Summary (Slide 21)

---

**End of Presentation Claims Extraction**

**Next Step**: Use these claims to populate JMP outline and draft sections.
