# Methodology Overview: Regime-Switching Estimation for Nonlinear DSGE Models

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Mathematical Framework](#mathematical-framework)
4. [Model Specification](#model-specification)
5. [Validation Strategy](#validation-strategy)
6. [References](#references)

---

## Problem Statement

### The Challenge

Bayesian estimation of nonlinear Dynamic Stochastic General Equilibrium (DSGE) models with occasionally-binding constraints (OBC) faces a fundamental computational challenge:

**Accurate nonlinear solutions are too slow for thousands of likelihood evaluations required by MCMC/HMC.**

Specifically:
- **Linear approximations** (perturbation methods) are fast but inaccurate when constraints bind (e.g., zero lower bound on interest rates)
- **Nonlinear solvers** (e.g., extended path, projection methods) are accurate but too slow (~1-10 seconds per evaluation)
- **HMC estimation** requires ~1,000-10,000 likelihood evaluations with gradient computations

### The Zero Lower Bound Problem

The zero lower bound (ZLB) on nominal interest rates is a canonical OBC:

```
r_t ≥ 0  (or r_t ≥ 1 in gross terms)
```

When this constraint binds:
- **Linear solutions are invalid**: They predict negative rates
- **Model dynamics change nonlinearly**: Output multipliers increase, forward guidance becomes powerful
- **Perturbation approximations fail**: First/second-order Taylor expansions around steady state don't capture constraint

### Existing Approaches and Limitations

| Approach | Speed | Accuracy | OBC Support | Bayesian Estimation |
|----------|-------|----------|-------------|---------------------|
| Linear (ROM1/ROM2) | Fast (ms) | Good (normal times) | ✗ No | ✓ Yes |
| Extended Path (EP) | Slow (seconds) | High | ✓ Yes | ✗ Infeasible |
| Projection Methods | Very Slow | Very High | ✓ Yes | ✗ Infeasible |
| Piecewise Linear | Fast | Medium | ✓ Approximate | ✓ Limited |

**Gap**: No method combines speed, accuracy, OBC support, and full Bayesian inference.

---

## Solution Architecture

### Core Idea: Hybrid Regime Switching

Our framework combines the best of both worlds:

1. **Use linear ROM** (reduced-order model) during **normal times** → Fast Kalman filter
2. **Switch to nonlinear surrogate** during **high-volatility/ZLB episodes** → Accurate dynamics
3. **Automatic regime detection** via empirical Bayes calibration → Data-driven gating

### Three-Component System

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Phase (Offline)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. SEP Dataset Generation                                  │
│     • Sample parameters θ from prior                        │
│     • Generate shock sequences ε                            │
│     • Solve SEP (stochastic extended path) → Y_sep          │
│     • Solve ROM1/ROM2 → Y_rom (baselines)                   │
│     • Store: (state_t, ε_t, θ) → (obs_t+1, state_t+1)       │
│                                                              │
│  2. Surrogate Training                                      │
│     • Train neural network: f(state_t, ε_t, θ) → Δ          │
│     • Residual learning: Δ = Y_sep - Y_rom1                 │
│     • Architecture: 2-3 layer MLP, obs-only output          │
│     • Validation: RMSE on held-out episodes                 │
│                                                              │
│  3. Gate Calibration (Empirical Bayes)                      │
│     • Generate synthetic data with known high-vol episodes  │
│     • Compute ROM errors + shock magnitudes                 │
│     • Calibrate threshold: P(gate=1 | features) = 0.10      │
│     • Episode padding: k_pre, k_post for smoothing          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Inference Phase (Online)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For each MCMC/HMC iteration:                               │
│    1. Propose θ ~ q(θ | θ_prev)                             │
│    2. For each time period t:                               │
│       a. Evaluate gate: g_t = I(high_volatility_t)          │
│       b. If g_t = 0: Use ROM1 Kalman filter                 │
│       c. If g_t = 1: Use ROM1 + Surrogate(state, ε, θ)      │
│    3. Compute log-likelihood: Σ_t log p(y_t | θ, g)         │
│    4. Accept/reject with MH or NUTS                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Residual Learning**: Train on `Y_sep - Y_rom1` instead of `Y_sep`
   - Surrogate only learns correction, not full dynamics
   - Smaller neural network (faster inference)
   - Better generalization (ROM1 provides structure)

2. **Hard Gating**: Binary regime switch instead of soft mixture
   - No mixing weights to estimate
   - Clear interpretation: "linear is good enough" vs "need nonlinear"
   - Faster inference (avoid dual evaluation)

3. **Empirical Bayes Calibration**: Data-driven threshold selection
   - Calibrate gate on synthetic data where ROM errors are known
   - Control gate frequency (e.g., 10% of periods)
   - Adaptive to model and shock distribution

4. **Baseline ROM Mode**: Compute ROM at fixed θ for AD compatibility
   - ROM1(θ_paper) used for all MCMC iterations
   - Differentiable through surrogate only
   - Avoids expensive re-solve of perturbation equations

---

## Mathematical Framework

### 1. Stochastic Extended Path (SEP)

SEP approximates the rational expectations equilibrium by solving a sequence of perfect foresight problems with stochastic terminal conditions.

**Algorithm** (Adjemian & Juillard, 2013):
```
For each simulation period t:
  1. Draw terminal shocks: ε_{t+1:t+H} ~ N(0, Σ)
  2. Solve perfect foresight path:
     Find {y_s}_{s=t}^{t+H} satisfying:
       E_t[f(y_{s+1}, y_s, y_{s-1}, ε_s)] = 0  for s = t,...,t+H-1
       y_{t+H} = ROM2(y_{t+H-1}, ε_{t+H})  (terminal condition)
  3. Return: y_t (first period of solution)
  4. Advance: y_{t+1} using y_t and new shock ε_{t+1}
```

**Parameters**:
- Horizon `H`: 8-12 periods typical (trade-off: accuracy vs speed)
- Terminal order: ROM2 provides stable boundary condition
- Solver: Newton's method with line search, tolerance ~1e-5

**Advantages**:
- Handles OBC naturally (inequality constraints in solver)
- Approximates news shocks and forward guidance effects
- Converges to REE as H → ∞

**Computational Cost**: ~0.1-2 seconds per period (model-dependent)

### 2. Reduced-Order Models (ROM)

Perturbation approximations around deterministic steady state.

**ROM1 (First-Order)**:
```
State transition:  x_{t+1} = g_x(θ) x_t + g_ε(θ) ε_t
Observation:       y_t = h_x(θ) x_t
```

- Linear in state and shocks
- Certainty equivalence holds
- Solve once per θ via generalized Schur decomposition
- Computational cost: ~10-50ms per θ

**ROM2 (Second-Order)**:
```
x_{t+1} = g_x x_t + g_ε ε_t + (1/2)[g_xx (x_t ⊗ x_t) + g_εε (ε_t ⊗ ε_t) + g_σσ σ²]
y_t = h_x x_t + (1/2) h_xx (x_t ⊗ x_t)
```

- Captures risk premia and precautionary savings
- More accurate far from steady state
- Computational cost: ~50-200ms per θ

**Limitation**: Cannot handle binding OBC → predicts constraint violations

### 3. Neural Network Surrogate

Approximate the SEP solution using a feedforward neural network.

**Architecture**:
```
Input:  [state_t, ε_t, θ] ∈ ℝ^{n_x + n_ε + n_θ}
Hidden: [256] → ReLU → [128] → ReLU
Output: Δ_obs ∈ ℝ^{n_y} (observable residuals only)
```

**Training**:
- Loss: MSE on residuals `Δ = Y_sep - Y_rom1`
- Optimizer: Adam with learning rate decay
- Validation split: 20% of episodes
- Early stopping: Monitor validation RMSE

**Inference** (at parameter θ):
```
Y_pred(state_t, ε_t, θ) = Y_rom1(state_t, ε_t, θ_baseline) + NN(state_t, ε_t, θ)
```

Where `θ_baseline` is the fixed calibration (e.g., posterior mode from linear estimation).

**Why obs-only output?**
- Kalman filter only needs observables for likelihood
- Full state prediction requires much larger network
- Obs-only: ~10× faster training, similar likelihood accuracy

### 4. Hard-Gate Regime Switching

**Gate Definition**:
```
g_t = I{score_t > threshold}

score_t = w_ε · |ε_t|_∞ + w_y · RMSE_ROM(y_{t-k:t})
```

- `|ε_t|_∞`: Largest shock magnitude at time t
- `RMSE_ROM`: Forecast error of ROM over recent window
- Weights: `w_ε`, `w_y` ∈ [0,1], typically (0.5, 0.5)

**Likelihood**:
```
log p(y_{1:T} | θ, g_{1:T}) = Σ_{t: g_t=0} log p_ROM(y_t | θ)
                             + Σ_{t: g_t=1} log p_Surr(y_t | θ)
```

Where:
- `p_ROM`: Kalman filter likelihood (Gaussian)
- `p_Surr`: Surrogate prediction likelihood (Gaussian, may have inflated noise)

**Episode Padding**:
To avoid abrupt transitions:
- If `g_t = 1`, activate `g_{t-k_pre : t+k_post} = 1`
- Minimum episode length: `min_len` periods
- Typical values: `k_pre=4`, `k_post=8`, `min_len=4`

### 5. Empirical Bayes Gate Calibration

**Objective**: Choose `threshold` such that gate activates in ~10% of periods.

**Procedure**:
1. Generate synthetic data with known high-volatility window
   ```
   Periods 1-79:    ε ~ N(0, Σ)
   Periods 80-120:  ε ~ N(0, 3·Σ)  (amplified shocks)
   Periods 121-200: ε ~ N(0, Σ)
   ```

2. Compute scores on synthetic data:
   ```
   score_t = w_ε · |ε_t|_∞ + w_y · RMSE_ROM(y_{t-k:t})
   ```

3. Select threshold at desired quantile:
   ```
   threshold = quantile(score_{1:T}, 0.90)  → 10% gate frequency
   ```

4. Validate: Check gate activates primarily in high-vol window (80-120)

**Rationale**: Calibration data has same model structure as estimation data, so empirical distribution of scores is representative.

---

## Model Specification

### HLT Model: Hybrid Linear-Nonlinear Trinomial

Based on Smets & Wouters (2007) medium-scale DSGE with modifications for OBC handling.

**Observables** (7):
1. `dy`: Output growth
2. `dc`: Consumption growth
3. `dinve`: Investment growth
4. `labobs`: Hours worked
5. `dw`: Wage growth
6. `pinfobs`: Inflation (quarterly)
7. `robs`: Nominal interest rate (quarterly)

**Shocks** (7):
1. `ea`: Productivity shock
2. `eb`: Risk premium shock
3. `eg`: Government spending shock
4. `eqs`: Investment-specific technology shock
5. `em`: Monetary policy shock
6. `epinf`: Price markup shock
7. `ew`: Wage markup shock

**State Variables** (19):
- Past observables (lags)
- Endogenous states (capital, price/wage dispersion)
- Exogenous AR(1) processes

**Parameters** (36):
- Technology: `ctrend`, `constepinf`, `calfa`, `czcap`, etc.
- Preferences: `csigma` (risk aversion), `chabb` (habit), `csadjcost` (investment adj cost)
- Frictions: `cfc` (fixed cost), `cindp` (price indexation), `cindw` (wage indexation)
- Policy: `crpi` (inflation response), `crr` (smoothing), `cry` (output response)
- Shock persistence: `crhoa`, `crhob`, etc.
- Shock volatility: `cstddev_a`, `cstddev_b`, etc.

### OBC Specification

**Zero Lower Bound Constraint** (in `Smets_Wouters_2007_HLT_obc.jl`):
```julia
# Monetary policy rule (original):
# r = crr·r_{-1} + (1-crr)·[crpi·π + cry·(y-y_pot)] + ε^m

# With ZLB:
r_desired = crr·r_{-1} + (1-crr)·[crpi·π + cry·(y-y_pot)] + ε^m
r = max(r_desired, 1.0)  # Gross rate ≥ 1 (net rate ≥ 0)
```

**Implementation**:
- SEP solver: Inequality constraint added to perfect foresight system
- ROM: Constraint ignored (may predict violations)
- Surrogate: Learns to correct ROM when constraint binds

**Alternative Specification**: Could also bound shadow rate or use smooth penalty function.

### State Definition

Following MacroModelling.jl conventions:

**State vector** `x_t`:
- `past_not_future_and_mixed`: Predetermined variables at time t
- `future_not_past_and_mixed`: Forward-looking variables at time t

**For SEP**:
- Input: `x_t`, `ε_t`, `θ`
- Output: `x_{t+1}`, `y_t`

**For ROM**:
- Same input/output structure
- Linear maps: `g_x`, `g_ε`, `h_x`

**For Surrogate**:
- Input: `[x_t, ε_t, θ]` (concatenated)
- Output: `Δy_t` (observable residuals)

---

## Validation Strategy

### 1. Synthetic Data Validation

**Setup**:
- Generate synthetic observables using SEP at `θ_true`
- Add measurement noise: `y^obs_t = y^sep_t + ν_t`, `ν_t ~ N(0, σ²_obs I)`
- Include high-volatility window (known ground truth)

**Metrics**:

a) **Approximation Accuracy** (vs SEP):
```
RMSE_full = √(1/T Σ_t |y^rom_t - y^sep_t|²)
RMSE_highvol = √(1/T_vol Σ_{t∈vol} |y^rom_t - y^sep_t|²)
```

Typical results:
- ROM1: RMSE_full ≈ 0.10, RMSE_highvol ≈ 0.15
- ROM1+Surrogate: RMSE_full ≈ 0.09, RMSE_highvol ≈ 0.11 (3% improvement)

b) **IRF Comparison**:
- Impulse responses to each shock (1 std dev)
- Horizon: 10-40 periods
- Compare ROM1, ROM2, SEP, Surrogate
- Visual inspection + correlation metrics

c) **Gate Diagnostics**:
- Gate activation rate: ~10% of periods (by construction)
- Gate concentration: >70% of high-vol window should activate
- Episode lengths: Mean ~8 periods, range [4, 20]

### 2. Estimation Validation

**Setup**:
- Estimate using ROM1+Surrogate with gating
- Compare to linear ROM1-only estimation

**Metrics**:

a) **Posterior Diagnostics**:
- R-hat < 1.01 (convergence)
- ESS > 400 per chain (mixing)
- ACF1 < 0.5 for key parameters (autocorrelation)
- No boundary concentration (cprobp, cindp)

b) **Model Fit**:
- Log marginal likelihood (via bridge sampling or importance sampling)
- In-sample RMSE
- Out-of-sample forecast errors

c) **Parameter Recovery** (synthetic data):
```
Bias = E[θ_hat - θ_true]
Coverage = P(θ_true ∈ CI_95%)
```

Targets: Bias < 0.1·σ_prior, Coverage ≈ 0.95

### 3. Computational Benchmarks

**Speed Comparison** (per likelihood evaluation):
- ROM1 Kalman filter: ~5ms
- ROM1 + Surrogate (10% gate): ~8ms (60% overhead)
- Full SEP: ~500-2000ms (100-400× slower)

**Estimation Time** (1000 HMC samples):
- Linear ROM1: ~5 minutes
- ROM1 + Surrogate: ~10 minutes
- Full SEP: ~8-33 hours (infeasible)

**Memory**:
- Surrogate model: ~5MB (256→128 architecture)
- Dataset: ~100MB-1GB (depends on sample size)

---

## References

### Core Methodology

- **Smets, F., & Wouters, R. (2007).** "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach." *American Economic Review*, 97(3), 586-606.
  - Source model specification (SW07)

- **Adjemian, S., & Juillard, M. (2013).** "Stochastic Extended Path Approach." Dynare Working Paper.
  - SEP algorithm and implementation

- **Den Haan, W. J., & De Wind, J. (2012).** "Nonlinear and stable perturbation-based approximations." *Journal of Economic Dynamics and Control*, 36(10), 1477-1497.
  - Perturbation method accuracy and stability

### Related Work

- **Guerrieri, L., & Iacoviello, M. (2015).** "OccBin: A toolkit for solving dynamic models with occasionally binding constraints easily." *Journal of Monetary Economics*, 70, 22-38.
  - Piecewise linear approach to OBC

- **Maliar, L., & Maliar, S. (2015).** "Merging simulation and projection approaches to solve high-dimensional problems with an application to a new Keynesian model." *Quantitative Economics*, 6(1), 1-47.
  - Projection methods and neural network approximations

- **Fernández-Villaverde, J., Hurtado, S., & Nuño, G. (2023).** "Financial Frictions and the Wealth Distribution." *Econometrica*, 91(3), 869-901.
  - Heterogeneous agent DSGE with neural network solutions

### Software

- **MacroModelling.jl**: Julia package for DSGE modeling
  - Repository: https://github.com/thorek1/MacroModelling.jl

- **Dynare**: MATLAB/Octave platform for DSGE models
  - Website: https://www.dynare.org/

- **Turing.jl**: Probabilistic programming in Julia (HMC/NUTS)
  - Repository: https://github.com/TuringLang/Turing.jl

---

## Summary

This framework achieves **fast, accurate Bayesian estimation** of nonlinear DSGE models with OBC through:

1. **Hybrid approximation**: Linear ROM for normal times + Surrogate for high-volatility/ZLB
2. **Residual learning**: Neural network learns corrections, not full dynamics
3. **Hard gating**: Data-driven regime switching via empirical Bayes
4. **Computational efficiency**: ~60% overhead vs linear, 100× faster than full SEP

**Validated performance**: ~3% RMSE improvement in high-volatility windows, feasible HMC estimation in ~10 minutes.

**Next steps**: See [CURRENT_APPROACHES.md](CURRENT_APPROACHES.md) for implementation details and [../tutorials/FULL_PIPELINE.md](../tutorials/FULL_PIPELINE.md) for hands-on walkthrough.

---

**Questions?** See [../README.md](../README.md) "Getting Help" or [../tutorials/TROUBLESHOOTING.md](../tutorials/TROUBLESHOOTING.md)
