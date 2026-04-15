# Hard-Gate Regime Switching

**Feature**: Automatic regime detection and switching between linear ROM and nonlinear surrogate
**Status**: Production-ready ✅
**Implementation**: Estimation scripts + gate calibration utilities

---

## Overview

**Regime switching** enables the estimation framework to automatically detect when linear approximations (ROM) are insufficient and switch to the nonlinear surrogate. This hybrid approach achieves near-SEP accuracy during high-volatility episodes while maintaining computational efficiency during normal times.

### Key Innovation: Hard Gating

**Binary regime switch** based on data-driven thresholds:
```
g_t = {
  0  if  score_t ≤ threshold  →  Use linear ROM (fast)
  1  if  score_t > threshold   →  Use nonlinear surrogate
}
```

**Advantages over soft gating**:
- No mixing weights to estimate
- Clear interpretation: "linear is adequate" vs "need nonlinear"
- Faster inference (no dual model evaluation)
- Easier to diagnose and calibrate

---

## Mathematical Framework

### Gate Score Function

**Composite score** combining shock magnitude and forecast errors:

```
score_t = w_ε · |ε_t|_∞ + w_y · RMSE_ROM(y_{t-k:t})
```

**Components**:

1. **Shock magnitude**: `|ε_t|_∞ = max_i |ε_{i,t}|`
   - Large shocks → likely constraint binding or nonlinearity
   - Instantaneous signal (forward-looking)

2. **ROM forecast error**: `RMSE_ROM(y_{t-k:t}) = √(1/k Σ_{s=t-k}^t |y_s - ŷ_ROM_s|²)`
   - Measures ROM approximation quality
   - Backward-looking (adapts to observed misfit)

**Weights**: Typically `w_ε = 0.5`, `w_y = 0.5` (equal contribution)

---

### Threshold Calibration: Empirical Bayes

**Objective**: Select threshold such that gate activates in ~10% of periods.

**Procedure**:

**1. Generate synthetic validation data** with known high-volatility window:
```julia
# Normal times: periods 1-79, 121-200
ε[1:79, :] = randn(79, n_shocks)
ε[121:200, :] = randn(80, n_shocks)

# High-volatility window: periods 80-120
ε[80:120, :] = 3.0 * randn(41, n_shocks)  # Amplified shocks

y_synth = sep_solve_mm!(model, θ_calib, ε)
```

**2. Compute scores** on synthetic data:
```julia
scores = zeros(T)
for t in 1:T
    # Shock component
    shock_score = maximum(abs, ε[:, t])

    # Forecast error component (if t > k)
    if t > k_window
        y_rom = rom1_predict(model, θ_calib, ε[:, t-k_window:t])
        fe_score = sqrt(mean(abs2, y_synth[:, t-k_window:t] .- y_rom))
    else
        fe_score = 0.0
    end

    scores[t] = w_ε * shock_score + w_y * fe_score
end
```

**3. Select threshold** at desired quantile:
```julia
# Target: 10% gate activation
threshold = quantile(scores, 0.90)

# Verify: gate should concentrate in high-vol window
gate_mask = scores .> threshold
println("Gate activation rate: ", mean(gate_mask))  # Should be ~0.10
println("Gate share in high-vol window: ", mean(gate_mask[80:120]))  # Should be >0.70
```

**4. Save calibration**:
```julia
gate_calib = Dict(
    :threshold => threshold,
    :w_eps => w_ε,
    :w_y => w_y,
    :k_window => k_window,
    :scores => scores,
    :gate_mask => gate_mask
)
serialize("gate_calibration.jls", gate_calib)
```

---

### Episode Padding

**Issue**: Binary gate can cause abrupt transitions:
```
Periods: ... 78  79  80  81 ... 119 120 121 122 ...
Gate:    ... 0   0   1   1  ...  1   1   0   0  ...
                 ↑ Sudden switch
```

**Solution**: Pad gate episodes for smooth transitions.

**Algorithm**:
```julia
function pad_gate_episodes(gate_raw::Vector{Bool};
                           k_pre::Int=4,
                           k_post::Int=8,
                           min_len::Int=4)
    gate_padded = copy(gate_raw)
    T = length(gate_raw)

    # Find gate activation periods
    active_periods = findall(gate_raw)

    if isempty(active_periods)
        return gate_padded
    end

    # Identify episodes (contiguous active periods)
    episodes = []
    ep_start = active_periods[1]
    ep_end = active_periods[1]

    for t in active_periods[2:end]
        if t == ep_end + 1
            ep_end = t  # Extend current episode
        else
            push!(episodes, (ep_start, ep_end))
            ep_start = t
            ep_end = t
        end
    end
    push!(episodes, (ep_start, ep_end))

    # Pad each episode
    for (start_t, end_t) in episodes
        # Pre-padding: activate k_pre periods before
        pad_start = max(1, start_t - k_pre)

        # Post-padding: activate k_post periods after
        pad_end = min(T, end_t + k_post)

        # Apply padding
        gate_padded[pad_start:pad_end] .= true
    end

    # Enforce minimum episode length
    # (Re-identify episodes in padded mask)
    active_periods = findall(gate_padded)
    episodes = identify_episodes(active_periods)

    for (start_t, end_t) in episodes
        if (end_t - start_t + 1) < min_len
            # Episode too short → deactivate
            gate_padded[start_t:end_t] .= false
        end
    end

    return gate_padded
end
```

**Typical values**:
- `k_pre = 4`: Activate 4 periods before episode start (anticipate constraint)
- `k_post = 8`: Activate 8 periods after episode end (smooth exit)
- `min_len = 4`: Minimum episode length (avoid single-period activations)

**Effect**: Smoother regime transitions, fewer artificial discontinuities in likelihood.

---

## Likelihood Computation

### Hybrid Likelihood

**Goal**: Combine ROM (linear) and Surrogate (nonlinear) likelihoods based on gate mask.

**Formula**:
```
log p(y_{1:T} | θ, g_{1:T}) = Σ_{t: g_t=0} log p_ROM(y_t | θ, y_{1:t-1})
                              + Σ_{t: g_t=1} log p_Surr(y_t | θ, y_{1:t-1})
```

**Implementation**:
```julia
function hybrid_likelihood(y_obs, θ, model, frozen, gate_mask;
                           obs_sigma_scale=1.0)
    T = size(y_obs, 2)
    loglik = 0.0

    # Infer shocks (ROM inversion)
    ε, _ = invert_for_shocks(model, y_obs, θ)

    # Recover states
    x = recover_states(model, y_obs, θ)

    # Observation noise covariance
    Σ_obs = obs_sigma_scale^2 * I(n_obs)

    for t in 1:T
        if gate_mask[t]
            # ========== Surrogate regime ==========
            # Predict with ROM + Surrogate
            y_rom = rom1_predict(x[:, t], ε[:, t], θ_baseline)
            Δ_surr = predict_frozen(frozen, [x[:, t]; ε[:, t]; θ])
            y_pred = y_rom + Δ_surr

            # Surrogate likelihood (may have inflated noise)
            loglik += log_mvnormal_pdf(y_obs[:, t], y_pred, Σ_obs)
        else
            # ========== ROM regime ==========
            # Predict with ROM only
            y_pred = rom1_predict(x[:, t], ε[:, t], θ_baseline)

            # ROM likelihood (standard noise)
            loglik += log_mvnormal_pdf(y_obs[:, t], y_pred, Σ_obs)
        end

        # Shock prior (applies in both regimes)
        loglik += log_mvnormal_pdf(ε[:, t], zeros(n_shocks), Σ_ε(θ))
    end

    return loglik
end
```

**Key features**:
- **Conditional on shocks**: Both regimes use inferred ε_t (filter-free sampling)
- **Gate-specific noise**: Can inflate `obs_sigma_scale` in surrogate periods (accounts for approximation error)
- **Common shock prior**: Regularizes ε_t in both regimes

---

### Gate-Consistent Likelihood Scaling

**Issue** (discovered in Step 63): Mixed unconditional/conditional likelihoods cause scale mismatch.

**Problem**:
```julia
# WRONG: Inconsistent likelihood scales
loglik_linear = kalman_filter(y, θ)          # Unconditional (integrates over ε)
loglik_gate = surrogate_loglik(y, θ, ε)      # Conditional (fixed ε)

# Result: loglik_gate >> loglik_linear → posterior concentrates in gate periods
```

**Fix**: Both regimes use **conditional likelihood** when sampling shocks:
```julia
# CORRECT: Both conditional on sampled ε
loglik_linear = conditional_rom_loglik(y, θ, ε)   # Conditional on ε
loglik_gate = surrogate_loglik(y, θ, ε)           # Conditional on ε

# Shock prior adds unconditional component
loglik_total = loglik_linear + loglik_gate + log p(ε)
```

**Implementation**: See `hybrid_likelihood` above (both regimes use inferred ε).

---

## Advanced Features

### 1. Observation Noise Relaxation

**Motivation**: Surrogate approximation introduces additional uncertainty beyond measurement noise.

**Solution**: Inflate observation noise in gate periods:
```julia
function hybrid_likelihood_relaxed(y_obs, θ, ..., gate_mask;
                                   obs_sigma_scale_rom=1.0,
                                   obs_sigma_scale_surr=2.0)
    for t in 1:T
        if gate_mask[t]
            Σ_obs = obs_sigma_scale_surr^2 * I(n_obs)  # Relaxed
        else
            Σ_obs = obs_sigma_scale_rom^2 * I(n_obs)   # Standard
        end

        loglik += log_mvnormal_pdf(y_obs[:, t], y_pred, Σ_obs)
    end
end
```

**Typical values**:
- ROM periods: `obs_sigma_scale_rom = 1.0` (trust ROM)
- Surrogate periods: `obs_sigma_scale_surr = 1.5-2.0` (account for surrogate error)

**Effect**: Prevents over-penalization of small surrogate prediction errors.

---

### 2. Gate-Window Shock Sampling (Experimental)

**Motivation**: Sampling all T shocks increases dimensionality. Most shocks in non-gate periods are well-approximated by ROM.

**Proposal**: Sample shocks **only in gate window**.

**Implementation**:
```julia
@model function estimate_gate_window(y_obs, model, frozen, gate_mask)
    # Structural parameters
    θ ~ prior_distribution()

    # Gate periods
    gate_periods = findall(gate_mask)

    # Sample shocks ONLY in gate window
    ε_gate ~ filldist(Normal(0, 1), n_shocks, length(gate_periods))

    # Fix shocks in non-gate periods (ROM inversion)
    ε = zeros(n_shocks, T)
    ε[:, gate_periods] = ε_gate
    ε[:, .!gate_mask] = rom_invert_shocks(y_obs[:, .!gate_mask], θ)

    # Likelihood
    loglik = hybrid_likelihood(y_obs, θ, model, frozen, gate_mask, ε)
    Turing.@addlogprob! loglik
end
```

**Benefits**:
- Reduced dimensionality (sample ~20-40 shocks instead of 200)
- Faster HMC mixing (fewer correlated parameters)
- Focuses sampling effort where surrogate is active

**Status**: Experimental (Step 64 proposal, not yet fully validated).

---

### 3. Guided Shock Priors

**Motivation**: Unconstrained shock sampling can lead to extreme trajectories that violate model assumptions.

**Solution**: Anchor shock prior to Kalman filter path.

**Implementation**:
```julia
function guided_shock_prior(y_obs, θ, model;
                            guidance_scale=0.5,
                            prior_scale=0.5)
    # Run Kalman filter to get "reasonable" shock path
    ε_kf, _ = kalman_smoother(model, y_obs, θ)

    # Guided prior: ε ~ N(guidance_scale * ε_kf, prior_scale^2 Σ_ε)
    @model function model_with_guided_shocks()
        θ ~ prior_distribution()

        # Guided shock distribution
        for t in 1:T
            ε[:, t] ~ MvNormal(guidance_scale * ε_kf[:, t],
                               prior_scale^2 * Σ_ε(θ))
        end

        # Likelihood
        loglik = hybrid_likelihood(y_obs, θ, model, frozen, gate_mask, ε)
        Turing.@addlogprob! loglik
    end
end
```

**Hyperparameters**:
- `guidance_scale = 0.5`: Shrink toward 50% of Kalman path
- `prior_scale = 0.5`: Tighten shock variance

**Effect**: Keeps sampler near plausible trajectories, prevents boundary concentration.

**Status**: Experimental (Step 64 proposal).

---

## Diagnostics

### Gate Activation Diagnostics

**1. Gate Share**:
```julia
gate_share = mean(gate_mask)
println("Gate activation rate: $(round(100*gate_share, digits=1))%")
# Target: ~10%
```

**2. Episode Statistics**:
```julia
episodes = identify_episodes(findall(gate_mask))
ep_lengths = [ep_end - ep_start + 1 for (ep_start, ep_end) in episodes]

println("Number of episodes: ", length(episodes))
println("Mean episode length: ", mean(ep_lengths))
println("Episode length range: ", extrema(ep_lengths))
# Typical: 3-10 episodes, mean length ~8 periods, range [4, 20]
```

**3. Concentration in High-Volatility Window**:
```julia
high_vol_window = 80:120  # Known from synthetic data
gate_share_highvol = mean(gate_mask[high_vol_window])

println("Gate share in high-vol window: $(round(100*gate_share_highvol, digits=1))%")
# Target: >70% (most high-vol periods should activate)
```

---

### Likelihood Diagnostics

**1. Loglik by Regime**:
```julia
loglik_rom_periods = sum(loglik_rom[.!gate_mask])
loglik_surr_periods = sum(loglik_surr[gate_mask])
loglik_total = loglik_rom_periods + loglik_surr_periods

println("Loglik (ROM periods): ", loglik_rom_periods)
println("Loglik (Surrogate periods): ", loglik_surr_periods)
println("Total loglik: ", loglik_total)

# Check: Should not be dominated by one regime
# If |loglik_surr| >> |loglik_rom|, may have scale issue
```

**2. Observation Fit by Regime**:
```julia
rmse_rom = sqrt(mean(abs2, y_obs[:, .!gate_mask] .- y_pred_rom[:, .!gate_mask]))
rmse_surr = sqrt(mean(abs2, y_obs[:, gate_mask] .- y_pred_surr[:, gate_mask]))

println("RMSE (ROM periods): ", rmse_rom)
println("RMSE (Surrogate periods): ", rmse_surr)
# Surrogate should have similar or slightly better RMSE
```

**3. Shock Magnitude by Regime**:
```julia
shock_mag_rom = mean(abs, ε[:, .!gate_mask])
shock_mag_surr = mean(abs, ε[:, gate_mask])

println("Mean |ε| (ROM periods): ", shock_mag_rom)
println("Mean |ε| (Surrogate periods): ", shock_mag_surr)
# Surrogate periods should have larger shocks (by construction)
```

---

## Troubleshooting

### Issue 1: Gate Activates Too Frequently (>20%)

**Symptoms**: Gate share >20%, excessive surrogate usage

**Causes**:
1. Threshold too low
2. ROM baseline poor (large forecast errors)
3. Shock prior variance too large

**Fix**:
```julia
# Recalibrate threshold with higher quantile
threshold = quantile(scores, 0.95)  # Instead of 0.90 → target 5% gate share

# Check ROM accuracy
rmse_rom = sqrt(mean(abs2, y_synth .- y_rom))
println("ROM RMSE: ", rmse_rom)  # Should be <0.15; if not, check ROM computation
```

---

### Issue 2: Gate Never Activates (<1%)

**Symptoms**: Gate share <1%, surrogate unused

**Causes**:
1. Threshold too high
2. Shocks too small (increase shock scale in synthetic data)
3. Score weights miscalibrated

**Fix**:
```julia
# Lower threshold
threshold = quantile(scores, 0.85)  # Target 15% gate share

# Increase shock scale in calibration
ε[80:120, :] = 5.0 * randn(41, n_shocks)  # Instead of 3.0

# Reweight score components
w_ε = 0.7  # Emphasize shocks
w_y = 0.3
```

---

### Issue 3: Posterior Concentrates at Boundaries

**Symptoms**: cprobp ≈ 0.95, cindp ≈ 0.01 (parameters at prior bounds)

**Causes**:
1. Likelihood scale mismatch (unconditional ROM + conditional surrogate)
2. Shock means absorbing misfit (|ε_mean| >> 1)
3. Gate-specific noise too tight

**Fix**:
```julia
# 1. Use gate-consistent likelihood (both conditional)
# See "Gate-Consistent Likelihood Scaling" above

# 2. Inflate observation noise in surrogate periods
obs_sigma_scale_surr = 2.0  # Relaxed

# 3. Use guided shock priors (Step 64)
guidance_scale = 0.5
prior_scale = 0.5
```

---

## Best Practices

### Gate Calibration

✅ **Do**:
- Calibrate on synthetic data with known high-volatility window
- Validate gate concentrates in high-vol periods (>70%)
- Use episode padding (k_pre=4, k_post=8, min_len=4)
- Save calibration for reproducibility

❌ **Don't**:
- Calibrate on real data (no ground truth for validation)
- Use extreme gate shares (<5% or >20%)
- Skip validation (always check activation patterns)

---

### Likelihood Implementation

✅ **Do**:
- Use gate-consistent scaling (both conditional or both unconditional)
- Inflate observation noise in surrogate periods (1.5-2x)
- Include shock prior in both regimes
- Monitor loglik by regime (diagnostic)

❌ **Don't**:
- Mix unconditional (ROM) + conditional (surrogate) likelihoods
- Use identical noise for both regimes (ignores surrogate error)
- Forget shock prior (leads to extreme ε values)

---

### Estimation

✅ **Do**:
- Start with standard hard gate (ROM vs Surrogate)
- Check posterior diagnostics (R-hat, ESS, ACF)
- Validate parameter recovery on synthetic data
- Document gate activation patterns

❌ **Don't**:
- Use experimental methods (gate-window sampling) for production
- Skip boundary diagnostics (cprobp, cindp)
- Ignore high ACF for curvature parameters (mixing issue)

---

## References

### Regime-Switching Methodologies

1. **Hamilton, J. D. (1989).** "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
   - Markov-switching models (soft gating ancestor)

2. **Guerrieri, L., & Iacoviello, M. (2015).** "OccBin: A toolkit for solving dynamic models with occasionally binding constraints easily." *Journal of Monetary Economics*, 70, 22-38.
   - Piecewise linear approach (alternative to hard gating)

3. **Binning, A., & Maih, J. (2017).** "Achieving the Limits of the Possible in DSGE Modeling." Norges Bank Working Paper.
   - Regime-switching DSGE estimation

---

### Empirical Bayes and Gate Calibration

4. **Efron, B., & Morris, C. (1973).** "Stein's Estimation Rule and Its Competitors—An Empirical Bayes Approach." *Journal of the American Statistical Association*, 68(341), 117-130.
   - Empirical Bayes foundations

5. **Carlin, B. P., & Louis, T. A. (2000).** *Bayes and Empirical Bayes Methods for Data Analysis* (2nd ed.). Chapman & Hall.
   - Empirical Bayes methodology

---

## Summary

**Hard-gate regime switching**:
- ✅ Binary switch (ROM vs Surrogate) based on data-driven threshold
- ✅ Empirical Bayes calibration (10% gate activation target)
- ✅ Episode padding for smooth transitions
- ✅ Gate-consistent likelihood (both conditional)
- ✅ Observation noise relaxation (accounts for surrogate error)

**Calibration**: Synthetic data with known high-volatility window
**Performance**: ~60% overhead vs ROM-only (10% gate share)

**Experimental extensions**:
- ⚠️ Gate-window shock sampling (reduced dimensionality)
- ⚠️ Guided shock priors (boundary issue mitigation)
- ⚠️ Robust measurement likelihood (Student-t)

**Next steps**:
- See `INVERSION_FILTER.md` for shock inference methodology
- See `NEURAL_NETWORK_SURROGATES.md` for surrogate training
- See `CURRENT_APPROACHES.md` for validated vs experimental methods
- See `PIPELINE_GUIDE.md` for complete workflow

---

*Last updated: January 2026*
*Feature status: Production-ready ✅*
