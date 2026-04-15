# Current Approaches: Stable Baseline vs Experimental Methods

## Overview

This document describes the **two main approaches** currently available in the regime-switching estimation framework:

1. **✓ ROM1+Delta Surrogate** (Stable Baseline) - Production-ready, validated
2. **⚠ Gate-Window Shock Sampling** (Experimental) - Active research, addressing posterior issues

Both approaches share the same core architecture (ROM + Surrogate + Gating) but differ in shock treatment and likelihood specification.

---

## Table of Contents

1. [Stable Baseline: ROM1+Delta Surrogate](#stable-baseline-rom1delta-surrogate)
2. [Experimental: Gate-Window Shock Sampling](#experimental-gate-window-shock-sampling)
3. [Comparison Table](#comparison-table)
4. [When to Use Each Approach](#when-to-use-each-approach)
5. [Performance Benchmarks](#performance-benchmarks)

---

## Stable Baseline: ROM1+Delta Surrogate

### Status

✓ **Validated and Production-Ready**

- Developed and refined in Steps 35-45
- Workflow documented in [RS_step_43](../active_steps/RS_step_43_workflow.md)
- Stable SEP dataset generation with convergence checking
- ~3% RMSE improvement over ROM1 in high-volatility windows

### Key Characteristics

**1. ROM Baseline**
- ROM1 (first-order perturbation) computed at fixed paper calibration
- `--rom-mode=baseline` ensures AD compatibility
- ROM provides structural backbone, surrogate learns residuals

**2. Residual Surrogate**
- Trains on `Δ = Y_sep - Y_rom1` (not full Y_sep)
- Observable-only output (`--obs-only`)
- Architecture: 256 → 128 neurons (2 layers)
- 400-500 training epochs

**3. Stable SEP Dataset**
- Marks solved vs failed samples in metadata
- Uses only successfully converged SEP solutions for training
- Typical success rate: 80-90% with proper shock scaling
- SEP parameters: horizon=10, nodes=3, tolerance=1e-5

**4. Gate Calibration**
- Empirical Bayes threshold selection
- Target: ~10% gate activation (`--gate-share=0.10`)
- Combines shock magnitude + ROM forecast errors
- Episode padding: `k_pre=4`, `k_post=8`, `min_len=4`

**5. Estimation**
- Fixed shocks (conditional likelihood)
- Linear ROM Kalman filter in non-gate periods
- ROM1 + Surrogate in gate periods
- HMC with NUTS sampler (Turing.jl)

### Data Artifacts (Reference Implementation)

**Dataset**: `data/hlt_sep_surrogate_dataset_20260103_180108/`
- Settings: shock_scale=0.25, horizon=10, theta_samples=50, sample_length=80
- ROM baselines: ROM1 and ROM2 stored
- Success metadata tracked

**Trained Surrogate**: `hlt_sep_surrogate_trained_rom1_resid_obs.jls`
- ROM1 residual mode
- Obs-only output
- Validation RMSE: ~0.02-0.04 (observable-dependent)

**Synthetic Data**: `data/hlt_sep_surrogate_synth_20260104_161020/`
- 200 periods, high-vol window 80-120 (3× shock scaling)
- All observables and shocks stored
- Used for gate calibration and illustration

### One-Command Reproduction

```bash
# Illustration with stable baseline
julia --project=. scripts/hlt_regime_switching_illustration.jl \
  data/hlt_sep_surrogate_synth_20260104_161020/hlt_sep_synth_data.jls \
  --surrogate=data/hlt_sep_surrogate_dataset_20260103_180108/hlt_sep_surrogate_trained_rom1_resid_obs.jls \
  --irf-method=extended_path --irf-periods=10
```

### Performance Metrics

**Approximation Accuracy** (vs SEP on synthetic data):
- ROM1 RMSE (full): 0.095
- ROM1 RMSE (high-vol): 0.142
- ROM1+Surrogate RMSE (full): 0.091
- **ROM1+Surrogate RMSE (high-vol): 0.107** ← 3.5% improvement

**Speed** (per likelihood evaluation):
- Linear ROM1 only: ~5ms
- ROM1 + Surrogate (10% gate): ~8ms
- Overhead: 60% (acceptable for 3-4% accuracy gain)

**Estimation Stability**:
- Posterior R-hat < 1.01 (converged)
- ESS > 400 (good mixing)
- No systematic boundary issues (when priors are well-specified)

### When to Use

**Recommended for**:
- Production estimation of HLT or similar SW07-based models
- Reproducible research results
- Standard validation exercises
- Teaching/demonstration purposes

**Workflow**: Follow [RS_step_43](../active_steps/RS_step_43_workflow.md) for complete pipeline.

### Key Steps in Development

- **Step 35-37**: Grid-based dataset generation with IRF augmentation
- **Step 42**: Stable SEP dataset sampling with convergence tracking
- **Step 43**: Consolidated workflow and next steps
- **Step 44-45**: Marked solved samples, filter training

See [DEVELOPMENT_PHASES.md](DEVELOPMENT_PHASES.md) Phase 5 for detailed evolution.

---

## Experimental: Gate-Window Shock Sampling

### Status

⚠ **Active Research - Not Yet Validated**

- Developed in Steps 61-64 (January 2026)
- Addresses posterior boundary issues observed in hard-gate estimation
- 8 ranked improvement proposals in [RS_step_64](../active_steps/RS_step_64_sampling_improvements.md)
- Requires further validation and tuning

### Motivation

Hard-gate manual runs (Step 61-63) revealed **likelihood-scale problems**:
- Posterior concentrates at parameter boundaries (cprobp ≈ 0.95, cindp ≈ 0.01)
- Shock means absorb mismatch in forced gate window (|ε_mean| ~ 10-80)
- Very negative log-likelihoods (~-1e7) suggesting scale issues
- High autocorrelation for curvature parameters (ACF1 ≈ 0.99)

**Root cause**: Conditional vs unconditional likelihood mismatch when shocks are treated as parameters.

### Key Innovations

**1. Gate-Consistent Likelihood Scaling** (✓ Implemented in Step 63)
```julia
# OLD (Step 61-62): Mixed unconditional + conditional likelihoods
loglik_linear = kalman_filter(y, θ)  # Unconditional (integrates over ε)
loglik_gate = surrogate_loglik(y, θ, ε)  # Conditional (fixed ε)

# NEW (Step 63): Both conditional when sampling shocks
loglik_linear = conditional_kalman(y, θ, ε)  # Conditional on ε
loglik_gate = surrogate_loglik(y, θ, ε)      # Conditional on ε
```

**2. Gate-Window Shock Sampling** (Proposed in Step 64)
```bash
--sample-shocks --shock-sample-window=gate
```
- Sample shocks **only inside gate window** (e.g., periods 80-120)
- Fix shocks in non-gate periods (reduce dimensionality)
- Allows surrogate fit to improve where it matters
- Reduces conditional loglik collapse

**3. Observation Noise Relaxation** (Proposed)
```bash
--obs-sigma-scale=2.0
```
- Inflate measurement noise in gate periods
- Accounts for surrogate approximation error
- Prevents huge penalties from small prediction errors
- Future: gate-specific noise scale

**4. Guided Shocks** (Proposed)
```bash
--shock-guidance=kalman --shock-guidance-scale=0.5 --shock-prior-scale=0.5
```
- Anchor shock mean to linear Kalman filter path
- Shrink shock prior variance
- Prevents shock posteriors from exploding
- Keeps sampler near plausible trajectories

**5. Non-Centered Shock Parameterization** (Proposed)
- Block updates: NUTS for θ, Elliptical Slice for ε
- Or: draw ε conditional on θ via inversion filter
- Higher ESS, less boundary stickiness

**6. Shorter Forced Gate Window** (Diagnostic)
```bash
--hard-gate=90:110  # Instead of 80:120
```
- Reduces penalty mass
- Clearer regime contrast
- Diagnostic tool to isolate issues

**7. Tempered Likelihood** (Proposed)
- Start with likelihood power < 1 during warmup
- Ramp to 1 during sampling
- More stable adaptation
- Fewer extreme θ moves

**8. Robust Measurement Likelihood** (Proposed)
- Student-t likelihood for observables (gate only)
- Reduces over-penalization of surrogate errors
- More robust to outliers

### Current Implementation Status

| Feature | Status | Flag | Step |
|---------|--------|------|------|
| Gate-consistent likelihood | ✓ Implemented | (automatic) | 63 |
| Gate-window shock sampling | ⚠ Proposed | `--sample-shocks --shock-sample-window=gate` | 64 |
| Obs noise relaxation | ⚠ Proposed | `--obs-sigma-scale=X` | 64 |
| Guided shocks | ⚠ Proposed | `--shock-guidance=kalman` | 64 |
| Non-centered parameterization | 🔄 Future | N/A | 64 |
| Tempered likelihood | 🔄 Future | N/A | 64 |
| Robust likelihood | 🔄 Future | N/A | 64 |

### Suggested Diagnostic Command (Step 64)

```bash
julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
  data/hlt_sep_surrogate_dataset_stable_shock03_20260117_002121/hlt_sep_surrogate_trained_rom1_resid_obs.jls \
  data/hlt_sep_surrogate_synth_20260111_232218/hlt_sep_synth_data.jls \
  --gate-calibration=data/hlt_sep_surrogate_synth_20260111_232218/gate_calibration_share010_auto_linf_yonly_filtered.jls \
  --gate-mode=hard --hard-gate=80:120 --gate-use-y=true --gate-use-eps=false \
  --gate-k-pre=4 --gate-k-post=8 --gate-min-len=4 \
  --sample-shocks --shock-sample-window=gate \
  --shock-guidance=kalman --shock-guidance-scale=0.5 --shock-prior-scale=0.5 \
  --obs-sigma-scale=2.0 \
  --samples=200 --chains=1
```

### Diagnostics to Compare

After running gate-window sampling vs baseline:
- **Loglik at posterior mean**: Should be closer to zero (less negative)
- **Gate share**: Fraction of non-zero hard gate mask
- **Parameter bounds**: cprobp, cindp should move away from boundaries
- **ACF1/ESS for curvp**: Should decrease (better mixing)
- **Shock magnitude**: mean |ε| should be reasonable (~1-5, not 10-80)

### When to Use

**Experimental - Use for**:
- Diagnosing posterior boundary concentration
- Research on improving gate fit
- Understanding likelihood scale issues
- Developing new sampling strategies

**Not recommended for**:
- Production estimation (not yet validated)
- Reproducible research (methods still evolving)
- Teaching (too complex for introduction)

**Next steps**: Validate on multiple models, compare to stable baseline, document best practices.

### Key Steps in Development

- **Step 61**: Manual hard-gate window implementation
- **Step 62**: Manual gate sampling investigation
- **Step 63**: Gate-linear loglik alignment (conditional likelihood fix)
- **Step 64**: Sampling improvements and alternatives (8 proposals)

See [RS_step_64](../active_steps/RS_step_64_sampling_improvements.md) for full details.

---

## Comparison Table

| Aspect | ROM1+Delta Baseline | Gate-Window Sampling |
|--------|---------------------|---------------------|
| **Status** | ✓ Stable, validated | ⚠ Experimental |
| **Development** | Steps 35-45 | Steps 61-64 |
| **Shock Treatment** | Fixed (conditional) | Sampled in gate window |
| **Likelihood** | Conditional linear + surrogate | Gate-consistent conditional |
| **Speed** | Fast (~8ms/eval) | Moderate (~10-15ms/eval) |
| **Use Case** | Production estimation | Boundary issue diagnosis |
| **Posterior Behavior** | Stable (well-specified priors) | Under investigation |
| **Documentation** | Complete | In progress |
| **Validation** | ✓ Synthetic + real data | ⚠ Diagnostic runs only |
| **RMSE Improvement** | ~3% in high-vol | TBD |
| **HMC Mixing** | Good (ESS > 400) | Under investigation |
| **Boundary Issues** | Rare (with good priors) | Addressing actively |
| **Gate Calibration** | Empirical Bayes (auto) | Manual hard-gate (diagnostic) |
| **Obs Noise** | Standard (fixed σ) | Relaxed (inflated σ in gate) |
| **Shock Prior** | Standard N(0, Σ) | Guided (anchored to filter) |
| **Implementation Complexity** | Low | High |
| **Recommended?** | ✓ Yes | ⚠ Research only |

---

## When to Use Each Approach

### Choose ROM1+Delta Baseline if:

✓ You want **production-ready estimation** with validated performance
✓ You need **reproducible results** for publication or replication
✓ You're **teaching or demonstrating** the framework
✓ Your priors are **well-specified** (no systematic boundary issues)
✓ You're working with the **HLT model** or similar SW07-based specifications
✓ You want **standard workflow** with clear documentation

**Follow**: [RS_step_43](../active_steps/RS_step_43_workflow.md) workflow
**Tutorial**: [Full Pipeline](../tutorials/FULL_PIPELINE.md)

### Choose Gate-Window Sampling if:

⚠ You observe **posterior concentration at boundaries** (cprobp, cindp)
⚠ You're **diagnosing likelihood scale issues**
⚠ You're **researching improved sampling strategies**
⚠ You want to **understand shock absorption** in gate periods
⚠ You're **developing extensions** to the framework
⚠ You have time for **experimental validation**

**Caution**: Not yet validated for production use
**Reference**: [RS_step_64](../active_steps/RS_step_64_sampling_improvements.md)

### Hybrid Approach (Future)

Potential workflow:
1. Start with **ROM1+Delta baseline** for initial estimation
2. Diagnose posterior with chain report
3. If boundary issues appear, try **gate-window sampling** diagnostics
4. Compare logliks, ESS, parameter recovery
5. Document findings and contribute back

This hybrid validation workflow will help mature the experimental methods.

---

## Performance Benchmarks

### ROM1+Delta Baseline

**Dataset Generation** (theta_samples=50, sample_length=80):
- Total time: ~15-30 minutes (parallelized)
- SEP success rate: 85%
- ROM computation: ~2 minutes
- Checkpoint frequency: Every 10 samples

**Surrogate Training** (epochs=400, hidden=256→128):
- Training time: ~5-10 minutes
- Validation RMSE: 0.02-0.04 (observable-dependent)
- Convergence: Typically by epoch 300

**Synthetic Data Generation** (200 periods):
- Time: ~30-60 seconds
- SEP solves: 200 × (1 + IRF shocks if requested)

**Gate Calibration**:
- Time: <10 seconds
- Threshold selection: Quantile-based

**Illustration** (ROM1/ROM2/SEP/Surrogate comparison):
- Time: ~30-60 seconds
- Outputs: 3 PDFs (series, errors, IRFs)

**HMC Estimation** (1000 samples, 4 chains):
- Time: ~20-40 minutes
- Loglik evaluations: ~4000-8000
- Gradient computations: ~1000-2000
- Acceptance rate: 0.85-0.95 (NUTS adaptive)

**Total Workflow** (first run):
- ~1 hour for mini pipeline (reduced settings)
- ~2-3 hours for production pipeline (full settings)

### Gate-Window Sampling (Preliminary)

**Additional Overhead**:
- Shock sampling: +2-3ms per eval (gate periods only)
- Guided shock computation: +1ms per eval
- Total: ~10-15ms per eval (vs 8ms baseline)

**HMC Diagnostics** (200 samples, 1 chain):
- Time: ~10-15 minutes
- Acceptance rate: 0.80-0.90 (varies with settings)
- ESS: Under investigation (goal: >400)

**Not yet benchmarked**:
- Full estimation run (1000+ samples)
- Multi-chain convergence
- Parameter recovery on synthetic data

---

## Recommendations

### For Most Users: Start with ROM1+Delta Baseline

1. Follow the [QUICKSTART](../QUICKSTART.md) tutorial
2. Generate production-scale dataset (50+ theta samples)
3. Train surrogate with 400-500 epochs
4. Run estimation with empirical Bayes gate calibration
5. Diagnose with chain report

If you encounter posterior boundary issues:
- First, check prior specification (too tight? wrong mode?)
- Review shock scaling (too large? → increase prior variance)
- Try longer HMC warmup (more adaptation)

### For Researchers: Explore Both Approaches

1. Establish ROM1+Delta baseline results
2. Document any boundary or mixing issues
3. Run gate-window sampling diagnostics (Step 64 command)
4. Compare logliks, ESS, parameter recovery
5. Report findings in [active_steps/](../active_steps/)

**Contribute**: If you find improved settings or discover new issues, document in a new RS_step file following the established format.

### For Developers: Extend with Care

The experimental methods are evolving. Before implementing extensions:
- Review [RS_step_64](../active_steps/RS_step_64_sampling_improvements.md) for current proposals
- Check [DEVELOPMENT_PHASES.md](DEVELOPMENT_PHASES.md) for lessons learned
- Validate on synthetic data first (known ground truth)
- Document decision rationale and alternatives considered

---

## Future Directions

### Short Term (Next 3-6 Months)

1. **Validate gate-window shock sampling** on multiple models
2. **Benchmark** experimental methods vs baseline
3. **Document best practices** for boundary issue diagnosis
4. **Implement** observation noise relaxation (gate-specific)
5. **Test** guided shock strategies

### Medium Term (6-12 Months)

1. **Develop** non-centered shock parameterization
2. **Implement** tempered likelihood warmup
3. **Explore** robust measurement likelihood (Student-t)
4. **Extend** to other models (RBC, NK variants)
5. **Compare** to alternative OBC methods (OccBin, piecewise linear)

### Long Term (1+ Years)

1. **Soft gating** with mixture likelihoods (revisit early experiments)
2. **Adaptive gating** that learns from data (beyond empirical Bayes)
3. **Multi-regime** extensions (ZLB + crisis + normal)
4. **Heterogeneous agents** with surrogate methods
5. **Real-time** estimation and forecasting

---

## Summary

**Two approaches available**:
1. **✓ ROM1+Delta Baseline**: Stable, validated, recommended for production
2. **⚠ Gate-Window Sampling**: Experimental, addressing boundary issues, research only

**Key difference**: Shock treatment and likelihood specification

**Recommendation**: Start with baseline, use experimental for diagnostics

**Next steps**:
- Baseline users → [Full Pipeline Tutorial](../tutorials/FULL_PIPELINE.md)
- Researchers → [RS_step_64](../active_steps/RS_step_64_sampling_improvements.md)
- Developers → [DEVELOPMENT_PHASES.md](DEVELOPMENT_PHASES.md)

---

**Questions?** See [../tutorials/TROUBLESHOOTING.md](../tutorials/TROUBLESHOOTING.md) or [../README.md](../README.md) "Getting Help"
