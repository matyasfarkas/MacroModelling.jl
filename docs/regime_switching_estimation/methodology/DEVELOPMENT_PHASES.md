# Development Phases: Evolution of the Regime-Switching Framework

This document traces the development history across 64 documented steps (January-January 2026), highlighting key decisions, lessons learned, and how the current stable baseline emerged.

---

## Table of Contents

1. [Timeline Overview](#timeline-overview)
2. [Phase 1: Infrastructure & Foundations](#phase-1-infrastructure--foundations-steps-01-11)
3. [Phase 2: Surrogate Discovery](#phase-2-surrogate-discovery-steps-12-19)
4. [Phase 3: HMC Integration](#phase-3-hmc-integration-steps-20-28)
5. [Phase 4: SEP Stabilization](#phase-4-sep-stabilization-steps-29-33)
6. [Phase 5: Scalable Datasets](#phase-5-scalable-datasets-steps-34-45)
7. [Phase 6: Advanced Techniques](#phase-6-advanced-techniques-steps-46-64)
8. [Key Architectural Decisions](#key-architectural-decisions)
9. [Lessons Learned](#lessons-learned)

---

## Timeline Overview

| Phase | Steps | Duration | Focus | Status |
|-------|-------|----------|-------|--------|
| 1 | 01-11 | Days 1-15 | Infrastructure, OBC, gates | ✓ Complete |
| 2 | 12-19 | Days 16-30 | Surrogate training, validation | ✓ Complete |
| 3 | 20-28 | Days 31-45 | HMC integration, diagnostics | ✓ Complete |
| 4 | 29-33 | Days 46-55 | SEP solver tuning | ✓ Complete |
| 5 | 34-45 | Days 56-80 | Production datasets, baseline | ✓ **Stable Baseline** |
| 6 | 46-64 | Days 81-120 | Advanced gating, sampling | ⚠ **Active Research** |

**Current Recommended Approach**: Phase 5 stable baseline (Steps 35-45)
**Experimental Frontier**: Phase 6 gate-window sampling (Steps 61-64)

---

## Phase 1: Infrastructure & Foundations (Steps 01-11)

### Goals
- Establish scope and architectural decisions
- Implement OBC support in HLT model
- Build gate calibration infrastructure
- Integrate gating into estimation

### Key Steps

**Step 01: Scope Definition**
- **Decision**: Hard gating only (no soft mixture)
- **Rationale**: Simpler implementation, clearer interpretation
- **Alternative considered**: Soft gating with mixture weights (deferred)

**Step 02: OBC Model**
- **Implementation**: `Smets_Wouters_2007_HLT_obc.jl`
- **ZLB Specification**: `r = max(r_desired, 1.0)` (gross rate ≥ 1)
- **Challenge**: SEP solver must handle inequality constraints

**Step 03-04: Gate Calibration**
- **Approach**: Empirical Bayes on synthetic data
- **Score Function**: Combine shock magnitude + ROM forecast errors
- **Per-Period Likelihood**: Enable gated estimation

**Step 05-08: Integration**
- **Hybrid Likelihood**: Linear KF for normal times, surrogate for gates
- **First Runs**: Validation on small synthetic datasets
- **Issue Discovered**: Gate calibration sensitive to window definition

**Step 09-11: OBC Refinement**
- **Padding Parameters**: `k_pre`, `k_post`, `min_len` added
- **Smoke Pipeline**: End-to-end test with OBC
- **Outcome**: Basic framework functional

### Outcomes
✓ Core architecture established
✓ OBC support working
✓ Gate calibration functional
✗ Soft gating deferred (complexity vs benefit)

### Lessons Learned
1. **Hard gating is simpler and sufficient** - Soft mixture adds complexity without clear benefit in this application
2. **OBC requires careful SEP tuning** - Tolerance and line search critical
3. **Episode padding essential** - Prevents unstable gate sequences

---

## Phase 2: Surrogate Discovery (Steps 12-19)

### Goals
- Determine optimal surrogate training strategy
- Validate approximation quality
- Establish IRF comparison framework

### Key Steps

**Step 12-13: High-Volatility Episodes**
- **Motivation**: Surrogate most valuable when ROM fails
- **Synthetic Data**: Engineered high-vol windows (3× shock amplification)
- **Visualization**: Series and error plots added

**Step 14: ROM-Residual Breakthrough**
- **Key Innovation**: Train on `Δ = Y_sep - Y_rom1` instead of `Y_sep`
- **Rationale**:
  - Surrogate learns correction, not full dynamics
  - ROM provides structural backbone
  - Better generalization with smaller network
- **Implementation**: `--rom-residual=1` flag, ROM baselines stored in dataset
- **Outcome**: **Major improvement in surrogate efficiency**

**Step 14b: Obs-Only Output**
- **Discovery**: Full state output unnecessary for likelihood
- **Implementation**: `--obs-only` predicts 7 observables (not 26 states)
- **Benefit**: 2-3× faster training, smaller model, less overfitting

**Step 15-16: IRF Experiments**
- **Challenge**: IRF mismatch between surrogate and SEP
- **Root Cause**: Shock scaling inconsistency
- **Solution**: Align `shock_scaling` across dataset, surrogate, IRF
- **Windowed Correction**: Attempted localized surrogates (abandoned)

**Step 17: Obs-Only Validation**
- **Confirmation**: Obs-only output sufficient for estimation
- **RMSE**: ~3% improvement in high-vol windows
- **Decision**: Adopt obs-only as default

**Step 18-19: IRF Augmentation**
- **Idea**: Include IRF data in training (not just time series)
- **Outcome**: Marginal benefit, added complexity
- **Decision**: Keep time-series only (simpler)

### Outcomes
✓ **ROM-residual learning** - Key architectural choice
✓ **Obs-only output** - Speed and generalization
✓ IRF comparison framework
✗ IRF-augmented training (not worth complexity)
✗ Windowed corrections (abandoned)

### Lessons Learned
1. **Residual learning > full approximation** - Structure from ROM reduces surrogate burden
2. **Obs-only sufficient for likelihood** - No need to predict full state
3. **Shock scaling must be consistent** - Dataset, IRF, estimation
4. **Simpler is better** - IRF augmentation didn't justify complexity

---

## Phase 3: HMC Integration (Steps 20-28)

### Goals
- Integrate surrogate into HMC estimation
- Develop chain diagnostics
- Compare regime-switching vs linear estimation

### Key Steps

**Step 20-21: Data Management**
- **Volatility Window Audit**: Verify high-vol realization
- **Data Consolidation**: Organize datasets for reproducibility

**Step 22-23: Regime vs Linear Comparison**
- **Linear Baseline**: ROM1-only estimation (no surrogate)
- **Regime-Switching**: Gated hybrid likelihood
- **Metrics**: Log marginal likelihood proxy, posterior diagnostics

**Step 24-28: HMC Diagnostics**
- **100-Sample Runs**: Quick diagnostics for iteration
- **Manual HMC**: Reparameterization experiments
- **Linear ROM1 Switching**: Non-OBC baseline tests
- **Challenge**: Posterior boundary issues (cprobp, cindp)
- **Issue**: Parameter bounds concentration observed

### Outcomes
✓ HMC integration functional
✓ Diagnostic pipeline established
⚠ Boundary issues identified (addressed in Phase 6)

### Lessons Learned
1. **Diagnostic runs essential** - 100-sample quick checks before full estimation
2. **Compare to linear** - Regime-switching must improve on ROM1-only
3. **Watch parameter bounds** - Concentration suggests prior or likelihood issues

---

## Phase 4: SEP Stabilization (Steps 29-33)

### Goals
- Improve SEP convergence rates
- Establish stable solver settings
- Enable reliable dataset generation

### Key Steps

**Step 29-30: Non-OBC Issues**
- **Discovery**: SEP fails even without OBC (not just ZLB issue)
- **Cause**: Shock scaling too aggressive for some θ samples
- **Acceptance Tolerance**: Experimented with non-OBC tolerance

**Step 31-32: Convergence Tuning**
- **Line Search**: Stabilized Newton steps
- **Tolerance**: 1e-5 standard, 1e-4 permissive
- **Shock Scaling**: Reduced to 0.25 (from 0.5)
- **Outcome**: Success rate 80-90% (from 60-70%)

**Step 33: Convergence Knobs**
- **Documented Settings**: Horizon, tolerance, max iterations
- **Trade-offs**: Speed vs accuracy
- **Dataset Regeneration**: With stable settings

### Outcomes
✓ Reliable SEP convergence (85-95% success)
✓ Documented solver knobs
✓ Shock scaling guidelines

### Lessons Learned
1. **Shock scaling critical** - 0.25 sweet spot for HLT model
2. **Line search essential** - Stabilizes Newton in nonlinear regions
3. **Track success rates** - Should be >80% for production datasets

---

## Phase 5: Scalable Datasets (Steps 34-45)

**This phase produced the current stable baseline approach**

### Goals
- Generate large-scale training datasets
- Systematically grid parameters and shocks
- Establish production workflow

### Key Steps

**Step 34-37: Grid-Based Generation**
- **Grid6 Dataset**: 6 theta samples × extensive shock coverage
- **IRF Augmentation**: Attempted (minimal benefit, kept simple)
- **Checkpointing**: Resume-from-failure capability
- **Grid8**: Extended to 8 samples

**Step 42: Stable Prefix Critical**
- **Problem**: Some samples converge only partially (50/80 periods)
- **Solution**: `--stable-prefix --stable-min-periods=40`
  - Marks successful vs failed samples
  - Filters training data to converged episodes only
- **Metadata**: `theta_success`, `theta_stable_periods` tracked
- **Outcome**: **Training quality dramatically improved**

**Step 43: Workflow Summary**
- **Consolidated Pipeline**: Dataset → Train → Synthetic → Gate → Estimate
- **Command Templates**: Copy-paste production settings
- **Documentation**: Scripts README, diagnostics README
- **Status**: **This is the stable baseline workflow**

**Step 44-45: Filter Training**
- **Marked Solved Samples**: Use only high-quality SEP solutions
- **SEP Shock Scale**: Final tuning to 0.25 baseline
- **Success Metrics**: 85-95% convergence typical

### Outcomes
✓ **Stable Baseline Established** (Steps 35-45)
✓ Production-scale datasets (50-100 theta samples)
✓ Robust success tracking
✓ Documented workflow ([RS_step_43](../active_steps/RS_step_43_workflow.md))

### Lessons Learned
1. **Filter failed samples** - Only train on converged SEP solutions
2. **Metadata crucial** - Track success rates, stable periods
3. **Checkpointing saves time** - Resume from interruptions
4. **Document working pipeline** - Step 43 is reference implementation

---

## Phase 6: Advanced Techniques (Steps 46-64)

**Active research - experimental methods**

### Goals
- Improve gate fit in forced windows
- Address posterior boundary issues
- Explore advanced sampling strategies

### Key Steps

**Step 46-56: Synthetic Data & Illustration**
- **SEP Shock Scaling**: Match dataset generation
- **Gate Use-Y-Only**: Experiment with observable-only gate scores
- **Increased Shocks**: Higher volatility windows
- **Gate Window Sampling**: Shock sampling in gate periods only
- **100-Draw Runs**: Statistical validation
- **Forecast Error Diagnostics**: Per-period surrogate accuracy

**Step 57-60: Performance**
- **Large Pilot Runs**: Speed benchmarks
- **Inversion Filter**: Alternative to Kalman
- **Hybrid Linear KF**: Optimize inference speed

**Step 61-63: Manual Gating**
- **Manual Hard-Gate**: Force gate in specific window (80-120)
  - Diagnostic tool, not production
- **Observation**: Posterior concentrates at boundaries (cprobp ≈ 0.95)
- **Root Cause**: Likelihood scale mismatch (conditional vs unconditional)
- **Step 63 Fix**: Gate-consistent likelihood scaling
  - Use conditional linear KF when shocks are parameters
  - Aligns with conditional surrogate likelihood

**Step 64: Sampling Improvements**
- **Problem**: Even with likelihood fix, boundary issues persist
- **Diagnosis**: Shocks absorb model misfit in forced gate window
- **8 Proposed Improvements**:
  1. Gate-consistent likelihood (✓ implemented Step 63)
  2. Gate-window shock sampling (sample ε only in gate)
  3. Observation noise relaxation (inflate σ_obs in gate)
  4. Guided shocks (anchor to KF path)
  5. Non-centered parameterization
  6. Shorter forced gate window (diagnostic)
  7. Tempered likelihood (warmup)
  8. Robust measurement likelihood (Student-t)
- **Status**: Proposals documented, validation ongoing

### Outcomes
✓ Performance characterized (ROM1+surrogate ~60% overhead)
✓ Likelihood scaling fixed (Step 63)
⚠ Boundary issues partially addressed
⚠ Gate-window sampling proposed (Step 64)
🔄 Experimental methods under validation

### Lessons Learned
1. **Manual gates useful for diagnostics** - But not production (too restrictive)
2. **Likelihood scale matters** - Conditional vs unconditional must be consistent
3. **Shocks can absorb misfit** - Need regularization (guidance, priors)
4. **Boundary concentration** ≠ **sampler failure** - Often a likelihood or prior issue

---

## Key Architectural Decisions

### Decision 1: Hard Gating (Step 01)
**Chosen**: Binary regime switch
**Alternative**: Soft mixture with weights
**Rationale**: Simplicity, clear interpretation
**Status**: Validated, stable

### Decision 2: ROM-Residual Learning (Step 14)
**Chosen**: Train on `Δ = Y_sep - Y_rom1`
**Alternative**: Train directly on `Y_sep`
**Rationale**: Smaller network, better generalization, ROM provides structure
**Status**: **Key innovation, widely adopted**

### Decision 3: Obs-Only Output (Step 14b, 17)
**Chosen**: Predict 7 observables only
**Alternative**: Full state (26 dimensions)
**Rationale**: Sufficient for likelihood, faster, less overfitting
**Status**: Default for estimation

### Decision 4: Baseline ROM Mode (Step 14)
**Chosen**: ROM computed once at paper calibration
**Alternative**: Recompute per θ (theta mode)
**Rationale**: AD compatibility, speed
**Status**: Required for HMC

### Decision 5: Empirical Bayes Gates (Step 03-04)
**Chosen**: Data-driven threshold calibration
**Alternative**: Manual thresholds
**Rationale**: Automatic, adaptive to model/data
**Status**: Production standard

### Decision 6: Fixed Shocks (Step 64 revisiting)
**Chosen**: Shocks not sampled (baseline)
**Alternative**: Sample shocks (full Bayesian)
**Rationale**: Speed, lower dimensionality
**Status**: Default; sampling experimental (Phase 6)

### Decision 7: Stable Prefix (Step 42)
**Chosen**: Filter to converged SEP samples
**Alternative**: Use all samples (including partial)
**Rationale**: Training quality
**Status**: **Critical for robust surrogates**

---

## Lessons Learned

### Technical

1. **Residual learning >> full approximation**
   - ROM provides inductive bias
   - Surrogate focuses on correction
   - Smaller networks generalize better

2. **Quality over quantity in training data**
   - 50 good samples > 100 mixed-quality samples
   - Filter failed SEP solutions (`--stable-prefix`)
   - Track success metadata

3. **Shock scaling is a goldilocks problem**
   - Too small (0.1): Little nonlinearity, easy SEP, nothing to learn
   - Just right (0.25): Nonlinear effects, stable convergence
   - Too large (0.5): SEP failures, unstable training data

4. **Consistency across pipeline**
   - `shock_scaling` must match: dataset, synthetic, IRF
   - ROM mode must be `baseline` for estimation
   - Metadata versioning prevents mismatches

5. **Diagnostics before production**
   - Quick 100-sample runs for iteration
   - Compare to linear ROM1 baseline
   - Check ESS, R-hat, boundary concentration

6. **Likelihood scale matters**
   - Conditional vs unconditional must align
   - Particularly critical when sampling shocks
   - Step 63 fix was essential

### Methodological

7. **Hard gating works**
   - No need for soft mixture complexity
   - ~10% gate activation sufficient
   - Episode padding prevents instability

8. **Empirical Bayes calibration is robust**
   - Synthetic data with known high-vol window
   - Generalizes to real data
   - Threshold selection automatic

9. **Obs-only sufficient for estimation**
   - No need to predict full state
   - Kalman filter only needs observables
   - Dramatic speed improvement

10. **Manual gates useful for diagnostics**
    - Force specific windows to understand behavior
    - Not for production (too restrictive)
    - Reveals likelihood scale issues

### Workflow

11. **Document working configurations**
    - Step 43 provides reference implementation
    - Command templates prevent errors
    - Metadata tracking aids reproducibility

12. **Checkpoint long runs**
    - Dataset generation can resume
    - Prevents loss from interruptions
    - Enables incremental refinement

13. **Phase experiments, stabilize, move on**
    - Phases 1-5: Build stable baseline
    - Phase 6: Experimental extensions
    - Don't mix experimental in production

14. **Validate each stage**
    - Dataset: Check success rates
    - Surrogate: Validate RMSE
    - Estimation: Chain diagnostics
    - Don't proceed if stage fails

### Research

15. **Simple baselines first**
    - ROM1-only estimation before regime-switching
    - Fixed shocks before sampled shocks
    - Standard gates before experimental

16. **Understand failure modes**
    - Boundary concentration → prior or likelihood issue
    - Poor mixing → warmup or reparameterization
    - SEP failures → shock scaling or tolerance

17. **Document experiments, even failures**
    - Windowed corrections tried (abandoned)
    - IRF augmentation (minimal benefit)
    - Lessons prevent repeating mistakes

---

## Current Status (January 2026)

### Production-Ready
- **Stable Baseline** (Phase 5): ROM1+delta surrogate
  - Workflow: [RS_step_43](../active_steps/RS_step_43_workflow.md)
  - Performance: ~3% RMSE improvement, 60% speed overhead
  - Recommended for: Production estimation, reproducible research

### Experimental
- **Gate-Window Sampling** (Phase 6, Steps 61-64)
  - Status: Proposals documented, validation ongoing
  - Use case: Diagnosing boundary issues, research
  - Not recommended for: Production

### Deprecated/Abandoned
- Soft gating (Step 10 - too complex)
- Windowed corrections (Step 16 - marginal benefit)
- IRF-augmented datasets (Step 18-19 - complexity not justified)
- Theta ROM mode for estimation (too slow)

---

## Future Directions

Based on Phase 6 exploration:

**Short Term** (Next 3-6 months):
- Validate gate-window shock sampling
- Implement observation noise relaxation
- Test guided shock strategies

**Medium Term** (6-12 months):
- Non-centered shock parameterization
- Tempered likelihood warmup
- Extend to other DSGE models

**Long Term** (1+ years):
- Revisit soft gating with improved methods
- Adaptive gating beyond empirical Bayes
- Heterogeneous agent extensions

---

## For Researchers

### If Starting Fresh
1. Begin with **Phase 5 stable baseline**
2. Validate on synthetic data
3. Only explore Phase 6 if encountering specific issues (boundaries, mixing)

### If Extending
1. Review **Step 64 proposals** for current research frontier
2. Check [CURRENT_APPROACHES.md](CURRENT_APPROACHES.md) for experimental methods
3. Document new experiments in active_steps/ following Step format

### If Debugging
1. Check **relevant phase** for similar issues
2. Lessons learned may provide solutions
3. See [TROUBLESHOOTING.md](../tutorials/TROUBLESHOOTING.md) for systematic debugging

---

## Summary

**64 steps → 6 phases → 1 stable baseline**

The development progressed from basic infrastructure (Phase 1) through surrogate discovery (Phase 2) and integration (Phase 3), stabilized the solver (Phase 4), established production workflow (Phase 5), and is now exploring advanced techniques (Phase 6).

**Key Milestone**: Phase 5 (Steps 35-45) produced the **stable baseline** currently recommended for production use.

**Current Focus**: Phase 6 (Steps 61-64) addresses refinements for challenging posterior geometries.

**Next**: See [CURRENT_APPROACHES.md](CURRENT_APPROACHES.md) for detailed comparison of stable vs experimental methods.

---

**Questions?** See [../README.md](../README.md) "Getting Help" or review specific step files in main directory and [active_steps/](../active_steps/)
