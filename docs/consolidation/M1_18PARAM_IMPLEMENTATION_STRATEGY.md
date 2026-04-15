# M1: 18-Parameter Validation Implementation Strategy

**Created**: 2026-03-07
**Status**: In Progress
**Priority**: MUST-HAVE (Critical for journal submission)
**Timeline**: 6-8 weeks (Weeks 1-8 of implementation plan)

---

## Objective

Extend surrogate-based estimation from 3 parameters (shock standard deviations only) to 18 structural parameters, demonstrating that the method works for deep parameter estimation, not just shock inference.

**Success Criteria**:
- All 18 parameters recovered within 10% relative error
- All 90% credible intervals cover truth
- All $\hat{R} < 1.01$, ESS > 1000
- Shock recovery correlation > 0.7
- MCMC completes in < 8 hours

---

## 18 Parameter Set

Based on HLT 2016 and Smets-Wouters 2007, the following 18 parameters are selected:

### 1. Structural Preferences and Technology (4 params)
- `csigma`: Risk aversion (SW: ~1.5, range 0.5-4.0)
- `csigl`: Labor supply elasticity inverse (SW: ~2.0, range 0.5-4.0)
- `chabb`: Habit persistence (SW: ~0.6, range 0.3-0.9)
- `calfa`: Capital share (SW: ~0.24, range 0.15-0.35)

### 2. Price and Wage Rigidities (4 params)
- `cprobp`: Price stickiness (Calvo) (SW: ~0.6, range 0.5-0.95)
- `cindp`: Price indexation to past inflation (SW: ~0.47, range 0.01-0.99)
- `cprobw`: Wage stickiness (Calvo) (SW: ~0.81, range 0.5-0.95)
- `cindw`: Wage indexation to past inflation (SW: ~0.32, range 0.01-0.99)

### 3. Kimball Curvatures (2 params)
- `curvp`: Price markup curvature (HLT: baseline 10, range 1-150)
- `curvw`: Wage markup curvature (HLT: baseline 10, range 1-150)

### 4. Taylor Rule (3 params)
- `crpi`: Response to inflation (SW: ~1.5, Taylor principle 1.1-3.0)
- `cry`: Response to output gap (SW: ~0.06, range 0.01-0.5)
- `crr`: Interest rate smoothing (SW: ~0.88, range 0.5-0.95)

### 5. Shock Persistence (3 params)
- `crhoa`: Technology shock AR(1) (SW: ~0.998, range 0.8-0.999)
- `crhob`: Preference shock AR(1) (SW: ~0.58, range 0.3-0.95)
- `crhog`: Government spending AR(1) (SW: ~0.996, range 0.8-0.999)

### 6. Shock Standard Deviations (2 params)
- `z_ea`: Technology shock SD (SW: ~0.46, range 0.2-1.0)
- `z_eb`: Preference shock SD (SW: ~1.85, range 0.5-4.0)

---

## Implementation Phases

### Phase 1: Parameter Configuration Infrastructure (Week 1)

**Goal**: Extend existing parameter configuration system to support 18-parameter set

**Files to Modify**:
1. `scripts/hlt_surrogate/parameter_config.jl` - Add `:param_set_18` configuration
2. `scripts/hlt_surrogate/hlt_sep_surrogate_cli_utils.jl` - Support new param set in CLI

**Tasks**:
- [ ] Define `param_set_18` in `parameter_config.jl` with:
  - Parameter names (18 symbols)
  - Prior bounds (Dict of (lower, upper) pairs)
  - Baseline values (from SW2007 posterior modes)
- [ ] Add parameter-dependent ROM mode support (some params affect ROM dynamics)
- [ ] Test parameter configuration loads correctly

**Deliverable**: Parameter configuration module supporting 18-parameter set

---

### Phase 2: Dataset Generation (Weeks 2-3)

**Goal**: Generate 200 parameter vectors × 180 periods ≈ 36,000 training samples

**Approach**: Leverage existing `hlt_sep_surrogate_dataset_generate.jl` infrastructure

**Tasks**:
- [ ] Generate 200-point parameter grid (Sobol sequence or Latin hypercube)
- [ ] Run SEP solver for each parameter vector:
  ```bash
  julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
    --param-set param_set_18 \
    --theta-samples 200 \
    --samples-per-theta 180 \
    --sep-horizon 40 \
    --sep-order 1 \
    --sep-nnodes 3 \
    --output-dir .local_artifacts/hlt_18param_dataset
  ```
- [ ] Validate dataset:
  - Check all 200 parameter vectors solved successfully
  - Verify state variable ranges are plausible
  - Ensure observables match data moments
- [ ] Compute dataset statistics (means, std devs, ranges)

**Challenges**:
- **SEP solver failures**: Some parameter combinations may not solve (e.g., Taylor principle violations, explosive dynamics)
  - **Mitigation**: Accept 80% success rate (160/200 vectors), filter failed solves
- **ROM vs FOM consistency**: For parameter-dependent surrogates, ROM must be updated per θ
  - **Decision**: Use **baseline ROM** (fixed at calibrated params) for simplicity
  - **Justification**: Surrogate learns full correction, not just residual from varying ROM

**Deliverable**: HDF5 dataset with ~30,000 valid samples

---

### Phase 3: Surrogate Training (Week 4)

**Goal**: Train larger neural network (512-256-128 architecture) to handle 18D parameter space

**Network Architecture**:
```
Input: [state (22), shock (7), theta (18)] = 47 dimensions
Hidden: [512, 256, 128] (3 layers with ReLU)
Output: [observables (7)] = 7 dimensions
```

**Training Configuration**:
- Optimizer: Adam with learning rate 1e-3 → 1e-4 (decay schedule)
- Batch size: 512 (larger batches for stability with 18D input)
- Epochs: 500-1000 (early stopping on validation loss)
- Train/val split: 80/20
- Regularization: Dropout (0.1) + weight decay (1e-5)

**Tasks**:
- [ ] Adapt `scripts/hlt_sep_surrogate_train.jl` for 512-256-128 architecture
- [ ] Train surrogate on 18-param dataset
- [ ] Validate surrogate accuracy:
  - Target: RRMSE < 0.002 on validation set
  - Check error distribution across parameter space (no systematic bias)
  - Test extrapolation: Evaluate on parameters outside training grid
- [ ] Save trained surrogate bundle (frozen network + metadata)

**Deliverable**: Trained surrogate with RRMSE < 0.002

---

### Phase 4: Synthetic Data Generation (Week 5)

**Goal**: Create synthetic "truth" dataset with known parameter values for recovery test

**Tasks**:
- [ ] Select "true" parameter vector from interior of prior support (avoid boundaries)
- [ ] Generate 180-period synthetic data using SEP:
  ```julia
  true_params = [csigma=1.5, csigl=2.0, chabb=0.65, ...] # 18 values
  synthetic_data = simulate_sep(model; periods=180, params=true_params)
  ```
- [ ] Add measurement error (small, σ_obs = 0.01)
- [ ] Save synthetic data + true parameters for estimation

**Deliverable**: Synthetic dataset with known ground truth

---

### Phase 5: Estimation (Week 6-7)

**Goal**: Run HMC estimation with 18-parameter surrogate, recover true values

**Script**: `test/test_sw07_hlt_estimation_18param.jl`

**Configuration**:
- MCMC: 4,000 samples (2× baseline for more parameters)
- Warmup: 2,000 samples
- Sampler: NUTS with AutoZygote
- Priors: Same as used for dataset generation
- Filter: Regime-switching (ROM: Kalman, FOM: Surrogate)
- Gate: ZLB-based probability (as in 3-param case)

**Tasks**:
- [ ] Create estimation script (adapt from `test_sw07_hlt_estimation_real_surrogate.jl`)
- [ ] Run mode-finding (Nelder-Mead or LBFGS)
- [ ] Run MCMC (expect 6-8 hours on local machine)
- [ ] Monitor convergence:
  - Check $\hat{R}$ every 500 iterations
  - Track acceptance rate (target 60-70%)
  - Watch for divergent transitions
- [ ] Diagnose issues if MCMC fails:
  - Reduce step size if acceptance too low
  - Add more warmup if $\hat{R} > 1.05$
  - Check prior bounds if hitting boundaries

**Deliverable**: 4,000 posterior samples with diagnostics

---

### Phase 6: Analysis and Tables (Week 8)

**Goal**: Analyze results, create tables for paper

**Tasks**:
- [ ] Compute parameter recovery metrics:
  ```julia
  for each parameter θ_j:
    bias = mean(posterior) - truth
    rel_error = abs(bias / truth) * 100
    coverage = truth in quantile(posterior, [0.05, 0.95])
  ```
- [ ] Compute MCMC diagnostics:
  - $\hat{R}$ (Gelman-Rubin)
  - ESS (bulk and tail)
  - Acceptance rate
  - Number of divergent transitions
- [ ] Create **Table: 18-Parameter Recovery** (for paper Section 7):
  | Parameter | True | Post Mean | 90% CI | Abs Error (%) | Bias |
  |-----------|------|-----------|--------|---------------|------|
  | csigma    | 1.50 | 1.48      | [1.42, 1.55] | 1.3% | -0.02 |
  | ...       | ...  | ...       | ...    | ...  | ...  |
- [ ] Create **Table: 18-Parameter MCMC Diagnostics**:
  | Parameter | $\hat{R}$ | ESS (bulk) | ESS (tail) | Accept Rate |
  |-----------|-----------|------------|------------|-------------|
  | csigma    | 1.002     | 1847       | 1654       | 0.67        |
  | ...       | ...       | ...        | ...        | ...         |
- [ ] Create runtime report:
  - Offline cost: Dataset generation + surrogate training time
  - Online cost: Total MCMC time, time per iteration
  - Comparison: Speedup vs direct SEP (estimated 50-100×)
- [ ] Update paper:
  - Fill Section 7.X "18-Parameter Validation" (currently TBA at line 1163)
  - Add tables to main text
  - Discuss results: Which parameters hardest to identify? Any correlations?

**Deliverable**: Complete 18-parameter results for paper

---

## Risk Mitigation

### Risk 1: MCMC Doesn't Converge for 18 Parameters

**Symptoms**: $\hat{R} > 1.05$, low ESS, acceptance rate < 40%

**Mitigations**:
1. **Reduce parameter count**: Start with 10-12 most identifiable params, expand if successful
2. **Better initialization**: Use mode from 3-param case as starting point for shared params
3. **Stronger priors**: Tighten prior bounds around SW2007 posterior modes
4. **More warmup**: Increase warmup from 2,000 to 5,000 samples
5. **Parallel chains**: Run 4 chains in parallel, check cross-chain convergence

**Fallback**: If 18-param fails, demonstrate 10-12 param validation and discuss 18-param as future work

---

### Risk 2: Surrogate Accuracy Degrades with 18D Parameter Space

**Symptoms**: RRMSE > 0.005, systematic bias in certain parameter regions

**Mitigations**:
1. **Larger network**: Try 1024-512-256 architecture (even bigger)
2. **More data**: Increase to 400 parameter vectors (80,000 samples)
3. **Better sampling**: Use Latin hypercube instead of Sobol for better 18D coverage
4. **Ensemble**: Train 5 surrogates with different seeds, use average prediction

**Fallback**: Use lower-dimensional parameter subset (e.g., 12 params)

---

### Risk 3: SEP Dataset Generation Takes Too Long (200 params × 180 periods)

**Estimate**: 200 × 180 / 60 ≈ 600 SEP solves (if reusing solutions across time)

**Time per SEP solve**: ~30 seconds (from existing runs)

**Total time**: 600 × 30s ≈ 5 hours (acceptable on local machine)

**Mitigations if too slow**:
1. **Reduce periods**: Use 120 periods instead of 180 (still substantial)
2. **Reduce grid**: Use 150 parameter vectors instead of 200
3. **Parallelize**: Run on 4 cores (speedup 3-4×)

---

## File Structure

```
scripts/
  generate_18param_surrogate_dataset.jl       # NEW (draft created)
  hlt_surrogate/
    parameter_config.jl                       # MODIFY (add param_set_18)

test/
  test_sw07_hlt_estimation_18param.jl         # NEW (to create in Phase 5)

docs/consolidation/
  M1_18PARAM_IMPLEMENTATION_STRATEGY.md       # THIS FILE

.local_artifacts/
  hlt_18param_dataset_YYYYMMDD/               # Dataset output
    sep_dataset.h5
    metadata.json
  hlt_18param_surrogate_YYYYMMDD/             # Trained surrogate
    hlt_sep_surrogate_trained.jls
  hlt_18param_estimation_YYYYMMDD/            # Estimation results
    mcmc_chain.jls
    estimation_summary.jls
    posterior_estimates.csv
```

---

## Next Steps (Immediate)

1. **This week**:
   - [ ] Review and approve this strategy document
   - [ ] Extend `parameter_config.jl` with `param_set_18`
   - [ ] Test parameter configuration loads correctly

2. **Next week**:
   - [ ] Generate 200-point parameter grid
   - [ ] Run dataset generation overnight (expect 5-6 hours)
   - [ ] Validate dataset quality

3. **Weeks 3-4**:
   - [ ] Train surrogate (expect 2-3 hours on local machine)
   - [ ] Validate surrogate accuracy

4. **Weeks 5-8**:
   - [ ] Generate synthetic data
   - [ ] Run estimation (6-8 hours)
   - [ ] Analyze results and create tables

---

## Dependencies

**Julia Packages** (already in Project.toml):
- MacroModelling.jl (SEP solver)
- Turing.jl + Zygote.jl (HMC with AD)
- Flux.jl (neural network training)
- HDF5.jl, CSV.jl, DataFrames.jl (data I/O)
- MCMCChains.jl (diagnostics)

**New Dependencies Needed**:
- `Sobol.jl` - For low-discrepancy parameter sampling (add to Project.toml)
  ```julia
  ] add Sobol
  ```

**Compute Resources**:
- Local machine only (no cluster access)
- Estimated runtime breakdown:
  - Dataset generation: 5-6 hours
  - Surrogate training: 2-3 hours
  - Estimation MCMC: 6-8 hours
  - **Total**: ~15-20 hours compute time (spread over 8 weeks)

---

## Questions for User

1. Should we start with 10-12 parameters as intermediate step, or go straight to 18?
   - **Current plan**: Go straight to 18 per user instruction

2. If 18-param MCMC doesn't converge, what's acceptable fallback?
   - **Proposal**: 12-parameter validation (drop 6 hardest-to-identify params)

3. Should we add Sobol.jl as dependency, or use simpler random grid?
   - **Recommendation**: Add Sobol for better 18D coverage

---

## Status Tracking

- [x] Strategy document created
- [ ] Parameter configuration extended
- [ ] Dataset generated
- [ ] Surrogate trained
- [ ] Estimation complete
- [ ] Results analyzed
- [ ] Paper tables created

Last updated: 2026-03-07
