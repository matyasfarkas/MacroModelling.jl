# Codebase Review Findings
**Date**: 2026-02-27
**Commit**: codex/consolidation-hlt-switching-audit
**Reviewer**: Claude Code (Sonnet 4.5)

## Executive Summary

This review inventories the RS/SEP/surrogate functionality across the SurrogateNN_Estimation.jl repository to identify script-local logic that should be promoted to the `src/` API, logic that should remain script-local, and deprecated code.

**Key Finding**: The regime-switching API in `src/regime_switching/` is well-architected with clean separation of concerns, but **significant business logic resides in script-local utilities** (`scripts/hlt_surrogate/*.jl`) that should be promoted to the library API for reusability, testability, and maintainability.

---

## 1. Current Architecture Assessment

### 1.1 Core API (`src/regime_switching/`)

**Status**: ✅ **GOOD** - Clean, well-factored, reusable

| File | Purpose | Lines | Quality |
|------|---------|-------|---------|
| `types.jl` | Config structs (`RegimeSwitchConfig`, `SwitchingLikelihoodConfig`, etc.) | 58 | Excellent |
| `likelihood.jl` | Switching LL composition, inversion filter, FOM LL evaluation | 541 | Excellent |
| `gating.jl` | Gate calibration, regime assignment, probability computation | 656 | Excellent |
| `diagnostics.jl` | Gate statistics, episode analysis, chain utilities | 219 | Very Good |
| `io.jl` | Payload serialization/deserialization, validation | 212 | Very Good |

**Strengths**:
- Pure functions with explicit dependencies
- Comprehensive input validation
- Type-stable implementations
- Clear separation between policy (configs) and implementation
- Good test coverage via `test_regime_switching_api.jl`

**Weaknesses**:
- Missing dataset generation API
- Missing FOM benchmark runner API
- No chain reporting/analysis API (exists only in `diagnostics.jl` partially)

---

## 2. Script-Local Functionality Audit

### 2.1 Orchestration Scripts (`scripts/hlt_sep_surrogate_*.jl`)

**Count**: 12 scripts
**Total Functionality**: ~3000+ LOC of orchestration + business logic

| Script | Purpose | Promote to API? | Rationale |
|--------|---------|----------------|-----------|
| `hlt_sep_surrogate_train.jl` | Train surrogate NN | ❌ Keep | True orchestration, CLI-heavy |
| `hlt_sep_surrogate_dataset_generate.jl` | Generate SEP dataset | ⚠️ **Partial** | Core dataset generation should be API |
| `hlt_sep_surrogate_synthetic_data.jl` | Generate synthetic obs | ✅ **Yes** | Reusable synthetic data protocol |
| `hlt_sep_surrogate_gate_calibration.jl` | Calibrate gate thresholds | ❌ Keep | Wrapper around `gating.jl` API |
| `hlt_sep_surrogate_validate_hlt3.jl` | Validation harness orchestrator | ❌ Keep | Top-level orchestration |
| `hlt_sep_surrogate_fom_benchmark.jl` | FOM benchmark runner | ⚠️ **Partial** | FOM execution logic should be API |
| `hlt_sep_surrogate_acceptance_smoke.jl` | Acceptance test runner | ❌ Keep | Test orchestration |
| `hlt_sep_surrogate_chain_report.jl` | Chain analysis/reporting | ✅ **Yes** | Reusable reporting API needed |
| `hlt_sep_surrogate_synthetic_estimation.jl` | Full estimation workflow | ❌ Keep | Example workflow script |
| `hlt_sep_surrogate_hmc_diagnose.jl` | HMC diagnostics | ✅ **Yes** | Diagnostic utilities |
| `hlt_sep_surrogate_forecast_errors.jl` | Forecast error analysis | ✅ **Yes** | Analysis utilities |
| `hlt_sep_surrogate_fom_probe_search.jl` | FOM configuration search | ❌ Keep | Experimental tuning script |

### 2.2 Script-Local Utility Modules (`scripts/hlt_surrogate/*.jl`)

**Count**: 9 utility modules
**Total Functionality**: ~2000+ LOC

| Module | Purpose | Promote to API? | Rationale |
|--------|---------|----------------|-----------|
| `hlt_sep_surrogate_cli_utils.jl` | CLI argument parsing (54 LOC) | ❌ Keep | Script-specific CLI patterns |
| `hlt_model_loader_utils.jl` | Model loading, parameter extraction | ⚠️ **Partial** | Core model utilities should be API |
| `hlt_sep_surrogate_nn_utils.jl` | NN training, frozen MLP management | ✅ **Yes** | Core surrogate training API |
| `hlt_turing_model_factory.jl` | Dynamic Turing model generation | ✅ **Yes** | Reusable model construction |
| `hlt_sep_surrogate_rom_utils.jl` | ROM residual computation | ✅ **Yes** | Core ROM approximation logic |
| `hlt_sep_surrogate_dataset_parallel.jl` | Parallel dataset generation | ✅ **Yes** | Core dataset generation engine |
| `parameter_config.jl` | Parameter specifications (legacy 3, phase1 18) | ✅ **Yes** | Reusable parameter config API |
| `test_lhs_sampling.jl` | LHS sampling tests | ❌ Keep | Test utility |
| `test_backward_compat.jl` | Backward compat tests | ❌ Keep | Test utility |

---

## 3. Detailed Findings by Functional Area

### 3.1 Dataset Generation (⚠️ **CRITICAL GAP**)

**Current State**: Logic split between:
- `scripts/hlt_sep_surrogate_dataset_generate.jl` (orchestration, ~300 LOC)
- `scripts/hlt_surrogate/hlt_sep_surrogate_dataset_parallel.jl` (parallel execution)
- `scripts/hlt_surrogate/hlt_model_loader_utils.jl` (model setup)

**Issue**: No reusable API for:
- SEP simulation loop over parameter grid
- Training data extraction from SEP trees
- Feature/target construction
- Train/validation split

**Recommendation**: **PROMOTE** core dataset generation logic to:
```
src/regime_switching/dataset.jl
```

**Functions to Expose**:
```julia
# Core dataset generation
function generate_surrogate_dataset(model, param_grid, sep_config; kwargs...)
function extract_training_pairs(sep_solution, observables)
function split_train_validation(X, Y, val_fraction)

# Parallel execution
function generate_dataset_parallel(model, param_grid, sep_config; n_workers, kwargs...)
```

**Severity**: HIGH - Dataset generation is a core workflow, currently not reusable outside scripts

---

### 3.2 FOM Benchmark Runner (⚠️ **CRITICAL GAP**)

**Current State**: Logic in:
- `scripts/hlt_sep_surrogate_fom_benchmark.jl` (~800 LOC)
- Includes: preset configuration, period selection, recovery ladder, result validation

**Issue**: FOM benchmark execution is script-local, making it:
- Untestable in isolation
- Not reusable for custom benchmark configurations
- Difficult to extend (e.g., new FOM presets)

**Recommendation**: **PROMOTE** FOM benchmark logic to:
```
src/regime_switching/fom_benchmark.jl
```

**Functions to Expose**:
```julia
# FOM benchmark execution
function run_fom_benchmark(model, scenario, chain_payload, benchmark_config)
function make_fom_preset(preset_symbol)  # :direct_sep_gated_smoke, etc.
function validate_fom_results(results, benchmark_config)

# Recovery ladder (SEP failure handling)
function apply_recovery_ladder(model, params, data; ladder_config)
```

**Severity**: HIGH - FOM benchmarking is a core validation step, should be library functionality

---

### 3.3 Synthetic Data Generation (✅ **PROMOTE**)

**Current State**: Logic in:
- `scripts/hlt_sep_surrogate_synthetic_data.jl` (~250 LOC)
- Includes: volatility episode injection, sample period selection, truth shock construction

**Recommendation**: **PROMOTE** to:
```
src/regime_switching/synthetic.jl
```

**Functions to Expose**:
```julia
# Synthetic scenario construction
function generate_synthetic_scenario(model, params, T; volatility_config, kwargs...)
function inject_volatility_episode(shocks, episode_start, episode_length, scale_factor)
function extract_sample_period(data, sample_start, sample_length)
```

**Severity**: MEDIUM - Needed for reproducible synthetic experiments

---

### 3.4 Chain Reporting & Analysis (✅ **PROMOTE**)

**Current State**: Partial logic in:
- `src/regime_switching/diagnostics.jl` (chain utilities, ~50 LOC)
- `scripts/hlt_sep_surrogate_chain_report.jl` (full reporting, ~400 LOC)

**Issue**: Chain reporting mixes reusable analysis (should be API) with formatting (OK to be script-local)

**Recommendation**: **PROMOTE** analysis functions to:
```
src/regime_switching/chain_analysis.jl
```

**Functions to Expose**:
```julia
# Chain summary statistics
function compute_parameter_summary(chain, theta_names)
function compute_shock_recovery_rmse(chain, truth_shocks, sample_idx)
function compute_gate_diagnostics(chain, gate_mask_truth)

# Convergence diagnostics
function compute_rhat(chain)
function compute_ess_bulk(chain)
function compute_acceptance_rate(chain)
```

**Severity**: MEDIUM - Improves reusability of chain analysis

---

### 3.5 Surrogate Training (✅ **PROMOTE**)

**Current State**: Logic in:
- `scripts/hlt_surrogate/hlt_sep_surrogate_nn_utils.jl` (~400 LOC)
- Includes: MLP construction, training loop, frozen MLP serialization, normalization

**Recommendation**: **PROMOTE** to:
```
src/regime_switching/surrogate_nn.jl
```

**Functions to Expose**:
```julia
# Surrogate NN construction
function train_surrogate_mlp(X, Y; hidden_dim, nepochs, learning_rate, kwargs...)
function freeze_mlp(trained_mlp, normstats)
function predict_surrogate(frozen_mlp, state, shock, theta)
```

**Severity**: HIGH - Core algorithmic component, currently not in library

---

### 3.6 Turing Model Factory (✅ **PROMOTE**)

**Current State**: Logic in:
- `scripts/hlt_surrogate/hlt_turing_model_factory.jl` (~300 LOC)
- Includes: dynamic model generation for 3-param, 18-param, future 28+ param

**Recommendation**: **PROMOTE** to:
```
src/regime_switching/turing_models.jl
```

**Functions to Expose**:
```julia
# Turing model construction
function create_switching_likelihood_model(frozen_mlp, data, param_specs; kwargs...)
function get_parameter_specs(param_set::Symbol)  # :legacy_3params, :phase1_18params
```

**Severity**: HIGH - Core estimation interface, enables scale-up

---

### 3.7 ROM Residual Utilities (✅ **PROMOTE**)

**Current State**: Logic in:
- `scripts/hlt_surrogate/hlt_sep_surrogate_rom_utils.jl` (~200 LOC)
- Includes: ROM approximation, residual computation, ROM1 baseline

**Recommendation**: **PROMOTE** to:
```
src/regime_switching/rom.jl
```

**Functions to Expose**:
```julia
# ROM approximation
function compute_rom1_residual(linear_obs, full_obs)
function make_rom_predictor(model, params, algorithm)
```

**Severity**: MEDIUM - Useful for ROM vs direct-SEP comparisons

---

## 4. Deprecation Candidates

**Files to Archive** (move to `archive/` or delete):

1. `ChatGPT/SEP_port_step_*.md` (26 files) - Already deleted in git status
   Status: ✅ Deleted in current branch

2. `SEP archive/old_docs/*.md` (28 files) - Old documentation artifacts
   Status: ✅ Deleted in current branch

3. `scripts/*_backup.jl` or `*_old.jl` - None found
   Status: ✅ No deprecated scripts found

---

## 5. Test Coverage Assessment

### 5.1 Existing Tests

| Test File | Coverage Area | Status |
|-----------|--------------|--------|
| `test_regime_switching_api.jl` | Core API (types, likelihood, gating) | ✅ Good |
| `test_hlt_validation_harness.jl` | Validation workflow | ✅ Good |
| `test_hlt_acceptance_smoke.jl` | Acceptance criteria | ✅ Good |
| `test_hlt_obc_sep.jl` | SEP with OBC | ✅ Good |

### 5.2 Missing Tests

**High Priority**:
- Dataset generation API (when promoted)
- FOM benchmark runner API (when promoted)
- Surrogate NN training (when promoted)
- Turing model factory (when promoted)

**Medium Priority**:
- Synthetic data generation
- Chain analysis utilities
- ROM residual computation

---

## 6. File Structure Recommendation

### 6.1 Proposed `src/regime_switching/` Structure

```
src/regime_switching/
├── types.jl                    # ✅ Exists
├── likelihood.jl               # ✅ Exists
├── gating.jl                   # ✅ Exists
├── diagnostics.jl              # ✅ Exists
├── io.jl                       # ✅ Exists
├── dataset.jl                  # ⚠️ NEW (promote from scripts)
├── fom_benchmark.jl            # ⚠️ NEW (promote from scripts)
├── synthetic.jl                # ⚠️ NEW (promote from scripts)
├── chain_analysis.jl           # ⚠️ NEW (promote from scripts + expand diagnostics)
├── surrogate_nn.jl             # ⚠️ NEW (promote from scripts)
├── turing_models.jl            # ⚠️ NEW (promote from scripts)
├── rom.jl                      # ⚠️ NEW (promote from scripts)
└── parameter_specs.jl          # ⚠️ NEW (promote from scripts)
```

### 6.2 Scripts to Keep (Orchestration Only)

```
scripts/
├── hlt_sep_surrogate_train.jl                  # Orchestration
├── hlt_sep_surrogate_dataset_generate.jl       # Wrapper around new dataset.jl API
├── hlt_sep_surrogate_gate_calibration.jl       # Wrapper around gating.jl
├── hlt_sep_surrogate_validate_hlt3.jl          # Top-level harness
├── hlt_sep_surrogate_fom_benchmark.jl          # Wrapper around new fom_benchmark.jl API
├── hlt_sep_surrogate_acceptance_smoke.jl       # Test runner
├── hlt_sep_surrogate_synthetic_estimation.jl   # Example workflow
└── hlt_surrogate/
    ├── hlt_sep_surrogate_cli_utils.jl          # CLI-specific utilities
    ├── test_lhs_sampling.jl                    # Test utility
    └── test_backward_compat.jl                 # Test utility
```

---

## 7. Priority Ranking

### P0: Critical for Submission (Correctness/Reproducibility)
1. **Verify current HEAD works end-to-end** (Phase B)
2. **Promote FOM benchmark API** - Core validation workflow
3. **Promote Turing model factory** - Enables 18-param scale-up narrative

### P1: High Priority (Publication Quality)
4. **Promote dataset generation API** - Core training workflow
5. **Promote surrogate NN API** - Core algorithmic component
6. **Promote synthetic data API** - Reproducible experiments

### P2: Medium Priority (Code Quality)
7. **Promote chain analysis API** - Reporting utilities
8. **Promote ROM utilities** - ROM vs FOM comparisons
9. **Expand test coverage** for promoted APIs

### P3: Low Priority (Nice to Have)
10. **Refactor CLI utilities** - Extract common patterns
11. **Documentation generation** - API reference docs

---

## 8. Risk Assessment

### 8.1 Correctness Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Script-local logic untested in isolation | **HIGH** | Promote to API + add unit tests |
| FOM benchmark configuration drift | **MEDIUM** | Centralize presets in `fom_benchmark.jl` |
| Parameter config inconsistency (3 vs 18 param) | **MEDIUM** | Centralize in `parameter_specs.jl` |

### 8.2 Robustness Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SEP floor-hit handling undocumented | **MEDIUM** | Document recovery ladder protocol |
| Gating sensitivity not quantified | **MEDIUM** | Add robustness tests (seed sweeps) |
| Chain deserialization module conflicts | **LOW** | Already handled by `ensure_chain_deserialize_modules!()` |

### 8.3 Performance Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Dataset generation allocates repeatedly | **MEDIUM** | Profile + optimize SEP loop |
| Redundant model reloads in benchmarks | **LOW** | Cache model instances where safe |

---

## 9. Recommendations Summary

### Immediate Actions (Before Submission)
1. ✅ **Phase B**: Run full verification suite on current HEAD
2. ⚠️ **Promote FOM benchmark API** to `src/regime_switching/fom_benchmark.jl`
3. ⚠️ **Promote Turing model factory** to `src/regime_switching/turing_models.jl`
4. ⚠️ **Document recovery ladder** in FOM benchmark docs

### Short-Term Actions (Submission Prep)
5. ⚠️ **Promote dataset generation API** to `src/regime_switching/dataset.jl`
6. ⚠️ **Promote surrogate NN API** to `src/regime_switching/surrogate_nn.jl`
7. ⚠️ **Add tests** for promoted APIs

### Medium-Term Actions (Post-Submission)
8. ⚠️ **Promote remaining utilities** (synthetic, chain analysis, ROM)
9. ⚠️ **Refactor scripts** to use new APIs
10. ⚠️ **Expand test coverage** to 80%+

---

## 10. References

### Codebase Files Reviewed
- `src/regime_switching/*.jl` (5 files, ~1700 LOC)
- `scripts/hlt_sep_surrogate_*.jl` (12 files, ~3000+ LOC)
- `scripts/hlt_surrogate/*.jl` (9 files, ~2000+ LOC)
- `test/test_regime_switching*.jl` (2 files)
- `test/test_hlt_*.jl` (3 files)

### Total Codebase
- **Core API**: ~1700 LOC
- **Script orchestration**: ~3000+ LOC
- **Script utilities**: ~2000+ LOC
- **Total RS/SEP/surrogate LOC**: ~6700+

### Architecture Quality
- **Core API quality**: **Excellent** (9/10)
- **Script organization**: **Fair** (5/10) - Too much business logic in scripts
- **Test coverage**: **Good** (7/10) - API tested, scripts not

---

**End of Phase A Review**
