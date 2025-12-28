# MacroModelling.jl Test Suite

Organized test files for SEP (Stochastic Extended Path) solver validation and development.

## Directory Structure

```
tests/
├── sep_validation/      # Validation tests against Dynare benchmarks
├── sep_development/     # Development and diagnostic tests
└── archived/            # Historical development tests
```

## Validation Tests (`sep_validation/`)

Primary tests validating MacroModelling.jl SEP implementation against Dynare benchmarks.

### test_rbc_sparse_tree_irf_validation.jl ✅

**Purpose**: Validate deterministic shock implementation against Dynare RBC IRF

**Model**: RBC with CES production function
**Shock**: +3σ and -3σ technology shocks
**Periods**: 60
**Order**: 10 (maximum branching)

**Run**:
```bash
julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl
```

**Expected outcome**: tt path matches Dynare to machine precision

### test_sw07_HLT_sep_for_dynare_comparison.jl

**Purpose**: SW07_HLT model IRF comparison

**Model**: Smets-Wouters 2007 with Kimball aggregator
**Status**: In development

**Run**:
```bash
julia --project=. tests/sep_validation/test_sw07_HLT_sep_for_dynare_comparison.jl
```

### test_dynare_rs_replication.jl

**Purpose**: Replicate Dynare recursive utility model

**Model**: Epstein-Zin preferences (from Dynare test suite)

##  Development Tests (`sep_development/`)

Development, diagnostic, and debugging tests.

### Core SEP Tests

- `test_sep_simulation.jl` - SEP simulation functionality
- `test_sep_vs_pert.jl` - Compare SEP vs perturbation IRFs
- `test_sep_warm_start.jl` - Warm start efficiency test
- `test_sep_scaling.jl` / `test_sep_scaling_simple.jl` - Shock scaling tests
- `test_sep_convergence.jl` - Convergence diagnostics
- `test_sep_irf_simulation.jl` - Simulation-based IRF extraction

### Diagnostic Tests

- `test_shock_cov.jl` - Shock covariance handling
- `test_shock_extraction_bug.jl` - Shock extraction debugging
- `test_weight_indexing.jl` - GH weight indexing tests
- `test_gh_nodes.jl` - Gauss-Hermite node computation
- `test_sep_group_indexing.jl` - Group indexing for sparse tree

### Sparse Tree Tests

- `test_sparse_tree_compilation.jl` - Compilation test
- `test_stochastic_funnel_irf.jl` - ts funnel baseline construction

## Archived Tests (`archived/`)

Historical development tests kept for reference.

### Development Phases

- `test_sep_step1.jl` / `test_sep_step2.jl` - Early integration steps
- `test_sep_phase2.jl` / `test_sep_phase3.jl` - Phase-by-phase implementation
- `test_sep_minimal.jl` - Minimal SEP test
- `test_sep_integration.jl` - Integration test
- `test_sep_comprehensive.jl` - Comprehensive SEP test
- `test_sep_components.jl` - Component tests

### Model-Specific Old Tests

- `test_gali_sep_comparison.jl` - Gali model comparison
- `test_fs2000_sep_comparison.jl` - FS2000 model comparison

## Running Tests

### Quick Validation

```bash
# Validate RBC deterministic shocks
cd /Users/matyasfarkas/Documents/GitHub/MacroModelling.jl
julia --project=. tests/sep_validation/test_rbc_sparse_tree_irf_validation.jl
```

### Full SEP Test Suite

```bash
# Run all validation tests
for f in tests/sep_validation/*.jl; do
    echo "Running $f..."
    julia --project=. "$f"
done
```

### Development Testing

```bash
# Run specific development test
julia --project=. tests/sep_development/test_sep_vs_pert.jl
```

## Test Status Legend

- ✅ **Working** - Test passes reliably
- 🔧 **In Progress** - Test under development
- ⏸️ **On Hold** - Waiting for feature implementation
- 📦 **Archived** - Historical reference only

## SW07_HLT IRF Comparison

### Current Status

**Model**: Smets-Wouters 2007 with Kimball aggregator
**Location**: `models/Smets_Wouters_2007_HLT.jl`
**Test**: `tests/sep_validation/test_sw07_HLT_sep_for_dynare_comparison.jl`

**Challenge**: High-order nonlinearity from Kimball aggregator

**Next Steps**:
1. Verify steady state matches Dynare
2. Compare first-order perturbation IRFs
3. Run SEP with deterministic shocks
4. Compare against Dynare extended_path output

### Running SW07 Test

```bash
julia --project=. tests/sep_validation/test_sw07_HLT_sep_for_dynare_comparison.jl
```

## Test File Naming Convention

```
test_<category>_<description>.jl

Categories:
- sep_       SEP solver tests
- dynare_    Dynare replication tests
- shock_     Shock handling tests
- sparse_    Sparse tree tests
- stochastic_  Stochastic simulation tests
```

## Diagnostic Scripts

Located in `scripts/diagnostics/`:

- `diagnose_sep_shock_scaling.jl` - Extract and analyze SEP tree responses
- `time_sep_solver.jl` - Performance benchmarking
- `compare_markup_irf.jl` - Compare markup variable IRFs

**Run diagnostic**:
```bash
julia --project=. scripts/diagnostics/diagnose_sep_shock_scaling.jl
```

## Contributing

When adding new tests:

1. Place in appropriate directory (`sep_validation/` or `sep_development/`)
2. Follow naming convention: `test_<category>_<description>.jl`
3. Add entry to this README
4. Include header comment explaining purpose
5. Add expected outcome and run instructions

## Documentation

See `docs/` for comprehensive documentation:
- `docs/README.md` - Documentation index
- `docs/SEP_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `docs/DETERMINISTIC_SHOCKS_AND_IRF_METHODOLOGY.md` - Technical guide

---

**Last Updated**: December 27, 2024
