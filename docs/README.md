# MacroModelling.jl Documentation & Replication Package

Comprehensive documentation for MacroModelling.jl features, including SEP solver and regime-switching estimation.

**📄 This repository contains the complete replication package for:**
> **"Structural Bias from Linearization in DSGE Estimation"** (Farkas, 2026)  
> *Full paper with code, data, and results*

**See the [main README](../README.md) for paper overview and full replication instructions.**

## Features Documentation

### ⭐ NEW: Regime-Switching Estimation with Neural Network Surrogates

- **[Regime-Switching Estimation](regime_switching_estimation/README.md)** - Complete framework documentation
  - Neural network surrogate-based estimation
  - Hard-gate regime switching for OBC models
  - Inversion filter and filter-free sampling
  - Performance optimizations (1.5-2x speedup)
  - **Status**: Production-ready ✅

### Stochastic Extended Path (SEP) Solver

- **[SEP Implementation Summary](SEP_IMPLEMENTATION_SUMMARY.md)** - Executive summary
- **[SEP Methodology Guide](DETERMINISTIC_SHOCKS_AND_IRF_METHODOLOGY.md)** - Technical documentation
- **Status**: Validated against Adjemian & Juillard (2025) ✅

## Quick Links by Feature

### Regime-Switching Estimation (NEW!)

1. **[Overview & Quick Start](regime_switching_estimation/README.md)** - Feature overview
2. **[Quick Start Tutorial](regime_switching_estimation/QUICKSTART.md)** - 15-minute tutorial
3. **[Methodology](regime_switching_estimation/methodology/OVERVIEW.md)** - Academic documentation
4. **[Pipeline Guide](regime_switching_estimation/PIPELINE_GUIDE.md)** - Complete workflow
5. **[Performance Optimizations](regime_switching_estimation/PERFORMANCE_OPTIMIZATIONS.md)** - Speed improvements

### SEP Solver

1. **[Implementation Summary](SEP_IMPLEMENTATION_SUMMARY.md)** - What was implemented
2. **[Methodology Guide](DETERMINISTIC_SHOCKS_AND_IRF_METHODOLOGY.md)** - Technical details
3. **[Status Tracking](../DETERMINISTIC_SHOCKS_STATUS.md)** - Development log

## Overview

MacroModelling.jl provides advanced tools for nonlinear DSGE modeling:

### Quick Start

```julia
using MacroModelling

# Load model
include("models/RBC_Dynare.jl")

# Create shock sequence: +3σ at t=1
shocks = zeros(60, 1)
shocks[1, 1] = 0.3  # Absolute shock value

# Solve with deterministic shocks
solve!(RBC_Dynare,
       algorithm=:stochastic_extended_path,
       sep_periods=60,
       sep_order=10,
       sep_sparse_tree=true,
       sep_deterministic_shocks=shocks)

# Extract solution
sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
```

## Document Overview

### 1. SEP_IMPLEMENTATION_SUMMARY.md

**Purpose**: Executive summary for developers and users
**Contents**:
- What was implemented
- Technical details
- Usage examples
- Validation results
- Known limitations
- Future enhancements

**Audience**: All users

### 2. DETERMINISTIC_SHOCKS_AND_IRF_METHODOLOGY.md

**Purpose**: Comprehensive technical documentation
**Contents**:
- Theoretical background
- Mathematical formulation
- Implementation details
- IRF computation methodology
- Detailed API reference
- Comparison with Dynare
- Code examples

**Audience**: Developers, researchers, power users

### 3. DETERMINISTIC_SHOCKS_STATUS.md

**Purpose**: Development tracking and status log
**Contents**:
- Completed phases (1-5)
- Implementation history
- Bug fixes and resolutions
- Remaining work items

**Audience**: Developers

## Features

✅ **Deterministic shock sequences** (T×dε matrices)
✅ **Perfect foresight solver** with Newton method
✅ **Sparse Jacobian** (tridiagonal block structure)
✅ **Fishbone sparse tree** algorithm
✅ **Validated against Dynare** benchmarks

## Performance

**RBC Model** (7 variables, 60 periods):
- Convergence: 11 iterations
- Time: 0.5 seconds
- Accuracy: Machine precision vs Dynare

## API

### New Parameter

```julia
solve!(model;
       algorithm=:stochastic_extended_path,
       sep_deterministic_shocks::Union{Nothing,Matrix{Float64}}=nothing,
       # ... other parameters
       )
```

**Parameter**: `sep_deterministic_shocks`
- Type: `Matrix{Float64}` of size (T × dε)
- T = number of periods
- dε = number of shocks
- Each row = shock values for that period
- When provided: Uses deterministic mode (perfect foresight)
- When `nothing`: Uses stochastic mode (Gauss-Hermite)

## Validation

Validated against Adjemian & Juillard (2025) replication package:
- Reference: ep-mj-30-years-master
- Model: RBC with CES production
- Test: +3σ and -3σ technology shocks
- Result: ✅ Numerical agreement to machine precision

## References

### Primary Reference

**Adjemian, Stéphane, and Michel Juillard (2025)**
"Stochastic Extended Path"
- Paper: https://stephane-adjemian.fr/papers/sep-2025.pdf
- Slides: https://stephane-adjemian.fr/dynare/slides/sep-2025.pdf

### Replication Package

Location: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/ep-mj-30-years-master/`

Key files:
- `models/irf/rbc.mod` - Dynare RBC IRF test
- `matlab/spfirf.m` - IRF plotting function
- `matlab/pdss.m` - Percentage deviation transformation
- `matlab/sparsity/` - Sparse tree analysis

## Future Work

### Priority 1: ts Funnel Baseline

Enable full IRF computation: IRF = pdss(tt) - pdss(ts)

**Required**: `sep_initial_state` parameter

**Algorithm**:
1. Start from deterministic steady state
2. Iteratively solve with decreasing order (10 → 0)
3. Use end state as initial state for next iteration
4. Construct stochastic funnel baseline

### Priority 2: IRF Helper Functions

```julia
# Proposed API
irf = get_sep_irf(model, shock_idx, shock_magnitude;
                  periods=60, order=10)
```

Automates:
- tt path construction
- ts funnel baseline
- IRF computation and transformation

### Priority 3: Extended Documentation

- User guide with tutorials
- API reference for all SEP functions
- Performance optimization guide
- Replication of Adjemian-Juillard figures

## Contributing

Documentation improvements welcome! Please:

1. Ensure accuracy against Adjemian-Juillard (2025)
2. Test all code examples
3. Maintain consistency across documents
4. Update this README if adding new documents

## License

This documentation follows the license of MacroModelling.jl.

---

**Last Updated**: December 27, 2024
**Version**: 1.0
**Status**: ✅ Complete
