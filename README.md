# Structural Bias from Linearization in DSGE Estimation

**Replication Package** | **Paper & Code** | **Mátyás Farkas (2026)**

## Overview

This repository contains the complete replication package for the paper:

> **"Structural Bias from Linearization in DSGE Estimation"**
>
> This paper decomposes the gap between full nonlinear and linearly-approximated DSGE dynamics, discovering that investment adjustment costs account for 69% of the total nonlinearity—not the zero-lower bound or pricing constraints typically emphasized. First-order perturbation renders the investment adjustment cost invisible because it satisfies $S(1) = S'(1) = 0$ at steady state.
>
> To estimate the nonlinear Smets-Wouters model on US data, I develop a **neural network surrogate** that learns the residual between nonlinear and linear transitions, coupled with a **regime-switching filter** that activates the nonlinear correction only during periods of large displacement from steady state. The methodology provides explicit control over approximation error with formal consistency bounds on posterior distortion.
>
> Bayesian estimation via NUTS-HMC completes in **2.4 hours** (on Apple M4) with zero divergent transitions. Compared to linear estimation, nonlinear results show doubled risk-premium volatility, 80% compression of wage markup persistence, and dramatically reduced shock sizes to fit COVID-19 data—revealing an economy driven more by endogenous investment-channel amplification than canonically estimated.

---

## Key Results

### Invisibility Decomposition
- **Investment channel**: 69% of nonlinearity
- **Price Phillips curve**: 0.2% of nonlinearity
- **Other channels**: ~31% combined

### Estimation Findings (1959-2004)
| Parameter | Linear | Nonlinear | Change |
|-----------|--------|-----------|--------|
| $\sigma_b$ (risk-premium volatility) | 0.12 | 0.28 | +133% |
| $\rho_w$ (wage markup persistence) | 0.37 | 0.08 | -78% |
| $\varepsilon_p$ (Kimball curvature) | 10.0 | 27.0 | +170% |

### Extended Sample (through 2025Q1)
- Linear model requires **56σ risk-premium shocks** for COVID collapse
- Nonlinear model fits same data with **49% smaller shocks** via endogenous amplification
- Risk-premium volatility **falls** to 0.09 in extended sample

---

## Methodology: Neural Network Surrogate + Regime Switching

### 1. **Nonlinear Solver**
Uses **Stochastic Extended Path (SEP)** method with occasionally binding constraints (OBC), scaling to medium-scale models and handling ZLB natively.

### 2. **Neural Network Surrogate**
- Learns residual $\hat{r}(s_{t-1}, \epsilon_t, \theta) = g^{\text{SEP}} - g^{\text{ROM1}}$ (two orders of magnitude smaller than full transition)
- Trained on 22,080 parameter-state-shock combinations spanning the full prior
- Provides smooth gradients for HMC and observable approximation error (validation RRMSE)

### 3. **Regime-Switching Filter**
- Hard gate classifies periods as "calm" or "stressed" based on economy's distance from steady state
- Applies nonlinear correction only where it matters; uses exact Kalman filter elsewhere
- Avoids particle filter weight degeneracy

### 4. **Filter-Free NUTS-HMC Sampling**
- Treats shocks as explicit unknowns in the posterior
- Sidesteps weight collapse plaguing bootstrap particle filters
- Achieves 1,000 effective samples with zero divergent transitions

---

## Computational Performance

| Stage | Time | Hardware |
|-------|------|----------|
| Offline: Surrogate training | ~12 hours | Apple M4 |
| Online: NUTS-HMC estimation (1000 draws) | ~2.4 hours | Apple M4 |
| **Total** | **~14.4 hours** | - |
| Particle filter baseline | diverges | asymptotes to 7 orders of magnitude error |

---

## Repository Structure

```
.
├── README.md                            # This file
├── CREDITS.md                           # Attribution to MacroModelling.jl
├── Project.toml                         # Julia dependencies
├── LICENSE                              # MIT License
│
├── src/                                 # Main package code
│   ├── MacroModelling.jl                # Core functionality
│   ├── particle_filter_*.jl             # Filter implementations
│   └── [supporting modules]
│
├── models/                              # DSGE model definitions
│   ├── Smets_Wouters_2007_HLT_obc.jl  # OBC DSGE model (main application)
│   └── [other comparative models]
│
├── docs/
│   ├── SurrogateNN_paper/               # Replication paper
│   │   ├── SurrogateNN_paper.tex        # Main paper
│   │   ├── SurrogateNN_paper.bib        # Bibliography
│   │   ├── figures*/                    # Generated figures & tables
│   │   └── generated/                   # Empirical results tables
│   │
│   └── [documentation]
│
├── scripts/                             # Replication scripts
│   ├── hlt_real_data_update_pipeline.jl # Full pipeline (1959-2025Q1)
│   ├── hlt_surrogate/                   # Surrogate training
│   │   ├── hlt_surrogate_training.jl    # Train neural network
│   │   ├── hlt_surrogate_validation.jl  # Validation diagnostics
│   │   └── hlt_model_loader_utils.jl
│   ├── monte_carlo_coverage.jl          # Posterior coverage diagnostics
│   ├── particle_filter_benchmark.jl     # PF degeneracy demonstration
│   └── [analysis scripts]
│
├── data/                                # Empirical datasets
│   └── hlt_**/                          # Timestamped dataset snapshots
│
├── test/                                # Unit tests & validation
│   ├── data/usmodel_update.csv          # US macroeconomic data (1959-2025Q1)
│   └── [test suites]
│
└── examples/                            # Usage examples
    └── sep_example.jl                   # SEP solver tutorial
```

---

## Quick Start: Running the Replication

### Environment Setup

```bash
cd /path/to/SurrogateNN_Estimation.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Full End-to-End Pipeline (with Offline Training)

```bash
julia --project=. scripts/hlt_surrogate/hlt_surrogate_training.jl \
  --design-size=22080 \
  --batch-size=1024
  
julia --project=. scripts/hlt_real_data_update_pipeline.jl \
  --run-dir=.local_artifacts/hlt_real_data_runs/hlt_update_$(date +%Y%m%d_%H%M%S)
```

This creates:
- `.local_artifacts/hlt_*/dataset/` – Synthetic training data for surrogate
- `.local_artifacts/hlt_*/real_data/` – NUTS-HMC posterior samples
- `.local_artifacts/hlt_*/tables/` – Posterior summaries & decomposition tables
- `.local_artifacts/hlt_*/diagnostics/` – Convergence diagnostics & sensitivity analysis

### Generating Paper Figures & Tables

```bash
julia --project=. scripts/generate_paper_output.jl \
  --run-dir=.local_artifacts/hlt_real_data_runs/[your_run]
```

### Key Configuration Parameters

**Surrogate Training:**
```julia
design_size = 22080              # Parameter-state-shock combinations
model_order = 1                  # SEP order (quadratic pruning)
network_width = [128, 128, 64]  # Hidden layer sizes
validation_fraction = 0.2        # Test set for RRMSE
```

**Estimation:**
```julia
sampler = "nuts"                 # NUTS-HMC (required for smooth surrogates)
n_samples = 1000                 # Posterior samples per chain
n_chains = 4                      # Parallel chains
target_acceptance = 0.8          # HMC step size tuning
```

**Data:**
```julia
data_file = "test/data/usmodel_update.csv"
sample_window = 47:290           # 1959Q1:2004Q4 (baseline)
sample_window = 47:409           # 1959Q1:2025Q1 (extended)
```

---

## Credits & Attribution

### Original Framework: MacroModelling.jl

This replication package builds entirely on **[MacroModelling.jl](https://github.com/thorek1/MacroModelling.jl)**, authored and maintained by **Thore Kockerols** (@thorek1).

MacroModelling.jl provides:
- The core DSGE model solving and estimation infrastructure
- State-space linearization (first-order perturbation baseline)
- Parameter definition and steady-state solvers
- Integration with Turing.jl and StatsPlots.jl

**Reference:**
> Kockerols, T. (2024). MacroModelling.jl: A Julia package for DSGE model estimation and simulation.
> [https://github.com/thorek1/MacroModelling.jl](https://github.com/thorek1/MacroModelling.jl)

### This Repository's Contributions

Mátyás Farkas extends MacroModelling.jl with:
- **Stochastic Extended Path (SEP) solver** for nonlinear DSGE dynamics with occasionally binding constraints
- **Neural network surrogate framework** for fast likelihood evaluation via residual learning
- **Regime-switching filter** for selective application of nonlinear corrections
- **Filter-free NUTS-HMC sampling** avoiding particle filter weight collapse
- **Empirical decomposition** quantifying the contribution of investment adjustment costs to nonlinearity
- **Full Bayesian inference** on the Smets-Wouters OBC model with US macroeconomic data (1959-2025Q1)

---

## Reproducibility & Validation

- **Surrogate validation**: `RRMSE < 0.1%` on held-out test set
- **Posterior coverage**: Formal consistency bounds on posterior RMSE distortion (Appendix, main paper)
- **Convergence diagnostics**: $\hat{R} < 1.01$ across all parameters; zero divergent transitions
- **Sensitivity analysis**: Results robust to network architecture, training sample size, and regime threshold

**Monte Carlo Coverage**: `scripts/monte_carlo_coverage.jl` demonstrates posterior coverage distortion empirically.

**Particle Filter Baseline**: `scripts/particle_filter_benchmark.jl` reproduces the degeneracy issue (log-likelihood 7 orders of magnitude below true value).

---

## Requirements

- **Julia**: v1.8+
- **Key dependencies**: MacroModelling.jl, AdvancedHMC.jl, Turing.jl, Flux.jl, Distributions.jl
- **Data**: US macroeconomic time series (1959-2025Q1); included in `test/data/`
- **Hardware**: ~32GB RAM for SEP solve; 2-4 CPU cores recommended for parallel chains

---

## License

MIT License (same as MacroModelling.jl)

Copyright (c) 2022 Thore Kockerols  
Copyright (c) 2026 Mátyás Farkas

See [LICENSE](LICENSE) for full terms.

---

## References

### Core Methodological Papers

- Adjemian, S. & Juillard, M. (2025). "Stochastic Extended Path." *arXiv:2501.xxxxx*
- Guerrieri, L. & Iacoviello, M. (2015). "OccBin: A toolkit to solve, simulate, and estimate DSGE models with occasionally binding constraints." *Journal of Economic Dynamics and Control*, 70, 200-221.
- Kennedy, M.C. & O'Hagan, A. (2001). "Bayesian calibration of computer models." *Journal of the Royal Statistical Society*, 63(3), 425-464.

### DSGE & ZLB Estimation

- Smets, F. & Wouters, R. (2007). "Shocks and frictions in US business cycles: A Bayesian DSGE approach." *American Economic Review*, 97(3), 586-606.
- Gust, C. et al. (2012). "The empirical implications of the interest rate lower bound." *Finance and Economics Discussion Series*, Federal Reserve Board.
- Boehl, G. (2022). "Efficient solution and computation of nonlinear DSGE models." *Journal of Economic Dynamics and Control*, 136, 104306.

### Surrogate & Emulation Methods

- Peherstorfer, B., Willcox, K., Gunzburger, M. (2018). "Survey of multifidelity methods in uncertainty propagation, inference, and optimization." *SIAM Review*, 60(3), 550-591.

Full citation list available in [docs/SurrogateNN_paper/SurrogateNN_paper.bib](docs/SurrogateNN_paper/SurrogateNN_paper.bib).

---

## Contact & Support

For questions about:
- **Paper methodology**: Contact Mátyás Farkas (author, [GitHub](https://github.com/matyasfarkas))
- **MacroModelling.jl core**: Contact Thore Kockerols ([@thorek1](https://github.com/thorek1))
- **Issues & reproducibility**: Open an issue on this repository

---

## Citation

If you use this replication package in your research, please cite:

```bibtex
@article{farkas2026structural,
  title={Structural Bias from Linearization in DSGE Estimation},
  author={Farkas, M\'aty\'as},
  year={2026}
}
```

And the underlying framework:

```bibtex
@software{kockerols2024macromodelling,
  title={MacroModelling.jl: A Julia package for DSGE model estimation and simulation},
  author={Kockerols, Thore},
  year={2024},
  url={https://github.com/thorek1/MacroModelling.jl}
}
```
