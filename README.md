# Investment Adjustment Costs and the Nonlinear Posterior of Smets-Wouters

**Replication Package** | **Paper & Code** | **Mátyás Farkas (2026)**

## Overview

This repository contains the complete replication package for the paper:

> **"Investment Adjustment Costs and the Nonlinear Posterior of Smets-Wouters"**
>
> This paper decomposes the gap between full nonlinear and linearly-approximated DSGE dynamics in a Smets-Wouters model, finding that investment adjustment costs account for 69% of the measured nonlinearity, not the zero-lower bound or pricing constraints typically emphasized. First-order perturbation renders the investment adjustment cost invisible because it satisfies $S(1) = S'(1) = 0$ at steady state.
>
> To estimate the nonlinear Smets-Wouters model on US data, I develop a **neural network surrogate** that learns the residual between nonlinear and linear transitions, coupled with a **regime-switching filter** that activates the nonlinear correction only during periods of large displacement from steady state. The methodology provides explicit control over approximation error with formal consistency bounds on posterior distortion.
>
> Bayesian estimation via NUTS-HMC completes in **2.4 hours** (on Apple M4) with zero divergent transitions in the baseline run. The empirical posterior comparison is conditional on the HLT specification, the warm-started high-likelihood surrogate mode, the gate design, and COVID out-of-distribution risk; within that benchmark, the nonlinear posterior shifts shock volatilities and persistence in ways consistent with investment-channel amplification. A quiet-sample OOS pilot fails under the original gate, but a fixed-posterior q95 padded-gate diagnostic reduces the aggregate RMSE ratio from **2.40** to **1.04**. Re-estimating the surrogate chain under that q95 gate gives **1.09** overall, **0.70** in the early quiet window, and **1.19** thereafter, so the issue is localized but not closed.

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
- Warm-started nonlinear mode improves the posterior-mean profiled objective by **35.92 nats** (`-1,781.68` vs. `-1,817.59`)
- Fixed-parameter no-NN ablation shows the gate/inversion architecture alone lowers LL; the NN residual correction drives the positive gain
- Risk-premium volatility **falls** to 0.092 in the extended sample
- Caveat: restored ARMA(1,1) markup terms improve the baseline linear LL by 34 nats, comparable in scale

### OOS Gate Diagnostic
- COVID-window OOS: switching surrogate reduces aggregate RMSE by **16%** over 2020Q1-2021Q2
- Quiet-sample pilot: estimating through 1994Q4 and scoring conditional predictions for 1995Q1-2007Q4 gives aggregate RMSE ratio **2.40** (switch/ROM1), with the original hard gate active throughout the holdout
- Fixed-posterior recalibration: stricter q95 padded-gate thresholds lower the quiet aggregate ratio to **1.04** and hard-gate activation to **1.9%**
- q95 re-estimation: a 150-draw surrogate chain under the q95 padded gate gives quiet aggregate ratio **1.09**, early-window ratio **0.70**, and remaining-window ratio **1.19**
- Interpretation: the default gate can help under COVID stress but over-activates in calm samples; q95 gate calibration sharply reduces the failure but does not yet make calm-period predictive performance uniformly better than ROM1

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
| Particle filter baseline | degenerates | bootstrap PF LL magnitude is roughly four orders worse than Kalman at 5,000 particles |

---

## Repository Structure

```
.
├── README.md                            # This file
├── REPLICATION.md                       # Curated replication workflow
├── CREDITS.md                           # Attribution to MacroModelling.jl
├── Project.toml / Manifest.toml         # Julia dependencies and pinned environment
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
│   │   ├── figures/                     # Versioned paper figure snapshots
│   │   └── generated/                   # Versioned paper table snapshots
│   │
│   └── [documentation]
│
├── scripts/                             # Replication scripts
│   ├── replication_smoke.sh             # Fast package verification
│   ├── hlt_sep_surrogate_dataset_generate.jl
│   ├── hlt_sep_surrogate_train.jl
│   ├── run_linear_hmc_advancedhmc.jl
│   ├── run_surrogate_hmc_advancedhmc.jl
│   ├── decompose_ll_gap.jl
│   ├── mode_sensitivity_report.jl
│   ├── oos_forecast_evaluate.jl
│   ├── sep_sensitivity_study.jl
│   ├── hlt_surrogate/                   # Shared surrogate utilities
│   └── [analysis scripts]
│
├── test/data/                           # Small empirical input snapshots
├── .local_artifacts/                    # Local heavy outputs (ignored)
│
├── test/                                # Unit tests & validation
│   ├── data/usmodel_update.csv          # US macroeconomic data (1959-2025Q1)
│   └── [test suites]
│
└── examples/                            # Usage examples
    └── sep_example.jl                   # SEP solver tutorial
```

---

## Quick Start

The curated replication guide is [REPLICATION.md](REPLICATION.md). The commands
below are the recommended minimal entry points.

### Environment

```bash
cd /path/to/SurrogateNN_Estimation.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### Smoke Verification

```bash
bash scripts/replication_smoke.sh
```

To skip the paper compile during smoke checks:

```bash
RUN_PAPER=0 bash scripts/replication_smoke.sh
```

### Paper Build

```bash
bash docs/SurrogateNN_paper/compile.sh
```

The paper build uses versioned figure snapshots in
`docs/SurrogateNN_paper/figures/` and versioned table snapshots in
`docs/SurrogateNN_paper/generated/`. Heavy `.jls` chains and timestamped
outputs are intentionally regenerated under `.local_artifacts/`.

### Heavy Replication Runs

Use [REPLICATION.md](REPLICATION.md) for the supported commands to regenerate
SEP datasets, train the surrogate, rerun linear/surrogate HMC, recompute
likelihood decomposition, mode sensitivity, OOS diagnostics, SEP sensitivity,
the Galí hard-ELB validation package, and direct-SEP smoke artifacts.

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
- **Convergence diagnostics**: warm-started surrogate chains have max $\hat{R} \approx 1.05$ and zero divergent transitions; cold-start failures are documented
- **Sensitivity/provenance status**: supported surrogate accuracy, convergence, mode-sensitivity, linear+gate ablation, quiet-sample OOS, bounded SEP-sensitivity pilot, and the HLT direct-SEP/MH validation harness are included. The HLT harness now returns a finite direct SEP likelihood with the exact determinant after tightening inversion finite-difference prediction solves. The maintained Galí hard-ELB validation package passes for the ROM1-residual/inversion/HMC pipeline on the locally identified two-parameter design; the full three-parameter direct-SEP HMC comparison remains a documented scaling target. The full 265-period SEP-sensitivity production run completed under `.local_artifacts/sep_sensitivity/full_265_20260521/`: all cells converge and tolerance is stable within each fixed `K`, while the `K=3` cells sit just above the original `1e-2` RMSE screen against the `K=5` reference.

**Monte Carlo Coverage**: `scripts/monte_carlo_coverage.jl` demonstrates posterior coverage distortion empirically.

**Particle Filter Baseline**: `scripts/particle_filter_benchmark.jl` reproduces the degeneracy issue (bootstrap PF log-likelihood magnitude roughly four orders worse than Kalman at 5,000 particles).

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
  title={Investment Adjustment Costs and the Nonlinear Posterior of Smets-Wouters},
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
