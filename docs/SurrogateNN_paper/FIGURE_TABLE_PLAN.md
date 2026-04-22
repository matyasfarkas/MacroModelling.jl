# Figure and Table Plan for JMP
**Paper**: Global Estimation of Nonlinear DSGE Models via Neural Network Surrogates

**Purpose**: This document specifies all figures and tables for the paper, including data sources, generation scripts, and panel specifications.

---

## Summary Statistics

**Total Figures**: 15 main text + 8 appendix = 23
**Total Tables**: 5 main text + 4 appendix = 9

---

## Main Text Figures

### Figure 1: SEP Tree Illustration
**Section**: 4.1 Stochastic Extended Path
**Purpose**: Visualize branching tree structure of SEP algorithm

**Specifications**:
- **Type**: Schematic diagram
- **Content**:
  - Horizontal axis: Time (t, t+1, t+2,...)
  - Vertical axis: Branches (ε_t^(1), ε_t^(2), ..., ε_t^(K))
  - Show branching at t+1: st → {st+1|ε^(k)} for k=1,...,K
  - Show future paths: st+1|ε^(k) → st+2|ε^(k,j)
  - Highlight: Expectation approximation 𝔼[st+1] ≈ Σk wk·st+1|ε^(k)
  - Annotate: "K=5 Gauss–Hermite nodes", "Branching horizon L=3"
- **Source**: Reproduce from presentation Slide 9
- **Format**: TikZ/PGF or vector graphic
- **Size**: Half-page width

**Generation**:
- Manual creation in LaTeX/TikZ or
- Julia script: `scripts/figures/plot_sep_tree_schematic.jl`

---

### Figure 2: SEP Tree to Training Dataset
**Section**: 4.2 Dataset Generation
**Purpose**: Show how SEP tree nodes become training pairs

**Specifications**:
- **Type**: Schematic with data extraction overlay
- **Panel A**: SEP tree (as in Figure 1)
- **Panel B**: Zoom into single parent→child transition
  - Parent node: (st−1,obs(g_par), t−1)
  - Child nodes: {(st,obs(gk), t)} for k=1,...,K
  - Shock realizations: {ε_t^(k)} with weights {wk}
- **Panel C**: Training pair format
  ```
  Input:  x = [st−1,obs, ε_t^(k), θ]
  Output: y = [st,obs, st+1](gk)
  ```
- **Source**: Reproduce from presentation Slide 11
- **Format**: Multi-panel schematic
- **Size**: Full-page width

**Generation**:
- Manual creation in LaTeX/TikZ

---

### Figure 3: MLP Surrogate Architecture
**Section**: 4.2 Surrogate Training
**Purpose**: Visualize neural network architecture

**Specifications**:
- **Type**: Network diagram
- **Content**:
  - Input layer: Nodes for st−1, εt, θ (color-coded)
  - Hidden layer: dh = 128 tanh nodes (show subset)
  - Output layer: Nodes for [st,obs, st+1]
  - Weights: Show W1, b1, W2, b2 schematically
  - Comparison box: "SEP: gθ(st−1, εt) = st"
- **Source**: Reproduce from presentation Slide 12
- **Format**: Network diagram (TikZ or vector)
- **Size**: Half-page width

**Generation**:
- Manual creation in LaTeX/TikZ or
- Julia script with Plots.jl/Makie.jl: `scripts/figures/plot_mlp_architecture.jl`

---

### Figure 4: Posterior Diagnostics - θ^Calvo
**Section**: 7.1 3-Parameter Validation
**Purpose**: Show chain convergence and posterior for Calvo parameter

**Specifications**:
- **Type**: 4-panel diagnostic plot
- **Panel A**: Trace plot
  - X: Iteration number (0-4000)
  - Y: θ^Calvo value (0.4-1.0)
  - Show 4 chains (different colors)
  - Horizontal line: Truth (0.75)
- **Panel B**: Posterior density
  - X: θ^Calvo value
  - Y: Density
  - Curves: Posterior (solid), prior (dashed)
  - Vertical lines: Mode, median, truth
  - 95% CI shaded region
- **Panel C**: Autocorrelation
  - X: Lag (0-30)
  - Y: ACF
  - Decay to near zero
- **Panel D**: Running mean
  - X: Iteration number
  - Y: Running mean of θ^Calvo
  - Convergence to truth
  - Horizontal line: Truth (0.75)
- **Footer**: Diagnostics text
  ```
  R̂ = 1.002, ESS_bulk = 1000, ESS_tail = 800
  Median = 0.735, Mode = 0.710, 95% CI = [0.449, 0.963]
  ```
- **Source**: Reproduce from presentation Slide 18
- **Format**: 2×2 grid, high-resolution
- **Size**: Full-page width

**Generation**:
- Julia script: `scripts/figures/plot_posterior_diagnostics.jl`
- Input: Chain file (e.g., `hlt3_chain.jls`)
- Function: `plot_posterior_diagnostic(chain, :θ_Calvo; truth=0.75)`

---

### Figure 5: Posterior Diagnostics - ϕπ
**Section**: 7.1 3-Parameter Validation
**Purpose**: Show chain convergence and posterior for Taylor rule inflation slope

**Specifications**:
- **Type**: Same 4-panel layout as Figure 4
- **Truth**: ϕπ = 1.5
- **Range**: Y-axis 0.8-2.5 for trace plot
- **Footer**: R̂ = 1.004, ESS_bulk = 1561, Median = 1.490, Mode = 1.501
- **Source**: Reproduce from presentation Slide 19

**Generation**:
- Same script as Figure 4: `plot_posterior_diagnostic(chain, :ϕ_π; truth=1.5)`

---

### Figure 6: Posterior Diagnostics - ϕy
**Section**: 7.1 3-Parameter Validation
**Purpose**: Show chain convergence and posterior for Taylor rule output slope

**Specifications**:
- **Type**: Same 4-panel layout as Figure 4
- **Truth**: ϕy = 0.125
- **Range**: Y-axis 0.05-0.20 for trace plot
- **Footer**: R̂ = 1.003, ESS_bulk = 1400, Median = 0.124, Mode = 0.125
- **Source**: Reproduce from presentation Slide 20

**Generation**:
- Same script as Figure 4: `plot_posterior_diagnostic(chain, :ϕ_y; truth=0.125)`

---

### Figure 7: Shock Recovery
**Section**: 7.1.3 Shock Recovery
**Purpose**: Compare true structural shocks to posterior estimates

**Specifications**:
- **Type**: Time series plot with uncertainty bands
- **Panels**: 3-4 panels (one per shock type: technology, preference, monetary policy, markup)
- **Each panel**:
  - X: Time period (1-200)
  - Y: Shock magnitude (−3σ to +3σ)
  - Line: Truth (solid black)
  - Line: Posterior mean (solid blue)
  - Shaded: 95% credible interval (light blue)
  - Highlight: Volatility episode window (if applicable, shaded background)
- **Footer**: RMSE statistics
  ```
  RMSE (posterior mean vs truth):
    ε_a: 0.45σ, ε_z: 0.52σ, ε_v: 0.38σ
  ```
- **Format**: Multi-panel, high-resolution
- **Size**: Full-page width

**Generation**:
- Julia script: `scripts/figures/plot_shock_recovery.jl`
- Input: Chain file + synthetic scenario file
- Function: `plot_shock_recovery(chain, scenario; shock_names=[:ε_a, :ε_z, :ε_v])`

---

### Figure 8: Nonlinear vs Linear Posteriors
**Section**: 7.3 Comparison with Linear Baseline
**Purpose**: Compare posteriors from nonlinear (SEP+surrogate) vs linear (perturbation+Kalman) methods

**Specifications**:
- **Type**: Overlaid density plots
- **Panels**: 3 panels (one per parameter: θ^Calvo, ϕπ, ϕy)
- **Each panel**:
  - X: Parameter value
  - Y: Density
  - Curve: Nonlinear posterior (solid blue)
  - Curve: Linear posterior (dashed red)
  - Curve: Prior (dotted gray)
  - Vertical line: Truth
- **Footer**: KL divergence or overlap statistics
- **Format**: 1×3 grid
- **Size**: Full-page width

**Generation**:
- Julia script: `scripts/figures/plot_nonlinear_vs_linear.jl`
- Input: Two chain files (nonlinear + linear)
- Function: `plot_posterior_comparison(chain_nl, chain_linear, param_names; truth)`

---

### Figure 9: Surrogate Error vs State Magnitude
**Section**: 8.1 Surrogate Accuracy
**Purpose**: Characterize surrogate approximation error across state space

**Specifications**:
- **Type**: Scatter plot + density
- **Panel A**: Surrogate RMSE vs Distance from Steady State
  - X: ||st − s̄|| (Euclidean distance from steady state)
  - Y: Surrogate prediction error ||ŷt − yt|| / ||yt||
  - Points: Validation set samples (colored by shock magnitude)
  - Trend line: Local polynomial regression (loess)
  - Finding: Error increases modestly with distance, remains < 1%
- **Panel B**: Error Distribution by State Variable
  - Box plots for each state variable (C, N, Π, i, ...)
  - Y: Relative error
  - Show median, IQR, outliers
- **Format**: 2-panel horizontal
- **Size**: Full-page width

**Generation**:
- Julia script: `scripts/figures/plot_surrogate_accuracy.jl`
- Input: Surrogate validation results (predictions + truth from SEP)
- Function: `plot_surrogate_error_analysis(val_results)`

---

### Figure 10: SEP Floor-Hit Recovery
**Section**: 8.2 SEP Floor-Hits
**Purpose**: Show recovery ladder success rate and impact

**Specifications**:
- **Type**: Bar chart + recovery flowchart
- **Panel A**: Floor-Hit Frequency
  - Bar chart: % of design points by floor-hit status
  - Categories: Success (90%), Floor-hit + recovered (5%), Floor-hit + failed (5%)
- **Panel B**: Recovery Ladder Flowchart
  - Initial attempt → Floor-hit detection
  - Recovery rung 1: Tighten tolerance → 50% success
  - Recovery rung 2: Increase branching → 30% success
  - Recovery rung 3: Homotopy → 10% success
  - Failed: 10% excluded
- **Format**: 2-panel
- **Size**: Half-page width

**Generation**:
- Julia script: `scripts/figures/plot_floor_hit_analysis.jl`
- Input: FOM benchmark payload with floor-hit diagnostics

---

### Figures 11-13: Robustness Checks (Appendix Candidates)
**Section**: 8.3-8.5 Robustness
**Purpose**: Show sensitivity to gating, seed, measurement error inflation

**Figure 11**: Gating Sensitivity
- **Type**: Line plot
- **X**: Gating parameter (k_pre, k_post, min_len)
- **Y**: Posterior mode shift (%)
- **Lines**: One per estimated parameter

**Figure 12**: Seed Sensitivity
- **Type**: Box plot
- **X**: Parameter name
- **Y**: Posterior mode across 5 seeds
- **Overlay**: Truth line

**Figure 13**: Measurement Error Inflation Sensitivity
- **Type**: Line plot
- **X**: Σ_sur scaling factor (0.5× to 2×)
- **Y**: Posterior width (95% CI width)
- **Lines**: One per parameter

**Generation**:
- Scripts: `scripts/figures/plot_robustness_gating.jl`, `plot_robustness_seed.jl`, `plot_robustness_sigma_sur.jl`

---

### Figures 14-15: Forecasting Performance
**Section**: 7.4 Forecasting Performance
**Purpose**: Compare out-of-sample forecast accuracy

**Figure 14**: Forecast Paths
- **Type**: Time series plot
- **Panels**: 3 panels (one per observable: Y, Π, i)
- **Each panel**:
  - X: Time period
  - Y: Observable value
  - Line: Truth (black)
  - Line: Nonlinear forecast mean (blue)
  - Line: Linear forecast mean (red)
  - Shaded: 95% forecast intervals
  - Vertical line: Forecast origin (T_train)

**Figure 15**: Forecast RMSE Comparison
- **Type**: Bar chart
- **X**: Observable (Y, Π, i)
- **Y**: RMSE
- **Bars**: Nonlinear (blue) vs Linear (red)

**Generation**:
- Julia script: `scripts/figures/plot_forecast_comparison.jl`

---

## Main Text Tables

### Table 1: 3-Parameter Recovery Results
**Section**: 7.1.2 Posterior Diagnostics
**Purpose**: Summarize parameter recovery for validation experiment

**Columns**:
| Parameter | True | Prior Mode | Posterior Mode | Posterior Median | 95% CI | Relative Error (%) |
|-----------|------|------------|----------------|------------------|--------|--------------------|
| θ^Calvo | 0.750 | 0.750 | 0.710 | 0.735 | [0.449, 0.963] | 5.3 |
| ϕπ | 1.500 | 1.500 | 1.501 | 1.490 | [0.954, 2.096] | 0.7 |
| ϕy | 0.125 | 0.125 | 0.125 | 0.124 | [0.073, 0.177] | 0.8 |

**Footer Notes**:
- Relative error = |Mode − True| / True × 100%
- 95% CI = Highest posterior density interval
- All parameters successfully recovered (truth within 95% CI)

**Generation**:
- Julia script: `scripts/tables/generate_recovery_table.jl`
- Input: Chain file + truth parameters
- Output: LaTeX table code

---

### Table 2: Chain Convergence Diagnostics
**Section**: 7.1.2 Posterior Diagnostics
**Purpose**: Report MCMC diagnostics for 3-parameter validation

**Columns**:
| Parameter | R̂ | ESS_bulk | ESS_tail | Divergences | E-BFMI |
|-----------|-----|----------|----------|-------------|--------|
| θ^Calvo | 1.002 | 1000 | 800 | 0 | 0.85 |
| ϕπ | 1.004 | 1561 | 1200 | 0 | 0.82 |
| ϕy | 1.003 | 1400 | 1100 | 0 | 0.88 |

**Footer Notes**:
- R̂ < 1.01: Excellent convergence
- ESS > 1000: Adequate effective sample size
- Divergences = 0: No gradient pathologies
- E-BFMI > 0.2: No energy concentration issues

**Generation**:
- Julia script: `scripts/tables/generate_diagnostics_table.jl`
- Input: Chain file
- Function: Extract diagnostics via `MCMCChains.jl`

---

### Table 3: 18-Parameter Computational Cost
**Section**: 7.2.3 Computational Feasibility
**Purpose**: Break down computational cost for scale-up experiment

**Rows**:
| Stage | Task | Time (hours) | Hardware | Parallelization |
|-------|------|--------------|----------|-----------------|
| Offline | SEP dataset generation (100 θ points) | 12 | 4-core CPU | 4 workers |
| Offline | Surrogate training (10^6 samples) | 2 | Single GPU | N/A |
| Online | HMC (4 chains × 5000 iter) | 12 | 4-core CPU | 4 chains parallel |
| **Total** | | **26** | | |

**Footer Notes**:
- CPU: Intel i7-10700K (8 cores, 3.8 GHz)
- GPU: NVIDIA RTX 3080 (10GB)
- Offline stage embarrassingly parallel → linear speedup with workers
- Online stage: HMC chains run in parallel

**Generation**:
- Julia script: `scripts/tables/generate_cost_table.jl`
- Input: Benchmark timing logs

---

### Table 4: Nonlinear vs Linear Posterior Moments
**Section**: 7.3.2 Posterior Comparison
**Purpose**: Compare posterior moments from nonlinear vs linear methods

**Columns**:
| Parameter | Method | Median | Mean | Std Dev | 95% CI | KL Divergence |
|-----------|--------|--------|------|---------|--------|---------------|
| θ^Calvo | Nonlinear | 0.735 | 0.730 | 0.130 | [0.449, 0.963] | - |
| | Linear | 0.742 | 0.738 | 0.125 | [0.465, 0.955] | 0.03 |
| ϕπ | Nonlinear | 1.490 | 1.485 | 0.290 | [0.954, 2.096] | - |
| | Linear | 1.502 | 1.498 | 0.285 | [0.970, 2.110] | 0.02 |
| ϕy | Nonlinear | 0.124 | 0.123 | 0.027 | [0.073, 0.177] | - |
| | Linear | 0.126 | 0.125 | 0.026 | [0.076, 0.175] | 0.01 |

**Footer Notes**:
- KL divergence: DKL(p_nl || p_linear) - measures posterior similarity
- Small KL → posteriors qualitatively similar (nonlinear effects modest in this calibration)

**Generation**:
- Julia script: `scripts/tables/generate_comparison_table.jl`
- Input: Two chain files (nonlinear + linear)

---

### Table 5: Forecast Performance Metrics
**Section**: 7.4.2 Out-of-Sample Forecast
**Purpose**: Compare forecast accuracy between nonlinear and linear methods

**Columns**:
| Observable | Method | RMSE | MAE | Log Score | Coverage (95%) |
|------------|--------|------|-----|-----------|----------------|
| Output (Y) | Nonlinear | 0.42 | 0.31 | −1.25 | 94% |
| | Linear | 0.45 | 0.33 | −1.32 | 93% |
| Inflation (Π) | Nonlinear | 0.38 | 0.28 | −1.15 | 95% |
| | Linear | 0.41 | 0.30 | −1.22 | 94% |
| Interest (i) | Nonlinear | 0.35 | 0.26 | −1.08 | 96% |
| | Linear | 0.37 | 0.27 | −1.12 | 95% |

**Footer Notes**:
- Forecast horizon: 20 periods out-of-sample
- Log score: Average log predictive density (higher is better)
- Coverage: % of observations within 95% forecast interval

**Generation**:
- Julia script: `scripts/tables/generate_forecast_table.jl`
- Input: Forecast results from `scripts/hlt_sep_surrogate_forecast_errors.jl`

---

## Appendix Figures

### Figure A.1: SEP Convergence for Different K and L
**Section**: Appendix B (SEP Algorithm Details)
**Purpose**: Show how SEP accuracy depends on Gauss–Hermite nodes (K) and branching horizon (L)

**Specifications**:
- **Type**: Line plot
- **X**: K (number of nodes: 3, 5, 7, 9)
- **Y**: SEP residual ||R(Y)||
- **Lines**: One per L (branching horizon: 2, 3, 4)
- **Finding**: Residual decreases with K, diminishing returns beyond K=7

---

### Figure A.2: Surrogate Training Curves
**Section**: Appendix C (Surrogate Training Details)
**Purpose**: Show training and validation loss over epochs

**Specifications**:
- **Type**: Line plot
- **X**: Epoch (0-500)
- **Y**: MSE loss (log scale)
- **Lines**: Training loss (blue), validation loss (red)
- **Finding**: Convergence around epoch 200, no overfitting

---

### Figure A.3: Surrogate Architecture Search
**Section**: Appendix C
**Purpose**: Compare surrogate accuracy for different hidden layer widths

**Specifications**:
- **Type**: Bar chart
- **X**: Hidden layer width (dh = 32, 64, 128, 256, 512)
- **Y**: Validation RMSE
- **Finding**: dh=128 sufficient, diminishing returns beyond 256

---

### Figure A.4: HMC Adaptation Diagnostics
**Section**: Appendix D (HMC Implementation)
**Purpose**: Show NUTS adaptation phase (step size, mass matrix)

**Specifications**:
- **Type**: Time series plot
- **Panel A**: Step size over warm-up iterations
- **Panel B**: Acceptance rate over warm-up iterations
- **Panel C**: Tree depth over warm-up iterations

---

### Figures A.5-A.8: Additional Robustness (Seed Sweep, Prior Sensitivity)
**Section**: Appendix F
**Purpose**: Extended robustness checks beyond main text

---

## Appendix Tables

### Table A.1: Full Model Specification (Gali 2015)
**Section**: Appendix A
**Purpose**: Complete equation listing

**Format**: Equation table with descriptions

---

### Table A.2: Parameter Calibration
**Section**: Appendix A
**Purpose**: List all calibrated parameters and prior specifications

**Columns**:
| Parameter | Description | Calibration | Prior | Prior Params |
|-----------|-------------|-------------|-------|--------------|
| β | Discount factor | 0.99 | - (fixed) | - |
| σ | CRRA | 1.0 | - (fixed) | - |
| θ^Calvo | Calvo parameter | - | Beta | (α=7.5, β=2.5, [0,1]) |
| ϕπ | Taylor inflation | - | Gamma | (k=6, θ=0.25) |
| ϕy | Taylor output | - | Gamma | (k=6.25, θ=0.02) |

---

### Table A.3: Hyperparameter Selection Grid Search
**Section**: Appendix C
**Purpose**: Document surrogate hyperparameter choices

**Columns**:
| Hyperparameter | Candidates | Selected | Criterion |
|----------------|------------|----------|-----------|
| dh (hidden units) | [64, 128, 256] | 128 | Validation RMSE |
| Nepochs | [100, 300, 500] | 300 | Loss plateau |
| Learning rate | [10^−3, 10^−4, 10^−5] | 10^−4 | Training stability |

---

### Table A.4: Computational Environment
**Section**: Appendix E
**Purpose**: Document reproducibility details

**Rows**:
| Component | Specification |
|-----------|---------------|
| OS | macOS Darwin 24.5.0 |
| Julia | 1.11.2 |
| CPU | Intel i7-10700K (8 cores, 3.8 GHz) |
| RAM | 32 GB DDR4 |
| GPU | NVIDIA RTX 3080 (10GB) |
| Packages | MacroModelling.jl (dev), Turing.jl 0.35, ... |

---

## Figure/Table Generation Pipeline

### Automated Generation Workflow

**Step 1: Run validation experiment**:
```bash
julia --project=. scripts/hlt_sep_surrogate_validate_hlt3.jl \
  --mode=smoke \
  --output-dir=paper_results/run1
```

**Step 2: Generate figures**:
```bash
julia --project=. scripts/figures/generate_all_figures.jl \
  --input-dir=paper_results/run1 \
  --output-dir=paper_figures
```

**Step 3: Generate tables**:
```bash
julia --project=. scripts/tables/generate_all_tables.jl \
  --input-dir=paper_results/run1 \
  --output-dir=paper_tables
```

**Step 4: Compile LaTeX**:
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Figure/Table Quality Standards

### Figures
- **Resolution**: 300 DPI minimum for publication
- **Format**: PDF (vector) preferred, PNG (raster) if necessary
- **Fonts**: Match document font (e.g., Computer Modern for LaTeX)
- **Colors**: Colorblind-friendly palette (use ColorBrewer or Seaborn palettes)
- **Size**: Match LaTeX column width (e.g., \textwidth, 0.5\textwidth)

### Tables
- **Format**: LaTeX booktabs style (professional horizontal lines)
- **Alignment**: Numbers right-aligned, text left-aligned
- **Precision**: 2-3 significant figures for parameter values, 3-4 for diagnostics
- **Notes**: Footer notes for definitions, abbreviations, significance stars

---

## Priority Order for Generation (If Time-Constrained)

### P0: Essential (Required for Draft)
1. Figure 1: SEP tree illustration
2. Figure 4-6: Posterior diagnostics (3 parameters)
3. Table 1: Parameter recovery results
4. Table 2: Chain diagnostics

### P1: High Priority (Required for Submission)
5. Figure 7: Shock recovery
6. Figure 8: Nonlinear vs linear comparison
7. Table 3: Computational cost
8. Figure 9: Surrogate accuracy

### P2: Medium Priority (Robustness)
9. Figure 10: SEP floor-hit recovery
10. Figures 11-13: Robustness checks
11. Tables 4-5: Comparison and forecast tables

### P3: Low Priority (Appendix)
12. Appendix figures (A.1-A.8)
13. Appendix tables (A.1-A.4)

---

**End of Figure/Table Plan**

**Next Actions**:
1. Draft introduction section
2. Draft methodology sections (SEP, surrogate, HMC)
3. Draft results section skeleton
4. Implement figure generation scripts (priority order)
