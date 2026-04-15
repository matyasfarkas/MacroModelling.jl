# Quick Start: Regime-Switching Estimation

**Goal**: Get a working example running in 15 minutes and understand what it does.

**What you'll learn**:
- How to run the regime-switching illustration
- What the outputs mean (plots and metrics)
- Where to go next for customization

---

## Prerequisites

### Required Software
- **Julia** 1.9 or higher ([download here](https://julialang.org/downloads/))
- **Git** (to clone the repository)

### Environment Setup

1. **Navigate to the project root**:
```bash
cd /path/to/MacroModelling.jl
```

2. **Activate the Julia environment**:
```julia
julia --project=.
```

3. **Install dependencies** (if not already installed):
```julia
using Pkg
Pkg.instantiate()
```

This will install MacroModelling.jl, Turing.jl, and all required dependencies.

---

## Option 1: Pre-Built Demo (5 minutes)

If you have access to pre-generated data artifacts, this is the fastest way to see results.

### Step 1: Verify Data Artifacts Exist

Check if these files exist:
```bash
ls data/hlt_sep_surrogate_synth_20260104_161020/hlt_sep_synth_data.jls
ls data/hlt_sep_surrogate_dataset_20260103_180108/hlt_sep_surrogate_trained_rom1_resid_obs.jls
```

If these files don't exist, skip to **Option 2** below.

### Step 2: Run Illustration

From the project root:
```bash
julia --project=. scripts/hlt_regime_switching_illustration.jl \
  data/hlt_sep_surrogate_synth_20260104_161020/hlt_sep_synth_data.jls \
  --surrogate=data/hlt_sep_surrogate_dataset_20260103_180108/hlt_sep_surrogate_trained_rom1_resid_obs.jls \
  --irf-method=extended_path --irf-periods=10
```

**Expected runtime**: ~30-60 seconds

### Step 3: View Outputs

The command generates three PDFs in the synthetic data directory:

1. **`hlt_regime_switching_series.pdf`**
   - Observed series over time
   - Highlights high-volatility window (shaded region)

2. **`hlt_regime_switching_errors.pdf`**
   - Forecast errors: ROM1 vs ROM1+Surrogate vs SEP
   - Shows surrogate improves fit in high-volatility window

3. **`hlt_regime_switching_irf_comparison.pdf`**
   - Impulse responses to a productivity shock
   - Compares ROM1, ROM2, SEP, and Surrogate predictions

**Key metrics printed to console**:
```
ROM1 RMSE (full):     X.XX
ROM1 RMSE (high-vol): Y.YY
Surrogate RMSE (full):     X.XX
Surrogate RMSE (high-vol): Z.ZZ  ← Should be ~3% better than ROM1 in high-vol
```

### What Just Happened?

1. **Loaded synthetic observables** generated with SEP (ground truth)
2. **Loaded ROM1+delta surrogate** neural network
3. **Compared predictions**: ROM1 (linear) vs ROM1+Surrogate (hybrid) vs SEP (exact)
4. **Generated IRFs**: Impulse responses showing nonlinear model dynamics

The surrogate should show ~3% RMSE improvement over ROM1 in the high-volatility window (typically periods 80-120).

---

## Option 2: Mini Pipeline (10-15 minutes)

If you don't have pre-built artifacts, run a minimal pipeline to generate them.

This creates a small dataset for speed (~5 minutes total). For production use, you'll want larger datasets (see [Full Pipeline Tutorial](tutorials/FULL_PIPELINE.md)).

### Step 1: Generate SEP Dataset

Create training data for the surrogate:

```bash
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --rom-orders=1,2 --rom-mode=baseline \
  --theta-samples=5 --sample-length=40 \
  --sep-horizon=8 --sep-order=1 --sep-nnodes=3 \
  --sep-maxit=60 --sep-tol=1e-4 --shock-scale=0.25 \
  --use-obc --timing
```

**What this does**:
- Samples 5 parameter draws (`--theta-samples=5`)
- Generates 40 periods per draw (`--sample-length=40`)
- Solves SEP with 8-period horizon (`--sep-horizon=8`)
- Computes ROM1 and ROM2 baselines (`--rom-orders=1,2`)
- Stores dataset in `data/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/`

**Expected runtime**: ~2-3 minutes

**Output directory**: Note the timestamp directory created (e.g., `data/hlt_sep_surrogate_dataset_20260127_103045/`)

### Step 2: Train Surrogate

Train a neural network on the ROM1 residuals:

```bash
julia --project=. scripts/hlt_sep_surrogate_train.jl \
  data/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/hlt_sep_surrogate_dataset.jls \
  --rom-residual=1 --obs-only --epochs=200 --hidden=128 --hidden2=64
```

Replace `YYYYMMDD_HHMMSS` with your actual dataset timestamp.

**What this does**:
- Trains on residual `Y - Y_rom1` (`--rom-residual=1`)
- Uses observables only, not full state (`--obs-only`)
- 2-layer network: 128 → 64 neurons (`--hidden=128 --hidden2=64`)
- 200 training epochs

**Expected runtime**: ~1-2 minutes

**Output**: `hlt_sep_surrogate_trained_rom1_resid_obs.jls` in the same directory

### Step 3: Generate Synthetic Data

Create synthetic observables for validation:

```bash
julia --project=. scripts/hlt_sep_surrogate_synthetic_data.jl \
  --sample-length=60 --sample-start=1 --burn-in=10 \
  --shock-scale=0.25 --sep-horizon=8 --sep-maxit=60 --sep-tol=1e-4 \
  --vol-start=20 --vol-end=40 --vol-mult=3 --vol-shocks=all \
  --use-obc
```

**What this does**:
- Generates 60-period time series
- High-volatility window: periods 20-40 (`--vol-start=20 --vol-end=40`)
- Shocks scaled 3× in high-vol window (`--vol-mult=3`)

**Expected runtime**: ~30 seconds

**Output directory**: `data/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/`

### Step 4: Run Illustration

Now run the comparison (same as Option 1, Step 2):

```bash
julia --project=. scripts/hlt_regime_switching_illustration.jl \
  data/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/hlt_sep_synth_data.jls \
  --surrogate=data/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/hlt_sep_surrogate_trained_rom1_resid_obs.jls \
  --irf-method=extended_path --irf-periods=10
```

Replace timestamps with your actual directories.

**Expected runtime**: ~30-60 seconds

**Outputs**: Same three PDFs as Option 1

---

## Understanding the Outputs

### 1. Series Plot (`hlt_regime_switching_series.pdf`)

Shows the 7 observables over time:
- Output growth (`dy`)
- Consumption growth (`dc`)
- Investment growth (`dinve`)
- Wages (`labobs`)
- Hours worked (`dw`)
- Inflation (`pinfobs`)
- Interest rate (`robs`)

**Shaded region**: High-volatility window where shocks are amplified (3× in the example)

**What to look for**: Series should be stationary, with higher variance in the shaded region.

### 2. Errors Plot (`hlt_regime_switching_errors.pdf`)

Compares forecast errors for each observable:
- **ROM1** (blue): First-order perturbation approximation
- **Surrogate** (orange): ROM1 + neural network correction
- **SEP** (green, if available): Stochastic extended path "ground truth"

**What to look for**: Surrogate errors should be smaller than ROM1, especially in the high-volatility window.

**Key metric**: RMSE improvement ~3% in high-vol window for a well-trained surrogate.

### 3. IRF Comparison (`hlt_regime_switching_irf_comparison.pdf`)

Impulse responses to a 1-standard-deviation productivity shock (`ea`):
- **ROM1** (blue): Linear approximation
- **ROM2** (green): Second-order approximation
- **SEP** (red): Nonlinear perfect foresight
- **Surrogate** (orange): Neural network prediction

7 observables × 10 periods shown.

**What to look for**:
- ROM1 and ROM2 should be similar for small shocks
- SEP may show nonlinear effects (asymmetry, state-dependence)
- Surrogate should approximate SEP closely

**Common patterns**:
- Output, consumption, investment increase (positive productivity shock)
- Hours may fall (wealth effect) or rise (substitution effect)
- Inflation typically falls (marginal cost reduction)
- Interest rate responds (Taylor rule)

---

## What's Next?

### Immediate Next Steps

1. **Explore the plots** in more detail
   - Open the PDFs and examine each observable
   - Check RMSE metrics in console output

2. **Try different parameters** in the mini pipeline:
   - Increase `--theta-samples` to 10 (more training data)
   - Increase `--epochs` to 400 (better training)
   - Change `--vol-mult` to see different volatility regimes

3. **Read the overview** to understand the methodology:
   - [methodology/OVERVIEW.md](methodology/OVERVIEW.md) - Academic explanation
   - [RS_program_overview.md](RS_program_overview.md) - Pipeline details

### Deeper Dives

4. **Full pipeline tutorial**: [tutorials/FULL_PIPELINE.md](tutorials/FULL_PIPELINE.md)
   - Production-scale datasets (theta_samples=50+)
   - Gate calibration for regime switching
   - HMC estimation
   - Chain diagnostics and reporting

5. **Configuration guide**: [tutorials/WORKFLOW_DECISIONS.md](tutorials/WORKFLOW_DECISIONS.md)
   - ROM1 vs ROM2 trade-offs
   - Surrogate architecture choices
   - Shock scaling strategies
   - When to use OBC

6. **Troubleshooting**: [tutorials/TROUBLESHOOTING.md](tutorials/TROUBLESHOOTING.md)
   - SEP convergence issues
   - Surrogate training problems
   - HMC posterior diagnostics

### Research Extensions

7. **Current approaches**: [methodology/CURRENT_APPROACHES.md](methodology/CURRENT_APPROACHES.md)
   - Stable baseline (ROM1+delta)
   - Experimental methods (gate-window sampling)

8. **Development history**: [methodology/DEVELOPMENT_PHASES.md](methodology/DEVELOPMENT_PHASES.md)
   - How the framework evolved
   - Key decisions and alternatives

9. **Active research**: [active_steps/](active_steps/)
   - Latest experiments (Steps 61-64)
   - Open problems and proposed solutions

---

## Troubleshooting Quick Start

### "File not found" errors
**Solution**: Verify you're in the correct directory (`MacroModelling.jl/`) and that paths match your actual timestamp directories.

### Julia package errors
**Solution**: Re-run `Pkg.instantiate()` from `julia --project=.`

### SEP fails to converge
**Solution**: This is normal for some parameter draws. The mini pipeline has forgiving tolerances (`--sep-tol=1e-4`). For production, see [tutorials/TROUBLESHOOTING.md](tutorials/TROUBLESHOOTING.md) "SEP Solver Issues".

### Surrogate doesn't improve ROM
**Solution**:
- Ensure `--rom-residual=1` is used in training
- Try more training epochs (`--epochs=400`)
- Check dataset quality: should have ROM baselines (see Step 1 output)

### Plots don't show improvement
**Solution**:
- Mini pipeline uses small dataset (5 thetas, 40 periods) which may undertrain
- Follow [tutorials/FULL_PIPELINE.md](tutorials/FULL_PIPELINE.md) for production-scale datasets
- Check high-vol window RMSE specifically (overall RMSE may not improve much)

### Out of memory
**Solution**:
- Reduce `--theta-samples` (try 3 instead of 5)
- Reduce `--sample-length` (try 30 instead of 40)
- Close other applications

---

## Summary

You now know how to:
- ✓ Run the regime-switching illustration
- ✓ Generate a minimal SEP dataset
- ✓ Train a ROM-residual surrogate
- ✓ Create synthetic data with volatility windows
- ✓ Interpret plots and RMSE metrics

**Total time**: 15 minutes for Option 2 mini pipeline, 5 minutes for Option 1 pre-built demo

**Next**: [tutorials/FULL_PIPELINE.md](tutorials/FULL_PIPELINE.md) for the complete workflow including estimation.

---

**Questions?** See [tutorials/TROUBLESHOOTING.md](tutorials/TROUBLESHOOTING.md) or [README.md](README.md) "Getting Help"
