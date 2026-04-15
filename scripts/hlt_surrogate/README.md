HLT SEP surrogate scripts (ROM1 + delta)

Pipeline scripts
- `scripts/hlt_sep_surrogate_dataset_generate.jl`
  - Generates SEP training data (supports ROM1 residual targets, IRF augmentation).
  - Default output directory is `data/` when `--output-dir` is omitted.
  - **Stable-only sampling**: use `--stable-prefix` to keep only stable SEP periods when SEP fails mid‑path (see below).
- `scripts/hlt_sep_surrogate_train.jl`
  - Trains a surrogate on the dataset (use `--rom-residual=1 --obs-only` for ROM1 delta on observables).
- `scripts/hlt_sep_surrogate_synthetic_data.jl`
  - Creates synthetic SEP data with a high-volatility window.
  - Volatility window is specified on observed periods; shocks are rescaled in total-period indices.
- `scripts/hlt_regime_switching_illustration.jl`
  - Produces series, error, and IRF comparison plots for ROM1 vs ROM1+delta surrogate.
- `scripts/hlt_sep_surrogate_gate_calibration.jl`
  - Calibrates regime-switching gates using synthetic data and ROM vs surrogate errors.
- `scripts/hlt_sep_surrogate_synthetic_estimation.jl`
  - Runs regime-switching estimation on synthetic data (ROM1 + delta surrogate).
- `scripts/hlt_sep_surrogate_chain_report.jl`
  - Generates a LaTeX posterior report with diagnostics and plots.

Shared utilities
- `scripts/hlt_surrogate/hlt_sep_surrogate_nn_utils.jl`: MLP training and frozen predictor helpers.
- `scripts/hlt_surrogate/hlt_sep_surrogate_rom_utils.jl`: ROM cache/build/predict helpers.
- `scripts/hlt_surrogate/hlt_sep_surrogate_cli_utils.jl`: shared CLI parsing helpers.

Recommended run sequence
1) Generate dataset:
```
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl --rom-orders=1 --rom-mode=baseline
```
Stable-only dataset (keep only SEP-stable prefixes if SEP fails mid‑path):
```
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --rom-orders=1 --rom-mode=baseline \
  --sep-shock-scale=0.6 \
  --stable-prefix --stable-min-periods=10
```
2) Train surrogate:
```
julia --project=. scripts/hlt_sep_surrogate_train.jl data/<dataset>/hlt_sep_surrogate_dataset.jls \
  --rom-residual=1 --obs-only --only-full-success \
  --out=data/<dataset>/hlt_sep_surrogate_trained_rom1_resid_obs.jls
```
3) Generate synthetic data with a realized high-volatility episode:
```
julia --project=. scripts/hlt_sep_surrogate_synthetic_data.jl \
  --vol-start=80 --vol-end=120 --vol-mult=3 --shock-scale=0.15 \
  --sep-tol=1e-4 --sep-maxit=120
```
4) Run the illustration:
```
julia --project=. scripts/hlt_regime_switching_illustration.jl \
  --synthetic=data/<synth>/hlt_sep_synth_data.jls \
  --surrogate=data/<dataset>/hlt_sep_surrogate_trained_rom1_resid_obs.jls
```

5) Calibrate gates (optional):
```
julia --project=. scripts/hlt_sep_surrogate_gate_calibration.jl \
  --synthetic=data/<synth>/hlt_sep_synth_data.jls \
  --surrogate=data/<dataset>/hlt_sep_surrogate_trained_rom1_resid_obs.jls
```

6) Run synthetic estimation:
```
julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
  data/<dataset>/hlt_sep_surrogate_trained_rom1_resid_obs.jls \
  data/<synth>/hlt_sep_synth_data.jls
```
