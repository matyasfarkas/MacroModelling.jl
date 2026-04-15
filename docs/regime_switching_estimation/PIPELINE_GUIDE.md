# Regime-Switching SEP/ROM Program Overview

Purpose
- End-to-end pipeline to (1) solve/approximate HLT with SEP, (2) train a surrogate, (3) compare ROM vs SEP vs surrogate, and (4) run regime-switching estimation with optional gate calibration.

Core components (files and roles)
1) Models
- `models/Smets_Wouters_2007_HLT.jl`
- `models/Smets_Wouters_2007_HLT_obc.jl`
- Reference calibration: `test/models/SW07_nonlinear.jl`

2) SEP dataset generation (surrogate training data)
- `scripts/hlt_sep_surrogate_dataset_generate.jl`
  - Produces `hlt_sep_surrogate_dataset.jls`
  - Keys: `X` (state_t, shocks_t, theta), `Y` (obs_t+1, state_t+1)
  - Optional ROM baselines: `Y_rom1`, `Y_rom2`
  - Metadata (`meta`): model info, state/obs names, theta grid, SEP settings, shock scaling, ROM options, etc.

3) Surrogate training
- `scripts/hlt_sep_surrogate_train.jl`
  - Trains a 2-layer or 3-layer MLP on dataset.
  - `--rom-residual=1|2` trains on residuals `Y - Y_romX`.
  - Output: `hlt_sep_surrogate_trained.jls` with `frozen`, `meta`, `validation_rmse`.

4) Synthetic data for illustration/estimation
- `scripts/hlt_sep_surrogate_synthetic_data.jl`
  - Produces `hlt_sep_synth_data.jls`
  - Keys: `obs_data`, `s0`, `shocks`, `theta_true`, `obs_sigma`, `shock_sigmas`,
    SEP settings, `vol_*` high-volatility window, and state/obs metadata.

5) Illustration plots (ROM vs SEP vs surrogate)
- `scripts/hlt_regime_switching_illustration.jl`
  - Outputs PDFs:
    - `hlt_regime_switching_series.pdf`
    - `hlt_regime_switching_errors.pdf`
    - `hlt_regime_switching_irf_comparison.pdf`
  - IRF comparison includes ROM1, ROM2, SEP, Surrogate (residual-aware if trained).

6) Gate calibration (for regime switching)
- `scripts/hlt_sep_surrogate_gate_calibration.jl`
  - Input: `hlt_sep_synth_data.jls`
  - Output: `gate_calibration.jls` (defaults to same directory)
  - Stores masks/probabilities and summary stats used by estimation.

7) Synthetic estimation (regime switching)
- `scripts/hlt_sep_surrogate_synthetic_estimation.jl`
  - Uses gate calibration to mix ROM (Kalman) and SEP surrogate loglik.
  - Supports ROM-residual surrogates in loglik (baseline mode only).
  - Writes: `hlt_sep_surrogate_estimation_chain.jls`

8) HMC diagnostics (lightweight)
- `scripts/hlt_sep_surrogate_hmc_diagnose.jl`
  - Runs a smaller HMC diagnostic using the surrogate loglik.
  - Supports ROM-residual surrogates in loglik (baseline mode only).

9) Chain reporting
- `scripts/hlt_sep_surrogate_chain_report.jl`
  - Generates LaTeX report from `hlt_sep_surrogate_estimation_chain.jls`.
  - Uses `SurrogateNN/diagnostics` helpers.

10) ROM helper
- `scripts/hlt_surrogate/hlt_sep_surrogate_rom_utils.jl`
  - ROM baselines for order 1/2, used by dataset, surrogate rollout, and loglik.

Data flow
1) SEP dataset
   - Inputs: HLT model + SEP settings + theta sampling + shocks.
   - Outputs: `hlt_sep_surrogate_dataset.jls` with `X`, `Y`, optional `Y_rom1`, `Y_rom2`.
2) Train surrogate
   - Inputs: dataset; optional residual target via `--rom-residual`.
   - Output: `hlt_sep_surrogate_trained.jls`.
3) Synthetic data
   - Inputs: HLT model + SEP settings + shock scaling.
   - Output: `hlt_sep_synth_data.jls`.
4) Gate calibration
   - Inputs: synthetic data (ROM vs SEP errors, shocks).
   - Output: `gate_calibration.jls`.
5) Estimation/diagnostics
   - Inputs: surrogate, synthetic data, gate calibration.
   - Output: chain file and diagnostics.
6) Illustration
   - Inputs: synthetic data + surrogate.
   - Output: PDFs with series/errors/IRFs.

ROM residual behavior
- Dataset generation stores ROM baselines with `--rom-orders=1,2` (default) and `--rom-mode`:
  - `baseline`: ROM is computed at paper calibration (recommended for AD/HMC).
  - `theta`: ROM computed per theta draw (slow; not used in AD flows).
- Training residual:
  - `--rom-residual=1` trains residual vs ROM1.
  - `--rom-residual=2` trains residual vs ROM2.
- Inference:
  - Surrogate output is added to ROM baseline before computing loglik or IRFs.

Recommended execution order (templates)
1) Generate SEP dataset (with ROM baselines)
```
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --rom-orders=1,2 --rom-mode=baseline \
  --theta-samples=10 --sample-length=80 \
  --sep-horizon=10 --sep-order=1 --sep-nnodes=3 \
  --sep-maxit=80 --sep-tol=1e-5 --shock-scale=0.25 \
  --use-obc --timing
```

2) Train residual surrogate (ROM2)
```
julia --project=. scripts/hlt_sep_surrogate_train.jl \
  scripts/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/hlt_sep_surrogate_dataset.jls \
  --rom-residual=2 --epochs=500 --hidden=256 --hidden2=128
```

3) Synthetic data
```
julia --project=. scripts/hlt_sep_surrogate_synthetic_data.jl \
  --sep-horizon=10 --sep-order=1 --sep-nnodes=3 --sep-maxit=80 --sep-tol=1e-5 \
  --shock-scale=0.25 --use-obc
```

4) Gate calibration
```
julia --project=. scripts/hlt_sep_surrogate_gate_calibration.jl \
  scripts/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/hlt_sep_synth_data.jls
```

5) Illustration (ROM1/ROM2/SEP/Surrogate)
```
julia --project=. scripts/hlt_regime_switching_illustration.jl \
  scripts/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/hlt_sep_synth_data.jls \
  --surrogate=scripts/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/hlt_sep_surrogate_trained.jls \
  --irf-method=extended_path --irf-periods=10 --irf-sep-periods=10
```

6) Synthetic estimation (gated)
```
julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
  scripts/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/hlt_sep_surrogate_trained.jls \
  scripts/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/hlt_sep_synth_data.jls
```

7) HMC diagnostics (optional)
```
julia --project=. scripts/hlt_sep_surrogate_hmc_diagnose.jl \
  scripts/hlt_sep_surrogate_dataset_YYYYMMDD_HHMMSS/hlt_sep_surrogate_trained.jls \
  scripts/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/hlt_sep_synth_data.jls
```

8) Chain report
```
julia --project=. scripts/hlt_sep_surrogate_chain_report.jl \
  scripts/hlt_sep_surrogate_synth_YYYYMMDD_HHMMSS/hlt_sep_surrogate_estimation_chain.jls
```

Key invariants to keep aligned
- Shock scaling: keep `shock_scaling` and `shock_scale` consistent across dataset, surrogate, and IRF comparison.
- Output ordering: datasets always use `[observables_t, state_next]`.
- State definition: `past_not_future_and_mixed + future_not_past_and_mixed`.
- OBC: use `--use-obc` consistently across SEP simulation, synthetic data, and ROM evaluation.

Troubleshooting (common)
- Residual surrogate not improving ROM: ensure dataset was generated with `Y_rom1`/`Y_rom2` and retrain with `--rom-residual`.
- IRF sign/scaling mismatch: check `shock_scaling`, `shock_scale`, and `--irf-shock-scale`.
- Estimation slow or unstable: use `rom_mode=baseline`, shorter horizons, and smaller shock scales for validation runs.
