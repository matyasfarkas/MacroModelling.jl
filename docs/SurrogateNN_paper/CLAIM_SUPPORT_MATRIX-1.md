# Claim Support Matrix

Date: 2026-03-04  
Run root: `/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl/.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023`

Rule: every quantitative claim in the manuscript must be listed here and tagged as `supported` or `TBA`.

| Claim ID | Claim | Evidence file | Table/Figure label | Status | Notes |
|---|---|---|---|---|---|
| C1 | Updated real-data application uses sample window `47:290` (`244` observations). | `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_real_data_payload.jls`; `docs/paper/generated/table_hlt_realdata_gate.tex` | `tab:realdata_gate` | supported | `Total periods = 244`. |
| C2 | Observable mapping/order is `[:dy,:dc,:dinve,:labobs,:pinfobs,:dwobs,:robs]` with `dw -> dwobs`. | `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_real_data_payload.jls` | N/A | supported | Built and validated by payload script/tests. |
| C3 | Gate calibration is non-degenerate and inside bounds `(0.01, 0.99)` with share `0.0983606557`. | `docs/paper/generated/hlt_real_data_summary.json`; `docs/paper/generated/table_hlt_realdata_gate.tex` | `tab:realdata_gate` | supported | `gate_share = 0.0983606557`, nonlinear periods `24`, linear `220`. |
| C4 | Full-depth estimation run completed at target depth (`2000` per chain, `4` chains). | `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/real_data/hlt_sep_surrogate_estimation_chain.jls`; `docs/paper/generated/hlt_real_data_summary.json` | `tab:realdata_runmeta` | supported | `chain_meta.n_samples_per_chain = 2000`, `chain_meta.n_chains = 4`. |
| C5 | Posterior table values from the full-depth run are exactly `cprobp=0.5`, `cindp=0.5`, `curvp=75.0` with zero posterior dispersion. | `docs/paper/generated/table_hlt_realdata_posterior.tex`; `docs/paper/generated/hlt_real_data_summary.json` | `tab:realdata_posterior` | supported | Numerically supported, but indicates degenerate chain movement. |
| C6 | Runtime for full switching estimation step is `1704.3` seconds. | `.local_artifacts/hlt_real_data_runs/hlt_update_20260304_151023/manifests/run_manifest.toml` | `tab:realdata_runmeta` | supported | `steps.switching_estimation.elapsed_s = 1704.3`. |
| C7 | MCMC diagnostics satisfy publication-quality convergence thresholds (`\hat{R}`, ESS, MCSE). | `docs/paper/generated/table_hlt_realdata_mcmc.tex`; `docs/paper/generated/hlt_real_data_summary.json` | `tab:realdata_mcmc` | TBA | Diagnostics are `null` due degenerate chain; no convergence claim supported. |
| C8 | Quantitative comparison versus particle filtering or direct SEP benchmarks. | Dedicated benchmark artifacts (not generated in this run) | N/A | TBA | Must remain `TBA, to be added`. |
| C9 | Any statement on 18-parameter scaling in this revision. | Dedicated 18-parameter artifacts (not generated in this run) | N/A | TBA | Must remain `TBA, to be added`. |
| C10 | Economic interpretation of posterior shifts relative to baseline. | Full posterior + non-degenerate diagnostics | N/A | TBA | With degenerate movement, interpretation claims are not supported. |

## Current Artifact Caveat

The full `2000 x 4` run completed and produced required artifacts, but the posterior chain is degenerate (no movement from initial values). Therefore:

- reporting raw posterior table values is `supported` as a descriptive fact;
- inferential/convergence/economic claims must remain `TBA, to be added`.
