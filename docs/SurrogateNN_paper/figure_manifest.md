# Figure and Table Provenance Map

Scope: active submission paper `docs/SurrogateNN_paper/SurrogateNN_paper.tex` under the default `\paperappendixfalse` compile, plus generated table files that are present under `docs/SurrogateNN_paper/generated/`. This map records matching local raw artifacts and producer scripts found on 2026-06-01 without rerunning heavy jobs.

## Active Short-Paper Figures

| Paper label | Figure file | Producer found | Matching raw artifact found | Status |
|---|---|---|---|---|
| `fig:nonlinearity_decomposition` | `figures/fig_nonlinearity_combined.pdf` | No exact producer found. Related analysis artifact exists at `.local_artifacts/nonlinearity_analysis/fom_rom1_decomposition.jls`. | Partial: `.local_artifacts/nonlinearity_analysis/fom_rom1_decomposition.jls`. | Blocked: figure has no exact checked producer script or manifest tying the PDF to the raw artifact. |
| `fig:all_vars_eqs` | `figures/fig_sep_all_vars_eqs_4sigma.pdf` | No exact producer found by filename search. Related `figures/irf_cache.jls` exists in the figure directory but is not an external raw artifact. | Partial: `docs/SurrogateNN_paper/figures/irf_cache.jls`. | Blocked: active PDF lacks a discoverable producer/raw-artifact chain. |
| Removed from active paper | `figures/fig_rom1_forecast_errors.pdf` | No exact producer found. Related script `scripts/hlt_sep_surrogate_forecast_errors.jl` produces `hlt_surrogate_forecast_errors.pdf`, not this filename. | No exact raw artifact found. Related real-data payloads and ROM1 trajectories exist under `.local_artifacts/hlt_18param_realdata/`. | Removed from the active paper because the visible sample-period mismatch could not be resolved from reproducible artifacts. |
| `fig:decomposition_vs_scale` | `figures/decomposition_vs_shock_scale.pdf` | `scripts/shock_scale_decomposition.jl`. | `.local_artifacts/shock_scale_decomposition/decomposition_by_scale.jls`. | Matched. Current caption already limits support to plotted grid 0.8--1.5 and matches the producer's block-share object. |
| `fig:gali_obc_vs_no_obc` | `figures/gali_obc_vs_no_obc.pdf` | `scripts/gali_sep_decomposition.jl`. | `.local_artifacts/gali_decomposition/gali_decomposition_results.jls`. | Matched, with a residual caption weakness: the script labels the y-axis as `Mean |FOM - ROM1| gap`; units are not otherwise documented in the figure file. |
| `fig:covid_shock_comparison` | `figures/covid_shock_comparison.pdf` | No exact producer found. Related COVID decomposition script/artifact produce different filenames (`covid_shock_decomposition.pdf`, `covid_shock_bar_decomposition.pdf`, `covid_ll_decomposition.pdf`). | Partial: `.local_artifacts/covid_decomposition/covid_decomposition_results.jls`, but no exact link to this PDF. | Blocked: no producer chain for the active PDF. Prior reviews also record a corrupted TFP panel label and undefined dotted bands in the visible figure. |

## Inactive Legacy Appendix Figures In This TeX File

These figures are inside the legacy in-file appendix guarded by `\ifpaperappendix`; they are not active in the default short-paper compile.

| Paper label | Figure file | Producer found | Matching raw artifact found | Status |
|---|---|---|---|---|
| `fig:gap_magnitude_vs_scale` | `figures/gap_magnitude_vs_shock_scale.pdf` | `scripts/shock_scale_decomposition.jl`. | `.local_artifacts/shock_scale_decomposition/decomposition_by_scale.jls`. | Matched, inactive by default. |
| `fig:gali_block_decomposition` | `figures/gali_block_decomposition.pdf` | `scripts/gali_sep_decomposition.jl`. | `.local_artifacts/gali_decomposition/gali_decomposition_results.jls`. | Matched, inactive by default. Caption should say observable-block shares if the legacy appendix is reactivated; the script title is `Gali (no ZLB): observable-block decomposition`. |
| `fig:zlb_binding_simulation` | `figures/fig_zlb_binding_simulation.pdf` | Multiple possible producers: `scripts/zlb_binding_simulation_plot.jl`, `scripts/zlb_replot_only.jl`, `scripts/zlb_sep_nonlinear_simulation.jl`, `scripts/compare_c1_c2_irfs.jl`, `scripts/test_zlb_sep_fix.jl`. | `.local_artifacts/zlb_sep_simulation_cache.jls`. | Ambiguous producer, inactive by default. |
| `fig:zlb_correction_robs` | `figures/zlb_correction_robs_closeup.pdf` | `scripts/plot_zlb_correction_timeseries.jl`. | Combined ZLB surrogate artifacts under `.local_artifacts/hlt_18param_validation_v2_combined/`; no single run manifest found for this PDF. | Partial, inactive by default. |
| `fig:zlb_correction_4panel` | `figures/zlb_correction_timeseries_4panel.pdf` | `scripts/plot_zlb_correction_timeseries.jl`. | Combined ZLB surrogate artifacts under `.local_artifacts/hlt_18param_validation_v2_combined/`; no single run manifest found for this PDF. | Partial, inactive by default. |

## Generated And Input Tables

| Table file | Active location | Producer found | Matching raw artifact found | Status |
|---|---|---|---|---|
| `generated/sep_sensitivity_table.tex` | Not active in the short-paper default compile. Active in `SurrogateNN_technical_appendix.tex`, which is outside this task's write scope. | `scripts/sep_sensitivity_study.jl`. | `.local_artifacts/sep_sensitivity/full_265_20260521/sep_sensitivity_table.tex` and `SEP_SENSITIVITY_SUMMARY.md`. | Matched. The checked generated table already reports the full 265-period, five-draw run and includes the source-artifact note. |
| `generated/table_mode_sensitivity.tex` | Only included by the legacy in-file appendix when `\WITHPAPERAPPENDIX` is defined; not active in the default short-paper compile. | `scripts/mode_sensitivity_report.jl`. | `.local_artifacts/mode_sensitivity/MODE_SENSITIVITY_SUMMARY.md`, `mode_sensitivity.csv`, and listed chain artifacts under `.local_artifacts/hlt_18param_realdata/`. | Matched as a polished table copied from generated artifacts. Not part of the active short paper. |
| `csadjcost_sensitivity_table.tex` | Active short paper via `\input{csadjcost_sensitivity_table.tex}`. | `scripts/csadjcost_sensitivity.jl`. | `.local_artifacts/csadjcost_sensitivity/csadjcost_sensitivity.jls` and `csadjcost_sensitivity_table.tex`. | Matched, with a local presentation edit relative to the raw artifact: the paper copy uses booktabs and expanded headers. Remaining note issue: `FOM-ROM1` should be `FOM--ROM1`, but this file is outside Worker A's owned paths. |

## Remaining Provenance Blockers

- `figures/fig_rom1_forecast_errors.pdf`: removed from the active paper because no exact producer/raw artifact was found and prior review recorded a sample-period conflict.
- `figures/covid_shock_comparison.pdf`: no exact producer/raw artifact link found; visible figure issues from prior review remain: corrupted TFP shock label and undefined dotted reference bands.
- `figures/fig_nonlinearity_combined.pdf`: no exact producer script or run manifest found.
- `figures/fig_sep_all_vars_eqs_4sigma.pdf`: no exact producer script or external raw artifact found.
- `figures/gali_obc_vs_no_obc.pdf`: producer and raw artifact match, but the caption should document the y-axis units/metric more explicitly before submission.
- Legacy appendix figures are inactive under the default compile; if `\WITHPAPERAPPENDIX` is used, `gali_block_decomposition` needs caption harmonization with the producer's observable-block decomposition.
