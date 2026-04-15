#!/usr/bin/env julia
#
# Validate ZLB-trained surrogate and generate correction plots.
#
# Usage:
#   julia --project=. scripts/validate_zlb_surrogate.jl

using Serialization, Statistics, LinearAlgebra

const REPO_ROOT = dirname(@__DIR__)
cd(REPO_ROOT)

include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

# ── Load surrogates ──
println("Loading surrogates...")
surr_zlb = deserialize(joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls"))
surr_v3 = deserialize(joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_rom1_v3.jls"))

f_zlb = surr_zlb["frozen"]
f_v3 = surr_v3["frozen"]

summarize_frozen(f_zlb)
println()
summarize_frozen(f_v3)

# ── Load ZLB test data ──
println("\nLoading ZLB dataset for testing...")
dz = deserialize(joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls"))
cursor = dz["cursor"]
X_test = Float64.(dz["X"][:, 1:cursor])
Y_test = Float64.(dz["Y"][:, 1:cursor])
Y_rom1 = Float64.(dz["Y_rom1"][:, 1:cursor])
sep_residuals = Float64.(dz["sep_residuals"][1:cursor])

# Filter to finite samples
finite_mask = [all(isfinite, view(X_test, :, j)) && all(isfinite, view(Y_test, :, j)) &&
               all(isfinite, view(Y_rom1, :, j)) for j in 1:cursor]
keep = findall(finite_mask)
X_test = X_test[:, keep]
Y_test = Y_test[:, keep]
Y_rom1 = Y_rom1[:, keep]
sep_residuals = sep_residuals[keep]
N = length(keep)
println("Test samples: $N (from ZLB dataset)")

# ── Predict ──
println("\nRunning batch predictions...")
Y_resid_zlb = predict_frozen_batch(f_zlb, X_test)   # residual prediction
Y_resid_v3 = predict_frozen_batch(f_v3, X_test)

Y_pred_zlb = Y_rom1 .+ Y_resid_zlb
Y_pred_v3 = Y_rom1 .+ Y_resid_v3

# True residual (what the NN should learn)
Y_resid_true = Y_test .- Y_rom1

# ── Compute errors ──
err_zlb = Y_pred_zlb .- Y_test
err_v3 = Y_pred_v3 .- Y_test
err_rom1 = Y_rom1 .- Y_test

obs_names = ["dy", "dc", "dinve", "labobs", "pinfobs", "dwobs", "robs"]
n_obs = length(obs_names)

rmse_zlb = [sqrt(mean(err_zlb[d, :].^2)) for d in 1:size(Y_test, 1)]
rmse_v3 = [sqrt(mean(err_v3[d, :].^2)) for d in 1:size(Y_test, 1)]
rmse_rom1 = [sqrt(mean(err_rom1[d, :].^2)) for d in 1:size(Y_test, 1)]
rrmse_zlb = rmse_zlb ./ max.(rmse_rom1, 1e-10)
rrmse_v3 = rmse_v3 ./ max.(rmse_rom1, 1e-10)

# ── Split by high/low residual (proxy for ZLB binding) ──
high_resid_mask = sep_residuals .> 0.01
n_high = count(high_resid_mask)
high_idx = findall(high_resid_mask)
low_idx = findall(.!high_resid_mask)

println("\n" * "="^90)
println("SURROGATE VALIDATION ON ZLB-BINDING TEST DATA ($N samples, $n_high high-residual)")
println("="^90)

println("\nPer-Observable RMSE:")
println("  Observable | ROM1 RMSE | v3 RMSE  | ZLB RMSE | v3 RRMSE | ZLB RRMSE | Improvement")
println("  " * "-"^85)
for i in 1:n_obs
    imp = (rmse_v3[i] - rmse_zlb[i]) / max(rmse_v3[i], 1e-10) * 100
    println("  $(rpad(obs_names[i], 10)) | $(lpad(round(rmse_rom1[i], sigdigits=4), 9)) | " *
            "$(lpad(round(rmse_v3[i], sigdigits=4), 8)) | $(lpad(round(rmse_zlb[i], sigdigits=4), 8)) | " *
            "$(lpad(round(rrmse_v3[i], sigdigits=3), 8)) | $(lpad(round(rrmse_zlb[i], sigdigits=3), 9)) | " *
            "$(round(imp, digits=1))%")
end

# Split analysis
if !isempty(high_idx) && !isempty(low_idx)
    println("\n  --- Split by SEP residual (high > 0.01, proxy for ZLB) ---")
    println("  Observable | v3(low)  | v3(high) | ZLB(low) | ZLB(high)| ZLB helps high?")
    println("  " * "-"^80)
    for i in 1:n_obs
        v3_l = sqrt(mean(err_v3[i, low_idx].^2))
        v3_h = sqrt(mean(err_v3[i, high_idx].^2))
        zl_l = sqrt(mean(err_zlb[i, low_idx].^2))
        zl_h = sqrt(mean(err_zlb[i, high_idx].^2))
        imp = (v3_h - zl_h) / max(v3_h, 1e-10) * 100
        println("  $(rpad(obs_names[i], 10)) | $(lpad(round(v3_l, sigdigits=3), 8)) | " *
                "$(lpad(round(v3_h, sigdigits=3), 8)) | $(lpad(round(zl_l, sigdigits=3), 8)) | " *
                "$(lpad(round(zl_h, sigdigits=3), 8)) | $(round(imp, digits=1))%")
    end
end

mean_rrmse_v3 = mean(rrmse_v3[1:n_obs])
mean_rrmse_zlb = mean(rrmse_zlb[1:n_obs])
println("\n  Mean RRMSE (v3):  $(round(mean_rrmse_v3, sigdigits=3))")
println("  Mean RRMSE (ZLB): $(round(mean_rrmse_zlb, sigdigits=3))")
println("  Improvement: $(round((1 - mean_rrmse_zlb/mean_rrmse_v3)*100, digits=1))%")

# ── Generate plots ──
println("\n" * "="^90)
println("GENERATING PLOTS")
println("="^90)

using Plots
gr()

plot_dir = joinpath(REPO_ROOT, "docs", "paper", "Figures")
mkpath(plot_dir)

# --- Plot 1: Residual correction scatter for investment growth (most important observable) ---
dinve_idx = 3  # investment growth is 3rd observable

fig1 = scatter(
    Y_resid_true[dinve_idx, low_idx], Y_resid_zlb[dinve_idx, low_idx],
    ms=1.5, alpha=0.3, color=:steelblue, label="Non-binding",
    xlabel="True FOM-ROM1 residual (investment growth)",
    ylabel="Surrogate-predicted residual",
    title="Residual Correction: Investment Growth",
    legend=:topleft, dpi=200, size=(700, 500)
)
scatter!(fig1,
    Y_resid_true[dinve_idx, high_idx], Y_resid_zlb[dinve_idx, high_idx],
    ms=2.0, alpha=0.5, color=:firebrick, label="ZLB-binding (resid>0.01)"
)
# 45-degree line
lims = extrema(vcat(Y_resid_true[dinve_idx, :], Y_resid_zlb[dinve_idx, :]))
plot!(fig1, [lims[1], lims[2]], [lims[1], lims[2]], ls=:dash, color=:gray, label="45° line", lw=1.5)

savefig(fig1, joinpath(plot_dir, "zlb_residual_correction_dinve.pdf"))
savefig(fig1, joinpath(plot_dir, "zlb_residual_correction_dinve.png"))
println("  Saved: zlb_residual_correction_dinve.pdf/png")

# --- Plot 2: RRMSE comparison using side-by-side bars ---
x_pos = 1:n_obs
w = 0.35
fig2 = bar(x_pos .- w/2, rrmse_v3[1:n_obs], bar_width=w, label="v3 (no ZLB data)",
    color=:steelblue, alpha=0.8, dpi=200, size=(700, 400))
bar!(fig2, x_pos .+ w/2, rrmse_zlb[1:n_obs], bar_width=w, label="ZLB-trained",
    color=:firebrick, alpha=0.8)
plot!(fig2, xticks=(x_pos, obs_names),
    title="Relative RMSE vs ROM1 Baseline (ZLB test data)",
    ylabel="RRMSE (lower = better)", legend=:topright)
savefig(fig2, joinpath(plot_dir, "zlb_rrmse_comparison.pdf"))
savefig(fig2, joinpath(plot_dir, "zlb_rrmse_comparison.png"))
println("  Saved: zlb_rrmse_comparison.pdf/png")

# --- Plot 3: Residual correction scatter for robs (interest rate) ---
robs_idx = 7

fig3 = scatter(
    Y_resid_true[robs_idx, low_idx], Y_resid_zlb[robs_idx, low_idx],
    ms=1.5, alpha=0.3, color=:steelblue, label="Non-binding",
    xlabel="True FOM-ROM1 residual (interest rate obs)",
    ylabel="Surrogate-predicted residual",
    title="Residual Correction: Interest Rate",
    legend=:topleft, dpi=200, size=(700, 500)
)
scatter!(fig3,
    Y_resid_true[robs_idx, high_idx], Y_resid_zlb[robs_idx, high_idx],
    ms=2.0, alpha=0.5, color=:firebrick, label="ZLB-binding (resid>0.01)"
)
lims3 = extrema(vcat(Y_resid_true[robs_idx, :], Y_resid_zlb[robs_idx, :]))
plot!(fig3, [lims3[1], lims3[2]], [lims3[1], lims3[2]], ls=:dash, color=:gray, label="45° line", lw=1.5)

savefig(fig3, joinpath(plot_dir, "zlb_residual_correction_robs.pdf"))
savefig(fig3, joinpath(plot_dir, "zlb_residual_correction_robs.png"))
println("  Saved: zlb_residual_correction_robs.pdf/png")

# --- Plot 4: Error distribution: CDF of absolute errors for dinve ---
abs_err_zlb_dinve = sort(abs.(err_zlb[dinve_idx, :]))
abs_err_v3_dinve = sort(abs.(err_v3[dinve_idx, :]))
abs_err_rom_dinve = sort(abs.(err_rom1[dinve_idx, :]))
cdf_x = range(0, 1, length=N)

fig4 = plot(
    abs_err_rom_dinve, cdf_x,
    label="ROM1 (linear)", color=:gray, lw=2,
    xlabel="Absolute error (investment growth)",
    ylabel="Cumulative fraction",
    title="Error CDF: Investment Growth on ZLB Data",
    legend=:bottomright, dpi=200, size=(700, 500),
    xlim=(0, quantile(abs_err_rom_dinve, 0.99))
)
plot!(fig4, abs_err_v3_dinve, cdf_x, label="v3 surrogate", color=:steelblue, lw=2)
plot!(fig4, abs_err_zlb_dinve, cdf_x, label="ZLB-trained surrogate", color=:firebrick, lw=2)

savefig(fig4, joinpath(plot_dir, "zlb_error_cdf_dinve.pdf"))
savefig(fig4, joinpath(plot_dir, "zlb_error_cdf_dinve.png"))
println("  Saved: zlb_error_cdf_dinve.pdf/png")

# --- Plot 5: Correction magnitude vs distance from steady state ---
# Compute L2 norm of state displacement from mean (proxy for SS distance)
state_part = X_test[1:51, :]  # first 51 dims are state+shock before theta
state_mean = vec(mean(state_part, dims=2))
state_dist = [norm(state_part[:, j] .- state_mean) for j in 1:N]

resid_magnitude = [norm(Y_resid_true[:, j]) for j in 1:N]

fig5 = scatter(
    state_dist[low_idx], resid_magnitude[low_idx],
    ms=1.5, alpha=0.2, color=:steelblue, label="Non-binding",
    xlabel="State displacement from mean (L2 norm)",
    ylabel="True residual magnitude (L2 norm)",
    title="Nonlinear Correction Size vs State Displacement",
    legend=:topleft, dpi=200, size=(700, 500)
)
scatter!(fig5,
    state_dist[high_idx], resid_magnitude[high_idx],
    ms=2.0, alpha=0.5, color=:firebrick, label="ZLB-binding"
)

savefig(fig5, joinpath(plot_dir, "zlb_correction_vs_displacement.pdf"))
savefig(fig5, joinpath(plot_dir, "zlb_correction_vs_displacement.png"))
println("  Saved: zlb_correction_vs_displacement.pdf/png")

# --- Plot 6: Per-observable RRMSE split by ZLB-binding vs non-binding ---
rrmse_high_zlb = Float64[]
rrmse_low_zlb = Float64[]
rrmse_high_v3 = Float64[]
rrmse_low_v3 = Float64[]

for i in 1:n_obs
    rom_h = sqrt(mean(err_rom1[i, high_idx].^2))
    rom_l = sqrt(mean(err_rom1[i, low_idx].^2))
    push!(rrmse_high_zlb, sqrt(mean(err_zlb[i, high_idx].^2)) / max(rom_h, 1e-10))
    push!(rrmse_low_zlb, sqrt(mean(err_zlb[i, low_idx].^2)) / max(rom_l, 1e-10))
    push!(rrmse_high_v3, sqrt(mean(err_v3[i, high_idx].^2)) / max(rom_h, 1e-10))
    push!(rrmse_low_v3, sqrt(mean(err_v3[i, low_idx].^2)) / max(rom_l, 1e-10))
end

fig6 = bar(x_pos .- w/2, rrmse_low_zlb, bar_width=w, label="Non-binding",
    color=:steelblue, alpha=0.8, dpi=200, size=(700, 400))
bar!(fig6, x_pos .+ w/2, rrmse_high_zlb, bar_width=w, label="ZLB-binding",
    color=:firebrick, alpha=0.8)
plot!(fig6, xticks=(x_pos, obs_names),
    title="ZLB-Trained Surrogate RRMSE: Binding vs Non-binding",
    ylabel="RRMSE (lower = better)", legend=:topright)
savefig(fig6, joinpath(plot_dir, "zlb_rrmse_binding_split.pdf"))
savefig(fig6, joinpath(plot_dir, "zlb_rrmse_binding_split.png"))
println("  Saved: zlb_rrmse_binding_split.pdf/png")

println("\n" * "="^90)
println("ALL PLOTS SAVED TO: $plot_dir")
println("="^90)
