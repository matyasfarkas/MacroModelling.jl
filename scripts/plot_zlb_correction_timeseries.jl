#!/usr/bin/env julia
#
# Plot a ZLB-binding simulation showing FOM, ROM1, and ROM1+surrogate trajectories.
# This illustrates HOW the residual correction works in a time-series context.
#
# Usage:
#   julia --project=. scripts/plot_zlb_correction_timeseries.jl

using Serialization, Statistics, LinearAlgebra

const REPO_ROOT = dirname(@__DIR__)
cd(REPO_ROOT)

include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

# ── Load surrogate ──
println("Loading surrogate...")
surr = deserialize(joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls"))
f_zlb = surr["frozen"]

# ── Load ZLB checkpoint ──
println("Loading ZLB dataset...")
dz = deserialize(joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls"))
cursor = dz["cursor"]
X_all = Float64.(dz["X"][:, 1:cursor])
Y_all = Float64.(dz["Y"][:, 1:cursor])
Y_rom1_all = Float64.(dz["Y_rom1"][:, 1:cursor])
theta_ids = dz["theta_ids"][1:cursor]
sep_residuals = Float64.(dz["sep_residuals"][1:cursor])

# Observable names (first 7 dims of Y)
obs_names = ["dy", "dc", "dinve", "labobs", "pinfobs", "dwobs", "robs"]
n_obs = length(obs_names)

# ── Find a theta with substantial ZLB binding ──
function find_best_zlb_theta(Y_all, Y_rom1_all, theta_ids)
    unique_thetas = sort(unique(theta_ids))
    best_theta = 0
    best_score = 0.0
    best_n_binding = 0

    for ti in unique_thetas
        mask = theta_ids .== ti
        idx = findall(mask)
        n_samples = length(idx)
        n_samples < 50 && continue

        robs_resid = Y_all[7, idx] .- Y_rom1_all[7, idx]
        n_binding_proxy = count(abs.(robs_resid) .> 0.5)
        score = sum(abs.(robs_resid))

        if n_binding_proxy > best_n_binding || (n_binding_proxy == best_n_binding && score > best_score)
            best_theta = ti
            best_score = score
            best_n_binding = n_binding_proxy
        end
    end
    return best_theta, best_n_binding
end

# Use theta 3: FOM floors at robs≈0 (ZLB binding) while ROM1 goes to -4.7%
best_theta = 3
println("\nUsing theta $best_theta (known strong ZLB-binding episode)")

# ── Extract trajectory for chosen theta ──
mask = theta_ids .== best_theta
idx = findall(mask)
T = length(idx)

Y_fom = Y_all[1:n_obs, idx]      # FOM (true SEP) observables
Y_rom = Y_rom1_all[1:n_obs, idx]  # ROM1 (linear) observables

# Compute surrogate prediction
Y_resid_surr = predict_frozen_batch(f_zlb, X_all[:, idx])
Y_surr = Y_rom1_all[:, idx] .+ Y_resid_surr
Y_surr_obs = Y_surr[1:n_obs, :]  # ROM1 + surrogate correction

println("Trajectory length: $T periods")
println("robs range (FOM): [$(round(minimum(Y_fom[7,:]), digits=3)), $(round(maximum(Y_fom[7,:]), digits=3))]")
println("robs range (ROM1): [$(round(minimum(Y_rom[7,:]), digits=3)), $(round(maximum(Y_rom[7,:]), digits=3))]")

# ── Generate plots ──
using Plots
gr()

plot_dir = joinpath(REPO_ROOT, "docs", "paper", "Figures")
mkpath(plot_dir)

# Focus on a window around the most interesting ZLB-binding episode
robs_gap = abs.(Y_fom[7, :] .- Y_rom[7, :])
peak_period = argmax(robs_gap)
# Window: 20 periods before and 30 after the peak
t_start = max(1, peak_period - 20)
t_end = min(T, peak_period + 30)
window = t_start:t_end
t_axis = collect(window)

println("Plotting window: periods $t_start to $t_end (peak ZLB gap at period $peak_period)")

# Nice labels
obs_labels = Dict(
    "dy" => "Output growth",
    "dc" => "Consumption growth",
    "dinve" => "Investment growth",
    "labobs" => "Hours worked",
    "pinfobs" => "Inflation",
    "dwobs" => "Wage growth",
    "robs" => "Interest rate"
)

# Color scheme
c_fom = :black
c_rom = :steelblue
c_surr = :firebrick

# --- Main figure: 4-panel plot of key observables ---
key_obs = [7, 3, 1, 5]  # robs, dinve, dy, pinfobs
key_names = [obs_names[i] for i in key_obs]

fig_main = plot(layout=(2,2), size=(1000, 700), dpi=200,
    plot_title="ZLB-Binding Episode: FOM vs ROM1 vs Surrogate",
    plot_titlefontsize=12)

for (pi, oi) in enumerate(key_obs)
    label_fom = pi == 1 ? "FOM (SEP)" : ""
    label_rom = pi == 1 ? "ROM1 (linear)" : ""
    label_surr = pi == 1 ? "ROM1 + surrogate" : ""

    plot!(fig_main, t_axis, Y_fom[oi, window],
        color=c_fom, lw=2.5, label=label_fom, subplot=pi,
        ylabel=obs_labels[obs_names[oi]], xlabel=pi > 2 ? "Period" : "")
    plot!(fig_main, t_axis, Y_rom[oi, window],
        color=c_rom, lw=1.5, ls=:dash, label=label_rom, subplot=pi)
    plot!(fig_main, t_axis, Y_surr_obs[oi, window],
        color=c_surr, lw=1.5, ls=:dot, label=label_surr, subplot=pi)

    # ZLB line for robs panel
    if oi == 7
        hline!(fig_main, [0.0], color=:gray, lw=0.8, ls=:dashdot, label="", subplot=pi)
    end
end

savefig(fig_main, joinpath(plot_dir, "zlb_correction_timeseries_4panel.pdf"))
savefig(fig_main, joinpath(plot_dir, "zlb_correction_timeseries_4panel.png"))
println("  Saved: zlb_correction_timeseries_4panel.pdf/png")

# --- Single-panel interest rate plot (for close-up) ---
fig_robs = plot(t_axis, Y_fom[7, window],
    color=c_fom, lw=3, label="FOM (SEP truth)",
    ylabel="Interest rate (annualized %)",
    xlabel="Simulation period",
    title="Interest Rate: ZLB-Binding Episode",
    size=(800, 400), dpi=200, legend=:topright)
plot!(fig_robs, t_axis, Y_rom[7, window],
    color=c_rom, lw=2, ls=:dash, label="ROM1 (linear)")
plot!(fig_robs, t_axis, Y_surr_obs[7, window],
    color=c_surr, lw=2, ls=:dot, label="ROM1 + surrogate")
hline!(fig_robs, [0.0], color=:gray, lw=1, ls=:dashdot, label="ZLB (0%)")

savefig(fig_robs, joinpath(plot_dir, "zlb_correction_robs_closeup.pdf"))
savefig(fig_robs, joinpath(plot_dir, "zlb_correction_robs_closeup.png"))
println("  Saved: zlb_correction_robs_closeup.pdf/png")

# --- Single-panel investment growth plot ---
fig_dinve = plot(t_axis, Y_fom[3, window],
    color=c_fom, lw=3, label="FOM (SEP truth)",
    ylabel="Investment growth",
    xlabel="Simulation period",
    title="Investment Growth: Nonlinear Amplification During ZLB Episode",
    size=(800, 400), dpi=200, legend=:topleft)
plot!(fig_dinve, t_axis, Y_rom[3, window],
    color=c_rom, lw=2, ls=:dash, label="ROM1 (linear)")
plot!(fig_dinve, t_axis, Y_surr_obs[3, window],
    color=c_surr, lw=2, ls=:dot, label="ROM1 + surrogate")

savefig(fig_dinve, joinpath(plot_dir, "zlb_correction_dinve_closeup.pdf"))
savefig(fig_dinve, joinpath(plot_dir, "zlb_correction_dinve_closeup.png"))
println("  Saved: zlb_correction_dinve_closeup.pdf/png")

# --- Residual decomposition: show the correction itself ---
fig_resid = plot(layout=(1,2), size=(1000, 400), dpi=200,
    plot_title="Neural Network Residual Correction During ZLB Episode")

# Panel 1: robs correction
resid_true_r = Y_fom[7, window] .- Y_rom[7, window]
resid_surr_r = Y_surr_obs[7, window] .- Y_rom[7, window]
plot!(fig_resid, t_axis, resid_true_r,
    color=c_fom, lw=2.5, label="True correction (FOM−ROM1)", subplot=1,
    ylabel="Interest rate correction", xlabel="Period")
plot!(fig_resid, t_axis, resid_surr_r,
    color=c_surr, lw=2, ls=:dot, label="Surrogate correction", subplot=1)
hline!(fig_resid, [0.0], color=:gray, lw=0.5, label="", subplot=1)

# Panel 2: dinve correction
resid_true_d = Y_fom[3, window] .- Y_rom[3, window]
resid_surr_d = Y_surr_obs[3, window] .- Y_rom[3, window]
plot!(fig_resid, t_axis, resid_true_d,
    color=c_fom, lw=2.5, label="True correction", subplot=2,
    ylabel="Investment growth correction", xlabel="Period")
plot!(fig_resid, t_axis, resid_surr_d,
    color=c_surr, lw=2, ls=:dot, label="Surrogate correction", subplot=2)
hline!(fig_resid, [0.0], color=:gray, lw=0.5, label="", subplot=2)

savefig(fig_resid, joinpath(plot_dir, "zlb_residual_correction_timeseries.pdf"))
savefig(fig_resid, joinpath(plot_dir, "zlb_residual_correction_timeseries.png"))
println("  Saved: zlb_residual_correction_timeseries.pdf/png")

println("\n" * "="^70)
println("ALL TIME-SERIES PLOTS SAVED TO: $plot_dir")
println("="^70)
