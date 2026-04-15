#!/usr/bin/env julia
#
# HLT Nonlinearity -- Deep Analysis (v2)
# ========================================
#
# New angles on SW07-HLT nonlinearity beyond equation-block decomposition.
# Uses existing 43,699-sample dataset (shock_scale=0.1) with FOM (SEP) and ROM1.
#
# FIXES from v1:
#   - delta_norm computed on obs-only portion (not full Y including states)
#   - Per-state-variable nonlinearity contribution added
#   - All charts have self-contained titles, axis labels, and interpretive notes
#   - Top-5 state variables by nonlinearity contribution figure added
#
# Produces 6 figures:
#   5. Temporal Clustering of Nonlinearity (ACF + burst histogram)
#   6. State-Conditional Amplification (expansion vs recession)
#   7. Shock Interaction Heatmap (dataset-based FOM-ROM1, super-additivity)
#   8. 3D Nonlinearity Surface (state distance x shock magnitude -> delta)
#   NEW:
#   8b. Per-State-Variable Nonlinearity Contribution (top-10 ranked)
#   8c. State-Observable Nonlinearity Coupling (which states predict which obs gaps)
#
# Usage:
#   julia --project=. scripts/hlt_nonlinearity_deep_analysis.jl

using MacroModelling
using Plots
using Serialization
using Statistics
using LinearAlgebra
using Printf
using StatsBase
using Random

# -- Setup --
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const FIG_DIR = joinpath(REPO_ROOT, "docs", "paper", "Figures")
mkpath(FIG_DIR)

gr()
default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
        tickfontsize=9, legendfontsize=8, linewidth=1.5, dpi=300)

# -- Load model --
include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "hlt_model_loader_utils.jl"))
const HLT = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
MacroModelling.solve!(HLT, silent = true)

# -- Load dataset --
const DATASET_PATH = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2/hlt_sep_surrogate_dataset_checkpoint.jls")

println("Loading dataset: $DATASET_PATH")
chk = deserialize(DATASET_PATH)

X = chk["X"]          # (d_state + d_eps + d_theta) x N samples
Y = chk["Y"]          # (d_obs + d_state_y) x N -- FOM outputs
Y_rom1 = chk["Y_rom1"] # (d_obs + d_state_y) x N -- ROM1 outputs
theta_ids = chk["theta_ids"]
cursor = chk["cursor"]

# Trim to actual data
X = X[:, 1:cursor]
Y = Y[:, 1:cursor]
Y_rom1 = Y_rom1[:, 1:cursor]
theta_ids = theta_ids[1:cursor]

N_samples = cursor
println("Dataset loaded: $N_samples samples")

# -- Derive dimensions from checkpoint --
settings = chk["settings"]
theta_names_chk = get(settings, "theta_names", Symbol[])
state_names_chk = get(settings, "state_names", Symbol[])
d_theta = length(theta_names_chk)
d_x = size(X, 1)  # total X dimension = d_state_x + d_eps + d_theta

# Infer d_eps from model name in settings
model_name_chk = get(settings, "model_name", "Smets_Wouters_2007_HLT_obc")
if contains(string(model_name_chk), "obc") || contains(string(model_name_chk), "zlb")
    structural_exo = [s for s in HLT.exo if !contains(string(s), "ᵒᵇᶜ")]
    d_eps = length(structural_exo)
else
    d_eps = 7  # SW07 has 7 structural shocks
end
d_state_x = d_x - d_eps - d_theta  # states in X

println("  X layout: d_state_x=$d_state_x, d_eps=$d_eps, d_theta=$d_theta (total=$d_x)")

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
d_obs = length(observables)
d_y = size(Y, 1)
d_state_y = d_y - d_obs

println("  Y layout: d_obs=$d_obs, d_state_y=$d_state_y (total=$d_y)")

# State labels from checkpoint
if !isempty(state_names_chk)
    state_labels = string.(state_names_chk)
else
    state_labels = ["state_$i" for i in 1:d_state_y]
end
println("  State variables ($d_state_y): ", join(state_labels[1:min(8, d_state_y)], ", "),
        d_state_y > 8 ? " ..." : "")

# -- CRITICAL FIX: Separate obs and state portions of delta --
delta_full = Y .- Y_rom1
delta_obs = delta_full[1:d_obs, :]           # obs-only
delta_state = delta_full[d_obs+1:end, :]     # state-only

# delta_norm on obs-only (this is what matters for likelihood / estimation)
delta_norm_obs = [norm(delta_obs[:, i]) for i in 1:N_samples]
# delta_norm on full (for decomposition accounting)
delta_norm_full = [norm(delta_full[:, i]) for i in 1:N_samples]
# delta_norm on state-only
delta_norm_state = [norm(delta_state[:, i]) for i in 1:N_samples]

println("\n  delta_norm DIAGNOSTIC:")
@printf("    Full Y:      median=%.6f, mean=%.6f, max=%.6f\n",
        median(delta_norm_full), mean(delta_norm_full), maximum(delta_norm_full))
@printf("    Obs only:    median=%.6f, mean=%.6f, max=%.6f\n",
        median(delta_norm_obs), mean(delta_norm_obs), maximum(delta_norm_obs))
@printf("    State only:  median=%.6f, mean=%.6f, max=%.6f\n",
        median(delta_norm_state), mean(delta_norm_state), maximum(delta_norm_state))
@printf("    Obs share of total MSE: %.1f%%\n",
        sum(delta_norm_obs.^2) / max(sum(delta_norm_full.^2), 1e-20) * 100)

# -- Extract shocks and states from X --
shock_start = d_state_x + 1
shock_end = d_state_x + d_eps
shock_matrix = X[shock_start:shock_end, :]  # (d_eps x N)
state_matrix = X[1:d_state_x, :]  # (d_state_x x N)
state_dist = [norm(state_matrix[:, i]) for i in 1:N_samples]
shock_mag = [norm(shock_matrix[:, i]) for i in 1:N_samples]

# ============================================================================
# FIGURE 5: Temporal Clustering of Nonlinearity
# ============================================================================
println("\n" * "="^60)
println("FIGURE 5: Temporal Clustering of Nonlinearity")
println("="^60)

# Group samples by theta_id to get per-trajectory sequences
unique_thetas = sort(unique(theta_ids))
println("  Unique thetas: $(length(unique_thetas))")

fig5 = plot(layout=(1, 2), size=(900, 400), margin=6Plots.mm, bottom_margin=10Plots.mm)

# Compute ACF of delta_norm sequences within each trajectory
all_acf_values = zeros(20)  # up to lag 20
acf_count = 0
burst_lengths = Int[]
median_delta = median(delta_norm_obs)

for tid in unique_thetas
    mask = findall(theta_ids .== tid)
    if length(mask) < 25  # need enough points for meaningful ACF
        continue
    end

    seq = delta_norm_obs[mask]
    n = length(seq)

    # ACF
    centered = seq .- mean(seq)
    var_seq = Statistics.var(seq)
    if var_seq < 1e-20
        continue
    end

    for lag in 1:min(20, n-1)
        acf_val = sum(centered[1:n-lag] .* centered[lag+1:n]) / ((n - lag) * var_seq)
        global all_acf_values[lag] += acf_val
    end
    global acf_count += 1

    # Burst lengths: consecutive periods where delta_norm > median
    above = seq .> median_delta
    current_burst = 0
    for i in 1:length(above)
        if above[i]
            current_burst += 1
        else
            if current_burst > 0
                push!(burst_lengths, current_burst)
            end
            current_burst = 0
        end
    end
    if current_burst > 0
        push!(burst_lengths, current_burst)
    end
end

if acf_count > 0
    all_acf_values ./= acf_count
end

# Panel 1: ACF
bar!(fig5[1], 1:20, all_acf_values, color=:steelblue, alpha=0.7, label="",
     bar_width=0.6)
# 95% confidence band
avg_len = N_samples / max(length(unique_thetas), 1)
ci = 1.96 / sqrt(avg_len)
hline!(fig5[1], [ci, -ci], color=:red, linestyle=:dash, label="95% CI", linewidth=1)
hline!(fig5[1], [0], color=:black, linestyle=:dot, label="", linewidth=0.5)
plot!(fig5[1], title="(a) Autocorrelation of ||delta_obs|| Within Trajectories",
      xlabel="Lag (periods)", ylabel="Autocorrelation",
      ylims=(-0.1, 1.0))
@printf("  ACF(1) = %.4f, ACF(2) = %.4f, ACF(5) = %.4f\n",
        all_acf_values[1], all_acf_values[2], all_acf_values[5])

# Panel 2: Burst length histogram
if !isempty(burst_lengths)
    max_burst = min(maximum(burst_lengths), 50)
    histogram!(fig5[2], burst_lengths, bins=1:max_burst+1, color=:coral, alpha=0.7,
               label="", normalize=:probability)
    vline!(fig5[2], [mean(burst_lengths)], color=:black, linewidth=2, linestyle=:dash,
           label="mean=$(round(mean(burst_lengths), digits=1))")
    vline!(fig5[2], [median(burst_lengths)], color=:darkblue, linewidth=1.5, linestyle=:dot,
           label="median=$(round(median(Float64.(burst_lengths)), digits=1))")
    plot!(fig5[2], title="(b) Burst Lengths (delta > median)",
          xlabel="Consecutive periods above median", ylabel="Frequency",
          xlims=(0, min(max_burst, 50)))
    @printf("  Mean burst length: %.1f, Median: %.1f, Max: %d, Total bursts: %d\n",
            mean(burst_lengths), median(Float64.(burst_lengths)),
            maximum(burst_lengths), length(burst_lengths))
end

plot!(fig5, plot_title="Temporal Clustering of FOM-ROM1 Gap")
savefig(fig5, joinpath(FIG_DIR, "hlt_temporal_clustering.pdf"))
println("  Saved: hlt_temporal_clustering.pdf")

# ============================================================================
# FIGURE 6: State-Conditional Amplification
# ============================================================================
println("\n" * "="^60)
println("FIGURE 6: State-Conditional Amplification")
println("="^60)

fig6 = plot(layout=(1, 2), size=(900, 400), margin=6Plots.mm, bottom_margin=10Plots.mm)

# Use output (dy) from Y to classify expansion vs recession.
dy_vals = Y[1, :]  # dy is the first observable
dy_median = median(dy_vals)
expansion = dy_vals .> dy_median
recession = dy_vals .<= dy_median

n_exp = sum(expansion)
n_rec = sum(recession)
println("  Expansion periods: $n_exp, Recession periods: $n_rec")
println("  dy median: $(@sprintf("%.4f", dy_median))")

# Structural shock ordering in the dataset (non-OBC model, 7 shocks):
structural_shock_names = [:ea, :eb, :eg, :eqs, :em, :epinf, :ew]

# Helper: binned conditional mean
function binned_conditional_means(shock_vals, delta_norms, mask; n_bins=10)
    sm = abs.(shock_vals[mask])
    dn = delta_norms[mask]
    perm = sortperm(sm)
    sm_sorted = sm[perm]
    dn_sorted = dn[perm]

    bin_size = max(1, length(sm_sorted) / n_bins)
    bin_x = Float64[]
    bin_y = Float64[]
    for b in 1:n_bins
        i_start = round(Int, (b-1) * bin_size) + 1
        i_end = round(Int, b * bin_size)
        i_end = min(i_end, length(sm_sorted))
        i_start > length(sm_sorted) && break
        i_start > i_end && break
        push!(bin_x, mean(sm_sorted[i_start:i_end]))
        push!(bin_y, mean(dn_sorted[i_start:i_end]))
    end
    return bin_x, bin_y
end

begin
    # Panel 1: monetary shock (em) -- index 5 in structural shocks
    em_shock_idx = findfirst(==(:em), structural_shock_names)

    if !isnothing(em_shock_idx) && em_shock_idx <= d_eps
        for (label, mask, col, ms) in [
            ("Expansion (n=$n_exp)", expansion, :steelblue, :circle),
            ("Recession (n=$n_rec)", recession, :indianred, :diamond)
        ]
            sum(mask) < 10 && continue
            bx, by = binned_conditional_means(shock_matrix[em_shock_idx, :], delta_norm_obs, mask)
            plot!(fig6[1], bx, by, label=label, color=col, linewidth=2, marker=ms,
                  markersize=3)
        end
        plot!(fig6[1], title="(a) Monetary Shock (em)",
              xlabel="|shock magnitude| (std devs)",
              ylabel="Mean ||delta_obs||")
    end

    # Panel 2: investment shock (eqs) -- index 4 in structural shocks
    eqs_shock_idx = findfirst(==(:eqs), structural_shock_names)

    if !isnothing(eqs_shock_idx) && eqs_shock_idx <= d_eps
        for (label, mask, col, ms) in [
            ("Expansion (n=$n_exp)", expansion, :steelblue, :circle),
            ("Recession (n=$n_rec)", recession, :indianred, :diamond)
        ]
            sum(mask) < 10 && continue
            bx, by = binned_conditional_means(shock_matrix[eqs_shock_idx, :], delta_norm_obs, mask)
            plot!(fig6[2], bx, by, label=label, color=col, linewidth=2, marker=ms,
                  markersize=3)
        end
        plot!(fig6[2], title="(b) Investment Shock (eqs)",
              xlabel="|shock magnitude| (std devs)",
              ylabel="Mean ||delta_obs||")
    end
end  # begin block

plot!(fig6, plot_title="State-Conditional Amplification: Expansion vs Recession")
savefig(fig6, joinpath(FIG_DIR, "hlt_state_conditional_amplification.pdf"))
println("  Saved: hlt_state_conditional_amplification.pdf")

# Diagnostic: check asymmetry magnitude
if n_exp > 0 && n_rec > 0
    @printf("  Mean delta_obs_norm -- Expansion: %.6f, Recession: %.6f, Ratio: %.3f\n",
            mean(delta_norm_obs[expansion]), mean(delta_norm_obs[recession]),
            mean(delta_norm_obs[recession]) / max(mean(delta_norm_obs[expansion]), 1e-15))
end

# ============================================================================
# FIGURE 7: Shock Interaction Heatmap (Dataset-Based FOM vs ROM1)
# ============================================================================
println("\n" * "="^60)
println("FIGURE 7: Shock Interaction Heatmap (FOM-ROM1 from dataset)")
println("="^60)

# Use the dataset's actual shock values and FOM-ROM1 gap to show how
# pairs of shocks interact in producing nonlinearity.
# Bin samples by quantiles of two shock magnitudes and compute mean delta_norm.

# Shock pairs to analyze
shock_pair_indices = [(5, 4, "Monetary (em) x Investment (eqs)"),
                      (5, 1, "Monetary (em) x TFP (ea)"),
                      (4, 1, "Investment (eqs) x TFP (ea)")]

n_hmap_bins = 8

fig7 = plot(layout=(1, 3), size=(1200, 400), margin=6Plots.mm, bottom_margin=10Plots.mm)

for (pi, (s1_idx, s2_idx, pair_label)) in enumerate(shock_pair_indices)
    println("  Computing $pair_label heatmap...")

    if s1_idx > d_eps || s2_idx > d_eps
        println("    WARNING: shock index out of range, skipping")
        continue
    end

    s1_vals = shock_matrix[s1_idx, :]
    s2_vals = shock_matrix[s2_idx, :]

    # Create 2D bins based on shock values
    s1_edges = range(quantile(s1_vals, 0.02), quantile(s1_vals, 0.98), length=n_hmap_bins+1)
    s2_edges = range(quantile(s2_vals, 0.02), quantile(s2_vals, 0.98), length=n_hmap_bins+1)

    delta_grid = fill(NaN, n_hmap_bins, n_hmap_bins)

    for i in 1:n_hmap_bins
        for j in 1:n_hmap_bins
            mask = (s1_vals .>= s1_edges[i]) .& (s1_vals .< s1_edges[i+1]) .&
                   (s2_vals .>= s2_edges[j]) .& (s2_vals .< s2_edges[j+1])
            if sum(mask) >= 3
                delta_grid[j, i] = mean(delta_norm_obs[mask])
            end
        end
    end

    s1_centers = [(s1_edges[i] + s1_edges[i+1]) / 2 for i in 1:n_hmap_bins]
    s2_centers = [(s2_edges[j] + s2_edges[j+1]) / 2 for j in 1:n_hmap_bins]

    delta_grid_plot = replace(delta_grid, NaN => 0.0)
    peak_val = maximum(filter(!isnan, delta_grid))
    println("    Peak FOM-ROM1 gap in $pair_label: $(@sprintf("%.6f", peak_val))")

    # Check super-additivity: are corners (both shocks large) worse than edges (one shock large)?
    corner_vals = filter(!isnan, [delta_grid[1,1], delta_grid[1,end], delta_grid[end,1], delta_grid[end,end]])
    edge_vals = filter(!isnan, vcat(delta_grid[1,:], delta_grid[end,:], delta_grid[:,1], delta_grid[:,end]))
    if !isempty(corner_vals) && !isempty(edge_vals)
        corner_mean = mean(corner_vals)
        edge_mean = mean(edge_vals)
        @printf("    Corner mean: %.6f, Edge mean: %.6f (ratio: %.2f)\n",
                corner_mean, edge_mean, corner_mean / max(edge_mean, 1e-15))
    end

    s1_name = structural_shock_names[s1_idx]
    s2_name = structural_shock_names[s2_idx]

    # Format tick labels to avoid overlap
    s1_ticks = round.(s1_centers, sigdigits=2)
    s2_ticks = round.(s2_centers, sigdigits=2)
    heatmap!(fig7[pi], s1_centers, s2_centers, delta_grid_plot,
             color=:viridis, xlabel=string(s1_name),
             ylabel=string(s2_name),
             title="($('a' + pi - 1)) " * pair_label,
             colorbar_title="mean ||delta_obs||",
             xticks=(s1_centers[1:2:end], [@sprintf("%.3f", v) for v in s1_ticks[1:2:end]]),
             yticks=(s2_centers[1:2:end], [@sprintf("%.3f", v) for v in s2_ticks[1:2:end]]),
             xrotation=30)
end

plot!(fig7, plot_title="Shock Interaction: FOM-ROM1 Nonlinearity Gap")
savefig(fig7, joinpath(FIG_DIR, "hlt_shock_interaction_heatmap.pdf"))
println("  Saved: hlt_shock_interaction_heatmap.pdf")

# ============================================================================
# FIGURE 8: Nonlinearity Surface
# ============================================================================
println("\n" * "="^60)
println("FIGURE 8: Nonlinearity Surface")
println("="^60)

fig8 = plot(layout=(1, 2), size=(1000, 450), margin=6Plots.mm, bottom_margin=10Plots.mm)

# Panel 1: Contour plot -- state distance x shock magnitude -> delta_norm_obs
n_contour_bins = 30
sd_edges = range(quantile(state_dist, 0.01), quantile(state_dist, 0.99), length=n_contour_bins+1)
sm_edges = range(quantile(shock_mag, 0.01), quantile(shock_mag, 0.99), length=n_contour_bins+1)

contour_z = fill(NaN, n_contour_bins, n_contour_bins)

for i in 1:n_contour_bins
    for j in 1:n_contour_bins
        mask = (state_dist .>= sd_edges[i]) .& (state_dist .< sd_edges[i+1]) .&
               (shock_mag .>= sm_edges[j]) .& (shock_mag .< sm_edges[j+1])
        if sum(mask) >= 3
            contour_z[j, i] = mean(delta_norm_obs[mask])
        end
    end
end

sd_centers = [(sd_edges[i] + sd_edges[i+1]) / 2 for i in 1:n_contour_bins]
sm_centers = [(sm_edges[j] + sm_edges[j+1]) / 2 for j in 1:n_contour_bins]

# Replace NaN for plotting
contour_z_plot = replace(contour_z, NaN => 0.0)

contourf!(fig8[1], sd_centers, sm_centers, contour_z_plot, color=:viridis,
          levels=15, xlabel="||state deviation from SS||",
          ylabel="||shock vector||",
          title="(a) Nonlinearity Landscape (obs-only delta)",
          colorbar_title="Mean ||delta_obs||")

# Panel 2: Scatter with color = delta_norm (subsample for readability)
n_plot = min(5000, N_samples)
Random.seed!(42)
plot_idx = randperm(N_samples)[1:n_plot]

scatter!(fig8[2], state_dist[plot_idx], shock_mag[plot_idx],
         zcolor=log10.(delta_norm_obs[plot_idx] .+ 1e-12),
         color=:viridis, markersize=1.5, markerstrokewidth=0, alpha=0.6,
         xlabel="||state deviation from SS||", ylabel="||shock vector||",
         title="(b) Per-Sample Nonlinearity (n=$n_plot subsample)",
         label="", colorbar_title="log10(||delta_obs||)")

# R-squared of state_dist + shock_mag as predictor of delta_norm_obs
X_reg = hcat(ones(N_samples), state_dist, shock_mag, state_dist .* shock_mag)
beta_reg = X_reg \ delta_norm_obs
yhat = X_reg * beta_reg
ss_res = sum((delta_norm_obs .- yhat).^2)
ss_tot = sum((delta_norm_obs .- mean(delta_norm_obs)).^2)
r2 = 1.0 - ss_res / ss_tot
@printf("  R-squared (state_dist + shock_mag + interaction -> delta_obs_norm): %.4f\n", r2)

# State distance alone
X_sd = hcat(ones(N_samples), state_dist)
beta_sd = X_sd \ delta_norm_obs
yhat_sd = X_sd * beta_sd
r2_sd = 1.0 - sum((delta_norm_obs .- yhat_sd).^2) / ss_tot
@printf("  R-squared (state_dist alone -> delta_obs_norm): %.4f\n", r2_sd)

# Shock magnitude alone
X_sm = hcat(ones(N_samples), shock_mag)
beta_sm = X_sm \ delta_norm_obs
yhat_sm = X_sm * beta_sm
r2_sm = 1.0 - sum((delta_norm_obs .- yhat_sm).^2) / ss_tot
@printf("  R-squared (shock_mag alone -> delta_obs_norm): %.4f\n", r2_sm)

plot!(fig8, plot_title="Nonlinearity Surface: State Distance x Shock Magnitude (R2 = $(@sprintf("%.3f", r2)))")
savefig(fig8, joinpath(FIG_DIR, "hlt_nonlinearity_surface.pdf"))
println("  Saved: hlt_nonlinearity_surface.pdf")

# ============================================================================
# FIGURE 8b: Per-State-Variable Nonlinearity Contribution (NEW)
# ============================================================================
println("\n" * "="^60)
println("FIGURE 8b: Per-State-Variable Nonlinearity Contribution")
println("="^60)

# delta_state is (d_state_y x N_samples)
# Compute per-state-variable contribution to total state delta-squared
state_mse = zeros(d_state_y)
for si in 1:d_state_y
    global state_mse[si] = mean(delta_state[si, :].^2)
end
total_state_mse = sum(state_mse)
state_share = state_mse ./ max(total_state_mse, 1e-20)

# Also compute per-obs contribution
obs_mse = zeros(d_obs)
for oi in 1:d_obs
    global obs_mse[oi] = mean(delta_obs[oi, :].^2)
end
total_obs_mse = sum(obs_mse)
obs_share = obs_mse ./ max(total_obs_mse, 1e-20)

# Sort states by contribution and take top 10
n_show = min(10, d_state_y)
perm_state = sortperm(state_share, rev=true)
top_state_idx = perm_state[1:n_show]

fig8b = plot(layout=(1, 2), size=(1000, 450), margin=6Plots.mm, bottom_margin=14Plots.mm)

# Panel 1: Top-10 state variables by delta-squared share
top_state_labels = [si <= length(state_labels) ? state_labels[si] : "s_$si" for si in top_state_idx]
bar!(fig8b[1], 1:n_show, state_share[top_state_idx] .* 100,
     color=:teal, alpha=0.8, label="")
plot!(fig8b[1], xticks=(1:n_show, top_state_labels), xrotation=35,
      title="(a) Top-$n_show State Variables by delta-squared Share",
      ylabel="% of total state delta-squared")

# Panel 2: RMSE for top-10 states
bar!(fig8b[2], 1:n_show, sqrt.(state_mse[top_state_idx]),
     color=:darkorange, alpha=0.8, label="")
plot!(fig8b[2], xticks=(1:n_show, top_state_labels), xrotation=35,
      title="(b) RMSE by State Variable (FOM - ROM1)",
      ylabel="RMSE (level units)")

plot!(fig8b, plot_title="State-Variable FOM-ROM1 Decomposition (Baseline, shock_scale=0.1)")
savefig(fig8b, joinpath(FIG_DIR, "hlt_state_variable_nonlinearity.pdf"))
println("  Saved: hlt_state_variable_nonlinearity.pdf")

println("\n  State variable nonlinearity ranking:")
println("  " * "-"^70)
@printf("  %-5s  %-15s  %12s  %12s  %12s\n",
        "Rank", "State", "Share (%)", "RMSE", "Max |delta|")
println("  " * "-"^70)
for (rank, si) in enumerate(top_state_idx)
    lab = si <= length(state_labels) ? state_labels[si] : "s_$si"
    max_delta = maximum(abs.(delta_state[si, :]))
    @printf("  %-5d  %-15s  %11.2f%%  %12.6f  %12.6f\n",
            rank, lab, state_share[si]*100, sqrt(state_mse[si]), max_delta)
end

# Cumulative share of top-k states
cumshare = cumsum(state_share[perm_state])
for k in [3, 5, 10, d_state_y]
    if k <= d_state_y
        @printf("  Top-%d states explain %.1f%% of state delta-squared\n", k, cumshare[k]*100)
    end
end

# Observable contribution ranking
println("\n  Observable nonlinearity ranking:")
println("  " * "-"^55)
@printf("  %-10s  %12s  %12s\n", "Observable", "Share (%)", "RMSE")
println("  " * "-"^55)
obs_perm = sortperm(obs_share, rev=true)
for oi in obs_perm
    @printf("  %-10s  %11.2f%%  %12.6f\n",
            observables[oi], obs_share[oi]*100, sqrt(obs_mse[oi]))
end

# ============================================================================
# FIGURE 8c: State-Observable Nonlinearity Coupling (NEW)
# ============================================================================
println("\n" * "="^60)
println("FIGURE 8c: State-Observable Nonlinearity Coupling")
println("="^60)

# Which state variables predict which observable gaps?
# Compute correlation between |delta_state_i| and |delta_obs_j|
n_top_states = min(10, d_state_y)
top_states_for_coupling = perm_state[1:n_top_states]

coupling_matrix = zeros(d_obs, n_top_states)
for (si_idx, si) in enumerate(top_states_for_coupling)
    for oi in 1:d_obs
        state_dev = abs.(delta_state[si, :])
        obs_dev = abs.(delta_obs[oi, :])
        if Statistics.std(state_dev) > 1e-15 && Statistics.std(obs_dev) > 1e-15
            global coupling_matrix[oi, si_idx] = cor(state_dev, obs_dev)
        end
    end
end

fig8c = plot(size=(700, 450), margin=6Plots.mm, bottom_margin=14Plots.mm,
             left_margin=8Plots.mm)

coupling_labels_x = [si <= length(state_labels) ? state_labels[si] : "s_$si"
                     for si in top_states_for_coupling]
obs_labels_y = string.(observables)

heatmap!(fig8c, coupling_labels_x, obs_labels_y, coupling_matrix,
         color=:RdBu, clims=(-1, 1),
         xlabel="State Variable (ranked by nonlinearity contribution)",
         ylabel="Observable",
         title="State-Observable Nonlinearity Coupling\n" *
               "(Pearson corr between |delta_state_i| and |delta_obs_j|)",
         colorbar_title="Correlation")

savefig(fig8c, joinpath(FIG_DIR, "hlt_state_obs_coupling.pdf"))
println("  Saved: hlt_state_obs_coupling.pdf")

println("\n  State-Observable coupling (top correlations):")
for oi in 1:d_obs
    max_corr_idx = argmax(coupling_matrix[oi, :])
    si = top_states_for_coupling[max_corr_idx]
    lab = si <= length(state_labels) ? state_labels[si] : "s_$si"
    @printf("  %-10s  <->  %-12s  r = %+.4f\n",
            observables[oi], lab, coupling_matrix[oi, max_corr_idx])
end

# ============================================================================
# ADDITIONAL DIAGNOSTIC: Per-individual-state contribution to delta_obs
# ============================================================================
println("\n" * "="^60)
println("DIAGNOSTIC: Which X-states (inputs) predict delta_obs_norm?")
println("="^60)

# Compute correlation between each input state variable value and delta_norm_obs
if d_state_x > 0
    state_input_corrs = zeros(d_state_x)
    for si in 1:d_state_x
        state_vals = X[si, :]
        if Statistics.std(state_vals) > 1e-15
            global state_input_corrs[si] = cor(abs.(state_vals), delta_norm_obs)
        end
    end

    perm_input = sortperm(abs.(state_input_corrs), rev=true)
    n_show_input = min(10, d_state_x)

    println("  Top-$n_show_input X-input states by |corr| with delta_obs_norm:")
    println("  " * "-"^45)
    for (rank, si) in enumerate(perm_input[1:n_show_input])
        lab = si <= length(state_labels) ? state_labels[si] : "x_$si"
        @printf("  %-5d  %-15s  corr = %+.4f\n", rank, lab, state_input_corrs[si])
    end
end

# -- Summary --
println("\n" * "="^60)
println("HLT DEEP ANALYSIS v2 -- COMPLETE")
println("="^60)
println("All figures saved to: $FIG_DIR")
println("  5.  hlt_temporal_clustering.pdf")
println("  6.  hlt_state_conditional_amplification.pdf")
println("  7.  hlt_shock_interaction_heatmap.pdf")
println("  8.  hlt_nonlinearity_surface.pdf")
println("  8b. hlt_state_variable_nonlinearity.pdf     [NEW: per-state decomposition]")
println("  8c. hlt_state_obs_coupling.pdf               [NEW: state-obs coupling matrix]")

println("\n--- RECOMMENDATIONS FOR FURTHER ANALYSIS ---")
println("1. NONLINEAR PROPAGATION PATH: The top state variables from Fig 8b likely")
println("   correspond to capital/investment states. Verify by inspecting the state")
println("   labels. If kp/inve dominate, this confirms the investment channel finding.")
println("")
println("2. CONDITIONAL ACF: Split the ACF analysis (Fig 5) by above-median vs")
println("   below-median state distance. Does nonlinearity persistence differ?")
println("")
println("3. NONLINEAR IMPULSE RESPONSE FUNCTIONS: For the top-2 shock interactions")
println("   from Fig 7, compute full nonlinear IRFs via SEP and compare with ROM1.")
println("   This reveals higher-order terms missed by pruned perturbation.")
println("")
println("4. FORECAST ERROR VARIANCE DECOMPOSITION: Apply the standard FEVD but")
println("   for delta_norm_obs instead of levels. Which shock type explains the")
println("   most variance in nonlinearity, not just in output?")
println("")
println("5. BUSINESS CYCLE ASYMMETRY DEEP DIVE: The expansion/recession split in")
println("   Fig 6 should show asymmetric amplification if the model has meaningful")
println("   nonlinearity. If it does not, the nonlinearity may be state-invariant")
println("   (driven by functional form, not regime).")
