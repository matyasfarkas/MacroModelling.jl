#!/usr/bin/env julia
#
# ZLB/OBC Nonlinearity Decomposition Analysis (v2)
# ==================================================
#
# Extends nonlinearity analysis to the OBC/ZLB case using the
# shock_scale=0.4 dataset (20,240 samples, ~8.7% ZLB binding).
#
# Produces 6 figures:
#   9.  Equation-Block Decomposition — ZLB Binding vs Non-Binding (obs + state blocks)
#   10. Nonlinearity Density Shift (shock_scale=0.1 vs 0.4) [FIXED: log x-axis]
#   11. Per-Variable Divergence — Binding vs Non-Binding
#   12. Nonlinearity Landscape Comparison (0.1 vs 0.4)
#   13. Per-State-Variable Nonlinearity Contribution (top-10 states)
#   14. Per-Theta Nonlinearity Heatmap (which parameter combos drive nonlinearity)
#
# Usage:
#   julia --project=. scripts/zlb_decomposition_analysis.jl

using MacroModelling
using Plots
using Serialization
using Statistics
using LinearAlgebra
using Printf
using StatsBase
using Distributions  # for Normal pdf in manual KDE

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

# -- Index maps --
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
d_obs = length(observables)

# -- Load ZLB dataset (shock_scale=0.4) --
const ZLB_PATH = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls")

println("Loading ZLB dataset: $ZLB_PATH")
zlb_chk = deserialize(ZLB_PATH)

X_zlb = zlb_chk["X"]
Y_zlb = zlb_chk["Y"]
Y_rom1_zlb = zlb_chk["Y_rom1"]
theta_ids_zlb = zlb_chk["theta_ids"]
cursor_zlb = zlb_chk["cursor"]

X_zlb = X_zlb[:, 1:cursor_zlb]
Y_zlb = Y_zlb[:, 1:cursor_zlb]
Y_rom1_zlb = Y_rom1_zlb[:, 1:cursor_zlb]
theta_ids_zlb = theta_ids_zlb[1:cursor_zlb]

N_zlb = cursor_zlb
println("ZLB dataset: $N_zlb samples")

# -- Load baseline dataset (shock_scale=0.1) --
const BASELINE_PATH = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2/hlt_sep_surrogate_dataset_checkpoint.jls")

println("Loading baseline dataset: $BASELINE_PATH")
base_chk = deserialize(BASELINE_PATH)

X_base = base_chk["X"]
Y_base = base_chk["Y"]
Y_rom1_base = base_chk["Y_rom1"]
cursor_base = base_chk["cursor"]

X_base = X_base[:, 1:cursor_base]
Y_base = Y_base[:, 1:cursor_base]
Y_rom1_base = Y_rom1_base[:, 1:cursor_base]

N_base = cursor_base
println("Baseline dataset: $N_base samples")

# -- Derive dimensions from checkpoint --
zlb_settings = zlb_chk["settings"]
zlb_theta_names = get(zlb_settings, "theta_names", Symbol[])
zlb_state_names = get(zlb_settings, "state_names", Symbol[])
d_theta_zlb = length(zlb_theta_names)
d_x_zlb = size(X_zlb, 1)
d_eps = 7  # SW07 structural shocks (ea, eb, eg, eqs, em, epinf, ew)
d_state_x = d_x_zlb - d_eps - d_theta_zlb  # states in X input
d_y = size(Y_zlb, 1)                         # total Y dimension
d_state_y = d_y - d_obs                      # states in Y output
println("  ZLB X layout: d_state_x=$d_state_x, d_eps=$d_eps, d_theta=$d_theta_zlb (total=$d_x_zlb)")
println("  ZLB Y layout: d_obs=$d_obs, d_state_y=$d_state_y (total=$d_y)")

# Baseline dimensions
base_settings = base_chk["settings"]
base_theta_names = get(base_settings, "theta_names", Symbol[])
base_state_names = get(base_settings, "state_names", Symbol[])
d_theta_base = length(base_theta_names)
d_x_base = size(X_base, 1)
d_state_x_base = d_x_base - d_eps - d_theta_base

# Use state names from checkpoint (falls back to generic labels)
if !isempty(zlb_state_names)
    state_labels = string.(zlb_state_names)
elseif !isempty(base_state_names)
    state_labels = string.(base_state_names)
else
    state_labels = ["state_$i" for i in 1:d_state_y]
end
println("  State variables ($d_state_y): ", join(state_labels[1:min(10, d_state_y)], ", "),
        d_state_y > 10 ? " ..." : "")

# -- Compute delta (obs-only vs full) --
delta_zlb_full = Y_zlb .- Y_rom1_zlb
delta_base_full = Y_base .- Y_rom1_base

# IMPORTANT: delta_norm should be computed on obs-only portion for
# interpretable relative errors, but full for total gap decomposition
delta_obs_zlb = delta_zlb_full[1:d_obs, :]
delta_obs_base = delta_base_full[1:d_obs, :]
delta_state_zlb = delta_zlb_full[d_obs+1:end, :]
delta_state_base = delta_base_full[d_obs+1:end, :]

# Obs-only norms (for density and relative error plots)
delta_norm_obs_zlb = [norm(delta_obs_zlb[:, i]) for i in 1:N_zlb]
delta_norm_obs_base = [norm(delta_obs_base[:, i]) for i in 1:N_base]

# Full norms (for total gap decomposition)
delta_norm_full_zlb = [norm(delta_zlb_full[:, i]) for i in 1:N_zlb]
delta_norm_full_base = [norm(delta_base_full[:, i]) for i in 1:N_base]

# -- Detect ZLB binding --
# Y layout: first d_obs rows are observables [dy, dc, dinve, labobs, pinfobs, dwobs, robs]
# robs = 100*(r-1), so robs near 0 means ZLB binding.
robs_obs_pos = findfirst(==(:robs), observables)  # = 7
robs_zlb = Y_zlb[robs_obs_pos, :]

# Use quantile-based threshold (~8.7% binding as reported for this dataset)
robs_threshold = quantile(robs_zlb, 0.087)
zlb_binding = robs_zlb .< robs_threshold
println("ZLB detection via robs threshold ($(@sprintf("%.4f", robs_threshold))): $(sum(zlb_binding)) binding")

n_binding = sum(zlb_binding)
n_nonbinding = N_zlb - n_binding
println("  Binding: $n_binding, Non-binding: $n_nonbinding")

# ============================================================================
# FIGURE 9: Equation-Block Decomposition — Binding vs Non-Binding
# ============================================================================
println("\n" * "="^60)
println("FIGURE 9: Equation-Block Decomposition")
println("="^60)

# Observable blocks (7 observables mapped to economic equation blocks)
block_map_obs = Dict(
    "Output/Resource"     => [:dy],
    "Consumption/Euler"   => [:dc],
    "Investment/Capital"  => [:dinve],
    "Labor Market"        => [:labobs],
    "Price Phillips"      => [:pinfobs],
    "Wage Phillips"       => [:dwobs],
    "Taylor Rule/Rate"    => [:robs],
)

block_names_obs = ["Output/Resource", "Consumption/Euler", "Investment/Capital",
                   "Labor Market", "Price Phillips", "Wage Phillips", "Taylor Rule/Rate"]

# Compute per-block contribution to total obs-delta-squared for binding/non-binding
binding_shares = Float64[]
nonbinding_shares = Float64[]
binding_abs = Float64[]
nonbinding_abs = Float64[]

for bname in block_names_obs
    vars = block_map_obs[bname]
    var_positions = [findfirst(==(v), observables) for v in vars]
    filter!(!isnothing, var_positions)

    if isempty(var_positions)
        push!(binding_shares, 0.0)
        push!(nonbinding_shares, 0.0)
        push!(binding_abs, 0.0)
        push!(nonbinding_abs, 0.0)
        continue
    end

    # Only observable portion of delta
    delta_block = delta_obs_zlb[var_positions, :]

    # Binding
    if n_binding > 0
        block_sq_binding = mean(sum(delta_block[:, zlb_binding].^2, dims=1))
        total_sq_binding = mean(delta_norm_obs_zlb[zlb_binding].^2)
        push!(binding_shares, block_sq_binding / max(total_sq_binding, 1e-20))
        push!(binding_abs, sqrt(block_sq_binding))
    else
        push!(binding_shares, 0.0)
        push!(binding_abs, 0.0)
    end

    # Non-binding
    nonbinding_mask = .!zlb_binding
    if n_nonbinding > 0
        block_sq_nonbinding = mean(sum(delta_block[:, nonbinding_mask].^2, dims=1))
        total_sq_nonbinding = mean(delta_norm_obs_zlb[nonbinding_mask].^2)
        push!(nonbinding_shares, block_sq_nonbinding / max(total_sq_nonbinding, 1e-20))
        push!(nonbinding_abs, sqrt(block_sq_nonbinding))
    else
        push!(nonbinding_shares, 0.0)
        push!(nonbinding_abs, 0.0)
    end
end

fig9 = plot(layout=(1, 2), size=(1000, 450), margin=6Plots.mm, bottom_margin=12Plots.mm)

# Panel 1: Share decomposition
x_pos = 1:length(block_names_obs)
bar_width = 0.35
bar!(fig9[1], x_pos .- bar_width/2, binding_shares .* 100,
     label="ZLB Binding (n=$n_binding)", color=:red, alpha=0.8, bar_width=bar_width)
bar!(fig9[1], x_pos .+ bar_width/2, nonbinding_shares .* 100,
     label="Non-Binding (n=$n_nonbinding)", color=:steelblue, alpha=0.8, bar_width=bar_width)
plot!(fig9[1], xticks=(x_pos, block_names_obs), xrotation=30,
      title="(a) Observable Share of Total delta-squared (%)",
      ylabel="% of total observable gap")

# Panel 2: Absolute magnitude (RMSE)
bar!(fig9[2], x_pos .- bar_width/2, binding_abs,
     label="ZLB Binding", color=:red, alpha=0.8, bar_width=bar_width)
bar!(fig9[2], x_pos .+ bar_width/2, nonbinding_abs,
     label="Non-Binding", color=:steelblue, alpha=0.8, bar_width=bar_width)
plot!(fig9[2], xticks=(x_pos, block_names_obs), xrotation=30,
      title="(b) RMSE by Observable Block",
      ylabel="RMSE (model units)")

for (i, bname) in enumerate(block_names_obs)
    @printf("  %-20s  Binding: %5.1f%%  Non-Binding: %5.1f%%  |  RMSE Bind: %.4f  Non: %.4f\n",
            bname, binding_shares[i]*100, nonbinding_shares[i]*100,
            binding_abs[i], nonbinding_abs[i])
end

plot!(fig9, plot_title="Equation-Block Decomposition: ZLB Binding vs Non-Binding")
savefig(fig9, joinpath(FIG_DIR, "zlb_block_decomposition.pdf"))
println("  Saved: zlb_block_decomposition.pdf")

# ============================================================================
# FIGURE 10: Nonlinearity Density Shift (FIXED: log x-axis)
# ============================================================================
println("\n" * "="^60)
println("FIGURE 10: Nonlinearity Density Shift (log x-axis)")
println("="^60)

# Fast histogram-based KDE (O(n), not O(n^2))
function simple_kde(data::Vector{Float64}; n_points=300, log_space=false)
    n = length(data)
    work_data = log_space ? log10.(max.(data, 1e-15)) : data
    h = 1.06 * Statistics.std(work_data) * n^(-0.2)  # Silverman bandwidth
    lo = minimum(work_data) - 3h
    hi = maximum(work_data) + 3h
    x = collect(range(lo, hi, length=n_points))
    dx = x[2] - x[1]

    # Bin data into histogram, then smooth with Gaussian kernel
    counts = zeros(n_points)
    for d in work_data
        idx = clamp(round(Int, (d - lo) / dx) + 1, 1, n_points)
        counts[idx] += 1.0
    end

    # Gaussian smoothing (convolution with truncated kernel)
    kernel_width = max(1, round(Int, 3h / dx))
    density = zeros(n_points)
    for i in 1:n_points
        for j in max(1, i-kernel_width):min(n_points, i+kernel_width)
            density[i] += counts[j] * exp(-0.5 * ((x[i] - x[j]) / h)^2)
        end
    end
    density ./= (n * h * sqrt(2pi))

    if log_space
        return 10.0 .^ x, density
    end
    return x, density
end

fig10 = plot(size=(700, 450), margin=6Plots.mm, bottom_margin=10Plots.mm)

# Use log-space KDE: spread out the mass that clusters near zero
# (a) Baseline
kx_base, ky_base = simple_kde(delta_norm_obs_base; log_space=true)
plot!(fig10, kx_base, ky_base, label="Baseline (shock_scale=0.1, n=$N_base)",
      color=:steelblue, linewidth=2, fillalpha=0.12, fill=true)

# (b) ZLB non-binding
if n_nonbinding > 10
    kx_nb, ky_nb = simple_kde(delta_norm_obs_zlb[.!zlb_binding]; log_space=true)
    plot!(fig10, kx_nb, ky_nb, label="Large shocks, non-binding (shock_scale=0.4, n=$n_nonbinding)",
          color=:green, linewidth=2, fillalpha=0.12, fill=true)
end

# (c) ZLB binding
if n_binding > 10
    kx_b, ky_b = simple_kde(delta_norm_obs_zlb[zlb_binding]; log_space=true)
    plot!(fig10, kx_b, ky_b, label="Large shocks, ZLB binding (shock_scale=0.4, n=$n_binding)",
          color=:red, linewidth=2, fillalpha=0.12, fill=true)
end

# Median vertical lines with labels
med_base = median(delta_norm_obs_base)
vline!(fig10, [med_base], color=:steelblue, linestyle=:dash, linewidth=1,
       label="Median baseline = $(@sprintf("%.3f", med_base))")
if n_nonbinding > 0
    med_nb = median(delta_norm_obs_zlb[.!zlb_binding])
    vline!(fig10, [med_nb], color=:green, linestyle=:dash, linewidth=1,
           label="Median non-binding = $(@sprintf("%.3f", med_nb))")
end
if n_binding > 0
    med_b = median(delta_norm_obs_zlb[zlb_binding])
    vline!(fig10, [med_b], color=:red, linestyle=:dash, linewidth=1,
           label="Median binding = $(@sprintf("%.3f", med_b))")
end

plot!(fig10, xscale=:log10,
      xlabel="Observable nonlinearity norm ||delta_obs||  (log10 scale)\n" *
             "[delta = y_FOM - y_ROM1 for 7 observables only]",
      ylabel="Density (KDE in log-space)",
      title="Distribution of FOM-ROM1 Observable Gap: Baseline vs ZLB\n" *
            "(log10 x-axis reveals tail structure hidden by mass at small values)",
      legend=:topright)

@printf("  Median delta_obs_norm -- Baseline: %.6f\n", med_base)
@printf("  Median delta_obs_norm -- ZLB non-binding: %.6f\n",
        n_nonbinding > 0 ? median(delta_norm_obs_zlb[.!zlb_binding]) : NaN)
@printf("  Median delta_obs_norm -- ZLB binding: %.6f\n",
        n_binding > 0 ? median(delta_norm_obs_zlb[zlb_binding]) : NaN)

savefig(fig10, joinpath(FIG_DIR, "zlb_nonlinearity_density.pdf"))
println("  Saved: zlb_nonlinearity_density.pdf")

# ============================================================================
# FIGURE 11: Per-Variable Divergence — Binding vs Non-Binding
# ============================================================================
println("\n" * "="^60)
println("FIGURE 11: Per-Observable Divergence (absolute error)")
println("="^60)

fig11 = plot(size=(900, 450), margin=6Plots.mm, bottom_margin=10Plots.mm)

obs_labels = string.(observables)
n_obs = length(observables)

# Compute ABSOLUTE error |delta| for each observable (not relative error,
# which inflates near zero crossings and gives misleading 100-200% values)
binding_medians = Float64[]
nonbinding_medians = Float64[]
binding_q75 = Float64[]
nonbinding_q75 = Float64[]

# Also compute normalized error: |delta_i| / std(y_FOM_i)
binding_normed = Float64[]
nonbinding_normed = Float64[]

for (oi, obs) in enumerate(observables)
    delta_vals = abs.(delta_obs_zlb[oi, :])
    fom_std = Statistics.std(Y_zlb[oi, :])

    # Binding
    if n_binding > 0
        push!(binding_medians, median(delta_vals[zlb_binding]))
        push!(binding_q75, quantile(delta_vals[zlb_binding], 0.75))
        push!(binding_normed, median(delta_vals[zlb_binding]) / max(fom_std, 1e-10))
    else
        push!(binding_medians, 0.0)
        push!(binding_q75, 0.0)
        push!(binding_normed, 0.0)
    end

    # Non-binding
    if n_nonbinding > 0
        push!(nonbinding_medians, median(delta_vals[.!zlb_binding]))
        push!(nonbinding_q75, quantile(delta_vals[.!zlb_binding], 0.75))
        push!(nonbinding_normed, median(delta_vals[.!zlb_binding]) / max(fom_std, 1e-10))
    else
        push!(nonbinding_medians, 0.0)
        push!(nonbinding_q75, 0.0)
        push!(nonbinding_normed, 0.0)
    end
end

x_pos = 1:n_obs
bar_width = 0.35

# Use normalized error (|delta|/std) for comparability across variables
bar!(fig11, x_pos .- bar_width/2, binding_normed .* 100,
     label="ZLB Binding (median)", color=:red, alpha=0.8, bar_width=bar_width)
bar!(fig11, x_pos .+ bar_width/2, nonbinding_normed .* 100,
     label="Non-Binding (median)", color=:steelblue, alpha=0.8, bar_width=bar_width)

plot!(fig11, xticks=(x_pos, obs_labels), xrotation=20,
      ylabel="Normalized error (%): median |delta_i| / std(y_FOM_i) x 100",
      title="Per-Observable Normalized Error: ZLB Binding vs Non-Binding\n" *
            "(error normalized by unconditional std of each observable)")

println("\n  Per-observable error diagnostics:")
println("  " * "-"^80)
@printf("  %-10s  %12s  %12s  %12s  %12s\n",
        "Variable", "Bind median", "NonB median", "Bind normed%", "NonB normed%")
println("  " * "-"^80)
for (oi, obs) in enumerate(observables)
    fom_std = Statistics.std(Y_zlb[oi, :])
    @printf("  %-10s  %12.6f  %12.6f  %11.2f%%  %11.2f%%\n",
            obs, binding_medians[oi], nonbinding_medians[oi],
            binding_normed[oi]*100, nonbinding_normed[oi]*100)
end

savefig(fig11, joinpath(FIG_DIR, "zlb_per_variable_divergence.pdf"))
println("  Saved: zlb_per_variable_divergence.pdf")

# ============================================================================
# FIGURE 12: Nonlinearity Landscape Comparison
# ============================================================================
println("\n" * "="^60)
println("FIGURE 12: Nonlinearity Landscape Comparison")
println("="^60)

fig12 = plot(layout=(1, 2), size=(1000, 450), margin=6Plots.mm)

# State distance and shock magnitude for both datasets
state_dist_base = [norm(X_base[1:d_state_x_base, i]) for i in 1:N_base]
shock_mag_base = [norm(X_base[d_state_x_base+1:d_state_x_base+d_eps, i]) for i in 1:N_base]

state_dist_zlb = [norm(X_zlb[1:d_state_x, i]) for i in 1:N_zlb]
shock_mag_zlb = [norm(X_zlb[d_state_x+1:d_state_x+d_eps, i]) for i in 1:N_zlb]

n_bins = 25

# Shared axis ranges
all_sd = vcat(state_dist_base, state_dist_zlb)
all_sm = vcat(shock_mag_base, shock_mag_zlb)
sd_lo, sd_hi = quantile(all_sd, 0.01), quantile(all_sd, 0.99)
sm_lo, sm_hi = quantile(all_sm, 0.01), quantile(all_sm, 0.99)
sd_edges = range(sd_lo, sd_hi, length=n_bins+1)
sm_edges = range(sm_lo, sm_hi, length=n_bins+1)

# Use obs-only delta norm for landscape coloring
all_dn = vcat(delta_norm_obs_base, delta_norm_obs_zlb)
clims_range = (0.0, quantile(all_dn, 0.95))

for (panel, sd_arr, sm_arr, dn_arr, n_arr, label) in [
    (1, state_dist_base, shock_mag_base, delta_norm_obs_base, N_base, "(a) Baseline (shock_scale=0.1)"),
    (2, state_dist_zlb, shock_mag_zlb, delta_norm_obs_zlb, N_zlb, "(b) ZLB (shock_scale=0.4)")
]
    contour_z = fill(NaN, n_bins, n_bins)
    for i in 1:n_bins
        for j in 1:n_bins
            mask = (sd_arr .>= sd_edges[i]) .& (sd_arr .< sd_edges[i+1]) .&
                   (sm_arr .>= sm_edges[j]) .& (sm_arr .< sm_edges[j+1])
            if sum(mask) >= 3
                contour_z[j, i] = mean(dn_arr[mask])
            end
        end
    end

    sd_centers = [(sd_edges[i] + sd_edges[i+1]) / 2 for i in 1:n_bins]
    sm_centers = [(sm_edges[j] + sm_edges[j+1]) / 2 for j in 1:n_bins]

    contour_z_plot = replace(contour_z, NaN => 0.0)
    contourf!(fig12[panel], sd_centers, sm_centers, contour_z_plot,
              color=:viridis, levels=15, clims=clims_range,
              xlabel="||state deviation from SS||", ylabel="||shock vector||",
              title=label, colorbar_title="mean ||delta_obs||")
end

plot!(fig12, plot_title="Nonlinearity Landscape: State Distance × Shock Magnitude")
savefig(fig12, joinpath(FIG_DIR, "zlb_nonlinearity_landscape.pdf"))
println("  Saved: zlb_nonlinearity_landscape.pdf")

# ============================================================================
# FIGURE 13: Per-State-Variable Nonlinearity Contribution
# ============================================================================
println("\n" * "="^60)
println("FIGURE 13: Per-State-Variable Nonlinearity Contribution")
println("="^60)

# Decompose delta in the state portion of Y by individual state variable
# delta_state_zlb is (d_state_y x N_zlb)
# Compute mean(delta_i^2) for each state variable i, then rank

state_mse_binding = zeros(d_state_y)
state_mse_nonbinding = zeros(d_state_y)

for si in 1:d_state_y
    if n_binding > 0
        global state_mse_binding[si] = mean(delta_state_zlb[si, zlb_binding].^2)
    end
    if n_nonbinding > 0
        global state_mse_nonbinding[si] = mean(delta_state_zlb[si, .!zlb_binding].^2)
    end
end

# Total state MSE
total_state_mse_binding = sum(state_mse_binding)
total_state_mse_nonbinding = sum(state_mse_nonbinding)

state_share_binding = state_mse_binding ./ max(total_state_mse_binding, 1e-20)
state_share_nonbinding = state_mse_nonbinding ./ max(total_state_mse_nonbinding, 1e-20)

# Sort by binding share (descending) and take top 10
n_show = min(10, d_state_y)
perm_binding = sortperm(state_share_binding, rev=true)
top_idx = perm_binding[1:n_show]

fig13 = plot(layout=(1, 2), size=(1000, 450), margin=6Plots.mm, bottom_margin=14Plots.mm)

# Panel 1: Top-10 state variables by share during ZLB binding
top_labels = [si <= length(state_labels) ? state_labels[si] : "s_$si" for si in top_idx]
x_pos = 1:n_show
bar_width = 0.35

bar!(fig13[1], x_pos .- bar_width/2, state_share_binding[top_idx] .* 100,
     label="ZLB Binding", color=:red, alpha=0.8, bar_width=bar_width)
bar!(fig13[1], x_pos .+ bar_width/2, state_share_nonbinding[top_idx] .* 100,
     label="Non-Binding", color=:steelblue, alpha=0.8, bar_width=bar_width)
plot!(fig13[1], xticks=(x_pos, top_labels), xrotation=35,
      title="(a) Top-$n_show State Variables by delta-squared Share",
      ylabel="% of total state delta-squared")

# Panel 2: Absolute RMSE for top-10 states
bar!(fig13[2], x_pos .- bar_width/2, sqrt.(state_mse_binding[top_idx]),
     label="ZLB Binding", color=:red, alpha=0.8, bar_width=bar_width)
bar!(fig13[2], x_pos .+ bar_width/2, sqrt.(state_mse_nonbinding[top_idx]),
     label="Non-Binding", color=:steelblue, alpha=0.8, bar_width=bar_width)
plot!(fig13[2], xticks=(x_pos, top_labels), xrotation=35,
      title="(b) RMSE by State Variable (top-$n_show)",
      ylabel="RMSE (level units)")

plot!(fig13, plot_title="State-Variable FOM-ROM1 Decomposition: ZLB Binding vs Non-Binding")
savefig(fig13, joinpath(FIG_DIR, "zlb_state_variable_decomposition.pdf"))
println("  Saved: zlb_state_variable_decomposition.pdf")

println("\n  Top-$n_show state variables by binding share:")
println("  " * "-"^75)
@printf("  %-5s  %-15s  %12s  %12s  %12s  %12s\n",
        "Rank", "State", "Bind share%", "NonB share%", "Bind RMSE", "NonB RMSE")
println("  " * "-"^75)
for (rank, si) in enumerate(top_idx)
    lab = si <= length(state_labels) ? state_labels[si] : "s_$si"
    @printf("  %-5d  %-15s  %11.2f%%  %11.2f%%  %12.6f  %12.6f\n",
            rank, lab, state_share_binding[si]*100, state_share_nonbinding[si]*100,
            sqrt(state_mse_binding[si]), sqrt(state_mse_nonbinding[si]))
end

# ============================================================================
# FIGURE 14: Per-Theta Nonlinearity Heatmap
# ============================================================================
println("\n" * "="^60)
println("FIGURE 14: Per-Theta Nonlinearity Heatmap")
println("="^60)

# Extract theta portion of X for each sample
theta_start = d_state_x + d_eps + 1
theta_end = d_state_x + d_eps + d_theta_zlb

if d_theta_zlb > 0 && theta_end <= d_x_zlb
    unique_tids = sort(unique(theta_ids_zlb))
    n_thetas = length(unique_tids)

    # Compute mean delta_norm per theta_id
    theta_mean_delta = zeros(n_thetas)
    theta_count = zeros(Int, n_thetas)
    theta_param_vals = zeros(d_theta_zlb, n_thetas)

    for (ti, tid) in enumerate(unique_tids)
        mask = theta_ids_zlb .== tid
        global theta_mean_delta[ti] = mean(delta_norm_obs_zlb[mask])
        global theta_count[ti] = sum(mask)
        # Extract theta values from first sample with this theta_id
        first_idx = findfirst(mask)
        if first_idx !== nothing
            global theta_param_vals[:, ti] = X_zlb[theta_start:theta_end, first_idx]
        end
    end

    # Find the 2 most correlated theta parameters with delta_norm
    theta_corrs = zeros(d_theta_zlb)
    for pi in 1:d_theta_zlb
        # Correlation between per-sample theta value and per-sample delta_norm
        theta_vals_all = X_zlb[theta_start + pi - 1, :]
        if Statistics.std(theta_vals_all) > 1e-12
            global theta_corrs[pi] = cor(theta_vals_all, delta_norm_obs_zlb)
        end
    end

    # Sort by absolute correlation
    perm_corr = sortperm(abs.(theta_corrs), rev=true)
    top2 = perm_corr[1:min(2, d_theta_zlb)]

    theta_name_labels = if !isempty(zlb_theta_names)
        string.(zlb_theta_names)
    else
        ["theta_$i" for i in 1:d_theta_zlb]
    end

    println("  Theta-nonlinearity correlations (top 5):")
    for pi in perm_corr[1:min(5, d_theta_zlb)]
        @printf("    %-12s  corr = %+.4f\n", theta_name_labels[pi], theta_corrs[pi])
    end

    if length(top2) >= 2
        p1, p2 = top2[1], top2[2]
        p1_vals = X_zlb[theta_start + p1 - 1, :]
        p2_vals = X_zlb[theta_start + p2 - 1, :]

        n_hbins = 20
        p1_edges = range(quantile(p1_vals, 0.01), quantile(p1_vals, 0.99), length=n_hbins+1)
        p2_edges = range(quantile(p2_vals, 0.01), quantile(p2_vals, 0.99), length=n_hbins+1)

        heatmap_z = fill(NaN, n_hbins, n_hbins)
        for i in 1:n_hbins
            for j in 1:n_hbins
                mask = (p1_vals .>= p1_edges[i]) .& (p1_vals .< p1_edges[i+1]) .&
                       (p2_vals .>= p2_edges[j]) .& (p2_vals .< p2_edges[j+1])
                if sum(mask) >= 2
                    heatmap_z[j, i] = mean(delta_norm_obs_zlb[mask])
                end
            end
        end

        p1_centers = [(p1_edges[i] + p1_edges[i+1]) / 2 for i in 1:n_hbins]
        p2_centers = [(p2_edges[j] + p2_edges[j+1]) / 2 for j in 1:n_hbins]

        heatmap_z_plot = replace(heatmap_z, NaN => 0.0)

        fig14 = plot(size=(650, 500), margin=6Plots.mm, bottom_margin=10Plots.mm)
        heatmap!(fig14, p1_centers, p2_centers, heatmap_z_plot,
                 color=:inferno, colorbar_title="mean ||delta_obs||",
                 xlabel=theta_name_labels[p1] * " (corr=$(@sprintf("%+.3f", theta_corrs[p1])))",
                 ylabel=theta_name_labels[p2] * " (corr=$(@sprintf("%+.3f", theta_corrs[p2])))",
                 title="Parameter-Nonlinearity Heatmap: Which Theta Combinations\n" *
                       "Produce Largest FOM-ROM1 Gap? (ZLB dataset)")

        savefig(fig14, joinpath(FIG_DIR, "zlb_theta_nonlinearity_heatmap.pdf"))
        println("  Saved: zlb_theta_nonlinearity_heatmap.pdf")
    else
        println("  WARNING: fewer than 2 theta parameters; skipping heatmap")
    end

    # Also produce a bar chart of all theta correlations
    fig14b = plot(size=(700, 400), margin=6Plots.mm, bottom_margin=12Plots.mm)
    bar!(fig14b, 1:d_theta_zlb, theta_corrs[perm_corr],
         color=[c >= 0 ? :indianred : :steelblue for c in theta_corrs[perm_corr]],
         alpha=0.8, label="")
    hline!(fig14b, [0], color=:black, linewidth=0.5, label="")
    plot!(fig14b, xticks=(1:d_theta_zlb, theta_name_labels[perm_corr]), xrotation=35,
          ylabel="Pearson correlation with ||delta_obs||",
          title="Parameter-Nonlinearity Correlation: Which Parameters\n" *
                "Are Most Associated with FOM-ROM1 Gap?")

    savefig(fig14b, joinpath(FIG_DIR, "zlb_theta_nonlinearity_correlations.pdf"))
    println("  Saved: zlb_theta_nonlinearity_correlations.pdf")
else
    println("  WARNING: no theta parameters in X; skipping per-theta figures")
end

# ============================================================================
# DIAGNOSTIC: Investigate the high relative errors (100-200%)
# ============================================================================
println("\n" * "="^60)
println("DIAGNOSTIC: Why are relative errors so high?")
println("="^60)

println("\n  Comparing absolute delta magnitudes with FOM variable scales:")
println("  " * "-"^90)
@printf("  %-10s  %12s  %12s  %12s  %12s  %12s\n",
        "Variable", "FOM mean", "FOM std", "delta mean", "delta/std %", "delta/mean %")
println("  " * "-"^90)
for (oi, obs) in enumerate(observables)
    fom_mean = mean(Y_zlb[oi, :])
    fom_std_val = Statistics.std(Y_zlb[oi, :])
    delta_mean = mean(abs.(delta_obs_zlb[oi, :]))
    @printf("  %-10s  %12.4f  %12.4f  %12.6f  %11.2f%%  %11.2f%%\n",
            obs, fom_mean, fom_std_val, delta_mean,
            delta_mean / max(fom_std_val, 1e-10) * 100,
            delta_mean / max(abs(fom_mean), 1e-10) * 100)
end
println("\n  Note: relative error |delta|/|y_FOM| inflates near zero crossings.")
println("  The normalized metric |delta|/std(y_FOM) is more informative.")
println("  High values (>10%) indicate economically significant nonlinearity.")

# -- Summary --
println("\n" * "="^60)
println("ZLB DECOMPOSITION ANALYSIS v2 -- COMPLETE")
println("="^60)
println("All figures saved to: $FIG_DIR")
println("   9. zlb_block_decomposition.pdf")
println("  10. zlb_nonlinearity_density.pdf          [FIXED: log x-axis KDE]")
println("  11. zlb_per_variable_divergence.pdf       [FIXED: normalized error metric]")
println("  12. zlb_nonlinearity_landscape.pdf")
println("  13. zlb_state_variable_decomposition.pdf  [NEW: per-state decomposition]")
println("  14. zlb_theta_nonlinearity_heatmap.pdf    [NEW: parameter heatmap]")
println("  14b.zlb_theta_nonlinearity_correlations.pdf [NEW: parameter correlations]")

println("\n--- RECOMMENDATIONS FOR FURTHER ANALYSIS ---")
println("1. TEMPORAL STATE CLUSTERING: Track how top-3 nonlinear states (from Fig 13)")
println("   evolve during ZLB binding episodes. Do they anticipate binding?")
println("2. CONDITIONAL BLOCK DECOMPOSITION: Slice the obs-block decomposition by")
println("   investment state (kp) quintile. Does inv dominance persist when kp is near SS?")
println("3. PARAMETER SENSITIVITY: For the most-correlated parameter (from Fig 14b),")
println("   run a 1D sweep of delta_norm vs that parameter, holding others fixed.")
println("4. NONLINEAR PROPAGATION CHANNELS: Use the top-3 states from Fig 13 as")
println("   mediators in a causal decomposition: shock -> state -> obs nonlinearity.")
