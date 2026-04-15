#!/usr/bin/env julia
#
# ZLB-binding simulation: produce a shadow-rate IRF where the Taylor rule
# prescribes a negative rate, and overlay the ZLB-clamped actual rate.
#
# Uses the non-OBC HLT model with ignore_obc=true to get clean unconstrained
# ("shadow") rate paths. Manually clamps at the ZLB floor.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Plots; gr()

# ── Load model ──────────────────────────────────────────────────────
println("Loading MacroModelling...")
include(joinpath(@__DIR__, "..", "src", "MacroModelling.jl"))
using .MacroModelling

println("Loading HLT model (non-OBC)...")
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_HLT.jl"))

𝓂 = Smets_Wouters_2007_HLT

# Verify model and get steady state
println("Computing steady state...")
SS = get_steady_state(𝓂, derivatives=false)
r_ss = Float64(SS(:r))
pinf_ss = Float64(SS(:pinf))
r_ss_annual = 400.0 * (r_ss - 1.0)
println("  r[ss] = $r_ss  (gross quarterly rate, ann=$(round(r_ss_annual, digits=2))%)")
println("  pinf[ss] = $pinf_ss")

# ── Probe all shocks for rate effect ──────────────────────────────
shock_names = 𝓂.exo
nshocks = length(shock_names)
println("\nProbing rate response to each shock (1σ):")
rate_effects = Dict{Symbol, Float64}()

for sname in shock_names
    probe = get_irf(𝓂,
        shocks = sname,
        periods = 40,
        algorithm = :first_order,
        levels = true,
        verbose = false
    )
    probe_key = axiskeys(probe, 3)[1]
    r_probe = Float64.(probe(:r, :, probe_key))
    # Maximum deviation from SS (could be positive or negative)
    max_dev = maximum(abs.(r_probe .- r_ss))
    min_r = minimum(r_probe)
    rate_effects[sname] = r_ss - min_r  # drop from SS (positive = rate falls)
    ann_drop = 400.0 * (r_ss - min_r)
    println("  $sname: min_r=$(round(min_r, digits=6)), drop=$(round(ann_drop, digits=2)) pp ann")
end

# Find shock with largest rate-lowering effect
best_shock = argmax(rate_effects)
best_drop = rate_effects[best_shock]
println("\nBest rate-lowering shock: $best_shock ($(round(400*best_drop, digits=2)) pp ann per σ)")

# ── Calibrate shock size ──────────────────────────────────────────
# Want shadow rate to dip to about -3% annualized
# Target drop from SS: r_ss_annual + 3 = 8.21 + 3 = 11.21 pp
target_drop_ann = r_ss_annual + 3.0  # pp annualized
target_drop_qtr = target_drop_ann / 400.0  # quarterly level terms
sigma_needed = target_drop_qtr / best_drop

println("Target: shadow rate at -3% ann → need $(round(sigma_needed, digits=1))σ of $best_shock")

# Cap sigma to reasonable range
sigma_main = clamp(sigma_needed, 1.0, 6.0)
println("Using $(round(sigma_main, digits=1))σ $best_shock shock")

# ── Build shock matrix ────────────────────────────────────────────
T_total = 40

# Single-period large shock + smaller follow-ups for sustained crisis
shock_matrix = zeros(nshocks, 3)
main_idx = findfirst(x -> x == best_shock, shock_names)
shock_matrix[main_idx, 1] = -sigma_main       # Main shock
shock_matrix[main_idx, 2] = -sigma_main * 0.4 # Follow-up
shock_matrix[main_idx, 3] = -sigma_main * 0.15

# Add preference shock for amplification if best_shock != :eb
eb_idx = findfirst(x -> x == :eb, shock_names)
if best_shock != :eb && !isnothing(eb_idx)
    shock_matrix[eb_idx, 1] = -sigma_main * 0.5
    shock_matrix[eb_idx, 2] = -sigma_main * 0.25
end

println("\nShock matrix (non-zero entries):")
for (i, sn) in enumerate(shock_names)
    row = shock_matrix[i, :]
    if any(row .!= 0)
        println("  $sn: $(round.(row, digits=2))")
    end
end

# ── Run unconstrained IRF ─────────────────────────────────────────
println("\n" * "="^60)
println("Running unconstrained IRF (shadow rate)")
println("="^60)

irf_data = get_irf(𝓂,
    shocks = shock_matrix,
    periods = T_total,
    algorithm = :first_order,
    levels = true,
    verbose = false
)

println("IRF computed! Size: ", size(irf_data))

# ── Extract observables ────────────────────────────────────────────
shock_key = axiskeys(irf_data, 3)[1]
r_shadow = Float64.(irf_data(:r, :, shock_key))
robs_shadow = Float64.(irf_data(:robs, :, shock_key))
pinf_series = Float64.(irf_data(:pinfobs, :, shock_key))
dy_series = Float64.(irf_data(:dy, :, shock_key))
dinve_series = Float64.(irf_data(:dinve, :, shock_key))

T = length(r_shadow)
t_plot = 1:T

# Shadow rate (annualized, %)
shadow_annual = 4.0 .* robs_shadow

# ZLB-clamped rate: max(0, shadow_rate)
clamped_annual = max.(0.0, shadow_annual)

# ZLB binding detection
zlb_binding = shadow_annual .< 0.0
n_zlb = sum(zlb_binding)

# Annualized inflation
pinf_annual = 4.0 .* pinf_series
pinf_ss_annual = 400.0 * (pinf_ss - 1.0)
ctrend = Float64(SS(:dy))

# Print summary
println("\nShadow rate per period:")
for (i, (rs, rc)) in enumerate(zip(shadow_annual, clamped_annual))
    flag = rs < 0 ? " ← ZLB binds" : ""
    println("  t=$i: shadow=$(round(rs, digits=2))%, clamped=$(round(rc, digits=2))%$flag")
    if i > 25; println("  ... (truncated)"); break; end
end

println("\nZLB binding periods: $n_zlb / $T")

# ── Plot ──────────────────────────────────────────────────────────
println("\nGenerating plot...")

figdir = joinpath(@__DIR__, "..", "docs", "paper", "figures")
mkpath(figdir)

# ZLB binding region coordinates
zlb_starts = Int[]
zlb_ends = Int[]
if n_zlb > 0
    zlb_starts = findall(diff([false; zlb_binding]) .== 1)
    zlb_ends = findall(diff([zlb_binding; false]) .== -1)
end

p = plot(layout=(2,2), size=(1000, 700),
         titlefontsize=11,
         guidefontsize=9,
         tickfontsize=8,
         legendfontsize=7,
         left_margin=5Plots.mm,
         bottom_margin=3Plots.mm)

# Panel 1: Policy rate — shadow vs. clamped
plot!(p[1], t_plot, shadow_annual,
      label="Shadow rate (Taylor rule)", color=:steelblue, linewidth=1.5, linestyle=:dash, alpha=0.7)
plot!(p[1], t_plot, clamped_annual,
      label="Actual rate (ZLB enforced)", color=:navy, linewidth=2.5)
hline!(p[1], [0.0], label="ZLB floor", color=:red, linestyle=:solid, linewidth=1.5)
hline!(p[1], [r_ss_annual], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (k, (s, e)) in enumerate(zip(zlb_starts, zlb_ends))
    vspan!(p[1], [s-0.5, e+0.5], alpha=0.10, color=:red, label=(k==1 ? "ZLB regime" : ""))
end
title!(p[1], "Policy Rate (annualized)")
ylabel!(p[1], "% p.a.")
xlabel!(p[1], "Quarter")

# Panel 2: Inflation
plot!(p[2], t_plot, pinf_annual,
      label="Inflation", color=:darkred, linewidth=2)
hline!(p[2], [pinf_ss_annual], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p[2], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p[2], "Inflation (annualized)")
ylabel!(p[2], "% p.a.")
xlabel!(p[2], "Quarter")

# Panel 3: Output growth
plot!(p[3], t_plot, dy_series,
      label="Output growth", color=:darkgreen, linewidth=2)
hline!(p[3], [ctrend], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p[3], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p[3], "Output Growth")
ylabel!(p[3], "% (quarterly)")
xlabel!(p[3], "Quarter")

# Panel 4: Investment growth
plot!(p[4], t_plot, dinve_series,
      label="Investment growth", color=:purple, linewidth=2)
hline!(p[4], [ctrend], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p[4], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p[4], "Investment Growth")
ylabel!(p[4], "% (quarterly)")
xlabel!(p[4], "Quarter")

# Save
outpath = joinpath(figdir, "fig_zlb_binding_simulation.pdf")
savefig(p, outpath)
println("\nPlot saved to: $outpath")

outpath_png = joinpath(figdir, "fig_zlb_binding_simulation.png")
savefig(p, outpath_png)
println("PNG saved to: $outpath_png")

# ── Summary ──────────────────────────────────────────────────────
println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("  Shock used: $best_shock at $(round(sigma_main, digits=1))σ")
println("  Periods: $T")
println("  ZLB-binding periods: $n_zlb ($(round(100*n_zlb/T, digits=1))%)")
println("  Shadow rate range: [$(round(minimum(shadow_annual), digits=2)), $(round(maximum(shadow_annual), digits=2))] %")
println("  Inflation range: [$(round(minimum(pinf_annual), digits=2)), $(round(maximum(pinf_annual), digits=2))] %")
println("  Output growth range: [$(round(minimum(dy_series), digits=2)), $(round(maximum(dy_series), digits=2))] %")
println("  Investment growth range: [$(round(minimum(dinve_series), digits=2)), $(round(maximum(dinve_series), digits=2))] %")
println("="^60)
