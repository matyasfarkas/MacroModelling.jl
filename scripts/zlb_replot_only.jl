#!/usr/bin/env julia
#
# Quick replot of the ZLB figure from cached SEP simulation data.
# Run zlb_sep_nonlinear_simulation.jl first to generate the cache.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Serialization
using Plots; gr()

# ── Load cached simulation ──────────────────────────────────────────
cache_path = joinpath(@__DIR__, "..", ".local_artifacts", "zlb_sep_simulation_cache.jls")
if !isfile(cache_path)
    error("Cache not found at $cache_path. Run zlb_sep_nonlinear_simulation.jl first.")
end

println("Loading cached SEP simulation...")
data = deserialize(cache_path)
sim_matrix = data.simulation  # raw matrix (nvars × ntime)
var_names = data.var_names
time_keys = data.time_keys

println("Variables: $(length(var_names)), Time points: $(length(time_keys))")

# ── Steady state reference values ───────────────────────────────────
# These are from the Smets_Wouters_2007_HLT_zlb model
r_ss_ann = 8.21    # 400*(1.020537-1)
pinf_ss_ann = 2.8  # 400*(1.007-1)
ctrend = 0.3982

# ── Extract variables ───────────────────────────────────────────────
function get_var(name::Symbol)
    idx = findfirst(==(name), var_names)
    isnothing(idx) && error("Variable $name not found")
    return Float64.(sim_matrix[idx, :])
end

robs_path = get_var(:robs)
robs_tilde_path = get_var(:robs_tilde)
pinfobs_path = get_var(:pinfobs)
dy_path = get_var(:dy)
dinve_path = get_var(:dinve)

# Annualize (robs is 100*(r-1), annualize = 4×)
robs_ann = 4.0 .* robs_path
robs_tilde_ann = 4.0 .* robs_tilde_path
pinf_ann = 4.0 .* pinfobs_path

# Skip period 0 (initial state = SS)
T = length(robs_ann) - 1
t_plot = 1:T
robs_ann_plot = robs_ann[2:end]
robs_tilde_ann_plot = robs_tilde_ann[2:end]
pinf_ann_plot = pinf_ann[2:end]
dy_plot = dy_path[2:end]
dinve_plot = dinve_path[2:end]

# ── ZLB detection based on ACTUAL rate ──────────────────────────────
# Shade only when the actual rate is pinned near zero
zlb_threshold_ann = 1.0  # % annualized
zlb_binding = robs_ann_plot .< zlb_threshold_ann
n_zlb = sum(zlb_binding)

println("\nPeriod-by-period rates:")
for t in 1:min(25, T)
    flag = zlb_binding[t] ? " ← ZLB" : ""
    println("  t=$t: shadow=$(round(robs_tilde_ann_plot[t], digits=2))%, actual=$(round(robs_ann_plot[t], digits=2))%$flag")
end

println("\nZLB binding (actual < $(zlb_threshold_ann)%): $n_zlb / $T periods")

# ZLB shading coordinates
zlb_starts = Int[]
zlb_ends = Int[]
if n_zlb > 0
    zlb_starts = findall(diff([false; zlb_binding]) .== 1)
    zlb_ends = findall(diff([zlb_binding; false]) .== -1)
end
println("Shaded regions: ", collect(zip(zlb_starts, zlb_ends)))

# ── Plot ────────────────────────────────────────────────────────────
figdir = joinpath(@__DIR__, "..", "docs", "paper", "figures")
mkpath(figdir)

p = plot(layout=(2,2), size=(1000, 700),
         titlefontsize=11, guidefontsize=9, tickfontsize=8,
         legendfontsize=7, left_margin=5Plots.mm, bottom_margin=3Plots.mm)

# Panel 1: Policy rate — shadow vs actual
plot!(p[1], t_plot, robs_tilde_ann_plot,
      label="Shadow rate (Taylor rule)", color=:steelblue, linewidth=1.5,
      linestyle=:dash, alpha=0.7)
plot!(p[1], t_plot, robs_ann_plot,
      label="Actual rate (softplus ZLB)", color=:navy, linewidth=2.5)
hline!(p[1], [0.0], label="ZLB floor", color=:red, linestyle=:solid, linewidth=1.5)
hline!(p[1], [r_ss_ann], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (k, (s, e)) in enumerate(zip(zlb_starts, zlb_ends))
    vspan!(p[1], [s-0.5, e+0.5], alpha=0.10, color=:red,
           label=(k==1 ? "ZLB regime" : ""))
end
title!(p[1], "Policy Rate (annualized)")
ylabel!(p[1], "% p.a.")
xlabel!(p[1], "Quarter")

# Panel 2: Inflation
plot!(p[2], t_plot, pinf_ann_plot,
      label="Inflation", color=:darkred, linewidth=2)
hline!(p[2], [pinf_ss_ann], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p[2], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p[2], "Inflation (annualized)")
ylabel!(p[2], "% p.a.")
xlabel!(p[2], "Quarter")

# Panel 3: Output growth
plot!(p[3], t_plot, dy_plot,
      label="Output growth", color=:darkgreen, linewidth=2)
hline!(p[3], [ctrend], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p[3], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p[3], "Output Growth")
ylabel!(p[3], "% (quarterly)")
xlabel!(p[3], "Quarter")

# Panel 4: Investment growth
plot!(p[4], t_plot, dinve_plot,
      label="Investment growth", color=:purple, linewidth=2)
hline!(p[4], [ctrend], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p[4], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p[4], "Investment Growth")
ylabel!(p[4], "% (quarterly)")
xlabel!(p[4], "Quarter")

savefig(p, joinpath(figdir, "fig_zlb_binding_simulation.pdf"))
savefig(p, joinpath(figdir, "fig_zlb_binding_simulation.png"))
println("\nSaved: fig_zlb_binding_simulation.pdf/.png")

println("\nSUMMARY")
println("  ZLB binding: $n_zlb / $T ($(round(100*n_zlb/T, digits=1))%)")
println("  Actual rate range: [$(round(minimum(robs_ann_plot), digits=2)), $(round(maximum(robs_ann_plot), digits=2))] %")
println("  Shadow rate range: [$(round(minimum(robs_tilde_ann_plot), digits=2)), $(round(maximum(robs_tilde_ann_plot), digits=2))] %")
