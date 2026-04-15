#!/usr/bin/env julia
#
# Compare IRFs: Baseline (no ZLB) vs C2 (softplus ZLB)
# C1 (sqrt-smooth-max) NSSS fails, so we skip it.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots; gr()

include(joinpath(@__DIR__, "..", "src", "MacroModelling.jl"))
using .MacroModelling

# ── Load models ───────────────────────────────────────────────────
println("="^70)
println("Loading models...")
println("="^70)

println("\n[1/2] Baseline (non-OBC)...")
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_HLT.jl"))
m_base = Smets_Wouters_2007_HLT

println("\n[2/2] C2: softplus ZLB...")
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_HLT_zlb.jl"))
m_c2 = Smets_Wouters_2007_HLT_zlb

# ── Steady states ─────────────────────────────────────────────────
println("\n" * "="^70)
println("Steady States")
println("="^70)

ss_base = get_steady_state(m_base, derivatives=false)
ss_c2 = get_steady_state(m_c2, derivatives=false)

for (name, ss) in [("Baseline", ss_base), ("C2 softplus", ss_c2)]
    r_val = Float64(ss(:r))
    pinf_val = Float64(ss(:pinf))
    println("  $name:  r[ss]=$(round(r_val, digits=6)) (ann=$(round(400*(r_val-1), digits=2))%),  pinf[ss]=$(round(pinf_val, digits=6))")
end

# Check C2 shadow rate
if :r_tilde in m_c2.var
    r_tilde_ss = Float64(ss_c2(:r_tilde))
    println("  C2 r_tilde[ss] = $(round(r_tilde_ss, digits=6)) (shadow rate)")
end

# ── Standard 1σ IRFs ──────────────────────────────────────────────
println("\n" * "="^70)
println("Standard IRFs (-1σ shocks, first-order, 40 periods)")
println("="^70)

T_irf = 40
obs_vars = [:robs, :pinfobs, :dy, :dinve]
obs_labels = ["Policy Rate (robs)" "Inflation (pinfobs)" "Output Growth (dy)" "Invest. Growth (dinve)"]

figdir = joinpath(@__DIR__, "..", "docs", "paper", "figures")
mkpath(figdir)

for shock_sym in [:eb, :ea, :em, :eqs]
    println("\n--- Shock: $shock_sym (-1σ) ---")

    irf_base = get_irf(m_base, shocks=shock_sym, periods=T_irf, algorithm=:first_order,
                       levels=true, negative_shock=true, verbose=false)
    irf_c2 = get_irf(m_c2, shocks=shock_sym, periods=T_irf, algorithm=:first_order,
                     levels=true, negative_shock=true, verbose=false)

    sk_base = axiskeys(irf_base, 3)[1]
    sk_c2 = axiskeys(irf_c2, 3)[1]

    p = plot(layout=(2,2), size=(1000, 700),
             titlefontsize=10, legendfontsize=7,
             left_margin=5Plots.mm, bottom_margin=3Plots.mm)

    for (panel_idx, (vname, vlabel)) in enumerate(zip(obs_vars, obs_labels))
        y_base = Float64.(irf_base(vname, :, sk_base))
        y_c2 = Float64.(irf_c2(vname, :, sk_c2))

        t_base = 1:length(y_base)
        t_c2 = 1:length(y_c2)

        plot!(p[panel_idx], t_base, y_base, label="Baseline (no ZLB)",
              color=:black, linewidth=2, linestyle=:solid)
        plot!(p[panel_idx], t_c2, y_c2, label="C2 (softplus ZLB)",
              color=:red, linewidth=2, linestyle=:dash)

        title!(p[panel_idx], vlabel)
        xlabel!(p[panel_idx], "Quarter")

        T_min = min(length(y_base), length(y_c2))
        diff = maximum(abs.(y_base[1:T_min] .- y_c2[1:T_min]))
        println("  $vname: max|base-C2|=$(round(diff, digits=6))")
    end

    plot!(p, plot_title="Negative $shock_sym shock (-1σ): Baseline vs Softplus ZLB")
    savefig(p, joinpath(figdir, "irf_compare_c2_$(shock_sym).png"))
    println("  Saved: irf_compare_c2_$(shock_sym).png")
end

# ── Large shock comparison ────────────────────────────────────────
println("\n" * "="^70)
println("Large shock comparison: eb=[-3,-2,-1.5,-1]σ, eqs=[-2,-1.5,-1]σ")
println("="^70)

p_big = plot(layout=(2,2), size=(1000, 700),
             titlefontsize=10, legendfontsize=7,
             left_margin=5Plots.mm, bottom_margin=3Plots.mm)

for (m_obj, m_label, m_color, m_style) in [
    (m_base, "Baseline (no ZLB)", :black, :solid),
    (m_c2, "C2: softplus ZLB", :red, :dash)
]
    shock_names = m_obj.exo
    nshocks = length(shock_names)
    shock_matrix = zeros(nshocks, 4)

    eb_idx = findfirst(x -> x == :eb, shock_names)
    eqs_idx = findfirst(x -> x == :eqs, shock_names)
    if !isnothing(eb_idx)
        shock_matrix[eb_idx, :] = [-3.0, -2.0, -1.5, -1.0]
    end
    if !isnothing(eqs_idx)
        shock_matrix[eqs_idx, 1:3] = [-2.0, -1.5, -1.0]
    end

    irf = get_irf(m_obj, shocks=shock_matrix, periods=T_irf, algorithm=:first_order,
                  levels=true, verbose=false)
    sk = axiskeys(irf, 3)[1]

    r_path = Float64.(irf(:r, :, sk))
    println("\n  $m_label:")
    println("    min(r) = $(round(minimum(r_path), digits=6)) (ann=$(round(400*(minimum(r_path)-1), digits=2))%)")
    println("    Periods r<1.0: $(sum(r_path .< 1.0)) / $(length(r_path))")

    for (panel_idx, (vname, vlabel)) in enumerate(zip(obs_vars, obs_labels))
        y = Float64.(irf(vname, :, sk))
        t = 1:length(y)
        plot!(p_big[panel_idx], t, y, label=m_label,
              color=m_color, linewidth=2, linestyle=m_style)
        title!(p_big[panel_idx], vlabel)
        xlabel!(p_big[panel_idx], "Quarter")
    end
end

hline!(p_big[1], [0.0], label="ZLB floor", color=:red, linewidth=1.5, linestyle=:solid, alpha=0.5)
plot!(p_big, plot_title="Large crisis shocks: Baseline vs Softplus ZLB")

savefig(p_big, joinpath(figdir, "irf_compare_c2_large_shock.png"))
savefig(p_big, joinpath(figdir, "irf_compare_c2_large_shock.pdf"))
println("\nSaved: irf_compare_c2_large_shock.png/.pdf")

# ── C2 shadow rate vs actual rate ─────────────────────────────────
println("\n" * "="^70)
println("C2 shadow rate vs actual rate (large shocks)")
println("="^70)

shock_names = m_c2.exo
nshocks = length(shock_names)
shock_matrix = zeros(nshocks, 4)
eb_idx = findfirst(x -> x == :eb, shock_names)
eqs_idx = findfirst(x -> x == :eqs, shock_names)
if !isnothing(eb_idx); shock_matrix[eb_idx, :] = [-3.0, -2.0, -1.5, -1.0]; end
if !isnothing(eqs_idx); shock_matrix[eqs_idx, 1:3] = [-2.0, -1.5, -1.0]; end

irf_c2_big = get_irf(m_c2, shocks=shock_matrix, periods=T_irf, algorithm=:first_order,
                     levels=true, verbose=false)
sk = axiskeys(irf_c2_big, 3)[1]

robs_actual = Float64.(irf_c2_big(:robs, :, sk))
r_actual = Float64.(irf_c2_big(:r, :, sk))

has_shadow = :robs_tilde in axiskeys(irf_c2_big, 1)
if has_shadow
    robs_shadow = Float64.(irf_c2_big(:robs_tilde, :, sk))
    r_tilde = Float64.(irf_c2_big(:r_tilde, :, sk))
else
    println("WARNING: robs_tilde not in IRF output")
    robs_shadow = robs_actual
    r_tilde = r_actual
end

pinf_c2 = Float64.(irf_c2_big(:pinfobs, :, sk))
dy_c2 = Float64.(irf_c2_big(:dy, :, sk))
dinve_c2 = Float64.(irf_c2_big(:dinve, :, sk))

T = length(robs_actual)
t_plot = 1:T

println("\nPeriod-by-period comparison:")
for t in 1:min(20, T)
    flag = has_shadow && r_tilde[t] < 1.001 ? " ← ZLB" : ""
    if has_shadow
        println("  t=$t: shadow=$(round(4*robs_shadow[t], digits=2))%, actual=$(round(4*robs_actual[t], digits=2))%$flag")
    else
        println("  t=$t: actual=$(round(4*robs_actual[t], digits=2))%$flag")
    end
end

# ZLB binding detection
if has_shadow
    zlb_binding = r_tilde .< 1.0
    n_zlb = sum(zlb_binding)
    println("\nZLB binding (shadow < 1.0): $n_zlb / $T periods")
else
    zlb_binding = r_actual .< 1.001
    n_zlb = sum(zlb_binding)
end

# ── Final paper figure ────────────────────────────────────────────
println("\n" * "="^70)
println("Generating paper figure")
println("="^70)

ss_r_ann = 400.0 * (Float64(ss_c2(:r)) - 1.0)
ss_pinf_ann = 400.0 * (Float64(ss_c2(:pinf)) - 1.0)
ctrend = Float64(ss_c2(:dy))

# ZLB binding region coordinates
zlb_starts = Int[]
zlb_ends = Int[]
if n_zlb > 0
    zlb_starts = findall(diff([false; zlb_binding]) .== 1)
    zlb_ends = findall(diff([zlb_binding; false]) .== -1)
end

p_paper = plot(layout=(2,2), size=(1000, 700),
               titlefontsize=11, guidefontsize=9, tickfontsize=8,
               legendfontsize=7, left_margin=5Plots.mm, bottom_margin=3Plots.mm)

# Panel 1: Policy rate
if has_shadow
    plot!(p_paper[1], t_plot, 4.0 .* robs_shadow,
          label="Shadow rate (Taylor rule)", color=:steelblue, linewidth=1.5, linestyle=:dash, alpha=0.7)
end
plot!(p_paper[1], t_plot, 4.0 .* robs_actual,
      label="Actual rate (softplus ZLB)", color=:navy, linewidth=2.5)
hline!(p_paper[1], [0.0], label="ZLB floor", color=:red, linestyle=:solid, linewidth=1.5)
hline!(p_paper[1], [ss_r_ann], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (k, (s, e)) in enumerate(zip(zlb_starts, zlb_ends))
    vspan!(p_paper[1], [s-0.5, e+0.5], alpha=0.10, color=:red, label=(k==1 ? "ZLB regime" : ""))
end
title!(p_paper[1], "Policy Rate (annualized)")
ylabel!(p_paper[1], "% p.a.")
xlabel!(p_paper[1], "Quarter")

# Panel 2: Inflation
plot!(p_paper[2], t_plot, 4.0 .* pinf_c2,
      label="Inflation", color=:darkred, linewidth=2)
hline!(p_paper[2], [ss_pinf_ann], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p_paper[2], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p_paper[2], "Inflation (annualized)")
ylabel!(p_paper[2], "% p.a.")
xlabel!(p_paper[2], "Quarter")

# Panel 3: Output growth
plot!(p_paper[3], t_plot, dy_c2,
      label="Output growth", color=:darkgreen, linewidth=2)
hline!(p_paper[3], [ctrend], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p_paper[3], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p_paper[3], "Output Growth")
ylabel!(p_paper[3], "% (quarterly)")
xlabel!(p_paper[3], "Quarter")

# Panel 4: Investment growth
plot!(p_paper[4], t_plot, dinve_c2,
      label="Investment growth", color=:purple, linewidth=2)
hline!(p_paper[4], [ctrend], label="", color=:gray, linestyle=:dot, linewidth=0.8)
for (s, e) in zip(zlb_starts, zlb_ends)
    vspan!(p_paper[4], [s-0.5, e+0.5], alpha=0.10, color=:red, label="")
end
title!(p_paper[4], "Investment Growth")
ylabel!(p_paper[4], "% (quarterly)")
xlabel!(p_paper[4], "Quarter")

savefig(p_paper, joinpath(figdir, "fig_zlb_binding_simulation.pdf"))
savefig(p_paper, joinpath(figdir, "fig_zlb_binding_simulation.png"))
println("Saved: fig_zlb_binding_simulation.pdf/.png")

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("  Model: Smets_Wouters_2007_HLT_zlb (softplus, κ=100)")
println("  Periods: $T")
println("  ZLB binding: $n_zlb / $T ($(round(100*n_zlb/T, digits=1))%)")
println("  Actual rate range: [$(round(minimum(4*robs_actual), digits=2)), $(round(maximum(4*robs_actual), digits=2))] % ann")
if has_shadow
    println("  Shadow rate range: [$(round(minimum(4*robs_shadow), digits=2)), $(round(maximum(4*robs_shadow), digits=2))] % ann")
end
println("  Inflation range: [$(round(minimum(4*pinf_c2), digits=2)), $(round(maximum(4*pinf_c2), digits=2))] % ann")
println("  Output growth range: [$(round(minimum(dy_c2), digits=2)), $(round(maximum(dy_c2), digits=2))] %")
println("  Investment growth range: [$(round(minimum(dinve_c2), digits=2)), $(round(maximum(dinve_c2), digits=2))] %")
println("="^70)
