#!/usr/bin/env julia
#
# ZLB-binding simulation using the SEP nonlinear solver.
#
# Uses the C2 softplus ZLB model (Smets_Wouters_2007_HLT_zlb.jl) where
# the Taylor rule embeds r = 1 + log(1+exp(κ(r̃-1)))/κ directly in the
# model equations. The SEP Newton solver evaluates this at full nonlinear
# precision, so the ZLB is properly enforced through general equilibrium.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Serialization
using Plots; gr()

# ── Load model ──────────────────────────────────────────────────────
println("Loading MacroModelling...")
include(joinpath(@__DIR__, "..", "src", "MacroModelling.jl"))
using .MacroModelling

println("Loading C2 softplus ZLB model...")
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_HLT_zlb.jl"))
𝓂 = Smets_Wouters_2007_HLT_zlb

# ── Steady state ────────────────────────────────────────────────────
println("Computing steady state...")
SS = get_steady_state(𝓂, derivatives=false)
r_ss = Float64(SS(:r))
r_tilde_ss = Float64(SS(:r_tilde))
pinf_ss = Float64(SS(:pinf))
r_ss_ann = 400.0 * (r_ss - 1.0)
pinf_ss_ann = 400.0 * (pinf_ss - 1.0)
ctrend = Float64(SS(:dy))
println("  r[ss]       = $r_ss  (ann=$(round(r_ss_ann, digits=2))%)")
println("  r_tilde[ss] = $r_tilde_ss (shadow rate)")
println("  pinf[ss]    = $pinf_ss  (ann=$(round(pinf_ss_ann, digits=2))%)")
println("  dy[ss]      = $ctrend")

# ── Build crisis shock matrix ───────────────────────────────────────
# We want to push the shadow rate well below zero so the softplus
# ZLB binding is clearly visible. Use a sequence of large negative
# preference (eb) and equity premium (eqs) shocks.
#
# Shocks enter the model as: z_e*/100 * SCALE * e[x]
# So to get a -Nσ shock we pass -N in the shock matrix (the model
# equations handle the scaling internally via z_e*/100 * SCALE).

T_sim = 40
nshocks = length(𝓂.exo)
shock_names = 𝓂.exo

# Deterministic shock sequence: large crisis in first 3 periods
shocks = zeros(nshocks, T_sim)

eb_idx = findfirst(x -> x == :eb, shock_names)
eqs_idx = findfirst(x -> x == :eqs, shock_names)

# Moderate negative preference shocks — enough to push shadow rate
# below zero but not so large that Newton diverges.
# At 1σ eb the rate drops ~2.5% ann from SS (8.21%), so ~1.5σ
# should push it near zero, and 2σ should go negative.
if !isnothing(eb_idx)
    shocks[eb_idx, 1] = -2.0   # Main crisis shock
    shocks[eb_idx, 2] = -1.0   # Follow-up
    shocks[eb_idx, 3] = -0.5   # Tail
end
if !isnothing(eqs_idx)
    shocks[eqs_idx, 1] = -1.0  # Equity premium spike
    shocks[eqs_idx, 2] = -0.5
end

println("\nShock matrix (non-zero entries):")
for (i, sn) in enumerate(shock_names)
    row = shocks[i, :]
    if any(row .!= 0)
        nz = findall(row .!= 0)
        println("  $sn: t=$(nz) → $(round.(row[nz], digits=2))")
    end
end

# ── Run SEP extended-path simulation ────────────────────────────────
println("\n" * "="^60)
println("Running SEP nonlinear extended-path simulation")
println("="^60)
println("  Periods: $T_sim")
println("  SEP horizon: 10")
println("  SEP order: 1")
println("  SEP nnodes: 3")
println("  SEP maxit: 200")
println("  SEP tol: 1e-5")
println("  Accept tol: 1.0")

t_start = time()
result = simulate_sep_extended_path(𝓂;
    periods = T_sim,
    shocks = shocks,
    burn_in = 0,
    sep_horizon = 10,
    sep_order = 1,
    sep_nnodes = 3,
    sep_maxit = 200,
    sep_tol = 1e-5,
    sep_accept_tol = 1.0,
    sep_sparse_tree = true,
    sep_shock_scale = 1.0,
    shock_scaling = :none,
    random_seed = 42,
    silent = false
)
elapsed = time() - t_start

println("\nSEP simulation completed in $(round(elapsed, digits=1)) seconds")
if result.errorflag
    println("WARNING: SEP failed at period $(result.failure_period)")
end

# ── Serialize result for fast replotting ─────────────────────────────
cache_path = joinpath(@__DIR__, "..", ".local_artifacts", "zlb_sep_simulation_cache.jls")
mkpath(dirname(cache_path))
serialize(cache_path, (simulation=parent(parent(result.simulation)),
                       var_names=collect(axiskeys(result.simulation, 1)),
                       time_keys=collect(axiskeys(result.simulation, 2)),
                       sep_errors=result.sep_errors,
                       errorflag=result.errorflag,
                       failure_period=result.failure_period))
println("Cached simulation to: $cache_path")

# ── Extract variables ───────────────────────────────────────────────
sim = result.simulation
var_names = axiskeys(sim, 1)

# Find variable indices
r_idx = findfirst(==(:r), var_names)
r_tilde_idx = findfirst(==(:r_tilde), var_names)
robs_idx = findfirst(==(:robs), var_names)
robs_tilde_idx = findfirst(==(:robs_tilde), var_names)
pinf_idx = findfirst(==(:pinf), var_names)
pinfobs_idx = findfirst(==(:pinfobs), var_names)
dy_idx = findfirst(==(:dy), var_names)
dinve_idx = findfirst(==(:dinve), var_names)

# Time axis: simulation starts from period 0 (SS), periods 1:T_sim are simulation
time_ax = axiskeys(sim, 2)

# Extract series (skip period 0 = initial state)
r_path = Float64.(sim[r_idx, :])
robs_path = Float64.(sim[robs_idx, :])
pinf_path = Float64.(sim[pinf_idx, :])

# Shadow rate
has_shadow = !isnothing(r_tilde_idx)
if has_shadow
    r_tilde_path = Float64.(sim[r_tilde_idx, :])
    robs_tilde_path = Float64.(sim[robs_tilde_idx, :])
end

pinfobs_path = Float64.(sim[pinfobs_idx, :])
dy_path = Float64.(sim[dy_idx, :])
dinve_path = Float64.(sim[dinve_idx, :])

# Annualize: robs is already 100*(r-1), so annualize = 4*robs
robs_ann = 4.0 .* robs_path
pinf_ann = 4.0 .* pinfobs_path
if has_shadow
    robs_tilde_ann = 4.0 .* robs_tilde_path
end

# Use periods 1:end (skip initial state at index 1 which is period 0)
T = length(robs_ann) - 1
t_plot = 1:T
robs_ann_plot = robs_ann[2:end]
pinf_ann_plot = pinf_ann[2:end]
dy_plot = dy_path[2:end]
dinve_plot = dinve_path[2:end]
if has_shadow
    robs_tilde_ann_plot = robs_tilde_ann[2:end]
end

# ── Summary statistics ──────────────────────────────────────────────
println("\n" * "="^60)
println("Period-by-period rates (annualized)")
println("="^60)

# ZLB binding = actual rate pinned near zero.
# With softplus κ=100 the floor is 0 from above; define "at ZLB"
# as actual annualized rate < 1.0% (well below normal ~8%).
zlb_threshold_ann = 1.0  # % annualized
zlb_binding = zeros(Bool, T)
for t in 1:T
    zlb_binding[t] = robs_ann_plot[t] < zlb_threshold_ann
end

for t in 1:min(25, T)
    flag = zlb_binding[t] ? " ← ZLB binds" : ""
    if has_shadow
        println("  t=$t: shadow=$(round(robs_tilde_ann_plot[t], digits=2))%, actual=$(round(robs_ann_plot[t], digits=2))%$flag")
    else
        println("  t=$t: actual=$(round(robs_ann_plot[t], digits=2))%$flag")
    end
end
if T > 25; println("  ... (truncated)"); end

n_zlb = sum(zlb_binding)
println("\nZLB binding periods (actual < $(zlb_threshold_ann)% ann): $n_zlb / $T")
println("Actual rate range: [$(round(minimum(robs_ann_plot), digits=2)), $(round(maximum(robs_ann_plot), digits=2))] % ann")
if has_shadow
    println("Shadow rate range: [$(round(minimum(robs_tilde_ann_plot), digits=2)), $(round(maximum(robs_tilde_ann_plot), digits=2))] % ann")
end

# ── Plot paper figure ───────────────────────────────────────────────
println("\n" * "="^60)
println("Generating paper figure (SEP nonlinear)")
println("="^60)

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
         titlefontsize=11, guidefontsize=9, tickfontsize=8,
         legendfontsize=7, left_margin=5Plots.mm, bottom_margin=3Plots.mm)

# Panel 1: Policy rate — shadow vs. actual (SEP nonlinear)
if has_shadow
    plot!(p[1], t_plot, robs_tilde_ann_plot,
          label="Shadow rate (Taylor rule)", color=:steelblue, linewidth=1.5,
          linestyle=:dash, alpha=0.7)
end
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
println("Saved: fig_zlb_binding_simulation.pdf/.png")

# ── SEP residual quality ────────────────────────────────────────────
println("\n" * "="^60)
println("SEP solver residuals per period")
println("="^60)
for (i, err) in enumerate(result.sep_errors)
    if i <= T_sim
        flag = isnan(err) ? " (NaN)" : (err > 0.1 ? " ← HIGH" : "")
        println("  t=$i: $(round(err, digits=6))$flag")
    end
end

# ── Final summary ───────────────────────────────────────────────────
println("\n" * "="^60)
println("SUMMARY (SEP NONLINEAR)")
println("="^60)
println("  Model: Smets_Wouters_2007_HLT_zlb (softplus, κ=100)")
println("  Solver: SEP extended-path (Newton, nonlinear)")
println("  Periods: $T")
println("  Runtime: $(round(elapsed, digits=1)) seconds")
println("  SEP failures: $(result.errorflag ? "YES at period $(result.failure_period)" : "None")")
println("  ZLB binding: $n_zlb / $T ($(round(100*n_zlb/T, digits=1))%)")
println("  Actual rate range: [$(round(minimum(robs_ann_plot), digits=2)), $(round(maximum(robs_ann_plot), digits=2))] % ann")
if has_shadow
    println("  Shadow rate range: [$(round(minimum(robs_tilde_ann_plot), digits=2)), $(round(maximum(robs_tilde_ann_plot), digits=2))] % ann")
end
println("  Inflation range: [$(round(minimum(pinf_ann_plot), digits=2)), $(round(maximum(pinf_ann_plot), digits=2))] % ann")
println("  Output growth range: [$(round(minimum(dy_plot), digits=2)), $(round(maximum(dy_plot), digits=2))] %")
println("  Investment growth range: [$(round(minimum(dinve_plot), digits=2)), $(round(maximum(dinve_plot), digits=2))] %")
println("="^60)
