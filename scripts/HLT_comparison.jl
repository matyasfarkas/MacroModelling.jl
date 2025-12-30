using MacroModelling, StatsPlots, Printf, AxisKeys

include("../models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

shock = :epinf
shock_size = 1.0
periods = 20

key_vars = [:y, :c, :inve, :pinfobs, :r, :w, :lab]

println("="^70)
println("HLT IRF COMPARISON: PERTURBATION ORDERS vs SEP")
println("="^70)
println("Shock: $(shock)")
println("Periods: $(periods)")
println("Variables: $(key_vars)")

println("\n1. Computing perturbation IRFs...")
irf_fo = get_irf(m; shocks=shock, variables=key_vars, periods=periods, algorithm=:first_order)
irf_so = get_irf(m; shocks=shock, variables=key_vars, periods=periods, algorithm=:second_order)
# irf_p3 = get_irf(m; shocks=shock, variables=key_vars, periods=periods, algorithm=:pruned_third_order)

println("✓ Perturbation IRFs computed")

sep_periods = max(periods, 40)
sep_order = 1
sep_nnodes = 3
sep_tol = 5e-3

println("\n2. Computing SEP IRF (funnel baseline)...")
irf_sep = get_sep_irf(m, shock, shock_size;
                      variables=key_vars,
                      periods=periods,
                      method=:funnel,
                      baseline=:zero_shock,
                      shock_scaling=:none,
                      sep_periods=sep_periods,
                      sep_order=sep_order,
                      sep_nnodes=sep_nnodes,
                      sep_tol=sep_tol,
                      sep_sparse_tree=true,
                      silent=false)

println("✓ SEP IRF computed")

function series_for(irf, var)
    var_idx = findfirst(==(var), axiskeys(irf, 1))
    if isnothing(var_idx)
        return nothing
    end
    return Float64.(irf[var_idx, :, 1])
end

function series_for_sep(irf, var)
    var_idx = findfirst(==(var), axiskeys(irf, 1))
    if isnothing(var_idx)
        return nothing
    end
    return Float64.(irf[var_idx, 2:end])
end

time_mm = collect(axiskeys(irf_fo, 2))
time_sep = time_mm

println("\n3. Plotting comparison...")
p = plot(layout=(3,3), size=(1200, 900),
         plot_title="Smets_Wouters_2007_HLT: epinf IRFs (SEP vs perturbation)")

for (i, var) in enumerate(key_vars)
    fo = series_for(irf_fo, var)
    so = series_for(irf_so, var)
    sep = series_for_sep(irf_sep, var)

    show_legend = (i == 1)
    fo_label = show_legend ? "1st order" : ""
    so_label = show_legend ? "2nd order" : ""
    sep_label = show_legend ? "SEP" : ""

    plot!(p[i], time_mm, fo, label=fo_label, color=:blue, linewidth=2)
    plot!(p[i], time_mm, so, label=so_label, color=:green, linewidth=2, linestyle=:dash)
    plot!(p[i], time_sep, sep, label=sep_label, color=:black, linewidth=2, linestyle=:dashdot)
    hline!(p[i], [0], color=:black, linestyle=:dot, label="")
    plot!(p[i], title=string(var))
end

pdf_path = joinpath(@__DIR__, "HLT_comparison_sep_irf.pdf")
savefig(p, pdf_path)
println("✓ Saved: $(pdf_path)")
