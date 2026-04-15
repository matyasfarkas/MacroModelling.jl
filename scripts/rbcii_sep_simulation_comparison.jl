# RBC-II SEP simulation comparison against Dynare `.mat` outputs
#
# This script reads Dynare dseries `.mat` files (DATA__/NAMES__) and:
# 1) recovers the shock sequence from `efficiency`
# 2) runs MacroModelling SEP extended-path simulations
# 3) compares Investment paths and reproduces the paper-style figure

using MacroModelling
using MAT
using StatsPlots
using Statistics

include("../models/RBCII_Dynare.jl")

function sep_validation_root()
    candidates = (
        joinpath(@__DIR__, "..", "test", "fixtures", "sep_validation"),
        joinpath(@__DIR__, "..", "tests", "sep_validation"),
    )
    for path in candidates
        isdir(path) && return path
    end
    return first(candidates)
end

const DATA_DIR = joinpath(sep_validation_root(), "sep_simulation_data", "accuracy-sc")

struct DynareDSeries
    names::Vector{String}
    data::Matrix{Float64}
end

function load_dynare_dseries(path::AbstractString)
    isfile(path) || error("Missing Dynare .mat file: $path")
    d = matread(path)
    names = vec(String.(d["NAMES__"]))
    data = Matrix{Float64}(d["DATA__"])
    return DynareDSeries(names, data)
end

function get_series(ds::DynareDSeries, name::AbstractString)
    idx = findfirst(==(name), ds.names)
    idx === nothing && error("Series \"$name\" not found in $(join(ds.names, ", ")).")
    return vec(ds.data[:, idx])
end

function implied_shocks_from_efficiency(eff::AbstractVector{<:Real}; rho::Real, sigma::Real)
    n = length(eff)
    n < 2 && error("Need at least 2 observations to infer shocks.")
    shocks = zeros(Float64, n - 1)
    for t in 2:n
        shocks[t - 1] = (eff[t] - rho * eff[t - 1]) / sigma
    end
    return shocks
end

function dynare_rbcii_path(order::Int; sigma_tag::AbstractString="007", algo::Int=1, hybrid::Int=0)
    fname = "rbcii-$(sigma_tag)-sep-$(order)-algo-$(algo)-hybrid-$(hybrid).mat"
    return joinpath(DATA_DIR, fname)
end

function main()
    m = RBCII_Dynare

# -------------------------------------------------------------------
# User configuration (paper defaults)
# -------------------------------------------------------------------
orders = [0, 1, 2, 5]
sigma_tag = "007"
algo = 1
hybrid = 0

rho = 0.950
sigma = 0.007

sep_horizon = 200
sep_nnodes = 3
sep_tol = 1e-5
sep_sparse_tree = true
sep_periods = 220   # paper figure uses 220 periods, plotting 120:220

plot_start = 120
plot_end = 220
# -------------------------------------------------------------------

dynare_data = Dict{Int, NamedTuple}()
shocks_ref = nothing
shock_diff = Dict{Int, Float64}()

for order in orders
    path = dynare_rbcii_path(order; sigma_tag=sigma_tag, algo=algo, hybrid=hybrid)
    ds = load_dynare_dseries(path)
    investment = get_series(ds, "Investment")
    efficiency = get_series(ds, "efficiency")

    shocks = implied_shocks_from_efficiency(efficiency; rho=rho, sigma=sigma)
    if shocks_ref === nothing
        shocks_ref = shocks
    else
        if length(shocks) != length(shocks_ref)
            error("Shock length mismatch for order=$order.")
        end
        shock_diff[order] = maximum(abs.(shocks .- shocks_ref))
    end

    dynare_data[order] = (investment=investment, efficiency=efficiency, path=path)
end

if !isempty(shock_diff)
    for (order, maxdiff) in sort(collect(shock_diff); by=first)
        if maxdiff > 1e-10
            println("Warning: shock sequence differs for order=$order (max diff = $maxdiff).")
        end
    end
end

if shocks_ref === nothing
    error("Failed to infer shocks from Dynare data.")
end

periods = min(sep_periods, length(shocks_ref))
shocks_used = reshape(shocks_ref[1:periods], 1, :)

investment_ss = get_steady_state(m, derivatives=false)(:Investment)
zlb_idx = findfirst(==(:ZLB), m.parameters)
zlb_idx === nothing && error("ZLB parameter not found.")
investment_floor = m.parameter_values[zlb_idx] * investment_ss

dynare_series_by_order = Dict{Int, Vector{Float64}}()
mm_series_by_order = Dict{Int, Vector{Float64}}()
metrics = Dict{Int, NamedTuple}()

for order in orders
    dynare_inv = dynare_data[order].investment[1:(periods + 1)]
    dynare_series_by_order[order] = dynare_inv

    println("Running MacroModelling SEP extended-path simulation (order=$order)...")
        res = MacroModelling.simulate_sep_extended_path(
            m;
            periods = periods,
            shocks = shocks_used,
        sep_horizon = sep_horizon,
        sep_order = order,
        sep_nnodes = sep_nnodes,
        sep_tol = sep_tol,
        sep_sparse_tree = sep_sparse_tree,
        shock_scaling = :none,
        silent = true
    )

    if res.errorflag
        println("SEP failed in period ", res.failure_period, " for order=$order.")
    end

    inv_idx = findfirst(==(:Investment), axiskeys(res.simulation, 1))
    mm_series = Float64.(res.simulation[inv_idx, :])
    mm_series_by_order[order] = mm_series

    n = min(length(mm_series), length(dynare_inv))
    mm_trim = mm_series[2:n]
    dyn_trim = dynare_inv[2:n]
    rmse = sqrt(mean((mm_trim .- dyn_trim).^2))
    max_abs = maximum(abs.(mm_trim .- dyn_trim))
    metrics[order] = (rmse=rmse, max_abs=max_abs, n=n)
end

println("Comparison metrics (Investment, t>=1):")
for order in orders
    mtr = metrics[order]
    println("  order=$(order): RMSE=$(mtr.rmse), Max abs diff=$(mtr.max_abs), n=$(mtr.n)")
end

plot_start = max(plot_start, 1)
plot_end = min(plot_end, periods + 1)
plot_idx = plot_start:plot_end

colors = cgrad([:black, :red], length(orders), categorical=true)

paper_dyn = plot(title="Dynare RBCII investment (orders $(join(orders, ",")))",
                 xlabel="Index", ylabel="Investment", legend=:topright)
paper_mm = plot(title="MacroModelling RBCII investment (orders $(join(orders, ",")))",
                xlabel="Index", ylabel="Investment", legend=:topright)

for (i, order) in enumerate(orders)
    dyn = dynare_series_by_order[order]
    mm = mm_series_by_order[order]
    lw = order == 0 ? 2.5 : 1.5
    plot!(paper_dyn, plot_idx, dyn[plot_idx], label="order=$(order)", color=colors[i], linewidth=lw)
    plot!(paper_mm, plot_idx, mm[plot_idx], label="order=$(order)", color=colors[i], linewidth=lw)
end

hline!(paper_dyn, [investment_ss], label="SS", linestyle=:dot, color=:black)
hline!(paper_dyn, [investment_floor], label="Floor", linestyle=:dashdot, color=:red)
hline!(paper_mm, [investment_ss], label="SS", linestyle=:dot, color=:black)
hline!(paper_mm, [investment_floor], label="Floor", linestyle=:dashdot, color=:red)

paper_plot = plot(paper_dyn, paper_mm, layout=(2, 1), size=(900, 900))
paper_path = joinpath(@__DIR__, "rbcii_sep_simulation_paper_defaults.pdf")
savefig(paper_plot, paper_path)
println("Saved paper-style plot: ", paper_path)

comparison_plot = plot(layout=(2, 2), size=(900, 700))
for (i, order) in enumerate(orders)
    dyn = dynare_series_by_order[order]
    mm = mm_series_by_order[order]
    plot!(comparison_plot[i], plot_idx, dyn[plot_idx], label="Dynare", linewidth=2)
    plot!(comparison_plot[i], plot_idx, mm[plot_idx], label="MacroModelling", linewidth=2, linestyle=:dash)
    hline!(comparison_plot[i], [investment_ss], label="SS", linestyle=:dot, color=:black)
    hline!(comparison_plot[i], [investment_floor], label="Floor", linestyle=:dashdot, color=:red)
    plot!(comparison_plot[i], title="order=$(order)", xlabel="Index", ylabel="Investment")
end

comparison_path = joinpath(@__DIR__, "rbcii_sep_simulation_comparison_orders.pdf")
savefig(comparison_plot, comparison_path)
println("Saved comparison plot: ", comparison_path)

end

main()
