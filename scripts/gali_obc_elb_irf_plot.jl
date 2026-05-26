#!/usr/bin/env julia

# Plot a Galí (2015) Ch. 3 OBC impulse response in which the ELB binds.
#
# The binding indicator is based on the shadow Taylor-rule rate:
#
#   R_shadow = (1 / beta) * Pi^phi_pi * (Y / Y_ss)^phi_y * exp(nu)
#
# The plotted policy rate is the solved OBC rate R.

ENV["GKSwstype"] = "100"

using MacroModelling
using Plots
using Printf
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "Gali_2015_chapter_3_obc.jl"))

Base.@kwdef struct IrfOptions
    shock_size::Float64 = -20.0
    periods::Int = 16
    sep_horizon::Int = 10
    sep_maxit::Int = 150
    sep_accept_tol::Float64 = 1e-2
    out_dir::String = joinpath(REPO_ROOT, ".local_artifacts", "gali_elb_irf")
end

function parse_irf_args(args)
    opts = IrfOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    return IrfOptions(
        shock_size = parse(Float64, get(values, "shock-size", string(opts.shock_size))),
        periods = parse(Int, get(values, "periods", string(opts.periods))),
        sep_horizon = parse(Int, get(values, "sep-horizon", string(opts.sep_horizon))),
        sep_maxit = parse(Int, get(values, "sep-maxit", string(opts.sep_maxit))),
        sep_accept_tol = parse(Float64, get(values, "sep-accept-tol", string(opts.sep_accept_tol))),
        out_dir = get(values, "out-dir", opts.out_dir),
    )
end

function find_idx(names, name::Symbol)
    idx = findfirst(==(name), Symbol.(names))
    idx === nothing && error("Missing symbol $name")
    return idx
end

function run_irf(opts::IrfOptions)
    mkpath(opts.out_dir)
    model = Gali_2015_chapter_3_obc
    var_idx = Dict(s => find_idx(model.var, s) for s in [:R, :Y, :Pi, :nu, :log_y, :pi_ann])
    shock_idx = find_idx(model.exo, :eps_nu)
    par_idx = Dict(s => find_idx(model.parameters, s) for s in [:R̄, :β, :ϕᵖⁱ, :ϕʸ])
    params = Float64.(model.parameter_values)
    rbar = params[par_idx[:R̄]]
    beta = params[par_idx[:β]]
    phi_pi = params[par_idx[:ϕᵖⁱ]]
    phi_y = params[par_idx[:ϕʸ]]

    shocks = zeros(Float64, length(model.exo), opts.periods)
    shocks[shock_idx, 1] = opts.shock_size

    println("Running Galí OBC SEP IRF with eps_nu shock $(opts.shock_size)...")
    res = MacroModelling.simulate_sep_extended_path(
        model;
        periods = opts.periods,
        shocks = shocks,
        burn_in = 0,
        sep_horizon = opts.sep_horizon,
        sep_order = 1,
        sep_nnodes = 3,
        sep_sparse_tree = true,
        sep_maxit = opts.sep_maxit,
        sep_tol = 1e-7,
        sep_accept_tol = opts.sep_accept_tol,
        shock_scaling = :none,
        silent = true,
    )

    sim = Float64.(Array(res.simulation))
    T = min(opts.periods, size(sim, 2) - 1)
    t = collect(1:T)
    y_ss = sim[var_idx[:Y], 1]
    log_y_ss = sim[var_idx[:log_y], 1]

    R = vec(sim[var_idx[:R], 2:(T + 1)])
    Y = vec(sim[var_idx[:Y], 2:(T + 1)])
    Pi = vec(sim[var_idx[:Pi], 2:(T + 1)])
    nu = vec(sim[var_idx[:nu], 2:(T + 1)])
    log_y = vec(sim[var_idx[:log_y], 2:(T + 1)])
    pi_ann = vec(sim[var_idx[:pi_ann], 2:(T + 1)])
    shadow = (1 / beta) .* (Pi .^ phi_pi) .* ((Y ./ y_ss) .^ phi_y) .* exp.(nu)
    binding = shadow .<= rbar

    actual_rate_ann_pct = 400 .* log.(R)
    shadow_rate_ann_pct = 400 .* log.(shadow)
    floor_ann_pct = 400 * log(rbar)
    output_gap_pct = 100 .* (log_y .- log_y_ss)
    inflation_ann_pct = 100 .* pi_ann
    nu_pct = 100 .* nu

    gr()
    default(
        fontfamily = "Computer Modern",
        linewidth = 2,
        grid = true,
        framestyle = :box,
        legendfontsize = 9,
        guidefontsize = 10,
        tickfontsize = 9,
        titlefontsize = 11,
        dpi = 220,
    )

    p1 = plot(t, actual_rate_ann_pct;
              label = "OBC policy rate",
              color = :steelblue,
              ylabel = "% annualized",
              title = "Policy Rate and Shadow Rate",
              xlabel = "Quarters")
    plot!(p1, t, shadow_rate_ann_pct; label = "Shadow Taylor-rule rate", color = :crimson, linestyle = :dash)
    hline!(p1, [floor_ann_pct]; label = "ELB", color = :black, linestyle = :dot)
    scatter!(p1, t[binding], actual_rate_ann_pct[binding]; label = "ELB binds", color = :black, markersize = 4)

    p2 = plot(t, output_gap_pct;
              label = "",
              color = :forestgreen,
              ylabel = "% log deviation",
              title = "Output",
              xlabel = "Quarters")
    hline!(p2, [0.0]; label = "", color = :black, linestyle = :dot)

    p3 = plot(t, inflation_ann_pct;
              label = "",
              color = :darkorange,
              ylabel = "% annualized",
              title = "Inflation",
              xlabel = "Quarters")
    hline!(p3, [0.0]; label = "", color = :black, linestyle = :dot)

    p4 = plot(t, nu_pct;
              label = "nu",
              color = :purple,
              ylabel = "%",
              title = "Monetary Policy Wedge",
              xlabel = "Quarters")
    hline!(p4, [0.0]; label = "", color = :black, linestyle = :dot)

    fig = plot(p1, p2, p3, p4;
               layout = (2, 2),
               size = (1050, 720),
               plot_title = "Galí OBC IRF: ELB-Binding Monetary Policy Shock",
               margin = 6Plots.mm)

    stem = joinpath(opts.out_dir, "gali_obc_elb_irf_eps_nu_m$(round(Int, abs(opts.shock_size)))")
    png_path = stem * ".png"
    pdf_path = stem * ".pdf"
    savefig(fig, png_path)
    savefig(fig, pdf_path)

    summary_path = stem * "_summary.md"
    open(summary_path, "w") do io
        println(io, "# Galí OBC ELB-Binding IRF")
        println(io)
        println(io, "- Shock: `eps_nu = $(opts.shock_size)` in period 1")
        println(io, "- Periods: $(T)")
        println(io, "- SEP horizon: $(opts.sep_horizon)")
        println(io, "- SEP maxit: $(opts.sep_maxit)")
        println(io, "- SEP accept tolerance: $(opts.sep_accept_tol)")
        println(io, "- SEP error flag: $(hasproperty(res, :errorflag) ? res.errorflag : missing)")
        println(io, "- ELB binding periods by shadow-rate criterion: $(sum(binding)) / $(T)")
        println(io, "- Minimum shadow rate: $(@sprintf("%.4f", minimum(shadow)))")
        println(io, "- Minimum shadow rate, annualized percent: $(@sprintf("%.3f", minimum(shadow_rate_ann_pct)))")
        println(io, "- Minimum OBC policy rate, annualized percent: $(@sprintf("%.3f", minimum(actual_rate_ann_pct)))")
        println(io, "- Figure PNG: `$png_path`")
        println(io, "- Figure PDF: `$pdf_path`")
    end

    println("Saved PNG: $png_path")
    println("Saved PDF: $pdf_path")
    println("Saved summary: $summary_path")
    println("ELB binding periods: $(sum(binding)) / $(T)")
    println("Min shadow annualized %: $(round(minimum(shadow_rate_ann_pct), digits = 3))")
    println("Min actual annualized %: $(round(minimum(actual_rate_ann_pct), digits = 3))")
    return (png = png_path, pdf = pdf_path, summary = summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_irf(parse_irf_args(ARGS))
end
