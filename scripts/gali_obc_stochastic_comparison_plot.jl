#!/usr/bin/env julia

# Stochastic Galí OBC simulation against the same-shock linearized model.
#
# The constrained path uses MacroModelling's first-order OBC simulator
# (`ignore_obc=false`).  The linear path uses the same shock matrix with OBC
# enforcement disabled (`ignore_obc=true`).  The plot is limited to output,
# inflation, and the policy rate.

ENV["GKSwstype"] = "100"

using MacroModelling
using AxisKeys
using Plots
using Printf
using Random
using Serialization
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "Gali_2015_chapter_3_obc.jl"))

Base.@kwdef struct StochCompareOptions
    periods::Int = 24
    seed::Int = 20260515
    shock_scale::Float64 = 0.0
    elb_period::Int = 6
    elb_shock_name::Symbol = :eps_z
    elb_shock::Float64 = 0.8
    elb_span::Int = 5
    elb_decay::Float64 = 1.0
    tail_periods::Int = 0
    floor_tolerance_pct::Float64 = 0.25
    out_dir::String = joinpath(REPO_ROOT, ".local_artifacts", "gali_elb_stochastic")
end

function parse_stoch_args(args)
    opts = StochCompareOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    return StochCompareOptions(
        periods = parse(Int, get(values, "periods", string(opts.periods))),
        seed = parse(Int, get(values, "seed", string(opts.seed))),
        shock_scale = parse(Float64, get(values, "shock-scale", string(opts.shock_scale))),
        elb_period = parse(Int, get(values, "elb-period", string(opts.elb_period))),
        elb_shock_name = Symbol(get(values, "elb-shock-name", string(opts.elb_shock_name))),
        elb_shock = parse(Float64, get(values, "elb-shock", string(opts.elb_shock))),
        elb_span = parse(Int, get(values, "elb-span", string(opts.elb_span))),
        elb_decay = parse(Float64, get(values, "elb-decay", string(opts.elb_decay))),
        tail_periods = parse(Int, get(values, "tail-periods", string(opts.tail_periods))),
        floor_tolerance_pct = parse(Float64, get(values, "floor-tolerance-pct", string(opts.floor_tolerance_pct))),
        out_dir = get(values, "out-dir", opts.out_dir),
    )
end

function find_idx(names, name::Symbol)
    idx = findfirst(==(name), Symbol.(names))
    idx === nothing && error("Missing symbol $name")
    return idx
end

function build_stochastic_shocks(model, opts::StochCompareOptions)
    rng = MersenneTwister(opts.seed)
    shocks = zeros(Float64, length(model.exo), opts.periods)
    structural = [:eps_a, :eps_z, :eps_nu]
    for shock_name in structural
        shocks[find_idx(model.exo, shock_name), :] .= opts.shock_scale .* randn(rng, opts.periods)
    end
    elb_idx = find_idx(model.exo, opts.elb_shock_name)
    opts.elb_span >= 1 || error("--elb-span must be at least 1.")
    0.0 < opts.elb_decay <= 1.0 || error("--elb-decay must be in (0, 1].")
    for j in 0:(opts.elb_span - 1)
        t = opts.elb_period + j
        t <= opts.periods || break
        shocks[elb_idx, t] = opts.elb_shock * opts.elb_decay^j
    end
    return shocks
end

function keyed_series(irf, variable::Symbol, T::Int)
    shock_key = axiskeys(irf, 3)[1]
    return vec(Float64.(irf(variable, :, shock_key)))[1:T]
end

function run_stochastic_comparison(opts::StochCompareOptions)
    mkpath(opts.out_dir)
    model = deepcopy(Gali_2015_chapter_3_obc)
    shocks = build_stochastic_shocks(model, opts)
    shock_keys = model.timings.exo
    keyed_shocks = KeyedArray(shocks; Shocks = shock_keys, Periods = 1:opts.periods)

    println("Running Galí OBC first-order simulation...")
    obc_model = deepcopy(model)
    obc_irf = MacroModelling.get_irf(
        obc_model;
        shocks = keyed_shocks,
        periods = opts.tail_periods,
        variables = [:log_y, :pi_ann, :i_ann],
        levels = true,
        ignore_obc = false,
        verbose = false,
    )

    println("Running same shocks through first-order linearized model...")
    lin_model = deepcopy(model)
    lin_irf = MacroModelling.get_irf(
        lin_model;
        shocks = keyed_shocks,
        periods = opts.tail_periods,
        variables = [:log_y, :pi_ann, :i_ann],
        levels = true,
        ignore_obc = true,
        verbose = false,
    )

    T = opts.periods
    t = collect(1:T)
    rbar = Float64(model.parameter_values[find_idx(model.parameters, :R̄)])
    ss = MacroModelling.get_steady_state(model; verbose = false)
    log_y_ss = Float64(ss[find_idx(model.var, :log_y), 1])

    out_sep = 100 .* (keyed_series(obc_irf, :log_y, T) .- log_y_ss)
    out_lin = 100 .* (keyed_series(lin_irf, :log_y, T) .- log_y_ss)
    pi_sep = 100 .* keyed_series(obc_irf, :pi_ann, T)
    pi_lin = 100 .* keyed_series(lin_irf, :pi_ann, T)
    i_sep = 100 .* keyed_series(obc_irf, :i_ann, T)
    i_lin = 100 .* keyed_series(lin_irf, :i_ann, T)
    floor_ann_pct = 400 * log(rbar)
    binding = i_sep .<= floor_ann_pct + opts.floor_tolerance_pct
    linear_violates_elb = i_lin .< floor_ann_pct
    forced_window = opts.elb_period:min(opts.periods, opts.elb_period + opts.elb_span - 1)

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

    p_output = plot(t, out_sep;
                    label = "OBC SEP",
                    color = :steelblue,
                    ylabel = "% log deviation",
                    title = "Output",
                    xlabel = "Quarters")
    plot!(p_output, t, out_lin; label = "Linearized", color = :crimson, linestyle = :dash)
    hline!(p_output, [0.0]; label = "", color = :black, linestyle = :dot)

    p_inflation = plot(t, pi_sep;
                       label = "OBC SEP",
                       color = :steelblue,
                       ylabel = "% annualized",
                       title = "Inflation",
                       xlabel = "Quarters")
    plot!(p_inflation, t, pi_lin; label = "Linearized", color = :crimson, linestyle = :dash)
    hline!(p_inflation, [0.0]; label = "", color = :black, linestyle = :dot)

    p_policy = plot(t, i_sep;
                    label = "OBC SEP",
                    color = :steelblue,
                    ylabel = "% annualized",
                    title = "Policy Rate",
                    xlabel = "Quarters")
    plot!(p_policy, t, i_lin; label = "Linearized", color = :crimson, linestyle = :dash)
    hline!(p_policy, [floor_ann_pct]; label = "ELB", color = :black, linestyle = :dot)
    scatter!(p_policy, t[binding], i_sep[binding]; label = "ELB binds", color = :black, markersize = 4)

    fig = plot(
        p_output,
        p_inflation,
        p_policy;
        layout = (3, 1),
        size = (950, 930),
        plot_title = "Galí OBC Adverse eps_z Stress Path vs Same-Shock Linearized Model",
        margin = 6Plots.mm,
    )

    shock_label = replace(string(opts.elb_shock), "." => "p", "-" => "m")
    bg_label = replace(string(opts.shock_scale), "." => "p", "-" => "m")
    stem = joinpath(opts.out_dir, "gali_obc_$(opts.elb_shock_name)_same_shocks_actualfloor_span$(opts.elb_span)_shock$(shock_label)_bg$(bg_label)")
    png_path = stem * ".png"
    pdf_path = stem * ".pdf"
    data_path = stem * "_payload.jls"
    summary_path = stem * "_summary.md"
    savefig(fig, png_path)
    savefig(fig, pdf_path)
    serialize(data_path, Dict(
        "options" => opts,
        "shocks" => shocks,
        "obc_irf" => obc_irf,
        "linear_irf" => lin_irf,
        "binding" => binding,
        "linear_violates_elb" => linear_violates_elb,
    ))

    open(summary_path, "w") do io
        println(io, "# Galí OBC Stochastic Same-Shock Comparison")
        println(io)
        println(io, "- Periods: $(T)")
        println(io, "- Seed: $(opts.seed)")
        println(io, "- Random structural shock scale: $(opts.shock_scale)")
        println(io, "- Forced ELB shock block: `$(opts.elb_shock_name)[$(opts.elb_period):$(min(opts.periods, opts.elb_period + opts.elb_span - 1))]`, first shock `$(opts.elb_shock)`, decay `$(opts.elb_decay)`")
        println(io, "- OBC engine: first-order simulator with `ignore_obc=false`")
        println(io, "- Linear engine: first-order simulator with `ignore_obc=true`")
        println(io, "- Actual ELB periods by policy-rate floor criterion: $(sum(binding)) / $(T)")
        println(io, "- Floor tolerance for marker: $(opts.floor_tolerance_pct) annualized percentage points")
        println(io, "- Linearized sub-ELB periods: $(sum(linear_violates_elb)) / $(T)")
        println(io, "- Minimum OBC policy rate, annualized percent: $(@sprintf("%.3f", minimum(i_sep)))")
        println(io, "- Minimum linearized policy rate, annualized percent: $(@sprintf("%.3f", minimum(i_lin)))")
        println(io, "- Forced-window OBC output range, percent log deviation: [$(@sprintf("%.3f", minimum(out_sep[forced_window]))), $(@sprintf("%.3f", maximum(out_sep[forced_window])))]")
        println(io, "- Forced-window OBC inflation range, annualized percent: [$(@sprintf("%.3f", minimum(pi_sep[forced_window]))), $(@sprintf("%.3f", maximum(pi_sep[forced_window])))]")
        println(io, "- Forced-window OBC policy range, annualized percent: [$(@sprintf("%.3f", minimum(i_sep[forced_window]))), $(@sprintf("%.3f", maximum(i_sep[forced_window])))]")
        println(io, "- Figure PNG: `$png_path`")
        println(io, "- Figure PDF: `$pdf_path`")
        println(io, "- Payload: `$data_path`")
    end

    println("Saved PNG: $png_path")
    println("Saved PDF: $pdf_path")
    println("Saved summary: $summary_path")
    println("Actual ELB periods: $(sum(binding)) / $(T)")
    println("Linearized sub-ELB periods: $(sum(linear_violates_elb)) / $(T)")
    println("Min OBC policy annualized %: $(round(minimum(i_sep), digits = 3))")
    println("Min linearized policy annualized %: $(round(minimum(i_lin), digits = 3))")
    return (png = png_path, pdf = pdf_path, summary = summary_path, payload = data_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_stochastic_comparison(parse_stoch_args(ARGS))
end
