#!/usr/bin/env julia

# Compare Galí OBC IRFs across the solution methods that are short-run feasible
# in the current repo.  The default shock is an adverse natural-rate/preference
# shock, not a policy-rate shock: eps_z > 0 lowers the shadow policy rate and
# pushes output and inflation down on impact.

ENV["GKSwstype"] = "100"

using AxisKeys
using DelimitedFiles
using MacroModelling
using Plots
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "Gali_2015_chapter_3_obc.jl"))

Base.@kwdef struct Options
    shock::Symbol = :eps_z
    shock_size::Float64 = 3.0
    periods::Int = 20
    sep_horizon::Int = 12
    sep_maxit::Int = 250
    sep_accept_tol::Float64 = 1e-2
    include_third::Bool = false
    out_dir::String = joinpath(REPO_ROOT, "docs", "SurrogateNN_paper", "figures")
    artifact_dir::String = joinpath(REPO_ROOT, ".local_artifacts", "gali_irf_solution_comparison")
end

function parse_args(args)
    opts = Options()
    vals = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected argument $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        vals[keyval[1]] = keyval[2]
    end
    return Options(
        shock = Symbol(get(vals, "shock", string(opts.shock))),
        shock_size = parse(Float64, get(vals, "shock-size", string(opts.shock_size))),
        periods = parse(Int, get(vals, "periods", string(opts.periods))),
        sep_horizon = parse(Int, get(vals, "sep-horizon", string(opts.sep_horizon))),
        sep_maxit = parse(Int, get(vals, "sep-maxit", string(opts.sep_maxit))),
        sep_accept_tol = parse(Float64, get(vals, "sep-accept-tol", string(opts.sep_accept_tol))),
        include_third = lowercase(get(vals, "include-third", string(opts.include_third))) in ["1", "true", "yes"],
        out_dir = get(vals, "out-dir", opts.out_dir),
        artifact_dir = get(vals, "artifact-dir", opts.artifact_dir),
    )
end

function find_idx(names, name::Symbol)
    idx = findfirst(==(name), Symbol.(names))
    idx === nothing && error("Missing symbol $name")
    return idx
end

clean_shock_name(x) = Symbol(replace(string(x), "₍ₓ₎" => ""))

function shock_matrix(model, opts::Options)
    shock_names = Symbol.(model.exo)
    shocks = zeros(Float64, length(shock_names), opts.periods)
    idx = findfirst(x -> clean_shock_name(x) == opts.shock, shock_names)
    idx === nothing && error("Missing shock $(opts.shock). Available shocks: $(shock_names)")
    shocks[idx, 1] = opts.shock_size
    return shocks
end

function pack_series(label, t, policy, inflation, output; status = "ok", elapsed = NaN, note = "")
    return Dict(
        "label" => label,
        "t" => t,
        "policy" => policy,
        "inflation" => inflation,
        "output" => output,
        "status" => status,
        "elapsed" => elapsed,
        "note" => note,
    )
end

function perturbation_irf(model, opts::Options, alg::Symbol, label::String, log_y_ss::Float64)
    elapsed = @elapsed irf = MacroModelling.get_irf(
        deepcopy(model);
        algorithm = alg,
        shocks = opts.shock,
        shock_size = opts.shock_size,
        negative_shock = false,
        periods = opts.periods,
        variables = [:i_ann, :pi_ann, :log_y],
        levels = true,
        ignore_obc = true,
        verbose = false,
    )
    arr = Array(irf)
    T = size(arr, 2)
    return pack_series(
        label,
        collect(1:T),
        100 .* vec(arr[1, :, 1]),
        100 .* vec(arr[2, :, 1]),
        100 .* (vec(arr[3, :, 1]) .- log_y_ss);
        elapsed = elapsed,
        note = "Perturbation solution with OBC ignored; shows the unconstrained local response.",
    )
end

function sep_irf(model, opts::Options, order::Int, label::String, shocks::Matrix{Float64}, log_y_ss::Float64)
    local_model = deepcopy(model)
    elapsed = @elapsed res = MacroModelling.simulate_sep_extended_path(
        local_model;
        periods = opts.periods,
        shocks = shocks,
        burn_in = 0,
        sep_horizon = opts.sep_horizon,
        sep_order = order,
        sep_nnodes = order == 0 ? 1 : 3,
        sep_sparse_tree = true,
        sep_maxit = opts.sep_maxit,
        sep_tol = 1e-7,
        sep_accept_tol = opts.sep_accept_tol,
        shock_scaling = :none,
        silent = true,
    )
    sim = Float64.(Array(res.simulation))
    T = min(opts.periods, size(sim, 2) - 1)
    sim_names = Symbol.(axiskeys(res.simulation, 1))
    ix_i = find_idx(sim_names, :i_ann)
    ix_pi = find_idx(sim_names, :pi_ann)
    ix_y = find_idx(sim_names, :log_y)
    return pack_series(
        label,
        collect(1:T),
        100 .* vec(sim[ix_i, 2:(T + 1)]),
        100 .* vec(sim[ix_pi, 2:(T + 1)]),
        100 .* (vec(sim[ix_y, 2:(T + 1)]) .- log_y_ss);
        elapsed = elapsed,
        note = "simulate_sep_extended_path, sep_order=$order, errorflag=$(hasproperty(res, :errorflag) ? res.errorflag : missing)",
    )
end

function try_method(f, label)
    try
        return f()
    catch err
        return pack_series(label, Int[], Float64[], Float64[], Float64[];
                           status = "failed", note = sprint(showerror, err))
    end
end

function write_csv(path, series)
    header = ["method" "t" "policy_rate_ann_pct" "inflation_ann_pct" "output_gap_pct"]
    rows = Vector{Vector{Any}}()
    for s in series
        s["status"] == "ok" || continue
        for i in eachindex(s["t"])
            push!(rows, Any[s["label"], s["t"][i], s["policy"][i], s["inflation"][i], s["output"][i]])
        end
    end
    body = isempty(rows) ? Matrix{Any}(undef, 0, 5) : reduce(vcat, permutedims.(rows))
    writedlm(path, vcat(header, body), ',')
end

function plot_comparison(opts::Options)
    mkpath(opts.out_dir)
    mkpath(opts.artifact_dir)

    model = Gali_2015_chapter_3_obc
    ss = MacroModelling.get_steady_state(model; derivatives = false, verbose = false)
    log_y_ss = Float64(ss(:log_y))
    rbar_idx = findfirst(==(:R̄), Symbol.(model.parameters))
    rbar = rbar_idx === nothing ? 1.0 : Float64(model.parameter_values[rbar_idx])
    floor_ann_pct = 400 * log(rbar)
    shocks = shock_matrix(model, opts)

    methods = [
        (:first_order, "first order"),
        (:second_order, "second order"),
    ]
    opts.include_third && push!(methods, (:pruned_third_order, "pruned third order"))

    series = Dict{String,Any}[]
    for (alg, label) in methods
        println("Running $label...")
        push!(series, try_method(() -> perturbation_irf(model, opts, alg, label, log_y_ss), label))
    end

    println("Running perfect foresight SEP...")
    push!(series, try_method(() -> sep_irf(model, opts, 0, "perfect foresight", shocks, log_y_ss), "perfect foresight"))
    println("Running stochastic SEP...")
    push!(series, try_method(() -> sep_irf(model, opts, 1, "stochastic SEP", shocks, log_y_ss), "stochastic SEP"))

    gr()
    default(fontfamily = "Computer Modern", linewidth = 2.2, dpi = 220,
            framestyle = :box, grid = true, legendfontsize = 8,
            guidefontsize = 10, tickfontsize = 9, titlefontsize = 11)

    styles = Dict(
        "first order" => (:black, :solid),
        "second order" => (:royalblue, :dash),
        "pruned third order" => (:darkgreen, :dot),
        "perfect foresight" => (:darkorange, :dashdot),
        "stochastic SEP" => (:firebrick, :solid),
    )

    panels = [
        ("Policy-rate response", "Annualized percent", "policy"),
        ("Inflation", "Annualized percent", "inflation"),
        ("Output", "Log deviation, percent", "output"),
    ]
    plots = []
    for (title, ylabel, key) in panels
        p = plot(title = title, xlabel = "Quarters", ylabel = ylabel)
        for s in series
            s["status"] == "ok" || continue
            color, ls = styles[s["label"]]
            plot!(p, s["t"], s[key]; label = s["label"], color = color, linestyle = ls)
        end
        hline!(p, [0.0]; label = "", color = :gray, linestyle = :dot, linewidth = 1.0)
        key == "policy" && hline!(p, [floor_ann_pct]; label = "ELB", color = :red, linestyle = :dash, linewidth = 1.2)
        push!(plots, p)
    end

    shock_label = "$(opts.shock) = $(@sprintf("%.2f", opts.shock_size))"
    fig = plot(plots...; layout = (1, 3), size = (1260, 390),
               plot_title = "Galí OBC solution-method IRFs: adverse natural-rate shock ($shock_label)",
               bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)

    stem = "gali_solution_method_irf_comparison"
    png_path = joinpath(opts.out_dir, stem * ".png")
    pdf_path = joinpath(opts.out_dir, stem * ".pdf")
    csv_path = joinpath(opts.artifact_dir, stem * ".csv")
    summary_path = joinpath(opts.artifact_dir, "SUMMARY.md")

    savefig(fig, png_path)
    savefig(fig, pdf_path)
    write_csv(csv_path, series)

    open(summary_path, "w") do io
        println(io, "# Galí Solution-Method IRF Comparison")
        println(io)
        println(io, "- Model: `models/Gali_2015_chapter_3_obc.jl`")
        println(io, "- Shock: `$(opts.shock) = $(opts.shock_size)` in period 1")
        println(io, "- Interpretation: adverse natural-rate/preference shock, not a monetary policy shock.")
        println(io, "- Periods: $(opts.periods)")
        println(io, "- SEP horizon: $(opts.sep_horizon)")
        println(io, "- SEP max iterations: $(opts.sep_maxit)")
        println(io, "- Figure PNG: `$png_path`")
        println(io, "- Figure PDF: `$pdf_path`")
        println(io, "- Raw CSV: `$csv_path`")
        println(io)
        println(io, "| Method | Status | Runtime seconds | Note |")
        println(io, "|---|---:|---:|---|")
        for s in series
            elapsed = isnan(s["elapsed"]) ? "" : @sprintf("%.2f", s["elapsed"])
            note = replace(String(s["note"]), "\n" => " ")
            println(io, "| $(s["label"]) | $(s["status"]) | $elapsed | $note |")
        end
    end

    println("Saved figure: $png_path")
    println("Saved figure: $pdf_path")
    println("Saved CSV: $csv_path")
    println("Saved summary: $summary_path")
    for s in series
        println(rpad(s["label"], 22), " status=", s["status"], " elapsed=", s["elapsed"])
    end
    return (png = png_path, pdf = pdf_path, csv = csv_path, summary = summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_comparison(parse_args(ARGS))
end
