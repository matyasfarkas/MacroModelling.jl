#!/usr/bin/env julia

# Compare HLT Smets-Wouters OBC IRFs across short-run feasible solution methods
# at the pooled surrogate-HMC posterior mean.  The plotted variables are the
# policy rate (robs), inflation (pinfobs), and an output response.  For
# presentation IRFs the default output measure is the level of output, converted
# to percent deviations from the baseline path.  The likelihood observable
# `dy` is still available with `--output-var=dy`.

ENV["GKSwstype"] = "100"

using AxisKeys
using Dates
using DelimitedFiles
using MacroModelling
using Plots
using Printf
using Serialization
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "Smets_Wouters_2007_HLT_obc.jl"))

Base.@kwdef struct Options
    # HLT shock naming follows the model equations.  The monetary-policy
    # innovation is `em`: ms[0] = ... + z_em / 100 * em[x], and log(ms[0])
    # enters the Taylor rule.  A positive `em` shock is contractionary.
    shock::Symbol = :em
    shock_size::Float64 = 1.0
    # The HLT shock processes already multiply raw innovations by their
    # standard-deviation parameters, e.g. z_em / 100 * em[x].  Therefore
    # `shock_size=1, shock_scaling=:none` is a one-standard-normal innovation.
    shock_scaling::Symbol = :none
    output_var::Symbol = :y
    periods::Int = 12
    sep_horizon::Int = 6
    sep_maxit::Int = 220
    sep_accept_tol::Float64 = 1e-3
    # The high-Kimball HLT posterior mean makes second-order perturbation IRFs
    # numerically fragile for large lower-bound shocks, so keep them opt-in.
    include_second::Bool = false
    include_stochastic::Bool = true
    show_elb::Bool = false
    posterior_path::String = joinpath(REPO_ROOT, ".local_artifacts", "hlt_18param_realdata",
                                      "hlt_surrogate_hmc_extended_18p_pooled_8000_20260609.jls")
    out_dir::String = joinpath(REPO_ROOT, "docs", "SurrogateNN_paper", "figures")
    artifact_dir::String = joinpath(REPO_ROOT, ".local_artifacts", "hlt_irf_solution_comparison")
end

function shock_description(shock::Symbol)
    descriptions = Dict(
        :ea => "technology",
        :eb => "risk-premium",
        :eg => "exogenous-spending",
        :eqs => "investment-specific",
        :em => "monetary-policy",
        :epinf => "price-markup",
        :ew => "wage-markup",
    )
    return get(descriptions, shock, string(shock))
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
    boolval(x) = lowercase(x) in ["1", "true", "yes"]
    return Options(
        shock = Symbol(get(vals, "shock", string(opts.shock))),
        shock_size = parse(Float64, get(vals, "shock-size", string(opts.shock_size))),
        shock_scaling = Symbol(get(vals, "shock-scaling", string(opts.shock_scaling))),
        output_var = Symbol(get(vals, "output-var", string(opts.output_var))),
        periods = parse(Int, get(vals, "periods", string(opts.periods))),
        sep_horizon = parse(Int, get(vals, "sep-horizon", string(opts.sep_horizon))),
        sep_maxit = parse(Int, get(vals, "sep-maxit", string(opts.sep_maxit))),
        sep_accept_tol = parse(Float64, get(vals, "sep-accept-tol", string(opts.sep_accept_tol))),
        include_second = boolval(get(vals, "include-second", string(opts.include_second))),
        include_stochastic = boolval(get(vals, "include-stochastic", string(opts.include_stochastic))),
        show_elb = boolval(get(vals, "show-elb", string(opts.show_elb))),
        posterior_path = get(vals, "posterior-path", opts.posterior_path),
        out_dir = get(vals, "out-dir", opts.out_dir),
        artifact_dir = get(vals, "artifact-dir", opts.artifact_dir),
    )
end

function find_idx(names, name::Symbol)
    idx = findfirst(==(name), Symbol.(names))
    idx === nothing && error("Missing symbol $name")
    return idx
end

function posterior_mean(path::String)
    payload = deserialize(path)
    names = Symbol.(payload["theta_names"])
    chain = Matrix{Float64}(payload["chain"])
    means = vec(mean(chain, dims = 1))
    return names, means, payload
end

function apply_posterior_mean!(model, path::String)
    names, means, payload = posterior_mean(path)
    params = copy(Float64.(model.parameter_values))
    model_param_names = Symbol.(model.parameters)
    for (name, val) in zip(names, means)
        idx = findfirst(==(name), model_param_names)
        idx === nothing && error("Posterior parameter $name not found in model.")
        params[idx] = val
    end
    MacroModelling.write_parameters_input!(model, params, verbose = false)
    return Dict(name => means[i] for (i, name) in enumerate(names)), payload
end

function scaled_shock_value(model, shock::Symbol, size::Float64, scaling::Symbol)
    scaling == :none && return size
    scaling == :parameter && return MacroModelling.sep_irf_shock_scale(
        model, shock, size; shock_scaling = :parameter, negative_shock = false)
    error("Unsupported shock scaling $scaling. Use :parameter or :none.")
end

function shock_matrix(model, opts::Options)
    shock_idx = findfirst(==(opts.shock), Symbol.(model.exo))
    shock_idx === nothing && error("Missing shock $(opts.shock). Available shocks: $(model.exo)")
    shocks = zeros(Float64, length(model.exo), opts.periods)
    shocks[shock_idx, 1] = scaled_shock_value(model, opts.shock, opts.shock_size, opts.shock_scaling)
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

function plot_vars(opts::Options)
    opts.output_var in [:y, :ygap, :dy] ||
        error("Unsupported --output-var=$(opts.output_var). Use y, ygap, or dy.")
    return [:robs, :pinfobs, opts.output_var]
end

function output_title_and_note(output_var::Symbol)
    output_var == :y && return ("Output", "Percent deviation from baseline")
    output_var == :ygap && return ("Output gap", "Percentage points")
    output_var == :dy && return ("Output growth", "Percentage points")
    error("Unsupported output variable $output_var")
end

function perturbation_irf(model, opts::Options, alg::Symbol, label::String, ss_vec::Vector{Float64})
    shock_value = scaled_shock_value(model, opts.shock, opts.shock_size, opts.shock_scaling)
    vars = plot_vars(opts)
    elapsed = @elapsed irf = MacroModelling.get_irf(
        deepcopy(model);
        algorithm = alg,
        shocks = opts.shock,
        shock_size = shock_value,
        negative_shock = false,
        periods = opts.periods,
        variables = vars,
        levels = false,
        ignore_obc = true,
        verbose = false,
    )
    arr = Array(irf)
    T = size(arr, 2)
    irf_names = Symbol.(axiskeys(irf, 1))
    irf_idx = [find_idx(irf_names, v) for v in vars]
    response = arr[irf_idx, :, 1]
    output_response = if opts.output_var == :y
        y_ss = ss_vec[find_idx(model.var, :y)]
        100 .* log.(max.(y_ss .+ vec(response[3, :]), eps()) ./ y_ss)
    else
        vec(response[3, :])
    end
    return pack_series(
        label,
        collect(1:T),
        vec(response[1, :]),
        vec(response[2, :]),
        output_response;
        elapsed = elapsed,
        note = "Perturbation solution with OBC ignored; get_irf returns deviations from the relevant steady state.",
    )
end

function sep_path(model, opts::Options, shocks::Matrix{Float64}, order::Int)
    local_model = deepcopy(model)
    res = MacroModelling.simulate_sep_extended_path(
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
    return res
end

function sep_irf(model, opts::Options, order::Int, label::String, shocks::Matrix{Float64})
    elapsed = @elapsed begin
        zero_res = sep_path(model, opts, zeros(size(shocks)), order)
        shock_res = sep_path(model, opts, shocks, order)
    end
    sim_zero = Float64.(Array(zero_res.simulation))
    sim_shock = Float64.(Array(shock_res.simulation))
    sim_names = Symbol.(axiskeys(shock_res.simulation, 1))
    vars = plot_vars(opts)
    var_idx = [find_idx(sim_names, v) for v in vars]
    T = min(opts.periods, size(sim_shock, 2) - 1, size(sim_zero, 2) - 1)
    response = sim_shock[var_idx, 2:(T + 1)] .- sim_zero[var_idx, 2:(T + 1)]
    output_response = if opts.output_var == :y
        100 .* (log.(max.(vec(sim_shock[var_idx[3], 2:(T + 1)]), eps())) .-
                log.(max.(vec(sim_zero[var_idx[3], 2:(T + 1)]), eps())))
    else
        vec(response[3, :])
    end
    errorflag = (hasproperty(zero_res, :errorflag) ? zero_res.errorflag : missing,
                 hasproperty(shock_res, :errorflag) ? shock_res.errorflag : missing)
    return pack_series(
        label,
        collect(1:T),
        vec(response[1, :]),
        vec(response[2, :]),
        output_response;
        elapsed = elapsed,
        note = "simulate_sep_extended_path, sep_order=$order, zero/shock errorflag=$errorflag",
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
    header = ["method" "t" "policy_rate_pp" "inflation_pp" "output_response_pp"]
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

    model = Smets_Wouters_2007_HLT_obc
    post_means, posterior_payload = apply_posterior_mean!(model, opts.posterior_path)
    ss = MacroModelling.get_steady_state(model; derivatives = false, verbose = false)
    ss_vec = [Float64(ss(v)) for v in model.var]
    robs_ss = Float64(ss(:robs))
    rbar_idx = findfirst(==(:R_bar), Symbol.(model.parameters))
    rbar_log = rbar_idx === nothing ? 0.0 : Float64(model.parameter_values[rbar_idx])
    floor_response = 100 * (exp(rbar_log) - 1) - robs_ss
    shocks = shock_matrix(model, opts)

    series = Dict{String,Any}[]
    println("Running first order...")
    push!(series, try_method(() -> perturbation_irf(model, opts, :first_order, "first order", ss_vec), "first order"))

    if opts.include_second
        println("Running pruned second order...")
        push!(series, try_method(() -> perturbation_irf(model, opts, :pruned_second_order, "pruned second order", ss_vec), "pruned second order"))
    end

    println("Running perfect foresight SEP...")
    push!(series, try_method(() -> sep_irf(model, opts, 0, "perfect foresight", shocks), "perfect foresight"))

    if opts.include_stochastic
        println("Running stochastic SEP...")
        push!(series, try_method(() -> sep_irf(model, opts, 1, "stochastic SEP", shocks), "stochastic SEP"))
    end

    gr()
    default(fontfamily = "Computer Modern", linewidth = 2.2, dpi = 220,
            framestyle = :box, grid = true, legendfontsize = 8,
            guidefontsize = 10, tickfontsize = 9, titlefontsize = 11)

    styles = Dict(
        "first order" => (:black, :solid),
        "pruned second order" => (:royalblue, :dash),
        "perfect foresight" => (:darkorange, :dashdot),
        "stochastic SEP" => (:firebrick, :solid),
    )

    output_title, output_ylabel = output_title_and_note(opts.output_var)
    panels = [
        ("Policy-rate response", "Percentage points", "policy"),
        ("Inflation", "Percentage points", "inflation"),
        (output_title, output_ylabel, "output"),
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
        key == "policy" && opts.show_elb &&
            hline!(p, [floor_response]; label = "ELB", color = :red, linestyle = :dash, linewidth = 1.2)
        push!(plots, p)
    end

    scale_label = opts.shock_scaling == :parameter ? "parameter-scaled" : "standard-normal innovation"
    shock_label = "$(shock_description(opts.shock)) shock: $(opts.shock) = $(@sprintf("%.2f", opts.shock_size)) $scale_label"
    fig = plot(plots...; layout = (1, 3), size = (1260, 390),
               plot_title = "HLT posterior-mean solution-method IRFs ($shock_label)",
               bottom_margin = 5Plots.mm, left_margin = 5Plots.mm)

    size_tag = replace(@sprintf("%+.2f", opts.shock_size), "+" => "pos", "-" => "neg", "." => "p")
    stem = "hlt_posterior_mean_irf_solution_comparison_$(opts.shock)_$(size_tag)"
    png_path = joinpath(opts.out_dir, stem * ".png")
    pdf_path = joinpath(opts.out_dir, stem * ".pdf")
    csv_path = joinpath(opts.artifact_dir, stem * ".csv")
    summary_path = joinpath(opts.artifact_dir, "SUMMARY.md")
    param_path = joinpath(opts.artifact_dir, "posterior_mean_parameters.csv")

    savefig(fig, png_path)
    savefig(fig, pdf_path)
    write_csv(csv_path, series)

    param_rows = Vector{Vector{Any}}()
    push!(param_rows, Any["parameter", "posterior_mean"])
    for name in sort(collect(keys(post_means)); by = string)
        push!(param_rows, Any[String(name), post_means[name]])
    end
    writedlm(param_path, reduce(vcat, permutedims.(param_rows)), ',')

    open(summary_path, "w") do io
        println(io, "# HLT Posterior-Mean Solution-Method IRF Comparison")
        println(io)
        println(io, "- Model: `models/Smets_Wouters_2007_HLT_obc.jl`")
        println(io, "- Posterior artifact: `$(opts.posterior_path)`")
        println(io, "- Posterior draws: `$(get(posterior_payload, "n_samples", missing))`")
        println(io, "- Shock: `$(opts.shock) = $(opts.shock_size)` (`$(opts.shock_scaling)` scaling)")
        println(io, "- Shock interpretation: $(shock_description(opts.shock)) shock. In the HLT model, `em` is the monetary-policy innovation in `ms[0]`, which enters the Taylor rule through `log(ms[0])`; a positive `em` shock raises the policy-rate wedge.")
        println(io, "- Variables: `robs`, `pinfobs`, `$(opts.output_var)`. If output variable is `y`, the chart reports percent deviations from the baseline path.")
        println(io, "- Periods: $(opts.periods)")
        println(io, "- SEP horizon: $(opts.sep_horizon)")
        println(io, "- SEP max iterations: $(opts.sep_maxit)")
        println(io, "- Policy-rate steady state: $(@sprintf("%.6f", robs_ss))")
        println(io, "- Policy-rate floor response line: $(@sprintf("%.6f", floor_response))")
        println(io, "- ELB line shown in chart: `$(opts.show_elb)`")
        println(io, "- Figure PNG: `$png_path`")
        println(io, "- Figure PDF: `$pdf_path`")
        println(io, "- Raw CSV: `$csv_path`")
        println(io, "- Posterior means CSV: `$param_path`")
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
