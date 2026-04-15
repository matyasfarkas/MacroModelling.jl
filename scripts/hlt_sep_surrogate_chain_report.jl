#!/usr/bin/env julia
if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"
end
ENV["PLOTS_SHOW"] = "false"

using Serialization
using Statistics
using LinearAlgebra
using Printf
using Dates
using MCMCChains
using StatsBase
using StatsPlots
using KernelDensity
using Distributions
using JSON
using MacroModelling
using Turing

function parameter_pages_include_path()
    candidates = [
        joinpath(@__DIR__, "..", "SurrogateNN", "diagnostics", "parameter_pages.jl"),
        joinpath(@__DIR__, "..", "archive", "development", "SurrogateNN", "diagnostics", "parameter_pages.jl"),
        joinpath(@__DIR__, "..", "archive", "development", "SurrogateNN", "parameter_pages.jl"),
    ]
    for path in candidates
        if isfile(path)
            return path
        end
    end
    error("parameter_pages.jl not found. Checked: $(join(candidates, ", "))")
end

include(parameter_pages_include_path())
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))

const PP = ParameterPages

StatsPlots.default(show = false)

function escape_tex(s::AbstractString)
    replace(s,
        "\\" => "\\textbackslash{}",
        "_" => "\\_",
        "%" => "\\%",
        "&" => "\\&",
        "#" => "\\#",
        "{" => "\\{",
        "}" => "\\}",
        "\$" => "\\\$",
        "^" => "\\textasciicircum{}",
        "~" => "\\textasciitilde{}",
    )
end

function param_filename(sym::Symbol)
    name = string(sym)
    name = replace(name, "ε" => "eps", "ϵ" => "eps")
    name = replace(name, r"[^A-Za-z0-9]+" => "_")
    name = replace(name, r"_+" => "_")
    name = strip(name, '_')
    return name == "" ? "param" : name
end

function param_label_tex(sym::Symbol)
    name = string(sym)
    name = replace(name, "ϵ" => "ε")
    m = match(r"ε\[(\d+),\s*(\d+)\]", name)
    if m !== nothing
        return "\$\\varepsilon_{$(m.captures[1]),$(m.captures[2])}\$"
    end
    if occursin("ε", name)
        return "\$\\varepsilon\$"
    end
    return "\\texttt{" * escape_tex(name) * "}"
end

function wrap_tex_list(items::Vector{String}; per_line::Int=12)
    if isempty(items)
        return ""
    end
    chunks = [items[i:min(i + per_line - 1, end)] for i in 1:per_line:length(items)]
    return join((join(chunk, ", ") for chunk in chunks), ", \\\\ ")
end

function is_eps_param(sym::Symbol)
    name = string(sym)
    if occursin(r"(ε|ϵ)\[", name)
        return true
    end
    lower = lowercase(name)
    return occursin("epsilon[", lower) || occursin("eps[", lower)
end

function summarize_param(values::AbstractVector{Float64})
    n = length(values)
    runmean = cumsum(values) ./ (1:n)
    acfvals = StatsBase.autocor(values)
    return Dict(
        :mean => Statistics.mean(values),
        :std => Statistics.std(values),
        :median => Statistics.median(values),
        :mode => try
            kd = kde(values)
            kd.x[findmax(kd.density)[2]]
        catch
            Statistics.median(values)
        end,
        :q025 => quantile(values, 0.025),
        :q975 => quantile(values, 0.975),
        :acf1 => length(acfvals) >= 2 ? acfvals[2] : NaN,
        :acf5 => length(acfvals) >= 6 ? acfvals[6] : NaN,
        :acf10 => length(acfvals) >= 11 ? acfvals[11] : NaN,
        :runmean_last => runmean[end]
    )
end

function summarize_chain_local(ch::Chains)
    params = names(ch, :parameters)
    rhat = Dict{Symbol,Any}()
    ess = Dict{Symbol,Any}()

    for p in params
        rhat[p] = try
            gelman_rubin(ch[:, p])
        catch
            missing
        end
        ess[p] = try
            ess_per_variable(ch[:, p])
        catch
            missing
        end
    end

    info = get(ch.info, :internals, Dict{Symbol,Any}())
    return Dict(
        :rhat => rhat,
        :ess => ess,
        :n_divergent => get(info, :count_divergences, missing),
        :step_size => get(info, :step_size, missing),
        :avg_accept => get(info, :avg_acceptance_rate, missing),
        :max_tree_depth => get(info, :max_depth, missing),
        :n_samples => length(ch)
    )
end

function prior_map_from_payload(payload::Dict, param_syms::Vector{Symbol})
    priors = Dict{Symbol,Distribution}()
    config = get(payload, "prior_config", Dict{String,Any}())
    cprobp_mu = get(config, "cprobp_mean", 0.5)
    cprobp_sd = get(config, "cprobp_sd", 0.10)
    cindp_mu = get(config, "cindp_mean", 0.5)
    cindp_sd = get(config, "cindp_sd", 0.15)
    curvp_mu = get(config, "curvp_mean", 75.0)
    curvp_sd = get(config, "curvp_sd", 25.0)
    for s in param_syms
        if s == :cprobp
            priors[s] = MacroModelling.Beta(cprobp_mu, cprobp_sd, 0.5, 0.95, μσ = true)
        elseif s == :cindp
            priors[s] = MacroModelling.Beta(cindp_mu, cindp_sd, 0.01, 0.99, μσ = true)
        elseif s == :curvp
            priors[s] = Distributions.Normal(curvp_mu, curvp_sd)
        end
    end
    return priors
end

function load_truth_map(payload::Dict)
    truth = Dict{Symbol,Float64}()
    synthetic_path = get(payload, "synthetic_path", "")
    if synthetic_path != "" && isfile(synthetic_path)
        syn = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
        theta_names = get(syn, "theta_names", Symbol[])
        theta_true = get(syn, "theta_true", nothing)
        if theta_true !== nothing && !isempty(theta_names)
            for (i, name) in enumerate(theta_names)
                truth[Symbol(name)] = theta_true[i]
            end
        end
    elseif haskey(payload, "theta_true")
        theta_true = payload["theta_true"]
        if length(theta_true) == 3
            truth[:cprobp] = theta_true[1]
            truth[:cindp] = theta_true[2]
            truth[:curvp] = theta_true[3]
        end
    end
    return truth
end

function info_stat(info::Dict, key::Symbol)
    val = get(info, key, missing)
    if val isa AbstractArray
        return Statistics.mean(skipmissing(vec(val)))
    end
    return val
end

function write_latex_report(report_path::AbstractString,
                            meta::Dict,
                            param_rows::AbstractVector{<:AbstractDict},
                            diag_rows::AbstractVector{<:AbstractDict},
                            joint_plot::Union{Nothing,String},
                            include_eps_pages::Bool)
    title = get(meta, :title, "HLT Surrogate Posterior Report")
    date_str = Dates.format(now(), "yyyy-mm-dd HH:MM")
    out_dir = meta[:out_dir]

    io = IOBuffer()
    println(io, "\\documentclass[11pt]{article}")
    println(io, "\\usepackage[margin=1in]{geometry}")
    println(io, "\\usepackage{amsmath,amssymb}")
    println(io, "\\usepackage{booktabs}")
    println(io, "\\usepackage{longtable}")
    println(io, "\\usepackage{graphicx}")
    println(io, "\\usepackage{hyperref}")
    println(io, "\\usepackage{float}")
    println(io, "\\usepackage{xcolor}")
    println(io, "\\title{", escape_tex(title), "}")
    println(io, "\\author{Posterior Diagnostics}")
    println(io, "\\date{", escape_tex(date_str), "}")
    println(io, "\\begin{document}")
    println(io, "\\maketitle")

    println(io, "\\section{Run Metadata}")
    println(io, "\\begin{itemize}")
    println(io, "  \\item Chain file: \\texttt{", escape_tex(meta[:chain_path]), "}")
    if haskey(meta, :chain_mtime)
        println(io, "  \\item Chain timestamp: ", escape_tex(meta[:chain_mtime]))
    end
    if haskey(meta, :synthetic_path)
        println(io, "  \\item Synthetic data: \\texttt{", escape_tex(meta[:synthetic_path]), "}")
    end
    if haskey(meta, :surrogate_path)
        println(io, "  \\item Surrogate: \\texttt{", escape_tex(meta[:surrogate_path]), "}")
    end
    println(io, "  \\item Samples: ", meta[:n_samples], " \\quad Chains: ", meta[:n_chains])
    println(io, "  \\item Report is regenerated from the chain at run time; re-run the script to refresh.")
    if haskey(meta, :param_labels_tex)
        println(io, "  \\item Parameters (diagnostic pages): ", meta[:param_labels_tex])
    else
        println(io, "  \\item Parameters (diagnostic pages): ", escape_tex(join(meta[:param_labels], ", ")))
    end
    if !include_eps_pages
        println(io, "  \\item Epsilon diagnostic pages excluded (use \\texttt{--include-eps-pages=true} to include).")
    end
    println(io, "\\end{itemize}")

    if haskey(meta, :regime_section)
        println(io, "\\section{Regime Switching}")
        for line in meta[:regime_section]
            println(io, line)
        end
    end

    println(io, "\\section{Sampling Diagnostics}")
    println(io, "\\begin{itemize}")
    println(io, "  \\item Average acceptance rate: ", meta[:acceptance])
    println(io, "  \\item Step size: ", meta[:step_size])
    println(io, "  \\item Divergent transitions: ", meta[:n_divergent])
    println(io, "  \\item Max tree depth: ", meta[:max_depth])
    println(io, "\\end{itemize}")

    println(io, "\\paragraph{Interpretation} High acceptance rates (greater than 0.95) may indicate very small steps and slow exploration; low rates (below 0.6) may indicate unstable trajectories. Any divergent transitions suggest misspecified geometry or overly tight tolerances and can bias estimates.")

    if haskey(meta, :fit_metrics)
        println(io, "\\section{Fit Metrics}")
        println(io, "\\begin{itemize}")
        for line in meta[:fit_metrics]
            println(io, line)
        end
        println(io, "\\end{itemize}")
        println(io, "\\paragraph{Interpretation} The log-likelihood at the posterior mean provides a direct fit metric. The Laplace approximation (theta-only) is a rough evidence proxy; larger values indicate better fit, but comparisons are meaningful only when computed with the same likelihood definition and shock treatment.")
    end

    println(io, "\\section{Parameter Summary}")
    println(io, "\\small")
    println(io, "\\setlength{\\tabcolsep}{4pt}")
    println(io, "\\renewcommand{\\arraystretch}{1.1}")
    println(io, "\\begin{longtable}{lrrrrrrrr}")
    println(io, "\\caption{Posterior Summary with Diagnostics} \\\\")
    println(io, "\\toprule")
    println(io, "Param & Mean & SD & 95\\% CI & ESS & R-hat & MCSE & ACF(1) & Prior shift \\\\")
    println(io, "\\midrule")
    println(io, "\\endfirsthead")
    println(io, "\\toprule")
    println(io, "Param & Mean & SD & 95\\% CI & ESS & R-hat & MCSE & ACF(1) & Prior shift \\\\")
    println(io, "\\midrule")
    println(io, "\\endhead")
    println(io, "\\midrule")
    println(io, "\\multicolumn{9}{r}{Continued on next page} \\\\")
    println(io, "\\endfoot")
    println(io, "\\bottomrule")
    println(io, "\\endlastfoot")
    for row in param_rows
        println(io,
            row[:param_tex], " & ",
            @sprintf("%.4f", row[:mean]), " & ",
            @sprintf("%.4f", row[:std]), " & ",
            @sprintf("[%.4f, %.4f]", row[:q025], row[:q975]), " & ",
            @sprintf("%.1f", row[:ess]), " & ",
            row[:rhat], " & ",
            row[:mcse], " & ",
            row[:acf1], " & ",
            row[:prior_shift], " \\\\")
    end
    println(io, "\\end{longtable}")
    println(io, "\\normalsize")

    println(io, "\\paragraph{Interpretation}")
    println(io, "R-hat values near 1.00 indicate convergence; values above 1.01 suggest non-convergence. Low ESS indicates high autocorrelation and unreliable uncertainty estimates. High ACF at lag 1 indicates slow mixing. A small prior shift (posterior mean close to prior mean in SD units) suggests weak identification by the likelihood.")

    println(io, "\\section{Parameter Diagnostics}")
    println(io, "Each page combines trace, density (with prior overlay), autocorrelation, and running mean.")
    rows_for_pages = include_eps_pages ? param_rows : [row for row in param_rows if !row[:is_eps]]
    if isempty(rows_for_pages)
        println(io, "No parameter diagnostic pages requested.")
    else
        for row in rows_for_pages
            page = row[:page]
            println(io, "\\begin{figure}[H]")
            println(io, "\\centering")
            println(io, "\\includegraphics[width=0.95\\textwidth]{", escape_tex(page), "}")
            println(io, "\\caption{Diagnostics for ", row[:param_tex], ".}")
            println(io, "\\end{figure}")
        end
    end

    if joint_plot !== nothing
        println(io, "\\section{Joint Posterior}")
        println(io, "\\begin{figure}[H]")
        println(io, "\\centering")
        println(io, "\\includegraphics[width=0.8\\textwidth]{", escape_tex(joint_plot), "}")
        println(io, "\\caption{Joint posterior scatter with marginal distributions. Strong curvature or multimodality can signal weak identifiability.}")
        println(io, "\\end{figure}")
    end

    println(io, "\\section{Diagnostic Flags}")
    println(io, "\\begin{itemize}")
    for row in diag_rows
        if haskey(row, :message_tex)
            println(io, "  \\item ", row[:message_tex])
        else
            println(io, "  \\item ", escape_tex(row[:message]))
        end
    end
    println(io, "\\end{itemize}")

    println(io, "\\end{document}")

    open(report_path, "w") do f
        write(f, String(take!(io)))
    end
end

function build_regime_section(payload::Dict)
    if !haskey(payload, "gate_info") && !haskey(payload, "gate_mask") && !haskey(payload, "gate_probs")
        return ["\\noindent This chain was produced without regime switching (linear-only likelihood)."]
    end

    gate_info = get(payload, "gate_info", Dict{String,Any}())
    gate_mode = get(gate_info, "gate_mode", "unknown")
    gate_share = get(payload, "gate_share", nothing)
    gate_bias = get(gate_info, "gate_bias", "n/a")
    tau_eps = get(gate_info, "tau_eps", "n/a")
    tau_y = get(gate_info, "tau_y", "n/a")
    k_pre = get(gate_info, "k_pre", "n/a")
    k_post = get(gate_info, "k_post", "n/a")
    min_len = get(gate_info, "min_len", "n/a")
    filter = get(gate_info, "filter", "n/a")
    system_priors = get(gate_info, "system_priors", nothing)

    lines = String[]
    push!(lines, "\\paragraph{Implementation} The regime-switching likelihood combines a linear ROM and a nonlinear SEP surrogate. Period-level diagnostics are computed from a linear filter (default: Kalman). A gate is formed by comparing standardized shock size (\\texttt{e\\_stat}) and forecast error (\\texttt{f\\_stat}) against thresholds \\texttt{tau\\_eps} and \\texttt{tau\\_y}. Gate padding (pre/post/min length) ensures contiguous nonlinear episodes.")
    push!(lines, "\\paragraph{Soft vs. hard gating} Soft gating assigns a probability \\(p_t\\in(0,1)\\) to the nonlinear regime and mixes log-likelihoods as \\(\\log(p_t\\,e^{\\ell^{\\text{SEP}}_t} + (1-p_t)\\,e^{\\ell^{\\text{ROM}}_t})\\). Hard gating uses a binary mask and sums \\(\\ell^{\\text{SEP}}_t\\) inside the gate and \\(\\ell^{\\text{ROM}}_t\\) outside.")
    push!(lines, "\\begin{itemize}")
    push!(lines, "  \\item Gate mode: \\texttt{" * escape_tex(string(gate_mode)) * "}")
    if gate_share !== nothing
        push!(lines, "  \\item Gate share (mean probability or mask share): " * escape_tex(string(round(gate_share, digits=4))))
    end
    push!(lines, "  \\item Gate thresholds: tau\\_eps=" * escape_tex(string(tau_eps)) * ", tau\\_y=" * escape_tex(string(tau_y)))
    push!(lines, "  \\item Gate padding: k\\_pre=" * escape_tex(string(k_pre)) * ", k\\_post=" * escape_tex(string(k_post)) * ", min\\_len=" * escape_tex(string(min_len)))
    push!(lines, "  \\item Shock filter: \\texttt{" * escape_tex(string(filter)) * "}")
    if gate_bias != "n/a"
        push!(lines, "  \\item Soft-gate bias: " * escape_tex(string(gate_bias)))
    end
    if system_priors !== nothing
        push!(lines, "  \\item System priors: \\texttt{" * escape_tex(string(system_priors)) * "}")
    end
    push!(lines, "\\end{itemize}")
    return lines
end

function main()
    chain_path = get(ARGS, 1, "")
    chain_path == "" && error("Usage: julia hlt_sep_surrogate_chain_report.jl <chain.jls> [out_dir]")
    out_dir = get(ARGS, 2, dirname(chain_path))
    title = parse_arg_string(ARGS, "--title", "HLT Surrogate Posterior Report")
    include_priors = parse_arg_bool(ARGS, "--include-priors", true)
    include_eps_pages = parse_arg_bool(ARGS, "--include-eps-pages", false)

    payload = MacroModelling.load_hlt_chain_payload(chain_path)
    MacroModelling.validate_hlt_chain_payload(payload; require_chain = true, label = "HLT chain report payload")
    chain = payload["chain"]

    plot_dir = joinpath(out_dir, "diagnostic_plots")
    mkpath(plot_dir)

    param_syms = MCMCChains.names(chain, :parameters)
    param_syms_pages = include_eps_pages ? param_syms : filter(s -> !is_eps_param(s), param_syms)
    param_labels = String.(param_syms_pages)
    param_labels_tex = wrap_tex_list(param_label_tex.(param_syms_pages); per_line=12)
    prior_map = include_priors ? prior_map_from_payload(payload, param_syms) : Dict{Symbol,Distribution}()
    truth_map = load_truth_map(payload)

    if !isempty(param_syms_pages)
        PP.publish_param_pages(chain;
            param_syms = param_syms_pages,
            saveprefix = joinpath(plot_dir, "param_pages"),
            truth = isempty(truth_map) ? nothing : truth_map,
            prior_map = isempty(prior_map) ? nothing : prior_map
        )

        pages_dir = joinpath(plot_dir, "param_pages", "pages")
        for sym in param_syms_pages
            old_path = joinpath(pages_dir, string(sym) * ".png")
            new_path = joinpath(pages_dir, param_filename(sym) * ".png")
            if old_path != new_path && isfile(old_path)
                mv(old_path, new_path; force = true)
            end
        end
    end

    arr = Array(chain)
    if ndims(arr) == 2
        arr = reshape(arr, size(arr, 1), size(arr, 2), 1)
    end
    n_samples, n_params, n_chains = size(arr)
    rows = Dict{Symbol,Any}[]
    diag_flags = Dict{Symbol,Any}[]

    summary = summarize_chain_local(chain)
    rhat_map = summary[:rhat]
    ess_map = summary[:ess]

    for (i, p) in enumerate(param_syms)
        is_eps = is_eps_param(p)
        vals = vec(arr[:, i, :])
        summ = summarize_param(vals)
        ess = let v = get(ess_map, p, missing)
            if v isa AbstractDict
                get(v, p, first(values(v)))
            elseif v isa AbstractArray
                v[1]
            elseif v === missing
                length(vals)
            else
                v
            end
        end
        rhat = let v = get(rhat_map, p, missing)
            if v isa AbstractDict
                get(v, p, first(values(v)))
            elseif v === missing
                NaN
            else
                v
            end
        end
        mcse = isfinite(ess) && ess > 0 ? summ[:std] / sqrt(ess) : NaN

        prior_shift = "n/a"
        if haskey(prior_map, p)
            dprior = prior_map[p]
            try
                μp = Statistics.mean(dprior)
                σp = Statistics.std(dprior)
                if isfinite(σp) && σp > 0
                    prior_shift = @sprintf("\$%.2f\\sigma\$", abs(summ[:mean] - μp) / σp)
                end
            catch
                prior_shift = "n/a"
            end
        end

        page_path = ""
        if include_eps_pages || !is_eps
            page_path = joinpath("diagnostic_plots", "param_pages", "pages", param_filename(p) * ".png")
        end

        push!(rows, Dict(
            :param => p,
            :param_tex => param_label_tex(p),
            :mean => summ[:mean],
            :std => summ[:std],
            :q025 => summ[:q025],
            :q975 => summ[:q975],
            :ess => ess,
            :rhat => isfinite(rhat) ? @sprintf("%.3f", rhat) : "n/a",
            :mcse => isfinite(mcse) ? @sprintf("%.4f", mcse) : "n/a",
            :acf1 => isfinite(summ[:acf1]) ? @sprintf("%.3f", summ[:acf1]) : "n/a",
            :prior_shift => prior_shift,
            :page => page_path,
            :is_eps => is_eps
        ))

        if isfinite(rhat) && rhat > 1.01
            push!(diag_flags, Dict(:message_tex => "R-hat > 1.01 for $(param_label_tex(p)): potential non-convergence."))
        end
        if isfinite(ess) && ess < 200
            push!(diag_flags, Dict(:message_tex => "ESS < 200 for $(param_label_tex(p)): high autocorrelation or too few effective draws."))
        end
        if isfinite(summ[:acf1]) && summ[:acf1] > 0.8
            push!(diag_flags, Dict(:message_tex => "High lag-1 autocorrelation for $(param_label_tex(p)): slow mixing."))
        end
    end

    summary_txt = joinpath(plot_dir, "diagnostic_summary.txt")
    open(summary_txt, "w") do io
        println(io, "Parameter diagnostics:")
        for row in rows
            println(io, "$(row[:param]) mean=$(round(row[:mean], digits=4)) sd=$(round(row[:std], digits=4)) ",
                    "ESS=$(row[:ess]) Rhat=$(row[:rhat]) ACF1=$(row[:acf1]) prior_shift=$(row[:prior_shift])")
        end
    end

    summary_json = joinpath(plot_dir, "diagnostic_summary.json")
    open(summary_json, "w") do io
        JSON.print(io, rows)
    end

    # Joint posterior plot (first two parameters)
    joint_plot = nothing
    if n_params >= 2
        p1 = vec(arr[:, 1, :])
        p2 = vec(arr[:, 2, :])
        joint = scatter(p1, p2,
                        xlabel=param_labels[1],
                        ylabel=param_labels[2],
                        alpha=0.4,
                        markersize=2,
                        label=false,
                        title="Joint Posterior")
        joint_plot = joinpath(plot_dir, "joint_posterior.png")
        savefig(joint, joint_plot)
        joint_plot = joinpath("diagnostic_plots", "joint_posterior.png")
    end

    info = get(chain.info, :internals, Dict{Symbol,Any}())
    chain_mtime = try
        Dates.format(Dates.unix2datetime(stat(chain_path).mtime), "yyyy-mm-dd HH:MM")
    catch
        "unknown"
    end
    regime_section = build_regime_section(payload)
    fit_lines = String[]
    if haskey(payload, "loglik_post_mean")
        push!(fit_lines, "\\item Log-likelihood at posterior mean: " * escape_tex(string(round(payload["loglik_post_mean"], digits = 2))))
    end
    if haskey(payload, "loglik_post_mean_filtered") && payload["loglik_post_mean_filtered"] !== nothing
        push!(fit_lines, "\\item Log-likelihood at posterior mean (filtered shocks): " * escape_tex(string(round(payload["loglik_post_mean_filtered"], digits = 2))))
    end
    if haskey(payload, "logprior_theta_post_mean")
        push!(fit_lines, "\\item Log prior (theta) at posterior mean: " * escape_tex(string(round(payload["logprior_theta_post_mean"], digits = 2))))
    end
    if haskey(payload, "log_marginal_theta_laplace")
        push!(fit_lines, "\\item Laplace log marginal (theta-only): " * escape_tex(string(round(payload["log_marginal_theta_laplace"], digits = 2))))
    end
    if haskey(payload, "log_marginal_theta_laplace_filtered") && payload["log_marginal_theta_laplace_filtered"] !== nothing
        push!(fit_lines, "\\item Laplace log marginal (theta-only, filtered shocks): " * escape_tex(string(round(payload["log_marginal_theta_laplace_filtered"], digits = 2))))
    end
    if haskey(payload, "loglik_shocks_source")
        push!(fit_lines, "\\item Loglik shocks source: \\texttt{" * escape_tex(string(payload["loglik_shocks_source"])) * "}")
    end
    if haskey(payload, "loglik_shocks_source_filtered") && payload["loglik_shocks_source_filtered"] !== nothing
        push!(fit_lines, "\\item Loglik filtered shocks source: \\texttt{" * escape_tex(string(payload["loglik_shocks_source_filtered"])) * "}")
    end
    if haskey(payload, "linear_loglik_true_conditional") && payload["linear_loglik_true_conditional"] !== nothing
        push!(fit_lines, "\\item Linear loglik at true theta (conditional): " * escape_tex(string(round(payload["linear_loglik_true_conditional"], digits = 2))))
    end
    if haskey(payload, "linear_loglik_true_conditional_filtered") && payload["linear_loglik_true_conditional_filtered"] !== nothing
        push!(fit_lines, "\\item Linear loglik at true theta (conditional, filtered shocks): " * escape_tex(string(round(payload["linear_loglik_true_conditional_filtered"], digits = 2))))
    end
    if haskey(payload, "linear_loglik_true_conditional_filtered_source") && payload["linear_loglik_true_conditional_filtered_source"] !== nothing
        push!(fit_lines, "\\item Linear conditional filtered shocks source: \\texttt{" * escape_tex(string(payload["linear_loglik_true_conditional_filtered_source"])) * "}")
    end
    if haskey(payload, "regime_loglik_true_hard") && payload["regime_loglik_true_hard"] !== nothing
        push!(fit_lines, "\\item Regime loglik at true theta (hard gate, conditional): " * escape_tex(string(round(payload["regime_loglik_true_hard"], digits = 2))))
    end
    if haskey(payload, "regime_loglik_true_hard_filtered") && payload["regime_loglik_true_hard_filtered"] !== nothing
        push!(fit_lines, "\\item Regime loglik at true theta (hard gate, filtered shocks): " * escape_tex(string(round(payload["regime_loglik_true_hard_filtered"], digits = 2))))
    end
    if haskey(payload, "regime_loglik_true_hard_filtered_source") && payload["regime_loglik_true_hard_filtered_source"] !== nothing
        push!(fit_lines, "\\item Regime hard-gate filtered shocks source: \\texttt{" * escape_tex(string(payload["regime_loglik_true_hard_filtered_source"])) * "}")
    end
    if haskey(payload, "hard_gate_source")
        push!(fit_lines, "\\item Hard gate source: \\texttt{" * escape_tex(string(payload["hard_gate_source"])) * "}")
    end
    if haskey(payload, "hard_gate_share")
        push!(fit_lines, "\\item Hard gate share: " * escape_tex(string(round(payload["hard_gate_share"], digits = 4))))
    end
    if haskey(payload, "hard_gate_threshold")
        push!(fit_lines, "\\item Hard gate threshold: " * escape_tex(string(payload["hard_gate_threshold"])))
    end
    if haskey(payload, "prior_config")
        push!(fit_lines, "\\item Prior means: " * escape_tex(string(payload["prior_config"])))
    end
    if haskey(payload, "init_params")
        push!(fit_lines, "\\item Initial parameters: " * escape_tex(string(payload["init_params"])))
    end

    meta = Dict(
        :title => title,
        :chain_path => chain_path,
        :chain_mtime => chain_mtime,
        :synthetic_path => get(payload, "synthetic_path", ""),
        :surrogate_path => get(payload, "surrogate_path", ""),
        :n_samples => n_samples,
        :n_chains => n_chains,
        :param_labels => param_labels,
        :param_labels_tex => param_labels_tex,
        :acceptance => info_stat(info, :avg_acceptance_rate),
        :step_size => info_stat(info, :step_size),
        :n_divergent => info_stat(info, :count_divergences),
        :max_depth => info_stat(info, :max_depth),
        :out_dir => out_dir,
        :regime_section => regime_section,
        :fit_metrics => fit_lines
    )

    report_path = joinpath(out_dir, "posterior_report.tex")
    write_latex_report(report_path, meta, rows, diag_flags, joint_plot, include_eps_pages)

    println("Diagnostics written to: $plot_dir")
    println("LaTeX report: $report_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
