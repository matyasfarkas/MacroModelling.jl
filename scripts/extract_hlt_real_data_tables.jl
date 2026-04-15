#!/usr/bin/env julia
using Serialization
using Statistics
using Printf
using Dates
using TOML
using JSON
using MCMCChains
using MacroModelling
import Turing

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))

const CORE_PARAMS = [:cprobp, :cindp, :curvp]

function scalarize_stat(x)
    if x isa Number
        return Float64(x)
    elseif x isa Missing
        return NaN
    elseif x isa AbstractArray
        isempty(x) && return NaN
        return scalarize_stat(first(x))
    elseif x isa AbstractDict
        isempty(x) && return NaN
        return scalarize_stat(first(values(x)))
    elseif x isa NamedTuple
        vals = collect(values(x))
        isempty(vals) && return NaN
        return scalarize_stat(vals[1])
    end
    return NaN
end

function safe_gelman(chain, p::Symbol)
    try
        return scalarize_stat(gelman_rubin(chain[:, p]))
    catch
        return NaN
    end
end

function safe_ess(chain, p::Symbol)
    try
        return scalarize_stat(ess_per_variable(chain[:, p]))
    catch
        return NaN
    end
end

function available_params(chain)
    params = Set(MCMCChains.names(chain, :parameters))
    return [p for p in CORE_PARAMS if p in params]
end

function extract_param_rows(chain; params = available_params(chain))
    rows = Vector{Dict{String,Any}}()
    for p in params
        vals = vec(Array(chain[:, p, :]))
        μ = Statistics.mean(vals)
        σ = Statistics.std(vals)
        q025 = Statistics.quantile(vals, 0.025)
        q5 = Statistics.quantile(vals, 0.05)
        q95 = Statistics.quantile(vals, 0.95)
        q975 = Statistics.quantile(vals, 0.975)
        rhat = safe_gelman(chain, p)
        ess = safe_ess(chain, p)
        mcse = isfinite(ess) && ess > 0 ? σ / sqrt(ess) : NaN

        push!(rows, Dict(
            "parameter" => String(p),
            "mean" => μ,
            "std" => σ,
            "q025" => q025,
            "q5" => q5,
            "q95" => q95,
            "q975" => q975,
            "rhat" => rhat,
            "ess" => ess,
            "mcse" => mcse,
        ))
    end
    return rows
end

function extract_chain_meta(chain)
    n_samples_per_chain = size(chain, 1)
    n_chains = length(MCMCChains.chains(chain))
    n_parameters = length(MCMCChains.names(chain, :parameters))
    info = get(chain.info, :internals, Dict{Symbol,Any}())
    acceptance = scalarize_stat(get(info, :avg_acceptance_rate, NaN))
    step_size = scalarize_stat(get(info, :step_size, NaN))
    divergences = scalarize_stat(get(info, :count_divergences, NaN))
    max_depth = scalarize_stat(get(info, :max_depth, NaN))

    return Dict(
        "n_samples_per_chain" => n_samples_per_chain,
        "n_parameters" => n_parameters,
        "n_chains" => n_chains,
        "avg_acceptance_rate" => acceptance,
        "step_size" => step_size,
        "divergences" => divergences,
        "max_tree_depth" => max_depth,
    )
end

function extract_gate_stats(payload::Dict{Any,Any})
    gate_share = get(payload, "gate_share", NaN)
    gate_mask = haskey(payload, "gate_mask") ? Bool.(payload["gate_mask"]) : Bool[]
    gate_info = haskey(payload, "gate_info") ? Dict{Any,Any}(payload["gate_info"]) : Dict{Any,Any}()
    hard_gate_share = get(payload, "hard_gate_share", NaN)
    hard_gate_threshold = get(payload, "hard_gate_threshold", NaN)

    return Dict(
        "gate_mode" => string(get(gate_info, "gate_mode", "unknown")),
        "gate_share" => gate_share,
        "gate_periods" => length(gate_mask),
        "gate_nonlinear_periods" => isempty(gate_mask) ? missing : sum(gate_mask),
        "gate_linear_periods" => isempty(gate_mask) ? missing : (length(gate_mask) - sum(gate_mask)),
        "gate_k_pre" => get(gate_info, "k_pre", missing),
        "gate_k_post" => get(gate_info, "k_post", missing),
        "gate_min_len" => get(gate_info, "min_len", missing),
        "gate_filter" => get(gate_info, "filter", missing),
        "tau_eps" => get(gate_info, "tau_eps", missing),
        "tau_y" => get(gate_info, "tau_y", missing),
        "hard_gate_share" => hard_gate_share,
        "hard_gate_threshold" => hard_gate_threshold,
    )
end

function extract_manifest_meta(run_manifest_path::AbstractString)
    if run_manifest_path == "" || !isfile(run_manifest_path)
        return Dict{String,Any}()
    end
    m = TOML.parsefile(run_manifest_path)
    out = Dict{String,Any}(
        "run_name" => get(m, "name", missing),
        "run_mode" => get(m, "mode", missing),
        "run_created_at" => get(m, "created_at", missing),
        "run_git_commit" => get(m, "git_commit", missing),
        "run_dir" => get(m, "run_dir", missing),
        "requested_samples" => get(m, "samples", missing),
        "requested_chains" => get(m, "chains", missing),
    )

    steps = haskey(m, "steps") ? m["steps"] : Dict{String,Any}()
    if haskey(steps, "switching_estimation")
        s = steps["switching_estimation"]
        out["switching_elapsed_s"] = get(s, "elapsed_s", missing)
    end
    if haskey(steps, "gate_calibration")
        s = steps["gate_calibration"]
        out["gate_calibration_elapsed_s"] = get(s, "elapsed_s", missing)
    end
    if haskey(steps, "surrogate_train")
        s = steps["surrogate_train"]
        out["surrogate_train_elapsed_s"] = get(s, "elapsed_s", missing)
    end

    return out
end

function write_tex_posterior(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Updated US real-data application: posterior summaries for HLT parameters.}")
        println(io, "\\label{tab:realdata_posterior}")
        println(io, "\\begin{tabular}{lrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & Mean & Std. Dev. & 95\\% CI Lower & 95\\% CI Upper \\\\")
        println(io, "\\midrule")
        for row in rows
            println(io, @sprintf("%s & %.6f & %.6f & %.6f & %.6f \\\\", row["parameter"], row["mean"], row["std"], row["q025"], row["q975"]))
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
end

function write_tex_mcmc(path::AbstractString, rows, meta)
    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{MCMC diagnostics for updated US real-data HLT estimation.}")
        println(io, "\\label{tab:realdata_mcmc}")
        println(io, "\\begin{tabular}{lrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & \$\\hat{R}\$ & ESS & MCSE & 90\\% CI Width \\\\")
        println(io, "\\midrule")
        for row in rows
            ciw = row["q95"] - row["q5"]
            println(io, @sprintf("%s & %.4f & %.1f & %.6f & %.6f \\\\", row["parameter"], row["rhat"], row["ess"], row["mcse"], ciw))
        end
        println(io, "\\midrule")
        println(io, @sprintf("\\multicolumn{5}{l}{Samples/chain: %d; Chains: %d; Avg. acceptance: %.4f; Divergences: %.0f} \\\\",
            meta["n_samples_per_chain"], meta["n_chains"], meta["avg_acceptance_rate"], meta["divergences"]))
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
end

function write_tex_gate(path::AbstractString, gate)
    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Regime-switching gate statistics for updated US real-data HLT estimation.}")
        println(io, "\\label{tab:realdata_gate}")
        println(io, "\\begin{tabular}{lr}")
        println(io, "\\toprule")
        println(io, "Statistic & Value \\\\")
        println(io, "\\midrule")
        println(io, "Gate mode & $(gate["gate_mode"]) \\\\")
        println(io, @sprintf("Gate share & %.6f \\\\", gate["gate_share"]))
        if !(gate["gate_nonlinear_periods"] isa Missing)
            println(io, "Nonlinear periods & $(gate["gate_nonlinear_periods"]) \\\\")
            println(io, "Linear periods & $(gate["gate_linear_periods"]) \\\\")
            println(io, "Total periods & $(gate["gate_periods"]) \\\\")
        end
        println(io, "Gate filter & $(gate["gate_filter"]) \\\\")
        println(io, "k\\_pre & $(gate["gate_k_pre"]) \\\\")
        println(io, "k\\_post & $(gate["gate_k_post"]) \\\\")
        println(io, "min\\_len & $(gate["gate_min_len"]) \\\\")
        println(io, "tau\\_eps & $(gate["tau_eps"]) \\\\")
        println(io, "tau\\_y & $(gate["tau_y"]) \\\\")
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
end

function write_tex_runmeta(path::AbstractString, chain_path::AbstractString, payload::Dict{Any,Any}, chain_meta, manifest_meta)
    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "\\centering")
        println(io, "\\caption{Run metadata for updated US real-data HLT estimation.}")
        println(io, "\\label{tab:realdata_runmeta}")
        println(io, "\\begin{tabular}{ll}")
        println(io, "\\toprule")
        println(io, "Field & Value \\\\")
        println(io, "\\midrule")
        println(io, "Chain payload & \\texttt{$(replace(chain_path, "_" => "\\_"))} \\\\")
        println(io, "Synthetic/payload source & \\texttt{$(replace(string(get(payload, "synthetic_path", missing)), "_" => "\\_"))} \\\\")
        println(io, "Shock filter & $(get(payload, "shock_filter", missing)) \\\\")
        println(io, "Linear filter & $(get(payload, "linear_filter", missing)) \\\\")
        println(io, "Samples per chain & $(chain_meta["n_samples_per_chain"]) \\\\")
        println(io, "Chains & $(chain_meta["n_chains"]) \\\\")
        if haskey(manifest_meta, "run_git_commit")
            println(io, "Git commit & \\texttt{$(manifest_meta["run_git_commit"])} \\\\")
        end
        if haskey(manifest_meta, "run_created_at")
            println(io, "Run created at & $(manifest_meta["run_created_at"]) \\\\")
        end
        println(io, "Table extraction timestamp & $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) \\\\")
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
    end
end

function copy_to_generated(src_files::Vector{String}, generated_dir::AbstractString)
    generated_dir == "" && return String[]
    mkpath(generated_dir)
    out = String[]
    for src in src_files
        dst = joinpath(generated_dir, basename(src))
        cp(src, dst; force = true)
        push!(out, dst)
    end
    return out
end

function toml_safe(x)
    if x isa Missing
        return "missing"
    elseif x isa AbstractFloat
        return isfinite(x) ? x : string(x)
    elseif x isa AbstractDict
        return Dict(string(k) => toml_safe(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [toml_safe(v) for v in x]
    elseif x isa Tuple
        return [toml_safe(v) for v in x]
    end
    return x
end

function extract_hlt_real_data_tables(chain_path::AbstractString;
                                      out_dir::AbstractString,
                                      run_manifest_path::AbstractString = "",
                                      generated_dir::AbstractString = "")
    mkpath(out_dir)

    payload = MacroModelling.load_hlt_chain_payload(chain_path)
    MacroModelling.validate_hlt_chain_payload(payload; require_chain = true, label = "HLT real-data chain payload")

    chain = payload["chain"]
    param_rows = extract_param_rows(chain)
    chain_meta = extract_chain_meta(chain)
    gate_stats = extract_gate_stats(payload)
    manifest_meta = extract_manifest_meta(run_manifest_path)

    posterior_tex = joinpath(out_dir, "table_hlt_realdata_posterior.tex")
    mcmc_tex = joinpath(out_dir, "table_hlt_realdata_mcmc.tex")
    gate_tex = joinpath(out_dir, "table_hlt_realdata_gate.tex")
    runmeta_tex = joinpath(out_dir, "table_hlt_realdata_runmeta.tex")

    write_tex_posterior(posterior_tex, param_rows)
    write_tex_mcmc(mcmc_tex, param_rows, chain_meta)
    write_tex_gate(gate_tex, gate_stats)
    write_tex_runmeta(runmeta_tex, chain_path, payload, chain_meta, manifest_meta)

    summary = Dict(
        "source_chain" => chain_path,
        "run_manifest" => run_manifest_path,
        "param_rows" => param_rows,
        "chain_meta" => chain_meta,
        "gate_stats" => gate_stats,
        "manifest_meta" => manifest_meta,
        "generated_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "tex_tables" => Dict(
            "posterior" => posterior_tex,
            "mcmc" => mcmc_tex,
            "gate" => gate_tex,
            "runmeta" => runmeta_tex,
        ),
    )

    toml_path = joinpath(out_dir, "hlt_real_data_summary.toml")
    json_path = joinpath(out_dir, "hlt_real_data_summary.json")
    open(toml_path, "w") do io
        TOML.print(io, toml_safe(summary))
    end
    open(json_path, "w") do io
        JSON.print(io, summary)
    end

    copied = copy_to_generated([posterior_tex, mcmc_tex, gate_tex, runmeta_tex, toml_path, json_path], generated_dir)

    return Dict(
        "posterior_tex" => posterior_tex,
        "mcmc_tex" => mcmc_tex,
        "gate_tex" => gate_tex,
        "runmeta_tex" => runmeta_tex,
        "summary_toml" => toml_path,
        "summary_json" => json_path,
        "copied_generated" => copied,
    )
end

function main(args)
    chain_path = get(args, 1, "")
    chain_path == "" && error("Usage: julia extract_hlt_real_data_tables.jl <chain_path> [--out-dir=<dir>] [--run-manifest=<path>] [--generated-dir=<dir>]")

    out_dir = parse_arg_string(args, "--out-dir", joinpath(dirname(chain_path), "..", "tables"))
    run_manifest = parse_arg_string(args, "--run-manifest", "")
    generated_dir = parse_arg_string(args, "--generated-dir", "")

    outputs = extract_hlt_real_data_tables(chain_path;
        out_dir = out_dir,
        run_manifest_path = run_manifest,
        generated_dir = generated_dir,
    )

    println("Real-data tables extracted")
    println("  Chain: $chain_path")
    println("  Output dir: $out_dir")
    println("  Posterior table: $(outputs["posterior_tex"])")
    println("  MCMC table: $(outputs["mcmc_tex"])")
    println("  Gate table: $(outputs["gate_tex"])")
    println("  Run-meta table: $(outputs["runmeta_tex"])")
    println("  Summary TOML: $(outputs["summary_toml"])")
    println("  Summary JSON: $(outputs["summary_json"])")
    if !isempty(outputs["copied_generated"])
        println("  Copied to generated dir: $(generated_dir)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
