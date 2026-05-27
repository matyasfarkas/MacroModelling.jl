#!/usr/bin/env julia

using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

const COMPARE_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

Base.@kwdef struct CompareOptions
    dataset::String = ""
    surrogate::String = ""
    out_dir::String = joinpath(COMPARE_REPO_ROOT, ".local_artifacts", "hlt_reduced_bridge_validation", "posterior_grid_compare_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    param_set::Symbol = :investment_4p_supported
    truth_mode::String = "validation-nearest-center"
    truth_index::Int = 0
    split_seed::Int = 20260527
    obs_sigma_scale::Float64 = 0.05
    obs_sigma_floor::Float64 = 1.0e-3
    dgp_noise_scale::Float64 = 0.0
    dgp_noise_seed::Int = 20260528
end

function parse_arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix) + 1:end]
    end
    return default
end

function parse_args(args::Vector{String})
    opts = CompareOptions()
    dataset = parse_arg(args, "--dataset", opts.dataset)
    surrogate = parse_arg(args, "--surrogate", opts.surrogate)
    isempty(dataset) && error("--dataset=<hlt_sep_surrogate_dataset.jls> is required")
    isempty(surrogate) && error("--surrogate=<hlt_sep_surrogate_trained.jls> is required")
    return CompareOptions(
        dataset = dataset,
        surrogate = surrogate,
        out_dir = parse_arg(args, "--out-dir", opts.out_dir),
        param_set = Symbol(parse_arg(args, "--param-set", String(opts.param_set))),
        truth_mode = parse_arg(args, "--truth-mode", opts.truth_mode),
        truth_index = parse(Int, parse_arg(args, "--truth-index", string(opts.truth_index))),
        split_seed = parse(Int, parse_arg(args, "--split-seed", string(opts.split_seed))),
        obs_sigma_scale = parse(Float64, parse_arg(args, "--obs-sigma-scale", string(opts.obs_sigma_scale))),
        obs_sigma_floor = parse(Float64, parse_arg(args, "--obs-sigma-floor", string(opts.obs_sigma_floor))),
        dgp_noise_scale = parse(Float64, parse_arg(args, "--dgp-noise-scale", string(opts.dgp_noise_scale))),
        dgp_noise_seed = parse(Int, parse_arg(args, "--dgp-noise-seed", string(opts.dgp_noise_seed))),
    )
end

function git_commit()
    try
        return readchomp(`git -C $COMPARE_REPO_ROOT rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function logsumexp(x::AbstractVector{<:Real})
    finite = [Float64(v) for v in x if isfinite(Float64(v))]
    isempty(finite) && return -Inf
    m = maximum(finite)
    return m + log(sum(exp.(finite .- m)))
end

function weighted_quantile(vals::Vector{Float64}, weights::Vector{Float64}, p::Float64)
    order = sortperm(vals)
    v = vals[order]
    w = weights[order]
    total = sum(w)
    total > 0 || return NaN
    cdf = cumsum(w) ./ total
    idx = findfirst(>=(p), cdf)
    return v[idx === nothing ? length(v) : idx]
end

function posterior_summary(theta_grid::Matrix{Float64}, logpost::Vector{Float64}, theta_names::Vector{Symbol})
    lse = logsumexp(logpost)
    weights = isfinite(lse) ? exp.(logpost .- lse) : fill(NaN, length(logpost))
    map_idx = argmax(logpost)
    rows = Vector{Dict{String,Any}}()
    for j in 1:size(theta_grid, 2)
        vals = theta_grid[:, j]
        mean_j = sum(weights .* vals)
        var_j = sum(weights .* (vals .- mean_j) .^ 2)
        push!(rows, Dict{String,Any}(
            "parameter" => String(theta_names[j]),
            "mean" => mean_j,
            "sd" => sqrt(max(var_j, 0.0)),
            "q05" => weighted_quantile(vals, weights, 0.05),
            "q95" => weighted_quantile(vals, weights, 0.95),
            "map" => vals[map_idx],
        ))
    end
    return Dict{String,Any}(
        "rows" => rows,
        "log_marginal" => lse,
        "map_index" => map_idx,
        "map_theta" => vec(theta_grid[map_idx, :]),
        "max_logpost" => maximum(logpost),
        "weights" => weights,
    )
end

function matrix_from_theta_grid(raw)
    raw isa AbstractVector || error("theta_grid metadata must be a vector.")
    n = length(raw)
    n > 0 || error("theta_grid is empty.")
    p = length(raw[1])
    out = Matrix{Float64}(undef, n, p)
    for i in 1:n
        length(raw[i]) == p || error("theta_grid has ragged rows.")
        out[i, :] .= Float64.(raw[i])
    end
    return out
end

function nearest_index(theta_grid::Matrix{Float64}, target::Vector{Float64}, candidates::Vector{Int})
    isempty(candidates) && (candidates = collect(1:size(theta_grid, 1)))
    scales = vec(maximum(theta_grid, dims = 1) .- minimum(theta_grid, dims = 1))
    scales[scales .<= sqrt(eps(Float64))] .= 1.0
    scores = [sum(((theta_grid[i, :] .- target) ./ scales) .^ 2) for i in candidates]
    return candidates[argmin(scores)]
end

function training_split_indices(n::Int, seed::Int)
    Random.seed!(seed)
    n_train = n == 1 ? 1 : clamp(Int(floor(0.9 * n)), 1, n - 1)
    perm = randperm(n)
    return perm[1:n_train], perm[n_train + 1:end]
end

function choose_truth_index(theta_grid::Matrix{Float64},
                            specs::Vector{ParameterSpec},
                            opts::CompareOptions)
    n = size(theta_grid, 1)
    if opts.truth_index > 0
        1 <= opts.truth_index <= n || error("--truth-index must be between 1 and $n")
        return opts.truth_index, Int[], Int[]
    end
    center = Float64[Float64(spec.prior_params.μ) for spec in specs]
    train_idx, val_idx = training_split_indices(n, opts.split_seed)
    if opts.truth_mode == "validation-nearest-center"
        return nearest_index(theta_grid, center, val_idx), train_idx, val_idx
    elseif opts.truth_mode == "nearest-center"
        return nearest_index(theta_grid, center, collect(1:n)), train_idx, val_idx
    elseif opts.truth_mode == "first-validation"
        isempty(val_idx) && error("No validation points available for truth-mode=first-validation")
        return val_idx[1], train_idx, val_idx
    else
        error("Unknown --truth-mode=$(opts.truth_mode). Use validation-nearest-center, nearest-center, first-validation, or --truth-index.")
    end
end

function prior_logpdf(theta::Vector{Float64}, specs::Vector{ParameterSpec})
    length(theta) == length(specs) || error("theta/spec length mismatch")
    lp = 0.0
    for (x, spec) in zip(theta, specs)
        lo, hi = spec.bounds
        (lo <= x <= hi) || return -Inf
        if spec.prior_type == :Normal
            μ = Float64(spec.prior_params.μ)
            σ = Float64(spec.prior_params.σ)
            lp += -0.5 * ((x - μ) / σ)^2 - log(σ) - 0.5 * log(2π)
        elseif spec.prior_type == :Uniform
            lp += -log(hi - lo)
        else
            error("Unsupported prior type in bridge comparison: $(spec.prior_type)")
        end
    end
    return lp
end

function gaussian_loglik(y::Vector{Float64}, pred::Vector{Float64}, sigma::Vector{Float64})
    all(isfinite, pred) || return -Inf
    r = (y .- pred) ./ sigma
    return -0.5 * sum(r .^ 2 .+ log.(2π .* sigma .^ 2))
end

function fmt(x)
    !isfinite(Float64(x)) && return string(x)
    return @sprintf("%.6g", Float64(x))
end

function latex_escape(s::AbstractString)
    return replace(s, "_" => "\\_")
end

function write_latex_table(path::String, rows::Vector{Dict{String,Any}})
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrrrrrr}")
        println(io, "\\toprule")
        println(io, "Parameter & True & Direct mean & Direct 90\\% CI & Surrogate mean & Surrogate 90\\% CI & ROM1 mean & ROM1 90\\% CI \\\\")
        println(io, "\\midrule")
        for row in rows
            dci = "[$(fmt(row["direct_q05"])), $(fmt(row["direct_q95"]))]"
            sci = "[$(fmt(row["surrogate_q05"])), $(fmt(row["surrogate_q95"]))]"
            rci = "[$(fmt(row["rom1_q05"])), $(fmt(row["rom1_q95"]))]"
            println(io, "$(latex_escape(row["parameter"])) & $(fmt(row["true"])) & $(fmt(row["direct_mean"])) & $(dci) & $(fmt(row["surrogate_mean"])) & $(sci) & $(fmt(row["rom1_mean"])) & $(rci) \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function run_compare(opts::CompareOptions)
    mkpath(opts.out_dir)
    data = deserialize(opts.dataset)
    bundle = deserialize(opts.surrogate)
    haskey(data, "meta") || error("Dataset missing meta.")
    haskey(data, "X") || error("Dataset missing X.")
    haskey(data, "Y") || error("Dataset missing Y.")
    haskey(data, "Y_rom1") || error("Dataset missing Y_rom1.")
    haskey(bundle, "frozen") || error("Surrogate bundle missing frozen network.")

    meta = data["meta"]
    X = Matrix{Float64}(data["X"])
    Y = Matrix{Float64}(data["Y"])
    Y_rom1 = Matrix{Float64}(data["Y_rom1"])
    theta_grid = matrix_from_theta_grid(get(meta, "theta_grid", nothing))
    theta_names = Symbol.(get(meta, "theta_names", Symbol[]))
    isempty(theta_names) && error("Dataset metadata missing theta_names.")
    n = size(theta_grid, 1)
    n == size(X, 2) == size(Y, 2) == size(Y_rom1, 2) ||
        error("Dataset grid/sample mismatch: theta=$(n), X=$(size(X,2)), Y=$(size(Y,2)), Y_rom1=$(size(Y_rom1,2))")

    specs = get_parameter_specs(opts.param_set)
    spec_names = [spec.name for spec in specs]
    spec_names == theta_names || error("Param set $(opts.param_set) names $spec_names do not match dataset theta_names $theta_names")

    d_obs = length(get(meta, "observables", Symbol[]))
    d_obs > 0 || error("Dataset metadata missing observables.")
    Y_obs = Y[1:d_obs, :]
    Y_rom1_obs = Y_rom1[1:d_obs, :]
    frozen = bundle["frozen"]
    residual_pred = predict_frozen_batch(frozen, X)
    size(residual_pred, 1) == d_obs || error("Surrogate output dimension $(size(residual_pred,1)) does not match d_obs=$d_obs")
    Y_sur = Y_rom1_obs .+ residual_pred

    truth_idx, train_idx, val_idx = choose_truth_index(theta_grid, specs, opts)
    theta_true = vec(theta_grid[truth_idx, :])
    obs_sigma = max.(opts.obs_sigma_scale .* vec(Statistics.std(Y_obs, dims = 2; corrected = false)), opts.obs_sigma_floor)
    y_true_clean = vec(Y_obs[:, truth_idx])
    Random.seed!(opts.dgp_noise_seed)
    dgp_noise = opts.dgp_noise_scale .* obs_sigma .* randn(length(obs_sigma))
    y_true = y_true_clean .+ dgp_noise

    prior = [prior_logpdf(vec(theta_grid[i, :]), specs) for i in 1:n]
    direct_logpost = similar(prior)
    surrogate_logpost = similar(prior)
    rom1_logpost = similar(prior)
    for i in 1:n
        direct_logpost[i] = prior[i] + gaussian_loglik(y_true, vec(Y_obs[:, i]), obs_sigma)
        surrogate_logpost[i] = prior[i] + gaussian_loglik(y_true, vec(Y_sur[:, i]), obs_sigma)
        rom1_logpost[i] = prior[i] + gaussian_loglik(y_true, vec(Y_rom1_obs[:, i]), obs_sigma)
    end

    direct = posterior_summary(theta_grid, direct_logpost, theta_names)
    surrogate = posterior_summary(theta_grid, surrogate_logpost, theta_names)
    rom1 = posterior_summary(theta_grid, rom1_logpost, theta_names)

    rmse_rom1 = vec(sqrt.(mean((Y_rom1_obs .- Y_obs) .^ 2, dims = 2)))
    rmse_sur = vec(sqrt.(mean((Y_sur .- Y_obs) .^ 2, dims = 2)))
    improvement = 1 .- rmse_sur ./ rmse_rom1

    rows = Vector{Dict{String,Any}}()
    all_surrogate_overlap = true
    all_rom1_overlap = true
    for j in 1:length(theta_names)
        drow = direct["rows"][j]
        srow = surrogate["rows"][j]
        rrow = rom1["rows"][j]
        s_overlap = max(drow["q05"], srow["q05"]) <= min(drow["q95"], srow["q95"])
        r_overlap = max(drow["q05"], rrow["q05"]) <= min(drow["q95"], rrow["q95"])
        all_surrogate_overlap &= s_overlap
        all_rom1_overlap &= r_overlap
        push!(rows, Dict{String,Any}(
            "parameter" => String(theta_names[j]),
            "true" => theta_true[j],
            "direct_mean" => drow["mean"],
            "direct_q05" => drow["q05"],
            "direct_q95" => drow["q95"],
            "surrogate_mean" => srow["mean"],
            "surrogate_q05" => srow["q05"],
            "surrogate_q95" => srow["q95"],
            "rom1_mean" => rrow["mean"],
            "rom1_q05" => rrow["q05"],
            "rom1_q95" => rrow["q95"],
            "surrogate_direct_abs_diff" => abs(srow["mean"] - drow["mean"]),
            "rom1_direct_abs_diff" => abs(rrow["mean"] - drow["mean"]),
            "surrogate_interval_overlap" => s_overlap,
            "rom1_interval_overlap" => r_overlap,
        ))
    end

    pred_rmse_sur_vs_direct = sqrt(mean((Y_sur .- Y_obs) .^ 2))
    pred_rmse_rom_vs_direct = sqrt(mean((Y_rom1_obs .- Y_obs) .^ 2))
    surface_rmse_sur = sqrt(mean((surrogate_logpost .- direct_logpost) .^ 2))
    surface_rmse_rom = sqrt(mean((rom1_logpost .- direct_logpost) .^ 2))
    result = Dict{String,Any}(
        "created_at" => string(Dates.now()),
        "git_commit" => git_commit(),
        "dataset" => opts.dataset,
        "surrogate" => opts.surrogate,
        "param_set" => String(opts.param_set),
        "truth_mode" => opts.truth_mode,
        "truth_index" => truth_idx,
        "truth_in_training_split" => truth_idx in train_idx,
        "truth_in_validation_split" => truth_idx in val_idx,
        "train_idx" => train_idx,
        "val_idx" => val_idx,
        "theta_names" => String.(theta_names),
        "theta_true" => theta_true,
        "y_true_clean" => y_true_clean,
        "y_true" => y_true,
        "dgp_noise" => dgp_noise,
        "dgp_noise_scale" => opts.dgp_noise_scale,
        "dgp_noise_seed" => opts.dgp_noise_seed,
        "obs_sigma" => obs_sigma,
        "observables" => String.(get(meta, "observables", Symbol[])),
        "prior_logpdf" => prior,
        "direct_logpost" => direct_logpost,
        "surrogate_logpost" => surrogate_logpost,
        "rom1_logpost" => rom1_logpost,
        "direct_summary" => direct,
        "surrogate_summary" => surrogate,
        "rom1_summary" => rom1,
        "rows" => rows,
        "rmse_surrogate_vs_direct_by_obs" => rmse_sur,
        "rmse_rom1_vs_direct_by_obs" => rmse_rom1,
        "rmse_improvement_by_obs" => improvement,
        "prediction_rmse_surrogate_vs_direct" => pred_rmse_sur_vs_direct,
        "prediction_rmse_rom1_vs_direct" => pred_rmse_rom_vs_direct,
        "surface_rmse_surrogate_vs_direct" => surface_rmse_sur,
        "surface_rmse_rom1_vs_direct" => surface_rmse_rom,
        "surrogate_all_interval_overlap" => all_surrogate_overlap,
        "rom1_all_interval_overlap" => all_rom1_overlap,
        "surrogate_improves_prediction_rmse" => pred_rmse_sur_vs_direct < pred_rmse_rom_vs_direct,
        "surrogate_improves_surface_rmse" => surface_rmse_sur < surface_rmse_rom,
        "comparison_pass" => all_surrogate_overlap && pred_rmse_sur_vs_direct < pred_rmse_rom_vs_direct && surface_rmse_sur < surface_rmse_rom,
    )

    payload_path = joinpath(opts.out_dir, "hlt_bridge_posterior_grid_comparison.jls")
    summary_path = joinpath(opts.out_dir, "SUMMARY.md")
    table_path = joinpath(opts.out_dir, "comparison_table.tex")
    serialize(payload_path, result)
    write_latex_table(table_path, rows)
    open(summary_path, "w") do io
        println(io, "# HLT Bridge Posterior Grid Comparison")
        println(io)
        println(io, "- Created: `$(result["created_at"])`")
        println(io, "- Git commit: `$(result["git_commit"])`")
        println(io, "- Dataset: `$(opts.dataset)`")
        println(io, "- Surrogate: `$(opts.surrogate)`")
        println(io, "- Parameter set: `$(opts.param_set)`")
        println(io, "- Grid cells: `$n`")
        println(io, "- Truth mode/index: `$(opts.truth_mode)` / `$(truth_idx)`")
        println(io, "- Truth theta: `$(join(["$(theta_names[i])=$(fmt(theta_true[i]))" for i in eachindex(theta_names)], ", "))`")
        println(io, "- Truth in training split: `$(result["truth_in_training_split"])`")
        println(io, "- Truth in validation split: `$(result["truth_in_validation_split"])`")
        println(io, "- Observation sigma: `$(join(fmt.(obs_sigma), ", "))`")
        println(io, "- DGP measurement-noise scale/seed: `$(opts.dgp_noise_scale)` / `$(opts.dgp_noise_seed)`")
        println(io)
        println(io, "## Fit Against Direct SEP Grid")
        println(io)
        println(io, "- Prediction RMSE, ROM1 vs direct SEP: `$(fmt(pred_rmse_rom_vs_direct))`")
        println(io, "- Prediction RMSE, surrogate vs direct SEP: `$(fmt(pred_rmse_sur_vs_direct))`")
        println(io, "- Log-posterior surface RMSE, ROM1 vs direct SEP: `$(fmt(surface_rmse_rom))`")
        println(io, "- Log-posterior surface RMSE, surrogate vs direct SEP: `$(fmt(surface_rmse_sur))`")
        println(io, "- Surrogate interval overlap with direct: `$(all_surrogate_overlap)`")
        println(io, "- ROM1 interval overlap with direct: `$(all_rom1_overlap)`")
        println(io, "- Comparison pass: `$(result["comparison_pass"])`")
        println(io)
        println(io, "## Prediction RMSE By Observable")
        println(io)
        println(io, "| Observable | ROM1 RMSE | Surrogate RMSE | Improvement |")
        println(io, "|---|---:|---:|---:|")
        obs_names = String.(get(meta, "observables", Symbol[]))
        for i in 1:d_obs
            println(io, "| `$(obs_names[i])` | $(fmt(rmse_rom1[i])) | $(fmt(rmse_sur[i])) | $(fmt(100 * improvement[i]))% |")
        end
        println(io)
        println(io, "## Posterior Marginals")
        println(io)
        println(io, "| Parameter | True | Direct mean | Direct 90% CI | Surrogate mean | Surrogate 90% CI | ROM1 mean | ROM1 90% CI |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows
            dci = "[$(fmt(row["direct_q05"])), $(fmt(row["direct_q95"]))]"
            sci = "[$(fmt(row["surrogate_q05"])), $(fmt(row["surrogate_q95"]))]"
            rci = "[$(fmt(row["rom1_q05"])), $(fmt(row["rom1_q95"]))]"
            println(io, "| `$(row["parameter"])` | $(fmt(row["true"])) | $(fmt(row["direct_mean"])) | $dci | $(fmt(row["surrogate_mean"])) | $sci | $(fmt(row["rom1_mean"])) | $rci |")
        end
        println(io)
        println(io, "## Scope")
        println(io)
        println(io, "This is a one-period, known-feature posterior-grid comparison on the finite HLT bridge support. It validates that the ROM1-residual surrogate tracks the direct SEP one-step prediction surface substantially better than ROM1. It is not yet the full inversion-filter or HMC bridge result.")
    end
    println("Wrote payload: $payload_path")
    println("Wrote summary: $summary_path")
    println("Wrote table: $table_path")
    println("Comparison pass: $(result["comparison_pass"])")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_compare(parse_args(ARGS))
end
