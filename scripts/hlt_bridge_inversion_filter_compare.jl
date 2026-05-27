#!/usr/bin/env julia

using Dates
using Printf
using Random
using Serialization
using TOML

const INV_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

Base.@kwdef struct InversionBridgeOptions
    dataset::String = ""
    surrogate::String = ""
    out_dir::String = joinpath(INV_REPO_ROOT, ".local_artifacts", "hlt_reduced_bridge_validation", "inversion_bridge_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    param_set::Symbol = :investment_4p_supported
    periods::Int = 8
    truth_mode::String = "validation-nearest-center"
    truth_index::Int = 0
    split_seed::Int = 20260527
    obs_sigma_scale::Float64 = 1.0
    obs_sigma_floor::Float64 = 1.0e-3
    direct_eval_points::Int = 25
    sep_horizon::Int = 4
    sep_maxit::Int = 80
    dry_run::Bool = true
end

function parse_arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix) + 1:end]
    end
    return default
end

function parse_bool(x::AbstractString)
    v = lowercase(strip(x))
    v in ("1", "true", "yes", "y", "on") && return true
    v in ("0", "false", "no", "n", "off") && return false
    error("Cannot parse boolean: $x")
end

function parse_args(args::Vector{String})
    opts = InversionBridgeOptions()
    dataset = parse_arg(args, "--dataset", opts.dataset)
    surrogate = parse_arg(args, "--surrogate", opts.surrogate)
    isempty(dataset) && error("--dataset=<hlt_sep_surrogate_dataset.jls> is required")
    isempty(surrogate) && error("--surrogate=<hlt_sep_surrogate_trained.jls> is required")
    return InversionBridgeOptions(
        dataset = dataset,
        surrogate = surrogate,
        out_dir = parse_arg(args, "--out-dir", opts.out_dir),
        param_set = Symbol(parse_arg(args, "--param-set", String(opts.param_set))),
        periods = parse(Int, parse_arg(args, "--periods", string(opts.periods))),
        truth_mode = parse_arg(args, "--truth-mode", opts.truth_mode),
        truth_index = parse(Int, parse_arg(args, "--truth-index", string(opts.truth_index))),
        split_seed = parse(Int, parse_arg(args, "--split-seed", string(opts.split_seed))),
        obs_sigma_scale = parse(Float64, parse_arg(args, "--obs-sigma-scale", string(opts.obs_sigma_scale))),
        obs_sigma_floor = parse(Float64, parse_arg(args, "--obs-sigma-floor", string(opts.obs_sigma_floor))),
        direct_eval_points = parse(Int, parse_arg(args, "--direct-eval-points", string(opts.direct_eval_points))),
        sep_horizon = parse(Int, parse_arg(args, "--sep-horizon", string(opts.sep_horizon))),
        sep_maxit = parse(Int, parse_arg(args, "--sep-maxit", string(opts.sep_maxit))),
        dry_run = parse_bool(parse_arg(args, "--dry-run", string(opts.dry_run))),
    )
end

function git_commit()
    try
        return readchomp(`git -C $INV_REPO_ROOT rev-parse HEAD`)
    catch
        return "unknown"
    end
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

function training_split_indices(n::Int, seed::Int)
    Random.seed!(seed)
    n_train = n == 1 ? 1 : clamp(Int(floor(0.9 * n)), 1, n - 1)
    perm = randperm(n)
    return perm[1:n_train], perm[n_train + 1:end]
end

function nearest_index(theta_grid::Matrix{Float64}, target::Vector{Float64}, candidates::Vector{Int})
    isempty(candidates) && (candidates = collect(1:size(theta_grid, 1)))
    scales = vec(maximum(theta_grid, dims = 1) .- minimum(theta_grid, dims = 1))
    scales[scales .<= sqrt(eps(Float64))] .= 1.0
    scores = [sum(((theta_grid[i, :] .- target) ./ scales) .^ 2) for i in candidates]
    return candidates[argmin(scores)]
end

function choose_truth_index(theta_grid::Matrix{Float64}, specs::Vector{ParameterSpec}, opts::InversionBridgeOptions)
    n = size(theta_grid, 1)
    train_idx, val_idx = training_split_indices(n, opts.split_seed)
    if opts.truth_index > 0
        1 <= opts.truth_index <= n || error("--truth-index must be between 1 and $n")
        return opts.truth_index, train_idx, val_idx
    end
    center = Float64[Float64(spec.prior_params.μ) for spec in specs]
    if opts.truth_mode == "validation-nearest-center"
        return nearest_index(theta_grid, center, val_idx), train_idx, val_idx
    elseif opts.truth_mode == "nearest-center"
        return nearest_index(theta_grid, center, collect(1:n)), train_idx, val_idx
    elseif opts.truth_mode == "first-validation"
        isempty(val_idx) && error("No validation points available.")
        return val_idx[1], train_idx, val_idx
    else
        error("Unknown --truth-mode=$(opts.truth_mode).")
    end
end

function fmt(x)
    return @sprintf("%.6g", Float64(x))
end

function bridge_manifest(opts::InversionBridgeOptions)
    data = deserialize(opts.dataset)
    bundle = deserialize(opts.surrogate)
    for key in ("meta", "X", "Y", "Y_rom1")
        haskey(data, key) || error("Dataset missing key: $key")
    end
    haskey(bundle, "frozen") || error("Surrogate bundle missing frozen network.")
    meta = data["meta"]
    theta_grid = matrix_from_theta_grid(get(meta, "theta_grid", nothing))
    theta_names = Symbol.(get(meta, "theta_names", Symbol[]))
    isempty(theta_names) && error("Dataset metadata missing theta_names.")
    specs = get_parameter_specs(opts.param_set)
    spec_names = [s.name for s in specs]
    spec_names == theta_names ||
        error("Param set $(opts.param_set) names $spec_names do not match dataset theta_names $theta_names")
    X = Matrix{Float64}(data["X"])
    Y = Matrix{Float64}(data["Y"])
    Y_rom1 = Matrix{Float64}(data["Y_rom1"])
    d_obs = length(get(meta, "observables", Symbol[]))
    d_obs > 0 || error("Dataset metadata missing observables.")
    n = size(theta_grid, 1)
    n == size(X, 2) == size(Y, 2) == size(Y_rom1, 2) ||
        error("Dataset grid/sample count mismatch.")
    truth_idx, train_idx, val_idx = choose_truth_index(theta_grid, specs, opts)
    obs_sigma = max.(opts.obs_sigma_scale .* vec(std(Y[1:d_obs, :], dims = 2; corrected = false)), opts.obs_sigma_floor)
    direct_eval_points = min(opts.direct_eval_points, n)
    panel_idx = val_idx[1:min(length(val_idx), opts.periods)]
    if length(panel_idx) < opts.periods
        panel_idx = vcat(panel_idx, train_idx[1:(opts.periods - length(panel_idx))])
    end

    return Dict{String,Any}(
        "created_at" => string(Dates.now()),
        "git_commit" => git_commit(),
        "dry_run" => opts.dry_run,
        "dataset" => opts.dataset,
        "surrogate" => opts.surrogate,
        "param_set" => String(opts.param_set),
        "theta_names" => String.(theta_names),
        "observables" => String.(get(meta, "observables", Symbol[])),
        "dataset_samples" => n,
        "feature_dim" => size(X, 1),
        "output_dim" => size(Y, 1),
        "obs_dim" => d_obs,
        "periods" => opts.periods,
        "truth_mode" => opts.truth_mode,
        "truth_index" => truth_idx,
        "truth_in_training_split" => truth_idx in train_idx,
        "truth_in_validation_split" => truth_idx in val_idx,
        "theta_true" => vec(theta_grid[truth_idx, :]),
        "panel_indices" => panel_idx,
        "obs_sigma" => obs_sigma,
        "direct_eval_points" => direct_eval_points,
        "sep_horizon" => opts.sep_horizon,
        "sep_maxit" => opts.sep_maxit,
        "execution_plan" => [
            "construct a short synthetic HLT observation panel from a held-out validation sequence",
            "recover shocks with the ROM1 inversion filter under each candidate theta",
            "evaluate the direct SEP inversion objective on the same observation panel and parameter grid",
            "evaluate the ROM1-residual surrogate objective with the same recovered-shock architecture",
            "compare direct SEP, ROM1, and surrogate posterior surfaces by means, intervals, MAP ranking, and surface RMSE",
        ],
        "acceptance_criteria" => [
            "direct SEP inversion objective finite for the truth point and all direct evaluation anchors",
            "surrogate 90 percent intervals overlap direct SEP for every bridge parameter",
            "surrogate surface RMSE is materially below ROM1 surface RMSE",
            "truth value lies in direct and surrogate 90 percent intervals for at least three of four parameters",
        ],
        "artifact_schema" => [
            "manifest.toml",
            "SUMMARY.md",
            "synthetic_panel.jls",
            "direct_inversion_grid.jls",
            "surrogate_inversion_grid.jls",
            "comparison_table.tex",
        ],
    )
end

function write_summary(path::String, manifest::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT Multi-Period Inversion Bridge")
        println(io)
        println(io, "- Created: `$(manifest["created_at"])`")
        println(io, "- Git commit: `$(manifest["git_commit"])`")
        println(io, "- Dry run: `$(manifest["dry_run"])`")
        println(io, "- Dataset: `$(manifest["dataset"])`")
        println(io, "- Surrogate: `$(manifest["surrogate"])`")
        println(io, "- Parameter set: `$(manifest["param_set"])`")
        println(io, "- Periods: `$(manifest["periods"])`")
        println(io, "- Direct evaluation points: `$(manifest["direct_eval_points"])`")
        println(io, "- SEP horizon/maxit: `$(manifest["sep_horizon"]) / $(manifest["sep_maxit"])`")
        println(io, "- Truth index: `$(manifest["truth_index"])`")
        println(io, "- Truth theta: `$(join(["$(manifest["theta_names"][i])=$(fmt(manifest["theta_true"][i]))" for i in eachindex(manifest["theta_names"])], ", "))`")
        println(io, "- Truth in validation split: `$(manifest["truth_in_validation_split"])`")
        println(io, "- Observation sigma: `$(join(fmt.(manifest["obs_sigma"]), ", "))`")
        println(io)
        println(io, "## Execution Plan")
        println(io)
        for item in manifest["execution_plan"]
            println(io, "- $item")
        end
        println(io)
        println(io, "## Acceptance Criteria")
        println(io)
        for item in manifest["acceptance_criteria"]
            println(io, "- $item")
        end
        println(io)
        println(io, "## Current Status")
        println(io)
        if manifest["dry_run"]
            println(io, "Design scaffold only. The script has verified dataset/surrogate compatibility, parameter names, split provenance, and artifact schema. The executable inversion evaluator is the next code step.")
        else
            println(io, "Executable mode is not implemented yet; use this manifest to wire the inversion evaluator.")
        end
    end
end

function run_inversion_bridge(opts::InversionBridgeOptions)
    mkpath(opts.out_dir)
    manifest = bridge_manifest(opts)
    manifest_path = joinpath(opts.out_dir, "manifest.toml")
    summary_path = joinpath(opts.out_dir, "SUMMARY.md")
    open(manifest_path, "w") do io
        TOML.print(io, manifest)
    end
    write_summary(summary_path, manifest)
    println("Wrote manifest: $manifest_path")
    println("Wrote summary: $summary_path")
    if !opts.dry_run
        error("Executable multi-period inversion bridge is not implemented yet. Run with --dry-run=true to freeze the design.")
    end
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_inversion_bridge(parse_args(ARGS))
end
