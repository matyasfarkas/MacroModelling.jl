#!/usr/bin/env julia

using Dates
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

Base.@kwdef mutable struct SweepOptions
    stage::String = "postprocess"
    run_id::String = "hlt_bridge_robustness_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_root::String = joinpath(REPO_ROOT, ".local_artifacts", "hlt_reduced_bridge_validation")
    dataset_dir::String = ""
    param_set::String = "investment_4p_supported"
    grid::Int = 5
    epochs::Int = 400
    hidden::Int = 256
    hidden2::Int = 128
    arch::String = "mlp"
    seed::Int = 1
    obs_sigma_scale::Float64 = 1.0
    dgp_noise_scale::Float64 = 0.25
    wait_timeout_minutes::Int = 720
    poll_seconds::Int = 60
end

function parse_arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix) + 1:end]
    end
    return default
end

function parse_args(args::Vector{String})
    opts = SweepOptions()
    return SweepOptions(
        stage = parse_arg(args, "--stage", opts.stage),
        run_id = parse_arg(args, "--run-id", opts.run_id),
        out_root = parse_arg(args, "--out-root", opts.out_root),
        dataset_dir = parse_arg(args, "--dataset-dir", opts.dataset_dir),
        param_set = parse_arg(args, "--param-set", opts.param_set),
        grid = parse(Int, parse_arg(args, "--grid", string(opts.grid))),
        epochs = parse(Int, parse_arg(args, "--epochs", string(opts.epochs))),
        hidden = parse(Int, parse_arg(args, "--hidden", string(opts.hidden))),
        hidden2 = parse(Int, parse_arg(args, "--hidden2", string(opts.hidden2))),
        arch = parse_arg(args, "--arch", opts.arch),
        seed = parse(Int, parse_arg(args, "--seed", string(opts.seed))),
        obs_sigma_scale = parse(Float64, parse_arg(args, "--obs-sigma-scale", string(opts.obs_sigma_scale))),
        dgp_noise_scale = parse(Float64, parse_arg(args, "--dgp-noise-scale", string(opts.dgp_noise_scale))),
        wait_timeout_minutes = parse(Int, parse_arg(args, "--wait-timeout-minutes", string(opts.wait_timeout_minutes))),
        poll_seconds = parse(Int, parse_arg(args, "--poll-seconds", string(opts.poll_seconds))),
    )
end

function julia_cmd(args::AbstractString...)
    exe = joinpath(Sys.BINDIR, Base.julia_exename())
    return Cmd([exe, "--project=$REPO_ROOT", args...])
end

function timestamp()
    return Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
end

function append_line(path::String, msg::String)
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, "[$(timestamp())] $msg")
    end
end

function run_logged(cmd::Cmd, log_path::String, label::String)
    append_line(log_path, "START $label")
    append_line(log_path, "CMD $(cmd)")
    ok = false
    open(log_path, "a") do io
        ok = success(pipeline(cmd; stdout = io, stderr = io))
    end
    append_line(log_path, ok ? "PASS $label" : "FAIL $label")
    ok || error("$label failed; see $log_path")
    return nothing
end

function wait_for_dataset(dataset_path::String, log_path::String, opts::SweepOptions)
    deadline = time() + 60 * opts.wait_timeout_minutes
    while !isfile(dataset_path)
        time() > deadline && error("Timed out waiting for dataset: $dataset_path")
        checkpoint = joinpath(dirname(dataset_path), "hlt_sep_surrogate_dataset_checkpoint.jls")
        if isfile(checkpoint)
            sz_mb = stat(checkpoint).size / 1_000_000
            append_line(log_path, @sprintf("Waiting for dataset; checkpoint %.2f MB", sz_mb))
        else
            append_line(log_path, "Waiting for dataset; checkpoint not present yet")
        end
        sleep(opts.poll_seconds)
    end
    append_line(log_path, @sprintf("Dataset ready: %s (%.2f MB)", dataset_path, stat(dataset_path).size / 1_000_000))
end

function write_manifest(path::String, opts::SweepOptions, paths::Dict{String,String}, status::String)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT Bridge Robustness Sweep")
        println(io)
        println(io, "- Created/updated: `$(timestamp())`")
        println(io, "- Status: `$status`")
        println(io, "- Run ID: `$(opts.run_id)`")
        println(io, "- Stage: `$(opts.stage)`")
        println(io, "- Parameter set: `$(opts.param_set)`")
        println(io, "- Grid: `$(opts.grid)`")
        println(io, "- Training: `$(opts.arch)`, epochs `$(opts.epochs)`, hidden `$(opts.hidden)`, hidden2 `$(opts.hidden2)`, seed `$(opts.seed)`")
        println(io, "- Posterior comparison: obs sigma scale `$(opts.obs_sigma_scale)`, DGP noise scale `$(opts.dgp_noise_scale)`")
        println(io)
        println(io, "## Paths")
        println(io)
        for key in sort(collect(keys(paths)))
            println(io, "- $(key): `$(paths[key])`")
        end
        println(io)
        println(io, "## Scope")
        println(io)
        println(io, "This robustness sweep extends the finite-support HLT bridge. It remains a one-period known-feature grid comparison unless followed by a separate inversion-filter or HMC bridge run.")
    end
end

function dataset_generation_cmd(opts::SweepOptions, dataset_dir::String)
    return julia_cmd(
        joinpath(REPO_ROOT, "scripts", "hlt_sep_surrogate_dataset_generate.jl"),
        "--param-set=$(opts.param_set)",
        "--theta-sampling=grid",
        "--grid=$(opts.grid)",
        "--samples-per-theta=1",
        "--burn-in=1",
        "--sample-length=1",
        "--sample-start=47",
        "--rom-orders=1",
        "--sep-horizon=2",
        "--sep-maxit=40",
        "--sep-accept-tol=1e-2",
        "--shock-scaling=none",
        "--shock-scale=0.05",
        "--use-obc",
        "--theta-attempts-per-theta=2",
        "--retry-on-early-failure=true",
        "--output-dir=$dataset_dir",
    )
end

function run_sweep(opts::SweepOptions)
    dataset_dir = isempty(opts.dataset_dir) ? joinpath(opts.out_root, opts.run_id) : opts.dataset_dir
    dataset_path = joinpath(dataset_dir, "hlt_sep_surrogate_dataset.jls")
    support_path = joinpath(dataset_dir, "SUPPORT_REPORT.md")
    surrogate_path = joinpath(dataset_dir, "hlt_sep_surrogate_trained_$(opts.run_id).jls")
    compare_dir = joinpath(opts.out_root, "posterior_grid_compare_$(opts.run_id)")
    log_path = joinpath(opts.out_root, "logs", "$(opts.run_id).postprocess.log")
    manifest_path = joinpath(compare_dir, "ROBUSTNESS_SWEEP.md")
    paths = Dict(
        "dataset_dir" => dataset_dir,
        "dataset" => dataset_path,
        "support_report" => support_path,
        "surrogate" => surrogate_path,
        "posterior_compare_dir" => compare_dir,
        "log" => log_path,
        "manifest" => manifest_path,
    )

    write_manifest(manifest_path, opts, paths, "running")
    append_line(log_path, "Robustness sweep started")

    if opts.stage == "full"
        mkpath(dataset_dir)
        run_logged(dataset_generation_cmd(opts, dataset_dir), log_path, "dataset generation")
    elseif opts.stage == "postprocess"
        wait_for_dataset(dataset_path, log_path, opts)
    else
        error("--stage must be full or postprocess")
    end

    run_logged(julia_cmd(
        joinpath(REPO_ROOT, "scripts", "hlt_bridge_support_report.jl"),
        dataset_path,
        "--out=$support_path",
    ), log_path, "support report")

    run_logged(julia_cmd(
        joinpath(REPO_ROOT, "scripts", "hlt_sep_surrogate_train.jl"),
        dataset_path,
        "--rom-residual=1",
        "--obs-only",
        "--epochs=$(opts.epochs)",
        "--hidden=$(opts.hidden)",
        "--hidden2=$(opts.hidden2)",
        "--arch=$(opts.arch)",
        "--seed=$(opts.seed)",
        "--out=$surrogate_path",
    ), log_path, "surrogate training")

    run_logged(julia_cmd(
        joinpath(REPO_ROOT, "scripts", "hlt_bridge_posterior_grid_compare.jl"),
        "--dataset=$dataset_path",
        "--surrogate=$surrogate_path",
        "--out-dir=$compare_dir",
        "--param-set=$(opts.param_set)",
        "--truth-mode=validation-nearest-center",
        "--obs-sigma-scale=$(opts.obs_sigma_scale)",
        "--dgp-noise-scale=$(opts.dgp_noise_scale)",
    ), log_path, "posterior grid comparison")

    write_manifest(manifest_path, opts, paths, "complete")
    append_line(log_path, "Robustness sweep complete")
    println("Robustness sweep complete: $manifest_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep(parse_args(ARGS))
end
