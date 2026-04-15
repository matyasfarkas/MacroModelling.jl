#!/usr/bin/env julia
using Dates
import TOML

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))

repo_root() = normpath(joinpath(@__DIR__, ".."))

function build_julia_cmd(root::String, script_path::String, args::Vector{String})
    base = Base.julia_cmd()
    return `$base --project=$root $script_path $args`
end

cmd_to_string(cmd::Cmd) = sprint(show, cmd)

function write_toml(path::String, payload::Dict)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, payload)
    end
end

function write_summary(path::String, manifest::Dict, steps::Vector{Dict{String,Any}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT Real-Data Update Pipeline")
        println(io)
        println(io, "- Run dir: `", manifest["run_dir"], "`")
        println(io, "- Repo root: `", manifest["repo_root"], "`")
        println(io, "- Created: `", manifest["created_at"], "`")
        println(io, "- Dry run: `", manifest["dry_run"], "`")
        println(io, "- Model variant: `", manifest["model_variant"], "`")
        println(io, "- Samples per chain: `", manifest["samples"], "`")
        println(io, "- Chains: `", manifest["chains"], "`")
        println(io)
        println(io, "## Steps")
        for s in steps
            println(io, "- `", s["name"], "`: `", get(s, "status", "planned"), "`")
            println(io, "  - Script: `", s["script"], "`")
            println(io, "  - Command: `", s["command"], "`")
            if haskey(s, "elapsed_s")
                println(io, "  - Elapsed (s): `", s["elapsed_s"], "`")
            end
            if haskey(s, "error")
                println(io, "  - Error: `", replace(String(s["error"]), "\n" => " "), "`")
            end
        end
    end
end

function run_step!(root::String,
                   run_dir::String,
                   step::Dict{String,Any},
                   manifest::Dict{String,Any},
                   step_results::Vector{Dict{String,Any}};
                   dry_run::Bool)
    script_rel = String(step["script"])
    args = String.(step["args"])
    script_abs = joinpath(root, script_rel)
    isfile(script_abs) || error("Script not found: $script_abs")

    cmd = build_julia_cmd(root, script_abs, args)
    rec = Dict{String,Any}(
        "name" => String(step["name"]),
        "script" => script_rel,
        "command" => cmd_to_string(cmd),
        "args" => args,
    )

    for (k, v) in get(step, "outputs", Dict{String,Any}())
        rec[String(k)] = v
    end

    start_t = time()
    if dry_run
        rec["status"] = "dry_run"
    else
        try
            run(cmd)
            rec["status"] = "ok"
        catch err
            rec["status"] = "failed"
            rec["error"] = sprint(showerror, err)
        end
    end
    rec["elapsed_s"] = round(time() - start_t; digits = 3)

    push!(step_results, rec)
    manifest_steps = get!(manifest, "steps", Dict{String,Any}())
    manifest_steps[rec["name"]] = Dict(k => v for (k, v) in rec if k != "name")

    write_toml(joinpath(run_dir, "manifests", "run_manifest.toml"), manifest)
    write_summary(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"), manifest, step_results)

    return rec
end

function pipeline_steps(root::String, run_dir::String;
                        dataset_source::String,
                        csv_path::String,
                        sample_start::Int,
                        sample_end::Int,
                        prefix_end::Int,
                        samples::Int,
                        chains::Int,
                        seed::Int,
                        train_epochs::Int,
                        train_hidden::Int,
                        train_hidden2::Int,
                        obs_sigma_mode::Symbol,
                        obs_sigma_scale::Float64,
                        obs_sigma_floor::Float64,
                        gate_target_share::Float64)
    dataset_dir = joinpath(run_dir, "dataset")
    real_data_dir = joinpath(run_dir, "real_data")
    diagnostics_dir = joinpath(run_dir, "diagnostics")
    tables_dir = joinpath(run_dir, "tables")
    generated_dir = joinpath(root, "docs", "paper", "generated")

    mkpath.(String[dataset_dir, real_data_dir, diagnostics_dir, tables_dir, generated_dir])

    surrogate_path = joinpath(dataset_dir, "hlt_sep_surrogate_trained.jls")
    payload_path = joinpath(real_data_dir, "hlt_real_data_payload.jls")
    gate_path = joinpath(real_data_dir, "gate_calibration.jls")
    chain_path = joinpath(real_data_dir, "hlt_sep_surrogate_estimation_chain.jls")
    checkpoint_path = joinpath(real_data_dir, "hlt_sep_surrogate_estimation_checkpoint.jls")

    manifest_path = joinpath(run_dir, "manifests", "run_manifest.toml")

    return [
        Dict(
            "name" => "surrogate_train",
            "script" => "scripts/hlt_sep_surrogate_train.jl",
            "outputs" => Dict(
                "dataset_source" => dataset_source,
                "surrogate_path" => surrogate_path,
            ),
            "args" => String[
                dataset_source,
                "--rom-residual=1",
                "--epochs=$(train_epochs)",
                "--hidden=$(train_hidden)",
                "--hidden2=$(train_hidden2)",
                "--seed=$(seed)",
                "--out=$(surrogate_path)",
            ],
        ),
        Dict(
            "name" => "real_data_payload",
            "script" => "scripts/hlt_real_data_payload.jl",
            "outputs" => Dict(
                "csv_path" => csv_path,
                "payload_path" => payload_path,
            ),
            "args" => String[
                "--csv=$(csv_path)",
                "--out=$(payload_path)",
                "--sample-start=$(sample_start)",
                "--sample-end=$(sample_end)",
                "--prefix-end=$(prefix_end)",
                "--obs-sigma-mode=$(obs_sigma_mode)",
                "--obs-sigma-scale=$(obs_sigma_scale)",
                "--obs-sigma-floor=$(obs_sigma_floor)",
                "--state-init-filter=kalman",
                "--state-init-algorithm=first_order",
                "--use-obc",
            ],
        ),
        Dict(
            "name" => "gate_calibration",
            "script" => "scripts/hlt_sep_surrogate_gate_calibration.jl",
            "outputs" => Dict("gate_path" => gate_path),
            "args" => String[
                payload_path,
                "--out=$(gate_path)",
                "--target-share=$(gate_target_share)",
                "--tau-eps=NaN",
                "--tau-y=NaN",
                "--min-achieved-share=0.01",
                "--max-achieved-share=0.99",
                "--periods=1",
                "--shock-source=filtered",
                "--shock-filter=kalman",
                "--use-obc",
            ],
        ),
        Dict(
            "name" => "switching_estimation",
            "script" => "scripts/hlt_sep_surrogate_synthetic_estimation.jl",
            "outputs" => Dict(
                "chain_path" => chain_path,
                "checkpoint_path" => checkpoint_path,
            ),
            "args" => String[
                surrogate_path,
                payload_path,
                "--out=$(chain_path)",
                "--gate-calibration=$(gate_path)",
                "--samples=$(samples)",
                "--chains=$(chains)",
                "--sampler=mh",
                "--mh-rw-cprobp=1e-6",
                "--mh-rw-cindp=1e-6",
                "--mh-rw-curvp=1e-4",
                "--gate-mode=hard",
                "--gate-k-pre=0",
                "--gate-k-post=0",
                "--gate-min-len=1",
                "--gate-share-min=0.01",
                "--gate-share-max=0.99",
                "--fail-degenerate-gate=true",
                "--shock-filter=inversion",
                "--linear-filter=inversion",
                "--post-mean-filter=inversion",
                "--checkpoint-path=$(checkpoint_path)",
                "--use-obc",
            ],
        ),
        Dict(
            "name" => "chain_diagnostics",
            "script" => "scripts/hlt_sep_surrogate_chain_report.jl",
            "outputs" => Dict("diagnostics_dir" => diagnostics_dir),
            "args" => String[
                chain_path,
                diagnostics_dir,
                "--title=HLT Updated US Real-Data Posterior Diagnostics",
                "--include-priors=true",
                "--include-eps-pages=false",
            ],
        ),
        Dict(
            "name" => "table_extraction",
            "script" => "scripts/extract_hlt_real_data_tables.jl",
            "outputs" => Dict(
                "tables_dir" => tables_dir,
                "generated_dir" => generated_dir,
            ),
            "args" => String[
                chain_path,
                "--out-dir=$(tables_dir)",
                "--run-manifest=$(manifest_path)",
                "--generated-dir=$(generated_dir)",
            ],
        ),
    ]
end

function main(args)
    root = repo_root()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

    run_dir = parse_arg_string(args, "--run-dir", joinpath(root, ".local_artifacts", "hlt_real_data_runs", "hlt_update_$(timestamp)"))
    dataset_source = parse_arg_string(args, "--dataset-source", joinpath(root, ".local_artifacts", "hlt_validation_runs", "hlt3_20260301_082807", "dataset", "hlt_sep_surrogate_dataset.jls"))
    csv_path = parse_arg_string(args, "--csv", joinpath(root, "test", "data", "usmodel_update.csv"))

    sample_start = parse_arg_int(args, "--sample-start", 47)
    sample_end = parse_arg_int(args, "--sample-end", 290)
    prefix_end = parse_arg_int(args, "--prefix-end", 46)

    samples = parse_arg_int(args, "--samples", 2000)
    chains = parse_arg_int(args, "--chains", 4)
    seed = parse_arg_int(args, "--seed", 42)

    train_epochs = parse_arg_int(args, "--train-epochs", 300)
    train_hidden = parse_arg_int(args, "--train-hidden", 128)
    train_hidden2 = parse_arg_int(args, "--train-hidden2", 64)

    obs_sigma_mode = parse_arg_symbol(args, "--obs-sigma-mode", :data_std)
    obs_sigma_scale = parse_arg_float(args, "--obs-sigma-scale", 0.1)
    obs_sigma_floor = parse_arg_float(args, "--obs-sigma-floor", 1e-4)
    gate_target_share = parse_arg_float(args, "--gate-target-share", 0.1)

    dry_run = parse_arg_bool(args, "--dry-run", false)

    isfile(dataset_source) || error("Dataset source not found: $dataset_source")
    isfile(csv_path) || error("CSV path not found: $csv_path")
    samples > 0 || error("samples must be positive")
    chains > 0 || error("chains must be positive")

    mkpath(run_dir)
    mkpath(joinpath(run_dir, "dataset"))
    mkpath(joinpath(run_dir, "real_data"))
    mkpath(joinpath(run_dir, "manifests"))
    mkpath(joinpath(run_dir, "diagnostics"))
    mkpath(joinpath(run_dir, "tables"))

    manifest = Dict{String,Any}(
        "name" => "HLT real-data updated US run",
        "created_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "repo_root" => root,
        "run_dir" => run_dir,
        "dry_run" => dry_run,
        "model_variant" => "Smets_Wouters_2007_HLT_obc",
        "dataset_source" => dataset_source,
        "csv_path" => csv_path,
        "sample_start" => sample_start,
        "sample_end" => sample_end,
        "prefix_end" => prefix_end,
        "samples" => samples,
        "chains" => chains,
        "seed" => seed,
        "obs_sigma_mode" => string(obs_sigma_mode),
        "obs_sigma_scale" => obs_sigma_scale,
        "obs_sigma_floor" => obs_sigma_floor,
        "gate_target_share" => gate_target_share,
        "steps" => Dict{String,Any}(),
    )

    steps = pipeline_steps(root, run_dir;
        dataset_source = dataset_source,
        csv_path = csv_path,
        sample_start = sample_start,
        sample_end = sample_end,
        prefix_end = prefix_end,
        samples = samples,
        chains = chains,
        seed = seed,
        train_epochs = train_epochs,
        train_hidden = train_hidden,
        train_hidden2 = train_hidden2,
        obs_sigma_mode = obs_sigma_mode,
        obs_sigma_scale = obs_sigma_scale,
        obs_sigma_floor = obs_sigma_floor,
        gate_target_share = gate_target_share,
    )

    step_results = Dict{String,Any}[]
    write_toml(joinpath(run_dir, "manifests", "run_manifest.toml"), manifest)
    write_summary(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"), manifest, step_results)

    for step in steps
        rec = run_step!(root, run_dir, step, manifest, step_results; dry_run = dry_run)
        if rec["status"] == "failed"
            error("Step failed: $(rec["name"]). See manifest in $(joinpath(run_dir, "manifests", "run_manifest.toml"))")
        end
    end

    println("Pipeline complete")
    println("  Run directory: $run_dir")
    println("  Manifest: $(joinpath(run_dir, "manifests", "run_manifest.toml"))")
    println("  Summary: $(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
