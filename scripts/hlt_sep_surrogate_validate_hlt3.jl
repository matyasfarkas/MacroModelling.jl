#!/usr/bin/env julia
using Dates
using Serialization
import TOML
using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))

function repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function parse_mode(args)
    m = parse_arg_symbol(args, "--mode", :smoke)
    m in (:smoke, :benchmark) || error("Unknown --mode=$m. Use :smoke or :benchmark.")
    return m
end

function parse_benchmark_profile(args)
    p = parse_arg_symbol(args, "--benchmark-profile", :bounded)
    p in (:bounded, :full) || error("Unknown --benchmark-profile=$p. Use :bounded or :full.")
    return p
end

function git_commit(root::String)
    try
        return readchomp(`git -C $root rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function build_julia_cmd(root::String, script_path::String, args::Vector{String})
    base = Base.julia_cmd()
    return `$base --project=$root $script_path $args`
end

function cmd_to_string(cmd::Cmd)
    return sprint(show, cmd)
end

function write_toml(path::String, payload::Dict)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, payload)
    end
end

function write_summary(path::String, manifest::Dict, steps::Vector{Dict{String,Any}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT 3-Parameter Validation Run")
        println(io)
        println(io, "- Mode: `", manifest["mode"], "`")
        println(io, "- Dry run: `", manifest["dry_run"], "`")
        println(io, "- Repo root: `", manifest["repo_root"], "`")
        println(io, "- Git commit: `", manifest["git_commit"], "`")
        println(io, "- Run dir: `", manifest["run_dir"], "`")
        println(io, "- Created: `", manifest["created_at"], "`")
        println(io)
        println(io, "## Steps")
        for step in steps
            println(io, "- `", step["name"], "`: `", get(step, "status", "planned"), "`")
            println(io, "  - Script: `", step["script"], "`")
            println(io, "  - Command: `", step["command"], "`")
            if haskey(step, "elapsed_s")
                println(io, "  - Elapsed (s): `", step["elapsed_s"], "`")
            end
            if get(step, "name", "") == "fom_benchmark"
                if haskey(step, "fom_benchmark_preset")
                    println(io, "  - FOM preset: `", step["fom_benchmark_preset"], "`")
                end
                if haskey(step, "fom_direct_sep_ok_count")
                    println(io, "  - Direct SEP ok count: `", step["fom_direct_sep_ok_count"], "`")
                end
                if haskey(step, "fom_direct_sep_recovery_count")
                    println(io, "  - Direct SEP recovery count: `", step["fom_direct_sep_recovery_count"], "`")
                end
                if haskey(step, "fom_result_statuses")
                    println(io, "  - Result statuses: `", step["fom_result_statuses"], "`")
                end
            elseif get(step, "name", "") == "acceptance_smoke"
                if haskey(step, "acceptance_smoke_status")
                    println(io, "  - Acceptance status: `", step["acceptance_smoke_status"], "`")
                end
                if haskey(step, "acceptance_smoke_gate_share")
                    println(io, "  - Gate share: `", step["acceptance_smoke_gate_share"], "`")
                end
                if haskey(step, "acceptance_smoke_gate_vol_overlap_count")
                    println(io, "  - Gate/vol overlap count: `", step["acceptance_smoke_gate_vol_overlap_count"], "`")
                end
                if haskey(step, "acceptance_smoke_theta_abs_error")
                    println(io, "  - Theta abs error: `", step["acceptance_smoke_theta_abs_error"], "`")
                end
                if haskey(step, "acceptance_smoke_rmse_gain_rom1_minus_sep_fit_region")
                    println(io, "  - RMSE gain (ROM1-SEP): `", step["acceptance_smoke_rmse_gain_rom1_minus_sep_fit_region"], "`")
                end
                if haskey(step, "acceptance_smoke_direct_sep_better_than_rom1_fit_region")
                    println(io, "  - Direct SEP better (truth-shock fit): `", step["acceptance_smoke_direct_sep_better_than_rom1_fit_region"], "`")
                end
            end
            if haskey(step, "error")
                println(io, "  - Error: `", replace(String(step["error"]), "\n" => " "), "`")
            end
        end
    end
end

function get_dict_value(d::AbstractDict, key::String)
    if haskey(d, key)
        return d[key]
    elseif haskey(d, Symbol(key))
        return d[Symbol(key)]
    else
        return nothing
    end
end

function validate_dataset_nonempty(dataset_path::String)
    isfile(dataset_path) || error("Dataset output not found: $dataset_path")
    payload = MacroModelling.load_hlt_dataset_payload(dataset_path)
    X = get_dict_value(payload, "X")
    X === nothing && return
    if X isa AbstractMatrix && size(X, 2) == 0
        error("Dataset contains zero samples (X has 0 columns): $dataset_path")
    end
end

function validate_synthetic_output(synthetic_path::String)
    isfile(synthetic_path) || error("Synthetic output not found: $synthetic_path")
    payload = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
    obs = get_dict_value(payload, "obs_data")
    obs === nothing && return
    obs isa AbstractMatrix || error("Synthetic obs_data is not a matrix: $(typeof(obs))")
    size(obs, 2) > 0 || error("Synthetic obs_data has zero periods: $synthetic_path")
    all(isfinite, obs) || error("Synthetic obs_data contains non-finite values: $synthetic_path")
    sample_idx = get_dict_value(payload, "sample_idx")
    if sample_idx isa AbstractVector
        size(obs, 2) == length(sample_idx) || error("Synthetic obs_data/sample_idx mismatch: obs T=$(size(obs,2)) vs sample_idx=$(length(sample_idx))")
    end
    generated_periods = get_dict_value(payload, "generated_periods")
    if generated_periods isa Integer
        generated_periods == size(obs, 2) || error("Synthetic generated_periods mismatch: generated=$generated_periods vs obs T=$(size(obs,2))")
    end
end

function validate_gate_calibration(gate_path::String)
    isfile(gate_path) || error("Gate calibration output not found: $gate_path")
    payload = MacroModelling.load_hlt_gate_calibration(gate_path)
    share = get_dict_value(payload, "achieved_share")
    share isa Real || return
    isfinite(share) || error("Gate achieved_share is non-finite: $gate_path")
    (0 < share < 1) || error("Gate achieved_share is degenerate ($share). Adjust gate calibration or padding settings.")
    target_reachable = get_dict_value(payload, "target_reachable")
    fail_unreachable = get_dict_value(payload, "fail_unreachable")
    if target_reachable === false && fail_unreachable !== true
        error("Gate target was unreachable but calibration did not fail (target_reachable=false): $gate_path")
    end
end

function validate_benchmark_reuse_inputs(run_dir::String; require_chain::Bool=false)
    required = (
        joinpath(run_dir, "dataset", "hlt_sep_surrogate_trained.jls"),
        joinpath(run_dir, "synthetic", "hlt_sep_synth_data.jls"),
        joinpath(run_dir, "synthetic", "gate_calibration.jls"),
    )
    for p in required
        isfile(p) || error("Missing required benchmark input for --benchmark-skip-build=true: $p")
    end
    if require_chain
        chain_path = joinpath(run_dir, "synthetic", "hlt_sep_surrogate_estimation_chain.jls")
        isfile(chain_path) || error("Missing required benchmark chain for --benchmark-skip-estimation=true: $chain_path")
    end
    return nothing
end

function validate_fom_benchmark_output(fom_path::String)
    isfile(fom_path) || error("FOM benchmark output not found: $fom_path")
    payload = MacroModelling.load_hlt_fom_benchmark_payload(fom_path)

    results = get_dict_value(payload, "results")
    results isa AbstractDict || error("FOM payload missing results dict: $fom_path")
    isempty(results) && error("FOM payload has empty results dict: $fom_path")

    statuses = Dict{String,String}()
    direct_sep_ok_labels = String[]
    direct_sep_recovery_labels = String[]
    direct_sep_floor_failures = Dict{String,Any}()
    for (label_raw, entry_raw) in results
        label = String(label_raw)
        entry_raw isa AbstractDict || error("FOM result entry for '$label' is not a dict: $(typeof(entry_raw))")
        status = get_dict_value(entry_raw, "status")
        status isa AbstractString || error("FOM result entry for '$label' missing string status.")
        statuses[label] = String(status)

        alg_eff = get_dict_value(entry_raw, "algorithm_effective")
        is_direct_sep = alg_eff isa AbstractString && String(alg_eff) == "stochastic_extended_path"
        if is_direct_sep && String(status) == "ok"
            push!(direct_sep_ok_labels, label)
            rung_used = get_dict_value(entry_raw, "recovery_rung_used")
            attempted = get_dict_value(entry_raw, "recovery_ladder_attempted")
            if (attempted === true) || (rung_used isa AbstractString && !isempty(String(rung_used)))
                push!(direct_sep_recovery_labels, label)
            end
        elseif is_direct_sep && String(status) == "on_failure_loglikelihood"
            direct_sep_floor_failures[label] = Dict(
                "failure_class" => get_dict_value(entry_raw, "sep_floor_failure_class"),
                "recovery_rung_used" => get_dict_value(entry_raw, "recovery_rung_used"),
                "recovery_ladder_attempted" => get_dict_value(entry_raw, "recovery_ladder_attempted"),
            )
        end
    end
    failures = get_dict_value(payload, "failures")
    failures_count = failures isa AbstractVector ? length(failures) : 0

    return Dict{String,Any}(
        "fom_algorithm_requested" => string(get_dict_value(payload, "algorithm_requested")),
        "fom_algorithm_effective" => string(get_dict_value(payload, "algorithm_effective")),
        "fom_benchmark_preset" => get_dict_value(payload, "benchmark_preset"),
        "fom_algorithm_supported" => get_dict_value(payload, "algorithm_supported"),
        "fom_algorithm_fallback_used" => get_dict_value(payload, "algorithm_fallback_used"),
        "fom_period_selection" => get_dict_value(payload, "period_selection"),
        "fom_gated_block_strategy" => get_dict_value(payload, "gated_block_strategy"),
        "fom_selected_periods_count" => get_dict_value(payload, "selected_periods_count"),
        "fom_evaluation_periods_count" => get_dict_value(payload, "evaluation_periods_count"),
        "fom_context_periods_effective" => get_dict_value(payload, "context_periods_effective"),
        "fom_benchmark_is_subset" => get_dict_value(payload, "benchmark_is_subset"),
        "fom_recovery_ladder_enabled" => get_dict_value(payload, "recovery_ladder_enabled"),
        "fom_recovery_ladder_policy" => get_dict_value(payload, "recovery_ladder_policy"),
        "fom_chain_summary_used" => get_dict_value(payload, "chain_summary_used"),
        "fom_script_elapsed_s" => get_dict_value(payload, "script_elapsed_s"),
        "fom_direct_sep_ok_labels" => sort!(unique(direct_sep_ok_labels)),
        "fom_direct_sep_ok_count" => length(unique(direct_sep_ok_labels)),
        "fom_direct_sep_recovery_labels" => sort!(unique(direct_sep_recovery_labels)),
        "fom_direct_sep_recovery_count" => length(unique(direct_sep_recovery_labels)),
        "fom_direct_sep_floor_failures" => direct_sep_floor_failures,
        "fom_result_statuses" => statuses,
        "fom_failures_count" => failures_count,
    )
end

function parse_f64_csv_arg(args::Vector{String}, key::String, default::Vector{Float64})
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return copy(default)
    vals = Float64[]
    for token in split(raw, ",")
        t = strip(token)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    isempty(vals) && return copy(default)
    return vals
end

function validate_acceptance_smoke_output(result_toml_path::String)
    isfile(result_toml_path) || error("Acceptance smoke output not found: $result_toml_path")
    payload = TOML.parsefile(result_toml_path)

    status = get(payload, "status", nothing)
    status isa AbstractString || error("Acceptance smoke payload missing string status: $result_toml_path")

    metrics = Dict{String,Any}(
        "acceptance_smoke_status" => String(status),
    )

    if haskey(payload, "switching") && payload["switching"] isa AbstractDict
        sw = payload["switching"]
        metrics["acceptance_smoke_gate_share"] = get(sw, "gate_share", nothing)
        metrics["acceptance_smoke_gate_vol_overlap_count"] = get(sw, "gate_vol_overlap_count", nothing)
    end

    if haskey(payload, "recovery") && payload["recovery"] isa AbstractDict
        rec = payload["recovery"]
        metrics["acceptance_smoke_theta_abs_error"] = get(rec, "theta_abs_error", nothing)
        metrics["acceptance_smoke_theta_recovery_pass"] = get(rec, "theta_recovery_pass", nothing)
    end

    if haskey(payload, "truth_shock_fit") && payload["truth_shock_fit"] isa AbstractDict
        fit = payload["truth_shock_fit"]
        metrics["acceptance_smoke_truth_shock_fit_status"] = get(fit, "status", "ok")
        metrics["acceptance_smoke_direct_sep_better_than_rom1_fit_region"] = get(fit, "direct_sep_better_than_rom1_fit_region", nothing)
        metrics["acceptance_smoke_rmse_sep_fit_region"] = get(fit, "rmse_sep_fit_region", nothing)
        metrics["acceptance_smoke_rmse_rom1_fit_region"] = get(fit, "rmse_rom1_fit_region", nothing)
        gain = nothing
        if haskey(fit, "rmse_gain_rom1_minus_sep_fit_region")
            gain = fit["rmse_gain_rom1_minus_sep_fit_region"]
        elseif (get(fit, "rmse_sep_fit_region", nothing) isa Real) && (get(fit, "rmse_rom1_fit_region", nothing) isa Real)
            gain = fit["rmse_rom1_fit_region"] - fit["rmse_sep_fit_region"]
        end
        metrics["acceptance_smoke_rmse_gain_rom1_minus_sep_fit_region"] = gain
    end

    return metrics
end

function acceptance_smoke_step(run_dir::String;
                               theta_tols::Vector{Float64},
                               direct_fit_margin::Float64,
                               run_inversion_benchmark_panel::Bool)
    out_toml = joinpath(run_dir, "acceptance_smoke", "hlt_acceptance_smoke_result.toml")
    out_summary = joinpath(run_dir, "acceptance_smoke", "hlt_acceptance_smoke_summary.md")
    theta_tols_csv = join(string.(theta_tols), ",")
    return Dict(
        "name" => "acceptance_smoke",
        "script" => "scripts/hlt_sep_surrogate_acceptance_smoke.jl",
        "outputs" => Dict(
            "acceptance_smoke_path" => out_toml,
            "acceptance_smoke_summary_path" => out_summary,
        ),
        "args" => String[
            run_dir,
            "--out=$(out_toml)",
            "--summary=$(out_summary)",
            "--quiet=true",
            "--run-inversion-benchmark-panel=$(run_inversion_benchmark_panel)",
            "--min-truthshock-direct-improvement=$(direct_fit_margin)",
            "--theta-tols=$(theta_tols_csv)",
        ],
    )
end

function smoke_defaults(run_dir::String; samples::Int, chains::Int, seed::Int, use_obc::Bool)
    dataset_dir = joinpath(run_dir, "dataset")
    synthetic_dir = joinpath(run_dir, "synthetic")
    surrogate_path = joinpath(dataset_dir, "hlt_sep_surrogate_trained.jls")
    synthetic_path = joinpath(synthetic_dir, "hlt_sep_synth_data.jls")
    gate_path = joinpath(synthetic_dir, "gate_calibration.jls")
    chain_path = joinpath(synthetic_dir, "hlt_sep_surrogate_estimation_chain.jls")
    fom_path = joinpath(synthetic_dir, "hlt_sep_fom_benchmark.jls")
    checkpoint_path = joinpath(synthetic_dir, "hlt_sep_surrogate_estimation_checkpoint.jls")

    obc_flag = use_obc ? "--use-obc" : "--no-obc"

    return [
        Dict(
            "name" => "dataset_generate",
            "script" => "scripts/hlt_sep_surrogate_dataset_generate.jl",
            "outputs" => Dict("dataset_dir" => dataset_dir, "dataset_path" => joinpath(dataset_dir, "hlt_sep_surrogate_dataset.jls")),
            "args" => String[
                "--param-set=legacy_3params",
                "--theta-sampling=grid",
                "--grid=3",
                "--sample-length=80",
                "--sep-horizon=20",
                "--sep-order=1",
                "--sep-nnodes=3",
                "--sep-maxit=150",
                "--sep-tol=1e-4",
                "--sep-accept-tol=0.5",
                "--sep-expectation-method=hmc",
                "--hmc-samples=200",
                "--hmc-warmup=100",
                "--hmc-leapfrog-steps=15",
                "--hmc-step-size=0.1",
                "--use-subdifferential=true",
                "--shock-scale=0.15",
                "--rom-orders=1,2",
                "--rom-mode=baseline",
                "--theta-attempts-per-theta=5",
                "--retry-on-early-failure=true",
                "--retry-shock-scale-backoff=0.8",
                "--stable-prefix",
                "--stable-min-periods=20",
                "--output-dir=$(dataset_dir)",
                "--seed=$(seed)",
                obc_flag,
            ],
        ),
        Dict(
            "name" => "surrogate_train",
            "script" => "scripts/hlt_sep_surrogate_train.jl",
            "outputs" => Dict("surrogate_path" => surrogate_path),
            "args" => String[
                joinpath(dataset_dir, "hlt_sep_surrogate_dataset.jls"),
                "--rom-residual=1",
                "--epochs=300",
                "--hidden=128",
                "--hidden2=64",
                "--seed=$(seed)",
                "--out=$(surrogate_path)",
            ],
        ),
        Dict(
            "name" => "synthetic_data_generate",
            "script" => "scripts/hlt_sep_surrogate_synthetic_data.jl",
            "outputs" => Dict("synthetic_dir" => synthetic_dir, "synthetic_path" => synthetic_path),
            "args" => String[
                "--sample-length=160",
                "--sep-horizon=20",
                "--sep-order=1",
                "--sep-nnodes=3",
                "--sep-maxit=150",
                "--sep-tol=1e-4",
                "--sep-accept-tol=0.5",
                "--sep-expectation-method=hmc",
                "--hmc-samples=200",
                "--hmc-warmup=100",
                "--hmc-leapfrog-steps=15",
                "--hmc-step-size=0.1",
                "--use-subdifferential=true",
                "--shock-scale=0.15",
                "--vol-start=55",
                "--vol-end=95",
                "--vol-mult=2.0",
                "--stable-prefix",
                "--stable-min-periods=80",
                "--attempts=3",
                "--retry-on-early-failure=true",
                "--retry-shock-scale-backoff=0.8",
                "--min-generated-periods=80",
                "--output-dir=$(synthetic_dir)",
                "--seed=$(seed+1)",
                obc_flag,
            ],
        ),
        Dict(
            "name" => "gate_calibration",
            "script" => "scripts/hlt_sep_surrogate_gate_calibration.jl",
            "outputs" => Dict("gate_path" => gate_path),
            "args" => String[
                synthetic_path,
                "--out=$(gate_path)",
                "--target-share=0.1",
                "--tau-eps=NaN",
                "--fail-unreachable=true",
                "--min-achieved-share=0.01",
                "--max-achieved-share=0.99",
                "--periods=1",
                "--shock-filter=kalman",
                obc_flag,
            ],
        ),
        Dict(
            "name" => "switching_estimation",
            "script" => "scripts/hlt_sep_surrogate_synthetic_estimation.jl",
            "outputs" => Dict("chain_path" => chain_path, "checkpoint_path" => checkpoint_path),
            "args" => String[
                surrogate_path,
                synthetic_path,
                "--out=$(chain_path)",
                "--gate-calibration=$(gate_path)",
                "--samples=$(samples)",
                "--chains=$(chains)",
                "--gate-mode=hard",
                "--gate-share-min=0.01",
                "--gate-share-max=0.99",
                "--fail-degenerate-gate=true",
                "--gate-k-pre=0",
                "--gate-k-post=0",
                "--gate-min-len=1",
                "--shock-filter=inversion",
                "--linear-filter=inversion",
                "--checkpoint-path=$(checkpoint_path)",
                obc_flag,
            ],
        ),
        Dict(
            "name" => "fom_benchmark",
            "script" => "scripts/hlt_sep_surrogate_fom_benchmark.jl",
            "outputs" => Dict("fom_benchmark_path" => fom_path),
            "args" => String[
                chain_path,
                synthetic_path,
                "--out=$(fom_path)",
                "--filter=inversion",
                "--allow-fail=true",
                obc_flag,
            ],
        ),
    ]
end

function benchmark_defaults(run_dir::String; samples::Int, chains::Int, seed::Int, use_obc::Bool, profile::Symbol=:bounded)
    steps = smoke_defaults(run_dir; samples = max(samples, 1), chains = max(chains, 1), seed = seed, use_obc = use_obc)
    dataset_dir = joinpath(run_dir, "dataset")
    synthetic_dir = joinpath(run_dir, "synthetic")
    chain_path = joinpath(synthetic_dir, "hlt_sep_surrogate_estimation_chain.jls")
    synthetic_path = joinpath(synthetic_dir, "hlt_sep_synth_data.jls")
    fom_path = joinpath(synthetic_dir, "hlt_sep_fom_benchmark.jls")
    profile == :bounded || profile == :full || error("Unsupported benchmark profile: $profile")

    dataset_args = profile == :bounded ? String[
        "--param-set=legacy_3params",
        "--theta-sampling=grid",
        "--grid=2",
        "--cprobp-min=0.45",
        "--cprobp-max=0.75",
        "--cindp-min=0.30",
        "--cindp-max=0.65",
        "--curvp-min=6.0",
        "--curvp-max=18.0",
        "--sample-length=40",
        "--burn-in=16",
        "--sep-horizon=10",
        "--sep-order=1",
        "--sep-nnodes=3",
        "--sep-maxit=80",
        "--sep-tol=1e-5",
        "--sep-accept-tol=0.35",
        "--shock-scale=0.10",
        "--rom-orders=1",
        "--rom-mode=baseline",
        "--stable-prefix",
        "--stable-min-periods=20",
        "--theta-attempts-per-theta=3",
        "--retry-on-early-failure=true",
        "--retry-shock-scale-backoff=0.7",
        "--output-dir=$(dataset_dir)",
        "--checkpoint-every=1",
        "--seed=$(seed)",
        (use_obc ? "--use-obc" : "--no-obc"),
    ] : String[
        "--param-set=legacy_3params",
        "--theta-sampling=prior",
        "--theta-samples=36",
        "--sample-length=160",
        "--sep-horizon=12",
        "--sep-order=1",
        "--sep-nnodes=3",
        "--sep-maxit=120",
        "--sep-tol=1e-5",
        "--sep-accept-tol=0.25",
        "--shock-scale=0.25",
        "--rom-orders=1,2",
        "--rom-mode=baseline",
        "--output-dir=$(dataset_dir)",
        "--checkpoint-every=5",
        "--seed=$(seed)",
        (use_obc ? "--use-obc" : "--no-obc"),
    ]

    train_args = profile == :bounded ? String[
        joinpath(dataset_dir, "hlt_sep_surrogate_dataset.jls"),
        "--rom-residual=1",
        "--epochs=150",
        "--hidden=128",
        "--hidden2=64",
        "--seed=$(seed)",
        "--out=$(joinpath(dataset_dir, "hlt_sep_surrogate_trained.jls"))",
    ] : String[
        joinpath(dataset_dir, "hlt_sep_surrogate_dataset.jls"),
        "--rom-residual=1",
        "--epochs=600",
        "--hidden=256",
        "--hidden2=128",
        "--seed=$(seed)",
        "--out=$(joinpath(dataset_dir, "hlt_sep_surrogate_trained.jls"))",
    ]

    synthetic_args = profile == :bounded ? String[
        "--sample-length=80",
        "--burn-in=16",
        "--sep-horizon=10",
        "--sep-order=1",
        "--sep-nnodes=3",
        "--sep-maxit=80",
        "--sep-tol=1e-5",
        "--sep-accept-tol=0.35",
        "--shock-scale=0.10",
        "--vol-start=24",
        "--vol-end=56",
        "--vol-mult=2.0",
        "--stable-prefix",
        "--stable-min-periods=40",
        "--attempts=5",
        "--retry-on-early-failure=true",
        "--retry-shock-scale-backoff=0.7",
        "--min-generated-periods=40",
        "--output-dir=$(synthetic_dir)",
        "--seed=$(seed+1)",
        (use_obc ? "--use-obc" : "--no-obc"),
    ] : String[
        "--sample-length=240",
        "--sep-horizon=12",
        "--sep-order=1",
        "--sep-nnodes=3",
        "--sep-maxit=120",
        "--sep-tol=1e-5",
        "--sep-accept-tol=0.25",
        "--shock-scale=0.25",
        "--vol-start=70",
        "--vol-end=130",
        "--vol-mult=2.5",
        "--output-dir=$(synthetic_dir)",
        "--seed=$(seed+1)",
        (use_obc ? "--use-obc" : "--no-obc"),
    ]

    fom_args = profile == :bounded ? String[
        chain_path,
        synthetic_path,
        "--out=$(fom_path)",
        "--benchmark-preset=direct_sep_gated_smoke_order1_tuned",
        "--allow-fail=true",
        "--recovery-ladder=true",
        "--labels=true",
        "--fallback-algorithm=first_order",
        (use_obc ? "--use-obc" : "--no-obc"),
    ] : nothing

    for step in steps
        if step["name"] == "dataset_generate"
            step["args"] = dataset_args
        elseif step["name"] == "surrogate_train"
            step["args"] = train_args
        elseif step["name"] == "synthetic_data_generate"
            step["args"] = synthetic_args
        elseif step["name"] == "switching_estimation"
            args = String.(step["args"])
            # overwrite samples/chains if explicit call provided
            step["args"] = [a for a in args if !(startswith(a, "--samples=") || startswith(a, "--chains="))]
            push!(step["args"], "--samples=$(max(samples, 1))")
            push!(step["args"], "--chains=$(max(chains, 1))")
        elseif step["name"] == "fom_benchmark" && profile == :bounded
            step["args"] = fom_args
        end
    end
    return steps
end

function apply_quick_smoke_overrides!(steps::Vector{Dict{String,Any}}, run_dir::String;
                                      seed::Int, samples::Int, chains::Int, use_obc::Bool,
                                      quick_smoke_fom_preset::String = "direct_sep_gated_smoke")
    obc_flag = use_obc ? "--use-obc" : "--no-obc"

    dataset_dir = joinpath(run_dir, "dataset")
    synthetic_dir = joinpath(run_dir, "synthetic")
    surrogate_path = joinpath(dataset_dir, "hlt_sep_surrogate_trained.jls")
    synthetic_path = joinpath(synthetic_dir, "hlt_sep_synth_data.jls")
    gate_path = joinpath(synthetic_dir, "gate_calibration.jls")
    chain_path = joinpath(synthetic_dir, "hlt_sep_surrogate_estimation_chain.jls")
    checkpoint_path = joinpath(synthetic_dir, "hlt_sep_surrogate_estimation_checkpoint.jls")
    fom_path = joinpath(synthetic_dir, "hlt_sep_fom_benchmark.jls")

    for step in steps
        name = String(step["name"])
        if name == "dataset_generate"
            step["args"] = String[
                "--param-set=legacy_3params",
                "--theta-sampling=grid",
                "--grid=1",
                "--cprobp-min=0.60",
                "--cprobp-max=0.60",
                "--cindp-min=0.47",
                "--cindp-max=0.47",
                "--curvp-min=10.0",
                "--curvp-max=10.0",
                "--samples-per-theta=8",
                "--burn-in=0",
                "--sample-length=8",
                "--sep-horizon=12",
                "--sep-order=1",
                "--sep-nnodes=3",
                "--sep-maxit=100",
                "--sep-tol=1e-6",
                "--sep-shock-scale=0.5",
                "--sep-accept-tol=0.25",
                "--shock-scale=0.05",
                "--rom-orders=1",
                "--rom-mode=baseline",
                "--checkpoint-every=1",
                "--stable-prefix",
                "--stable-min-periods=6",
                "--theta-attempts-per-theta=5",
                "--retry-on-early-failure=true",
                "--retry-shock-scale-backoff=0.7",
                "--output-dir=$(dataset_dir)",
                "--seed=$(seed)",
                obc_flag,
            ]
        elseif name == "surrogate_train"
            step["args"] = String[
                joinpath(dataset_dir, "hlt_sep_surrogate_dataset.jls"),
                "--rom-residual=1",
                "--epochs=80",
                "--hidden=64",
                "--hidden2=32",
                "--seed=$(seed)",
                "--out=$(surrogate_path)",
            ]
        elseif name == "synthetic_data_generate"
            step["args"] = String[
                "--burn-in=0",
                "--sample-length=8",
                "--sep-horizon=12",
                "--sep-order=1",
                "--sep-nnodes=3",
                "--sep-maxit=100",
                "--sep-tol=1e-6",
                "--sep-shock-scale=0.5",
                "--sep-accept-tol=0.25",
                "--shock-scale=0.03",
                "--vol-start=3",
                "--vol-end=6",
                "--vol-mult=1.2",
                "--stable-prefix",
                "--stable-min-periods=4",
                "--attempts=5",
                "--retry-on-early-failure=true",
                "--retry-shock-scale-backoff=0.7",
                "--min-generated-periods=4",
                "--output-dir=$(synthetic_dir)",
                "--seed=$(seed + 1)",
                obc_flag,
            ]
        elseif name == "gate_calibration"
            step["args"] = String[
                synthetic_path,
                "--out=$(gate_path)",
                "--target-share=0.1",
                "--tau-eps=NaN",
                "--fail-unreachable=true",
                "--min-achieved-share=0.05",
                "--max-achieved-share=0.875",
                "--max-target-share-error=0.5",
                "--periods=1",
                "--shock-filter=kalman",
                obc_flag,
            ]
        elseif name == "switching_estimation"
            step["args"] = String[
                surrogate_path,
                synthetic_path,
                "--out=$(chain_path)",
                "--gate-calibration=$(gate_path)",
                "--samples=$(min(samples, 20))",
                "--gate-k-pre=0",
                "--gate-k-post=0",
                "--gate-min-len=1",
                "--chains=$(min(chains, 1))",
                "--gate-mode=hard",
                "--gate-share-min=0.05",
                "--gate-share-max=0.95",
                "--fail-degenerate-gate=true",
                "--shock-filter=inversion",
                "--linear-filter=inversion",
                "--checkpoint-path=$(checkpoint_path)",
                obc_flag,
            ]
        elseif name == "fom_benchmark"
            step["args"] = String[
                chain_path,
                synthetic_path,
                "--out=$(fom_path)",
                "--benchmark-preset=$(quick_smoke_fom_preset)",
                "--allow-fail=true",
                "--recovery-ladder=true",
                "--labels=true",
                "--fallback-algorithm=first_order",
                obc_flag,
            ]
        end
    end
    return steps
end

function main()
    root = repo_root()
    mode = parse_mode(ARGS)
    dry_run = parse_arg_bool(ARGS, "--dry-run", false)
    quick_smoke = parse_arg_bool(ARGS, "--quick-smoke", false)
    skip_fom = parse_arg_bool(ARGS, "--skip-fom", false)
    require_direct_fom_ok = parse_arg_bool(ARGS, "--require-direct-fom-ok", false)
    run_acceptance_smoke = parse_arg_bool(ARGS, "--run-acceptance-smoke", false)
    require_acceptance_smoke_ok = parse_arg_bool(ARGS, "--require-acceptance-smoke-ok", false)
    benchmark_profile = parse_benchmark_profile(ARGS)
    benchmark_skip_build = parse_arg_bool(ARGS, "--benchmark-skip-build", false)
    benchmark_skip_estimation = parse_arg_bool(ARGS, "--benchmark-skip-estimation", false)
    acceptance_smoke_run_inversion_benchmark_panel = parse_arg_bool(ARGS, "--acceptance-smoke-run-inversion-benchmark-panel", false)
    acceptance_smoke_direct_fit_margin = parse_arg_float(ARGS, "--acceptance-smoke-direct-fit-margin", 0.0)
    acceptance_smoke_theta_tols = parse_f64_csv_arg(ARGS, "--acceptance-smoke-theta-tols", [0.10, 0.15, 20.0])
    quick_smoke_fom_preset = parse_arg_string(ARGS, "--quick-smoke-fom-preset", "direct_sep_gated_smoke")
    artifact_root = parse_arg_string(ARGS, "--artifact-root", joinpath(root, ".local_artifacts", "hlt_validation_runs"))
    run_dir_arg = parse_arg_string(ARGS, "--run-dir", "")
    seed = parse_arg_int(ARGS, "--seed", 42)
    samples = parse_arg_int(ARGS, "--samples", mode == :smoke ? 150 : 1000)
    chains = parse_arg_int(ARGS, "--chains", mode == :smoke ? 1 : 4)
    use_obc = parse_arg_bool(ARGS, "--use-obc", true)

    run_dir = run_dir_arg == "" ? joinpath(artifact_root, "hlt3_$(Dates.format(now(), "yyyymmdd_HHMMSS"))") : run_dir_arg
    mkpath(run_dir)
    mkpath(joinpath(run_dir, "manifests"))

    steps = mode == :smoke ?
        smoke_defaults(run_dir; samples = samples, chains = chains, seed = seed, use_obc = use_obc) :
        benchmark_defaults(run_dir; samples = samples, chains = chains, seed = seed, use_obc = use_obc, profile = benchmark_profile)

    if quick_smoke
        mode == :smoke || error("--quick-smoke is only supported with --mode=smoke.")
        apply_quick_smoke_overrides!(steps, run_dir;
                                     seed = seed,
                                     samples = samples,
                                     chains = chains,
                                     use_obc = use_obc,
                                     quick_smoke_fom_preset = quick_smoke_fom_preset)
    end

    if skip_fom
        steps = [s for s in steps if s["name"] != "fom_benchmark"]
    end

    if mode == :benchmark && benchmark_skip_build
        steps = [s for s in steps if !(String(s["name"]) in ("dataset_generate", "surrogate_train", "synthetic_data_generate", "gate_calibration"))]
        if !dry_run
            validate_benchmark_reuse_inputs(run_dir; require_chain = benchmark_skip_estimation)
        end
    end

    if mode == :benchmark && benchmark_skip_estimation
        steps = [s for s in steps if String(s["name"]) != "switching_estimation"]
        if !dry_run
            validate_benchmark_reuse_inputs(run_dir; require_chain = true)
        end
    end

    if run_acceptance_smoke
        mode == :smoke || error("--run-acceptance-smoke is only supported with --mode=smoke.")
        push!(steps, acceptance_smoke_step(run_dir;
                                           theta_tols = acceptance_smoke_theta_tols,
                                           direct_fit_margin = acceptance_smoke_direct_fit_margin,
                                           run_inversion_benchmark_panel = acceptance_smoke_run_inversion_benchmark_panel))
    end

    manifest = Dict{String,Any}(
        "name" => "HLT switching validation (3param)",
        "mode" => String(mode),
        "dry_run" => dry_run,
        "quick_smoke" => quick_smoke,
        "require_direct_fom_ok" => require_direct_fom_ok,
        "run_acceptance_smoke" => run_acceptance_smoke,
        "require_acceptance_smoke_ok" => require_acceptance_smoke_ok,
        "acceptance_smoke_run_inversion_benchmark_panel" => acceptance_smoke_run_inversion_benchmark_panel,
        "acceptance_smoke_direct_fit_margin" => acceptance_smoke_direct_fit_margin,
        "acceptance_smoke_theta_tols" => acceptance_smoke_theta_tols,
        "quick_smoke_fom_preset" => quick_smoke_fom_preset,
        "benchmark_profile" => String(benchmark_profile),
        "benchmark_skip_build" => benchmark_skip_build,
        "benchmark_skip_estimation" => benchmark_skip_estimation,
        "repo_root" => root,
        "git_commit" => git_commit(root),
        "run_dir" => run_dir,
        "created_at" => string(Dates.now()),
        "seed" => seed,
        "samples" => samples,
        "chains" => chains,
        "use_obc" => use_obc,
        "steps" => Dict{String,Any}(),
    )

    step_results = Dict{String,Any}[]
    for step in steps
        script_rel = String(step["script"])
        args = String.(step["args"])
        cmd = build_julia_cmd(root, joinpath(root, script_rel), args)
        rec = Dict{String,Any}(
            "name" => String(step["name"]),
            "script" => script_rel,
            "command" => cmd_to_string(cmd),
            "status" => dry_run ? "planned" : "pending",
        )
        for (k, v) in step["outputs"]
            rec[String(k)] = String(v)
        end

        manifest["steps"][rec["name"]] = Dict(k => v for (k, v) in rec if k != "name")
        write_toml(joinpath(run_dir, "manifests", "run_manifest.toml"), manifest)
        write_summary(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"), manifest, vcat(step_results, [rec]))

        if !dry_run
            t0 = time()
            try
                run(cmd)
                if rec["name"] == "dataset_generate" && haskey(rec, "dataset_path")
                    validate_dataset_nonempty(String(rec["dataset_path"]))
                elseif rec["name"] == "synthetic_data_generate" && haskey(rec, "synthetic_path")
                    validate_synthetic_output(String(rec["synthetic_path"]))
                elseif rec["name"] == "gate_calibration" && haskey(rec, "gate_path")
                    validate_gate_calibration(String(rec["gate_path"]))
                elseif rec["name"] == "fom_benchmark" && haskey(rec, "fom_benchmark_path")
                    fom_meta = validate_fom_benchmark_output(String(rec["fom_benchmark_path"]))
                    for (k, v) in fom_meta
                        rec[String(k)] = v
                    end
                    if require_direct_fom_ok
                        direct_ok_count = get(rec, "fom_direct_sep_ok_count", 0)
                        if !(direct_ok_count isa Integer && direct_ok_count >= 1)
                            error("FOM benchmark produced no successful direct SEP results (fom_direct_sep_ok_count=$(direct_ok_count)).")
                        end
                    end
                elseif rec["name"] == "acceptance_smoke" && haskey(rec, "acceptance_smoke_path")
                    acceptance_meta = validate_acceptance_smoke_output(String(rec["acceptance_smoke_path"]))
                    for (k, v) in acceptance_meta
                        rec[String(k)] = v
                    end
                    if require_acceptance_smoke_ok && get(rec, "acceptance_smoke_status", "") != "ok"
                        status_val = get(rec, "acceptance_smoke_status", missing)
                        error("Acceptance smoke status was not ok (acceptance_smoke_status=$(status_val)).")
                    end
                end
                rec["status"] = "ok"
            catch err
                rec["status"] = "failed"
                rec["error"] = sprint(showerror, err)
                rec["elapsed_s"] = round(time() - t0; digits = 3)
                push!(step_results, rec)
                manifest["steps"][rec["name"]] = Dict(k => v for (k, v) in rec if k != "name")
                write_toml(joinpath(run_dir, "manifests", "run_manifest.toml"), manifest)
                write_summary(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"), manifest, step_results)
                rethrow()
            end
            rec["elapsed_s"] = round(time() - t0; digits = 3)
        end

        push!(step_results, rec)
        manifest["steps"][rec["name"]] = Dict(k => v for (k, v) in rec if k != "name")
        write_toml(joinpath(run_dir, "manifests", "run_manifest.toml"), manifest)
        write_summary(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"), manifest, step_results)
    end

    println("HLT 3-parameter validation harness")
    println("  Mode: $(mode)")
    println("  Benchmark profile: $(benchmark_profile)")
    println("  Dry run: $(dry_run)")
    println("  Quick smoke: $(quick_smoke)")
    println("  Run dir: $(run_dir)")
    println("  Manifest: $(joinpath(run_dir, "manifests", "run_manifest.toml"))")
    println("  Summary: $(joinpath(run_dir, "manifests", "RUN_SUMMARY.md"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
