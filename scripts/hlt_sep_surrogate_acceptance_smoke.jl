#!/usr/bin/env julia
using Dates
using Serialization
using Statistics
using LinearAlgebra
import TOML
using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

function repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function build_julia_cmd(root::String, script_path::String, args::Vector{String})
    base = Base.julia_cmd()
    return `$base --project=$root $script_path $args`
end

function write_toml(path::String, payload::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, payload)
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

function first_positional_arg_str(args::Vector{String})
    arg = first_positional_arg(args)
    arg === nothing && error("Usage: julia hlt_sep_surrogate_acceptance_smoke.jl <run_dir> [--force-benchmarks=true]")
    return String(arg)
end

function parse_blocks(args::Vector{String})
    raw = strip(parse_arg_string(args, "--blocks", "first,last"))
    isempty(raw) && return Symbol[:first, :last]
    vals = Symbol[]
    for token in split(raw, ",")
        s = Symbol(lowercase(strip(token)))
        s in (:first, :last, :longest) || error("Unsupported block selection '$token'. Use first,last,longest.")
        push!(vals, s)
    end
    isempty(vals) && error("--blocks produced an empty list.")
    return unique(vals)
end

function parse_optional_arg_float(args::Vector{String}, key::String)
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return nothing
    lowered = lowercase(raw)
    lowered in ("none", "null", "nothing") && return nothing
    return parse(Float64, raw)
end

function parse_f64_vector_arg(args::Vector{String}, key::String, default::Vector{Float64})
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return copy(default)
    vals = Float64[]
    for token in split(raw, ",")
        t = strip(token)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    return vals
end

function arg_present(args::Vector{String}, key::String)
    any(startswith(arg, key * "=") for arg in args)
end

function append_tag_before_ext(path::AbstractString, tag::AbstractString)
    path_s = String(path)
    tag_s = String(tag)
    isempty(tag_s) && return path_s
    base = basename(path_s)
    dir = dirname(path_s)
    if occursin('.', base)
        stem, ext = splitext(base)
        return joinpath(dir, stem * "_" * tag_s * ext)
    end
    return joinpath(dir, base * "_" * tag_s)
end

function bool_vec(x)
    return Bool.(vec(x))
end

function f64_vec(x)
    return Float64.(collect(x))
end

function load_payload_dict(path::String)
    isfile(path) || error("Missing payload: $path")
    payload = deserialize(path)
    payload isa AbstractDict || error("Payload is not a Dict: $path")
    return payload
end

function maybe_string(x)
    x === nothing && return ""
    return String(x)
end

function int_vec_or_empty(x)
    x === nothing && return Int[]
    return Int.(collect(x))
end

function benchmark_paths(bench_dir::String, tag::String)
    out = joinpath(bench_dir, "hlt_acceptance_" * tag * ".jls")
    summary = joinpath(bench_dir, "hlt_acceptance_" * tag * "_summary.md")
    return out, summary
end

function default_lock_path(run_dir::String, run_id_tag::AbstractString)
    tag = String(run_id_tag)
    base = isempty(tag) ? ".lock" : ".lock_" * tag
    return joinpath(run_dir, "acceptance_smoke", base)
end

function acquire_lockfile(lock_path::String)
    mkpath(dirname(lock_path))
    if isfile(lock_path)
        contents = try
            strip(read(lock_path, String))
        catch
            "<unreadable>"
        end
        error("Acceptance smoke lock file already exists: $lock_path" *
              (isempty(contents) ? "" : " (contents: $contents)"))
    end
    open(lock_path, "w") do io
        println(io, "pid=", getpid())
        println(io, "timestamp=", now())
    end
    return lock_path
end

function release_lockfile(lock_path::String)
    isfile(lock_path) && rm(lock_path; force = true)
    return nothing
end

function run_cmd(cmd; quiet::Bool = false, timeout_seconds::Float64 = 0.0)
    if timeout_seconds <= 0
        if quiet
            run(pipeline(cmd; stdout = devnull))
        else
            run(cmd)
        end
        return nothing
    end

    proc = if quiet
        run(pipeline(cmd; stdout = devnull); wait = false)
    else
        run(cmd; wait = false)
    end
    deadline = time() + timeout_seconds
    while process_running(proc)
        time() > deadline && break
        sleep(0.05)
    end
    if process_running(proc)
        try
            kill(proc)
        catch
        end
        try
            wait(proc)
        catch
        end
        error("Benchmark subprocess timed out after $(timeout_seconds) seconds: $(sprint(show, cmd))")
    end
    success(proc) || error("Benchmark subprocess failed: $(sprint(show, cmd))")
    return nothing
end

function benchmark_step(root::String,
                        benchmark_script::String,
                        chain_path::String,
                        synthetic_path::String,
                        bench_dir::String;
                        preset::String,
                        block::Symbol,
                        force::Bool,
                        recovery_ladder::Bool = false,
                        reuse_existing_direct_first_path::Union{Nothing,String} = nothing,
                        dry_run::Bool = false,
                        quiet::Bool = false,
                        timeout_seconds::Float64 = 0.0)
    tag = (startswith(preset, "direct_sep") ? "direct" : "rom1") * "_" * String(block)
    out_path, summary_path = benchmark_paths(bench_dir, tag)
    reused_existing = false

    if !force && !isfile(out_path) && reuse_existing_direct_first_path !== nothing && preset == "direct_sep_gated_smoke_order1_tuned" && block == :first
        if isfile(reuse_existing_direct_first_path)
            src_payload = load_payload_dict(reuse_existing_direct_first_path)
            if maybe_string(get_dict_value(src_payload, "benchmark_preset")) == preset &&
               maybe_string(get_dict_value(src_payload, "period_selection")) == "gated_block" &&
               maybe_string(get_dict_value(src_payload, "gated_block_strategy")) == "first"
                mkpath(dirname(out_path))
                cp(reuse_existing_direct_first_path, out_path; force = true)
                derived_summary = replace(reuse_existing_direct_first_path, r"\.jls$" => "_summary.md")
                isfile(derived_summary) && cp(derived_summary, summary_path; force = true)
                reused_existing = true
            end
        end
    end

    cmd = build_julia_cmd(root, benchmark_script, String[
        chain_path,
        synthetic_path,
        "--out=$(out_path)",
        "--benchmark-preset=$(preset)",
        "--gated-block=$(String(block))",
        "--prefer-chain-summary=true",
        "--build-chain-summary-cache=true",
        "--allow-fail=true",
        "--recovery-ladder=$(recovery_ladder)",
    ])

    if !reused_existing && (force || !isfile(out_path))
        if dry_run
            return Dict{String,Any}(
                "tag" => tag,
                "command" => sprint(show, cmd),
                "out_path" => out_path,
                "summary_path" => summary_path,
                "reused_existing" => false,
                "ran" => false,
                "dry_run" => true,
            )
        end
        t0 = time()
        run_cmd(cmd; quiet = quiet, timeout_seconds = timeout_seconds)
        elapsed = time() - t0
        return Dict{String,Any}(
            "tag" => tag,
            "command" => sprint(show, cmd),
            "out_path" => out_path,
            "summary_path" => summary_path,
            "reused_existing" => false,
            "ran" => true,
            "elapsed_s" => elapsed,
        )
    end

    return Dict{String,Any}(
        "tag" => tag,
        "command" => sprint(show, cmd),
        "out_path" => out_path,
        "summary_path" => summary_path,
        "reused_existing" => reused_existing,
        "ran" => false,
        "dry_run" => dry_run,
        "cache_hit" => isfile(out_path),
    )
end

function parse_true_benchmark_result(path::String)
    payload = load_payload_dict(path)
    results = get_dict_value(payload, "results")
    results isa AbstractDict || error("Benchmark payload missing results dict: $path")
    true_entry = get_dict_value(results, "true")
    true_entry isa AbstractDict || error("Benchmark payload missing true result entry: $path")

    status = String(get_dict_value(true_entry, "status"))
    ll = get_dict_value(true_entry, "fom_loglik")
    ll_val = ll isa Real ? Float64(ll) : NaN
    return Dict{String,Any}(
        "path" => path,
        "status" => status,
        "fom_loglik" => ll_val,
        "algorithm_requested" => string(get_dict_value(payload, "algorithm_requested")),
        "algorithm_effective" => string(get_dict_value(true_entry, "algorithm_effective")),
        "benchmark_preset" => get_dict_value(payload, "benchmark_preset"),
        "selected_period_indices" => int_vec_or_empty(get_dict_value(payload, "selected_period_indices")),
        "evaluation_period_indices" => int_vec_or_empty(get_dict_value(payload, "evaluation_period_indices")),
        "context_period_indices" => int_vec_or_empty(get_dict_value(payload, "context_period_indices")),
        "benchmark_is_subset" => get_dict_value(payload, "benchmark_is_subset"),
        "recovery_ladder_enabled" => get_dict_value(true_entry, "recovery_ladder_enabled"),
        "recovery_ladder_attempted" => get_dict_value(true_entry, "recovery_ladder_attempted"),
        "recovery_rung_used" => get_dict_value(true_entry, "recovery_rung_used"),
        "attempts_count" => get_dict_value(true_entry, "attempts_count"),
        "sep_floor_failure_class" => get_dict_value(true_entry, "sep_floor_failure_class"),
    )
end

function extract_vector_field(d::AbstractDict, key::String; what::String = key)
    v = get_dict_value(d, key)
    v === nothing && error("Missing $(what) in payload.")
    return v
end

function switching_metrics(chain_summary::AbstractDict, synthetic::AbstractDict; min_overlap::Int = 1)
    gate_mask = bool_vec(extract_vector_field(chain_summary, "gate_mask"; what = "chain_summary.gate_mask"))
    isempty(gate_mask) && error("Gate mask is empty.")
    gate_share_val = get_dict_value(chain_summary, "gate_share")
    gate_share = gate_share_val isa Real ? Float64(gate_share_val) : mean(gate_mask)
    isfinite(gate_share) || error("Gate share is non-finite.")
    (0.0 < gate_share < 1.0) || error("Gate share is degenerate ($gate_share).")
    any(gate_mask) || error("Gate mask has no nonlinear periods.")
    any(.!gate_mask) || error("Gate mask has no linear periods.")

    vol_start = Int(get_dict_value(synthetic, "vol_start"))
    vol_end = Int(get_dict_value(synthetic, "vol_end"))
    vol_start <= vol_end || error("Invalid volatility window: vol_start=$vol_start > vol_end=$vol_end")
    length(gate_mask) >= vol_end || error("Gate mask length $(length(gate_mask)) is shorter than vol_end=$vol_end")
    gated_idx = findall(gate_mask)
    vol_window = collect(vol_start:vol_end)
    overlap = intersect(gated_idx, vol_window)
    overlap_count = length(overlap)
    overlap_count >= min_overlap || error("Gate does not overlap the synthetic volatility window enough (overlap=$overlap_count, required=$min_overlap).")

    return Dict{String,Any}(
        "gate_share" => gate_share,
        "gate_mask" => gate_mask,
        "gated_indices" => gated_idx,
        "vol_start" => vol_start,
        "vol_end" => vol_end,
        "vol_window_indices" => vol_window,
        "gate_vol_overlap_indices" => overlap,
        "gate_vol_overlap_count" => overlap_count,
    )
end

function recovery_metrics(chain_summary::AbstractDict; thresholds::Vector{Float64})
    theta_true = f64_vec(extract_vector_field(chain_summary, "theta_true"; what = "chain_summary.theta_true"))
    theta_est = f64_vec(extract_vector_field(chain_summary, "post_mean_theta"; what = "chain_summary.post_mean_theta"))
    length(theta_true) == length(theta_est) || error("theta_true length $(length(theta_true)) != post_mean_theta length $(length(theta_est))")
    length(theta_true) == length(thresholds) || error("theta threshold length $(length(thresholds)) must match theta length $(length(theta_true))")

    abs_err = abs.(theta_est .- theta_true)
    passes = abs_err .<= thresholds
    all(passes) || error("Posterior mean recovery failed thresholds. abs_err=$(abs_err), thresholds=$(thresholds)")

    return Dict{String,Any}(
        "theta_true" => theta_true,
        "theta_est" => theta_est,
        "theta_abs_error" => abs_err,
        "theta_error_thresholds" => thresholds,
        "theta_recovery_pass" => all(passes),
    )
end

function load_hlt_model_from_synthetic(root::String, synthetic::AbstractDict)
    model_name = String(get_dict_value(synthetic, "model"))
    return load_hlt_model(root, model_name; mod = @__MODULE__)
end

function params_from_theta(model, theta::AbstractVector, theta_names_raw)
    theta_names = Symbol.(collect(theta_names_raw))
    params = copy(model.parameter_values)
    isempty(theta_names) && return params
    idx = indexin(theta_names, model.parameters)
    any(isnothing, idx) && error("Some theta_names were not found in $(model.model_name) parameters.")
    params[Int.(idx)] .= theta
    return params
end

function synthetic_sample_shocks(synthetic::AbstractDict, T::Int)
    shocks = Matrix{Float64}(extract_vector_field(synthetic, "shocks"; what = "synthetic.shocks"))
    size(shocks, 2) >= T || error("Synthetic shocks have only $(size(shocks,2)) columns for T=$T.")
    shock_sigmas = get_dict_value(synthetic, "shock_sigmas")
    if shock_sigmas !== nothing
        sig = Vector{Float64}(collect(shock_sigmas))
        if length(sig) == size(shocks, 1)
            structural_idx = findall(sig .> 0)
            !isempty(structural_idx) || error("No structural shocks identified from synthetic.shock_sigmas.")
            return shocks[structural_idx, 1:T]
        end
    end
    return shocks[:, 1:T]
end

function truth_shock_fit_metrics(root::String, synthetic::AbstractDict, switching::AbstractDict; direct_fit_margin::Float64 = 0.0)
    _ = switching  # not used directly; switching validity is enforced separately
    mm_model = load_hlt_model_from_synthetic(root, synthetic)
    theta_true = f64_vec(extract_vector_field(synthetic, "theta_true"; what = "synthetic.theta_true"))
    theta_names = extract_vector_field(synthetic, "theta_names"; what = "synthetic.theta_names")
    params = params_from_theta(mm_model, theta_true, theta_names)
    MacroModelling.write_parameters_input!(mm_model, params, verbose = false)

    obs_data_ref = Matrix{Float64}(extract_vector_field(synthetic, "obs_data"; what = "synthetic.obs_data"))
    T_full = size(obs_data_ref, 2)
    T_full > 0 || error("Synthetic obs_data is empty.")
    all_struct_shocks = synthetic_sample_shocks(synthetic, T_full)

    observables = Symbol.(collect(extract_vector_field(synthetic, "observables"; what = "synthetic.observables")))
    obs_idx_raw = get_dict_value(synthetic, "obs_idx")
    state_idx_raw = get_dict_value(synthetic, "state_idx")
    if obs_idx_raw === nothing || state_idx_raw === nothing
        error("Synthetic payload missing obs_idx/state_idx required for truth-shock fit check.")
    end
    obs_idx = Int.(collect(obs_idx_raw))
    state_idx = Int.(collect(state_idx_raw))

    vol_start = Int(get_dict_value(synthetic, "vol_start"))
    vol_end = Int(get_dict_value(synthetic, "vol_end"))
    1 <= vol_start <= vol_end <= T_full || error("Invalid synthetic volatility window for truth-shock fit: $vol_start:$vol_end with T=$T_full")

    ctx = 1
    micro_start = max(1, vol_start - ctx)
    micro_end = vol_end
    micro_T = micro_end - micro_start + 1
    micro_T > 0 || error("Invalid microcase window: $micro_start:$micro_end")
    sample_shocks = copy(all_struct_shocks[:, micro_start:micro_end])
    fit_idx = collect((vol_start - micro_start + 1):(vol_end - micro_start + 1))

    shock_scaling_raw = get_dict_value(synthetic, "shock_scaling")
    shock_scaling = shock_scaling_raw isa Symbol ? shock_scaling_raw : Symbol(String(shock_scaling_raw))

    attempt_specs = [
        Dict{String,Any}(
            "name" => "order1_tuned_smoke",
            "sep_horizon" => max(4, micro_T),
            "sep_order" => 1,
            "sep_nnodes" => 3,
            "sep_maxit" => 20,
            "sep_tol" => 1e-4,
            "sep_sparse_tree" => true,
            "sep_accept_tol" => 1.0,
            "sep_shock_scale" => 0.5,
        ),
        Dict{String,Any}(
            "name" => "order0_fallback",
            "sep_horizon" => max(4, micro_T),
            "sep_order" => 0,
            "sep_nnodes" => 1,
            "sep_maxit" => 80,
            "sep_tol" => 1e-5,
            "sep_sparse_tree" => true,
            "sep_accept_tol" => 0.5,
            "sep_shock_scale" => 1.0,
        ),
        Dict{String,Any}(
            "name" => "order1_relaxed",
            "sep_horizon" => max(4, micro_T),
            "sep_order" => 1,
            "sep_nnodes" => 3,
            "sep_maxit" => 60,
            "sep_tol" => 1e-4,
            "sep_sparse_tree" => true,
            "sep_accept_tol" => 1.0,
            "sep_shock_scale" => 0.5,
        ),
    ]

    sep_res = nothing
    sim = nothing
    chosen_attempt = nothing
    attempt_logs = Dict{String,Any}[]
    for spec in attempt_specs
        try
            res_try = MacroModelling.simulate_sep_extended_path(
                mm_model;
                periods = micro_T,
                burn_in = 0,
                shocks = sample_shocks,
                sep_horizon = Int(spec["sep_horizon"]),
                sep_order = Int(spec["sep_order"]),
                sep_nnodes = Int(spec["sep_nnodes"]),
                sep_maxit = Int(spec["sep_maxit"]),
                sep_tol = Float64(spec["sep_tol"]),
                sep_sparse_tree = Bool(spec["sep_sparse_tree"]),
                sep_accept_tol = Float64(spec["sep_accept_tol"]),
                sep_shock_scale = Float64(spec["sep_shock_scale"]),
                shock_scaling = shock_scaling,
                silent = true,
            )
            if !res_try.errorflag
                sim_try = Array(res_try.simulation)
                if size(sim_try, 2) >= micro_T + 1
                    sep_res = res_try
                    sim = sim_try
                    chosen_attempt = spec
                    push!(attempt_logs, Dict("name" => spec["name"], "status" => "ok"))
                    break
                else
                    push!(attempt_logs, Dict("name" => spec["name"], "status" => "too_short", "sim_cols" => size(sim_try, 2)))
                end
            else
                push!(attempt_logs, Dict("name" => spec["name"], "status" => "sep_errorflag", "failure_period" => res_try.failure_period))
            end
        catch err
            push!(attempt_logs, Dict("name" => spec["name"], "status" => "error", "error" => sprint(showerror, err)))
        end
    end
    sep_res === nothing && error("Direct SEP truth-shock microcase failed for all attempts: $(attempt_logs)")
    sim === nothing && error("Internal error: SEP simulation array missing after successful attempt.")

    obs_sep = Matrix{Float64}(sim[obs_idx, 2:(micro_T + 1)])
    obs_data = copy(obs_sep)  # Direct SEP defines the microcase reference.
    d_obs = size(obs_data, 1)
    rom_model = load_hlt_linear_model(root; mod = @__MODULE__)
    rom_params = params_from_theta(rom_model, theta_true, theta_names)
    rom_fallback_note = "ROM1 baseline simulated with MacroModelling.get_irf on the non-OBC HLT model (first_order)."
    rom_use_obc = false
    irf_lin = Base.invokelatest(
        MacroModelling.get_irf,
        rom_model;
        algorithm = :first_order,
        shocks = sample_shocks,
        periods = micro_T,
        variables = observables,
        parameters = rom_params,
        levels = true,
        ignore_obc = true,
        verbose = false,
    )
    A_lin = Array(irf_lin)
    A_lin2 = ndims(A_lin) == 3 ? A_lin[:, :, 1] : A_lin
    if size(A_lin2, 2) >= micro_T + 1
        obs_rom = Matrix{Float64}(A_lin2[:, 2:(micro_T + 1)])
    elseif size(A_lin2, 2) == micro_T
        obs_rom = Matrix{Float64}(A_lin2)
    else
        error("Unexpected ROM1 simulation shape $(size(A_lin2)) for micro_T=$micro_T")
    end
    size(obs_rom, 1) == d_obs || error("ROM1 observable count mismatch: $(size(obs_rom,1)) vs $d_obs")
    size(obs_rom, 2) == micro_T || error("ROM1 period count mismatch: $(size(obs_rom,2)) vs $micro_T")

    rmse_sep_t = vec(sqrt.(mean((obs_data .- obs_sep) .^ 2, dims = 1)))
    rmse_rom1_t = vec(sqrt.(mean((obs_data .- obs_rom) .^ 2, dims = 1)))
    mae_sep_t = vec(mean(abs.(obs_data .- obs_sep), dims = 1))
    mae_rom1_t = vec(mean(abs.(obs_data .- obs_rom), dims = 1))

    rmse_sep_fit = mean(rmse_sep_t[fit_idx])
    rmse_rom1_fit = mean(rmse_rom1_t[fit_idx])
    mae_sep_fit = mean(mae_sep_t[fit_idx])
    mae_rom1_fit = mean(mae_rom1_t[fit_idx])
    direct_better = (rmse_sep_fit + direct_fit_margin) < rmse_rom1_fit
    direct_better || error("Direct SEP truth-shock microcase fit did not beat ROM1 in volatility window. " *
                           "rmse_sep=$(rmse_sep_fit), rmse_rom1=$(rmse_rom1_fit), idx=$(fit_idx)")
    rmse_rom1_fit > 1e-10 || error("ROM1 fit error is too small in the microcase volatility window; shock episode is not informative.")

    return Dict{String,Any}(
        "fit_period_indices" => fit_idx,
        "fit_period_source" => "synthetic_vol_window_microcase",
        "microcase_source_periods_in_strict_run" => collect(micro_start:micro_end),
        "microcase_total_periods" => micro_T,
        "rmse_sep_per_period" => rmse_sep_t,
        "rmse_rom1_per_period" => rmse_rom1_t,
        "mae_sep_per_period" => mae_sep_t,
        "mae_rom1_per_period" => mae_rom1_t,
        "rmse_sep_fit_region" => rmse_sep_fit,
        "rmse_rom1_fit_region" => rmse_rom1_fit,
        "mae_sep_fit_region" => mae_sep_fit,
        "mae_rom1_fit_region" => mae_rom1_fit,
        "rmse_gain_rom1_minus_sep_fit_region" => rmse_rom1_fit - rmse_sep_fit,
        "direct_sep_better_than_rom1_fit_region" => direct_better,
        "direct_fit_margin" => direct_fit_margin,
        "microcase_uses_direct_sep_generated_observations" => true,
        "model" => String(mm_model.model_name),
        "rom1_use_obc" => rom_use_obc,
        "rom1_fallback_note" => rom_fallback_note,
        "sep_attempt_used" => chosen_attempt["name"],
        "sep_attempt_logs" => attempt_logs,
        "sep_order" => chosen_attempt["sep_order"],
        "sep_nnodes" => chosen_attempt["sep_nnodes"],
        "sep_horizon" => chosen_attempt["sep_horizon"],
        "sep_maxit" => chosen_attempt["sep_maxit"],
        "sep_accept_tol" => chosen_attempt["sep_accept_tol"],
    )
end

function compare_direct_vs_rom1(root::String,
                                benchmark_script::String,
                                chain_path::String,
                                synthetic_path::String,
                                bench_dir::String,
                                blocks::Vector{Symbol};
                                force_benchmarks::Bool,
                                direct_preset::String,
                                rom1_preset::String,
                                recovery_ladder::Bool,
                                reuse_existing_direct_first_path::Union{Nothing,String},
                                require_any_direct_better::Bool,
                                min_improvement::Float64,
                                dry_run::Bool,
                                quiet::Bool,
                                timeout_seconds::Float64)
    comparisons = Dict{String,Any}()
    direct_better_blocks = String[]
    benchmark_runs = Dict{String,Any}()

    for block in blocks
        block_key = String(block)
        benchmark_runs["direct_" * block_key] = benchmark_step(
            root, benchmark_script, chain_path, synthetic_path, bench_dir;
            preset = direct_preset,
            block = block,
            force = force_benchmarks,
            recovery_ladder = recovery_ladder,
            reuse_existing_direct_first_path = reuse_existing_direct_first_path,
            dry_run = dry_run,
            quiet = quiet,
            timeout_seconds = timeout_seconds,
        )
        benchmark_runs["rom1_" * block_key] = benchmark_step(
            root, benchmark_script, chain_path, synthetic_path, bench_dir;
            preset = rom1_preset,
            block = block,
            force = force_benchmarks,
            recovery_ladder = false,
            dry_run = dry_run,
            quiet = quiet,
            timeout_seconds = timeout_seconds,
        )
        if dry_run
            continue
        end

        direct = parse_true_benchmark_result(benchmark_runs["direct_" * block_key]["out_path"])
        rom1 = parse_true_benchmark_result(benchmark_runs["rom1_" * block_key]["out_path"])

        direct["status"] == "ok" || error("Direct SEP benchmark failed for block=$block_key with status=$(direct["status"]).")
        rom1["status"] == "ok" || error("ROM1 benchmark failed for block=$block_key with status=$(rom1["status"]).")
        direct["selected_period_indices"] == rom1["selected_period_indices"] ||
            error("Benchmark period mismatch for block=$block_key (selected periods).")
        direct["evaluation_period_indices"] == rom1["evaluation_period_indices"] ||
            error("Benchmark period mismatch for block=$block_key (evaluation periods).")
        direct["context_period_indices"] == rom1["context_period_indices"] ||
            error("Benchmark period mismatch for block=$block_key (context periods).")

        ll_direct = Float64(direct["fom_loglik"])
        ll_rom1 = Float64(rom1["fom_loglik"])
        ll_diff = ll_direct - ll_rom1
        direct_better = isfinite(ll_diff) && (ll_diff > min_improvement)
        direct_better && push!(direct_better_blocks, block_key)

        comparisons[block_key] = Dict{String,Any}(
            "selected_period_indices" => direct["selected_period_indices"],
            "evaluation_period_indices" => direct["evaluation_period_indices"],
            "context_period_indices" => direct["context_period_indices"],
            "direct" => direct,
            "rom1" => rom1,
            "ll_diff_direct_minus_rom1" => ll_diff,
            "direct_better_than_rom1" => direct_better,
            "min_improvement" => min_improvement,
        )
    end

    if dry_run
        return Dict{String,Any}(
            "benchmark_runs" => benchmark_runs,
            "comparisons" => comparisons,
            "direct_better_blocks" => String[],
            "require_any_direct_better" => require_any_direct_better,
            "dry_run" => true,
        )
    end

    unique!(direct_better_blocks)
    if require_any_direct_better && isempty(direct_better_blocks)
        error("Direct SEP FOM did not beat ROM1 on any tested gated block ($(join(String.(blocks), ","))).")
    end

    return Dict{String,Any}(
        "benchmark_runs" => benchmark_runs,
        "comparisons" => comparisons,
        "direct_better_blocks" => sort(direct_better_blocks),
        "require_any_direct_better" => require_any_direct_better,
        "min_improvement" => min_improvement,
        "direct_preset" => direct_preset,
        "rom1_preset" => rom1_preset,
        "recovery_ladder" => recovery_ladder,
    )
end

function write_summary(path::String, payload::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT Switching Acceptance Smoke")
        println(io)
        println(io, "- Status: `", payload["status"], "`")
        println(io, "- Run Dir: `", payload["run_dir"], "`")
        println(io, "- Created: `", payload["created_at"], "`")
        println(io, "- Dry Run: `", payload["dry_run"], "`")
        println(io)
        if haskey(payload, "failure_messages") && !isempty(payload["failure_messages"])
            println(io, "## Failures")
            for msg in payload["failure_messages"]
                println(io, "- ", replace(String(msg), "\n" => " "))
            end
            println(io)
        end

        if haskey(payload, "switching")
            sw = payload["switching"]
            println(io, "## Switching")
            println(io, "- Gate Share: `", sw["gate_share"], "`")
            println(io, "- Gated Indices: `", sw["gated_indices"], "`")
            println(io, "- Vol Window: `", sw["vol_start"], ":", sw["vol_end"], "`")
            println(io, "- Overlap Indices: `", sw["gate_vol_overlap_indices"], "`")
            println(io, "- Overlap Count: `", sw["gate_vol_overlap_count"], "`")
            println(io)
        end

        if haskey(payload, "recovery")
            rec = payload["recovery"]
            println(io, "## Parameter Recovery")
            println(io, "- Theta True: `", rec["theta_true"], "`")
            println(io, "- Posterior Mean: `", rec["theta_est"], "`")
            println(io, "- Abs Error: `", rec["theta_abs_error"], "`")
            println(io, "- Thresholds: `", rec["theta_error_thresholds"], "`")
            println(io)
        end

        if haskey(payload, "truth_shock_fit")
            fit = payload["truth_shock_fit"]
            println(io, "## Truth-Shock Fit (Direct SEP vs ROM1)")
            if get(fit, "status", "") == "skipped_dry_run"
                println(io, "- Status: `skipped_dry_run`")
                println(io, "- Reason: `", get(fit, "reason", ""), "`")
                println(io, "- Direct fit margin: `", get(fit, "direct_fit_margin", 0.0), "`")
            else
                println(io, "- Fit region source: `", fit["fit_period_source"], "`")
                println(io, "- Fit periods: `", fit["fit_period_indices"], "`")
                println(io, "- RMSE SEP (fit region): `", fit["rmse_sep_fit_region"], "`")
                println(io, "- RMSE ROM1 (fit region): `", fit["rmse_rom1_fit_region"], "`")
                println(io, "- RMSE Gain (ROM1-SEP): `", fit["rmse_gain_rom1_minus_sep_fit_region"], "`")
                println(io, "- Direct SEP better: `", fit["direct_sep_better_than_rom1_fit_region"], "`")
            end
            println(io)
        end

        if haskey(payload, "fom_vs_rom1")
            cmp = payload["fom_vs_rom1"]
            println(io, "## Direct SEP FOM vs ROM1")
            println(io, "- Require any direct > ROM1: `", cmp["require_any_direct_better"], "`")
            println(io, "- Direct-better blocks: `", get(cmp, "direct_better_blocks", String[]), "`")
            if haskey(cmp, "comparisons")
                for block in sort!(collect(keys(cmp["comparisons"])))
                    c = cmp["comparisons"][block]
                    println(io, "- Block `", block, "`")
                    println(io, "  - Selected: `", c["selected_period_indices"], "`")
                    println(io, "  - Eval: `", c["evaluation_period_indices"], "`")
                    println(io, "  - Context: `", c["context_period_indices"], "`")
                    println(io, "  - Direct status/loglik: `", c["direct"]["status"], "` / `", c["direct"]["fom_loglik"], "`")
                    println(io, "  - ROM1 status/loglik: `", c["rom1"]["status"], "` / `", c["rom1"]["fom_loglik"], "`")
                    println(io, "  - Diff (direct-rom1): `", c["ll_diff_direct_minus_rom1"], "`")
                    println(io, "  - Direct better: `", c["direct_better_than_rom1"], "`")
                end
            end
            println(io)
        end

        if haskey(payload, "benchmarks")
            println(io, "## Benchmark Calls")
            for k in sort!(collect(keys(payload["benchmarks"])))
                b = payload["benchmarks"][k]
                println(io, "- `", k, "`")
                println(io, "  - Out: `", b["out_path"], "`")
                println(io, "  - Ran: `", get(b, "ran", false), "`")
                println(io, "  - Reused Existing: `", get(b, "reused_existing", false), "`")
                if haskey(b, "elapsed_s")
                    println(io, "  - Elapsed (s): `", b["elapsed_s"], "`")
                end
            end
        end
    end
end

function main()
    root = repo_root()
    run_dir = normpath(first_positional_arg_str(ARGS))
    dry_run = parse_arg_bool(ARGS, "--dry-run", false)
    force_benchmarks = parse_arg_bool(ARGS, "--force-benchmarks", false)
    recovery_ladder = parse_arg_bool(ARGS, "--recovery-ladder", true)
    require_any_direct_better = parse_arg_bool(ARGS, "--require-any-direct-better", false)
    run_inversion_benchmark_panel = parse_arg_bool(ARGS, "--run-inversion-benchmark-panel", false)
    min_gate_overlap = parse_arg_int(ARGS, "--min-gate-overlap", 1)
    min_direct_improvement = parse_arg_float(ARGS, "--min-direct-improvement", 0.0)
    min_truthshock_direct_improvement = parse_arg_float(ARGS, "--min-truthshock-direct-improvement", 0.0)
    timeout_seconds = parse_arg_float(ARGS, "--timeout-seconds", 0.0)
    theta_tols = parse_f64_vector_arg(ARGS, "--theta-tols", [0.10, 0.15, 20.0])
    blocks = parse_blocks(ARGS)
    direct_preset = parse_arg_string(ARGS, "--direct-preset", "direct_sep_gated_smoke_order1_tuned")
    rom1_preset = parse_arg_string(ARGS, "--rom1-preset", "first_order_gated_smoke")
    run_id_tag = strip(parse_arg_string(ARGS, "--run-id-tag", ""))
    default_benchmark_dir = joinpath(run_dir, "acceptance_smoke", isempty(run_id_tag) ? "benchmarks" : "benchmarks_" * run_id_tag)
    default_out_toml = append_tag_before_ext(joinpath(run_dir, "acceptance_smoke", "hlt_acceptance_smoke_result.toml"), run_id_tag)
    default_out_summary = append_tag_before_ext(joinpath(run_dir, "acceptance_smoke", "hlt_acceptance_smoke_summary.md"), run_id_tag)
    benchmark_dir = arg_present(ARGS, "--benchmark-dir") ? parse_arg_string(ARGS, "--benchmark-dir", default_benchmark_dir) : default_benchmark_dir
    out_toml = arg_present(ARGS, "--out") ? parse_arg_string(ARGS, "--out", default_out_toml) : default_out_toml
    out_summary = arg_present(ARGS, "--summary") ? parse_arg_string(ARGS, "--summary", default_out_summary) : default_out_summary
    lock_file = parse_arg_string(ARGS, "--lock-file", default_lock_path(run_dir, run_id_tag))
    reuse_existing_direct_first = parse_arg_bool(ARGS, "--reuse-existing-direct-first", true)
    quiet = parse_arg_bool(ARGS, "--quiet", false)

    synthetic_path = joinpath(run_dir, "synthetic", "hlt_sep_synth_data.jls")
    chain_path = joinpath(run_dir, "synthetic", "hlt_sep_surrogate_estimation_chain.jls")
    chain_summary_path = joinpath(run_dir, "synthetic", "hlt_sep_surrogate_estimation_chain_summary.jls")
    existing_direct_fom_path = joinpath(run_dir, "synthetic", "hlt_sep_fom_benchmark.jls")
    benchmark_script = joinpath(root, "scripts", "hlt_sep_surrogate_fom_benchmark.jl")

    payload = Dict{String,Any}(
        "created_at" => string(now()),
        "repo_root" => root,
        "run_dir" => run_dir,
        "dry_run" => dry_run,
        "status" => "ok",
        "inputs" => Dict(
            "synthetic_path" => synthetic_path,
            "chain_path" => chain_path,
            "chain_summary_path" => chain_summary_path,
            "benchmark_script" => benchmark_script,
            "benchmark_dir" => benchmark_dir,
            "blocks" => String.(blocks),
            "theta_tols" => theta_tols,
            "min_gate_overlap" => min_gate_overlap,
            "min_direct_improvement" => min_direct_improvement,
            "direct_preset" => direct_preset,
            "rom1_preset" => rom1_preset,
            "recovery_ladder" => recovery_ladder,
            "require_any_direct_better" => require_any_direct_better,
            "reuse_existing_direct_first" => reuse_existing_direct_first,
            "run_inversion_benchmark_panel" => run_inversion_benchmark_panel,
            "min_truthshock_direct_improvement" => min_truthshock_direct_improvement,
            "timeout_seconds" => timeout_seconds,
            "run_id_tag" => run_id_tag,
            "lock_file" => lock_file,
            "quiet" => quiet,
        ),
    )

    failures = String[]
    benchmarks_for_summary = Dict{String,Any}()
    acquire_lockfile(lock_file)
    try
        try
            synthetic = load_payload_dict(synthetic_path)
            chain_summary = load_payload_dict(chain_summary_path)
            switching = switching_metrics(chain_summary, synthetic; min_overlap = min_gate_overlap)
            payload["switching"] = switching
            payload["recovery"] = recovery_metrics(chain_summary; thresholds = theta_tols)
            if dry_run
                payload["truth_shock_fit"] = Dict{String,Any}(
                    "status" => "skipped_dry_run",
                    "reason" => "Dry run skips direct SEP truth-shock microcase for fast validation.",
                    "direct_fit_margin" => min_truthshock_direct_improvement,
                )
            else
                payload["truth_shock_fit"] = truth_shock_fit_metrics(root, synthetic, switching;
                                                                     direct_fit_margin = min_truthshock_direct_improvement)
            end

            if run_inversion_benchmark_panel
                existing_direct_path = (reuse_existing_direct_first && isfile(existing_direct_fom_path)) ? existing_direct_fom_path : nothing
                fom_cmp = compare_direct_vs_rom1(
                    root,
                    benchmark_script,
                    chain_path,
                    synthetic_path,
                    benchmark_dir,
                    blocks;
                    force_benchmarks = force_benchmarks,
                    direct_preset = direct_preset,
                    rom1_preset = rom1_preset,
                    recovery_ladder = recovery_ladder,
                    reuse_existing_direct_first_path = existing_direct_path,
                    require_any_direct_better = require_any_direct_better,
                    min_improvement = min_direct_improvement,
                    dry_run = dry_run,
                    quiet = quiet,
                    timeout_seconds = timeout_seconds,
                )
                payload["fom_vs_rom1"] = Dict{String,Any}(filter(p -> p.first != "benchmark_runs", pairs(fom_cmp)))
                benchmarks_for_summary = Dict{String,Any}(fom_cmp["benchmark_runs"])
                payload["benchmarks"] = benchmarks_for_summary
            end
        catch err
            payload["status"] = "failed"
            msg = sprint(showerror, err, catch_backtrace())
            push!(failures, msg)
        end

        payload["failure_messages"] = failures
        write_toml(out_toml, payload)
        write_summary(out_summary, payload)

        println("HLT acceptance smoke result: ", payload["status"])
        println("  TOML: ", out_toml)
        println("  Summary: ", out_summary)
        if !isempty(failures)
            error("HLT acceptance smoke failed. See summary: $out_summary")
        end
        return nothing
    finally
        release_lockfile(lock_file)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
