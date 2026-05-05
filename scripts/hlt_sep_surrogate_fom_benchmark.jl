#!/usr/bin/env julia
using Serialization
using AxisKeys
using Dates
using Statistics
using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function ensure_chain_deserialize_modules!()
    try
        @eval import Turing
    catch err
        error("Failed to load Turing before deserializing chain payload. " *
              "Activate the project and ensure Turing is available. Original error: $(sprint(showerror, err))")
    end
    try
        @eval import MCMCChains
    catch err
        error("Failed to load MCMCChains before deserializing chain payload. " *
              "Activate the project and ensure MCMCChains is available. Original error: $(sprint(showerror, err))")
    end
    return nothing
end

function parse_optional_arg_symbol(args::Vector{String}, key::String)
    raw = parse_arg_string(args, key, "")
    raw = strip(raw)
    isempty(raw) && return nothing
    lowered = lowercase(raw)
    lowered in ("none", "null", "nothing") && return nothing
    return Symbol(raw)
end

function parse_optional_arg_int(args::Vector{String}, key::String)
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return nothing
    lowercase(raw) in ("none", "null", "nothing") && return nothing
    return parse(Int, raw)
end

function parse_optional_arg_float(args::Vector{String}, key::String)
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return nothing
    lowercase(raw) in ("none", "null", "nothing") && return nothing
    return parse(Float64, raw)
end

function parse_optional_arg_bool(args::Vector{String}, key::String)
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return nothing
    lowered = lowercase(raw)
    lowered in ("none", "null", "nothing") && return nothing
    lowered in ("true", "1", "yes", "y", "on") && return true
    lowered in ("false", "0", "no", "n", "off") && return false
    error("Invalid boolean for $(key): $(raw)")
end

function has_cli_arg(args::Vector{String}, key::String)
    return any(arg -> arg == key || startswith(arg, key * "="), args)
end

function parse_benchmark_preset(args::Vector{String})
    preset = parse_optional_arg_symbol(args, "--benchmark-preset")
    preset === nothing && return nothing
    if preset in (:direct_sep_gated_smoke, :direct_sep_gated_smoke_order1_tuned, :first_order_gated_smoke)
        return preset
    elseif preset == :hlt_quick_fom
        return :direct_sep_gated_smoke
    else
        error("Unsupported --benchmark-preset=$(preset). Supported: direct_sep_gated_smoke, direct_sep_gated_smoke_order1_tuned, first_order_gated_smoke")
    end
end

function parse_requested_labels(args::Vector{String})
    raw = strip(parse_arg_string(args, "--labels", ""))
    isempty(raw) && return nothing
    labels = unique(filter(x -> !isempty(x), strip.(split(raw, ","))))
    isempty(labels) && return nothing
    allowed = Set(["baseline", "true", "post_mean"])
    invalid = filter(lbl -> !(lbl in allowed), labels)
    isempty(invalid) || error("Invalid labels in --labels: $(join(invalid, ", ")). Allowed: baseline,true,post_mean")
    return Set(labels)
end

function parse_period_selection(args::Vector{String})
    sel = parse_arg_symbol(args, "--period-selection", :all)
    if sel == :nonlinear
        return :gated
    elseif sel == :gated_window
        return :gated_block
    end
    sel in (:all, :gated, :gated_block) ||
        error("Unsupported --period-selection=$(sel). Supported: all, gated, nonlinear, gated_block")
    return sel
end

function get_payload_value(d::AbstractDict, key::String)
    if haskey(d, key)
        return d[key]
    elseif haskey(d, Symbol(key))
        return d[Symbol(key)]
    else
        return nothing
    end
end

function parse_gated_block_strategy(args::Vector{String})
    raw = lowercase(strip(parse_arg_string(args, "--gated-block", "first")))
    raw in ("first", "last", "longest") || error("Unsupported --gated-block=$(raw). Supported: first, last, longest")
    return Symbol(raw)
end

function fom_loglikelihood_supported(alg_sym::Symbol, filter_sym::Symbol)
    # SEP FOM likelihood is currently implemented only for the inversion filter.
    if alg_sym == :stochastic_extended_path && filter_sym != :inversion
        return false
    end
    return true
end

function make_result_entry(status::String,
                           requested_alg_sym::Symbol,
                           effective_alg_sym::Symbol,
                           filter_sym::Symbol;
                           elapsed_s::Real = 0.0,
                           ll_fom = nothing,
                           err_msg::Union{Nothing,String} = nothing,
                           backend::String = "MacroModelling.get_loglikelihood")
    entry = Dict{String,Any}(
        "status" => status,
        "elapsed_s" => round(float(elapsed_s); digits = 3),
        "algorithm" => String(effective_alg_sym),
        "algorithm_requested" => String(requested_alg_sym),
        "algorithm_effective" => String(effective_alg_sym),
        "filter" => String(filter_sym),
        "fom_backend" => backend,
    )
    if ll_fom !== nothing
        entry["fom_loglik"] = ll_fom
    end
    if err_msg !== nothing
        entry["error"] = err_msg
    end
    return entry
end

function maybe_reset_sep_inversion_diagnostics!()
    if isdefined(MacroModelling, :reset_sep_inversion_last_diagnostics!)
        try
            return getfield(MacroModelling, :reset_sep_inversion_last_diagnostics!)()
        catch
            return nothing
        end
    end
    return nothing
end

function maybe_get_sep_inversion_diagnostics()
    if isdefined(MacroModelling, :get_sep_inversion_last_diagnostics)
        try
            return getfield(MacroModelling, :get_sep_inversion_last_diagnostics)()
        catch err
            return Dict{String,Any}(
                "kind" => "sep_inversion_filter",
                "status" => "diagnostics_error",
                "error" => sprint(showerror, err),
            )
        end
    end
    return nothing
end

function make_sep_override_dict(; sep_periods,
                                 sep_order,
                                 sep_nnodes,
                                 sep_sparse_tree,
                                 sep_maxit,
                                 sep_tol,
                                 sep_accept_tol,
                                 sep_shock_scale,
                                 sep_inv_maxit,
                                 sep_inv_step_tol,
                                 sep_inv_resid_tol,
                                 sep_inv_lambda,
                                 sep_inv_predict_tol = nothing,
                                 sep_inv_logdet_method = nothing,
                                 sep_inv_logdet_sv_tol = nothing)
    return Dict{String,Any}(
        "sep_periods" => sep_periods,
        "sep_order" => sep_order,
        "sep_nnodes" => sep_nnodes,
        "sep_sparse_tree" => sep_sparse_tree,
        "sep_maxit" => sep_maxit,
        "sep_tol" => sep_tol,
        "sep_accept_tol" => sep_accept_tol,
        "sep_shock_scale" => sep_shock_scale,
        "sep_inv_maxit" => sep_inv_maxit,
        "sep_inv_step_tol" => sep_inv_step_tol,
        "sep_inv_resid_tol" => sep_inv_resid_tol,
        "sep_inv_lambda" => sep_inv_lambda,
        "sep_inv_predict_tol" => sep_inv_predict_tol,
        "sep_inv_logdet_method" => sep_inv_logdet_method,
        "sep_inv_logdet_sv_tol" => sep_inv_logdet_sv_tol,
    )
end

function merge_sep_overrides(base::Dict{String,Any}, patch::Dict{String,Any})
    merged = copy(base)
    for (k, v) in patch
        merged[String(k)] = v
    end
    return merged
end

function _getnum(v, default)
    return (v === nothing || !(v isa Real) || !isfinite(float(v))) ? default : float(v)
end

function build_sep_floor_recovery_rungs(base_sep_overrides::Dict{String,Any})
    base_sep_maxit = Int(round(_getnum(get(base_sep_overrides, "sep_maxit", nothing), 80)))
    base_sep_inv_maxit = Int(round(_getnum(get(base_sep_overrides, "sep_inv_maxit", nothing), 2)))
    base_sep_accept_tol = _getnum(get(base_sep_overrides, "sep_accept_tol", nothing), 0.25)

    return [
        Dict{String,Any}(
            "name" => "more_iterations",
            "note" => "Increase SEP and inversion iterations before relaxing acceptance.",
            "patch" => Dict{String,Any}(
                "sep_maxit" => max(base_sep_maxit, 120),
                "sep_inv_maxit" => max(base_sep_inv_maxit, 4),
            ),
        ),
        Dict{String,Any}(
            "name" => "partial_sep_acceptance",
            "note" => "Allow bounded partial SEP convergence acceptance if residual remains finite.",
            "patch" => Dict{String,Any}(
                "sep_accept_tol" => max(base_sep_accept_tol, 1.0),
            ),
        ),
        Dict{String,Any}(
            "name" => "order1_sparse_tuned_smoke",
            "note" => "Switch to tuned bounded stochastic-SEP smoke settings that produced finite HLT/OBC direct FOM values.",
            "patch" => Dict{String,Any}(
                "sep_periods" => 4,
                "sep_order" => 1,
                "sep_nnodes" => 3,
                "sep_sparse_tree" => true,
                "sep_maxit" => 20,
                "sep_tol" => 1e-4,
                "sep_accept_tol" => 1.0,
                "sep_shock_scale" => 0.5,
                "sep_inv_maxit" => 1,
                "sep_inv_step_tol" => 1e-4,
                "sep_inv_resid_tol" => 1e-3,
                "sep_inv_lambda" => 1e-3,
                "sep_inv_predict_tol" => get(base_sep_overrides, "sep_inv_predict_tol", 1e-10),
                "sep_inv_logdet_method" => get(base_sep_overrides, "sep_inv_logdet_method", nothing),
                "sep_inv_logdet_sv_tol" => get(base_sep_overrides, "sep_inv_logdet_sv_tol", nothing),
            ),
        ),
        Dict{String,Any}(
            "name" => "order1_sparse_tuned_more_iters",
            "note" => "Keep tuned stochastic-SEP settings but allow more SEP/inversion iterations.",
            "patch" => Dict{String,Any}(
                "sep_periods" => 4,
                "sep_order" => 1,
                "sep_nnodes" => 3,
                "sep_sparse_tree" => true,
                "sep_maxit" => 40,
                "sep_tol" => 1e-4,
                "sep_accept_tol" => 1.0,
                "sep_shock_scale" => 0.5,
                "sep_inv_maxit" => 2,
                "sep_inv_step_tol" => 1e-4,
                "sep_inv_resid_tol" => 1e-3,
                "sep_inv_lambda" => 1e-3,
                "sep_inv_predict_tol" => get(base_sep_overrides, "sep_inv_predict_tol", 1e-10),
                "sep_inv_logdet_method" => get(base_sep_overrides, "sep_inv_logdet_method", nothing),
                "sep_inv_logdet_sv_tol" => get(base_sep_overrides, "sep_inv_logdet_sv_tol", nothing),
            ),
        ),
    ]
end

function should_run_sep_recovery_ladder(requested_alg_sym::Symbol,
                                        effective_alg_sym::Symbol,
                                        filter_sym::Symbol,
                                        status::String)
    status in ("on_failure_loglikelihood", "invalid_loglikelihood", "error") || return false
    requested_alg_sym == :stochastic_extended_path || return false
    effective_alg_sym == :stochastic_extended_path || return false
    filter_sym == :inversion || return false
    return true
end

function run_fom_attempt(model,
                         obs_data_ka,
                         params;
                         requested_alg_sym::Symbol,
                         effective_alg_sym::Symbol,
                         filter_sym::Symbol,
                         on_failure_loglikelihood::Float64,
                         presample_periods_effective::Int,
                         sep_overrides::Dict{String,Any},
                         attempt_name::String,
                         attempt_note::Union{Nothing,String} = nothing)
    maybe_reset_sep_inversion_diagnostics!()
    t0 = time()
    status = "ok"
    ll_fom = nothing
    err_msg = nothing
    try
        ll_fom = MacroModelling.get_loglikelihood(
            model,
            obs_data_ka,
            params;
            algorithm = effective_alg_sym,
            filter = filter_sym,
            verbose = false,
            on_failure_loglikelihood = on_failure_loglikelihood,
            presample_periods = presample_periods_effective,
            sep_periods = get(sep_overrides, "sep_periods", nothing),
            sep_order = get(sep_overrides, "sep_order", nothing),
            sep_nnodes = get(sep_overrides, "sep_nnodes", nothing),
            sep_sparse_tree = get(sep_overrides, "sep_sparse_tree", nothing),
            sep_maxit = get(sep_overrides, "sep_maxit", nothing),
            sep_tol = get(sep_overrides, "sep_tol", nothing),
            sep_accept_tol = get(sep_overrides, "sep_accept_tol", nothing),
            sep_shock_scale = get(sep_overrides, "sep_shock_scale", nothing),
            sep_inv_maxit = get(sep_overrides, "sep_inv_maxit", nothing),
            sep_inv_step_tol = get(sep_overrides, "sep_inv_step_tol", nothing),
            sep_inv_resid_tol = get(sep_overrides, "sep_inv_resid_tol", nothing),
            sep_inv_lambda = get(sep_overrides, "sep_inv_lambda", nothing),
            sep_inv_predict_tol = get(sep_overrides, "sep_inv_predict_tol", nothing),
            sep_inv_logdet_method = get(sep_overrides, "sep_inv_logdet_method", nothing),
            sep_inv_logdet_sv_tol = get(sep_overrides, "sep_inv_logdet_sv_tol", nothing),
        )
        if !isfinite(ll_fom)
            status = "invalid_loglikelihood"
            err_msg = "FOM returned non-finite loglikelihood."
        elseif ll_fom == on_failure_loglikelihood
            status = "on_failure_loglikelihood"
            err_msg = "FOM returned on_failure_loglikelihood sentinel ($(on_failure_loglikelihood))."
        end
    catch err
        status = "error"
        err_msg = sprint(showerror, err)
    end
    elapsed = round(time() - t0; digits = 3)
    sep_diag = maybe_get_sep_inversion_diagnostics()

    attempt = Dict{String,Any}(
        "name" => attempt_name,
        "status" => status,
        "elapsed_s" => elapsed,
        "sep_overrides" => copy(sep_overrides),
    )
    if attempt_note !== nothing
        attempt["note"] = attempt_note
    end
    if ll_fom !== nothing
        attempt["fom_loglik"] = ll_fom
    end
    if err_msg !== nothing
        attempt["error"] = err_msg
    end
    if sep_diag !== nothing
        attempt["sep_inversion_diagnostics"] = sep_diag
        if sep_diag isa AbstractDict
            diag_status = get_payload_value(sep_diag, "status")
            diag_code = get_payload_value(sep_diag, "failure_code")
            if status == "on_failure_loglikelihood"
                attempt["failure_class"] = diag_code === nothing ? "unknown_or_prefilter_failure" : String(diag_code)
            elseif status == "error" && diag_status == "failure"
                attempt["failure_class"] = diag_code === nothing ? "sep_inversion_failure" : String(diag_code)
            end
        end
    elseif status == "on_failure_loglikelihood"
        attempt["failure_class"] = "unknown_or_prefilter_failure"
    end
    return attempt
end

function first_two_positional_args(args::Vector{String})
    out = String[]
    for arg in args
        if !startswith(arg, "--")
            push!(out, arg)
            length(out) == 2 && break
        end
    end
    return out
end

function select_model(model_name::String)
    return load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)
end

function inject_theta(base_params::AbstractVector, model, theta_names::Vector{Symbol}, theta_vals::AbstractVector)
    params = copy(base_params)
    idx = indexin(theta_names, model.parameters)
    if any(isnothing, idx)
        error("Theta names not found in $(model.model_name) parameters.")
    end
    for (i, j) in enumerate(Int.(idx))
        params[j] = theta_vals[i]
    end
    return params
end

function switching_total_for_label(chain_payload::AbstractDict, label::String)
    if label == "true"
        return get(chain_payload, "regime_loglik_true", nothing)
    elseif label == "post_mean"
        return get(chain_payload, "loglik_post_mean", nothing)
    else
        return nothing
    end
end

positional = first_two_positional_args(ARGS)
length(positional) == 2 || error("Usage: julia hlt_sep_surrogate_fom_benchmark.jl <chain.jls> <synthetic_data.jls> [--out=PATH] [--filter=inversion]")
script_t0 = time()
chain_path, synthetic_path = positional
out_path = parse_arg_string(ARGS, "--out", joinpath(dirname(synthetic_path), "hlt_sep_fom_benchmark.jls"))
filter_sym = parse_arg_symbol(ARGS, "--filter", :inversion)
alg_sym = parse_arg_symbol(ARGS, "--algorithm", :stochastic_extended_path)
fallback_alg_sym = parse_optional_arg_symbol(ARGS, "--fallback-algorithm")
benchmark_preset = parse_benchmark_preset(ARGS)
requested_labels = parse_requested_labels(ARGS)
period_selection = parse_period_selection(ARGS)
period_mask_key = parse_arg_string(ARGS, "--period-mask-key", "gate_mask")
gated_block_strategy = parse_gated_block_strategy(ARGS)
context_periods = parse_optional_arg_int(ARGS, "--context-periods")
allow_empty_selection = parse_arg_bool(ARGS, "--allow-empty-selection", false)
prefer_chain_summary = parse_arg_bool(ARGS, "--prefer-chain-summary", true)
build_chain_summary_cache = parse_arg_bool(ARGS, "--build-chain-summary-cache", true)
max_periods = parse_optional_arg_int(ARGS, "--max-periods")
presample_periods = parse_optional_arg_int(ARGS, "--presample-periods")
sep_periods = parse_optional_arg_int(ARGS, "--sep-periods")
sep_order = parse_optional_arg_int(ARGS, "--sep-order")
sep_nnodes = parse_optional_arg_int(ARGS, "--sep-nnodes")
sep_sparse_tree = parse_optional_arg_bool(ARGS, "--sep-sparse-tree")
sep_maxit = parse_optional_arg_int(ARGS, "--sep-maxit")
sep_tol = parse_optional_arg_float(ARGS, "--sep-tol")
sep_accept_tol = parse_optional_arg_float(ARGS, "--sep-accept-tol")
sep_shock_scale = parse_optional_arg_float(ARGS, "--sep-shock-scale")
sep_inv_maxit = parse_optional_arg_int(ARGS, "--sep-inv-maxit")
sep_inv_step_tol = parse_optional_arg_float(ARGS, "--sep-inv-step-tol")
sep_inv_resid_tol = parse_optional_arg_float(ARGS, "--sep-inv-resid-tol")
sep_inv_lambda = parse_optional_arg_float(ARGS, "--sep-inv-lambda")
sep_inv_predict_tol = parse_optional_arg_float(ARGS, "--sep-inv-predict-tol")
sep_inv_logdet_method = parse_optional_arg_symbol(ARGS, "--sep-inv-logdet-method")
sep_inv_logdet_sv_tol = parse_optional_arg_float(ARGS, "--sep-inv-logdet-sv-tol")
recovery_ladder = parse_arg_bool(ARGS, "--recovery-ladder", false)
recovery_ladder_max_rungs = parse_optional_arg_int(ARGS, "--recovery-ladder-max-rungs")
allow_fail = parse_arg_bool(ARGS, "--allow-fail", true)
on_failure_loglikelihood = -1e12
force_obc = "--use-obc" in ARGS
force_no_obc = "--no-obc" in ARGS
force_obc && force_no_obc && error("Specify only one of --use-obc or --no-obc.")
if sep_order !== nothing && sep_order == 1 && sep_nnodes !== nothing && !(sep_nnodes in (1, 3, 5))
    error("--sep-nnodes must be one of 1,3,5 when --sep-order=1 (got $(sep_nnodes)).")
end
recovery_ladder_max_rungs === nothing || recovery_ladder_max_rungs >= 0 || error("--recovery-ladder-max-rungs must be >= 0.")
preset_note = ""
if benchmark_preset !== nothing
    if benchmark_preset == :direct_sep_gated_smoke
        preset_note = "Applied bounded HLT/OBC direct-SEP gated-block smoke defaults (override with explicit flags)."
        has_cli_arg(ARGS, "--labels") || (requested_labels = Set(["true"]))
        has_cli_arg(ARGS, "--algorithm") || (alg_sym = :stochastic_extended_path)
        has_cli_arg(ARGS, "--filter") || (filter_sym = :inversion)
        has_cli_arg(ARGS, "--period-selection") || (period_selection = :gated_block)
        has_cli_arg(ARGS, "--gated-block") || (gated_block_strategy = :first)
        has_cli_arg(ARGS, "--context-periods") || (context_periods = 1)
        has_cli_arg(ARGS, "--max-periods") || (max_periods = 1)
        has_cli_arg(ARGS, "--presample-periods") || (presample_periods = nothing)
        has_cli_arg(ARGS, "--sep-periods") || (sep_periods = 10)
        has_cli_arg(ARGS, "--sep-order") || (sep_order = 0)
        has_cli_arg(ARGS, "--sep-nnodes") || (sep_nnodes = 1)
        has_cli_arg(ARGS, "--sep-maxit") || (sep_maxit = 80)
        has_cli_arg(ARGS, "--sep-tol") || (sep_tol = 1e-5)
        has_cli_arg(ARGS, "--sep-accept-tol") || (sep_accept_tol = 0.5)
        has_cli_arg(ARGS, "--sep-inv-maxit") || (sep_inv_maxit = 2)
        has_cli_arg(ARGS, "--sep-inv-resid-tol") || (sep_inv_resid_tol = 1e-4)
        has_cli_arg(ARGS, "--sep-inv-step-tol") || (sep_inv_step_tol = 1e-5)
        has_cli_arg(ARGS, "--sep-inv-lambda") || (sep_inv_lambda = 1e-3)
        has_cli_arg(ARGS, "--sep-inv-predict-tol") || (sep_inv_predict_tol = 1e-10)
        has_cli_arg(ARGS, "--allow-fail") || (allow_fail = true)
    elseif benchmark_preset == :direct_sep_gated_smoke_order1_tuned
        preset_note = "Applied bounded HLT/OBC direct-SEP gated-block stochastic (order-1) tuned smoke defaults that produced finite direct FOM values in HLT/OBC probes (override with explicit flags)."
        has_cli_arg(ARGS, "--labels") || (requested_labels = Set(["true"]))
        has_cli_arg(ARGS, "--algorithm") || (alg_sym = :stochastic_extended_path)
        has_cli_arg(ARGS, "--filter") || (filter_sym = :inversion)
        has_cli_arg(ARGS, "--period-selection") || (period_selection = :gated_block)
        has_cli_arg(ARGS, "--gated-block") || (gated_block_strategy = :first)
        has_cli_arg(ARGS, "--context-periods") || (context_periods = 1)
        has_cli_arg(ARGS, "--max-periods") || (max_periods = 1)
        has_cli_arg(ARGS, "--presample-periods") || (presample_periods = nothing)
        has_cli_arg(ARGS, "--sep-periods") || (sep_periods = 4)
        has_cli_arg(ARGS, "--sep-order") || (sep_order = 1)
        has_cli_arg(ARGS, "--sep-nnodes") || (sep_nnodes = 3)
        has_cli_arg(ARGS, "--sep-sparse-tree") || (sep_sparse_tree = true)
        has_cli_arg(ARGS, "--sep-shock-scale") || (sep_shock_scale = 0.5)
        has_cli_arg(ARGS, "--sep-maxit") || (sep_maxit = 20)
        has_cli_arg(ARGS, "--sep-tol") || (sep_tol = 1e-4)
        has_cli_arg(ARGS, "--sep-accept-tol") || (sep_accept_tol = 1.0)
        has_cli_arg(ARGS, "--sep-inv-maxit") || (sep_inv_maxit = 1)
        has_cli_arg(ARGS, "--sep-inv-resid-tol") || (sep_inv_resid_tol = 1e-3)
        has_cli_arg(ARGS, "--sep-inv-step-tol") || (sep_inv_step_tol = 1e-4)
        has_cli_arg(ARGS, "--sep-inv-lambda") || (sep_inv_lambda = 1e-3)
        has_cli_arg(ARGS, "--sep-inv-predict-tol") || (sep_inv_predict_tol = 1e-10)
        has_cli_arg(ARGS, "--sep-inv-logdet-method") || (sep_inv_logdet_method = :exact)
        has_cli_arg(ARGS, "--allow-fail") || (allow_fail = true)
    elseif benchmark_preset == :first_order_gated_smoke
        preset_note = "Applied bounded first-order inversion gated-block smoke defaults (override with explicit flags)."
        has_cli_arg(ARGS, "--labels") || (requested_labels = Set(["true"]))
        has_cli_arg(ARGS, "--algorithm") || (alg_sym = :first_order)
        has_cli_arg(ARGS, "--filter") || (filter_sym = :inversion)
        has_cli_arg(ARGS, "--period-selection") || (period_selection = :gated_block)
        has_cli_arg(ARGS, "--gated-block") || (gated_block_strategy = :first)
        has_cli_arg(ARGS, "--context-periods") || (context_periods = 1)
        has_cli_arg(ARGS, "--max-periods") || (max_periods = 1)
        has_cli_arg(ARGS, "--presample-periods") || (presample_periods = nothing)
        has_cli_arg(ARGS, "--allow-fail") || (allow_fail = true)
    end
end
context_periods === nothing || context_periods >= 0 || error("--context-periods must be >= 0.")
period_selection == :gated_block || context_periods === nothing || error("--context-periods is only supported with --period-selection=gated_block.")

chain_payload, chain_summary_path, chain_summary_used = MacroModelling.load_hlt_chain_payload_for_benchmark(
    chain_path;
    prefer_summary = prefer_chain_summary,
    build_summary_cache = build_chain_summary_cache,
    ensure_deserialize_modules! = ensure_chain_deserialize_modules!,
)
if period_selection in (:gated, :gated_block) && get_payload_value(chain_payload, period_mask_key) === nothing && chain_summary_used
    chain_payload, chain_summary_path, chain_summary_used = MacroModelling.load_hlt_chain_payload_for_benchmark(
        chain_path;
        prefer_summary = false,
        build_summary_cache = build_chain_summary_cache,
        ensure_deserialize_modules! = ensure_chain_deserialize_modules!,
    )
end
synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
model_name = String(get(synthetic, "model", "Smets_Wouters_2007_HLT"))
if force_obc
    model_name = "Smets_Wouters_2007_HLT_obc"
elseif force_no_obc
    model_name = "Smets_Wouters_2007_HLT"
end
hlt_model_file_and_symbol(model_name)  # fail fast on unsupported model names before benchmark setup
obs_data = synthetic["obs_data"]
observables = Symbol.(get(synthetic, "observables", Symbol[]))
isempty(observables) && error("Synthetic data missing observables.")
T_full = size(obs_data, 2)
if period_selection in (:gated, :gated_block)
    MacroModelling.validate_hlt_chain_payload(
        chain_payload;
        gate_mask_key = period_mask_key,
        require_gate_mask = true,
        sample_length = T_full,
        label = "HLT benchmark chain payload",
    )
end
selected_period_indices = collect(1:T_full)
evaluation_period_indices = collect(1:T_full)
context_period_indices = Int[]
period_selection_note = ""
if period_selection == :gated
    mask_val = get_payload_value(chain_payload, period_mask_key)
    mask_val === nothing && error("Requested --period-selection=gated but period mask key '$period_mask_key' was not found in chain payload or cache.")
    gate_mask = Bool.(vec(mask_val))
    length(gate_mask) == T_full || error("Gate mask length ($(length(gate_mask))) does not match synthetic sample length T=$(T_full).")
    selected_period_indices = findall(gate_mask)
    evaluation_period_indices = copy(selected_period_indices)
    period_selection_note = "Selected gated periods from chain payload key '$period_mask_key'."
elseif period_selection == :gated_block
    mask_val = get_payload_value(chain_payload, period_mask_key)
    mask_val === nothing && error("Requested --period-selection=gated_block but period mask key '$period_mask_key' was not found in chain payload or cache.")
    gate_mask = Bool.(vec(mask_val))
    length(gate_mask) == T_full || error("Gate mask length ($(length(gate_mask))) does not match synthetic sample length T=$(T_full).")
    max_periods === nothing || max_periods > 0 || error("--max-periods must be positive.")
    selected_period_indices, evaluation_period_indices, context_period_indices, helper_note =
        MacroModelling.select_gated_block_periods(
            gate_mask,
            gated_block_strategy,
            something(context_periods, 0),
            something(max_periods, 0),
        )
    if isempty(selected_period_indices)
        period_selection_note = "No gated periods found in mask key '$period_mask_key'."
    else
        period_selection_note = helper_note * " from mask key '$period_mask_key'."
        if !isempty(context_period_indices)
            period_selection_note *= " Prepended $(length(context_period_indices)) context period(s): $(first(context_period_indices)):$(last(context_period_indices))."
        end
    end
end
if period_selection != :gated_block && max_periods !== nothing
    max_periods > 0 || error("--max-periods must be positive.")
    if max_periods < length(selected_period_indices)
        selected_period_indices = selected_period_indices[1:max_periods]
        evaluation_period_indices = copy(selected_period_indices)
        period_selection_note = isempty(period_selection_note) ?
            "Selected first $(max_periods) periods." :
            period_selection_note * " Truncated to first $(max_periods) selected periods."
    elseif period_selection == :all && max_periods > T_full
        error("--max-periods=$(max_periods) exceeds available sample length $(T_full).")
    end
end
if isempty(selected_period_indices)
    allow_empty_selection || error("No periods selected for benchmark (period_selection=$(period_selection), mask_key=$(period_mask_key)). Use --allow-empty-selection=true to allow empty selections.")
end
context_periods_effective = length(context_period_indices)
presample_periods_effective = isnothing(presample_periods) ? context_periods_effective : presample_periods
presample_periods_effective >= 0 || error("presample_periods must be >= 0.")
if !isempty(selected_period_indices) && presample_periods_effective >= length(selected_period_indices)
    error("presample_periods=$(presample_periods_effective) must be smaller than selected sample length $(length(selected_period_indices)).")
end
obs_data = obs_data[:, selected_period_indices]
obs_data_ka = KeyedArray(obs_data; Variable = observables, Time = selected_period_indices)

theta_names = Symbol.(get(synthetic, "theta_names", Symbol[]))
isempty(theta_names) && error("Synthetic data missing theta_names.")
theta_true = synthetic["theta_true"]
post_mean = get(chain_payload, "post_mean_theta", nothing)

results = Dict{String,Any}()
comparisons = Dict{String,Any}()
failures = String[]
requested_alg_sym = alg_sym
effective_alg_sym = alg_sym
algorithm_supported = fom_loglikelihood_supported(requested_alg_sym, filter_sym)
algorithm_fallback_used = false
algorithm_support_note = ""
benchmark_is_subset = (length(selected_period_indices) != T_full) ||
                      any(selected_period_indices .!= collect(1:length(selected_period_indices))) ||
                      (presample_periods_effective != 0)
if !algorithm_supported
    if fallback_alg_sym === nothing
        algorithm_support_note =
            "Requested FOM backend call is unsupported in this codebase: " *
            "MacroModelling.get_loglikelihood(...; algorithm=$(requested_alg_sym), filter=$(filter_sym)). " *
            "Provide --fallback-algorithm=<supported_algorithm> to run a proxy benchmark."
        labels = String["baseline", "true"]
        if post_mean !== nothing
            push!(labels, "post_mean")
        end
        if requested_labels !== nothing
            labels = filter(lbl -> lbl in requested_labels, labels)
            isempty(labels) && error("Requested --labels selects no available labels in the current payload.")
        end
        for label in labels
            switch_total = switching_total_for_label(chain_payload, label)
            entry = make_result_entry(
                "unsupported_algorithm",
                requested_alg_sym,
                requested_alg_sym,
                filter_sym;
                elapsed_s = 0.0,
                err_msg = algorithm_support_note,
            )
            if switch_total !== nothing
                entry["switching_loglik"] = switch_total
            end
            results[label] = entry
            push!(failures, "$(label): $(algorithm_support_note)")
        end
    else
        effective_alg_sym = fallback_alg_sym
        algorithm_fallback_used = true
        algorithm_support_note =
            "Requested algorithm $(requested_alg_sym) is unsupported via MacroModelling.get_loglikelihood; " *
            "using fallback algorithm $(effective_alg_sym) for proxy FOM benchmark."
        println("Warning: ", algorithm_support_note)
    end
end

if isempty(results)
model = select_model(model_name)
panel = Dict{String,Any}(
    "baseline" => copy(model.parameter_values),
    "true" => inject_theta(model.parameter_values, model, theta_names, theta_true),
)
if post_mean !== nothing
    panel["post_mean"] = inject_theta(model.parameter_values, model, theta_names, post_mean)
end
if requested_labels !== nothing
    panel = Dict(k => v for (k, v) in panel if k in requested_labels)
    isempty(panel) && error("Requested --labels selects no available labels in the current payload.")
end
base_sep_overrides = make_sep_override_dict(
    sep_periods = sep_periods,
    sep_order = sep_order,
    sep_nnodes = sep_nnodes,
    sep_sparse_tree = sep_sparse_tree,
    sep_maxit = sep_maxit,
    sep_tol = sep_tol,
    sep_accept_tol = sep_accept_tol,
    sep_shock_scale = sep_shock_scale,
    sep_inv_maxit = sep_inv_maxit,
    sep_inv_step_tol = sep_inv_step_tol,
    sep_inv_resid_tol = sep_inv_resid_tol,
    sep_inv_lambda = sep_inv_lambda,
    sep_inv_predict_tol = sep_inv_predict_tol,
    sep_inv_logdet_method = sep_inv_logdet_method,
    sep_inv_logdet_sv_tol = sep_inv_logdet_sv_tol,
)
for (label, params) in sort(collect(panel); by = first)
    attempts = Dict{String,Any}[]
    base_attempt = run_fom_attempt(
        model,
        obs_data_ka,
        params;
        requested_alg_sym = requested_alg_sym,
        effective_alg_sym = effective_alg_sym,
        filter_sym = filter_sym,
        on_failure_loglikelihood = on_failure_loglikelihood,
        presample_periods_effective = presample_periods_effective,
        sep_overrides = base_sep_overrides,
        attempt_name = "base",
        attempt_note = "Requested benchmark settings.",
    )
    push!(attempts, base_attempt)
    final_attempt = base_attempt
    recovery_rung_used = nothing

    if recovery_ladder && should_run_sep_recovery_ladder(requested_alg_sym, effective_alg_sym, filter_sym, String(base_attempt["status"]))
        rungs = build_sep_floor_recovery_rungs(base_sep_overrides)
        if recovery_ladder_max_rungs !== nothing
            rungs = rungs[1:min(length(rungs), recovery_ladder_max_rungs)]
        end
        for rung in rungs
            merged_sep_overrides = merge_sep_overrides(base_sep_overrides, Dict{String,Any}(rung["patch"]))
            merged_sep_overrides == get(final_attempt, "sep_overrides", Dict{String,Any}()) && continue
            attempt = run_fom_attempt(
                model,
                obs_data_ka,
                params;
                requested_alg_sym = requested_alg_sym,
                effective_alg_sym = effective_alg_sym,
                filter_sym = filter_sym,
                on_failure_loglikelihood = on_failure_loglikelihood,
                presample_periods_effective = presample_periods_effective,
                sep_overrides = merged_sep_overrides,
                attempt_name = "recovery:" * String(rung["name"]),
                attempt_note = String(rung["note"]),
            )
            push!(attempts, attempt)
            final_attempt = attempt
            if String(attempt["status"]) == "ok"
                recovery_rung_used = String(rung["name"])
                break
            end
        end
    end

    status = String(final_attempt["status"])
    ll_fom = get(final_attempt, "fom_loglik", nothing)
    err_msg = get(final_attempt, "error", nothing)
    elapsed = get(final_attempt, "elapsed_s", 0.0)
    if status != "ok"
        push!(failures, "$(label): $(err_msg === nothing ? status : err_msg)")
    end
    entry = make_result_entry(
        status,
        requested_alg_sym,
        effective_alg_sym,
        filter_sym;
        elapsed_s = elapsed,
        ll_fom = ll_fom,
        err_msg = err_msg isa Nothing ? nothing : String(err_msg),
    )
    entry["attempts"] = attempts
    entry["attempts_count"] = length(attempts)
    entry["recovery_ladder_enabled"] = recovery_ladder
    entry["recovery_ladder_attempted"] = length(attempts) > 1
    entry["recovery_rung_used"] = recovery_rung_used
    if haskey(final_attempt, "sep_inversion_diagnostics")
        entry["sep_inversion_diagnostics"] = final_attempt["sep_inversion_diagnostics"]
    end
    if haskey(final_attempt, "failure_class")
        entry["sep_floor_failure_class"] = final_attempt["failure_class"]
    end
    switch_total = switching_total_for_label(chain_payload, label)
    if switch_total !== nothing
        if benchmark_is_subset
            entry["switching_loglik_fullsample"] = switch_total
            entry["comparison_skipped"] = "subset_or_presample_mismatch"
        elseif ll_fom !== nothing
            cmp = MacroModelling.evaluate_switching_vs_fom(switch_total, ll_fom)
            comparisons[label] = Dict(string(k) => v for (k, v) in pairs(cmp))
            entry["switching_loglik"] = switch_total
        end
    end
    results[label] = entry
end
end

payload = Dict(
    "created_at" => string(Dates.now()),
    "script_elapsed_s" => round(time() - script_t0; digits = 3),
    "chain_path" => chain_path,
    "chain_summary_path" => chain_summary_path,
    "chain_summary_used" => chain_summary_used,
    "synthetic_path" => synthetic_path,
    "model" => model_name,
    "period_selection" => String(period_selection),
    "gated_block_strategy" => period_selection == :gated_block ? String(gated_block_strategy) : nothing,
    "period_mask_key" => period_mask_key,
    "period_selection_note" => period_selection_note,
    "selected_period_indices" => selected_period_indices,
    "selected_periods_count" => length(selected_period_indices),
    "selected_periods_share" => T_full == 0 ? 0.0 : length(selected_period_indices) / T_full,
    "evaluation_period_indices" => evaluation_period_indices,
    "evaluation_periods_count" => length(evaluation_period_indices),
    "context_period_indices" => context_period_indices,
    "context_periods_requested" => context_periods,
    "context_periods_effective" => context_periods_effective,
    "benchmark_is_subset" => benchmark_is_subset,
    "algorithm" => String(effective_alg_sym),
    "benchmark_preset" => benchmark_preset === nothing ? nothing : String(benchmark_preset),
    "benchmark_preset_note" => preset_note,
    "algorithm_requested" => String(requested_alg_sym),
    "algorithm_effective" => String(effective_alg_sym),
    "algorithm_supported" => algorithm_supported,
    "algorithm_fallback_used" => algorithm_fallback_used,
    "algorithm_support_note" => algorithm_support_note,
    "filter" => String(filter_sym),
    "labels_requested" => requested_labels === nothing ? nothing : sort!(String.(collect(requested_labels))),
    "max_periods" => max_periods,
    "presample_periods" => presample_periods,
    "presample_periods_effective" => presample_periods_effective,
    "recovery_ladder_enabled" => recovery_ladder,
    "recovery_ladder_max_rungs" => recovery_ladder_max_rungs,
    "recovery_ladder_policy" => "deterministic_sep_floor_v1",
    "sep_overrides" => Dict(
        "sep_periods" => sep_periods,
        "sep_order" => sep_order,
        "sep_nnodes" => sep_nnodes,
        "sep_sparse_tree" => sep_sparse_tree,
        "sep_maxit" => sep_maxit,
        "sep_tol" => sep_tol,
        "sep_accept_tol" => sep_accept_tol,
        "sep_shock_scale" => sep_shock_scale,
        "sep_inv_maxit" => sep_inv_maxit,
        "sep_inv_step_tol" => sep_inv_step_tol,
        "sep_inv_resid_tol" => sep_inv_resid_tol,
        "sep_inv_lambda" => sep_inv_lambda,
        "sep_inv_predict_tol" => sep_inv_predict_tol,
        "sep_inv_logdet_method" => sep_inv_logdet_method === nothing ? nothing : String(sep_inv_logdet_method),
        "sep_inv_logdet_sv_tol" => sep_inv_logdet_sv_tol,
    ),
    "results" => results,
    "comparisons" => comparisons,
    "failures" => failures,
)
serialize(out_path, payload)

summary_stem = replace(basename(out_path), r"\.jls$" => "")
summary_path = joinpath(dirname(out_path), "$(summary_stem)_summary.md")
open(summary_path, "w") do io
    println(io, "# HLT FOM Benchmark Summary")
    println(io)
    println(io, "- Created: `", payload["created_at"], "`")
    println(io, "- Script Elapsed (s): `", payload["script_elapsed_s"], "`")
    println(io, "- Model: `", model_name, "`")
    println(io, "- Period Selection: `", payload["period_selection"], "`")
    if payload["gated_block_strategy"] !== nothing
        println(io, "- Gated Block Strategy: `", payload["gated_block_strategy"], "`")
    end
    println(io, "- Period Mask Key: `", payload["period_mask_key"], "`")
    println(io, "- Selected Periods: `", payload["selected_periods_count"], " / ", T_full, "`")
    println(io, "- Evaluation Periods: `", payload["evaluation_periods_count"], "`")
    println(io, "- Context Periods (requested/effective): `", payload["context_periods_requested"], " / ", payload["context_periods_effective"], "`")
    if !isempty(String(payload["period_selection_note"]))
        println(io, "- Period Selection Note: `", replace(String(payload["period_selection_note"]), "\n" => " "), "`")
    end
    println(io, "- Requested Algorithm: `", requested_alg_sym, "`")
    println(io, "- Effective Algorithm: `", effective_alg_sym, "`")
    if payload["benchmark_preset"] !== nothing
        println(io, "- Benchmark Preset: `", payload["benchmark_preset"], "`")
        if !isempty(String(payload["benchmark_preset_note"]))
            println(io, "- Benchmark Preset Note: `", replace(String(payload["benchmark_preset_note"]), "\n" => " "), "`")
        end
    end
    println(io, "- Algorithm Supported: `", algorithm_supported, "`")
    if !isempty(algorithm_support_note)
        println(io, "- Algorithm Note: `", replace(algorithm_support_note, "\n" => " "), "`")
    end
    println(io, "- Filter: `", filter_sym, "`")
    println(io, "- Labels Requested: `", payload["labels_requested"], "`")
    println(io, "- Max Periods: `", payload["max_periods"], "`")
    println(io, "- Presample Periods: `", payload["presample_periods"], "`")
    println(io, "- Presample Periods Effective: `", payload["presample_periods_effective"], "`")
    println(io, "- Recovery Ladder Enabled: `", payload["recovery_ladder_enabled"], "`")
    println(io, "- Recovery Ladder Max Rungs: `", payload["recovery_ladder_max_rungs"], "`")
    println(io, "- Recovery Ladder Policy: `", payload["recovery_ladder_policy"], "`")
    println(io, "- SEP Overrides: `", payload["sep_overrides"], "`")
    println(io, "- Chain: `", chain_path, "`")
    println(io, "- Chain Summary Path: `", chain_summary_path, "`")
    println(io, "- Chain Summary Used: `", chain_summary_used, "`")
    println(io, "- Synthetic: `", synthetic_path, "`")
    println(io, "- Output: `", out_path, "`")
    println(io)
    println(io, "## Results")
    for label in sort(collect(keys(results)))
        r = results[label]
        println(io, "- `", label, "`: status=`", r["status"], "`, elapsed_s=`", get(r, "elapsed_s", "n/a"), "`")
        if get(r, "recovery_ladder_attempted", false)
            println(io, "  - Recovery rung used: `", get(r, "recovery_rung_used", "none"), "`")
            println(io, "  - Attempts: `", get(r, "attempts_count", "n/a"), "`")
            if haskey(r, "attempts") && (r["attempts"] isa AbstractVector)
                trace_parts = String[]
                for a in r["attempts"]
                    if a isa AbstractDict
                        nm = get(a, "name", "attempt")
                        st = get(a, "status", "unknown")
                        fc = get(a, "failure_class", nothing)
                        part = fc === nothing ? "$(nm):$(st)" : "$(nm):$(st):$(fc)"
                        push!(trace_parts, part)
                    end
                end
                isempty(trace_parts) || println(io, "  - Attempt trace: `", join(trace_parts, " -> "), "`")
            end
        end
        if haskey(r, "fom_loglik")
            println(io, "  - FOM loglik: `", r["fom_loglik"], "`")
        end
        if haskey(r, "switching_loglik")
            println(io, "  - Switching loglik: `", r["switching_loglik"], "`")
            if haskey(comparisons, label)
                println(io, "  - Total diff: `", comparisons[label]["total_diff"], "`")
            end
        elseif haskey(r, "switching_loglik_fullsample")
            println(io, "  - Switching loglik (full sample): `", r["switching_loglik_fullsample"], "`")
            println(io, "  - Comparison skipped: `", get(r, "comparison_skipped", "unknown"), "`")
        end
        if haskey(r, "error")
            println(io, "  - Error: `", replace(String(r["error"]), "\n" => " "), "`")
        end
        if haskey(r, "sep_floor_failure_class")
            println(io, "  - SEP floor failure class: `", r["sep_floor_failure_class"], "`")
        end
        if haskey(r, "sep_inversion_diagnostics")
            diag = r["sep_inversion_diagnostics"]
            if diag isa AbstractDict
                diag_status = get_payload_value(diag, "status")
                diag_period = get_payload_value(diag, "period_index")
                diag_code = get_payload_value(diag, "failure_code")
                if diag_status !== nothing || diag_period !== nothing || diag_code !== nothing
                    println(io, "  - SEP diag: status=`", diag_status, "`, period=`", diag_period, "`, code=`", diag_code, "`")
                end
            end
        end
    end
end

println("HLT FOM benchmark")
println("  Output payload: $out_path")
println("  Summary: $summary_path")
if !isempty(failures)
    println("  Failures: $(length(failures))")
    if !isempty(algorithm_support_note)
        println("  Note: $(algorithm_support_note)")
    end
    if !allow_fail
        error("FOM benchmark failures encountered: $(join(failures, " | "))")
    end
end
