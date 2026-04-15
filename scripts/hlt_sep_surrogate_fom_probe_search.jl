#!/usr/bin/env julia
using Serialization
using AxisKeys
using Dates
using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

script_repo_root() = normpath(joinpath(@__DIR__, ".."))

function parse_csv_ints(args::Vector{String}, key::String, default::Vector{Int})
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return default
    vals = Int[]
    for part in split(raw, ",")
        s = strip(part)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    isempty(vals) && return default
    return vals
end

function parse_csv_symbols(args::Vector{String}, key::String, default::Vector{Symbol})
    raw = strip(parse_arg_string(args, key, ""))
    isempty(raw) && return default
    vals = Symbol[]
    for part in split(raw, ",")
        s = lowercase(strip(part))
        isempty(s) && continue
        push!(vals, Symbol(s))
    end
    isempty(vals) && return default
    return vals
end

function has_cli_arg(args::Vector{String}, key::String)
    return any(arg -> arg == key || startswith(arg, key * "="), args)
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

function ensure_chain_deserialize_modules!()
    try
        @eval import Turing
    catch
        # Chain summary sidecar path usually avoids this; defer error until full chain is required.
    end
    try
        @eval import MCMCChains
    catch
    end
    return nothing
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

function load_chain_payload_light(chain_path::String; period_mask_key::String = "gate_mask")
    summary_path = replace(chain_path, r"\.jls$" => "_summary.jls")
    if isfile(summary_path)
        payload = MacroModelling.load_hlt_chain_summary(summary_path)
        if get_payload_value(payload, period_mask_key) !== nothing
            MacroModelling.validate_hlt_chain_payload(
                payload;
                gate_mask_key = period_mask_key,
                require_gate_mask = true,
                label = "HLT probe chain summary payload",
            )
            return payload, summary_path, true
        end
    end
    ensure_chain_deserialize_modules!()
    payload = MacroModelling.load_hlt_chain_payload(chain_path)
    MacroModelling.validate_hlt_chain_payload(
        payload;
        gate_mask_key = period_mask_key,
        require_gate_mask = true,
        label = "HLT probe chain payload",
    )
    return payload, summary_path, false
end

function include_hlt_model!(model_name::String)
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

function classify_status(ll::Union{Nothing,Real}, sentinel::Float64)
    if ll === nothing
        return "missing"
    elseif !isfinite(ll)
        return "invalid_loglikelihood"
    elseif ll == sentinel
        return "on_failure_loglikelihood"
    else
        return "ok"
    end
end

positional = String[]
for arg in ARGS
    !startswith(arg, "--") && push!(positional, arg)
    length(positional) == 2 && break
end
length(positional) == 2 || error("Usage: julia hlt_sep_surrogate_fom_probe_search.jl <chain.jls> <synthetic_data.jls> [--out=PATH] [--gated-blocks=first,last] [--contexts=1,2]")
chain_path, synthetic_path = positional
total_t0 = time()

out_path = parse_arg_string(ARGS, "--out", joinpath(dirname(synthetic_path), "hlt_sep_fom_probe_search.jls"))
period_mask_key = parse_arg_string(ARGS, "--period-mask-key", "gate_mask")
gated_blocks = parse_csv_symbols(ARGS, "--gated-blocks", [:first, :last])
for s in gated_blocks
    s in (:first, :last, :longest) || error("Unsupported gated block strategy in --gated-blocks: $s")
end
contexts = parse_csv_ints(ARGS, "--contexts", [1, 2])
all(c -> c >= 0, contexts) || error("--contexts must be nonnegative.")
max_eval_periods = parse_arg_int(ARGS, "--max-eval-periods", 1)
max_eval_periods > 0 || error("--max-eval-periods must be positive.")
include_order1 = parse_arg_bool(ARGS, "--include-order1", false)
skip_order0 = parse_arg_bool(ARGS, "--skip-order0", false) || parse_arg_bool(ARGS, "--only-order1", false)
dry_run = parse_arg_bool(ARGS, "--dry-run", false)
allow_fail = parse_arg_bool(ARGS, "--allow-fail", true)
on_failure_loglikelihood = parse_arg_float(ARGS, "--on-failure-loglikelihood", -1e12)
sep_sparse_tree_override = parse_optional_arg_bool(ARGS, "--sep-sparse-tree")
sep_shock_scale_override = parse_optional_arg_float(ARGS, "--sep-shock-scale")

sep_order1_periods = parse_arg_int(ARGS, "--order1-sep-periods", 6)
sep_order1_nnodes = parse_arg_int(ARGS, "--order1-sep-nnodes", 3)
sep_order1_maxit = parse_arg_int(ARGS, "--order1-sep-maxit", 60)
sep_order1_tol = parse_arg_float(ARGS, "--order1-sep-tol", 1e-5)
sep_order1_accept_tol = parse_arg_float(ARGS, "--order1-sep-accept-tol", 0.75)
sep_order1_inv_maxit = parse_arg_int(ARGS, "--order1-sep-inv-maxit", 2)
sep_order1_inv_resid_tol = parse_arg_float(ARGS, "--order1-sep-inv-resid-tol", 1e-4)
sep_order1_inv_step_tol = parse_arg_float(ARGS, "--order1-sep-inv-step-tol", 1e-5)
sep_order1_inv_lambda = parse_arg_float(ARGS, "--order1-sep-inv-lambda", 1e-3)
include_order1 && !(sep_order1_nnodes in (1, 3, 5)) &&
    error("--order1-sep-nnodes must be one of 1,3,5 (got $(sep_order1_nnodes)).")

chain_payload, chain_summary_path, chain_summary_used = load_chain_payload_light(chain_path; period_mask_key = period_mask_key)
synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)

model_name = String(get(synthetic, "model", "Smets_Wouters_2007_HLT"))
if "--use-obc" in ARGS
    model_name = "Smets_Wouters_2007_HLT_obc"
elseif "--no-obc" in ARGS
    model_name = "Smets_Wouters_2007_HLT"
end

obs_data_full = synthetic["obs_data"]
observables = Symbol.(get(synthetic, "observables", Symbol[]))
isempty(observables) && error("Synthetic data missing observables.")
T_full = size(obs_data_full, 2)

gate_mask_val = get_payload_value(chain_payload, period_mask_key)
gate_mask_val === nothing && error("Gate mask key '$period_mask_key' not found in chain payload or summary cache.")
gate_mask = Bool.(vec(gate_mask_val))
length(gate_mask) == T_full || error("Gate mask length ($(length(gate_mask))) != synthetic sample length $(T_full).")
MacroModelling.validate_hlt_chain_payload(
    chain_payload;
    gate_mask_key = period_mask_key,
    require_gate_mask = true,
    sample_length = T_full,
    label = "HLT probe chain payload",
)

theta_names = Symbol.(get(synthetic, "theta_names", Symbol[]))
isempty(theta_names) && error("Synthetic data missing theta_names.")
theta_true = get(synthetic, "theta_true", nothing)
theta_true === nothing && error("Synthetic data missing theta_true.")

probe_cfgs = NamedTuple[]
if !skip_order0
    push!(probe_cfgs, (
        name = "o0_default",
        sep_periods = 10,
        sep_order = 0,
        sep_nnodes = 1,
        sep_maxit = 80,
        sep_tol = 1e-5,
        sep_accept_tol = 0.5,
        sep_inv_maxit = 2,
        sep_inv_resid_tol = 1e-4,
        sep_inv_step_tol = 1e-5,
        sep_inv_lambda = 1e-3,
    ))
    push!(probe_cfgs, (
        name = "o0_relaxed",
        sep_periods = 10,
        sep_order = 0,
        sep_nnodes = 1,
        sep_maxit = 80,
        sep_tol = 1e-5,
        sep_accept_tol = 1.0,
        sep_inv_maxit = 4,
        sep_inv_resid_tol = 1e-3,
        sep_inv_step_tol = 1e-5,
        sep_inv_lambda = 1e-3,
    ))
end
if include_order1
    push!(probe_cfgs, (
        name = "o1_smoke",
        sep_periods = sep_order1_periods,
        sep_order = 1,
        sep_nnodes = sep_order1_nnodes,
        sep_maxit = sep_order1_maxit,
        sep_tol = sep_order1_tol,
        sep_accept_tol = sep_order1_accept_tol,
        sep_inv_maxit = sep_order1_inv_maxit,
        sep_inv_resid_tol = sep_order1_inv_resid_tol,
        sep_inv_step_tol = sep_order1_inv_step_tol,
        sep_inv_lambda = sep_order1_inv_lambda,
    ))
end

planned_cases = NamedTuple[]
for strategy in gated_blocks, ctx in contexts, cfg in probe_cfgs
    selected_idx, eval_idx, ctx_idx, note = MacroModelling.select_gated_block_periods(gate_mask, strategy, ctx, max_eval_periods)
    push!(planned_cases, (
        case_id = "$(cfg.name)__$(strategy)__ctx$(ctx)",
        strategy = strategy,
        context_periods_requested = ctx,
        selected_idx = selected_idx,
        eval_idx = eval_idx,
        context_idx = ctx_idx,
        presample_periods = length(ctx_idx),
        selection_note = note,
        cfg = cfg,
    ))
end

if dry_run
    println("HLT SEP FOM probe search (dry-run)")
    println("  Planned cases: ", length(planned_cases))
    for c in planned_cases
        println("  - ", c.case_id, " selected=", c.selected_idx, " eval=", c.eval_idx, " ctx=", c.context_idx)
    end
    exit(0)
end

model_t0 = time()
model = include_hlt_model!(model_name)
model_setup_elapsed_s = round(time() - model_t0; digits = 3)
params_true = inject_theta(model.parameter_values, model, theta_names, theta_true)

results = Dict{String,Any}()
failures = String[]
probe_loop_t0 = time()
ok_cases = 0

println("HLT SEP FOM probe search")
println("  Model: $model_name")
println("  Gate mask key: $period_mask_key")
println("  Gate mask true periods: ", findall(gate_mask))
println("  Planned cases: ", length(planned_cases))
println("  Chain summary used: ", chain_summary_used)

for c in planned_cases
    if isempty(c.selected_idx)
        entry = Dict{String,Any}(
            "status" => "empty_selection",
            "error" => "No selected periods for case",
            "selected_period_indices" => c.selected_idx,
            "evaluation_period_indices" => c.eval_idx,
            "context_period_indices" => c.context_idx,
            "presample_periods" => c.presample_periods,
            "config" => Dict(string(k) => getfield(c.cfg, k) for k in propertynames(c.cfg)),
        )
        results[c.case_id] = entry
        push!(failures, "$(c.case_id): empty selection")
        continue
    end

    obs_sub = obs_data_full[:, c.selected_idx]
    obs_ka = KeyedArray(obs_sub; Variable = observables, Time = c.selected_idx)

    ll = nothing
    status = "ok"
    err_msg = nothing
    t0 = time()
    try
        ll = MacroModelling.get_loglikelihood(
            model,
            obs_ka,
            params_true;
            algorithm = :stochastic_extended_path,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = on_failure_loglikelihood,
            presample_periods = c.presample_periods,
            sep_periods = c.cfg.sep_periods,
            sep_order = c.cfg.sep_order,
            sep_nnodes = c.cfg.sep_nnodes,
            sep_sparse_tree = sep_sparse_tree_override,
            sep_maxit = c.cfg.sep_maxit,
            sep_tol = c.cfg.sep_tol,
            sep_accept_tol = c.cfg.sep_accept_tol,
            sep_shock_scale = sep_shock_scale_override,
            sep_inv_maxit = c.cfg.sep_inv_maxit,
            sep_inv_step_tol = c.cfg.sep_inv_step_tol,
            sep_inv_resid_tol = c.cfg.sep_inv_resid_tol,
            sep_inv_lambda = c.cfg.sep_inv_lambda,
        )
        status = classify_status(ll, on_failure_loglikelihood)
        if status != "ok"
            err_msg = status == "on_failure_loglikelihood" ?
                "FOM returned on_failure_loglikelihood sentinel ($(on_failure_loglikelihood))." :
                "FOM returned invalid loglikelihood."
            push!(failures, "$(c.case_id): $(err_msg)")
        else
            global ok_cases += 1
        end
    catch err
        status = "error"
        err_msg = sprint(showerror, err)
        push!(failures, "$(c.case_id): $(err_msg)")
    end
    elapsed_s = round(time() - t0; digits = 3)

    entry = Dict{String,Any}(
        "status" => status,
        "elapsed_s" => elapsed_s,
        "fom_loglik" => ll,
        "error" => err_msg,
        "selected_period_indices" => c.selected_idx,
        "evaluation_period_indices" => c.eval_idx,
        "context_period_indices" => c.context_idx,
        "presample_periods" => c.presample_periods,
        "selection_note" => c.selection_note,
        "config" => Dict(string(k) => getfield(c.cfg, k) for k in propertynames(c.cfg)),
    )
    results[c.case_id] = entry
    println("  ", c.case_id, " => status=", status, " ll=", ll, " elapsed_s=", elapsed_s,
            " selected=", c.selected_idx, " pres=", c.presample_periods)
    flush(stdout)
end

payload = Dict{String,Any}(
    "created_at" => string(Dates.now()),
    "script" => basename(@__FILE__),
    "probe_loop_elapsed_s" => round(time() - probe_loop_t0; digits = 3),
    "total_elapsed_s" => round(time() - total_t0; digits = 3),
    "model_setup_elapsed_s" => model_setup_elapsed_s,
    "chain_path" => chain_path,
    "chain_summary_path" => chain_summary_path,
    "chain_summary_used" => chain_summary_used,
    "synthetic_path" => synthetic_path,
    "model" => model_name,
    "period_mask_key" => period_mask_key,
    "gate_mask" => gate_mask,
    "gate_true_periods" => findall(gate_mask),
    "gated_blocks" => String.(gated_blocks),
    "contexts" => contexts,
    "max_eval_periods" => max_eval_periods,
    "include_order1" => include_order1,
    "skip_order0" => skip_order0,
    "sep_sparse_tree_override" => sep_sparse_tree_override,
    "sep_shock_scale_override" => sep_shock_scale_override,
    "planned_cases_count" => length(planned_cases),
    "ok_cases_count" => ok_cases,
    "results" => results,
    "failures" => failures,
)

serialize(out_path, payload)
summary_path = replace(out_path, r"\.jls$" => "_summary.md")
open(summary_path, "w") do io
    println(io, "# HLT SEP FOM Probe Search")
    println(io)
    println(io, "- Created: `", payload["created_at"], "`")
    println(io, "- Model: `", payload["model"], "`")
    println(io, "- Gate true periods: `", payload["gate_true_periods"], "`")
    println(io, "- Chain summary used: `", payload["chain_summary_used"], "`")
    println(io, "- Model setup elapsed (s): `", payload["model_setup_elapsed_s"], "`")
    println(io, "- Probe loop elapsed (s): `", payload["probe_loop_elapsed_s"], "`")
    println(io, "- Total elapsed (s): `", payload["total_elapsed_s"], "`")
    println(io, "- Planned cases: `", payload["planned_cases_count"], "`")
    println(io, "- Skip order-0 cases: `", payload["skip_order0"], "`")
    println(io, "- SEP sparse tree override: `", payload["sep_sparse_tree_override"], "`")
    println(io, "- SEP shock scale override: `", payload["sep_shock_scale_override"], "`")
    println(io, "- OK cases: `", payload["ok_cases_count"], "`")
    println(io)
    println(io, "## Results")
    for case_id in sort!(collect(keys(results)))
        r = results[case_id]
        println(io, "- `", case_id, "`: status=`", r["status"], "`, elapsed_s=`", get(r, "elapsed_s", "n/a"), "`")
        println(io, "  - Selected: `", r["selected_period_indices"], "`")
        println(io, "  - Eval: `", r["evaluation_period_indices"], "`")
        println(io, "  - Context: `", r["context_period_indices"], "`")
        println(io, "  - Presample: `", r["presample_periods"], "`")
        println(io, "  - Config: `", r["config"], "`")
        if get(r, "fom_loglik", nothing) !== nothing
            println(io, "  - FOM loglik: `", r["fom_loglik"], "`")
        end
        if get(r, "error", nothing) !== nothing
            println(io, "  - Error: `", replace(String(r["error"]), '\n' => ' '), "`")
        end
    end
end

println("Probe payload: $out_path")
println("Probe summary: $summary_path")
if !isempty(failures)
    println("Probe failures: $(length(failures)) / $(length(planned_cases))")
    allow_fail || error("Probe search encountered failures: $(join(failures, " | "))")
end
