#!/usr/bin/env julia

# Reduced SW07-HLT validation bridge.
#
# Purpose: define the medium-scale direct nonlinear benchmark needed between
# the Galí hard-ELB validation package and the full 18-parameter HLT
# application.  The default bridge targets the investment/risk-premium block
# that drives the paper's economic mechanism.

using Dates
using Printf
using Serialization
using TOML
using AxisKeys
using MacroModelling

const BRIDGE_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

const BRIDGE_PARAMETER_BLOCKS = Dict(
    "investment_4p" => ["crhob", "crhoqs", "z_eb", "z_eqs"],
    "investment_4p_supported" => ["crhob", "crhoqs", "z_eb", "z_eqs"],
    "investment_curvature_5p" => ["csadjcost", "crhob", "crhoqs", "z_eb", "z_eqs"],
    "price_legacy_3p" => ["cprobp", "cindp", "curvp"],
)

const BRIDGE_DEFAULT_OBSERVABLES = ["dy", "dinve", "labobs", "pinfobs", "robs"]
const BRIDGE_FAILURE_LL = -1.0e12

Base.@kwdef struct BridgeOptions
    stage::String = "design"
    dry_run::Bool = false
    run_id::String = "hlt_bridge_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir::String = joinpath(BRIDGE_REPO_ROOT, ".local_artifacts", "hlt_reduced_bridge_validation")
    parameter_block::String = "investment_4p"
    observables::Vector{String} = copy(BRIDGE_DEFAULT_OBSERVABLES)
    periods::Union{Nothing,Int} = nothing
    grid_axis::Union{Nothing,Int} = nothing
    train_samples::Union{Nothing,Int} = nothing
    direct_eval_points::Union{Nothing,Int} = nothing
    sep_horizon::Union{Nothing,Int} = nothing
    sep_maxit::Union{Nothing,Int} = nothing
    seed::Int = 20260527
end

Base.@kwdef struct BridgeConfig
    stage::String
    periods::Int
    grid_axis::Int
    train_samples::Int
    direct_eval_points::Int
    sep_horizon::Int
    sep_maxit::Int
    sep_accept_tol::Float64
    hmc_warmup::Int
    hmc_draws::Int
    hmc_chains::Int
end

function parse_bridge_bool(value::AbstractString)
    v = lowercase(strip(value))
    v in ("1", "true", "yes", "y", "on") && return true
    v in ("0", "false", "no", "n", "off") && return false
    error("Cannot parse boolean value: $value")
end

function parse_bridge_csv(value::AbstractString)
    vals = String[]
    for raw in split(value, ",")
        token = strip(raw)
        isempty(token) || push!(vals, token)
    end
    return vals
end

function parse_bridge_args(args::Vector{String})
    opts = BridgeOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        if occursin("=", arg)
            key, value = split(arg[3:end], "=", limit = 2)
            values[key] = value
        else
            values[arg[3:end]] = "true"
        end
    end

    stage = get(values, "stage", opts.stage)
    stage in ("design", "smoke", "pilot", "full") ||
        error("Unknown --stage=$stage. Use design, smoke, pilot, or full.")

    parameter_block = get(values, "parameter-block", opts.parameter_block)
    haskey(BRIDGE_PARAMETER_BLOCKS, parameter_block) ||
        error("Unknown --parameter-block=$parameter_block. Choices: $(sort(collect(keys(BRIDGE_PARAMETER_BLOCKS))))")

    return BridgeOptions(
        stage = stage,
        dry_run = parse_bridge_bool(get(values, "dry-run", string(opts.dry_run))),
        run_id = get(values, "run-id", opts.run_id),
        out_dir = get(values, "out-dir", opts.out_dir),
        parameter_block = parameter_block,
        observables = haskey(values, "observables") ? parse_bridge_csv(values["observables"]) : opts.observables,
        periods = haskey(values, "periods") ? parse(Int, values["periods"]) : nothing,
        grid_axis = haskey(values, "grid-axis") ? parse(Int, values["grid-axis"]) : nothing,
        train_samples = haskey(values, "train-samples") ? parse(Int, values["train-samples"]) : nothing,
        direct_eval_points = haskey(values, "direct-eval-points") ? parse(Int, values["direct-eval-points"]) : nothing,
        sep_horizon = haskey(values, "sep-horizon") ? parse(Int, values["sep-horizon"]) : nothing,
        sep_maxit = haskey(values, "sep-maxit") ? parse(Int, values["sep-maxit"]) : nothing,
        seed = parse(Int, get(values, "seed", string(opts.seed))),
    )
end

function stage_defaults(stage::String)
    if stage == "design"
        return BridgeConfig(
            stage = stage,
            periods = 24,
            grid_axis = 3,
            train_samples = 0,
            direct_eval_points = 0,
            sep_horizon = 8,
            sep_maxit = 200,
            sep_accept_tol = 1e-2,
            hmc_warmup = 0,
            hmc_draws = 0,
            hmc_chains = 0,
        )
    elseif stage == "smoke"
        return BridgeConfig(
            stage = stage,
            periods = 12,
            grid_axis = 3,
            train_samples = 64,
            direct_eval_points = 5,
            sep_horizon = 6,
            sep_maxit = 200,
            sep_accept_tol = 1e-2,
            hmc_warmup = 0,
            hmc_draws = 0,
            hmc_chains = 0,
        )
    elseif stage == "pilot"
        return BridgeConfig(
            stage = stage,
            periods = 24,
            grid_axis = 3,
            train_samples = 512,
            direct_eval_points = 25,
            sep_horizon = 8,
            sep_maxit = 300,
            sep_accept_tol = 5e-3,
            hmc_warmup = 100,
            hmc_draws = 200,
            hmc_chains = 1,
        )
    elseif stage == "full"
        return BridgeConfig(
            stage = stage,
            periods = 40,
            grid_axis = 5,
            train_samples = 4096,
            direct_eval_points = 125,
            sep_horizon = 10,
            sep_maxit = 500,
            sep_accept_tol = 1e-3,
            hmc_warmup = 500,
            hmc_draws = 1000,
            hmc_chains = 4,
        )
    else
        error("Unknown stage: $stage")
    end
end

function apply_overrides(cfg::BridgeConfig, opts::BridgeOptions)
    return BridgeConfig(
        stage = cfg.stage,
        periods = something(opts.periods, cfg.periods),
        grid_axis = something(opts.grid_axis, cfg.grid_axis),
        train_samples = something(opts.train_samples, cfg.train_samples),
        direct_eval_points = something(opts.direct_eval_points, cfg.direct_eval_points),
        sep_horizon = something(opts.sep_horizon, cfg.sep_horizon),
        sep_maxit = something(opts.sep_maxit, cfg.sep_maxit),
        sep_accept_tol = cfg.sep_accept_tol,
        hmc_warmup = cfg.hmc_warmup,
        hmc_draws = cfg.hmc_draws,
        hmc_chains = cfg.hmc_chains,
    )
end

function bridge_git_commit()
    try
        return readchomp(`git -C $BRIDGE_REPO_ROOT rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function bridge_run_dir(opts::BridgeOptions)
    return joinpath(opts.out_dir, opts.run_id)
end

function bridge_manifest(opts::BridgeOptions, cfg::BridgeConfig)
    theta_names = BRIDGE_PARAMETER_BLOCKS[opts.parameter_block]
    grid_points = cfg.grid_axis ^ length(theta_names)
    return Dict{String,Any}(
        "created_at" => string(Dates.now()),
        "git_commit" => bridge_git_commit(),
        "stage" => opts.stage,
        "dry_run" => opts.dry_run,
        "model" => "Smets_Wouters_2007_HLT_obc",
        "purpose" => "Reduced SW07-HLT direct nonlinear validation bridge between Gali and full 18-parameter HLT.",
        "parameter_block" => opts.parameter_block,
        "theta_names" => theta_names,
        "observables" => opts.observables,
        "periods" => cfg.periods,
        "grid_axis" => cfg.grid_axis,
        "grid_points" => grid_points,
        "train_samples" => cfg.train_samples,
        "direct_eval_points" => cfg.direct_eval_points,
        "sep_horizon" => cfg.sep_horizon,
        "sep_maxit" => cfg.sep_maxit,
        "sep_accept_tol" => cfg.sep_accept_tol,
        "hmc_warmup" => cfg.hmc_warmup,
        "hmc_draws" => cfg.hmc_draws,
        "hmc_chains" => cfg.hmc_chains,
        "seed" => opts.seed,
        "benchmark_ladder" => [
            "finite direct SEP and ROM1/surrogate objectives at true parameter and posterior-mode anchors",
            "local direct SEP objective grid on reduced investment block",
            "surrogate objective grid using the same ROM1-inversion shocks and measurement-error scaling",
            "optional short matched HMC after grid posterior overlap passes",
        ],
        "acceptance_criteria" => [
            "all direct SEP grid anchors finite or explicitly classified with recovery diagnostics",
            "surrogate residual RRMSE below 0.005 on bridge observables and below 0.01 on all audited observables",
            "direct and surrogate reduced-block posterior means within two combined MCSE units in matched HMC stage",
            "direct and surrogate 90 percent intervals overlap for every bridge parameter",
            "ROM1 posterior differs from direct nonlinear in the direction predicted by the investment-channel decomposition",
        ],
        "artifact_schema" => [
            "manifest.toml",
            "SUMMARY.md",
            "direct_grid_payload.jls",
            "surrogate_grid_payload.jls",
            "comparison_table.tex",
            "posterior_draw_audit.jls",
        ],
    )
end

function write_bridge_manifest(path::String, manifest::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, manifest)
    end
end

function write_bridge_summary(path::String, manifest::Dict{String,Any})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Reduced SW07-HLT Validation Bridge")
        println(io)
        println(io, "- Stage: `$(manifest["stage"])`")
        println(io, "- Dry run: `$(manifest["dry_run"])`")
        println(io, "- Model: `$(manifest["model"])`")
        println(io, "- Parameter block: `$(manifest["parameter_block"])`")
        println(io, "- Parameters: `$(join(manifest["theta_names"], ", "))`")
        println(io, "- Observables: `$(join(manifest["observables"], ", "))`")
        println(io, "- Periods: `$(manifest["periods"])`")
        println(io, "- Grid points: `$(manifest["grid_points"])`")
        println(io, "- Direct evaluation points: `$(manifest["direct_eval_points"])`")
        println(io, "- SEP horizon/maxit/accept_tol: `$(manifest["sep_horizon"]) / $(manifest["sep_maxit"]) / $(manifest["sep_accept_tol"])`")
        println(io, "- HMC chains/warmup/draws: `$(manifest["hmc_chains"]) / $(manifest["hmc_warmup"]) / $(manifest["hmc_draws"])`")
        println(io, "- Git commit: `$(manifest["git_commit"])`")
        println(io)
        println(io, "## Why This Bridge Exists")
        println(io)
        println(io, "The Galí hard-ELB package validates the residual-learning and inversion machinery in a small OBC model. The full HLT direct SEP-HMC comparison is too expensive in its present finite-difference form. This bridge is the intermediate Econometrica-grade benchmark: a reduced HLT block close to the investment mechanism, with direct SEP evaluations on a small local posterior surface.")
        println(io)
        println(io, "## Benchmark Ladder")
        for item in manifest["benchmark_ladder"]
            println(io, "- $item")
        end
        println(io)
        println(io, "## Acceptance Criteria")
        for item in manifest["acceptance_criteria"]
            println(io, "- $item")
        end
        println(io)
        println(io, "## Next Implementation Step")
        println(io)
        if manifest["stage"] == "design" || manifest["dry_run"]
            println(io, "Run `--stage=smoke` to build the tiny HLT steady-state panel and verify finite ROM1/inversion and direct SEP/inversion likelihoods on the reduced grid.")
        elseif manifest["stage"] == "smoke"
            println(io, "If the smoke status is `pass`, expand the reduced direct grid and train the bridge ROM1-residual surrogate before launching any HMC.")
        else
            println(io, "Use the previous stage's diagnostics to decide whether the next heavier bridge stage is justified.")
        end
    end
end

function bridge_symbol_vector(values::Vector{String})
    return Symbol.(values)
end

function bridge_load_hlt_model()
    return load_hlt_model(BRIDGE_REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
end

function bridge_axiskeys(x)
    return axiskeys(x)
end

function bridge_keyed_array(values; Variable, Time)
    return KeyedArray(values; Variable = Variable, Time = Time)
end

function bridge_macro_modelling()
    return MacroModelling
end

function bridge_extract_steady_state(model, observables::Vector{Symbol})
    mm = bridge_macro_modelling()
    ss = mm.get_steady_state(model; derivatives = false, return_variables_only = true, silent = true)
    ss_keys = collect(bridge_axiskeys(ss)[1])
    ss_vals = collect(ss)
    vals = Float64[]
    for obs in observables
        idx = findfirst(==(obs), ss_keys)
        idx === nothing && error("Observable $(obs) not found in HLT steady-state output.")
        push!(vals, Float64(ss_vals[idx]))
    end
    return vals
end

function bridge_inject_theta(base_params::AbstractVector, model, theta_names::Vector{Symbol}, theta_vals::AbstractVector)
    params = copy(base_params)
    idx = indexin(theta_names, model.parameters)
    if any(isnothing, idx)
        missing = theta_names[findall(isnothing, idx)]
        error("Theta names not found in $(model.model_name) parameters: $(missing)")
    end
    for (i, j) in enumerate(Int.(idx))
        params[j] = theta_vals[i]
    end
    return params
end

function bridge_axis_values(name::Symbol, center::Float64, grid_axis::Int)
    grid_axis >= 1 || error("--grid-axis must be positive.")
    grid_axis == 1 && return [center]

    if name in (:crhob, :crhoqs)
        lo = max(0.05, center - 0.05)
        hi = min(0.98, center + 0.05)
    elseif startswith(String(name), "z_")
        lo = max(1.0e-8, center * 0.90)
        hi = center * 1.10
    elseif name == :csadjcost
        lo = max(1.0e-8, center * 0.90)
        hi = center * 1.10
    elseif name in (:cprobp, :cindp)
        lo = max(0.01, center - 0.05)
        hi = min(0.99, center + 0.05)
    elseif name == :curvp
        lo = max(1.0e-8, center * 0.90)
        hi = center * 1.10
    else
        lo = center * 0.95
        hi = center * 1.05
    end

    return collect(range(lo, hi; length = grid_axis))
end

function bridge_local_grid(theta_names::Vector{Symbol}, theta0::Vector{Float64}, grid_axis::Int)
    axes = [bridge_axis_values(name, theta0[i], grid_axis) for (i, name) in enumerate(theta_names)]
    return [Float64[x for x in tup] for tup in Iterators.product(axes...)]
end

function bridge_sep_settings(cfg::BridgeConfig)
    return Dict{String,Any}(
        "sep_periods" => cfg.sep_horizon,
        "sep_order" => 1,
        "sep_nnodes" => 3,
        "sep_sparse_tree" => true,
        "sep_maxit" => cfg.sep_maxit,
        "sep_tol" => 1.0e-4,
        "sep_accept_tol" => cfg.sep_accept_tol,
        "sep_shock_scale" => 0.5,
        "sep_inv_maxit" => 1,
        "sep_inv_step_tol" => 1.0e-4,
        "sep_inv_resid_tol" => 1.0e-3,
        "sep_inv_lambda" => 1.0e-3,
        "sep_inv_predict_tol" => 1.0e-10,
        "sep_inv_logdet_method" => :exact,
        "sep_inv_logdet_sv_tol" => sqrt(eps(Float64)),
    )
end

function bridge_get_sep_diagnostics()
    mm = bridge_macro_modelling()
    if isdefined(mm, :get_sep_inversion_last_diagnostics)
        try
            diag = getfield(mm, :get_sep_inversion_last_diagnostics)()
            diag === nothing && return nothing
            return diag
        catch err
            return Dict{String,Any}(
                "status" => "diagnostics_error",
                "error" => sprint(showerror, err),
            )
        end
    end
    return nothing
end

function bridge_reset_sep_diagnostics!()
    mm = bridge_macro_modelling()
    if isdefined(mm, :reset_sep_inversion_last_diagnostics!)
        try
            getfield(mm, :reset_sep_inversion_last_diagnostics!)()
        catch
        end
    end
    return nothing
end

function bridge_loglikelihood(model,
                              obs_ka,
                              params;
                              algorithm::Symbol,
                              sep_settings::Union{Nothing,Dict{String,Any}} = nothing)
    mm = bridge_macro_modelling()
    bridge_reset_sep_diagnostics!()
    if algorithm == :first_order
        return mm.get_loglikelihood(
            model,
            obs_ka,
            params;
            algorithm = :first_order,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = BRIDGE_FAILURE_LL,
            presample_periods = 0,
        )
    elseif algorithm == :stochastic_extended_path
        sep_settings === nothing && error("SEP settings required for direct SEP likelihood.")
        return mm.get_loglikelihood(
            model,
            obs_ka,
            params;
            algorithm = :stochastic_extended_path,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = BRIDGE_FAILURE_LL,
            presample_periods = 0,
            sep_periods = sep_settings["sep_periods"],
            sep_order = sep_settings["sep_order"],
            sep_nnodes = sep_settings["sep_nnodes"],
            sep_sparse_tree = sep_settings["sep_sparse_tree"],
            sep_maxit = sep_settings["sep_maxit"],
            sep_tol = sep_settings["sep_tol"],
            sep_accept_tol = sep_settings["sep_accept_tol"],
            sep_shock_scale = sep_settings["sep_shock_scale"],
            sep_inv_maxit = sep_settings["sep_inv_maxit"],
            sep_inv_step_tol = sep_settings["sep_inv_step_tol"],
            sep_inv_resid_tol = sep_settings["sep_inv_resid_tol"],
            sep_inv_lambda = sep_settings["sep_inv_lambda"],
            sep_inv_predict_tol = sep_settings["sep_inv_predict_tol"],
            sep_inv_logdet_method = sep_settings["sep_inv_logdet_method"],
            sep_inv_logdet_sv_tol = sep_settings["sep_inv_logdet_sv_tol"],
        )
    else
        error("Unsupported bridge likelihood algorithm: $(algorithm)")
    end
end

function bridge_classify_ll(ll)
    if !(ll isa Real) || !isfinite(Float64(ll))
        return "nonfinite"
    elseif Float64(ll) == BRIDGE_FAILURE_LL
        return "failure_sentinel"
    else
        return "ok"
    end
end

function bridge_smoke_status(results::Vector{Dict{String,Any}})
    direct_statuses = String[get(r, "direct_status", "not_evaluated") for r in results if haskey(r, "direct_status")]
    linear_statuses = String[get(r, "linear_status", "not_evaluated") for r in results]
    linear_ok = all(==("ok"), linear_statuses)
    direct_ok = !isempty(direct_statuses) && all(==("ok"), direct_statuses)
    return linear_ok && direct_ok ? "pass" : "fail"
end

function write_bridge_smoke_table(path::String, theta_names::Vector{Symbol}, results::Vector{Dict{String,Any}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "\\begin{tabular}{lrrr}")
        println(io, "\\toprule")
        println(io, "Cell & Linear LL & Direct SEP LL & Direct Status \\\\")
        println(io, "\\midrule")
        for (i, r) in enumerate(results)
            direct = haskey(r, "direct_loglik") ? @sprintf("%.3f", Float64(r["direct_loglik"])) : "--"
            linear = @sprintf("%.3f", Float64(r["linear_loglik"]))
            status = get(r, "direct_status", "not evaluated")
            println(io, "$(i) & $(linear) & $(direct) & $(status) \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function append_bridge_smoke_summary(path::String,
                                     smoke::Dict{String,Any},
                                     theta_names::Vector{Symbol},
                                     theta0::Vector{Float64},
                                     ss_values::Vector{Float64})
    open(path, "a") do io
        println(io)
        println(io, "## Smoke Execution Result")
        println(io)
        println(io, "- Status: `$(smoke["status"])`")
        println(io, "- Linear finite cells: `$(smoke["linear_ok_count"]) / $(smoke["grid_points"])`")
        println(io, "- Direct finite cells: `$(smoke["direct_ok_count"]) / $(smoke["direct_eval_points"])`")
        println(io, "- Elapsed seconds: `$(smoke["elapsed_s"])`")
        println(io, "- Payload: `$(smoke["payload_file"])`")
        println(io, "- Table: `$(smoke["table_file"])`")
        println(io)
        println(io, "### Baseline Theta")
        for (name, val) in zip(theta_names, theta0)
            println(io, "- `$(name)`: `$(@sprintf("%.8g", val))`")
        end
        println(io)
        println(io, "### Steady-State Observation Values")
        for (obs, val) in zip(smoke["observables"], ss_values)
            println(io, "- `$(obs)`: `$(@sprintf("%.8g", val))`")
        end
        if !isempty(get(smoke, "failures", String[]))
            println(io)
            println(io, "### Failure Diagnostics")
            for msg in smoke["failures"]
                println(io, "- $(msg)")
            end
        end
    end
end

function run_bridge_smoke!(opts::BridgeOptions,
                           cfg::BridgeConfig,
                           manifest::Dict{String,Any},
                           run_dir::String)
    t0 = time()
    model = bridge_load_hlt_model()
    theta_names = Symbol.(manifest["theta_names"])
    observables = bridge_symbol_vector(opts.observables)
    obs_values = bridge_extract_steady_state(model, observables)
    obs_panel = repeat(reshape(obs_values, :, 1), 1, cfg.periods)
    obs_ka = bridge_keyed_array(obs_panel; Variable = observables, Time = 1:cfg.periods)

    theta_idx = indexin(theta_names, model.parameters)
    any(isnothing, theta_idx) && error("Bridge parameter block has names not found in HLT model.")
    theta0 = Float64[model.parameter_values[Int(i)] for i in theta_idx]
    grid = bridge_local_grid(theta_names, theta0, cfg.grid_axis)
    direct_n = min(cfg.direct_eval_points, length(grid))
    sep_settings = bridge_sep_settings(cfg)

    results = Dict{String,Any}[]
    failures = String[]
    for (i, theta) in enumerate(grid)
        params = bridge_inject_theta(model.parameter_values, model, theta_names, theta)
        cell = Dict{String,Any}(
            "cell" => i,
            "theta" => Dict(String(name) => theta[j] for (j, name) in enumerate(theta_names)),
        )

        linear_ll = try
            bridge_loglikelihood(model, obs_ka, params; algorithm = :first_order)
        catch err
            push!(failures, "linear cell $(i): $(sprint(showerror, err))")
            NaN
        end
        cell["linear_loglik"] = linear_ll
        cell["linear_status"] = bridge_classify_ll(linear_ll)

        if i <= direct_n
            direct_ll = try
                bridge_loglikelihood(
                    model,
                    obs_ka,
                    params;
                    algorithm = :stochastic_extended_path,
                    sep_settings = sep_settings,
                )
            catch err
                push!(failures, "direct cell $(i): $(sprint(showerror, err))")
                NaN
            end
            cell["direct_loglik"] = direct_ll
            cell["direct_status"] = bridge_classify_ll(direct_ll)
            diag = bridge_get_sep_diagnostics()
            diag !== nothing && (cell["sep_inversion_diagnostics"] = diag)
        end

        push!(results, cell)
    end

    payload_file = joinpath(run_dir, "direct_grid_payload.jls")
    table_file = joinpath(run_dir, "comparison_table.tex")
    smoke = Dict{String,Any}(
        "status" => bridge_smoke_status(results),
        "stage" => opts.stage,
        "observables" => opts.observables,
        "theta_names" => manifest["theta_names"],
        "theta0" => theta0,
        "steady_state_observations" => Dict(String(obs) => obs_values[i] for (i, obs) in enumerate(observables)),
        "grid_points" => length(grid),
        "direct_eval_points" => direct_n,
        "linear_ok_count" => count(r -> get(r, "linear_status", "") == "ok", results),
        "direct_ok_count" => count(r -> get(r, "direct_status", "") == "ok", results),
        "sep_settings" => sep_settings,
        "results" => results,
        "failures" => failures,
        "elapsed_s" => round(time() - t0; digits = 3),
        "payload_file" => payload_file,
        "table_file" => table_file,
    )
    Serialization.serialize(payload_file, smoke)
    write_bridge_smoke_table(table_file, theta_names, results)
    manifest["smoke_status"] = smoke["status"]
    manifest["smoke_elapsed_s"] = smoke["elapsed_s"]
    manifest["smoke_payload"] = payload_file
    manifest["smoke_table"] = table_file
    manifest["smoke_linear_ok_count"] = smoke["linear_ok_count"]
    manifest["smoke_direct_ok_count"] = smoke["direct_ok_count"]
    append_bridge_smoke_summary(joinpath(run_dir, "SUMMARY.md"), smoke, theta_names, theta0, obs_values)
    return smoke
end

function run_bridge(opts::BridgeOptions)
    cfg = apply_overrides(stage_defaults(opts.stage), opts)
    run_dir = bridge_run_dir(opts)
    mkpath(run_dir)
    manifest = bridge_manifest(opts, cfg)
    write_bridge_manifest(joinpath(run_dir, "manifest.toml"), manifest)
    write_bridge_summary(joinpath(run_dir, "SUMMARY.md"), manifest)
    println("Wrote bridge manifest: $(joinpath(run_dir, "manifest.toml"))")
    println("Wrote bridge summary: $(joinpath(run_dir, "SUMMARY.md"))")
    if opts.stage != "design" && !opts.dry_run
        if opts.stage == "smoke"
            smoke = run_bridge_smoke!(opts, cfg, manifest, run_dir)
            write_bridge_manifest(joinpath(run_dir, "manifest.toml"), manifest)
            println("Wrote smoke payload: $(smoke["payload_file"])")
            println("Smoke status: $(smoke["status"])")
        else
            error("Executable $opts.stage stage is intentionally not launched yet. Run --stage=smoke first to prove finite direct-grid support.")
        end
    end
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bridge(parse_bridge_args(ARGS))
end
