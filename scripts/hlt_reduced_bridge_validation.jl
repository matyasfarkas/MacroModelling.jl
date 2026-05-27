#!/usr/bin/env julia

# Reduced SW07-HLT validation bridge scaffold.
#
# Purpose: define the medium-scale direct nonlinear benchmark needed between
# the Galí hard-ELB validation package and the full 18-parameter HLT
# application.  The default bridge targets the investment/risk-premium block
# that drives the paper's economic mechanism.

using Dates
using Printf
using TOML

const BRIDGE_REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

const BRIDGE_PARAMETER_BLOCKS = Dict(
    "investment_4p" => ["crhob", "crhoqs", "z_eb", "z_eqs"],
    "investment_curvature_5p" => ["csadjcost", "crhob", "crhoqs", "z_eb", "z_eqs"],
    "price_legacy_3p" => ["cprobp", "cindp", "curvp"],
)

const BRIDGE_DEFAULT_OBSERVABLES = ["dyobs", "dinveobs", "labobs", "pinfobs", "robs"]

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
        println(io, "Implement the smoke stage by reusing the current HLT quick-smoke artifacts, replacing the legacy price block with the reduced investment block, and evaluating direct SEP predictions on the local grid before launching any HMC.")
    end
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
        error("Executable $opts.stage stage is intentionally not launched yet. First inspect the design manifest and wire the direct-grid evaluator.")
    end
    return manifest
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bridge(parse_bridge_args(ARGS))
end
