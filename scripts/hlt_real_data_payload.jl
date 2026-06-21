#!/usr/bin/env julia
using CSV
using DataFrames
using AxisKeys
using MacroModelling
using Serialization
using Statistics

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

const SOURCE_OBSERVABLES = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
const MODEL_OBSERVABLES = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
const THETA_NAMES_3 = [:cprobp, :cindp, :curvp]
const THETA_NAMES_18 = [:crhoa, :crhob, :crhog, :crhoqs, :crhopinf, :crhow, :crhoms,
                         :z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em,
                         :cprobp, :cindp, :curvp, :cprobw]

script_repo_root() = normpath(joinpath(@__DIR__, ".."))

function normalize_column_symbols(df::DataFrame)
    old = names(df)
    new = Symbol.(strip.(String.(old)))
    rename!(df, old .=> new)
    return df
end

function require_columns(df::DataFrame, cols::Vector{Symbol})
    present = Set(Symbol.(names(df)))
    missing_cols = [c for c in cols if !(c in present)]
    isempty(missing_cols) || error("Missing required columns in input CSV: $(missing_cols)")
    return nothing
end

function state_index(model)
    return sort(unique(vcat(
        model.timings.past_not_future_and_mixed_idx,
        model.timings.future_not_past_and_mixed_idx,
    )))
end

function default_shock_sigmas(model; shock_scaling::Symbol = :none, shock_scale::Float64 = 1.0)
    sigmas = zeros(Float64, length(model.exo))
    obc_mask = contains.(string.(model.exo), "ᵒᵇᶜ")
    for i in eachindex(model.exo)
        if !obc_mask[i]
            sigmas[i] = shock_scaling == :parameter ?
                MacroModelling.sep_irf_shock_std(model, model.exo[i]; warn_missing = false) : 1.0
        end
    end
    sigmas .*= shock_scale
    all(isfinite, sigmas) || error("Computed shock_sigmas contain non-finite values.")
    all(sigmas .>= 0) || error("Computed shock_sigmas contain negative values.")
    return sigmas
end

function make_obs_sigma(obs_data::AbstractMatrix;
                        mode::Symbol = :data_std,
                        scale::Float64 = 0.1,
                        floor::Float64 = 1e-4)
    floor >= 0 || error("obs_sigma_floor must be nonnegative, got $floor")
    scale > 0 || error("obs_sigma_scale must be positive, got $scale")

    obs_sigma = if mode == :data_std
        vec(Statistics.std(obs_data, dims = 2)) .* scale
    elseif mode == :constant
        fill(scale, size(obs_data, 1))
    else
        error("Unknown obs_sigma_mode=$mode. Use :data_std or :constant.")
    end

    obs_sigma = max.(obs_sigma, floor)
    all(isfinite, obs_sigma) || error("obs_sigma contains non-finite values.")
    all(obs_sigma .> 0) || error("obs_sigma must be strictly positive.")
    return obs_sigma
end

function build_hlt_real_data_payload(; csv_path::AbstractString,
                                       model_name::AbstractString = "Smets_Wouters_2007_HLT_obc",
                                       theta_names::Vector{Symbol} = THETA_NAMES_3,
                                       sample_start::Int = 47,
                                       sample_end::Int = 290,
                                       prefix_end::Int = 46,
                                       shock_scaling::Symbol = :none,
                                       shock_scale::Float64 = 1.0,
                                       obs_sigma_mode::Symbol = :data_std,
                                       obs_sigma_scale::Float64 = 0.1,
                                       obs_sigma_floor::Float64 = 1e-4,
                                       state_init_filter::Symbol = :kalman,
                                       state_init_algorithm::Symbol = :first_order)
    isfile(csv_path) || error("CSV file not found: $csv_path")
    sample_start >= 1 || error("sample_start must be >= 1")
    sample_end >= sample_start || error("sample_end must be >= sample_start")
    prefix_end >= 1 || error("prefix_end must be >= 1")
    prefix_end < sample_start || error("prefix_end ($prefix_end) must be strictly less than sample_start ($sample_start)")

    model = load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)
    df = CSV.read(csv_path, DataFrame)
    normalize_column_symbols(df)
    require_columns(df, SOURCE_OBSERVABLES)

    T_total = nrow(df)
    sample_end <= T_total || error("sample_end=$sample_end exceeds data length $T_total")
    prefix_end <= T_total || error("prefix_end=$prefix_end exceeds data length $T_total")

    data_source = KeyedArray(
        Matrix(df)',
        Variable = Symbol.(names(df)),
        Time = 1:T_total,
    )

    sample_idx = sample_start:sample_end
    prefix_idx = 1:prefix_end

    sample_data = data_source(SOURCE_OBSERVABLES, sample_idx)
    sample_data = rekey(sample_data, :Variable => MODEL_OBSERVABLES)
    obs_data = Matrix(sample_data)

    prefix_data = data_source(SOURCE_OBSERVABLES, prefix_idx)
    prefix_data = rekey(prefix_data, :Variable => MODEL_OBSERVABLES)
    prefix_obs = Matrix(prefix_data)

    s_idx = state_index(model)
    state_names = model.var[s_idx]
    obs_idx = indexin(MODEL_OBSERVABLES, model.var)
    any(isnothing, obs_idx) && error("MODEL_OBSERVABLES not found in model variable list.")
    obs_idx = Int.(obs_idx)

    # Avoid world-age issues from runtime-generated model internals.
    s0 = Base.invokelatest(
        MacroModelling.linear_filter_initial_state,
        model,
        prefix_obs,
        MODEL_OBSERVABLES,
        state_names;
        parameters = model.parameter_values,
        filter = state_init_filter,
        algorithm = state_init_algorithm,
        label = "Real-data initial-state variables",
    )

    T_obs = size(obs_data, 2)
    shock_sigmas = default_shock_sigmas(model; shock_scaling = shock_scaling, shock_scale = shock_scale)
    shocks = zeros(Float64, length(model.exo), T_obs)
    obs_sigma = make_obs_sigma(obs_data; mode = obs_sigma_mode, scale = obs_sigma_scale, floor = obs_sigma_floor)

    payload = Dict(
        "model" => String(model.model_name),
        "source_csv" => String(csv_path),
        "source_columns" => SOURCE_OBSERVABLES,
        "observables" => MODEL_OBSERVABLES,
        "sample_idx" => collect(sample_idx),
        "sample_idx_requested" => collect(sample_idx),
        "generated_periods" => T_obs,
        "requested_periods" => T_obs,
        "burn_in" => 0,
        "s0" => s0,
        "state_idx" => s_idx,
        "state_names" => state_names,
        "state_definition" => "past_not_future_and_mixed + future_not_past_and_mixed",
        "obs_idx" => obs_idx,
        "obs_data" => obs_data,
        "obs_sigma" => obs_sigma,
        "shocks" => shocks,
        "shock_sigmas" => shock_sigmas,
        "shock_scaling" => shock_scaling,
        "shock_scale" => shock_scale,
        "theta_names" => theta_names,
        "theta_true" => nothing,
        "theta_baseline" => model.parameter_values[indexin(theta_names, model.parameters)],
        "prefix_idx" => collect(prefix_idx),
        "prefix_init_filter" => String(state_init_filter),
        "prefix_init_algorithm" => String(state_init_algorithm),
        "obs_sigma_mode" => String(obs_sigma_mode),
        "obs_sigma_scale" => obs_sigma_scale,
        "obs_sigma_floor" => obs_sigma_floor,
    )

    # Required by load_hlt_synthetic_scenario contract:
    # obs_data, s0, shocks, theta_true
    return payload
end

function write_payload(payload::Dict, out_path::AbstractString)
    mkpath(dirname(out_path))
    serialize(out_path, payload)
    return out_path
end

function main(args)
    csv_path = parse_arg_string(args, "--csv", "")
    csv_path == "" && error("Usage: julia hlt_real_data_payload.jl --csv=<path> [--out=<path>] [--sample-start=47 --sample-end=290 --prefix-end=46] [--use-obc|--no-obc]")

    out_path = parse_arg_string(args, "--out", "")
    sample_start = parse_arg_int(args, "--sample-start", 47)
    sample_end = parse_arg_int(args, "--sample-end", 290)
    prefix_end = parse_arg_int(args, "--prefix-end", 46)
    shock_scaling = parse_arg_symbol(args, "--shock-scaling", :none)
    shock_scale = parse_arg_float(args, "--shock-scale", 1.0)
    obs_sigma_mode = parse_arg_symbol(args, "--obs-sigma-mode", :data_std)
    obs_sigma_scale = parse_arg_float(args, "--obs-sigma-scale", 0.1)
    obs_sigma_floor = parse_arg_float(args, "--obs-sigma-floor", 1e-4)
    state_init_filter = parse_arg_symbol(args, "--state-init-filter", :kalman)
    state_init_algorithm = parse_arg_symbol(args, "--state-init-algorithm", :first_order)

    force_obc = "--use-obc" in args
    force_no_obc = "--no-obc" in args
    force_obc && force_no_obc && error("Specify only one of --use-obc or --no-obc.")
    model_name = force_no_obc ? "Smets_Wouters_2007_HLT" : "Smets_Wouters_2007_HLT_obc"

    theta_set = parse_arg_string(args, "--theta-set", "3")
    theta_names = if theta_set == "18"
        copy(THETA_NAMES_18)
    elseif theta_set == "3"
        copy(THETA_NAMES_3)
    else
        error("Unknown --theta-set=$theta_set. Use 3 or 18.")
    end

    payload = build_hlt_real_data_payload(
        csv_path = csv_path,
        model_name = model_name,
        theta_names = theta_names,
        sample_start = sample_start,
        sample_end = sample_end,
        prefix_end = prefix_end,
        shock_scaling = shock_scaling,
        shock_scale = shock_scale,
        obs_sigma_mode = obs_sigma_mode,
        obs_sigma_scale = obs_sigma_scale,
        obs_sigma_floor = obs_sigma_floor,
        state_init_filter = state_init_filter,
        state_init_algorithm = state_init_algorithm,
    )

    if out_path == ""
        out_path = joinpath(dirname(csv_path), "hlt_real_data_payload.jls")
    end
    write_payload(payload, out_path)

    println("Real-data payload generated")
    println("  CSV: $(csv_path)")
    println("  Output: $(out_path)")
    println("  Model: $(payload["model"])")
    println("  Observables: $(payload["observables"])")
    println("  Sample idx: $(first(payload["sample_idx"])):$(last(payload["sample_idx"])) (T=$(length(payload["sample_idx"])))")
    println("  obs_data size: $(size(payload["obs_data"]))")
    println("  s0 length: $(length(payload["s0"]))")
    println("  shock matrix size: $(size(payload["shocks"]))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
