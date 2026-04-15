# RBCII filter diagnostic: KF vs inversion on synthetic SEP data
#
# Uses Dynare .mat efficiency series to infer shocks, simulates SEP data,
# then compares log-likelihood/runtime and shock-recovery accuracy between:
#   - Kalman filter (linear, first order)
#   - Inversion filter (linear, first order)

using MacroModelling
using MAT
using AxisKeys
using Statistics
using Printf

include("../models/RBCII_Dynare.jl")

function sep_validation_root()
    candidates = (
        joinpath(@__DIR__, "..", "test", "fixtures", "sep_validation"),
        joinpath(@__DIR__, "..", "tests", "sep_validation"),
    )
    for path in candidates
        isdir(path) && return path
    end
    return first(candidates)
end

const DATA_DIR = joinpath(sep_validation_root(), "sep_simulation_data", "accuracy-sc")

function parse_arg(args::Vector{String}, name::String, default)
    prefix = name * "="
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "=", limit = 2)[2]
        end
    end
    return default
end

function parse_int(args, name, default)
    val = parse_arg(args, name, nothing)
    return val === nothing ? default : parse(Int, val)
end

function parse_float(args, name, default)
    val = parse_arg(args, name, nothing)
    return val === nothing ? default : parse(Float64, val)
end

function parse_bool(args, name, default)
    val = parse_arg(args, name, nothing)
    return val === nothing ? default : (val in ("1", "true", "yes", "on"))
end

function parse_list(args, name, default::Vector{String})
    val = parse_arg(args, name, nothing)
    return val === nothing ? default : split(val, ",")
end

struct DynareDSeries
    names::Vector{String}
    data::Matrix{Float64}
end

function load_dynare_dseries(path::AbstractString)
    isfile(path) || error("Missing Dynare .mat file: $path")
    d = matread(path)
    names = vec(String.(d["NAMES__"]))
    data = Matrix{Float64}(d["DATA__"])
    return DynareDSeries(names, data)
end

function get_series(ds::DynareDSeries, name::AbstractString)
    idx = findfirst(==(name), ds.names)
    idx === nothing && error("Series \"$name\" not found in $(join(ds.names, ", ")).")
    return vec(ds.data[:, idx])
end

function implied_shocks_from_efficiency(eff::AbstractVector{<:Real}; rho::Real, sigma::Real)
    n = length(eff)
    n < 2 && error("Need at least 2 observations to infer shocks.")
    shocks = zeros(Float64, n - 1)
    for t in 2:n
        shocks[t - 1] = (eff[t] - rho * eff[t - 1]) / sigma
    end
    return shocks
end

function dynare_rbcii_path(order::Int; sigma_tag::AbstractString, algo::Int, hybrid::Int)
    fname = "rbcii-$(sigma_tag)-sep-$(order)-algo-$(algo)-hybrid-$(hybrid).mat"
    return joinpath(DATA_DIR, fname)
end

function main()
    m = RBCII_Dynare

    # defaults
    sigma_tag = parse_arg(ARGS, "--sigma-tag", "007")
    dynare_order = parse_int(ARGS, "--dynare-order", 1)
    algo = parse_int(ARGS, "--algo", 1)
    hybrid = parse_int(ARGS, "--hybrid", 0)
    periods = parse_int(ARGS, "--periods", 200)
    sep_order = parse_int(ARGS, "--sep-order", 1)
    sep_horizon = parse_int(ARGS, "--sep-horizon", 200)
    sep_nnodes = parse_int(ARGS, "--sep-nnodes", 3)
    sep_tol = parse_float(ARGS, "--sep-tol", 1e-5)
    sep_sparse_tree = parse_bool(ARGS, "--sep-sparse-tree", true)
    observables = Symbol.(parse_list(ARGS, "--observables", ["Investment"]))
    out_path = parse_arg(ARGS, "--out", "")

    # load Dynare shocks
    path = dynare_rbcii_path(dynare_order; sigma_tag=sigma_tag, algo=algo, hybrid=hybrid)
    ds = load_dynare_dseries(path)
    efficiency = get_series(ds, "efficiency")

    rho = m.parameter_values[findfirst(==(:rho), m.parameters)]
    sigma = m.parameter_values[findfirst(==(:sigma), m.parameters)]
    shocks = implied_shocks_from_efficiency(efficiency; rho=rho, sigma=sigma)
    T = min(periods, length(shocks))
    shocks_used = reshape(shocks[1:T], 1, :)

    # simulate "true" data using SEP
    println("Simulating SEP data (order=$sep_order, periods=$T)...")
    res = MacroModelling.simulate_sep_extended_path(
        m;
        periods = T,
        shocks = shocks_used,
        sep_horizon = sep_horizon,
        sep_order = sep_order,
        sep_nnodes = sep_nnodes,
        sep_tol = sep_tol,
        sep_sparse_tree = sep_sparse_tree,
        shock_scaling = :none,
        silent = true
    )

    if res.errorflag
        println("Warning: SEP failed in period $(res.failure_period). Results may be unreliable.")
    end

    obs_idx = [findfirst(==(v), axiskeys(res.simulation, 1)) for v in observables]
    any(idx -> idx === nothing, obs_idx) && error("Observable(s) not found in simulation output.")
    obs_idx = Int.(obs_idx)
    obs_sim = res.simulation[obs_idx, :]
    if size(obs_sim, 2) == T + 1
        obs_sim = obs_sim[:, 2:end]
    end

    data = KeyedArray(Matrix{Float64}(obs_sim); Variable = observables, Time = 1:size(obs_sim, 2))
    params = m.parameter_values

    println("Running KF loglik...")
    t_kf = @elapsed begin
        ll_kf = get_loglikelihood(m, data, params; algorithm = :first_order, filter = :kalman)
        global loglik_kf = ll_kf
    end

    println("Running inversion loglik...")
    t_inv = @elapsed begin
        ll_inv = get_loglikelihood(m, data, params; algorithm = :first_order, filter = :inversion)
        global loglik_inv = ll_inv
    end

    shocks_true = vec(shocks_used)
    shocks_kf = get_estimated_shocks(m, data; parameters = params, algorithm = :first_order, filter = :kalman, smooth = false)
    shocks_inv = get_estimated_shocks(m, data; parameters = params, algorithm = :first_order, filter = :inversion, smooth = false)
    shocks_kf = vec(Array(shocks_kf))
    shocks_inv = vec(Array(shocks_inv))

    n = min(length(shocks_true), length(shocks_kf), length(shocks_inv))
    rmse_kf = sqrt(mean((shocks_kf[1:n] .- shocks_true[1:n]).^2))
    rmse_inv = sqrt(mean((shocks_inv[1:n] .- shocks_true[1:n]).^2))

    @printf("KF loglik: %.4f (time %.3fs)\n", loglik_kf, t_kf)
    @printf("Inversion loglik: %.4f (time %.3fs)\n", loglik_inv, t_inv)
    @printf("Shock RMSE vs true: KF=%.6f, inversion=%.6f\n", rmse_kf, rmse_inv)
    @printf("Observables: %s\n", join(string.(observables), ", "))

    if out_path != ""
        open(out_path, "w") do io
            println(io, "model=RBCII_Dynare")
            println(io, "dynare_path=$(path)")
            println(io, "periods=$(T)")
            println(io, "observables=$(join(string.(observables), ","))")
            println(io, "loglik_kf=$(loglik_kf)")
            println(io, "time_kf=$(t_kf)")
            println(io, "loglik_inversion=$(loglik_inv)")
            println(io, "time_inversion=$(t_inv)")
            println(io, "shock_rmse_kf=$(rmse_kf)")
            println(io, "shock_rmse_inversion=$(rmse_inv)")
        end
        println("Saved diagnostic summary: $out_path")
    end
end

main()
