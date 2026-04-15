# RBCII surrogate diagnostic: KF vs inversion vs surrogate inversion
#
# Steps:
# 1) Infer shocks from Dynare efficiency series.
# 2) Simulate SEP data (treated as truth).
# 3) Train ROM1 residual surrogate on (state_{t-1}, shock_t) -> (obs_t, state_t) residuals.
# 4) Compare loglik/runtime:
#    - Linear KF (MacroModelling)
#    - Linear inversion (MacroModelling)
#    - Surrogate conditional (true shocks)
#    - Surrogate inversion (MAP shocks)

using MacroModelling
using MAT
using AxisKeys
using Statistics
using Printf
using Random
using LinearAlgebra
import ForwardDiff

include("../models/RBCII_Dynare.jl")
include("hlt_surrogate/hlt_sep_surrogate_nn_utils.jl")
include("hlt_surrogate/hlt_sep_surrogate_rom_utils.jl")

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

function predict_surrogate(frozen::FrozenMLP,
                           rom_predictor::RomPredictor,
                           state::AbstractVector,
                           shock::AbstractVector,
                           d_obs::Int)
    rom_full = rom_predict(rom_predictor, state, shock, Float64[])
    y_resid = predict_frozen(frozen, vcat(state, shock))
    if length(y_resid) != length(rom_full)
        error("Residual output size mismatch: got $(length(y_resid)), expected $(length(rom_full)).")
    end
    y = rom_full .+ y_resid
    return y[1:d_obs], y[d_obs + 1:end]
end

function inversion_step(predict_fn,
                        state::AbstractVector,
                        y_obs::AbstractVector,
                        obs_sigma::AbstractVector,
                        shock_sigmas::AbstractVector,
                        structural_idx::AbstractVector{Int};
                        eps_init::Union{Nothing,AbstractVector} = nothing,
                        maxit::Int = 10,
                        tol::Float64 = 1e-6,
                        lambda::Float64 = 1e-4)
    d_eps = length(shock_sigmas)
    n_struct = length(structural_idx)
    shock_std = n_struct > 0 ? shock_sigmas[structural_idx] : zeros(eltype(obs_sigma), 0)
    eps_struct = eps_init === nothing ? zeros(eltype(obs_sigma), n_struct) : copy(eps_init)

    if n_struct == 0
        eps_full = zeros(eltype(obs_sigma), d_eps)
        obs_pred, state_next = predict_fn(state, eps_full)
        resid = (y_obs .- obs_pred) ./ obs_sigma
        ll = -0.5 * (sum(resid .^ 2) + sum(log.(2 * pi .* obs_sigma .^ 2)))
        return eps_full, state_next, ll
    end

    for _ in 1:maxit
        eps_full = zeros(eltype(eps_struct), d_eps)
        eps_full[structural_idx] .= eps_struct
        obs_pred, _ = predict_fn(state, eps_full)
        resid = (y_obs .- obs_pred) ./ obs_sigma
        r = vcat(resid, eps_struct ./ shock_std)

        J = ForwardDiff.jacobian(eps_s -> begin
            eps_full = zeros(eltype(eps_s), d_eps)
            eps_full[structural_idx] .= eps_s
            predict_fn(state, eps_full)[1]
        end, eps_struct)
        J_obs = -(J ./ obs_sigma)
        J_prior = Diagonal(1.0 ./ shock_std)
        J_aug = vcat(J_obs, J_prior)

        lhs = J_aug' * J_aug + lambda * I
        rhs = -J_aug' * r
        step = lhs \ rhs
        eps_struct .+= step
        if norm(step) <= tol * (1 + norm(eps_struct))
            break
        end
    end

    eps_full = zeros(eltype(eps_struct), d_eps)
    eps_full[structural_idx] .= eps_struct
    obs_pred, state_next = predict_fn(state, eps_full)
    resid = (y_obs .- obs_pred) ./ obs_sigma
    ll = -0.5 * (sum(resid .^ 2) +
                 sum((eps_struct ./ shock_std) .^ 2) +
                 sum(log.(2 * pi .* obs_sigma .^ 2)) +
                 sum(log.(2 * pi .* shock_std .^ 2)))
    return eps_full, state_next, ll
end

function inversion_loglik_per_period(predict_fn,
                                     s0::AbstractVector,
                                     obs_data::AbstractMatrix,
                                     obs_sigma::AbstractVector,
                                     shock_sigmas::AbstractVector;
                                     maxit::Int = 10,
                                     tol::Float64 = 1e-6,
                                     lambda::Float64 = 1e-4)
    state = copy(s0)
    d_eps = length(shock_sigmas)
    T = size(obs_data, 2)
    ll = zeros(Float64, T)
    shocks_out = zeros(Float64, d_eps, T)
    structural_idx = findall(shock_sigmas .> 0)
    eps_init = zeros(Float64, length(structural_idx))

    for t in 1:T
        eps_full, state_next, ll_t = inversion_step(predict_fn,
                                                    state,
                                                    obs_data[:, t],
                                                    obs_sigma,
                                                    shock_sigmas,
                                                    structural_idx;
                                                    eps_init = eps_init,
                                                    maxit = maxit,
                                                    tol = tol,
                                                    lambda = lambda)
        shocks_out[:, t] .= eps_full
        ll[t] = ll_t
        state = state_next
        eps_init = eps_full[structural_idx]
    end
    return ll, shocks_out
end

function main()
    m = RBCII_Dynare

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
    shock_scale = parse_float(ARGS, "--shock-scale", 1.0)
    train_epochs = parse_int(ARGS, "--train-epochs", 300)
    train_hidden = parse_int(ARGS, "--train-hidden", 128)
    train_hidden2 = parse_int(ARGS, "--train-hidden2", 64)
    train_seed = parse_int(ARGS, "--train-seed", 1)
    train_share = parse_float(ARGS, "--train-share", 0.9)
    obs_sigma_scale = parse_float(ARGS, "--obs-sigma-scale", 1.0)
    obs_sigma_floor = parse_float(ARGS, "--obs-sigma-floor", 1e-6)
    use_obc = parse_bool(ARGS, "--use-obc", false)
    observables = Symbol.(parse_list(ARGS, "--observables", ["Investment"]))
    out_path = parse_arg(ARGS, "--out", "")
    surrogate_out = parse_arg(ARGS, "--surrogate-out", "")

    path = dynare_rbcii_path(dynare_order; sigma_tag=sigma_tag, algo=algo, hybrid=hybrid)
    ds = load_dynare_dseries(path)
    efficiency = get_series(ds, "efficiency")
    rho = m.parameter_values[findfirst(==(:rho), m.parameters)]
    sigma = m.parameter_values[findfirst(==(:sigma), m.parameters)]
    shocks_eps = implied_shocks_from_efficiency(efficiency; rho=rho, sigma=sigma)
    T = min(periods, length(shocks_eps))

    n_exo = length(m.timings.exo)
    shock_idx = findfirst(==(:epsilon), m.timings.exo)
    shock_idx === nothing && error("epsilon shock not found in RBCII_Dynare.")
    shocks_full = zeros(Float64, n_exo, T)
    shocks_full[shock_idx, :] .= shock_scale .* shocks_eps[1:T]

    println("Simulating SEP data (order=$sep_order, periods=$T)...")
    res = MacroModelling.simulate_sep_extended_path(
        m;
        periods = T,
        shocks = shocks_full,
        sep_horizon = sep_horizon,
        sep_order = sep_order,
        sep_nnodes = sep_nnodes,
        sep_tol = sep_tol,
        sep_sparse_tree = sep_sparse_tree,
        shock_scaling = :none,
        silent = true
    )
    res.errorflag && println("Warning: SEP failed in period $(res.failure_period).")

    obs_idx = indexin(observables, m.var)
    any(isnothing, obs_idx) && error("Observable not found in RBCII_Dynare.")
    obs_idx = Int.(obs_idx)
    state_idx = m.timings.past_not_future_and_mixed_idx
    state_names = m.var[state_idx]

    inv_idx = findfirst(==(:Investment), m.var)
    zlb_idx = findfirst(==(:ZLB), m.parameters)
    if inv_idx !== nothing && zlb_idx !== nothing
        investment = Vector{Float64}(res.simulation[inv_idx, 2:(T + 1)])
        investment_ss = get_steady_state(m, derivatives=false)(:Investment)
        investment_floor = m.parameter_values[zlb_idx] * investment_ss
        bind_share = mean(investment .<= investment_floor + 1e-10)
        println("Investment floor bind share: $(round(bind_share, digits=4)) (floor=$(investment_floor))")
    end

    sim = res.simulation
    obs_data = Matrix{Float64}(sim[obs_idx, 2:(T + 1)])
    state_prev = Matrix{Float64}(sim[state_idx, 1:T])
    state_next = Matrix{Float64}(sim[state_idx, 2:(T + 1)])

    d_obs = size(obs_data, 1)
    d_state = size(state_prev, 1)
    d_in = d_state + n_exo
    d_out = d_obs + d_state

    base_params = copy(m.parameter_values)
    rom_predictor = RomPredictor(m,
                                 1,
                                 :baseline,
                                 use_obc,
                                 Int[],
                                 base_params,
                                 nothing,
                                 nothing,
                                 Int.(state_idx),
                                 Int.(obs_idx))
    ensure_rom_cache!(rom_predictor, Float64[])

    X = zeros(Float64, d_in, T)
    Y = zeros(Float64, d_out, T)
    for t in 1:T
        shock_vec = shocks_full[:, t]
        rom_pred = rom_predict(rom_predictor, state_prev[:, t], shock_vec, Float64[])
        true_next = vcat(obs_data[:, t], state_next[:, t])
        X[:, t] .= vcat(state_prev[:, t], shock_vec)
        Y[:, t] .= true_next .- rom_pred
    end

    rng = Random.MersenneTwister(train_seed)
    perm = randperm(rng, T)
    n_train = max(1, min(T - 1, floor(Int, train_share * T)))
    train_idx = perm[1:n_train]
    val_idx = perm[(n_train + 1):end]
    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]
    X_val = X[:, val_idx]
    Y_val = Y[:, val_idx]

    println("Training surrogate (n_train=$n_train, n_val=$(length(val_idx)))...")
    frozen = train_mlp!(X_train, Y_train;
                        d_hidden = train_hidden,
                        d_hidden2 = train_hidden2,
                        nepoch = train_epochs,
                        seed = train_seed,
                        verbose = true)

    val_rmse = NaN
    if !isempty(val_idx)
        Y_pred = similar(Y_val)
        for i in 1:size(X_val, 2)
            Y_pred[:, i] .= predict_frozen(frozen, X_val[:, i])
        end
        val_rmse = sqrt(mean((Y_pred .- Y_val).^2))
        println("Validation RMSE (residuals): $(round(val_rmse, digits=6))")
    end

    obs_sigma = vec(Statistics.std(obs_data, dims = 2)) .* obs_sigma_scale
    obs_sigma[obs_sigma .== 0.0] .= 1.0
    obs_sigma = max.(obs_sigma, obs_sigma_floor)

    data = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))
    params = m.parameter_values

    t_kf = @elapsed loglik_kf = get_loglikelihood(m, data, params; algorithm = :first_order, filter = :kalman)
    t_inv = @elapsed loglik_inv = get_loglikelihood(m, data, params; algorithm = :first_order, filter = :inversion)

    # surrogate conditional (true shocks)
    t_cond = @elapsed begin
        ll_cond = 0.0
        state = copy(state_prev[:, 1])
        for t in 1:T
            shock_vec = shocks_full[:, t]
            obs_pred, state_next_pred = predict_surrogate(frozen, rom_predictor, state, shock_vec, d_obs)
            resid = (obs_data[:, t] .- obs_pred) ./ obs_sigma
            ll_cond += -0.5 * sum(resid .^ 2 .+ log.(2 * pi .* obs_sigma .^ 2))
            state = state_next_pred
        end
        global loglik_surrogate_cond = ll_cond
    end

    shock_sigmas = zeros(Float64, n_exo)
    shock_sigmas[shock_idx] = sigma
    predict_lin = (state, shock_vec) -> begin
        y = rom_predict(rom_predictor, state, shock_vec, Float64[])
        return y[1:d_obs], y[d_obs + 1:end]
    end
    predict_sep = (state, shock_vec) -> predict_surrogate(frozen, rom_predictor, state, shock_vec, d_obs)

    t_inv_sur = @elapsed begin
        ll_sep_vec, shocks_sep_inv = inversion_loglik_per_period(predict_sep,
                                                                 state_prev[:, 1],
                                                                 obs_data,
                                                                 obs_sigma,
                                                                 shock_sigmas)
        global loglik_surrogate_inv = sum(ll_sep_vec)
        global shocks_sep_inv_full = shocks_sep_inv
    end

    t_inv_lin = @elapsed begin
        ll_lin_vec, shocks_lin_inv = inversion_loglik_per_period(predict_lin,
                                                                 state_prev[:, 1],
                                                                 obs_data,
                                                                 obs_sigma,
                                                                 shock_sigmas)
        global loglik_linear_inv = sum(ll_lin_vec)
        global shocks_lin_inv_full = shocks_lin_inv
    end

    true_shocks = shocks_full[shock_idx, :]
    rmse_inv_lin = sqrt(mean((shocks_lin_inv_full[shock_idx, :] .- true_shocks).^2))
    rmse_inv_sur = sqrt(mean((shocks_sep_inv_full[shock_idx, :] .- true_shocks).^2))

    @printf("KF loglik: %.4f (time %.3fs)\n", loglik_kf, t_kf)
    @printf("Inversion loglik (linear): %.4f (time %.3fs)\n", loglik_inv, t_inv)
    @printf("Surrogate conditional loglik (true shocks): %.4f (time %.3fs)\n", loglik_surrogate_cond, t_cond)
    @printf("Surrogate inversion loglik: %.4f (time %.3fs)\n", loglik_surrogate_inv, t_inv_sur)
    @printf("Inversion loglik (ROM1, MAP shocks): %.4f (time %.3fs)\n", loglik_linear_inv, t_inv_lin)
    @printf("Shock RMSE vs true (inversion): ROM1=%.6f, surrogate=%.6f\n", rmse_inv_lin, rmse_inv_sur)
    @printf("Observables: %s\n", join(string.(observables), ", "))
    if shock_scale != 1.0
        println("Shock scale applied: $shock_scale")
    end

    if surrogate_out != ""
        serialize(surrogate_out, Dict(
            "frozen" => frozen,
            "meta" => Dict(
                "model" => "RBCII_Dynare",
                "rom_residual" => true,
                "rom_residual_order" => 1,
                "rom_mode" => :baseline,
                "observables" => observables,
                "state_names" => state_names,
                "shock_idx" => shock_idx,
                "train_epochs" => train_epochs,
                "train_hidden" => train_hidden,
                "train_hidden2" => train_hidden2,
                "train_seed" => train_seed,
                "validation_rmse" => val_rmse,
            )
        ))
        println("Saved surrogate: $surrogate_out")
    end

    if out_path != ""
        open(out_path, "w") do io
            println(io, "model=RBCII_Dynare")
            println(io, "dynare_path=$(path)")
            println(io, "periods=$(T)")
            println(io, "observables=$(join(string.(observables), ","))")
            println(io, "shock_scale=$(shock_scale)")
            println(io, "loglik_kf=$(loglik_kf)")
            println(io, "time_kf=$(t_kf)")
            println(io, "loglik_inversion=$(loglik_inv)")
            println(io, "time_inversion=$(t_inv)")
            println(io, "loglik_surrogate_conditional=$(loglik_surrogate_cond)")
            println(io, "time_surrogate_conditional=$(t_cond)")
            println(io, "loglik_surrogate_inversion=$(loglik_surrogate_inv)")
            println(io, "time_surrogate_inversion=$(t_inv_sur)")
            println(io, "loglik_linear_inversion_map=$(loglik_linear_inv)")
            println(io, "time_linear_inversion_map=$(t_inv_lin)")
            println(io, "shock_rmse_inv_rom1=$(rmse_inv_lin)")
            println(io, "shock_rmse_inv_surrogate=$(rmse_inv_sur)")
        end
        println("Saved diagnostic summary: $out_path")
    end
end

main()
