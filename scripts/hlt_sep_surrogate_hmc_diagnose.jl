#!/usr/bin/env julia
import Pkg

const SURROGATE_ENV = joinpath(@__DIR__, "..", "SurrogateNN")
const REPO_ENV = normpath(joinpath(@__DIR__, ".."))

function activate_hmc_diag_env!()
    Pkg.activate(SURROGATE_ENV)
    if !("--no-instantiate" in ARGS)
        Pkg.instantiate()
    end
    # Fall back to the main repo environment when the surrogate env is incomplete.
    if Base.find_package("AdvancedHMC") === nothing ||
       Base.find_package("ForwardDiff") === nothing ||
       Base.find_package("LogDensityProblems") === nothing
        @warn "SurrogateNN env missing HMC dependencies; falling back to repo project env." surrogate_env=SURROGATE_ENV repo_env=REPO_ENV
        Pkg.activate(REPO_ENV)
    end
end

activate_hmc_diag_env!()

using Serialization, Random, LinearAlgebra, Statistics, Distributions, Printf
using AdvancedHMC, ForwardDiff, LogDensityProblems
using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

script_repo_root() = normpath(joinpath(@__DIR__, ".."))

function first_two_positional_args(args::Vector{String})
    out = String[]
    for arg in args
        if !startswith(arg, "--")
            push!(out, arg)
            if length(out) == 2
                break
            end
        end
    end
    return out
end

function beta_ab_from_mu_sigma(mu::Float64, sigma::Float64)
    alpha = ((1 - mu) / sigma^2 - 1 / mu) * mu^2
    beta = alpha * (1 / mu - 1)
    return alpha, beta
end

function snap_theta_to_grid(θ::AbstractVector, θ_grid::Vector{Vector{Float64}})
    best_idx = 0
    best_dist = Inf
    for (i, candidate) in enumerate(θ_grid)
        dist = sum(abs2, θ .- candidate)
        if dist < best_dist
            best_dist = dist
            best_idx = i
        end
    end
    return θ_grid[best_idx], best_idx, sqrt(best_dist)
end

function surrogate_loglik(frozen::FrozenMLP,
                           s0::AbstractVector,
                           shocks::AbstractMatrix,
                           θ::AbstractVector,
                           obs_data::AbstractMatrix,
                           obs_sigma::AbstractVector;
                           rom_predictor::Union{Nothing,RomPredictor} = nothing)
    d_obs = size(obs_data, 1)
    predict_surrogate = (state, shock_t, θ_local) -> begin
        x = vcat(state, shock_t, θ_local)
        y_resid = predict_frozen(frozen, x)
        if rom_predictor === nothing
            length(y_resid) < d_obs + 1 &&
                error("Obs-only surrogate requires state output to update state.")
            obs_pred = y_resid[1:d_obs]
            state_next = y_resid[d_obs + 1:end]
        else
            rom_full = rom_predict(rom_predictor, state, shock_t, θ_local)
            if length(y_resid) == d_obs
                obs_pred = rom_full[1:d_obs] .+ y_resid
                state_next = rom_full[d_obs + 1:end]
            elseif length(y_resid) == length(rom_full)
                y = rom_full .+ y_resid
                obs_pred = y[1:d_obs]
                state_next = y[d_obs + 1:end]
            else
                error("Residual output size mismatch: got $(length(y_resid)), expected $d_obs or $(length(rom_full)).")
            end
        end
        return obs_pred, state_next
    end
    ll = MacroModelling.conditional_loglik_per_period(
        predict_surrogate,
        s0,
        shocks,
        θ,
        obs_data,
        obs_sigma,
    )
    return sum(ll)
end

positional = first_two_positional_args(ARGS)
surrogate_path = length(positional) >= 1 ? positional[1] : nothing
synthetic_path = length(positional) >= 2 ? positional[2] : nothing

if surrogate_path === nothing || synthetic_path === nothing
    error("Usage: julia hlt_sep_surrogate_hmc_diagnose.jl <trained_surrogate.jls> <synthetic_data.jls>")
end

n_samples = parse_arg_int(ARGS, "--samples", 200)
n_adapt = parse_arg_int(ARGS, "--adapt", 100)
target_accept = parse_arg_float(ARGS, "--target-accept", 0.85)
seed = parse_arg_int(ARGS, "--seed", 1)
sample_shocks = "--sample-shocks" in ARGS
snap_theta = "--snap-theta" in ARGS
obs_sigma_mode = parse_arg_symbol(ARGS, "--obs-sigma-mode", :synthetic)
obs_sigma_scale = parse_arg_float(ARGS, "--obs-sigma-scale", 1.0)
obs_sigma_floor = parse_arg_float(ARGS, "--obs-sigma-floor", 0.0)
shock_prior_scale = parse_arg_float(ARGS, "--shock-prior-scale", 1.0)
T_cap = parse_arg_int(ARGS, "--T", 0)

Random.seed!(seed)

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
surrogate_data = surrogate_bundle.payload
frozen = surrogate_bundle.frozen
meta = surrogate_bundle.meta
rom_residual = get(meta, "rom_residual", false)
rom_order = Int(get(meta, "rom_residual_order", 0))
rom_mode_raw = get(meta, "rom_mode", :baseline)
rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)
val_rmse = get(surrogate_data, "validation_rmse", nothing)

synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
synthetic_model_name = get(synthetic, "model", "Smets_Wouters_2007_HLT")
obs_data = synthetic["obs_data"]
obs_sigma = synthetic["obs_sigma"]
s0 = synthetic["s0"]
shocks_fixed = synthetic["shocks"]
theta_true = synthetic["theta_true"]
theta_names = get(synthetic, "theta_names", Symbol[])
observables = get(synthetic, "observables", Symbol[])
state_names = get(synthetic, "state_names", Symbol[])
shock_sigmas = haskey(synthetic, "shock_sigmas") ? synthetic["shock_sigmas"] : vec(Statistics.std(shocks_fixed, dims = 2))
shock_sigmas = shock_sigmas .* shock_prior_scale

if T_cap > 0 && T_cap < size(obs_data, 2)
    obs_data = obs_data[:, 1:T_cap]
    shocks_fixed = shocks_fixed[:, 1:T_cap]
end

d_obs = size(obs_data, 1)
d_theta = length(theta_true)
d_eps = size(shocks_fixed, 1)
T_obs = size(obs_data, 2)
structural_idx = findall(shock_sigmas .> 0)

rom_predictor = nothing
if rom_residual && rom_order != 0
    use_obc = false
    if synthetic_model_name == "Smets_Wouters_2007_HLT_obc"
        use_obc = true
        mm_model = load_hlt_model(script_repo_root(), "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
    else
        mm_model = load_hlt_model(script_repo_root(), "Smets_Wouters_2007_HLT"; mod = @__MODULE__)
    end
    if isempty(observables) || isempty(state_names)
        error("ROM residual surrogate requires observables and state_names in synthetic data.")
    end
    obs_idx = indexin(observables, mm_model.var)
    state_idx = indexin(state_names, mm_model.var)
    if any(isnothing, obs_idx)
        error("Observable names not found in $(mm_model.model_name).")
    end
    if any(isnothing, state_idx)
        error("State names not found in $(mm_model.model_name).")
    end
    base_params = haskey(meta, "base_values") ? meta["base_values"] : mm_model.parameter_values
    theta_idx = Int[]
    if rom_mode == :theta
        error("rom_mode=:theta is not supported with AD-based HMC diagnostics. Retrain with --rom-mode=baseline.")
    end
    rom_predictor = RomPredictor(mm_model,
                                 rom_order,
                                 rom_mode,
                                 use_obc,
                                 Int.(theta_idx),
                                 base_params,
                                 nothing,
                                 nothing,
                                 Int.(state_idx),
                                 Int.(obs_idx))
    if rom_mode == :baseline
        ensure_rom_cache!(rom_predictor, theta_true)
    end
end

θ_grid = Vector{Vector{Float64}}()
if haskey(meta, "theta_grid")
    θ_grid = [θ for θ in meta["theta_grid"] if all(isfinite, θ)]
end

if obs_sigma_mode != :synthetic && val_rmse !== nothing
    if length(val_rmse) < d_obs
        error("Surrogate validation_rmse length ($(length(val_rmse))) < d_obs ($d_obs).")
    end
    obs_rmse = val_rmse[1:d_obs] .* obs_sigma_scale
    if obs_sigma_mode == :surrogate
        obs_sigma = obs_rmse
    elseif obs_sigma_mode == :max
        obs_sigma = max.(obs_sigma, obs_rmse)
    else
        error("Unknown obs_sigma_mode=$obs_sigma_mode. Use :synthetic, :surrogate, or :max.")
    end
elseif obs_sigma_mode != :synthetic && val_rmse === nothing
    println("Warning: surrogate validation_rmse not found; using synthetic obs_sigma.")
end

if obs_sigma_floor > 0
    obs_sigma = max.(obs_sigma, obs_sigma_floor)
end

αp, βp = beta_ab_from_mu_sigma(0.5, 0.10)
αi, βi = beta_ab_from_mu_sigma(0.5, 0.15)
prior_cprobp = Distributions.truncated(Distributions.Beta(αp, βp), 0.5, 0.95)
prior_cindp = Distributions.truncated(Distributions.Beta(αi, βi), 0.01, 0.99)
prior_curvp = Distributions.Normal(75.0, 25.0)
θ_priors = (prior_cprobp, prior_cindp, prior_curvp)

function logprior_theta(θ::AbstractVector)
    return logpdf(θ_priors[1], θ[1]) + logpdf(θ_priors[2], θ[2]) + logpdf(θ_priors[3], θ[3])
end

function build_shocks_from_eps(eps_std::AbstractVector,
                               shock_sigmas::AbstractVector,
                               structural_idx::AbstractVector{Int},
                               d_eps::Int,
                               T::Int)
    shocks = zeros(eltype(eps_std), d_eps, T)
    if isempty(structural_idx)
        return shocks
    end
    eps_mat = reshape(eps_std, length(structural_idx), T)
    for (j, idx) in enumerate(structural_idx)
        shocks[idx, :] .= eps_mat[j, :] .* shock_sigmas[idx]
    end
    return shocks
end

struct HLTLogDensity
    frozen::FrozenMLP
    obs_data::Matrix{Float64}
    obs_sigma::Vector{Float64}
    s0::Vector{Float64}
    shocks_fixed::Matrix{Float64}
    shock_sigmas::Vector{Float64}
    structural_idx::Vector{Int}
    θ_grid::Vector{Vector{Float64}}
    snap_theta::Bool
    sample_shocks::Bool
    d_theta::Int
    d_eps::Int
    T::Int
    dim::Int
    rom_predictor::Union{Nothing,RomPredictor}
end

function log_posterior(p::HLTLogDensity, x::AbstractVector)
    θ = x[1:p.d_theta]
    θ_eff = θ
    if p.snap_theta && !isempty(p.θ_grid)
        θ_eff, _, _ = snap_theta_to_grid(θ, p.θ_grid)
    end

    lp = logprior_theta(θ)
    if p.sample_shocks
        eps_std = x[p.d_theta + 1:end]
        lp += sum(logpdf.(Normal(0, 1), eps_std))
        shocks = build_shocks_from_eps(eps_std, p.shock_sigmas, p.structural_idx, p.d_eps, p.T)
    else
        shocks = p.shocks_fixed
    end

    ll = surrogate_loglik(p.frozen, p.s0, shocks, θ_eff, p.obs_data, p.obs_sigma;
                          rom_predictor = p.rom_predictor)
    return lp + ll
end

LogDensityProblems.logdensity(p::HLTLogDensity, x::AbstractVector) = log_posterior(p, x)
LogDensityProblems.dimension(p::HLTLogDensity) = p.dim
LogDensityProblems.capabilities(::Type{HLTLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::HLTLogDensity, x::AbstractVector)
    lp = log_posterior(p, x)
    grad = ForwardDiff.gradient(z -> log_posterior(p, z), x)
    return lp, grad
end

d_shock = sample_shocks ? length(structural_idx) * T_obs : 0
d_total = d_theta + d_shock

println("HMC diagnostic")
println("  Surrogate: $surrogate_path")
println("  Synthetic: $synthetic_path")
println("  T_obs: $T_obs")
println("  d_theta: $d_theta, d_eps: $d_eps, sample_shocks=$sample_shocks (dim=$d_total)")
println("  obs_sigma mode: $obs_sigma_mode (min/max=$(minimum(obs_sigma)) / $(maximum(obs_sigma)))")
println("  snap_theta: $snap_theta (theta_grid size=$(length(θ_grid)))")
if rom_predictor !== nothing
    println("  ROM residual: order=$rom_order mode=$rom_mode")
end

θ_init = copy(theta_true)
eps_init = sample_shocks ? zeros(d_shock) : zeros(0)
x_init = vcat(θ_init, eps_init)

log_density = HLTLogDensity(
    frozen,
    obs_data,
    obs_sigma,
    s0,
    shocks_fixed,
    shock_sigmas,
    structural_idx,
    θ_grid,
    snap_theta,
    sample_shocks,
    d_theta,
    d_eps,
    T_obs,
    d_total,
    rom_predictor,
)

lp_init = log_posterior(log_density, x_init)
println("  Log posterior at init: $lp_init")
grad_init = ForwardDiff.gradient(z -> log_posterior(log_density, z), x_init)
println("  Grad norm: $(norm(grad_init)) (all finite=$(all(isfinite, grad_init)))")

if snap_theta && !isempty(θ_grid)
    θ_snap, idx, dist = snap_theta_to_grid(θ_init, θ_grid)
    println("  Snap theta: idx=$idx dist=$(round(dist, digits=6)) θ_snap=$(θ_snap)")
end

# Mass matrix (diagonal)
θ_scales = [0.10, 0.15, 25.0]
M_diag = vcat(θ_scales .^ 2, ones(d_shock))
metric = DiagEuclideanMetric(M_diag)

hamiltonian = Hamiltonian(metric, log_density)
initial_ϵ = find_good_stepsize(hamiltonian, x_init)
integrator = Leapfrog(initial_ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_accept, integrator))

println("  Running NUTS: adapt=$n_adapt samples=$n_samples target_accept=$target_accept")
samples, stats = sample(hamiltonian, kernel, x_init, n_samples + n_adapt, adaptor, n_adapt;
                        progress=true, verbose=false)

θ_samples = hcat([s[1:d_theta] for s in samples[(n_adapt+1):end]]...)'

acceptance_rate = mean([s.acceptance_rate for s in stats[(n_adapt+1):end]])
step_size_final = mean([s.step_size for s in stats[(n_adapt+1):end]])

println("  Acceptance rate: $(round(acceptance_rate, digits=3))")
println("  Final step size: $(round(step_size_final, sigdigits=3))")
println("  Theta mean: $(round.(vec(mean(θ_samples, dims=1)), digits=6))")
println("  Theta std: $(round.(vec(Statistics.std(θ_samples, dims=1)), digits=6))")
