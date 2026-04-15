#!/usr/bin/env julia
using Serialization
using Statistics
using Profile
using LinearAlgebra
import Distributions
using MCMCChains
using Random

import Turing
import DynamicPPL
import ADTypes: AutoForwardDiff
using MacroModelling
using AxisKeys

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function parse_system_priors(args::Vector{String})
    priors = Tuple{Int,Int,Float64}[]
    for arg in args
        if startswith(arg, "--system-prior=")
            spec = split(arg, "=", limit = 2)[2]
            parts = split(spec, "=", limit = 2)
            if length(parts) != 2
                error("Invalid --system-prior format: $spec (expected start:stop=prob).")
            end
            range_part, prob_str = parts
            range_part = replace(range_part, "-" => ":")
            if occursin(":", range_part)
                bounds = split(range_part, ":", limit = 2)
                if length(bounds) != 2
                    error("Invalid --system-prior range: $range_part.")
                end
                start = parse(Int, bounds[1])
                stop = parse(Int, bounds[2])
            else
                start = parse(Int, range_part)
                stop = start
            end
            prob = parse(Float64, prob_str)
            push!(priors, (start, stop, prob))
        end
    end
    return priors
end

function parse_hard_gate_ranges(args::Vector{String})
    ranges = Tuple{Int,Int}[]
    for arg in args
        if startswith(arg, "--hard-gate=")
            spec = split(arg, "=", limit = 2)[2]
            for token in split(spec, ",")
                token = strip(token)
                token = replace(token, "-" => ":")
                if occursin(":", token)
                    bounds = split(token, ":", limit = 2)
                    if length(bounds) != 2
                        error("Invalid --hard-gate range: $token.")
                    end
                    start = parse(Int, bounds[1])
                    stop = parse(Int, bounds[2])
                else
                    start = parse(Int, token)
                    stop = start
                end
                push!(ranges, (start, stop))
            end
        end
    end
    return ranges
end

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

positional = first_two_positional_args(ARGS)
surrogate_path = length(positional) >= 1 ? positional[1] : nothing
synthetic_path = length(positional) >= 2 ? positional[2] : nothing

if surrogate_path === nothing || synthetic_path === nothing
    error("Usage: julia hlt_sep_surrogate_synthetic_estimation.jl <trained_surrogate.jls> <synthetic_data.jls> [--samples=500]")
end

n_samples = parse_arg_int(ARGS, "--samples", 500)
linear_samples = parse_arg_int(ARGS, "--linear-samples", 0)
n_chains = parse_arg_int(ARGS, "--chains", 1)
mcmc_threads = parse_arg_bool(ARGS, "--mcmc-threads", true)
sampler_mode = parse_arg_symbol(ARGS, "--sampler", :nuts)
nuts_adapt = parse_arg_int(ARGS, "--nuts-adapt", 1000)
nuts_target_accept = parse_arg_float(ARGS, "--nuts-target-accept", 0.8)
nuts_max_depth = parse_arg_int(ARGS, "--nuts-max-depth", 10)
nuts_init_eps = parse_arg_float(ARGS, "--nuts-init-eps", 0.0)
hmc_step_size = parse_arg_float(ARGS, "--hmc-step-size", 1e-4)
hmc_leapfrog = parse_arg_int(ARGS, "--hmc-leapfrog", 8)
mh_rw_cprobp = parse_arg_float(ARGS, "--mh-rw-cprobp", 5e-3)
mh_rw_cindp = parse_arg_float(ARGS, "--mh-rw-cindp", 5e-3)
mh_rw_curvp = parse_arg_float(ARGS, "--mh-rw-curvp", 1.0)
mh_rw_default = parse_arg_float(ARGS, "--mh-rw-default", 5e-3)
param_set_name = parse_arg_symbol(ARGS, "--param-set", :auto)
out_path = parse_arg_string(ARGS, "--out", "")
linear_out_path = parse_arg_string(ARGS, "--linear-out", "")
chain_in = parse_arg_string(ARGS, "--chain-in", "")
eval_only = parse_arg_bool(ARGS, "--eval-only", false)
post_mean_filter = parse_arg_symbol(ARGS, "--post-mean-filter", :none)
chunk_size = parse_arg_int(ARGS, "--chunk-size", 0)
checkpoint_path = parse_arg_string(ARGS, "--checkpoint-path", "")
shock_prior_scale = parse_arg_float(ARGS, "--shock-prior-scale", 1.0)
shock_filter = parse_arg_symbol(ARGS, "--shock-filter", :sampling)
linear_filter = parse_arg_symbol(ARGS, "--linear-filter", :kalman)
inversion_maxit = parse_arg_int(ARGS, "--inversion-maxit", 10)
inversion_tol = parse_arg_float(ARGS, "--inversion-tol", 1e-6)
inversion_lambda = parse_arg_float(ARGS, "--inversion-lambda", 1e-4)
refine_maxit = parse_arg_int(ARGS, "--refine-maxit", 0)
refine_tol = parse_arg_float(ARGS, "--refine-tol", 1e-4)
sample_shocks = "--sample-shocks" in ARGS
linear_sample_shocks = parse_arg_bool(ARGS, "--linear-sample-shocks", false)
profile = "--profile" in ARGS
profile_out = parse_arg_string(ARGS, "--profile-out", "")
obs_sigma_mode = parse_arg_symbol(ARGS, "--obs-sigma-mode", :max)
obs_sigma_scale = parse_arg_float(ARGS, "--obs-sigma-scale", 1.0)
obs_sigma_floor = parse_arg_float(ARGS, "--obs-sigma-floor", 0.0)
shock_guidance = parse_arg_symbol(ARGS, "--shock-guidance", :none)
shock_guidance_scale = parse_arg_float(ARGS, "--shock-guidance-scale", 1.0)
shock_guidance_theta = parse_arg_symbol(ARGS, "--shock-guidance-theta", :baseline)
obs_window = parse_arg_int(ARGS, "--obs-window", 0)
obs_window_mode = parse_arg_symbol(ARGS, "--obs-window-mode", :last)
obs_window_theta = parse_arg_symbol(ARGS, "--obs-window-theta", :synthetic)
obs_window_shocks = parse_arg_symbol(ARGS, "--obs-window-shocks", :auto)
obs_window_init = parse_arg_symbol(ARGS, "--obs-window-init", :surrogate)
obs_window_filter = parse_arg_symbol(ARGS, "--obs-window-filter", :inversion)
shock_sample_window = parse_arg_symbol(ARGS, "--shock-sample-window", :all)
force_obc = "--use-obc" in ARGS
force_no_obc = "--no-obc" in ARGS
if force_obc && force_no_obc
    error("Specify only one of --use-obc or --no-obc.")
end
gate_calibration_path = parse_arg_string(ARGS, "--gate-calibration", "")
gate_k_pre = parse_arg_int(ARGS, "--gate-k-pre", 4)
gate_k_post = parse_arg_int(ARGS, "--gate-k-post", 8)
gate_min_len = parse_arg_int(ARGS, "--gate-min-len", 4)
gate_shock_filter = parse_arg_symbol(ARGS, "--gate-shock-filter", :kalman)
gate_theta_mode = parse_arg_symbol(ARGS, "--gate-theta", :baseline)
gate_periods_arg = parse_arg_int(ARGS, "--gate-periods", -1)
gate_mode = parse_arg_symbol(ARGS, "--gate-mode", :soft)
gate_beta_eps = parse_arg_float(ARGS, "--gate-beta-eps", 1.0)
gate_beta_y = parse_arg_float(ARGS, "--gate-beta-y", 1.0)
gate_bias_arg = parse_arg_float(ARGS, "--gate-bias", NaN)
gate_prob_floor = parse_arg_float(ARGS, "--gate-prob-floor", 1e-4)
gate_prob_ceiling = parse_arg_float(ARGS, "--gate-prob-ceil", 1 - 1e-4)
gate_use_eps = parse_arg_bool(ARGS, "--gate-use-eps", true)
gate_use_y = parse_arg_bool(ARGS, "--gate-use-y", true)
gate_use_cached_stats = parse_arg_bool(ARGS, "--gate-use-cached-stats", true)
gate_share_min = parse_arg_float(ARGS, "--gate-share-min", 0.0)
gate_share_max = parse_arg_float(ARGS, "--gate-share-max", 1.0)
fail_degenerate_gate = parse_arg_bool(ARGS, "--fail-degenerate-gate", false)
use_chain_gate = "--use-chain-gate" in ARGS
true_theta_filter = parse_arg_symbol(ARGS, "--true-theta-filter", :none)
hard_gate_threshold = parse_arg_float(ARGS, "--hard-gate-threshold", 0.5)
system_prior_default = parse_arg_float(ARGS, "--system-prior-default", NaN)
system_priors = parse_system_priors(ARGS)
hard_gate_ranges = parse_hard_gate_ranges(ARGS)
prior_cprobp_mu = parse_arg_float(ARGS, "--prior-cprobp-mean", 0.5)
prior_cprobp_sd = parse_arg_float(ARGS, "--prior-cprobp-sd", 0.10)
prior_cindp_mu = parse_arg_float(ARGS, "--prior-cindp-mean", 0.5)
prior_cindp_sd = parse_arg_float(ARGS, "--prior-cindp-sd", 0.15)
prior_curvp_mu = parse_arg_float(ARGS, "--prior-curvp-mean", 75.0)
prior_curvp_sd = parse_arg_float(ARGS, "--prior-curvp-sd", 25.0)
init_cprobp_arg = parse_arg_float(ARGS, "--init-cprobp", NaN)
init_cindp_arg = parse_arg_float(ARGS, "--init-cindp", NaN)
init_curvp_arg = parse_arg_float(ARGS, "--init-curvp", NaN)
init_jitter = parse_arg_float(ARGS, "--init-jitter", 0.0)
if gate_prob_floor <= 0 || gate_prob_ceiling >= 1 || gate_prob_floor >= gate_prob_ceiling
    error("Invalid gate probability bounds: floor=$gate_prob_floor, ceil=$gate_prob_ceiling.")
end
if !(sampler_mode in (:nuts, :hmc, :mh))
    error("Unknown --sampler=$sampler_mode. Use :nuts, :hmc, or :mh.")
end
if nuts_adapt < 0
    error("Invalid --nuts-adapt=$nuts_adapt. Use a nonnegative integer.")
end
if !(0 < nuts_target_accept < 1)
    error("Invalid --nuts-target-accept=$nuts_target_accept. Use a value in (0,1).")
end
if nuts_max_depth < 1
    error("Invalid --nuts-max-depth=$nuts_max_depth. Use an integer >= 1.")
end
if nuts_init_eps < 0
    error("Invalid --nuts-init-eps=$nuts_init_eps. Use a nonnegative value.")
end
if hmc_step_size <= 0
    error("Invalid --hmc-step-size=$hmc_step_size. Use a positive value.")
end
if hmc_leapfrog < 1
    error("Invalid --hmc-leapfrog=$hmc_leapfrog. Use an integer >= 1.")
end
if mh_rw_cprobp <= 0 || mh_rw_cindp <= 0 || mh_rw_curvp <= 0
    error("MH random-walk scales must be positive.")
end
if shock_filter != :sampling && shock_filter != :inversion
    error("Unknown --shock-filter=$shock_filter. Use :sampling or :inversion.")
end
if linear_filter != :kalman && linear_filter != :inversion
    error("Unknown --linear-filter=$linear_filter. Use :kalman or :inversion.")
end
if shock_filter == :inversion
    if sample_shocks
        println("Warning: --sample-shocks ignored because --shock-filter=inversion.")
    end
    if linear_sample_shocks
        println("Warning: --linear-sample-shocks ignored because --shock-filter=inversion.")
    end
    sample_shocks = false
    linear_sample_shocks = false
end
if !gate_use_eps && !gate_use_y
    error("At least one of --gate-use-eps or --gate-use-y must be true.")
end
if !(0.0 <= gate_share_min < gate_share_max <= 1.0)
    error("Invalid gate share bounds: [$gate_share_min, $gate_share_max].")
end
if !(shock_sample_window in (:all, :gate))
    error("Unknown --shock-sample-window=$(shock_sample_window). Use :all or :gate.")
end
if gate_mode != :soft && gate_mode != :hard
    error("Unknown gate_mode=$gate_mode. Use :soft or :hard.")
end
if post_mean_filter != :none && post_mean_filter != :kalman && post_mean_filter != :inversion
    error("Unknown --post-mean-filter=$post_mean_filter. Use :none, :kalman, or :inversion.")
end
if true_theta_filter != :none && true_theta_filter != :kalman && true_theta_filter != :inversion
    error("Unknown --true-theta-filter=$true_theta_filter. Use :none, :kalman, or :inversion.")
end
if chunk_size < 0
    error("Invalid --chunk-size=$chunk_size. Use 0 to disable chunking or a positive integer.")
end
if !(hard_gate_threshold > 0 && hard_gate_threshold < 1)
    error("Invalid --hard-gate-threshold=$hard_gate_threshold. Use a value between 0 and 1.")
end
if eval_only && chain_in == ""
    error("--eval-only requires --chain-in to load an existing chain.")
end
chain_payload = nothing
if chain_in != ""
    if !isfile(chain_in)
        error("Chain file not found: $chain_in")
    end
    chain_payload = MacroModelling.load_hlt_chain_payload(chain_in)
end

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
surrogate_data = surrogate_bundle.payload
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
rom_residual = get(sur_meta, "rom_residual", false)
rom_order = Int(get(sur_meta, "rom_residual_order", 0))
rom_mode_raw = get(sur_meta, "rom_mode", :baseline)
rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)

synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
synthetic_model_name = get(synthetic, "model", "Smets_Wouters_2007_HLT")
if force_obc
    use_obc = true
elseif force_no_obc
    use_obc = false
else
    use_obc = false
    if synthetic_model_name == "Smets_Wouters_2007_HLT_obc"
        println("Warning: synthetic model is OBC; defaulting to non-OBC HLT. Use --use-obc to override.")
    elseif synthetic_model_name != "Smets_Wouters_2007_HLT"
        println("Warning: unknown synthetic model '$synthetic_model_name'; defaulting to non-OBC HLT. Use --use-obc to override.")
    end
end
model_name_for_estimation = use_obc ? "Smets_Wouters_2007_HLT_obc" : "Smets_Wouters_2007_HLT"
hlt_model_file_and_symbol(model_name_for_estimation)  # fail fast on unsupported name
mm_model = load_hlt_model(script_repo_root(), model_name_for_estimation; mod = @__MODULE__)
obs_data = synthetic["obs_data"]
obs_sigma = synthetic["obs_sigma"]
s0 = synthetic["s0"]
shocks = synthetic["shocks"]
theta_true = synthetic["theta_true"]
theta_names = haskey(synthetic, "theta_names") ? synthetic["theta_names"] : Symbol[]
observables = haskey(synthetic, "observables") ? synthetic["observables"] : Symbol[]
state_names = haskey(synthetic, "state_names") ? synthetic["state_names"] : Symbol[]
shock_sigmas = haskey(synthetic, "shock_sigmas") ? synthetic["shock_sigmas"] : vec(std(shocks, dims = 2))
shock_sigmas = shock_sigmas .* shock_prior_scale
d_obs = size(obs_data, 1)
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx_any)
        error("Theta names not found in $(mm_model.model_name) parameters.")
    end
    Int.(idx_any)
end

# --- Dynamic prior infrastructure for N-parameter estimation ---
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

n_theta = length(theta_names)
_param_set_sym = if param_set_name == :auto
    n_theta <= 3 ? :legacy_3params : :phase1_18params
else
    param_set_name
end
_param_specs = get_parameter_specs(_param_set_sym)
_spec_names = [s.name for s in _param_specs]

# Build ordered prior distributions, bounds, and MH step sizes for each theta
_prior_dists = Vector{Distributions.Distribution}(undef, n_theta)
_prior_bounds = Vector{Tuple{Float64,Float64}}(undef, n_theta)
_mh_rw_scales = Vector{Float64}(undef, n_theta)
_theta_init = Vector{Float64}(undef, n_theta)
_baseline = get_phase1_18param_baseline()

for (i, tname) in enumerate(theta_names)
    si = findfirst(==(tname), _spec_names)
    if si !== nothing
        spec = _param_specs[si]
        if spec.prior_type == :Beta
            # Use standard Distributions.Beta with shape params; bounds enforced separately
            _prior_dists[i] = Distributions.Beta(spec.prior_params.α, spec.prior_params.β)
        elseif spec.prior_type == :InvGamma
            _prior_dists[i] = Distributions.InverseGamma(spec.prior_params.α, spec.prior_params.θ)
        elseif spec.prior_type == :Normal
            _prior_dists[i] = Distributions.Normal(spec.prior_params.μ, spec.prior_params.σ)
        else
            _prior_dists[i] = Distributions.Uniform(spec.bounds...)
        end
        _prior_bounds[i] = spec.bounds
        # MH step: scale for ~23% acceptance in d dimensions (Roberts et al. 1997)
        # Optimal: 2.38 * σ_posterior / sqrt(d), approximate σ_posterior ≈ prior_std * 0.5
        d_scale = 2.38 / sqrt(n_theta)
        if spec.prior_type == :Beta
            prior_std = sqrt(spec.prior_params.α * spec.prior_params.β /
                            ((spec.prior_params.α + spec.prior_params.β)^2 *
                             (spec.prior_params.α + spec.prior_params.β + 1)))
            _mh_rw_scales[i] = prior_std * d_scale * 0.5
        elseif spec.prior_type == :InvGamma
            # For InvGamma(2, θ), mean = θ, std = θ/(α-1)/sqrt(α-2) undefined for α≤2
            # Use range-based scaling instead
            _mh_rw_scales[i] = (spec.bounds[2] - spec.bounds[1]) * 0.02 * d_scale
        elseif spec.prior_type == :Normal
            _mh_rw_scales[i] = spec.prior_params.σ * d_scale * 0.5
        else
            _mh_rw_scales[i] = mh_rw_default
        end
        _theta_init[i] = get(_baseline, tname, Distributions.mean(_prior_dists[i]))
    else
        # Fallback: use model baseline with wide normal prior
        bv = mm_model.parameter_values[theta_param_idx[i]]
        _prior_dists[i] = Distributions.Normal(bv, abs(bv) * 0.5 + 0.01)
        _prior_bounds[i] = (bv * 0.1, bv * 5.0)
        _mh_rw_scales[i] = mh_rw_default
        _theta_init[i] = bv
    end
end

# Override with CLI args for legacy 3-param case
if n_theta == 3 && theta_names == [:cprobp, :cindp, :curvp]
    _prior_dists[1] = MacroModelling.Beta(prior_cprobp_mu, prior_cprobp_sd, 0.5, 0.95, μσ = true)
    _prior_dists[2] = MacroModelling.Beta(prior_cindp_mu, prior_cindp_sd, 0.01, 0.99, μσ = true)
    _prior_dists[3] = Distributions.Normal(prior_curvp_mu, prior_curvp_sd)
    _mh_rw_scales[1] = mh_rw_cprobp
    _mh_rw_scales[2] = mh_rw_cindp
    _mh_rw_scales[3] = mh_rw_curvp
end

println("Estimating $n_theta parameters: $theta_names")
println("Prior distributions:")
for i in 1:n_theta
    println("  $(theta_names[i]): $(_prior_dists[i])  bounds=$(_prior_bounds[i])  mh_rw=$(_mh_rw_scales[i])")
end

function hlt_paper_baseline_params()
    return copy(mm_model.parameter_values)
end
baseline_params_linear_loglik = hlt_paper_baseline_params()

rom_predictor = nothing
if !rom_residual
    error("Regime switching requires a ROM1 residual surrogate. Retrain with --rom-residual and --rom-order=1.")
end
if rom_order != 1
    error("Regime switching expects rom_residual_order=1 (ROM1). Retrain with --rom-order=1.")
end
if rom_residual && rom_order != 0
    if rom_mode == :theta
        error("rom_mode=:theta is not supported with AD-based estimation. Retrain with --rom-mode=baseline.")
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
    base_params = hlt_paper_baseline_params()
    theta_idx = Int[]
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
        ensure_rom_cache!(rom_predictor, theta_true === nothing ? zeros(length(theta_idx)) : theta_true)
    end
end

val_rmse = get(surrogate_data, "validation_rmse", nothing)
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

if out_path == ""
    if chain_in != ""
        out_path = chain_in
    else
        out_path = joinpath(dirname(synthetic_path), "hlt_sep_surrogate_estimation_chain.jls")
    end
end
if profile_out == ""
    profile_out = joinpath(dirname(out_path), "hlt_sep_surrogate_estimation_profile.txt")
end
if chunk_size > 0 && checkpoint_path == ""
    checkpoint_path = out_path * ".checkpoint.jls"
end

function build_guided_shocks(obs_data::AbstractMatrix,
                             observables::Vector{Symbol},
                             shock_sigmas::AbstractVector,
                             theta_names::Vector{Symbol},
                             theta_true::Union{AbstractVector,Nothing};
                             guidance::Symbol,
                             guidance_scale::Float64,
                             theta_mode::Symbol)
    if guidance == :none
        return nothing
    end
    filter = guidance == :kalman ? :kalman :
             guidance == :inversion ? :inversion :
             error("Unknown shock guidance: $guidance. Use :none, :kalman, or :inversion.")

    params = MacroModelling.parameters_with_theta_mode(
        hlt_paper_baseline_params(),
        mm_model.parameters,
        theta_names,
        theta_true;
        theta_mode = theta_mode,
        mode_label = "shock guidance theta mode",
        theta_label = "Shock guidance theta names",
    )

    guided = MacroModelling.estimate_observed_shocks_matrix(
        mm_model,
        obs_data,
        observables;
        parameters = params,
        algorithm = :first_order,
        filter = filter,
        data_in_levels = true,
        smooth = false,
        verbose = false,
        expected_rows = length(shock_sigmas),
        expected_cols = size(obs_data, 2),
        label = "Guided shock matrix",
    )

    if guidance_scale != 1.0
        guided .*= guidance_scale
    end

    zero_idx = findall(shock_sigmas .== 0)
    if !isempty(zero_idx)
        guided[zero_idx, :] .= 0.0
    end

    return guided
end

shock_guided = nothing
if sample_shocks && shock_guidance != :none
    if isempty(observables)
        error("Synthetic data is missing observables; cannot compute guided shocks.")
    end
    shock_guided = build_guided_shocks(obs_data,
                                       observables,
                                       shock_sigmas,
                                       theta_names,
                                       theta_true;
                                       guidance = shock_guidance,
                                       guidance_scale = shock_guidance_scale,
                                       theta_mode = shock_guidance_theta)
elseif !sample_shocks && shock_guidance != :none
    println("Shock guidance requested but shock sampling is disabled; ignoring guided shocks.")
end

function filtered_shocks_from_theta(obs_data::AbstractMatrix,
                                    observables::Vector{Symbol},
                                    shock_sigmas::AbstractVector,
                                    theta_vec::AbstractVector,
                                    theta_names::Vector{Symbol};
                                    filter::Symbol)
    if filter != :kalman && filter != :inversion
        error("Unknown post-mean filter=$filter. Use :kalman or :inversion.")
    end
    params = MacroModelling.override_named_parameters(
        hlt_paper_baseline_params(),
        mm_model.parameters,
        theta_names,
        theta_vec;
        label = "Theta names",
    )
    shocks = MacroModelling.estimate_observed_shocks_matrix(
        mm_model,
        obs_data,
        observables;
        parameters = params,
        algorithm = :first_order,
        filter = filter,
        data_in_levels = true,
        smooth = false,
        verbose = false,
        expected_rows = length(shock_sigmas),
        expected_cols = size(obs_data, 2),
        label = "Filtered shock matrix",
    )
    zero_idx = findall(shock_sigmas .== 0)
    if !isempty(zero_idx)
        shocks[zero_idx, :] .= 0.0
    end
    return shocks
end

if obs_window > 0
    T_full = size(obs_data, 2)
    if obs_window >= T_full
        println("Obs window ($obs_window) >= full sample ($T_full); using full sample.")
    else
        if obs_window_mode != :last
            error("Only obs_window_mode=:last is supported.")
        end
        start_idx = T_full - obs_window + 1
        if obs_window_init == :linear
            if start_idx <= 1
                error("Linear window init requires a non-empty prefix.")
            end
            prefix_obs = obs_data[:, 1:(start_idx - 1)]
            params_window = MacroModelling.parameters_with_theta_mode(
                hlt_paper_baseline_params(),
                mm_model.parameters,
                theta_names,
                theta_true;
                theta_mode = obs_window_theta,
                mode_label = "obs_window_theta",
                theta_label = "Theta names",
            )
            s0 = MacroModelling.linear_filter_initial_state(
                mm_model,
                prefix_obs,
                observables,
                state_names;
                parameters = params_window,
                filter = obs_window_filter,
                algorithm = :first_order,
                label = "Linear filter variables",
            )
            println("Linear window init: filter=$obs_window_filter, theta=$obs_window_theta.")
        elseif obs_window_init == :surrogate
            θ_window = if obs_window_theta == :synthetic
                theta_true
            elseif obs_window_theta == :baseline
                MacroModelling.extract_named_parameters(
                    hlt_paper_baseline_params(),
                    mm_model.parameters,
                    theta_names;
                    label = "Theta names",
                )
            else
                error("Unknown obs_window_theta=$obs_window_theta. Use :synthetic or :baseline.")
            end

            window_shocks = if obs_window_shocks == :auto
                shock_guided !== nothing ? shock_guided : shocks
            elseif obs_window_shocks == :guided
                shock_guided === nothing && error("obs_window_shocks=:guided requires shock guidance.")
                shock_guided
            elseif obs_window_shocks == :true
                shocks
            else
                error("Unknown obs_window_shocks=$obs_window_shocks. Use :auto, :guided, or :true.")
            end

            predict_sur_window = function (state, shock_t, θ_local)
                x = vcat(state, shock_t, θ_local)
                y_resid = predict_frozen(frozen, x)
                if rom_predictor === nothing
                    y = y_resid
                else
                    y = rom_predict(rom_predictor, state, shock_t, θ_local) .+ y_resid
                end
                return y[1:d_obs], y[(d_obs + 1):end]
            end
            s0 = MacroModelling.advance_state(predict_sur_window, s0, window_shocks, θ_window, start_idx - 1)
        else
            error("Unknown obs_window_init=$obs_window_init. Use :linear or :surrogate.")
        end
        obs_data = obs_data[:, start_idx:end]
        shocks = shocks[:, start_idx:end]
        if shock_guided !== nothing
            shock_guided = shock_guided[:, start_idx:end]
        end
        println("Using last $obs_window periods (t=$start_idx:$T_full); s0 updated via surrogate.")
    end
end

gate_mask = trues(size(obs_data, 2))
gate_info = Dict{String,Any}()
gate_stats = Dict{String,Any}()
gate_periods = gate_periods_arg >= 0 ? gate_periods_arg : 1
gate_probs = nothing

gate_path = gate_calibration_path
if gate_path == ""
    candidate = joinpath(dirname(synthetic_path), "gate_calibration.jls")
    if isfile(candidate)
        gate_path = candidate
    end
end

if gate_path != "" && isfile(gate_path)
    gate_calib = MacroModelling.load_hlt_gate_calibration(gate_path)
    tau_eps = gate_calib["tau_eps"]
    tau_y = gate_calib["tau_y"]
    shock_norm = Symbol(gate_calib["shock_norm"])
    error_norm = Symbol(gate_calib["error_norm"])
    if gate_periods_arg < 0 && haskey(gate_calib, "periods")
        gate_periods = gate_calib["periods"]
    end

    if get(gate_calib, "model", mm_model.model_name) != mm_model.model_name
        println("Warning: gate calibration model $(get(gate_calib, "model", "unknown")) != $(mm_model.model_name).")
    end

    lin_obs = nothing
    gate_shocks = nothing
    e_stat = nothing
    f_stat = nothing
    cached_filter_ok = !haskey(gate_calib, "shock_filter") || Symbol(gate_calib["shock_filter"]) == gate_shock_filter
    cached_periods_ok = !haskey(gate_calib, "periods") || Int(gate_calib["periods"]) == gate_periods
    cached_lengths_ok = haskey(gate_calib, "e_stats") && haskey(gate_calib, "f_stats") &&
                        length(gate_calib["e_stats"]) == size(obs_data, 2) &&
                        length(gate_calib["f_stats"]) == size(obs_data, 2)
    use_cached_gate_stats = gate_use_cached_stats && cached_filter_ok && cached_periods_ok && cached_lengths_ok
    if use_cached_gate_stats
        e_stat = vec(Float64.(gate_calib["e_stats"]))
        f_stat = vec(Float64.(gate_calib["f_stats"]))
        if !all(isfinite, e_stat) || !all(isfinite, f_stat)
            println("Warning: cached gate statistics are non-finite; recomputing gate stats.")
            use_cached_gate_stats = false
        else
            println("Using cached gate statistics from calibration payload.")
        end
    elseif gate_use_cached_stats && haskey(gate_calib, "e_stats") && haskey(gate_calib, "f_stats")
        println("Warning: cached gate statistics incompatible with current gate settings; recomputing gate stats.")
    end
    if !use_cached_gate_stats
        params_gate = MacroModelling.parameters_with_theta_mode(
            hlt_paper_baseline_params(),
            mm_model.parameters,
            theta_names,
            theta_true;
            theta_mode = gate_theta_mode,
            mode_label = "gate theta mode",
            theta_label = "Theta names",
        )
        lin_obs, gate_shocks, e_stat, f_stat = MacroModelling.compute_linear_gate_stats_from_filter(
            mm_model,
            obs_data,
            observables,
            obs_sigma,
            shock_sigmas,
            state_names;
            periods = gate_periods,
            parameters = params_gate,
            filter = gate_shock_filter,
            algorithm = :first_order,
            shock_norm = shock_norm,
            error_norm = error_norm,
            ignore_obc = false,
            label = "Gate linear simulation",
        )
    end
    eps_mask = gate_use_eps ? (e_stat .> tau_eps) : falses(length(e_stat))
    y_mask = gate_use_y ? (f_stat .> tau_y) : falses(length(f_stat))
    base_mask = eps_mask .| y_mask
    gate_mask = MacroModelling.apply_gate_padding(base_mask, gate_k_pre, gate_k_post, gate_min_len)

    gate_info = Dict(
        "gate_calibration" => gate_path,
        "tau_eps" => tau_eps,
        "tau_y" => tau_y,
        "shock_norm" => String(shock_norm),
        "error_norm" => String(error_norm),
        "periods" => gate_periods,
        "k_pre" => gate_k_pre,
        "k_post" => gate_k_post,
        "min_len" => gate_min_len,
        "filter" => String(gate_shock_filter),
        "theta_mode" => String(gate_theta_mode),
        "gate_mode" => String(gate_mode),
        "gate_beta_eps" => gate_beta_eps,
        "gate_beta_y" => gate_beta_y,
        "gate_bias" => gate_bias_arg,
        "gate_prob_floor" => gate_prob_floor,
        "gate_prob_ceiling" => gate_prob_ceiling,
        "gate_use_eps" => gate_use_eps,
        "gate_use_y" => gate_use_y,
        "gate_use_cached_stats" => gate_use_cached_stats,
        "gate_used_cached_stats" => use_cached_gate_stats,
    )
    if gate_mode == :soft
        target_share = get(gate_calib, "target_share", mean(base_mask))
        soft_window = isempty(system_priors) ? trues(length(base_mask)) : falses(length(base_mask))
        for (start, stop, _) in system_priors
            lo = max(1, start)
            hi = min(length(soft_window), stop)
            if lo <= hi
                soft_window[lo:hi] .= true
            end
        end
        prior_default = isnan(system_prior_default) ? target_share : system_prior_default
        prior_probs = fill(prior_default, length(base_mask))
        for (start, stop, prob) in system_priors
            lo = max(1, start)
            hi = min(length(prior_probs), stop)
            if lo <= hi
                prior_probs[lo:hi] .= prob
            end
        end
        prior_probs = clamp.(prior_probs, gate_prob_floor, gate_prob_ceiling)
        prior_logit = MacroModelling.logit.(prior_probs)
        eps_scale = max(tau_eps, eps(Float64))
        y_scale = max(tau_y, eps(Float64))
        eps_score = gate_use_eps ? gate_beta_eps .* ((e_stat .- tau_eps) ./ eps_scale) : zeros(length(e_stat))
        y_score = gate_use_y ? gate_beta_y .* ((f_stat .- tau_y) ./ y_scale) : zeros(length(f_stat))
        score = eps_score .+ y_score
        target_soft = mean(soft_window) > 0 ? mean(base_mask[soft_window]) : target_share
        gate_bias = isnan(gate_bias_arg) ? MacroModelling.calibrate_gate_bias(score[soft_window] .+ prior_logit[soft_window], target_soft) : gate_bias_arg
        gate_probs = similar(score)
        for t in eachindex(score)
            if soft_window[t]
                gate_probs[t] = MacroModelling.logistic(gate_bias + score[t] + prior_logit[t])
            else
                gate_probs[t] = base_mask[t] ? gate_prob_ceiling : gate_prob_floor
            end
        end
        gate_probs = clamp.(gate_probs, gate_prob_floor, gate_prob_ceiling)
        gate_info["gate_bias"] = gate_bias
        gate_info["target_share"] = target_share
        gate_info["target_share_soft"] = target_soft
        gate_info["system_prior_default"] = prior_default
        gate_info["system_priors"] = system_priors
        gate_info["soft_window"] = soft_window
    end
    soft_window_local = gate_mode == :soft ? gate_info["soft_window"] : falses(length(base_mask))
    gate_stats = Dict(
        "e_stat" => e_stat,
        "f_stat" => f_stat,
        "eps_mask" => eps_mask,
        "y_mask" => y_mask,
        "base_mask" => base_mask,
        "gate_mask" => gate_mask,
        "gate_probs" => gate_probs,
        "soft_window" => soft_window_local,
        "lin_obs" => lin_obs,
        "gate_shocks" => gate_shocks,
        "system_priors" => system_priors,
    )
else
    println("Gate calibration not found; using SEP for all periods.")
    if gate_mode == :soft
        gate_probs = fill(1.0, size(obs_data, 2))
    end
end

if !isempty(hard_gate_ranges)
    manual_mask = falses(length(gate_mask))
    for (start, stop) in hard_gate_ranges
        lo = max(1, start)
        hi = min(length(manual_mask), stop)
        if lo <= hi
            manual_mask[lo:hi] .= true
        end
    end
    manual_mask = MacroModelling.apply_gate_padding(manual_mask, gate_k_pre, gate_k_post, gate_min_len)
    gate_mask = manual_mask
    gate_info["manual_gate_ranges"] = hard_gate_ranges
    gate_info["manual_gate_active"] = true
    gate_stats["manual_gate_mask"] = manual_mask
    gate_stats["manual_gate_ranges"] = hard_gate_ranges
    if gate_mode == :soft
        println("Warning: --hard-gate provided; forcing gate_mode=:hard.")
        gate_mode = :hard
        gate_probs = nothing
    end
    gate_info["gate_mode"] = String(gate_mode)
end

if chain_payload !== nothing && (eval_only || chain_in != "") && (use_chain_gate || gate_calibration_path == "")
    if haskey(chain_payload, "gate_mask")
        gate_mask = Bool.(chain_payload["gate_mask"])
    end
    if haskey(chain_payload, "gate_probs")
        gate_probs = chain_payload["gate_probs"]
    end
    if haskey(chain_payload, "gate_info")
        gate_info = chain_payload["gate_info"]
        if haskey(gate_info, "gate_mode")
            gate_mode = Symbol(gate_info["gate_mode"])
        end
    end
    if haskey(chain_payload, "gate_stats")
        gate_stats = chain_payload["gate_stats"]
    end
    if length(gate_mask) != size(obs_data, 2)
        error("Gate mask length ($(length(gate_mask))) does not match T=$(size(obs_data, 2)) from synthetic data.")
    end
    if gate_probs !== nothing && length(gate_probs) != size(obs_data, 2)
        error("Gate probs length ($(length(gate_probs))) does not match T=$(size(obs_data, 2)) from synthetic data.")
    end
    println("Using gate mask/probabilities from chain payload.")
end

gate_share = gate_mode == :soft && gate_probs !== nothing ? mean(gate_probs) : mean(gate_mask)
isfinite(gate_share) || error("Gate share is non-finite.")
if gate_share < gate_share_min || gate_share > gate_share_max
    error("Gate share $(round(gate_share, digits = 4)) is outside allowed bounds [$gate_share_min, $gate_share_max]. Adjust gate calibration or padding.")
end
if fail_degenerate_gate && (gate_share <= 0 || gate_share >= 1)
    error("Degenerate gate share $(round(gate_share, digits = 4)); disable padding or recalibrate gate.")
end
shock_sample_mask = trues(size(obs_data, 2))
if sample_shocks && shock_sample_window == :gate
    if gate_mode == :soft && haskey(gate_info, "soft_window")
        shock_sample_mask = Bool.(gate_info["soft_window"])
    else
        shock_sample_mask = gate_mask
    end
    if !any(shock_sample_mask)
        println("Warning: shock-sample-window=gate has no periods; sampling shocks for all periods.")
        shock_sample_mask .= true
    end
end
shock_sample_idx = findall(shock_sample_mask)

hard_gate_mask = gate_mask
hard_gate_source = "gate_mask"
if gate_mode == :soft && gate_probs !== nothing
    hard_gate_mask = gate_probs .>= hard_gate_threshold
    hard_gate_source = "gate_probs>=threshold"
end
hard_gate_share = mean(hard_gate_mask)
if fail_degenerate_gate && (hard_gate_share <= 0 || hard_gate_share >= 1)
    error("Degenerate hard gate share $(round(hard_gate_share, digits = 4)); adjust --hard-gate-threshold or gate calibration.")
end

d_obs = size(obs_data, 1)
rom_full_predict = function (state::AbstractVector, shock_t::AbstractVector, θ_local::AbstractVector)
    rom_predictor === nothing && error("ROM predictor missing; cannot evaluate linear/surrogate predictors.")
    return rom_predict(rom_predictor, state, shock_t, θ_local)
end

# Build theta padding: surrogate may have been trained with more theta parameters
# than are being estimated. Map estimation theta into the full surrogate theta vector.
surrogate_theta_names = get(sur_meta, "theta_names", Symbol[])
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    # Build mapping: for each surrogate theta, find index in estimation theta or use baseline
    _sur_theta_baseline = zeros(Float64, length(surrogate_theta_names))
    _sur_theta_est_idx = zeros(Int, length(surrogate_theta_names))  # 0 = use baseline
    _base_params = hlt_paper_baseline_params()
    for (si, sname) in enumerate(surrogate_theta_names)
        ei = findfirst(==(sname), theta_names)
        if ei !== nothing
            _sur_theta_est_idx[si] = ei
        else
            pi = findfirst(==(sname), mm_model.parameters)
            _sur_theta_baseline[si] = pi !== nothing ? _base_params[pi] : 0.0
        end
    end
    println("Surrogate theta padding: $(length(surrogate_theta_names)) surrogate params, $(length(theta_names)) estimation params")
    println("  Estimated params at surrogate positions: ", [(surrogate_theta_names[i], _sur_theta_est_idx[i]) for i in 1:length(surrogate_theta_names) if _sur_theta_est_idx[i] > 0])

    function _pad_theta(θ_local::AbstractVector)
        θ_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                θ_full[i] = θ_local[_sur_theta_est_idx[i]]
            end
        end
        return θ_full
    end
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, _pad_theta(θ_local)))
else
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, θ_local))
end
surrogate_step_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
    rom_full_predict,
    surrogate_residual_predict,
    state,
    shock_t,
    θ_local,
    d_obs;
    allow_full_residual = true,
)

# ROM-only predict (linear): used for robust shock recovery in the inversion filter.
# Returns (obs, state_next) using only the first-order perturbation solution.
rom_only_predict = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
    rom_full_predict,
    state,
    shock_t,
    θ_local,
    d_obs,
)

function linear_gate_loglik_per_period(obs_data_ka,
                                       θ::AbstractVector,
                                       theta_names::Vector{Symbol},
                                       s0::AbstractVector,
                                       shocks::AbstractMatrix;
                                       shock_filter::Symbol,
                                       linear_filter::Symbol)
    rom_predictor === nothing && error("ROM predictor missing; cannot compute linear gate loglikelihood.")
    predict_lin = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
        rom_full_predict,
        state,
        shock_t,
        θ_local,
        d_obs,
    )
    kalman_linear_loglik = θ_local -> MacroModelling.linear_model_loglik_per_period(
        mm_model,
        obs_data_ka,
        θ_local,
        theta_names;
        model_parameter_names = mm_model.parameters,
        base_parameters = baseline_params_linear_loglik,
        theta_idx = theta_param_idx,
        algorithm = :first_order,
        filter = :kalman,
        on_failure_loglikelihood = -1e12,
        presample_periods = 0,
        initial_covariance = :theoretical,
        verbose = false,
        theta_label = "Theta names",
    )
    # When linear_filter=:kalman, use the proper Kalman filter (re-solves model for theta)
    # rather than falling through to conditional_loglik which uses fixed ROM matrices.
    # This ensures the likelihood is theta-dependent for linear (non-gate) periods.
    if linear_filter == :kalman
        return kalman_linear_loglik(θ)
    end
    return MacroModelling.linear_reference_loglik_per_period(
        θ,
        s0,
        shocks,
        obs_data,
        obs_sigma,
        shock_sigmas;
        shock_filter = shock_filter,
        linear_filter = linear_filter,
        predict_linear = predict_lin,
        kalman_linear_loglik = kalman_linear_loglik,
        inversion_maxit = inversion_maxit,
        inversion_tol = inversion_tol,
        inversion_lambda = inversion_lambda,
    )
end

function logprior_theta(theta_vec::AbstractVector)
    lp = 0.0
    for i in eachindex(theta_vec)
        lb, ub = _prior_bounds[i]
        if theta_vec[i] < lb || theta_vec[i] > ub
            return -Inf
        end
        d = Distributions.truncated(_prior_dists[i], lb, ub)
        lp += Distributions.logpdf(d, theta_vec[i])
    end
    return lp
end

function loglik_at_theta(theta_vec::AbstractVector,
                         shocks_eval::AbstractMatrix)
    if shock_filter == :inversion
        # FIX AD-04: Use ROM1 (linear) for robust shock recovery,
        # then evaluate likelihood with ROM1 + surrogate (nonlinear).
        ll_sep, _ = MacroModelling.inversion_loglik_per_period(rom_only_predict,
                                                               s0,
                                                               theta_vec,
                                                               obs_data,
                                                               obs_sigma,
                                                               shock_sigmas;
                                                               eval_predict_fn = surrogate_step_predict,
                                                               maxit = inversion_maxit,
                                                               tol = inversion_tol,
                                                               lambda = inversion_lambda,
                                                               refine_maxit = refine_maxit,
                                                               refine_tol = refine_tol)
        ll_lin = linear_gate_loglik_per_period(obs_data_ka, theta_vec, theta_names, s0, shocks_eval;
                                               shock_filter = shock_filter,
                                               linear_filter = linear_filter)
        if gate_mode == :soft && gate_probs !== nothing
            return MacroModelling.mix_loglikelihood(ll_sep, ll_lin, gate_probs)
        end
        ll_total = sum(ll_sep[gate_mask])
        if use_linear_loglik
            ll_total += sum(ll_lin[.!gate_mask])
        end
        return ll_total
    end
    if gate_mode == :soft && gate_probs !== nothing
        ll_sep = MacroModelling.additive_residual_loglik_per_period(
            rom_full_predict,
            surrogate_residual_predict,
            s0,
            shocks_eval,
            theta_vec,
            obs_data,
            obs_sigma;
            d_obs = d_obs,
            allow_full_residual = true,
        )
        ll_lin = linear_gate_loglik_per_period(obs_data_ka, theta_vec, theta_names, s0, shocks_eval;
                                               shock_filter = shock_filter,
                                               linear_filter = linear_filter)
        return MacroModelling.mix_loglikelihood(ll_sep, ll_lin, gate_probs)
    end

    ll_sep_vec = MacroModelling.additive_residual_loglik_per_period(
        rom_full_predict,
        surrogate_residual_predict,
        s0,
        shocks_eval,
        theta_vec,
        obs_data,
        obs_sigma;
        d_obs = d_obs,
        allow_full_residual = true,
    )
    ll_sep = sum(ll_sep_vec[gate_mask])
    if use_linear_loglik
        ll_lin = linear_gate_loglik_per_period(obs_data_ka, theta_vec, theta_names, s0, shocks_eval;
                                               shock_filter = shock_filter,
                                               linear_filter = linear_filter)
        return ll_sep + sum(ll_lin[.!gate_mask])
    end
    return ll_sep
end

function theta_laplace_log_marginal(theta_vec::AbstractVector,
                                    theta_cov::AbstractMatrix,
                                    loglik_val::Float64,
                                    logprior_val::Float64)
    d = length(theta_vec)
    cov_sym = Symmetric(theta_cov)
    logdet_cov = LinearAlgebra.logabsdet(cov_sym)[1]
    return loglik_val + logprior_val + 0.5 * (d * log(2 * pi) + logdet_cov)
end

function sample_in_chunks(model, sampler, total_samples::Int;
                          chunk_size::Int,
                          n_chains::Int,
                          use_threads::Bool,
                          init_params,
                          checkpoint_path::String)
    return MacroModelling.run_chunked_sampling(
        total_samples,
        chunk_size;
        sample_chunk = function (n_i, i, n_chunks)
            println("Sampling chunk $i/$n_chunks ($n_i draws)...")
            flush(stdout)
            if use_threads
                return Turing.sample(model, sampler, Turing.MCMCThreads(), n_i, n_chains;
                                     progress = false,
                                     initial_params = init_params)
            elseif n_chains > 1
                return Turing.sample(model, sampler, Turing.MCMCSerial(), n_i, n_chains;
                                     progress = false,
                                     initial_params = init_params)
            end
            return Turing.sample(model, sampler, n_i;
                                 progress = false,
                                 initial_params = init_params)
        end,
        concat_chunks = (a, b) -> MCMCChains.chainscat(a, b),
        on_chunk = function (i, n_chunks, _n_i, chunk_chain, samps, elapsed_s)
            acc, div, step = MacroModelling.chunk_stats(chunk_chain)
            if acc !== nothing || div !== nothing || step !== nothing
                println("Chunk $i stats: accept=$(acc), divergences=$(div), step=$(step)")
            end
            println("Completed chunk $i/$n_chunks in $(round(elapsed_s, digits = 1))s")
            flush(stdout)
            if checkpoint_path != ""
                payload = MacroModelling.build_chain_checkpoint_payload(
                    samps;
                    chunks_done = i,
                    samples_done = min(i * chunk_size, total_samples),
                )
                serialize(checkpoint_path, payload)
                println("Saved checkpoint: $checkpoint_path")
                flush(stdout)
            end
        end,
    )
end

function build_sampler(sampler_mode::Symbol;
                       nuts_adapt::Int,
                       nuts_target_accept::Float64,
                       nuts_max_depth::Int,
                       nuts_init_eps::Float64,
                       hmc_step_size::Float64,
                       hmc_leapfrog::Int,
                       mh_rw_cprobp::Float64,
                       mh_rw_cindp::Float64,
                       mh_rw_curvp::Float64)
    if sampler_mode == :nuts
        return Turing.NUTS(nuts_adapt, nuts_target_accept;
                           max_depth = nuts_max_depth,
                           init_ϵ = nuts_init_eps,
                           adtype = AutoForwardDiff())
    elseif sampler_mode == :hmc
        return Turing.HMC(hmc_step_size, hmc_leapfrog; adtype = AutoForwardDiff())
    end
    # Build MH proposal: multivariate random walk for theta_vec
    cov_diag = [_mh_rw_scales[i]^2 for i in 1:n_theta]
    return Turing.MH(
        :theta_vec => Turing.Inference.AdvancedMH.RandomWalkProposal(
            Distributions.MvNormal(zeros(n_theta), Diagonal(cov_diag)))
    )
end

function sampler_descriptor(sampler_mode::Symbol;
                            nuts_adapt::Int,
                            nuts_target_accept::Float64,
                            nuts_max_depth::Int,
                            nuts_init_eps::Float64,
                            hmc_step_size::Float64,
                            hmc_leapfrog::Int,
                            mh_rw_cprobp::Float64,
                            mh_rw_cindp::Float64,
                            mh_rw_curvp::Float64)
    if sampler_mode == :nuts
        return "nuts(adapt=$(nuts_adapt), target_accept=$(nuts_target_accept), max_depth=$(nuts_max_depth), init_eps=$(nuts_init_eps))"
    elseif sampler_mode == :hmc
        return "hmc(step_size=$(hmc_step_size), leapfrog=$(hmc_leapfrog))"
    end
    return "mh(rw_cprobp=$(mh_rw_cprobp), rw_cindp=$(mh_rw_cindp), rw_curvp=$(mh_rw_curvp))"
end

obs_data_ka = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))
use_linear_loglik = gate_mode == :soft || any(.!gate_mask)

Turing.@model function hlt_surrogate_model(obs_data,
                                           frozen,
                                           s0,
                                           shocks_fixed,
                                           obs_sigma,
                                           shock_sigmas,
                                           shock_guided,
                                           sample_shocks::Bool,
                                           shock_sample_idx::AbstractVector{Int},
                                           obs_data_ka,
                                           theta_names::Vector{Symbol},
                                           gate_mask::AbstractVector{Bool},
                                           gate_probs,
                                           gate_mode::Symbol,
                                           use_linear_loglik::Bool,
                                           shock_filter::Symbol,
                                           linear_filter::Symbol,
                                           inversion_maxit::Int,
                                           inversion_tol::Float64,
                                           inversion_lambda::Float64)
    # Dynamic N-parameter prior using proper truncated distributions
    # This enables correct bijectors for HMC/NUTS (ForwardDiff-compatible)
    theta_vec ~ Turing.arraydist([
        Distributions.truncated(_prior_dists[i], _prior_bounds[i][1], _prior_bounds[i][2])
        for i in 1:n_theta
    ])
    θ = theta_vec

    d_eps = length(shock_sigmas)
    T = size(obs_data, 2)
    if sample_shocks
        structural_idx = findall(shock_sigmas .> 0)
        if isempty(structural_idx)
            shocks = zeros(eltype(θ), d_eps, T)
        else
            sample_idx = shock_sample_idx
            if isempty(sample_idx)
                sample_idx = collect(1:T)
            end
            ε ~ Turing.filldist(Normal(0, 1), length(structural_idx), length(sample_idx))
            Tε = eltype(ε)
            sigmas = Tε.(shock_sigmas[structural_idx])
            base = shock_guided === nothing ? shocks_fixed : shock_guided
            shocks = Tε.(base)
            for (j, idx) in enumerate(structural_idx)
                shocks[idx, sample_idx] .= shocks[idx, sample_idx] .+ ε[j, :] .* sigmas[j]
            end
        end
    else
        shocks = shocks_fixed
    end

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        if shock_filter == :inversion
            ll_sep, _ = MacroModelling.inversion_loglik_per_period(surrogate_step_predict,
                                                                   s0,
                                                                   θ,
                                                                   obs_data,
                                                                   obs_sigma,
                                                                   shock_sigmas;
                                                                   maxit = inversion_maxit,
                                                                   tol = inversion_tol,
                                                                   lambda = inversion_lambda,
                                                                   refine_maxit = refine_maxit,
                                                                   refine_tol = refine_tol)
            ll_lin = linear_gate_loglik_per_period(obs_data_ka, θ, theta_names, s0, shocks;
                                                   shock_filter = :inversion,
                                                   linear_filter = linear_filter)
            if gate_mode == :soft
                gate_probs === nothing && error("gate_mode=:soft requires gate_probs.")
                Turing.@addlogprob!(MacroModelling.mix_loglikelihood(ll_sep, ll_lin, gate_probs))
            else
                ll_total = sum(ll_sep[gate_mask])
                if use_linear_loglik
                    ll_total += sum(ll_lin[.!gate_mask])
                end
                Turing.@addlogprob!(ll_total)
            end
        else
            if gate_mode == :soft
                gate_probs === nothing && error("gate_mode=:soft requires gate_probs.")
                ll_sep = MacroModelling.additive_residual_loglik_per_period(
                    rom_full_predict,
                    surrogate_residual_predict,
                    s0,
                    shocks,
                    θ,
                    obs_data,
                    obs_sigma;
                    d_obs = d_obs,
                    allow_full_residual = true,
                )
                ll_lin = linear_gate_loglik_per_period(obs_data_ka, θ, theta_names, s0, shocks;
                                                       shock_filter = shock_filter,
                                                       linear_filter = linear_filter)
                Turing.@addlogprob!(MacroModelling.mix_loglikelihood(ll_sep, ll_lin, gate_probs))
            else
                ll_sep_vec = MacroModelling.additive_residual_loglik_per_period(
                    rom_full_predict,
                    surrogate_residual_predict,
                    s0,
                    shocks,
                    θ,
                    obs_data,
                    obs_sigma;
                    d_obs = d_obs,
                    allow_full_residual = true,
                )
                ll_sep = sum(ll_sep_vec[gate_mask])
                ll_linear = zero(eltype(θ))
                if use_linear_loglik
                    ll_vec = linear_gate_loglik_per_period(obs_data_ka, θ, theta_names, s0, shocks;
                                                           shock_filter = shock_filter,
                                                           linear_filter = linear_filter)
                    ll_linear = sum(ll_vec[.!gate_mask])
                end
                Turing.@addlogprob! (ll_sep + ll_linear)
            end
        end
    end
end

println("Running surrogate-based synthetic estimation")
println("Surrogate: $surrogate_path")
println("Synthetic data: $synthetic_path")
println("Model: $(mm_model.model_name) (use_obc=$use_obc)")
println("True theta: $theta_true")
println("Shock filter: $shock_filter")
println("Linear filter: $linear_filter")
println("Shock sampling: $(sample_shocks ? "enabled" : "fixed")")
if linear_samples > 0
    println("Linear shock sampling: $(linear_sample_shocks ? "enabled" : "fixed")")
end
println("Sampler: " * sampler_descriptor(sampler_mode;
                                         nuts_adapt = nuts_adapt,
                                         nuts_target_accept = nuts_target_accept,
                                         nuts_max_depth = nuts_max_depth,
                                         nuts_init_eps = nuts_init_eps,
                                         hmc_step_size = hmc_step_size,
                                         hmc_leapfrog = hmc_leapfrog,
                                         mh_rw_cprobp = mh_rw_cprobp,
                                         mh_rw_cindp = mh_rw_cindp,
                                         mh_rw_curvp = mh_rw_curvp))
if sample_shocks
    println("Shock sample window: $shock_sample_window (periods=$(length(shock_sample_idx))/$(size(obs_data, 2)))")
end
println("Chains: $n_chains (samples per chain: $n_samples)")
println("MCMC threads: $(mcmc_threads ? "enabled" : "disabled") (available=$(Threads.nthreads()))")
if sample_shocks
    println("Shock sigma min/max: $(minimum(shock_sigmas)) / $(maximum(shock_sigmas))")
    if shock_guided !== nothing
        println("Shock guidance: $shock_guidance (theta = $shock_guidance_theta, scale = $shock_guidance_scale)")
        println("Guided shock min/max: $(minimum(shock_guided)) / $(maximum(shock_guided))")
    end
end
println("Obs sigma mode: $obs_sigma_mode (min/max = $(minimum(obs_sigma)) / $(maximum(obs_sigma)))")
println("Priors ($n_theta parameters):")
for i in 1:n_theta
    println("  $(theta_names[i]): $(_prior_dists[i])  init=$(_theta_init[i])")
end
if isempty(gate_info)
    println("Gate calibration: none (SEP for all periods)")
    println("Gate mode: $gate_mode")
else
    println("Gate calibration: $(gate_info["gate_calibration"])")
    println("Gate mode: $gate_mode")
    println("Gate share: $(round(gate_share, digits = 4)) (k_pre=$(gate_info["k_pre"]), k_post=$(gate_info["k_post"]), min_len=$(gate_info["min_len"]))")
end
if gate_mode == :soft
    prior_label = isnan(system_prior_default) ? "auto" : string(system_prior_default)
    println("System prior default: $prior_label")
    if !isempty(system_priors)
        println("System priors: $(system_priors)")
    end
end

ll_lin_true = nothing
ll_rs_true = nothing
ll_rs_true_hard = nothing
ll_rs_true_hard_filtered = nothing
ll_rs_true_hard_filtered_source = nothing
ll_lin_true_conditional = nothing
ll_lin_true_conditional_filtered = nothing
ll_lin_true_conditional_filtered_source = nothing
if theta_true !== nothing
    if shock_filter == :inversion
        ll_lin_vec = linear_gate_loglik_per_period(obs_data_ka, theta_true, theta_names, s0, shocks;
                                                   shock_filter = shock_filter,
                                                   linear_filter = linear_filter)
        ll_lin_true = sum(ll_lin_vec)
        ll_sep_true, _ = MacroModelling.inversion_loglik_per_period(surrogate_step_predict,
                                                                    s0,
                                                                    theta_true,
                                                                    obs_data,
                                                                    obs_sigma,
                                                                    shock_sigmas;
                                                                    maxit = inversion_maxit,
                                                                    tol = inversion_tol,
                                                                    lambda = inversion_lambda,
                                                                    refine_maxit = refine_maxit,
                                                                    refine_tol = refine_tol)
        if gate_mode == :soft && gate_probs !== nothing
            ll_rs_true = MacroModelling.mix_loglikelihood(ll_sep_true, ll_lin_vec, gate_probs)
        else
            ll_rs_true = sum(ll_sep_true[gate_mask])
            if use_linear_loglik
                ll_rs_true += sum(ll_lin_vec[.!gate_mask])
            end
        end
    else
        ll_lin_vec = linear_gate_loglik_per_period(obs_data_ka, theta_true, theta_names, s0, shocks;
                                                   shock_filter = shock_filter,
                                                   linear_filter = linear_filter)
        ll_lin_true = sum(ll_lin_vec)
        if gate_mode == :soft && gate_probs !== nothing
            ll_sep_true = MacroModelling.additive_residual_loglik_per_period(
                rom_full_predict,
                surrogate_residual_predict,
                s0,
                shocks,
                theta_true,
                obs_data,
                obs_sigma;
                d_obs = d_obs,
                allow_full_residual = true,
            )
            ll_rs_true = MacroModelling.mix_loglikelihood(ll_sep_true, ll_lin_vec, gate_probs)
        else
            ll_sep_true = MacroModelling.additive_residual_loglik_per_period(
                rom_full_predict,
                surrogate_residual_predict,
                s0,
                shocks,
                theta_true,
                obs_data,
                obs_sigma;
                d_obs = d_obs,
                allow_full_residual = true,
            )
            ll_rs_true = sum(ll_sep_true[gate_mask])
            if use_linear_loglik
                ll_rs_true += sum(ll_lin_vec[.!gate_mask])
            end
        end
        if rom_predictor !== nothing
            predict_lin_true = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
                rom_full_predict,
                state,
                shock_t,
                θ_local,
                d_obs,
            )
            ll_lin_cond_true = MacroModelling.conditional_loglik_per_period(
                predict_lin_true,
                s0,
                shocks,
                theta_true,
                obs_data,
                obs_sigma,
            )
            ll_lin_true_conditional = sum(ll_lin_cond_true)
            ll_sep_true_cond = MacroModelling.additive_residual_loglik_per_period(
                rom_full_predict,
                surrogate_residual_predict,
                s0,
                shocks,
                theta_true,
                obs_data,
                obs_sigma;
                d_obs = d_obs,
                allow_full_residual = true,
            )
            ll_rs_true_hard = sum(ll_sep_true_cond[hard_gate_mask]) + sum(ll_lin_cond_true[.!hard_gate_mask])
        end
        if true_theta_filter != :none && rom_predictor !== nothing
            if isempty(observables)
                println("Warning: observables missing; skipping true-theta filtered shocks.")
            else
                shocks_true_filtered = filtered_shocks_from_theta(obs_data, observables, shock_sigmas, theta_true, theta_names;
                                                                  filter = true_theta_filter)
                predict_lin_true_f = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
                    rom_full_predict,
                    state,
                    shock_t,
                    θ_local,
                    d_obs,
                )
                ll_lin_cond_true_f = MacroModelling.conditional_loglik_per_period(
                    predict_lin_true_f,
                    s0,
                    shocks_true_filtered,
                    theta_true,
                    obs_data,
                    obs_sigma,
                )
                ll_lin_true_conditional_filtered = sum(ll_lin_cond_true_f)
                ll_lin_true_conditional_filtered_source = String(true_theta_filter)
                ll_sep_true_cond_f = MacroModelling.additive_residual_loglik_per_period(
                    rom_full_predict,
                    surrogate_residual_predict,
                    s0,
                    shocks_true_filtered,
                    theta_true,
                    obs_data,
                    obs_sigma;
                    d_obs = d_obs,
                    allow_full_residual = true,
                )
                ll_rs_true_hard_filtered = sum(ll_sep_true_cond_f[hard_gate_mask]) + sum(ll_lin_cond_true_f[.!hard_gate_mask])
                ll_rs_true_hard_filtered_source = String(true_theta_filter)
            end
        end
    end
    println("Loglik at true theta: linear=$(round(ll_lin_true, digits=2)) regime=$(round(ll_rs_true, digits=2))")
    if ll_lin_true_conditional !== nothing
        println("Loglik at true theta (linear conditional): $(round(ll_lin_true_conditional, digits=2))")
    end
    if ll_rs_true_hard !== nothing
        println("Loglik at true theta (hard gate, conditional; source=$hard_gate_source): $(round(ll_rs_true_hard, digits=2))")
    end
    if ll_lin_true_conditional_filtered !== nothing
        println("Loglik at true theta (linear conditional, filtered shocks): $(round(ll_lin_true_conditional_filtered, digits=2))")
    end
    if ll_rs_true_hard_filtered !== nothing
        println("Loglik at true theta (hard gate, filtered shocks; source=$hard_gate_source): $(round(ll_rs_true_hard_filtered, digits=2))")
    end
end

# Dynamic init params for N parameters
_init_theta_vec = copy(_theta_init)
if n_theta == 3 && theta_names == [:cprobp, :cindp, :curvp]
    # Legacy: use CLI overrides if provided
    if isfinite(init_cprobp_arg); _init_theta_vec[1] = init_cprobp_arg; end
    if isfinite(init_cindp_arg); _init_theta_vec[2] = init_cindp_arg; end
    if isfinite(init_curvp_arg); _init_theta_vec[3] = init_curvp_arg; end
end
# Use theta_true as init if available (synthetic data case)
if theta_true !== nothing && length(theta_true) == n_theta
    _init_theta_vec .= theta_true
end
# Clamp init values to strict interior of bounds (NUTS needs interior start)
for i in 1:n_theta
    lo, hi = _prior_bounds[i]
    pad = 0.02 * (hi - lo)
    _init_theta_vec[i] = clamp(_init_theta_vec[i], lo + pad, hi - pad)
end
base_init_params = (theta_vec = _init_theta_vec,)
init_params = base_init_params
if n_chains > 1
    if init_jitter > 0
        rng = Random.MersenneTwister(1234)
        init_params = [
            (theta_vec = [clamp(_init_theta_vec[i] + _mh_rw_scales[i] * 10 * randn(rng),
                                _prior_bounds[i][1], _prior_bounds[i][2]) for i in 1:n_theta],)
            for _ in 1:n_chains
        ]
    else
        init_params = [base_init_params for _ in 1:n_chains]
    end
end

turing_model = hlt_surrogate_model(obs_data,
                                    frozen,
                                    s0,
                                    shocks,
                                    obs_sigma,
                                    shock_sigmas,
                                    shock_guided,
                                    sample_shocks,
                                    shock_sample_idx,
                                    obs_data_ka,
                                    theta_names,
                                    gate_mask,
                                    gate_probs,
                                    gate_mode,
                                    use_linear_loglik,
                                    shock_filter,
                                    linear_filter,
                                    inversion_maxit,
                                    inversion_tol,
                                    inversion_lambda)
println("Initial params: $(init_params)")

if sampler_mode == :mh && mcmc_threads && n_chains > 1
    println("Warning: disabling MCMCThreads for sampler=:mh due DynamicPPL thread-safety limitations.")
end
use_threads = sampler_mode != :mh && mcmc_threads && n_chains > 1 && Threads.nthreads() >= n_chains
samps = nothing
if n_samples <= 0 && linear_samples > 0
    println("Skipping regime-switching sampling (--samples=$n_samples); proceeding to linear-only estimation.")
elseif eval_only || chain_in != ""
    payload_in = MacroModelling.load_hlt_chain_payload(chain_in)
    samps = payload_in["chain"]
    println("Loaded chain: $chain_in")
else
    start_time = time()
    if profile && chunk_size > 0
        error("--chunk-size is not supported with --profile.")
    end
    sampler = build_sampler(sampler_mode;
                            nuts_adapt = nuts_adapt,
                            nuts_target_accept = nuts_target_accept,
                            nuts_max_depth = nuts_max_depth,
                            nuts_init_eps = nuts_init_eps,
                            hmc_step_size = hmc_step_size,
                            hmc_leapfrog = hmc_leapfrog,
                            mh_rw_cprobp = mh_rw_cprobp,
                            mh_rw_cindp = mh_rw_cindp,
                            mh_rw_curvp = mh_rw_curvp)
    if chunk_size > 0
        samps = sample_in_chunks(turing_model, sampler, n_samples;
                                 chunk_size = chunk_size,
                                 n_chains = n_chains,
                                 use_threads = use_threads,
                                 init_params = init_params,
                                 checkpoint_path = checkpoint_path)
    elseif profile
        Profile.clear()
        Profile.@profile begin
            if use_threads
                samps = Turing.sample(turing_model,
                                      sampler,
                                      Turing.MCMCThreads(),
                                      n_samples,
                                      n_chains;
                                      progress = true,
                                      initial_params = init_params)
            elseif n_chains > 1
                samps = Turing.sample(turing_model,
                                      sampler,
                                      Turing.MCMCSerial(),
                                      n_samples,
                                      n_chains;
                                      progress = true,
                                      initial_params = init_params)
            else
                samps = Turing.sample(turing_model, sampler, n_samples;
                                      progress = true,
                                      initial_params = init_params)
            end
        end
        open(profile_out, "w") do io
            Profile.print(io)
        end
        println("Saved profile: $profile_out")
    else
        if use_threads
            samps = Turing.sample(turing_model,
                                  sampler,
                                  Turing.MCMCThreads(),
                                  n_samples,
                                  n_chains;
                                  progress = true,
                                  initial_params = init_params)
        elseif n_chains > 1
            samps = Turing.sample(turing_model,
                                  sampler,
                                  Turing.MCMCSerial(),
                                  n_samples,
                                  n_chains;
                                  progress = true,
                                  initial_params = init_params)
        else
            samps = Turing.sample(turing_model, sampler, n_samples;
                                  progress = true,
                                  initial_params = init_params)
        end
    end
    elapsed = time() - start_time
    println("Elapsed (s): $(round(elapsed, digits = 2))")
end

theta_mean = nothing
theta_cov = nothing
if samps !== nothing
    println(samps)
    println("Posterior mean (regime): $(mean(samps).nt.mean)")

    theta_syms = n_theta == 3 && theta_names == [:cprobp, :cindp, :curvp] ? [:cprobp, :cindp, :curvp] : [Symbol("theta_vec[$i]") for i in 1:n_theta]
    theta_mat = MacroModelling.theta_draws(samps, theta_syms)
    theta_mean = vec(mean(theta_mat, dims = 1))
    theta_cov = Statistics.cov(theta_mat)
    shocks_eval = shocks
    shocks_source = "fixed"
    logprior_eps = 0.0
    if sample_shocks
        eps_mean = MacroModelling.epsilon_means_from_chain(samps; sample_idx = shock_sample_idx)
        if eps_mean !== nothing
            shocks_base = shock_guided === nothing ? shocks : shock_guided
            shocks_eval = MacroModelling.build_shocks_from_eps(eps_mean, shock_sigmas, shock_guided;
                                                               sample_idx = shock_sample_idx,
                                                               shocks_base = shocks_base,
                                                               T_full = size(shocks, 2))
            shocks_source = "posterior_eps_mean"
            logprior_eps = sum(Distributions.logpdf.(Distributions.Normal(0, 1), eps_mean))
        else
            println("Warning: could not build posterior-mean shocks; using fixed shock path for loglik.")
        end
    end
    if shock_filter == :inversion
        shocks_source = "inversion_map"
    end
    loglik_post_mean = loglik_at_theta(theta_mean, shocks_eval)
    logprior_theta_val = logprior_theta(theta_mean)
    log_marginal_theta = theta_laplace_log_marginal(theta_mean, theta_cov, loglik_post_mean, logprior_theta_val)
    println("Loglik at posterior mean (regime): $(round(loglik_post_mean, digits = 2))")
    println("Laplace log marginal (theta-only, regime): $(round(log_marginal_theta, digits = 2))")

    loglik_post_mean_filtered = nothing
    log_marginal_theta_filtered = nothing
    loglik_filtered_source = nothing
    if post_mean_filter != :none
        if shock_filter == :inversion
            println("Post-mean filtered shocks skipped because --shock-filter=inversion is active.")
        elseif isempty(observables)
            println("Warning: observables missing; skipping post-mean filtered shocks.")
        else
            shocks_filtered = filtered_shocks_from_theta(obs_data, observables, shock_sigmas, theta_mean, theta_names;
                                                         filter = post_mean_filter)
            loglik_post_mean_filtered = loglik_at_theta(theta_mean, shocks_filtered)
            log_marginal_theta_filtered = theta_laplace_log_marginal(theta_mean, theta_cov, loglik_post_mean_filtered, logprior_theta_val)
            loglik_filtered_source = String(post_mean_filter)
            println("Loglik at posterior mean ($(post_mean_filter) shocks): $(round(loglik_post_mean_filtered, digits = 2))")
            println("Laplace log marginal (theta-only, $(post_mean_filter) shocks): $(round(log_marginal_theta_filtered, digits = 2))")
        end
    end
else
    # No regime samples — initialize defaults for the serialize block below
    shocks_eval = shocks
    shocks_source = shock_filter == :inversion ? "inversion_map" : "fixed"
    logprior_eps = 0.0
    loglik_post_mean = nothing
    logprior_theta_val = nothing
    log_marginal_theta = nothing
    loglik_post_mean_filtered = nothing
    log_marginal_theta_filtered = nothing
    loglik_filtered_source = nothing
end  # samps !== nothing

linear_chain = nothing
if linear_samples > 0 && !(eval_only || chain_in != "")
    linear_gate_mask = falses(size(obs_data, 2))
    linear_model = hlt_surrogate_model(obs_data,
                                        frozen,
                                        s0,
                                        shocks,
                                        obs_sigma,
                                        shock_sigmas,
                                        shock_guided,
                                        linear_sample_shocks,
                                        collect(1:size(obs_data, 2)),
                                        obs_data_ka,
                                        theta_names,
                                        linear_gate_mask,
                                        nothing,
                                        :hard,
                                        true,
                                        shock_filter,
                                        linear_filter,
                                        inversion_maxit,
                                        inversion_tol,
                                        inversion_lambda)
    linear_start = time()
    linear_sampler = build_sampler(sampler_mode;
                                   nuts_adapt = nuts_adapt,
                                   nuts_target_accept = nuts_target_accept,
                                   nuts_max_depth = nuts_max_depth,
                                   nuts_init_eps = nuts_init_eps,
                                   hmc_step_size = hmc_step_size,
                                   hmc_leapfrog = hmc_leapfrog,
                                   mh_rw_cprobp = mh_rw_cprobp,
                                   mh_rw_cindp = mh_rw_cindp,
                                   mh_rw_curvp = mh_rw_curvp)
    if use_threads
        linear_chain = Turing.sample(linear_model,
                                     linear_sampler,
                                     Turing.MCMCThreads(),
                                     linear_samples,
                                     n_chains;
                                     progress = true,
                                     initial_params = init_params)
    elseif n_chains > 1
        linear_chain = Turing.sample(linear_model,
                                     linear_sampler,
                                     Turing.MCMCSerial(),
                                     linear_samples,
                                     n_chains;
                                     progress = true,
                                     initial_params = init_params)
    else
        linear_chain = Turing.sample(linear_model, linear_sampler, linear_samples;
                                     progress = true,
                                     initial_params = init_params)
    end
    linear_elapsed = time() - linear_start
    println("Elapsed (linear, s): $(round(linear_elapsed, digits = 2))")
    if linear_out_path == ""
        linear_out_path = joinpath(dirname(out_path), "hlt_sep_surrogate_estimation_chain_linear.jls")
    end
    serialize(linear_out_path, Dict(
        "chain" => linear_chain,
        "theta_true" => theta_true,
        "synthetic_path" => synthetic_path,
        "prior_config" => Dict(
            "cprobp_mean" => prior_cprobp_mu,
            "cprobp_sd" => prior_cprobp_sd,
            "cindp_mean" => prior_cindp_mu,
            "cindp_sd" => prior_cindp_sd,
            "curvp_mean" => prior_curvp_mu,
            "curvp_sd" => prior_curvp_sd,
        ),
        "init_params" => Dict(String(theta_names[i]) => _init_theta_vec[i] for i in 1:n_theta),
        "linear_elapsed_s" => linear_elapsed,
        "linear_filter" => String(linear_filter),
        "loglik_shocks_source" => (linear_filter == :inversion ? "inversion_map" : "kalman"),
        "loglik_post_mean" => begin
            theta_lin_mat = MacroModelling.theta_draws(linear_chain, theta_syms)
            theta_lin_mean = vec(mean(theta_lin_mat, dims = 1))
            sum(linear_gate_loglik_per_period(obs_data_ka, theta_lin_mean, theta_names, s0, shocks;
                                              shock_filter = :inversion,
                                              linear_filter = linear_filter))
        end,
        "log_marginal_theta_laplace" => begin
            theta_lin_mat = MacroModelling.theta_draws(linear_chain, theta_syms)
            theta_lin_mean = vec(mean(theta_lin_mat, dims = 1))
            theta_lin_cov = Statistics.cov(theta_lin_mat)
            lin_loglik = sum(linear_gate_loglik_per_period(obs_data_ka, theta_lin_mean, theta_names, s0, shocks;
                                                           shock_filter = :inversion,
                                                           linear_filter = linear_filter))
            lin_logprior = logprior_theta(theta_lin_mean)
            theta_laplace_log_marginal(theta_lin_mean, theta_lin_cov, lin_loglik, lin_logprior)
        end,
        "logprior_theta_post_mean" => begin
            theta_lin_mat = MacroModelling.theta_draws(linear_chain, theta_syms)
            theta_lin_mean = vec(mean(theta_lin_mat, dims = 1))
            logprior_theta(theta_lin_mean)
        end,
        "note" => "Linear-only estimation (gate_mask all false).",
    ))
    println("Posterior mean (linear): $(mean(linear_chain).nt.mean)")
    println("Saved linear chain: $linear_out_path")
elseif linear_samples > 0
    println("Skipping linear sampling because --eval-only or --chain-in was provided.")
end

serialize(out_path, Dict(
    "chain" => samps,
    "theta_true" => theta_true,
    "surrogate_path" => surrogate_path,
    "synthetic_path" => synthetic_path,
    "linear_loglik_true" => ll_lin_true,
    "linear_loglik_true_conditional" => ll_lin_true_conditional,
    "linear_loglik_true_conditional_filtered" => ll_lin_true_conditional_filtered,
    "linear_loglik_true_conditional_filtered_source" => ll_lin_true_conditional_filtered_source,
    "regime_loglik_true" => ll_rs_true,
    "regime_loglik_true_hard" => ll_rs_true_hard,
    "regime_loglik_true_hard_filtered" => ll_rs_true_hard_filtered,
    "regime_loglik_true_hard_filtered_source" => ll_rs_true_hard_filtered_source,
    "hard_gate_source" => hard_gate_source,
    "hard_gate_share" => hard_gate_share,
    "hard_gate_threshold" => hard_gate_threshold,
    "loglik_post_mean" => loglik_post_mean,
    "loglik_post_mean_filtered" => loglik_post_mean_filtered,
    "logprior_theta_post_mean" => logprior_theta_val,
    "logprior_eps_post_mean" => logprior_eps,
    "log_marginal_theta_laplace" => log_marginal_theta,
    "log_marginal_theta_laplace_filtered" => log_marginal_theta_filtered,
    "post_mean_theta" => theta_mean,
    "post_cov_theta" => theta_cov,
    "loglik_shocks_source" => shocks_source,
    "loglik_shocks_source_filtered" => loglik_filtered_source,
    "shock_filter" => String(shock_filter),
    "linear_filter" => String(linear_filter),
    "sampler_mode" => String(sampler_mode),
    "sampler_config" => Dict(
        "nuts_adapt" => nuts_adapt,
        "nuts_target_accept" => nuts_target_accept,
        "nuts_max_depth" => nuts_max_depth,
        "nuts_init_eps" => nuts_init_eps,
        "hmc_step_size" => hmc_step_size,
        "hmc_leapfrog" => hmc_leapfrog,
        "mh_rw_cprobp" => mh_rw_cprobp,
        "mh_rw_cindp" => mh_rw_cindp,
        "mh_rw_curvp" => mh_rw_curvp,
    ),
    "linear_chain_path" => linear_out_path == "" ? nothing : linear_out_path,
    "shock_guidance" => shock_guidance,
    "shock_guidance_scale" => shock_guidance_scale,
    "shock_guidance_theta" => shock_guidance_theta,
    "shock_guided" => shock_guided,
    "prior_config" => Dict(String(theta_names[i]) => string(_prior_dists[i]) for i in 1:n_theta),
    "init_params" => Dict(String(theta_names[i]) => _init_theta_vec[i] for i in 1:n_theta),
    "gate_info" => gate_info,
    "gate_share" => gate_share,
    "gate_stats" => gate_stats,
    "gate_mask" => gate_mask,
    "gate_probs" => gate_probs,
))

summary_out_path = replace(out_path, r"\.jls$" => "_summary.jls")
serialize(summary_out_path, Dict(
    "theta_true" => theta_true,
    "synthetic_path" => synthetic_path,
    "regime_loglik_true" => ll_rs_true,
    "regime_loglik_true_hard" => ll_rs_true_hard,
    "regime_loglik_true_hard_filtered" => ll_rs_true_hard_filtered,
    "regime_loglik_true_hard_filtered_source" => ll_rs_true_hard_filtered_source,
    "loglik_post_mean" => loglik_post_mean,
    "loglik_post_mean_filtered" => loglik_post_mean_filtered,
    "post_mean_theta" => theta_mean,
    "post_cov_theta" => theta_cov,
    "shock_filter" => String(shock_filter),
    "linear_filter" => String(linear_filter),
    "sampler_mode" => String(sampler_mode),
    "gate_share" => gate_share,
    "gate_mask" => gate_mask,
    "gate_info" => gate_info,
    "summary_kind" => "hlt_sep_surrogate_estimation_chain_summary",
    "summary_source_chain_path" => out_path,
))

println("Saved chain: $out_path")
println("Saved chain summary: $summary_out_path")
