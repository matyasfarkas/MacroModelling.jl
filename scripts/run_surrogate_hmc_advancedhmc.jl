#!/usr/bin/env julia
# ============================================================================
# SURROGATE HMC ESTIMATION — AdvancedHMC.jl (NUTS + Inversion Filter + NN)
# ============================================================================
#
# 18-parameter nonlinear estimation using:
#   1. Inversion filter — recovers shocks via ROM1 (linear) inversion
#   2. Neural network surrogate — corrects ROM1 observables & state transitions
#   3. AdvancedHMC.jl NUTS with finite-difference gradients
#   4. Logit parameter transforms (constrained ↔ unconstrained)
#
# Architecture (FIX AD-04 design):
#   predict_fn     = ROM1 (linear) — for robust shock recovery
#   eval_predict_fn = ROM1 + NN surrogate — for evaluation & state propagation
#
# Usage:
#   julia --project=. scripts/run_surrogate_hmc_advancedhmc.jl \
#       --surrogate=.local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls \
#       --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls \
#       --out=.local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_advhmc.jls \
#       --samples=500 --adapt=200 --seed=42
# ============================================================================

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, var, quantile
import Distributions
using Printf, Dates
using AdvancedHMC, LogDensityProblems
using MacroModelling
using AxisKeys

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_kv_string(args, key, default)
    for arg in args
        if startswith(arg, "$key=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end
parse_kv_int(args, key, default) = parse(Int, parse_kv_string(args, key, string(default)))
parse_kv_float(args, key, default) = parse(Float64, parse_kv_string(args, key, string(default)))

surrogate_path = parse_kv_string(ARGS, "--surrogate", "")
data_path      = parse_kv_string(ARGS, "--data", "")
out_path       = parse_kv_string(ARGS, "--out", "hlt_surrogate_hmc_advhmc.jls")
n_samples      = parse_kv_int(ARGS, "--samples", 500)
n_adapt        = parse_kv_int(ARGS, "--adapt", 200)
target_accept  = parse_kv_float(ARGS, "--target-accept", 0.65)
max_depth      = parse_kv_int(ARGS, "--max-depth", 8)
seed           = parse_kv_int(ARGS, "--seed", 42)
fd_eps         = parse_kv_float(ARGS, "--fd-eps", 1e-5)
verbose        = any(==("--verbose"), ARGS)

# Inversion filter settings
inv_maxit      = parse_kv_int(ARGS, "--inv-maxit", 10)
inv_tol        = parse_kv_float(ARGS, "--inv-tol", 1e-6)
inv_lambda     = parse_kv_float(ARGS, "--inv-lambda", 1e-4)

# Observation sigma settings
obs_sigma_scale = parse_kv_float(ARGS, "--obs-sigma-scale", 2.0)
obs_sigma_floor = parse_kv_float(ARGS, "--obs-sigma-floor", 0.1)

# Checkpoint interval (saves every N draws)
checkpoint_every = parse_kv_int(ARGS, "--checkpoint-every", 100)

# Init from a previous chain (e.g., Kalman posterior mean)
init_from_path = parse_kv_string(ARGS, "--init-from", "")

# Gate calibration: regime-switching (surrogate on gate periods, Kalman on rest)
gate_path = parse_kv_string(ARGS, "--gate-calibration", "")
gate_mode = Symbol(parse_kv_string(ARGS, "--gate-mode", "soft"))
gate_k_pre  = parse_kv_int(ARGS, "--gate-k-pre", 4)
gate_k_post = parse_kv_int(ARGS, "--gate-k-post", 8)
gate_min_len = parse_kv_int(ARGS, "--gate-min-len", 4)

if surrogate_path == "" || data_path == ""
    error("""Usage: julia run_surrogate_hmc_advancedhmc.jl \\
        --surrogate=<surrogate.jls> --data=<payload.jls> \\
        [--out=...] [--samples=500] [--adapt=200] [--seed=42] \\
        [--init-from=<chain.jls>] [--gate-calibration=<gate.jls>]""")
end

Random.seed!(seed)

println("=" ^ 72)
println("SURROGATE HMC — AdvancedHMC.jl (NUTS + Inversion Filter)")
println("Started: $(now())")
println("=" ^ 72)
println("  Surrogate:      $surrogate_path")
println("  Data:           $data_path")
println("  Output:         $out_path")
println("  Samples:        $n_samples")
println("  Adapt:          $n_adapt")
println("  Target accept:  $target_accept")
println("  Max tree depth: $max_depth")
println("  FD epsilon:     $fd_eps")
println("  Inv maxit:      $inv_maxit")
println("  Inv tol:        $inv_tol")
println("  Inv lambda:     $inv_lambda")
println("  Seed:           $seed")
println("  Init from:      $(init_from_path == "" ? "(blended calib/prior)" : init_from_path)")
println("  Gate calib:     $(gate_path == "" ? "(none — full surrogate)" : gate_path)")
println("  Gate mode:      $gate_mode")

# ============================================================================
# Step 1: Load Data Payload
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 1: Loading data payload")
println("-" ^ 72)

payload = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data       = payload["obs_data"]          # (d_obs, T)
obs_sigma_base = payload["obs_sigma"]         # (d_obs,)
theta_true     = get(payload, "theta_true", nothing)
theta_names    = payload["theta_names"]       # Vector{Symbol}
observables    = payload["observables"]       # Vector{Symbol}
state_names    = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
s0             = payload["s0"]                # initial state vector
shock_sigmas   = haskey(payload, "shock_sigmas") ? payload["shock_sigmas"] : error("Missing shock_sigmas in payload")

d_obs = size(obs_data, 1)
T_obs = size(obs_data, 2)
n_theta = length(theta_names)
d_state = length(s0)

println("  Observables:    $observables ($d_obs)")
println("  State vars:     $(length(state_names))")
println("  Periods:        $T_obs")
println("  Parameters:     $theta_names ($n_theta)")
println("  Shock dims:     $(length(shock_sigmas)) ($(count(shock_sigmas .> 0)) structural)")
if theta_true !== nothing
    println("  Theta true:     $(round.(theta_true, digits=4))")
else
    println("  Theta true:     (real data — no true values)")
end

# ============================================================================
# Step 2: Load Model & Surrogate
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 2: Loading HLT model + NN surrogate")
println("-" ^ 72)

repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

# Load non-OBC linear model for ROM1
mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  Model:          $(mm_model.model_name)")

# Load surrogate bundle
surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)

println("  Surrogate:      d_in=$(frozen.d_in), d_out=$(frozen.d_out)")
println("  Activation:     $(get_activation(frozen))")
if val_rmse !== nothing
    println("  Val RMSE:       $(round.(val_rmse[1:min(7,length(val_rmse))], sigdigits=3))")
end

# ============================================================================
# Step 3: Build Observation Sigma
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 3: Building observation sigma")
println("-" ^ 72)

obs_sigma = copy(obs_sigma_base)
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_rmse = val_rmse[1:d_obs] .* obs_sigma_scale
    obs_sigma = max.(obs_sigma, obs_rmse)
    println("  obs_sigma mode: max(synthetic, surrogate_rmse × $obs_sigma_scale)")
end
if obs_sigma_floor > 0
    obs_sigma = max.(obs_sigma, obs_sigma_floor)
    println("  obs_sigma floor: $obs_sigma_floor")
end
println("  obs_sigma:      $(round.(obs_sigma, sigdigits=3))")

# ============================================================================
# Step 4: Build ROM Predictor & Predict Functions
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 4: Building ROM1 predictor + surrogate predict functions")
println("-" ^ 72)

# Compute state/obs indices in model variable ordering
obs_idx = indexin(observables, mm_model.var)
state_idx = indexin(state_names, mm_model.var)
any(isnothing, obs_idx) && error("Observable names not found in $(mm_model.model_name): $observables")
any(isnothing, state_idx) && error("State names not found in $(mm_model.model_name): $state_names")

# Parameter index mapping
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx_any)
        missing_names = theta_names[isnothing.(idx_any)]
        error("Theta names not found in model parameters: $missing_names")
    end
    Int.(idx_any)
end

base_parameters = copy(mm_model.parameter_values)

# Build ROM predictor (baseline mode: fixed at calibrated parameters)
rom_predictor = RomPredictor(mm_model,
                             1,          # ROM order 1 (first-order perturbation)
                             :baseline,  # fixed baseline — no per-theta re-solve
                             false,      # no OBC for ROM1
                             Int[],      # theta_idx empty for baseline
                             base_parameters,
                             nothing,
                             nothing,
                             Int.(state_idx),
                             Int.(obs_idx))

# Initialize ROM cache
ensure_rom_cache!(rom_predictor, Float64[])
println("  ROM1 cache initialized (baseline mode)")
println("  State subset: $(length(state_idx)) vars")
println("  Obs subset:   $(length(obs_idx)) vars")

# Predict functions: same architecture as Turing script (FIX AD-04)
rom_full_predict = function(state::AbstractVector, shock_t::AbstractVector, θ_local::AbstractVector)
    return rom_predict(rom_predictor, state, shock_t, θ_local)
end

# Surrogate theta padding (if surrogate has different theta count than estimation)
surrogate_theta_names = get(sur_meta, "theta_names", Symbol[])
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    _sur_theta_baseline = zeros(Float64, length(surrogate_theta_names))
    _sur_theta_est_idx = zeros(Int, length(surrogate_theta_names))
    for (si, sname) in enumerate(surrogate_theta_names)
        ei = findfirst(==(sname), theta_names)
        if ei !== nothing
            _sur_theta_est_idx[si] = ei
        else
            pi = findfirst(==(sname), mm_model.parameters)
            _sur_theta_baseline[si] = pi !== nothing ? base_parameters[pi] : 0.0
        end
    end
    println("  Surrogate theta padding: $(length(surrogate_theta_names)) surrogate → $(length(theta_names)) estimation")
    function _pad_theta(θ_local::AbstractVector)
        θ_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                θ_full[i] = θ_local[_sur_theta_est_idx[i]]
            end
        end
        return θ_full
    end
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, _pad_theta(θ_local)))[1:d_obs]
else
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, θ_local))[1:d_obs]
end

# Combined predict: ROM1 + NN correction
surrogate_step_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
    rom_full_predict,
    surrogate_residual_predict,
    state,
    shock_t,
    θ_local,
    d_obs;
    allow_full_residual = false,
)

# ROM-only predict: for shock recovery in inversion filter
rom_only_predict = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
    rom_full_predict,
    state,
    shock_t,
    θ_local,
    d_obs,
)

# Batch NN residual evaluation (for efficient Phase 2 in inversion filter)
# The inversion filter constructs X_nn = [state; shock; theta] and calls this in batch.
# Returns full NN output (d_out × T); the filter uses only the first d_obs rows.
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    batch_nn_residual = function(X_nn::AbstractMatrix)
        d_prefix = d_state + length(shock_sigmas)
        T_batch = size(X_nn, 2)
        X_padded = Matrix{eltype(X_nn)}(undef, frozen.d_in, T_batch)
        X_padded[1:d_prefix, :] .= X_nn[1:d_prefix, :]
        for t in 1:T_batch
            X_padded[(d_prefix+1):end, t] .= _pad_theta(X_nn[(d_prefix+1):end, t])
        end
        return predict_frozen_batch(frozen, X_padded)
    end
else
    batch_nn_residual = (X_nn) -> predict_frozen_batch(frozen, X_nn)
end

# Single-sample NN residual for gate-conditional correction (full d_out output)
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    single_nn_residual = function(x_nn::AbstractVector)
        d_prefix = d_state + length(shock_sigmas)
        x_padded = vcat(x_nn[1:d_prefix], _pad_theta(x_nn[(d_prefix+1):end]))
        return predict_frozen(frozen, x_padded)
    end
else
    single_nn_residual = (x_nn) -> predict_frozen(frozen, x_nn)
end

# Correction clamp bounds: ±3σ of training RMSE per output dimension
if val_rmse !== nothing && length(val_rmse) == frozen.d_out
    nn_correction_clamp = 3.0 .* val_rmse
    println("  Correction clamp: ±3×RMSE (obs max=$(round(maximum(nn_correction_clamp[1:d_obs]), sigdigits=3)), state max=$(round(maximum(nn_correction_clamp[(d_obs+1):end]), sigdigits=3)))")
else
    nn_correction_clamp = nothing
    println("  Correction clamp: disabled (no val_rmse)")
end

println("  Predict functions built: rom_only + surrogate_step + batch_nn + single_nn")

# Quick test
println("\n  Testing predict functions...")
t0 = time()
test_shock = zeros(Float64, length(shock_sigmas))
test_theta = Float64[base_parameters[i] for i in theta_param_idx]
test_obs_rom, test_state_rom = rom_only_predict(Float64.(s0), test_shock, test_theta)
t_rom = time() - t0
println("    ROM1 predict: $(round(t_rom*1e6, digits=0))μs, obs=$(round.(test_obs_rom[1:min(3,d_obs)], sigdigits=4))")

t0 = time()
test_obs_sur, test_state_sur = surrogate_step_predict(Float64.(s0), test_shock, test_theta)
t_sur = time() - t0
println("    Surrogate predict: $(round(t_sur*1e6, digits=0))μs, obs=$(round.(test_obs_sur[1:min(3,d_obs)], sigdigits=4))")

t0 = time()
test_x_nn = vcat(Float64.(s0), test_shock, test_theta)
test_y_nn = single_nn_residual(test_x_nn)
t_snn = time() - t0
println("    Single NN residual: $(round(t_snn*1e6, digits=0))μs, d_out=$(length(test_y_nn)), first 3 obs=$(round.(test_y_nn[1:min(3,d_obs)], sigdigits=4))")

# ============================================================================
# Step 5: Priors and Transforms
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 5: Setting up priors and parameter transforms")
println("-" ^ 72)

include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
specs = get_phase1_18param_specs()
spec_names = [s.name for s in specs]

prior_dists  = Vector{Distributions.Distribution}(undef, n_theta)
prior_bounds = Vector{Tuple{Float64,Float64}}(undef, n_theta)

for (i, tname) in enumerate(theta_names)
    si = findfirst(==(tname), spec_names)
    si === nothing && error("No prior specification found for parameter: $tname")
    spec = specs[si]
    if spec.prior_type == :Beta
        prior_dists[i] = Distributions.Beta(spec.prior_params.α, spec.prior_params.β)
    elseif spec.prior_type == :InvGamma
        prior_dists[i] = Distributions.InverseGamma(spec.prior_params.α, spec.prior_params.θ)
    elseif spec.prior_type == :Normal
        prior_dists[i] = Distributions.Normal(spec.prior_params.μ, spec.prior_params.σ)
    else
        prior_dists[i] = Distributions.Uniform(spec.bounds...)
    end
    prior_bounds[i] = spec.bounds
end

# Logit transforms
function constrained_to_unconstrained(θ::Vector{Float64})
    x = similar(θ)
    for i in 1:length(θ)
        lb, ub = prior_bounds[i]
        val = clamp(θ[i], lb + 1e-10, ub - 1e-10)
        x[i] = log((val - lb) / (ub - val))
    end
    return x
end

function unconstrained_to_constrained(x::Vector{Float64})
    θ = similar(x)
    for i in 1:length(x)
        lb, ub = prior_bounds[i]
        θ[i] = lb + (ub - lb) / (1.0 + exp(-x[i]))
    end
    return θ
end

function log_jacobian(x::Vector{Float64})
    lj = 0.0
    for i in 1:length(x)
        lb, ub = prior_bounds[i]
        s = 1.0 / (1.0 + exp(-x[i]))
        lj += log(ub - lb) + log(s) + log(1.0 - s)
    end
    return lj
end

# ============================================================================
# Step 6: Gate Calibration + Log-Density Function
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 6: Building log-density function")
println("-" ^ 72)

# Build KeyedArray for Kalman filter
obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)

# --- Gate calibration: regime-switching ---
gate_probs = nothing  # soft gate probabilities (T_obs,)
gate_mask  = trues(T_obs)  # hard gate mask (true = use surrogate)

if gate_path != ""
    gate_calib = MacroModelling.load_hlt_gate_calibration(gate_path)
    tau_eps = gate_calib["tau_eps"]
    tau_y   = gate_calib["tau_y"]
    gate_use_eps = get(gate_calib, "use_eps", true)
    gate_use_y   = get(gate_calib, "use_y", true)

    # Use cached gate statistics (e_stats, f_stats) from calibration
    if haskey(gate_calib, "e_stats") && haskey(gate_calib, "f_stats") &&
       length(gate_calib["e_stats"]) == T_obs && length(gate_calib["f_stats"]) == T_obs
        e_stat = vec(Float64.(gate_calib["e_stats"]))
        f_stat = vec(Float64.(gate_calib["f_stats"]))
        println("  Using cached gate statistics from calibration payload")
    else
        error("Gate calibration missing e_stats/f_stats or dimension mismatch. " *
              "Re-run gate calibration with --gate-use-cached-stats.")
    end

    # Compute gate mask: periods where shocks or forecast errors exceed thresholds
    eps_mask = gate_use_eps ? (e_stat .> tau_eps) : falses(T_obs)
    y_mask   = gate_use_y   ? (f_stat .> tau_y)   : falses(T_obs)
    base_mask = eps_mask .| y_mask

    # Apply padding (pre/post) and minimum run length for hard gate mask
    gate_mask = MacroModelling.apply_gate_padding(base_mask, gate_k_pre, gate_k_post, gate_min_len)

    n_base = count(base_mask)
    n_gate = count(gate_mask)
    println("  Gate: tau_eps=$(round(tau_eps, sigdigits=3)), tau_y=$(round(tau_y, sigdigits=3))")
    println("  Gate (hard): $n_base base → $n_gate/$T_obs padded periods")

    # Compute soft gate probabilities (matching Turing script logic)
    if gate_mode == :soft
        gate_prob_floor_val = 1e-4
        gate_prob_ceil_val  = 1.0 - 1e-4
        gate_beta_eps_val   = 1.0
        gate_beta_y_val     = 1.0

        target_share = mean(base_mask)
        prior_probs = fill(target_share, T_obs)
        prior_probs = clamp.(prior_probs, gate_prob_floor_val, gate_prob_ceil_val)
        prior_logit = MacroModelling.logit.(prior_probs)

        eps_scale = max(tau_eps, eps(Float64))
        y_scale   = max(tau_y, eps(Float64))
        eps_score = gate_use_eps ? gate_beta_eps_val .* ((e_stat .- tau_eps) ./ eps_scale) : zeros(T_obs)
        y_score   = gate_use_y   ? gate_beta_y_val   .* ((f_stat .- tau_y)   ./ y_scale)   : zeros(T_obs)
        score = eps_score .+ y_score

        gate_bias = MacroModelling.calibrate_gate_bias(score .+ prior_logit, target_share)
        gate_probs = MacroModelling.logistic.(gate_bias .+ score .+ prior_logit)
        gate_probs = clamp.(gate_probs, gate_prob_floor_val, gate_prob_ceil_val)

        gate_share = mean(gate_probs)
        println("  Gate (soft): mean_prob=$(round(gate_share, sigdigits=3)), range=[$(round(minimum(gate_probs), sigdigits=3)), $(round(maximum(gate_probs), sigdigits=3))]")
    else
        gate_share = mean(gate_mask)
        println("  Gate (hard): $(round(100*gate_share, digits=1))% periods use surrogate")
    end
else
    println("  No gate calibration — applying surrogate to ALL periods")
end

use_regime_switching = gate_path != "" && !all(gate_mask)

# Kalman filter log-likelihood (for non-gate periods)
function kalman_loglik_per_period(θ_constrained::Vector{Float64})
    return MacroModelling.linear_model_loglik_per_period(
        mm_model,
        obs_data_ka,
        θ_constrained,
        theta_names;
        model_parameter_names = mm_model.parameters,
        base_parameters       = base_parameters,
        theta_idx             = theta_param_idx,
        algorithm             = :first_order,
        filter                = :kalman,
        on_failure_loglikelihood = -1e12,
        presample_periods     = 0,
        initial_covariance    = :theoretical,
        verbose               = false,
        theta_label           = "Theta",
    )
end

# Inversion filter log-likelihood using surrogate (constrained θ)
# Architecture: ROM1 recovers shocks in Phase 1, batch NN corrects obs in Phase 2.
# State propagation always via ROM1 — prevents NN state drift over 184 periods.
function surrogate_inversion_loglik_per_period(θ_constrained::Vector{Float64})
    ll_vec, _ = MacroModelling.inversion_loglik_per_period(
        rom_only_predict,          # ROM1 for shock recovery + state propagation
        s0,
        θ_constrained,
        obs_data,
        obs_sigma,
        shock_sigmas;
        batch_eval_residual_fn = batch_nn_residual,  # NN obs correction in batch Phase 2
        single_eval_residual_fn = use_regime_switching ? single_nn_residual : nothing,
        gate_mask = use_regime_switching ? BitVector(gate_mask) : nothing,
        correction_clamp = use_regime_switching ? nn_correction_clamp : nothing,
        maxit  = inv_maxit,
        tol    = inv_tol,
        lambda = inv_lambda,
    )
    return ll_vec
end

# Combined log-likelihood: regime-switching or full surrogate
function total_loglik(θ_constrained::Vector{Float64})
    if use_regime_switching
        ll_surr = surrogate_inversion_loglik_per_period(θ_constrained)
        ll_kalman = kalman_loglik_per_period(θ_constrained)
        if gate_mode == :soft && gate_probs !== nothing
            # Soft mixing: p * surr_ll + (1-p) * kalman_ll (via logsumexp)
            # mix_loglikelihood returns a scalar (sum of per-period mixed LL)
            return MacroModelling.mix_loglikelihood(ll_surr, ll_kalman, gate_probs)
        else
            # Hard gate: surrogate on gate periods, Kalman on rest
            return sum(ll_surr[gate_mask]) + sum(ll_kalman[.!gate_mask])
        end
    else
        return sum(surrogate_inversion_loglik_per_period(θ_constrained))
    end
end

# Log prior (constrained space)
function log_prior(θ_constrained::Vector{Float64})
    lp = 0.0
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        if θ_constrained[i] < lb || θ_constrained[i] > ub
            return -Inf
        end
        d = Distributions.truncated(prior_dists[i], lb, ub)
        lp += Distributions.logpdf(d, θ_constrained[i])
    end
    return lp
end

# Full log-density in UNCONSTRAINED space
function log_density_unconstrained(x::Vector{Float64})
    θ = unconstrained_to_constrained(x)
    lp = log_prior(θ)
    isfinite(lp) || return -Inf
    ll = total_loglik(θ)
    isfinite(ll) || return -Inf
    lj = log_jacobian(x)
    return ll + lp + lj
end

# Finite-difference gradient
function fd_gradient!(∇f::Vector{Float64}, f::Function, x::Vector{Float64}, h::Float64)
    f0 = f(x)
    @inbounds for i in 1:length(x)
        x_old = x[i]
        x[i] = x_old + h
        fp = f(x)
        x[i] = x_old
        ∇f[i] = (fp - f0) / h
    end
    return f0
end

# Initialize: prefer --init-from chain, then calibrated values (clamped inward)
baseline_vals = get_phase1_18param_baseline()
θ_calib = Float64[get(baseline_vals, tname, NaN) for tname in theta_names]
any(isnan, θ_calib) && error("Missing baseline value for some parameters")

if init_from_path != ""
    println("  Loading initialization from: $init_from_path")
    init_chain = deserialize(init_from_path)
    if haskey(init_chain, "theta_post_mean")
        θ_init = Float64.(init_chain["theta_post_mean"])
    elseif haskey(init_chain, "chain") && init_chain["chain"] isa AbstractMatrix
        θ_init = Float64.(vec(mean(init_chain["chain"], dims=1)))
    else
        error("Cannot extract init from $init_from_path: no theta_post_mean or chain key")
    end
    # Clamp into bounds interior
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        θ_init[i] = clamp(θ_init[i], lb + 0.01*(ub-lb), ub - 0.01*(ub-lb))
    end
    println("  Init source: posterior mean from $(init_from_path)")
else
    # Use calibrated values, clamped inward (NOT the blended average that causes bad LL)
    θ_init = similar(θ_calib)
    for i in 1:n_theta
        lb, ub = prior_bounds[i]
        θ_init[i] = clamp(θ_calib[i], lb + 0.05*(ub-lb), ub - 0.05*(ub-lb))
    end
    println("  Init source: calibrated values (clamped to interior)")
end

println("  Initial values:")
for i in 1:n_theta
    @printf("    %-12s  calib=%.4f  init=%.4f  bounds=(%.2f, %.2f)\n",
            theta_names[i], θ_calib[i], θ_init[i], prior_bounds[i]...)
end

println("\n  Testing combined likelihood at init point...")
t0 = time()
ll_init = total_loglik(θ_init)
t_inv = time() - t0
println("  Total LL at init: $(round(ll_init, digits=2)) ($(round(t_inv*1000, digits=1)) ms)")

lp_init = log_prior(θ_init)
println("  Log prior at init: $(round(lp_init, digits=2))")

x_init = constrained_to_unconstrained(θ_init)
lj_init = log_jacobian(x_init)
println("  Log Jacobian at init: $(round(lj_init, digits=2))")

ld_init = log_density_unconstrained(x_init)
println("  Total log-density: $(round(ld_init, digits=2))")

# Test gradient
println("\n  Testing finite-difference gradient...")
∇f = zeros(n_theta)
t0 = time()
fd_gradient!(∇f, log_density_unconstrained, copy(x_init), fd_eps)
t_grad = time() - t0
println("  Gradient norm: $(round(norm(∇f), digits=4)) ($(round(t_grad*1000, digits=1)) ms)")
println("  Gradient:      $(round.(∇f, digits=3))")
all(isfinite, ∇f) || error("Gradient contains non-finite values!")

# Timing estimates
cost_per_eval_ms = t_inv * 1000
cost_per_grad_ms = t_grad * 1000
est_leapfrog_worst = 2^max_depth
est_time_per_draw_s = est_leapfrog_worst * cost_per_grad_ms / 1000
est_total_hours = (n_samples + n_adapt) * est_time_per_draw_s / 3600

println("\n  Timing estimates:")
println("  Inversion eval:     $(round(cost_per_eval_ms, digits=1)) ms")
println("  Gradient (FD, $n_theta evals): $(round(cost_per_grad_ms, digits=1)) ms")
println("  Est. per NUTS draw: $(round(est_time_per_draw_s, digits=1)) s (worst case, depth=$max_depth)")
println("  Est. total:         $(round(est_total_hours, digits=1)) hours (worst case)")

# ============================================================================
# Step 7: AdvancedHMC Setup
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 7: Setting up AdvancedHMC NUTS sampler")
println("-" ^ 72)

struct SurrogateLogDensity
    dim::Int
    fd_eps::Float64
end

LogDensityProblems.logdensity(p::SurrogateLogDensity, x::AbstractVector) =
    log_density_unconstrained(Vector{Float64}(x))

LogDensityProblems.dimension(p::SurrogateLogDensity) = p.dim

LogDensityProblems.capabilities(::Type{SurrogateLogDensity}) =
    LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::SurrogateLogDensity, x::AbstractVector)
    xv = Vector{Float64}(x)
    ∇f = zeros(p.dim)
    lp = fd_gradient!(∇f, log_density_unconstrained, xv, p.fd_eps)
    return lp, ∇f
end

log_density_obj = SurrogateLogDensity(n_theta, fd_eps)

metric = UnitEuclideanMetric(n_theta)
println("  Mass matrix: unit Euclidean (step-size-only adaptation)")

hamiltonian = Hamiltonian(metric, log_density_obj)

println("\n  Finding initial step size...")
initial_ϵ = find_good_stepsize(hamiltonian, x_init)
println("  Initial step size: $(round(initial_ϵ, sigdigits=3))")

integrator = Leapfrog(initial_ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(max_depth=max_depth)))
adaptor = StepSizeAdaptor(target_accept, integrator)

# ============================================================================
# Step 8: Run NUTS with Checkpointing
# ============================================================================

println("\n" * "=" ^ 72)
println("RUNNING NUTS SAMPLER ($n_samples draws + $n_adapt warmup)")
println("=" ^ 72)

# Use AdvancedHMC's high-level sample() — handles PhasePoint/adaptation internally
n_total = n_samples + n_adapt

t_start = time()
all_samples, stats = sample(hamiltonian, kernel, x_init, n_total, adaptor, n_adapt;
                            progress=true, verbose=verbose)
t_elapsed = time() - t_start

println("\n  Sampling complete! Elapsed: $(round(t_elapsed/60, digits=1)) minutes")
println("  Draws/sec: $(round(n_total / t_elapsed, digits=2))")

# ============================================================================
# Step 9: Post-processing
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 9: Post-processing")
println("-" ^ 72)

# Transform back to constrained space
θ_chain = hcat([unconstrained_to_constrained(Vector{Float64}(s)) for s in all_samples]...)'
θ_post = θ_chain[(n_adapt+1):end, :]

# Diagnostics
acceptance_rates = [s.acceptance_rate for s in stats[(n_adapt+1):end]]
tree_depths = [s.tree_depth for s in stats[(n_adapt+1):end]]
n_divergent = sum(s.numerical_error for s in stats[(n_adapt+1):end])

println("\n  NUTS Diagnostics:")
println("  Mean acceptance rate: $(round(mean(acceptance_rates), digits=3))")
println("  Mean tree depth:     $(round(mean(tree_depths), digits=1))")
println("  Max tree depth:      $(maximum(tree_depths))")
println("  Divergences:         $n_divergent / $n_samples")

# Compute log-likelihood at posterior mean
θ_post_mean = vec(mean(θ_post, dims=1))
ll_post_mean = total_loglik(θ_post_mean)
lp_post_mean = log_prior(θ_post_mean)

println("\n  Posterior mean log-likelihood: $(round(ll_post_mean, digits=2))")
println("  Posterior mean log-prior:     $(round(lp_post_mean, digits=2))")

# Parameter summary
has_true = theta_true !== nothing && length(theta_true) == n_theta
if has_true
    println("\n  " * "-" ^ 68)
    @printf("  %-12s %8s %8s %8s %8s %8s %8s\n",
            "Parameter", "True", "Mean", "Std", "Q2.5", "Q97.5", "Cover")
    println("  " * "-" ^ 68)
else
    println("\n  " * "-" ^ 60)
    @printf("  %-12s %8s %8s %8s %8s %8s\n",
            "Parameter", "Calib", "Mean", "Std", "Q2.5", "Q97.5")
    println("  " * "-" ^ 60)
end

coverage_count = 0
for i in 1:n_theta
    post_mean = mean(θ_post[:, i])
    post_std = std(θ_post[:, i])
    q025 = quantile(θ_post[:, i], 0.025)
    q975 = quantile(θ_post[:, i], 0.975)

    if has_true
        true_val = theta_true[i]
        covered = q025 < true_val < q975
        coverage_count += covered
        @printf("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f %5s\n",
                theta_names[i], true_val, post_mean, post_std, q025, q975,
                covered ? "yes" : "NO")
    else
        @printf("  %-12s %8.4f %8.4f %8.4f %8.4f %8.4f\n",
                theta_names[i], θ_calib[i], post_mean, post_std, q025, q975)
    end
end
if has_true
    println("  " * "-" ^ 68)
    println("  Coverage: $coverage_count / $n_theta ($(round(100*coverage_count/n_theta, digits=0))%)")
else
    println("  " * "-" ^ 60)
end

# ESS estimate
function ess_batch_means(chain::Vector{Float64}; batch_size::Int=50)
    n = length(chain)
    n < 2 * batch_size && return Float64(n)
    n_batches = n ÷ batch_size
    batch_means = [mean(chain[(i-1)*batch_size+1 : i*batch_size]) for i in 1:n_batches]
    var_total = var(chain)
    var_batch = var(batch_means)
    var_batch < 1e-20 && return Float64(n)
    return n * var_total / (batch_size * var_batch)
end

println("\n  Effective Sample Size (ESS):")
for i in 1:n_theta
    ess = ess_batch_means(θ_post[:, i])
    @printf("    %-12s  ESS = %.0f  (%.1f%%)\n", theta_names[i], ess, 100*ess/n_samples)
end

# ============================================================================
# Step 10: Save Results
# ============================================================================

println("\n" * "-" ^ 72)
println("STEP 10: Saving results")
println("-" ^ 72)

results = Dict{String,Any}(
    "chain"              => θ_post,
    "chain_full"         => θ_chain,
    "samples_unc"        => all_samples,
    "stats"              => stats,
    "theta_names"        => theta_names,
    "theta_true"         => theta_true,
    "theta_post_mean"    => θ_post_mean,
    "theta_init"         => θ_init,
    "prior_bounds"       => prior_bounds,
    "n_samples"          => n_samples,
    "n_adapt"            => n_adapt,
    "target_accept"      => target_accept,
    "max_depth"          => max_depth,
    "seed"               => seed,
    "fd_eps"             => fd_eps,
    "elapsed_seconds"    => t_elapsed,
    "ll_post_mean"       => ll_post_mean,
    "lp_post_mean"       => lp_post_mean,
    "n_divergent"        => n_divergent,
    "acceptance_rates"   => acceptance_rates,
    "tree_depths"        => tree_depths,
    "observables"        => observables,
    "state_names"        => state_names,
    "T_obs"              => T_obs,
    "model_name"         => "Smets_Wouters_2007_HLT",
    "surrogate_path"     => surrogate_path,
    "sampler"            => "NUTS (AdvancedHMC.jl)",
    "likelihood"         => "inversion_filter + surrogate",
    "gradient"           => "finite_differences",
    "inv_maxit"          => inv_maxit,
    "inv_tol"            => inv_tol,
    "inv_lambda"         => inv_lambda,
    "obs_sigma"          => obs_sigma,
    "obs_sigma_scale"    => obs_sigma_scale,
    "obs_sigma_floor"    => obs_sigma_floor,
    "timestamp"          => string(now()),
)

mkpath(dirname(out_path))
serialize(out_path, results)
println("  Saved: $out_path")

println("\n" * "=" ^ 72)
println("SURROGATE HMC ESTIMATION COMPLETE")
println("Finished: $(now())")
println("Elapsed:  $(round(t_elapsed/60, digits=1)) minutes")
println("=" ^ 72)
