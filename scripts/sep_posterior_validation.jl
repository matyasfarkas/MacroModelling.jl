#!/usr/bin/env julia
# ============================================================================
# SEP POSTERIOR VALIDATION — Direct FOM vs Surrogate Prediction Comparison
# ============================================================================
#
# Validates the surrogate (ROM1 + NN) against the full-order model (SEP) at
# posterior draws from a surrogate HMC chain.
#
# Architecture:
#   1. Recover shocks once using ROM1 inversion filter (θ-independent because
#      the ROM1 solution matrix is cached at baseline parameters).
#   2. Pre-compute the ROM1 state trajectory and obs predictions (also θ-indep).
#   3. At each posterior draw θ:
#      - Apply NN correction: surrogate_obs = ROM1_obs + NN(ROM1_state, ε, θ)
#      - Run SEP simulation with recovered shocks and θ → SEP obs
#      - Compute Gaussian LL under ROM1, surrogate, and SEP obs predictions
#      - Report per-variable RMSE and LL gaps
#
# Usage:
#   julia --project=. scripts/sep_posterior_validation.jl \
#       --chain=.local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls \
#       --surrogate=.local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls \
#       --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls \
#       --n-draws=20 --sep-horizon=40 --verbose
# ============================================================================

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, var, cor, quantile
using Printf, Dates
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

chain_path     = parse_kv_string(ARGS, "--chain", "")
surrogate_path = parse_kv_string(ARGS, "--surrogate", "")
data_path      = parse_kv_string(ARGS, "--data", "")
out_path       = parse_kv_string(ARGS, "--out", "sep_posterior_validation_results.jls")
n_draws        = parse_kv_int(ARGS, "--n-draws", 20)
sep_horizon    = parse_kv_int(ARGS, "--sep-horizon", 40)
sep_maxit      = parse_kv_int(ARGS, "--sep-maxit", 200)
sep_nnodes     = parse_kv_int(ARGS, "--sep-nnodes", 3)
sep_accept_tol = parse_kv_float(ARGS, "--sep-accept-tol", 0.35)
verbose        = any(==("--verbose"), ARGS)

# Inversion filter settings
inv_maxit  = parse_kv_int(ARGS, "--inv-maxit", 10)
inv_tol    = parse_kv_float(ARGS, "--inv-tol", 1e-6)
inv_lambda = parse_kv_float(ARGS, "--inv-lambda", 1e-4)

# Observation sigma settings
obs_sigma_scale = parse_kv_float(ARGS, "--obs-sigma-scale", 2.0)
obs_sigma_floor = parse_kv_float(ARGS, "--obs-sigma-floor", 0.1)

isempty(chain_path) && error("Must provide --chain=<path>")
isempty(surrogate_path) && error("Must provide --surrogate=<path>")
isempty(data_path) && error("Must provide --data=<path>")

println("=" ^ 72)
println("SEP POSTERIOR VALIDATION")
println("=" ^ 72)
println("  Chain:          $chain_path")
println("  Surrogate:      $surrogate_path")
println("  Data:           $data_path")
println("  N draws:        $n_draws")
println("  SEP horizon:    $sep_horizon")
println("  SEP maxit:      $sep_maxit")
println("  SEP accept tol: $sep_accept_tol")

# ============================================================================
# Step 1: Load Data Payload
# ============================================================================

println("\n--- Loading data payload ---")
payload = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data       = payload["obs_data"]
obs_sigma_base = payload["obs_sigma"]
theta_names    = payload["theta_names"]
observables    = payload["observables"]
state_names    = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
s0             = payload["s0"]
shock_sigmas   = payload["shock_sigmas"]

d_obs  = size(obs_data, 1)
T_obs  = size(obs_data, 2)
n_theta = length(theta_names)
d_state = length(s0)
d_exo   = length(shock_sigmas)

println("  Obs: $d_obs × $T_obs, State: $d_state, Shocks: $d_exo, Params: $n_theta")

# ============================================================================
# Step 2: Load Chain & Subsample
# ============================================================================

println("\n--- Loading chain ---")
chain_data = deserialize(chain_path)
chain = chain_data["chain"]  # (n_samples, n_theta)
n_total = size(chain, 1)

# Subsample: take every k-th draw
k = max(1, n_total ÷ n_draws)
draw_indices = 1:k:n_total
if length(draw_indices) > n_draws
    draw_indices = draw_indices[1:n_draws]
end
n_eval = length(draw_indices)
println("  Total chain: $n_total draws, subsampling every $k-th → $n_eval draws")

# ============================================================================
# Step 3: Load Model & Surrogate
# ============================================================================

println("\n--- Loading model + surrogate ---")
repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  Model: $(mm_model.model_name)")

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)
println("  Surrogate: d_in=$(frozen.d_in), d_out=$(frozen.d_out)")
if val_rmse !== nothing
    println("  val_rmse[1:min(7,end)]: $(round.(val_rmse[1:min(7,length(val_rmse))], sigdigits=3))")
end

# ============================================================================
# Step 4: Build Predict Functions
# ============================================================================

println("\n--- Building predict functions ---")

obs_idx   = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names, mm_model.var))

theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    any(isnothing, idx_any) && error("Theta names not found in model")
    Int.(idx_any)
end

base_parameters = copy(mm_model.parameter_values)

# Build observation sigma (matching chain: max(obs_sigma_base, val_rmse * scale, floor))
obs_sigma = copy(obs_sigma_base)
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_sigma = max.(obs_sigma, val_rmse[1:d_obs] .* obs_sigma_scale)
end
if obs_sigma_floor > 0
    obs_sigma = max.(obs_sigma, obs_sigma_floor)
end
println("  obs_sigma = $(round.(obs_sigma, sigdigits=3))")

# ROM predictor (dual-stripping, for NN input construction)
rom_predictor = RomPredictor(mm_model, 1, :baseline, false, Int[],
    base_parameters, nothing, nothing, Int.(state_idx), Int.(obs_idx))
ensure_rom_cache!(rom_predictor, Float64[])

rom_full_predict = (state, shock_t, θ_local) -> rom_predict(rom_predictor, state, shock_t, θ_local)

# ForwardDiff-compatible ROM predict for inversion filter Jacobian
_, matrix_rom_predict_tuple, _ = build_matrix_rom_predict(mm_model;
    state_idx=Int.(state_idx), obs_idx=Int.(obs_idx))

# Surrogate theta padding (if surrogate has different theta count)
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
    function _pad_theta(θ_local)
        θ_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                θ_full[i] = θ_local[_sur_theta_est_idx[i]]
            end
        end
        return θ_full
    end
    nn_residual_predict = (state, shock_t, θ_local) ->
        predict_frozen(frozen, vcat(state, shock_t, _pad_theta(θ_local)))[1:d_obs]
else
    nn_residual_predict = (state, shock_t, θ_local) ->
        predict_frozen(frozen, vcat(state, shock_t, θ_local))[1:d_obs]
end

println("  Predict functions ready")

# ============================================================================
# Step 5: Recover Shocks & Pre-compute ROM1 Trajectory (θ-independent)
# ============================================================================

println("\n--- Recovering shocks via ROM1 inversion filter ---")
println("  (Uses matrix_rom_predict which preserves ForwardDiff duals)")

# Use posterior mean θ for the inversion call (θ is ignored by ROM1 predict anyway)
θ_post_mean = Float64.(chain_data["theta_post_mean"])

ll_rom1_vec, shocks_recovered = MacroModelling.inversion_loglik_per_period(
    matrix_rom_predict_tuple, s0, θ_post_mean, obs_data, obs_sigma, shock_sigmas;
    maxit = inv_maxit, tol = inv_tol, lambda = inv_lambda)

ll_rom1_total = sum(ll_rom1_vec)
println("  ROM1 inversion LL (joint): $(round(ll_rom1_total, digits=1))")

# Pre-compute ROM1 state trajectory and obs (θ-independent)
rom1_obs = Matrix{Float64}(undef, d_obs, T_obs)
rom1_states = Matrix{Float64}(undef, d_state, T_obs)
let state_v = copy(Float64.(s0))
    for t in 1:T_obs
        rom1_states[:, t] = state_v
        obs_t, state_next = matrix_rom_predict_tuple(state_v, shocks_recovered[:, t], θ_post_mean)
        rom1_obs[:, t] = Float64.(obs_t)
        state_v = Float64.(state_next)
    end
end

# Shock statistics
structural_idx = findall(shock_sigmas .> 0)
shock_std = shock_sigmas[structural_idx]
for j in structural_idx
    vals = abs.(shocks_recovered[j, :])
    @printf("  Shock %d (σ=%.3f): mean|ε|=%.4f, max|ε|=%.4f, mean|ε/σ|=%.3f\n",
        j, shock_sigmas[j], mean(vals), maximum(vals), mean(vals ./ shock_sigmas[j]))
end

# ============================================================================
# Step 6: Compute LL constants
# ============================================================================

obs_log_norm_const   = sum(log.(2π .* obs_sigma .^ 2))
shock_log_norm_const = isempty(structural_idx) ? 0.0 : sum(log.(2π .* shock_std .^ 2))

function compute_obs_ll(obs_pred::Matrix{Float64})
    # Observation component of joint LL (same shocks → same shock penalty)
    ll = 0.0
    for t in 1:T_obs
        resid = obs_data[:, t] .- obs_pred[:, t]
        ll += -0.5 * (sum((resid ./ obs_sigma) .^ 2) + obs_log_norm_const)
    end
    return ll
end

function compute_shock_penalty()
    # Shock prior penalty (same for all methods since same shocks)
    ll = 0.0
    for t in 1:T_obs
        eps_struct = shocks_recovered[structural_idx, t]
        ll += -0.5 * (sum((eps_struct ./ shock_std) .^ 2) + shock_log_norm_const)
    end
    return ll
end

shock_penalty = compute_shock_penalty()
obs_ll_rom1 = compute_obs_ll(rom1_obs)
println("\n  ROM1 obs LL:      $(round(obs_ll_rom1, digits=1))")
println("  Shock penalty:    $(round(shock_penalty, digits=1))")
println("  ROM1 total (obs+shock): $(round(obs_ll_rom1 + shock_penalty, digits=1))")
println("  ROM1 total (inversion): $(round(ll_rom1_total, digits=1))")

# Kalman LL for reference
obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)
function kalman_ll(θ_constrained::Vector{Float64})
    ll_vec = MacroModelling.linear_model_loglik_per_period(
        mm_model, obs_data_ka, θ_constrained, theta_names;
        model_parameter_names = mm_model.parameters,
        base_parameters = base_parameters,
        theta_idx = theta_param_idx,
        algorithm = :first_order,
        filter = :kalman,
        on_failure_loglikelihood = -1e12)
    return sum(ll_vec)
end

ll_kalm_postmean = kalman_ll(θ_post_mean)
println("  Kalman LL (post mean): $(round(ll_kalm_postmean, digits=1))")

# ============================================================================
# Step 7: Evaluate at Posterior Draws
# ============================================================================

println("\n" * "=" ^ 72)
println("EVALUATING $n_eval POSTERIOR DRAWS")
println("=" ^ 72)

# Storage
ll_nn      = Vector{Float64}(undef, n_eval)   # ROM1+NN obs LL + shock penalty
ll_sep     = Vector{Float64}(undef, n_eval)   # SEP obs LL + shock penalty
ll_kalm    = Vector{Float64}(undef, n_eval)   # Kalman marginal LL
rmse_rom1_sep = Matrix{Float64}(undef, d_obs, n_eval)  # per-variable RMSE: ROM1 vs SEP
rmse_nn_sep   = Matrix{Float64}(undef, d_obs, n_eval)  # per-variable RMSE: surrogate vs SEP
sep_failed = falses(n_eval)

t_start = time()

for (i, di) in enumerate(draw_indices)
    θ_i = Float64.(vec(chain[di, :]))
    t0 = time()

    # --- ROM1+NN obs predictions (fast) ---
    nn_obs = copy(rom1_obs)
    for t in 1:T_obs
        nn_correction = nn_residual_predict(rom1_states[:, t], shocks_recovered[:, t], θ_i)
        nn_obs[:, t] .+= Float64.(nn_correction)
    end
    obs_ll_nn = compute_obs_ll(nn_obs)
    ll_nn[i] = obs_ll_nn + shock_penalty

    # --- Kalman LL (fast) ---
    ll_kalm[i] = kalman_ll(θ_i)

    # --- SEP simulation (slow) ---
    full_params = copy(base_parameters)
    for (j, idx) in enumerate(theta_param_idx)
        full_params[idx] = θ_i[j]
    end
    old_params = copy(mm_model.parameter_values)
    mm_model.parameter_values .= full_params

    local sep_obs
    try
        sep_result = simulate_sep_extended_path(mm_model;
            periods       = T_obs,
            initial_state = nothing,
            shocks        = shocks_recovered,
            burn_in       = 0,
            sep_horizon   = sep_horizon,
            sep_order     = 1,
            sep_nnodes    = sep_nnodes,
            sep_maxit     = sep_maxit,
            sep_tol       = 1e-7,
            sep_sparse_tree = true,
            sep_accept_tol  = sep_accept_tol,
            sep_shock_scale = 1.0,
            silent          = !verbose)

        mm_model.parameter_values .= old_params

        if sep_result.errorflag
            sep_failed[i] = true
            ll_sep[i] = NaN
            rmse_rom1_sep[:, i] .= NaN
            rmse_nn_sep[:, i] .= NaN
        else
            sep_sim = sep_result.simulation
            var_names = axiskeys(sep_sim, 1)
            sep_obs = Matrix{Float64}(undef, d_obs, T_obs)
            for (j, oname) in enumerate(observables)
                row = findfirst(==(oname), var_names)
                if row !== nothing
                    sep_obs[j, :] = sep_sim[row, 1:T_obs]
                end
            end

            obs_ll_sep = compute_obs_ll(sep_obs)
            ll_sep[i] = obs_ll_sep + shock_penalty

            # Per-variable RMSE
            for j in 1:d_obs
                rmse_rom1_sep[j, i] = sqrt(mean((rom1_obs[j, :] .- sep_obs[j, :]) .^ 2))
                rmse_nn_sep[j, i]   = sqrt(mean((nn_obs[j, :] .- sep_obs[j, :]) .^ 2))
            end
        end
    catch e
        mm_model.parameter_values .= old_params
        @warn "SEP failed at draw $i" exception=e
        sep_failed[i] = true
        ll_sep[i] = NaN
        rmse_rom1_sep[:, i] .= NaN
        rmse_nn_sep[:, i] .= NaN
    end

    dt = time() - t0
    elapsed = time() - t_start
    eta = elapsed / i * (n_eval - i)

    if verbose || i % 5 == 1 || i == n_eval
        sep_str = sep_failed[i] ? "FAILED" : @sprintf("%.1f", ll_sep[i])
        nn_gap = sep_failed[i] ? NaN : abs(ll_nn[i] - ll_sep[i])
        @printf("  [%3d/%3d] nn=%.1f  sep=%s  kalm=%.1f  |nn-sep|=%.1f  (%.1fs, ETA %.0fs)\n",
            i, n_eval, ll_nn[i], sep_str, ll_kalm[i],
            isnan(nn_gap) ? 0.0 : nn_gap, dt, eta)
    end
end

# ============================================================================
# Step 8: Summary Statistics
# ============================================================================

valid = .!sep_failed
n_valid = count(valid)
n_fail = count(sep_failed)

println("\n" * "=" ^ 72)
println("RESULTS")
println("=" ^ 72)
println("  Valid draws: $n_valid / $n_eval ($n_fail SEP failures)")

if n_valid > 0
    # LL gaps
    gap_nn_sep = abs.(ll_nn[valid] .- ll_sep[valid])
    gap_rom1_sep_ll = obs_ll_rom1 .- compute_obs_ll.(Ref(rom1_obs))  # always 0, for clarity

    println("\n  --- Log-likelihood comparison ---")
    println("  ROM1 joint LL (fixed): $(round(ll_rom1_total, digits=1))")
    @printf("  ROM1+NN LL:   mean=%.1f, std=%.1f\n", mean(ll_nn[valid]), std(ll_nn[valid]))
    @printf("  SEP LL:       mean=%.1f, std=%.1f\n", mean(ll_sep[valid]), std(ll_sep[valid]))
    @printf("  Kalman LL:    mean=%.1f, std=%.1f\n", mean(ll_kalm[valid]), std(ll_kalm[valid]))

    println("\n  --- |LL_surrogate - LL_SEP| (key validation metric) ---")
    @printf("    Mean:   %.1f nats\n", mean(gap_nn_sep))
    @printf("    Median: %.1f nats\n", quantile(gap_nn_sep, 0.5))
    @printf("    Max:    %.1f nats\n", maximum(gap_nn_sep))
    @printf("    Std:    %.1f nats\n", std(gap_nn_sep))
    @printf("    Per-period: %.2f nats/period\n", mean(gap_nn_sep) / T_obs)

    signed_nn_sep = ll_nn[valid] .- ll_sep[valid]
    @printf("    Signed mean: %.1f (positive = surrogate > SEP)\n", mean(signed_nn_sep))

    if n_valid >= 3
        r = cor(ll_nn[valid], ll_sep[valid])
        @printf("    Correlation(LL_surr, LL_SEP): %.4f\n", r)
    end

    # Per-variable RMSE
    println("\n  --- Per-variable RMSE at posterior draws ---")
    println("  Variable         ROM1-SEP    Surr-SEP    Capture%%    val_RMSE")
    for j in 1:d_obs
        r1 = mean(rmse_rom1_sep[j, valid])
        rs = mean(rmse_nn_sep[j, valid])
        capture = r1 > 0 ? (1.0 - rs / r1) * 100 : NaN
        vr = val_rmse !== nothing && j <= length(val_rmse) ? val_rmse[j] : NaN
        @printf("  %-16s  %8.4f    %8.4f    %6.1f%%    %.4f\n",
            string(observables[j]), r1, rs, capture, vr)
    end

    # Overall RMSE
    avg_r1 = mean(rmse_rom1_sep[:, valid])
    avg_rs = mean(rmse_nn_sep[:, valid])
    avg_cap = avg_r1 > 0 ? (1.0 - avg_rs / avg_r1) * 100 : NaN
    @printf("  %-16s  %8.4f    %8.4f    %6.1f%%\n", "AVERAGE", avg_r1, avg_rs, avg_cap)

    println("\n  Total wall time: $(round((time() - t_start) / 3600, digits=2)) hours")
end

# ============================================================================
# Step 9: Save Results
# ============================================================================

results = Dict{String,Any}(
    "ll_rom1_total"       => ll_rom1_total,
    "ll_nn"               => ll_nn,
    "ll_sep"              => ll_sep,
    "ll_kalman"           => ll_kalm,
    "shock_penalty"       => shock_penalty,
    "rmse_rom1_sep"       => rmse_rom1_sep,
    "rmse_nn_sep"         => rmse_nn_sep,
    "draw_indices"        => collect(draw_indices),
    "n_valid"             => n_valid,
    "n_failed"            => n_fail,
    "mean_abs_gap_nn_sep" => n_valid > 0 ? mean(abs.(ll_nn[valid] .- ll_sep[valid])) : NaN,
    "max_abs_gap_nn_sep"  => n_valid > 0 ? maximum(abs.(ll_nn[valid] .- ll_sep[valid])) : NaN,
    "correlation_nn_sep"  => n_valid >= 3 ? cor(ll_nn[valid], ll_sep[valid]) : NaN,
    "obs_sigma"           => obs_sigma,
    "shock_sigmas"        => shock_sigmas,
    "shocks_recovered"    => shocks_recovered,
    "chain_path"          => chain_path,
    "surrogate_path"      => surrogate_path,
    "data_path"           => data_path,
    "sep_horizon"         => sep_horizon,
    "sep_maxit"           => sep_maxit,
    "sep_accept_tol"      => sep_accept_tol,
    "timestamp"           => string(now()),
    "elapsed_seconds"     => time() - t_start,
)

serialize(out_path, results)
println("\nResults saved to: $out_path")
println("Done.")
