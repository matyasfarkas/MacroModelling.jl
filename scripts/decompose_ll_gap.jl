#!/usr/bin/env julia
# ============================================================================
# LL GAP DECOMPOSITION: PARAMETER RELOCATION vs MODEL CHANGE
# ============================================================================
#
# Computes a 2x2 table of log-likelihoods at the linear posterior mean
# (theta_linear) and the (pooled) surrogate posterior mean (theta_surrogate)
# under the linear (Kalman) and surrogate (regime-switching inversion + NN)
# likelihoods.
#
# Outputs:
#   .local_artifacts/ll_decomposition/ll_decomposition_table.tex
#   .local_artifacts/ll_decomposition/LL_DECOMPOSITION_SUMMARY.md
# ============================================================================

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, var, quantile
import Distributions
using Printf, Dates
using MacroModelling
using AxisKeys

# Tunables (match the HMC runners' production calls)
const SEED            = 42
const INV_MAXIT       = 10
const INV_TOL         = 1e-6
const INV_LAMBDA      = 1e-4
const OBS_SIGMA_SCALE = 2.0
const OBS_SIGMA_FLOOR = 0.1
const GATE_MODE       = :soft
const GATE_K_PRE      = 4
const GATE_K_POST     = 8
const GATE_MIN_LEN    = 4

const DATA_PATH      = ".local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls"
const LIN_CHAIN_PATH = ".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls"
const SUR_CHAIN_PATHS = [
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls",
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed5.jls",
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed6.jls",
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed7.jls",
]
const SURROGATE_PATH = ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls"
const GATE_PATH      = ".local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p.jls"
const OUT_DIR        = ".local_artifacts/ll_decomposition"

Random.seed!(SEED)

println("=" ^ 78)
println("LL GAP DECOMPOSITION: parameter relocation vs model change")
println("Started: $(now())")
println("=" ^ 78)

# ============================================================================
# Step 1: Load data + posterior means
# ============================================================================

println("\n[Step 1] Loading data + posterior means")

payload = MacroModelling.load_hlt_synthetic_scenario(DATA_PATH)
obs_data       = payload["obs_data"]
obs_sigma_base = payload["obs_sigma"]
theta_names    = payload["theta_names"]
observables    = payload["observables"]
state_names    = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
s0             = payload["s0"]
shock_sigmas   = payload["shock_sigmas"]

d_obs   = size(obs_data, 1)
T_obs   = size(obs_data, 2)
n_theta = length(theta_names)
d_state = length(s0)

println("  Observables: $observables ($d_obs)")
println("  Periods:     $T_obs   Parameters: $n_theta")

# theta_linear from the linear chain
lin_chain = deserialize(LIN_CHAIN_PATH)
theta_linear = haskey(lin_chain, "theta_post_mean") ?
    Float64.(lin_chain["theta_post_mean"]) :
    Float64.(vec(mean(lin_chain["chain"], dims=1)))
ll_post_mean_lin_recorded = get(lin_chain, "ll_post_mean", NaN)
println("  Linear chain LL_post_mean (recorded): $(round(ll_post_mean_lin_recorded, digits=2))")

# theta_surrogate from pooling 4 warm-started surrogate chains
sur_chains = [deserialize(p) for p in SUR_CHAIN_PATHS]
sur_mats   = [c["chain"] for c in sur_chains]
pooled     = vcat(sur_mats...)
theta_surrogate = vec(mean(pooled, dims=1))
sur_chain_ll = [get(c, "ll_post_mean", NaN) for c in sur_chains]
println("  Surrogate per-chain LL_post_mean (recorded): $(round.(sur_chain_ll, digits=2))")
println("  Pooled draws: $(size(pooled))")

println("\n  theta_linear:")
for i in 1:n_theta
    @printf("    %-12s lin=%.4f  sur=%.4f\n", theta_names[i], theta_linear[i], theta_surrogate[i])
end

# ============================================================================
# Step 2: Load model + surrogate
# ============================================================================

println("\n[Step 2] Loading HLT model + NN surrogate")

repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  Model: $(mm_model.model_name)")

# Solve the model so build_matrix_rom_predict is well-defined (avoids 0x0 mat-mul)
MacroModelling.solve!(mm_model)

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(SURROGATE_PATH)
frozen   = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)

# theta indexing
theta_param_idx = Int.(indexin(theta_names, mm_model.parameters))
any(==(nothing), theta_param_idx) && error("Theta names not found in model parameters")

obs_idx   = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names, mm_model.var))
any(==(nothing), obs_idx)   && error("Observables not found")
any(==(nothing), state_idx) && error("State names not found")

base_parameters = copy(mm_model.parameter_values)

# ============================================================================
# Step 3: Build observation sigma (matches surrogate runner)
# ============================================================================

println("\n[Step 3] Building observation sigma")
obs_sigma = copy(obs_sigma_base)
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_rmse = val_rmse[1:d_obs] .* OBS_SIGMA_SCALE
    obs_sigma = max.(obs_sigma, obs_rmse)
end
if OBS_SIGMA_FLOOR > 0
    obs_sigma = max.(obs_sigma, OBS_SIGMA_FLOOR)
end
println("  obs_sigma: $(round.(obs_sigma, sigdigits=3))")

# ============================================================================
# Step 4: Build ROM + NN predict functions (matches surrogate runner)
# ============================================================================

println("\n[Step 4] Building ROM + NN predict functions")

rom_predictor = RomPredictor(mm_model,
                             1,
                             :baseline,
                             false,
                             Int[],
                             base_parameters,
                             nothing,
                             nothing,
                             Int.(state_idx),
                             Int.(obs_idx))
ensure_rom_cache!(rom_predictor, Float64[])

rom_full_predict = function(state::AbstractVector, shock_t::AbstractVector, theta_local::AbstractVector)
    return rom_predict(rom_predictor, state, shock_t, theta_local)
end

surrogate_theta_names = get(sur_meta, "theta_names", Symbol[])
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    _sur_theta_baseline = zeros(Float64, length(surrogate_theta_names))
    _sur_theta_est_idx = zeros(Int, length(surrogate_theta_names))
    for (si, sname) in enumerate(surrogate_theta_names)
        ei = findfirst(==(sname), theta_names)
        if ei !== nothing
            _sur_theta_est_idx[si] = ei
        else
            pi_ = findfirst(==(sname), mm_model.parameters)
            _sur_theta_baseline[si] = pi_ !== nothing ? base_parameters[pi_] : 0.0
        end
    end
    function _pad_theta(theta_local::AbstractVector)
        theta_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                theta_full[i] = theta_local[_sur_theta_est_idx[i]]
            end
        end
        return theta_full
    end
    surrogate_residual_predict = (state, shock_t, theta_local) ->
        predict_frozen(frozen, vcat(state, shock_t, _pad_theta(theta_local)))[1:d_obs]
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
    single_nn_residual = function(x_nn::AbstractVector)
        d_prefix = d_state + length(shock_sigmas)
        x_padded = vcat(x_nn[1:d_prefix], _pad_theta(x_nn[(d_prefix+1):end]))
        return predict_frozen(frozen, x_padded)
    end
else
    surrogate_residual_predict = (state, shock_t, theta_local) ->
        predict_frozen(frozen, vcat(state, shock_t, theta_local))[1:d_obs]
    batch_nn_residual = (X_nn) -> predict_frozen_batch(frozen, X_nn)
    single_nn_residual = (x_nn) -> predict_frozen(frozen, x_nn)
end

surrogate_step_predict = (state, shock_t, theta_local) -> MacroModelling.predict_additive_residual(
    rom_full_predict,
    surrogate_residual_predict,
    state,
    shock_t,
    theta_local,
    d_obs;
    allow_full_residual = false,
)

rom_only_predict = (state, shock_t, theta_local) -> MacroModelling.predict_from_full(
    rom_full_predict,
    state,
    shock_t,
    theta_local,
    d_obs,
)

nn_correction_clamp = nothing
if val_rmse !== nothing && length(val_rmse) == frozen.d_out
    nn_correction_clamp = 3.0 .* val_rmse
end

# ============================================================================
# Step 5: Gate calibration (matches surrogate runner)
# ============================================================================

println("\n[Step 5] Gate calibration")

obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)

gate_probs = nothing
gate_mask  = trues(T_obs)
gate_calib = MacroModelling.load_hlt_gate_calibration(GATE_PATH)
tau_eps = gate_calib["tau_eps"]
tau_y   = gate_calib["tau_y"]
gate_use_eps = get(gate_calib, "use_eps", true)
gate_use_y   = get(gate_calib, "use_y", true)

if haskey(gate_calib, "e_stats") && haskey(gate_calib, "f_stats") &&
   length(gate_calib["e_stats"]) == T_obs && length(gate_calib["f_stats"]) == T_obs
    e_stat = vec(Float64.(gate_calib["e_stats"]))
    f_stat = vec(Float64.(gate_calib["f_stats"]))
else
    error("Gate calibration missing e_stats/f_stats or wrong length")
end

eps_mask = gate_use_eps ? (e_stat .> tau_eps) : falses(T_obs)
y_mask   = gate_use_y   ? (f_stat .> tau_y)   : falses(T_obs)
base_mask = eps_mask .| y_mask
gate_mask = MacroModelling.apply_gate_padding(base_mask, GATE_K_PRE, GATE_K_POST, GATE_MIN_LEN)

if GATE_MODE == :soft
    gate_prob_floor_val = 1e-4
    gate_prob_ceil_val  = 1.0 - 1e-4
    target_share = mean(base_mask)
    prior_probs = fill(target_share, T_obs)
    prior_probs = clamp.(prior_probs, gate_prob_floor_val, gate_prob_ceil_val)
    prior_logit = MacroModelling.logit.(prior_probs)
    eps_scale = max(tau_eps, eps(Float64))
    y_scale   = max(tau_y,   eps(Float64))
    eps_score = gate_use_eps ? ((e_stat .- tau_eps) ./ eps_scale) : zeros(T_obs)
    y_score   = gate_use_y   ? ((f_stat .- tau_y)   ./ y_scale)   : zeros(T_obs)
    score = eps_score .+ y_score
    gate_bias = MacroModelling.calibrate_gate_bias(score .+ prior_logit, target_share)
    gate_probs = MacroModelling.logistic.(gate_bias .+ score .+ prior_logit)
    gate_probs = clamp.(gate_probs, gate_prob_floor_val, gate_prob_ceil_val)
    println("  Gate (soft): mean_prob=$(round(mean(gate_probs), sigdigits=3))")
else
    println("  Gate (hard): $(round(100*mean(gate_mask), digits=1))% periods use surrogate")
end

use_regime_switching = !all(gate_mask)

# ============================================================================
# Step 6: LL evaluators
# ============================================================================

println("\n[Step 6] LL evaluator definitions")

# Linear (Kalman) LL (sum over T)
function ll_linear(theta_constrained::Vector{Float64})
    ll_vec = MacroModelling.linear_model_loglik_per_period(
        mm_model,
        obs_data_ka,
        theta_constrained,
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
    return sum(ll_vec)
end

# Surrogate (regime-switched, inversion-filter) LL (scalar)
function ll_surrogate(theta_constrained::Vector{Float64})
    # Per-period surrogate LL via inversion filter + NN
    ll_surr_vec, _ = MacroModelling.inversion_loglik_per_period(
        rom_only_predict,
        s0,
        theta_constrained,
        obs_data,
        obs_sigma,
        shock_sigmas;
        batch_eval_residual_fn  = batch_nn_residual,
        single_eval_residual_fn = use_regime_switching ? single_nn_residual : nothing,
        gate_mask               = use_regime_switching ? BitVector(gate_mask) : nothing,
        correction_clamp        = use_regime_switching ? nn_correction_clamp : nothing,
        maxit  = INV_MAXIT,
        tol    = INV_TOL,
        lambda = INV_LAMBDA,
    )
    if use_regime_switching
        ll_kal_vec = MacroModelling.linear_model_loglik_per_period(
            mm_model,
            obs_data_ka,
            theta_constrained,
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
        if GATE_MODE == :soft && gate_probs !== nothing
            return MacroModelling.mix_loglikelihood(ll_surr_vec, ll_kal_vec, gate_probs)
        else
            return sum(ll_surr_vec[gate_mask]) + sum(ll_kal_vec[.!gate_mask])
        end
    else
        return sum(ll_surr_vec)
    end
end

# Linear+gate ablation: keep the same inversion filter, observation scaling,
# soft gate, and Kalman mixture, but set all NN residual corrections to zero.
zero_batch_residual = function(X_nn::AbstractMatrix)
    return zeros(eltype(X_nn), frozen.d_out, size(X_nn, 2))
end
zero_single_residual = function(x_nn::AbstractVector)
    return zeros(eltype(x_nn), frozen.d_out)
end

function ll_linear_gate(theta_constrained::Vector{Float64})
    ll_gate_vec, _ = MacroModelling.inversion_loglik_per_period(
        rom_only_predict,
        s0,
        theta_constrained,
        obs_data,
        obs_sigma,
        shock_sigmas;
        batch_eval_residual_fn  = zero_batch_residual,
        single_eval_residual_fn = use_regime_switching ? zero_single_residual : nothing,
        gate_mask               = use_regime_switching ? BitVector(gate_mask) : nothing,
        correction_clamp        = nothing,
        maxit  = INV_MAXIT,
        tol    = INV_TOL,
        lambda = INV_LAMBDA,
    )
    if use_regime_switching
        ll_kal_vec = MacroModelling.linear_model_loglik_per_period(
            mm_model,
            obs_data_ka,
            theta_constrained,
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
        if GATE_MODE == :soft && gate_probs !== nothing
            return MacroModelling.mix_loglikelihood(ll_gate_vec, ll_kal_vec, gate_probs)
        else
            return sum(ll_gate_vec[gate_mask]) + sum(ll_kal_vec[.!gate_mask])
        end
    else
        return sum(ll_gate_vec)
    end
end

# ============================================================================
# Step 7: 2x2 evaluation + linear+gate ablation
# ============================================================================

println("\n[Step 7] Evaluating 2x2 LL table")

println("  -> LL_linear(theta_linear) ...")
t0 = time()
LL_lin_at_lin = ll_linear(theta_linear)
@printf("     %.4f   (%.1f s)\n", LL_lin_at_lin, time()-t0)

println("  -> LL_linear(theta_surrogate) ...")
t0 = time()
LL_lin_at_sur = ll_linear(theta_surrogate)
@printf("     %.4f   (%.1f s)\n", LL_lin_at_sur, time()-t0)

println("  -> LL_linear_gate(theta_linear) ...")
t0 = time()
LL_gate_at_lin = ll_linear_gate(theta_linear)
@printf("     %.4f   (%.1f s)\n", LL_gate_at_lin, time()-t0)

println("  -> LL_linear_gate(theta_surrogate) ...")
t0 = time()
LL_gate_at_sur = ll_linear_gate(theta_surrogate)
@printf("     %.4f   (%.1f s)\n", LL_gate_at_sur, time()-t0)

println("  -> LL_surrogate(theta_linear) ...")
t0 = time()
LL_sur_at_lin = ll_surrogate(theta_linear)
@printf("     %.4f   (%.1f s)\n", LL_sur_at_lin, time()-t0)

println("  -> LL_surrogate(theta_surrogate) ...")
t0 = time()
LL_sur_at_sur = ll_surrogate(theta_surrogate)
@printf("     %.4f   (%.1f s)\n", LL_sur_at_sur, time()-t0)

# ============================================================================
# Step 8: Decomposition
# ============================================================================

println("\n[Step 8] Decomposition")

dLL_total = LL_sur_at_sur - LL_lin_at_lin

# Path A: model change first, then parameter relocation
A_model     = LL_sur_at_lin - LL_lin_at_lin           # at theta_linear
A_param     = LL_sur_at_sur - LL_sur_at_lin           # under surrogate
A_total     = A_model + A_param

# Path B: parameter relocation first, then model change
B_param     = LL_lin_at_sur - LL_lin_at_lin           # under linear
B_model     = LL_sur_at_sur - LL_lin_at_sur           # at theta_surrogate
B_total     = B_param + B_model

println("  Total gap:  $(round(dLL_total, digits=2)) nats")
println("  Path A: model_first=$(round(A_model, digits=2))  + param_first=$(round(A_param, digits=2))  = $(round(A_total, digits=2))")
println("  Path B: param_first=$(round(B_param, digits=2))  + model_first=$(round(B_model, digits=2))  = $(round(B_total, digits=2))")

avg_model = 0.5 * (A_model + B_model)
avg_param = 0.5 * (A_param + B_param)
avg_total = avg_model + avg_param
share_model = avg_model / dLL_total
share_param = avg_param / dLL_total

gate_gain_at_lin = LL_gate_at_lin - LL_lin_at_lin
nn_gain_at_lin   = LL_sur_at_lin - LL_gate_at_lin
full_gain_at_lin = LL_sur_at_lin - LL_lin_at_lin

gate_gain_at_sur = LL_gate_at_sur - LL_lin_at_sur
nn_gain_at_sur   = LL_sur_at_sur - LL_gate_at_sur
full_gain_at_sur = LL_sur_at_sur - LL_lin_at_sur

avg_gate_gain = 0.5 * (gate_gain_at_lin + gate_gain_at_sur)
avg_nn_gain   = 0.5 * (nn_gain_at_lin + nn_gain_at_sur)
avg_full_gain = 0.5 * (full_gain_at_lin + full_gain_at_sur)
gate_share_fixed_theta = avg_gate_gain / avg_full_gain
nn_share_fixed_theta   = avg_nn_gain / avg_full_gain

println("  Symmetric (avg):  model=$(round(avg_model, digits=2))  param=$(round(avg_param, digits=2))  total=$(round(avg_total, digits=2))")
println("  Shares:           model=$(round(100*share_model, digits=1))%  param=$(round(100*share_param, digits=1))%")
println("  Fixed-theta ablation: gate=$(round(avg_gate_gain, digits=2))  nn=$(round(avg_nn_gain, digits=2))  full=$(round(avg_full_gain, digits=2))")
println("  Fixed-theta shares:   gate=$(round(100*gate_share_fixed_theta, digits=1))%  nn=$(round(100*nn_share_fixed_theta, digits=1))%")

# ============================================================================
# Step 9: Outputs (LaTeX + Markdown)
# ============================================================================

println("\n[Step 9] Writing outputs")

mkpath(OUT_DIR)
tex_path = joinpath(OUT_DIR, "ll_decomposition_table.tex")
md_path  = joinpath(OUT_DIR, "LL_DECOMPOSITION_SUMMARY.md")
jls_path = joinpath(OUT_DIR, "ll_decomposition.jls")
ablation_tex_path = joinpath(OUT_DIR, "linear_gate_ablation_table.tex")
ablation_md_path  = joinpath(OUT_DIR, "LINEAR_GATE_ABLATION_SUMMARY.md")

# --- LaTeX table ---
tex = """
% LL gap decomposition: parameter relocation vs model change
% Generated $(now())
% Total draws: $(size(pooled, 1))   T_obs = $T_obs
\\begin{table}[t]
\\centering
\\caption{Log-likelihood gap decomposition: model change vs parameter relocation}
\\label{tab:ll-decomposition}
\\begin{tabular}{lrr}
\\toprule
 & \$\\hat{\\theta}_{\\mathrm{linear}}\$ & \$\\hat{\\theta}_{\\mathrm{surrogate}}\$ \\\\
\\midrule
LL\$_{\\mathrm{linear}}\$ (Kalman)       & $(@sprintf("%.2f", LL_lin_at_lin)) & $(@sprintf("%.2f", LL_lin_at_sur)) \\\\
LL\$_{\\mathrm{surrogate}}\$ (RS+inv+NN) & $(@sprintf("%.2f", LL_sur_at_lin)) & $(@sprintf("%.2f", LL_sur_at_sur)) \\\\
\\midrule
\\multicolumn{3}{l}{\\emph{Path A:} model change \$\\to\$ parameter relocation} \\\\
\\quad model change at \$\\hat{\\theta}_{\\mathrm{linear}}\$    & \\multicolumn{2}{r}{$(@sprintf("%.2f", A_model))} \\\\
\\quad parameter shift under surrogate                          & \\multicolumn{2}{r}{$(@sprintf("%.2f", A_param))} \\\\
\\quad total                                                    & \\multicolumn{2}{r}{$(@sprintf("%.2f", A_total))} \\\\
\\midrule
\\multicolumn{3}{l}{\\emph{Path B:} parameter relocation \$\\to\$ model change} \\\\
\\quad parameter shift under linear                             & \\multicolumn{2}{r}{$(@sprintf("%.2f", B_param))} \\\\
\\quad model change at \$\\hat{\\theta}_{\\mathrm{surrogate}}\$ & \\multicolumn{2}{r}{$(@sprintf("%.2f", B_model))} \\\\
\\quad total                                                    & \\multicolumn{2}{r}{$(@sprintf("%.2f", B_total))} \\\\
\\midrule
\\multicolumn{3}{l}{\\emph{Symmetric average shares}} \\\\
\\quad model change share                                       & \\multicolumn{2}{r}{$(@sprintf("%.1f", 100*share_model))\\%} \\\\
\\quad parameter relocation share                               & \\multicolumn{2}{r}{$(@sprintf("%.1f", 100*share_param))\\%} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
open(tex_path, "w") do io
    write(io, tex)
end
println("  Wrote: $tex_path")

ablation_tex = """
% Linear+gate fixed-parameter ablation
% Generated $(now())
\\begin{table}[t]
\\centering
\\caption{Fixed-parameter ablation: gate/inversion architecture vs neural residual}
\\label{tab:linear-gate-ablation}
\\begin{tabular}{lrr}
\\toprule
 & \$\\hat{\\theta}_{\\mathrm{linear}}\$ & \$\\hat{\\theta}_{\\mathrm{surrogate}}\$ \\\\
\\midrule
LL\$_{\\mathrm{linear}}\$ (Kalman)             & $(@sprintf("%.2f", LL_lin_at_lin)) & $(@sprintf("%.2f", LL_lin_at_sur)) \\\\
LL\$_{\\mathrm{gate}}\$ (gate+inversion, no NN) & $(@sprintf("%.2f", LL_gate_at_lin)) & $(@sprintf("%.2f", LL_gate_at_sur)) \\\\
LL\$_{\\mathrm{surrogate}}\$ (gate+inversion+NN) & $(@sprintf("%.2f", LL_sur_at_lin)) & $(@sprintf("%.2f", LL_sur_at_sur)) \\\\
\\midrule
Gate/inversion gain over Kalman                & $(@sprintf("%.2f", gate_gain_at_lin)) & $(@sprintf("%.2f", gate_gain_at_sur)) \\\\
NN residual gain over gate-only                & $(@sprintf("%.2f", nn_gain_at_lin)) & $(@sprintf("%.2f", nn_gain_at_sur)) \\\\
Full fixed-parameter gain                      & $(@sprintf("%.2f", full_gain_at_lin)) & $(@sprintf("%.2f", full_gain_at_sur)) \\\\
\\midrule
Symmetric gate/inversion share                 & \\multicolumn{2}{r}{$(@sprintf("%.1f", 100*gate_share_fixed_theta))\\%} \\\\
Symmetric NN residual share                    & \\multicolumn{2}{r}{$(@sprintf("%.1f", 100*nn_share_fixed_theta))\\%} \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
open(ablation_tex_path, "w") do io
    write(io, ablation_tex)
end
println("  Wrote: $ablation_tex_path")

# --- Markdown summary ---
headline_who = abs(avg_model) > abs(avg_param) ? "MODEL CHANGE" : "PARAMETER RELOCATION"
md = """
# LL Gap Decomposition: Parameter Relocation vs Model Change

Generated: $(now())

## Inputs

- T = $T_obs quarters, 7 observables
- theta_linear: posterior mean from `$LIN_CHAIN_PATH`
- theta_surrogate: pooled posterior mean (4 chains, $(size(pooled,1)) total draws)

## 2x2 LL table

|                                    | theta_linear                              | theta_surrogate                           |
|------------------------------------|-------------------------------------------|-------------------------------------------|
| LL_linear (Kalman)                 | $(@sprintf("%.2f", LL_lin_at_lin))        | $(@sprintf("%.2f", LL_lin_at_sur))        |
| LL_surrogate (RS+inv+NN)           | $(@sprintf("%.2f", LL_sur_at_lin))        | $(@sprintf("%.2f", LL_sur_at_sur))        |

Total gap LL_surrogate(theta_surrogate) - LL_linear(theta_linear) = **$(@sprintf("%.2f", dLL_total)) nats**

## Path A: model change then parameter relocation

- model change at theta_linear:               **$(@sprintf("%.2f", A_model)) nats**
- parameter relocation under surrogate model: **$(@sprintf("%.2f", A_param)) nats**
- check sum:                                  $(@sprintf("%.2f", A_total)) nats

## Path B: parameter relocation then model change

- parameter relocation under linear model:    **$(@sprintf("%.2f", B_param)) nats**
- model change at theta_surrogate:            **$(@sprintf("%.2f", B_model)) nats**
- check sum:                                  $(@sprintf("%.2f", B_total)) nats

## Symmetric averages and shares

- Avg model-change contribution:        $(@sprintf("%.2f", avg_model)) nats ($(@sprintf("%.1f", 100*share_model))%)
- Avg parameter-relocation contribution: $(@sprintf("%.2f", avg_param)) nats ($(@sprintf("%.1f", 100*share_param))%)

## Headline

The $(@sprintf("%.2f", dLL_total))-nat gap between LL_surrogate(theta_surrogate) and
LL_linear(theta_linear) is dominated by **$headline_who**
(symmetric share $(@sprintf("%.1f", 100*max(share_model, share_param)))%). This
addresses the referee critique that linear estimation could absorb nonlinearity
into shock variances at different parameters: the model-change component
remains $(@sprintf("%.2f", avg_model)) nats even after allowing each model to
re-optimize over its own posterior mean.
"""
open(md_path, "w") do io
    write(io, md)
end
println("  Wrote: $md_path")

ablation_md = """
# Linear+Gate Fixed-Parameter Ablation

Generated: $(now())

This artifact evaluates a no-NN ablation at existing posterior means. It is not
a substitute for a full HMC run under the no-NN likelihood. It answers the
fixed-parameter question: holding parameters at the linear and surrogate
posterior means, how much of the likelihood change comes from the gate/inversion
architecture itself versus the learned NN residual correction?

## LL table

| Likelihood | theta_linear | theta_surrogate |
|---|---:|---:|
| Linear Kalman | $(@sprintf("%.2f", LL_lin_at_lin)) | $(@sprintf("%.2f", LL_lin_at_sur)) |
| Linear+gate, no NN | $(@sprintf("%.2f", LL_gate_at_lin)) | $(@sprintf("%.2f", LL_gate_at_sur)) |
| Full gate+NN | $(@sprintf("%.2f", LL_sur_at_lin)) | $(@sprintf("%.2f", LL_sur_at_sur)) |

## Fixed-parameter gains

| Contribution | theta_linear | theta_surrogate |
|---|---:|---:|
| Gate/inversion over Kalman | $(@sprintf("%.2f", gate_gain_at_lin)) | $(@sprintf("%.2f", gate_gain_at_sur)) |
| NN residual over gate-only | $(@sprintf("%.2f", nn_gain_at_lin)) | $(@sprintf("%.2f", nn_gain_at_sur)) |
| Full gain over Kalman | $(@sprintf("%.2f", full_gain_at_lin)) | $(@sprintf("%.2f", full_gain_at_sur)) |

## Symmetric fixed-parameter shares

- Gate/inversion architecture: $(@sprintf("%.1f", 100*gate_share_fixed_theta))%
- NN residual correction: $(@sprintf("%.1f", 100*nn_share_fixed_theta))%

## Required long-run follow-up

Run a dedicated no-NN HMC chain before presenting this as a posterior table:

```bash
julia --project=. scripts/run_surrogate_hmc_advancedhmc.jl \\
  --surrogate=$SURROGATE_PATH \\
  --data=$DATA_PATH \\
  --gate-calibration=$GATE_PATH \\
  --init-from=$LIN_CHAIN_PATH \\
  --out=.local_artifacts/hlt_18param_realdata/hlt_linear_gate_hmc_extended_18p_2000.jls \\
  --samples=2000 --adapt=500 --seed=42 \\
  --disable-nn-correction=true
```
"""
open(ablation_md_path, "w") do io
    write(io, ablation_md)
end
println("  Wrote: $ablation_md_path")

# --- Raw artifact ---
serialize(jls_path, Dict(
    "theta_linear"     => theta_linear,
    "theta_surrogate"  => theta_surrogate,
    "theta_names"      => theta_names,
    "LL_lin_at_lin"    => LL_lin_at_lin,
    "LL_lin_at_sur"    => LL_lin_at_sur,
    "LL_gate_at_lin"   => LL_gate_at_lin,
    "LL_gate_at_sur"   => LL_gate_at_sur,
    "LL_sur_at_lin"    => LL_sur_at_lin,
    "LL_sur_at_sur"    => LL_sur_at_sur,
    "dLL_total"        => dLL_total,
    "A_model"          => A_model,
    "A_param"          => A_param,
    "B_model"          => B_model,
    "B_param"          => B_param,
    "avg_model"        => avg_model,
    "avg_param"        => avg_param,
    "share_model"      => share_model,
    "share_param"      => share_param,
    "gate_gain_at_lin" => gate_gain_at_lin,
    "gate_gain_at_sur" => gate_gain_at_sur,
    "nn_gain_at_lin"   => nn_gain_at_lin,
    "nn_gain_at_sur"   => nn_gain_at_sur,
    "avg_gate_gain"    => avg_gate_gain,
    "avg_nn_gain"      => avg_nn_gain,
    "avg_full_gain"    => avg_full_gain,
    "gate_share_fixed_theta" => gate_share_fixed_theta,
    "nn_share_fixed_theta"   => nn_share_fixed_theta,
    "T_obs"            => T_obs,
    "n_pooled_draws"   => size(pooled, 1),
    "timestamp"        => string(now()),
))
println("  Wrote: $jls_path")

println("\n" * "=" ^ 78)
println("DECOMPOSITION COMPLETE")
println("Finished: $(now())")
println("=" ^ 78)
