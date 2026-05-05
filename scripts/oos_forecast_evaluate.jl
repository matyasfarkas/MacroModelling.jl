#!/usr/bin/env julia
# ============================================================================
# OOS FORECAST EVALUATION — 1-step-ahead RMSE (ROM1 vs Switching Surrogate)
# ============================================================================
#
# Uses posterior means from two truncated-sample OOS chains (linear &
# surrogate), evaluates 1-step-ahead forecast errors on a holdout window by
# running each model's filter through the entire sample and extracting the
# innovations in the OOS window. Default settings reproduce the pre-COVID
# exercise (t = 245..265). For the quiet-sample exercise, use:
#   --tag=quiet_1994 --t-pre=144 --t-end=196 --window-label=Early-holdout
#
#   Linear:    custom Kalman filter (A, B, C from first-order solution) →
#              innovation v_t = y_t - C·μ_{t|t-1}.
#   Surrogate: inversion filter (ROM1 shock recovery + optional NN gate
#              correction) up to t-1, then predict obs_t with ε=0 via
#              ROM1 + NN.
#
# Outputs:
#   .local_artifacts/oos_forecast/oos_comparison.tex  — LaTeX table
#   .local_artifacts/oos_forecast/oos_comparison.md   — Markdown summary
#   .local_artifacts/oos_forecast/OOS_SUMMARY.md      — High-level writeup
# ============================================================================

using Serialization, Printf, Statistics, LinearAlgebra, Dates
import LinearAlgebra as ℒ
using AxisKeys
using MacroModelling

repo_root = normpath(joinpath(@__DIR__, ".."))
cd(repo_root)

function parse_kv(args, key, default)
    for a in args
        if startswith(a, "$key=")
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

t_pre = parse(Int, parse_kv(ARGS, "--t-pre", "244"))  # last in-sample index (1-based)
tag = parse_kv(ARGS, "--tag", t_pre == 244 ? "preCOVID" : "T$(t_pre)")
t_end_arg = parse_kv(ARGS, "--t-end", "")
window_label = parse_kv(ARGS, "--window-label", t_pre == 244 ? "COVID" : "Early holdout")
linear_chain    = parse_kv(ARGS, "--linear-chain",
    ".local_artifacts/oos_forecast/hlt_linear_hmc_$(tag)_500.jls")
surrogate_chain = parse_kv(ARGS, "--surrogate-chain",
    ".local_artifacts/oos_forecast/hlt_surrogate_hmc_$(tag)_300.jls")
payload_full    = parse_kv(ARGS, "--data-full",
    ".local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls")
surrogate_bundle_path = parse_kv(ARGS, "--surrogate",
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls")
gate_path = parse_kv(ARGS, "--gate-calibration",
    ".local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p.jls")
gate_k_pre = parse(Int, parse_kv(ARGS, "--gate-k-pre", "4"))
gate_k_post = parse(Int, parse_kv(ARGS, "--gate-k-post", "8"))
gate_min_len = parse(Int, parse_kv(ARGS, "--gate-min-len", "4"))

println("=" ^ 72)
println("OOS FORECAST EVALUATION")
println("Started: $(now())")
println("=" ^ 72)
println("  Linear chain:    $linear_chain")
println("  Surrogate chain: $surrogate_chain")
println("  Data payload:    $payload_full")
println("  Surrogate NN:    $surrogate_bundle_path")
println("  Gate calib:      $gate_path")
println("  Gate padding:    pre=$gate_k_pre post=$gate_k_post min_len=$gate_min_len")
println("  T_pre:           $t_pre (last in-sample index)")

# ----------------------------------------------------------------------------
# Load full-sample data (includes OOS window)
# ----------------------------------------------------------------------------
payload = MacroModelling.load_hlt_synthetic_scenario(payload_full)
obs_data    = payload["obs_data"]            # (d_obs, T_full)
obs_sigma   = payload["obs_sigma"]
theta_names = payload["theta_names"]
observables = payload["observables"]
state_names = payload["state_names"]
s0          = payload["s0"]
shock_sigmas = payload["shock_sigmas"]
T_full = size(obs_data, 2)
d_obs  = size(obs_data, 1)
eval_end = t_end_arg == "" ? T_full : min(parse(Int, t_end_arg), T_full)
eval_end > t_pre || error("t_end=$eval_end must exceed t_pre=$t_pre")
T_oos  = eval_end - t_pre
window_start = parse(Int, parse_kv(ARGS, "--window-start", string(t_pre + 1)))
window_end   = parse(Int, parse_kv(ARGS, "--window-end", string(min(window_start + 5, eval_end))))
window_start = clamp(window_start, t_pre + 1, eval_end)
window_end = clamp(window_end, window_start, eval_end)
println("  T_full=$T_full  T_oos=$T_oos (OOS indices $(t_pre+1)..$eval_end)")
println("  Observables: $observables")
println("  $(window_label) window: $(window_start)..$(window_end)")

# ----------------------------------------------------------------------------
# Load posterior means from truncated-sample chains
# ----------------------------------------------------------------------------
isfile(linear_chain)    || error("Linear chain not found: $linear_chain")
isfile(surrogate_chain) || error("Surrogate chain not found: $surrogate_chain")
lin_res = deserialize(linear_chain)
sur_res = deserialize(surrogate_chain)
θ_lin = Float64.(lin_res["theta_post_mean"])
θ_sur = Float64.(sur_res["theta_post_mean"])
println("  θ_lin ($tag): ", round.(θ_lin, digits=4))
println("  θ_sur ($tag): ", round.(θ_sur, digits=4))

# ----------------------------------------------------------------------------
# Load HLT model (non-OBC linear ROM1)
# ----------------------------------------------------------------------------
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
theta_param_idx = Int.(indexin(theta_names, mm_model.parameters))
base_parameters = copy(mm_model.parameter_values)

# ============================================================================
# LINEAR FORECAST — Custom Kalman filter (stores 1-step-ahead innovations)
# ============================================================================
#
# Rebuilds A, B, C from the solved model exactly as MacroModelling's
# filter_and_smooth does, so the 1-step-ahead prediction innovation v_t =
# y_t − C·μ_{t|t-1} is recoverable even in the OOS window.
# ============================================================================

println("\n" * "-" ^ 72)
println("LINEAR 1-STEP-AHEAD INNOVATIONS ($tag posterior mean)")
println("-" ^ 72)

function build_linear_state_space(model, params)
    opts = MacroModelling.merge_calculation_options()
    # Set parameter values on the model and solve
    model.parameter_values .= params
    MacroModelling.solve!(model; opts = opts)
    SS_and_pars, (sol_err, _) =
        MacroModelling.get_NSSS_and_parameters(model, params; opts = opts)
    sol_err < opts.tol.NSSS_acceptance_tol ||
        error("NSSS solver failed: err=$sol_err at posterior mean")
    ∇₁ = MacroModelling.calculate_jacobian(params, SS_and_pars, model)
    sol, _, solved =
        MacroModelling.calculate_first_order_solution(∇₁; T = model.timings, opts = opts)
    solved || error("First-order solution failed at posterior mean")
    TT = model.timings
    A = @views sol[:, 1:TT.nPast_not_future_and_mixed] *
              ℒ.diagm(ones(TT.nVars))[TT.past_not_future_and_mixed_idx, :]
    B = @views sol[:, TT.nPast_not_future_and_mixed + 1 : end]
    # Observable row order: same as filter_and_smooth (sorted union of aux, var, exo_present)
    obs_rowset = sort(union(model.aux, model.var, model.exo_present))
    obs_sorted = sort(observables)
    C = ℒ.diagm(ones(TT.nVars))[sort(Int.(indexin(obs_sorted, obs_rowset))), :]
    P̄ = MacroModelling.calculate_covariance(model.parameter_values, model; opts = opts)[1]
    return A, B, C, P̄, obs_sorted
end

# Plug posterior mean into model parameters
params_lin = copy(base_parameters)
for (i, pi) in enumerate(theta_param_idx)
    params_lin[pi] = θ_lin[i]
end

A_lin, B_lin, C_lin, P̄_lin, obs_sorted = build_linear_state_space(mm_model, params_lin)
println("  Built A ($(size(A_lin))), B ($(size(B_lin))), C ($(size(C_lin)))")
println("  Observable order in filter: $obs_sorted")

# Reorder data to match sorted observable order
obs_perm = Int.(indexin(obs_sorted, observables))
any(isnothing, obs_perm) && error("Observable mismatch vs sorted filter order")
obs_data_sorted = obs_data[obs_perm, :]
obs_sigma_sorted = obs_sigma[obs_perm]

n_states = size(A_lin, 1)
𝐁 = B_lin * B_lin'
μ_pred = zeros(n_states, T_full + 1)
P_pred = zeros(n_states, n_states, T_full + 1)
P_pred[:, :, 1] = P̄_lin
# 1-step-ahead innovations in sorted-obs order
v_kalman = zeros(d_obs, T_full)

for t in 1:T_full
    # 1-step-ahead prediction at t is μ_pred[:, t] = A·μ_{t-1|t-1}
    # (Written in the form of Durbin–Koopman used by filter_and_smooth.)
    y_pred_t = C_lin * μ_pred[:, t]
    v_kalman[:, t] = obs_data_sorted[:, t] - y_pred_t

    F = C_lin * P_pred[:, :, t] * C_lin'
    F_lu = ℒ.lu(F, check=false)
    ℒ.issuccess(F_lu) || (println("  Kalman: singular F at t=$t — skipping update"); break)
    iF = inv(F_lu)
    PCiF = P_pred[:, :, t] * C_lin' * iF
    L = A_lin - A_lin * PCiF * C_lin
    P_pred[:, :, t+1] = A_lin * P_pred[:, :, t] * L' + 𝐁
    μ_pred[:, t+1] = A_lin * (μ_pred[:, t] + PCiF * v_kalman[:, t])
end

# Map innovation rows back to the original observable ordering
invperm_obs = Int.(indexin(observables, obs_sorted))
v_linear = v_kalman[invperm_obs, :]   # (d_obs, T_full) in the same order as observables

# ============================================================================
# SURROGATE FORECAST — Inversion filter state trajectory + 1-step predict
# ============================================================================

println("\n" * "-" ^ 72)
println("SURROGATE 1-STEP-AHEAD INNOVATIONS ($tag posterior mean)")
println("-" ^ 72)

obs_idx  = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names, mm_model.var))

# Plug posterior mean θ_sur into the base parameters so the ROM1 predictor
# solves the model at the posterior (not at calibrated baseline).
base_parameters_sur = copy(base_parameters)
for (i, pi) in enumerate(theta_param_idx)
    base_parameters_sur[pi] = θ_sur[i]
end

rom_predictor = RomPredictor(mm_model, 1, :baseline, false, Int[],
                              base_parameters_sur, nothing, nothing,
                              state_idx, obs_idx)
ensure_rom_cache!(rom_predictor, Float64[])

rom_full_predict(state, shock_t, θ_local) =
    rom_predict(rom_predictor, state, shock_t, θ_local)
rom_only_predict(state, shock_t, θ_local) =
    MacroModelling.predict_from_full(rom_full_predict, state, shock_t, θ_local, d_obs)

# Load the frozen surrogate
surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_bundle_path)
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)
nn_correction_clamp = (val_rmse !== nothing && length(val_rmse) == frozen.d_out) ?
                      3.0 .* val_rmse : nothing

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
    function _pad_theta(θ_local::AbstractVector)
        θ_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                θ_full[i] = θ_local[_sur_theta_est_idx[i]]
            end
        end
        return θ_full
    end
    single_nn_residual(x_nn) = begin
        d_prefix = length(s0) + length(shock_sigmas)
        x_padded = vcat(x_nn[1:d_prefix], _pad_theta(x_nn[(d_prefix+1):end]))
        return predict_frozen(frozen, x_padded)
    end
    batch_nn_residual(X_nn) = begin
        d_prefix = length(s0) + length(shock_sigmas)
        T_batch = size(X_nn, 2)
        X_padded = Matrix{eltype(X_nn)}(undef, frozen.d_in, T_batch)
        X_padded[1:d_prefix, :] .= X_nn[1:d_prefix, :]
        for t in 1:T_batch
            X_padded[(d_prefix+1):end, t] .= _pad_theta(X_nn[(d_prefix+1):end, t])
        end
        return predict_frozen_batch(frozen, X_padded)
    end
else
    single_nn_residual(x_nn) = predict_frozen(frozen, x_nn)
    batch_nn_residual(X_nn)  = predict_frozen_batch(frozen, X_nn)
end

# Load full-sample gate calibration (we reuse it since gate stats cover 265 periods)
gate_mask = trues(T_full)
gate_probs = nothing
if isfile(gate_path)
    g = MacroModelling.load_hlt_gate_calibration(gate_path)
    if haskey(g, "e_stats") && haskey(g, "f_stats") &&
       length(g["e_stats"]) >= T_full && length(g["f_stats"]) >= T_full
        e_stat = Float64.(g["e_stats"][1:T_full])
        f_stat = Float64.(g["f_stats"][1:T_full])
        tau_eps = g["tau_eps"]; tau_y = g["tau_y"]
        use_eps = get(g, "use_eps", true); use_y = get(g, "use_y", true)
        eps_mask = use_eps ? (e_stat .> tau_eps) : falses(T_full)
        y_mask   = use_y   ? (f_stat .> tau_y)   : falses(T_full)
        base_mask = eps_mask .| y_mask
        gate_mask = MacroModelling.apply_gate_padding(base_mask, gate_k_pre, gate_k_post, gate_min_len)
        # Soft gate probabilities (match training script logic)
        target_share = Float64(get(g, "target_share", mean(base_mask)))
        prior_probs = clamp.(fill(target_share, T_full), 1e-4, 1.0 - 1e-4)
        prior_logit = MacroModelling.logit.(prior_probs)
        eps_scale = max(tau_eps, eps(Float64))
        y_scale   = max(tau_y,   eps(Float64))
        eps_score = use_eps ? ((e_stat .- tau_eps) ./ eps_scale) : zeros(T_full)
        y_score   = use_y   ? ((f_stat .- tau_y)   ./ y_scale)   : zeros(T_full)
        score = eps_score .+ y_score
        gate_bias = MacroModelling.calibrate_gate_bias(score .+ prior_logit, target_share)
        gate_probs = clamp.(MacroModelling.logistic.(gate_bias .+ score .+ prior_logit),
                            1e-4, 1.0 - 1e-4)
        println("  Gate: hard $(count(gate_mask))/$T_full periods; " *
                "soft mean prob $(round(mean(gate_probs), sigdigits=3))")
    end
end

# Run inversion filter through the full sample using θ_sur → get state trajectory
println("  Running inversion filter on full sample (T=$T_full) ...")
_, shocks_sur = MacroModelling.inversion_loglik_per_period(
    rom_only_predict, s0, θ_sur, obs_data, obs_sigma, shock_sigmas;
    batch_eval_residual_fn = batch_nn_residual,
    single_eval_residual_fn = single_nn_residual,
    gate_mask = gate_mask,
    correction_clamp = nn_correction_clamp,
    maxit = 10, tol = 1e-6, lambda = 1e-4,
)

# Reconstruct state trajectory by replaying the filter deterministically
# (same gate-conditional correction used at filter time)
states = Matrix{Float64}(undef, length(s0), T_full + 1)
states[:, 1] = Float64.(s0)
obs_pred_sur_1step = Matrix{Float64}(undef, d_obs, T_full)

for t in 1:T_full
    st = states[:, t]
    # 1-step-ahead prediction from state at t-1, shocks = 0 (conditional expectation)
    if gate_mask[t]
        obs_rom, _ = rom_only_predict(st, zeros(length(shock_sigmas)), θ_sur)
        x_nn_t = vcat(st, zeros(length(shock_sigmas)), θ_sur)
        y_nn_t = single_nn_residual(x_nn_t)
        if nn_correction_clamp !== nothing
            y_nn_t = clamp.(y_nn_t, -nn_correction_clamp, nn_correction_clamp)
        end
        obs_pred_sur_1step[:, t] = obs_rom .+ y_nn_t[1:d_obs]
    else
        obs_pred_sur_1step[:, t], _ = rom_only_predict(st, zeros(length(shock_sigmas)), θ_sur)
    end
    # Now propagate state with actual recovered shocks (to get state at t+1 | t)
    eps_t = shocks_sur[:, t]
    if gate_mask[t]
        obs_rom, state_rom_next = rom_only_predict(st, eps_t, θ_sur)
        x_nn_t = vcat(st, eps_t, θ_sur)
        y_nn_t = single_nn_residual(x_nn_t)
        if nn_correction_clamp !== nothing
            y_nn_t = clamp.(y_nn_t, -nn_correction_clamp, nn_correction_clamp)
        end
        states[:, t+1] = state_rom_next .+ y_nn_t[(d_obs+1):end]
    else
        _, state_next = rom_only_predict(st, eps_t, θ_sur)
        states[:, t+1] = state_next
    end
    if !all(isfinite, states[:, t+1])
        println("  Non-finite state at t=$t — zeroing forward")
        states[:, t+1] .= 0.0
    end
end

v_surrogate = obs_data .- obs_pred_sur_1step  # (d_obs, T_full) in observable order

# ============================================================================
# RMSE summary
# ============================================================================

println("\n" * "=" ^ 72)
println("OOS RMSE SUMMARY (indices $(t_pre + 1)..$eval_end)")
println("=" ^ 72)

oos_rng   = (t_pre + 1):eval_end
covid_rng = window_start:window_end
noncovid_oos_rng = (window_end + 1):eval_end

gate_share_full = mean(gate_mask[oos_rng])
gate_share_covid = mean(gate_mask[covid_rng])
gate_share_post = isempty(collect(noncovid_oos_rng)) ? NaN : mean(gate_mask[noncovid_oos_rng])
soft_gate_share_full = gate_probs === nothing ? NaN : mean(gate_probs[oos_rng])
soft_gate_share_covid = gate_probs === nothing ? NaN : mean(gate_probs[covid_rng])
soft_gate_share_post = (gate_probs === nothing || isempty(collect(noncovid_oos_rng))) ?
                       NaN : mean(gate_probs[noncovid_oos_rng])

function rmse_per_obs(v, rng)
    return [sqrt(mean(v[i, rng].^2)) for i in 1:size(v, 1)]
end

rmse_lin_full  = rmse_per_obs(v_linear,    oos_rng)
rmse_sur_full  = rmse_per_obs(v_surrogate, oos_rng)
rmse_lin_covid = rmse_per_obs(v_linear,    covid_rng)
rmse_sur_covid = rmse_per_obs(v_surrogate, covid_rng)
rmse_lin_post  = rmse_per_obs(v_linear,    noncovid_oos_rng)
rmse_sur_post  = rmse_per_obs(v_surrogate, noncovid_oos_rng)

# Aggregate (root mean across observables — same as mean of mean-sq innovations)
function agg_rmse(v, rng)
    s2 = 0.0
    for i in 1:size(v, 1)
        s2 += mean(v[i, rng].^2)
    end
    return sqrt(s2 / size(v, 1))
end
agg_lin_full  = agg_rmse(v_linear,    oos_rng)
agg_sur_full  = agg_rmse(v_surrogate, oos_rng)
agg_lin_covid = agg_rmse(v_linear,    covid_rng)
agg_sur_covid = agg_rmse(v_surrogate, covid_rng)
agg_lin_post  = agg_rmse(v_linear,    noncovid_oos_rng)
agg_sur_post  = agg_rmse(v_surrogate, noncovid_oos_rng)

println("\nFull OOS window ($(length(oos_rng)) quarters):")
@printf("  Gate share: hard %.1f%%, soft mean %.1f%%\n",
        100 * gate_share_full,
        isfinite(soft_gate_share_full) ? 100 * soft_gate_share_full : NaN)
@printf("  %-10s %12s %12s %10s\n", "Obs", "Linear RMSE", "Switch RMSE", "Ratio S/L")
for (i, obs) in enumerate(observables)
    @printf("  %-10s %12.4f %12.4f %10.3f\n",
            String(obs), rmse_lin_full[i], rmse_sur_full[i],
            rmse_sur_full[i] / rmse_lin_full[i])
end
@printf("  %-10s %12.4f %12.4f %10.3f\n", "AGG",
        agg_lin_full, agg_sur_full, agg_sur_full / agg_lin_full)

println("\n$(window_label) window ($(length(covid_rng)) quarters):")
@printf("  Gate share: hard %.1f%%, soft mean %.1f%%\n",
        100 * gate_share_covid,
        isfinite(soft_gate_share_covid) ? 100 * soft_gate_share_covid : NaN)
@printf("  %-10s %12s %12s %10s\n", "Obs", "Linear RMSE", "Switch RMSE", "Ratio S/L")
for (i, obs) in enumerate(observables)
    @printf("  %-10s %12.4f %12.4f %10.3f\n",
            String(obs), rmse_lin_covid[i], rmse_sur_covid[i],
            rmse_sur_covid[i] / rmse_lin_covid[i])
end
@printf("  %-10s %12.4f %12.4f %10.3f\n", "AGG",
        agg_lin_covid, agg_sur_covid, agg_sur_covid / agg_lin_covid)

println("\nRemaining OOS window ($(length(noncovid_oos_rng)) quarters):")
@printf("  Gate share: hard %.1f%%, soft mean %.1f%%\n",
        100 * gate_share_post,
        isfinite(soft_gate_share_post) ? 100 * soft_gate_share_post : NaN)
@printf("  %-10s %12s %12s %10s\n", "Obs", "Linear RMSE", "Switch RMSE", "Ratio S/L")
for (i, obs) in enumerate(observables)
    @printf("  %-10s %12.4f %12.4f %10.3f\n",
            String(obs), rmse_lin_post[i], rmse_sur_post[i],
            rmse_sur_post[i] / rmse_lin_post[i])
end
@printf("  %-10s %12.4f %12.4f %10.3f\n", "AGG",
        agg_lin_post, agg_sur_post, agg_sur_post / agg_lin_post)

# ============================================================================
# LaTeX table
# ============================================================================

println("\n" * "-" ^ 72)
println("Writing LaTeX table...")
println("-" ^ 72)

tex_path = ".local_artifacts/oos_forecast/oos_comparison_$(tag).tex"
open(tex_path, "w") do io
    println(io, "% Out-of-sample 1-step-ahead forecast RMSE: ROM1 (linear Kalman) vs regime-switching surrogate")
    println(io, "% Generated $(now())")
    println(io, "% Estimation: first $t_pre periods. Forecast indices: $(t_pre+1)..$eval_end (T_oos=$T_oos).")
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{One-step-ahead out-of-sample forecast RMSE, ROM1 vs.\\ Switching Surrogate. " *
                "Estimation sample first $t_pre periods; forecast indices $(t_pre+1)--$eval_end. " *
                "Lower is better; ratio \$<\$1 favors the switching model.}")
    println(io, "\\label{tab:oos_forecast_rmse_$(replace(tag, '-' => '_'))}")
    println(io, "\\small")
    println(io, "\\begin{tabular}{lcccccccccc}")
    println(io, "\\toprule")
    println(io, "& \\multicolumn{3}{c}{Full OOS ($(length(oos_rng))Q)} " *
                "& \\multicolumn{3}{c}{$window_label ($(length(covid_rng))Q)} " *
                "& \\multicolumn{3}{c}{Remaining ($(length(noncovid_oos_rng))Q)} \\\\")
    println(io, "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
    println(io, "Observable & ROM1 & Switch & Ratio & ROM1 & Switch & Ratio & ROM1 & Switch & Ratio \\\\")
    println(io, "\\midrule")
    for (i, obs) in enumerate(observables)
        @printf(io, "%s & %.3f & %.3f & %.2f & %.3f & %.3f & %.2f & %.3f & %.3f & %.2f \\\\\n",
                String(obs),
                rmse_lin_full[i], rmse_sur_full[i], rmse_sur_full[i]/rmse_lin_full[i],
                rmse_lin_covid[i], rmse_sur_covid[i], rmse_sur_covid[i]/rmse_lin_covid[i],
                rmse_lin_post[i], rmse_sur_post[i], rmse_sur_post[i]/rmse_lin_post[i])
    end
    println(io, "\\midrule")
    @printf(io, "Aggregate & %.3f & %.3f & %.2f & %.3f & %.3f & %.2f & %.3f & %.3f & %.2f \\\\\n",
            agg_lin_full, agg_sur_full, agg_sur_full/agg_lin_full,
            agg_lin_covid, agg_sur_covid, agg_sur_covid/agg_lin_covid,
            agg_lin_post, agg_sur_post, agg_sur_post/agg_lin_post)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
end
println("  Wrote: $tex_path")

# Companion markdown
md_path = ".local_artifacts/oos_forecast/oos_comparison_$(tag).md"
open(md_path, "w") do io
    println(io, "# OOS forecast RMSE (1-step ahead)")
    println(io, "")
    println(io, "- Estimation sample: first $t_pre periods")
    println(io, "- Forecast indices:  $(t_pre + 1)..$eval_end (T_oos=$T_oos)")
    println(io, "- $window_label window: indices $(window_start)..$(window_end)")
    println(io, "- Linear chain:      `$linear_chain` (n=$(lin_res["n_samples"]) draws)")
    println(io, "- Surrogate chain:   `$surrogate_chain` (n=$(sur_res["n_samples"]) draws)")
    @printf(io, "- Gate share, full OOS: hard %.1f%%, soft mean %.1f%%\n",
            100 * gate_share_full,
            isfinite(soft_gate_share_full) ? 100 * soft_gate_share_full : NaN)
    @printf(io, "- Gate share, %s: hard %.1f%%, soft mean %.1f%%\n",
            window_label,
            100 * gate_share_covid,
            isfinite(soft_gate_share_covid) ? 100 * soft_gate_share_covid : NaN)
    println(io, "")
    println(io, "## Per-observable RMSE")
    println(io, "")
    println(io, "| Obs | ROM1 (full OOS) | Switch (full) | Ratio | ROM1 ($window_label) | Switch ($window_label) | Ratio | ROM1 (remaining) | Switch (remaining) | Ratio |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (i, obs) in enumerate(observables)
        @printf(io, "| %s | %.3f | %.3f | %.2f | %.3f | %.3f | %.2f | %.3f | %.3f | %.2f |\n",
                String(obs),
                rmse_lin_full[i], rmse_sur_full[i], rmse_sur_full[i]/rmse_lin_full[i],
                rmse_lin_covid[i], rmse_sur_covid[i], rmse_sur_covid[i]/rmse_lin_covid[i],
                rmse_lin_post[i], rmse_sur_post[i], rmse_sur_post[i]/rmse_lin_post[i])
    end
    @printf(io, "| **Aggregate** | **%.3f** | **%.3f** | **%.2f** | **%.3f** | **%.3f** | **%.2f** | **%.3f** | **%.3f** | **%.2f** |\n",
            agg_lin_full, agg_sur_full, agg_sur_full/agg_lin_full,
            agg_lin_covid, agg_sur_covid, agg_sur_covid/agg_lin_covid,
            agg_lin_post, agg_sur_post, agg_sur_post/agg_lin_post)
end
println("  Wrote: $md_path")

# ============================================================================
# Save raw arrays
# ============================================================================
serialize(".local_artifacts/oos_forecast/oos_innovations_$(tag).jls", Dict(
    "v_linear"      => v_linear,
    "v_surrogate"   => v_surrogate,
    "obs_data"      => obs_data,
    "observables"   => observables,
    "t_pre"         => t_pre,
    "eval_end"      => eval_end,
    "window_start"  => window_start,
    "window_end"    => window_end,
    "window_label"  => window_label,
    "theta_lin"     => θ_lin,
    "theta_sur"     => θ_sur,
    "rmse_lin_full" => rmse_lin_full,
    "rmse_sur_full" => rmse_sur_full,
    "rmse_lin_covid"=> rmse_lin_covid,
    "rmse_sur_covid"=> rmse_sur_covid,
    "rmse_lin_post" => rmse_lin_post,
    "rmse_sur_post" => rmse_sur_post,
    "agg_lin_full"  => agg_lin_full,
    "agg_sur_full"  => agg_sur_full,
    "agg_lin_covid" => agg_lin_covid,
    "agg_sur_covid" => agg_sur_covid,
    "agg_lin_post"  => agg_lin_post,
    "agg_sur_post"  => agg_sur_post,
    "gate_share_full" => gate_share_full,
    "gate_share_covid" => gate_share_covid,
    "gate_share_post" => gate_share_post,
    "soft_gate_share_full" => soft_gate_share_full,
    "soft_gate_share_covid" => soft_gate_share_covid,
    "soft_gate_share_post" => soft_gate_share_post,
    "gate_k_pre" => gate_k_pre,
    "gate_k_post" => gate_k_post,
    "gate_min_len" => gate_min_len,
))

# ============================================================================
# Headline summary
# ============================================================================

summary_path = ".local_artifacts/oos_forecast/OOS_SUMMARY_$(tag).md"
covid_ratio = agg_sur_covid / agg_lin_covid
full_ratio  = agg_sur_full  / agg_lin_full
post_ratio  = agg_sur_post  / agg_lin_post

open(summary_path, "w") do io
    println(io, "# OOS forecast summary — ROM1 vs Switching Surrogate")
    println(io, "")
    println(io, "_Generated $(now())_")
    println(io, "")
    println(io, "## Headline finding")
    println(io, "")
    if covid_ratio < 1.0
        @printf(io, "- **%s window**: switching model beats ROM1, RMSE ratio %.2f (%.0f%% reduction).\n",
                window_label, covid_ratio, 100*(1-covid_ratio))
    else
        @printf(io, "- **%s window**: ROM1 wins, switching-model RMSE ratio %.2f (%.0f%% worse than linear).\n",
                window_label, covid_ratio, 100*(covid_ratio-1))
    end
    if full_ratio < 1.0
        @printf(io, "- **Full OOS (%d..%d)**: switching model wins overall, ratio %.2f (%.0f%% reduction).\n",
                t_pre + 1, eval_end, full_ratio, 100*(1-full_ratio))
    else
        @printf(io, "- **Full OOS (%d..%d)**: ROM1 wins overall, switching ratio %.2f (%.0f%% worse).\n",
                t_pre + 1, eval_end, full_ratio, 100*(full_ratio-1))
    end
    if post_ratio < 1.0
        @printf(io, "- **Remaining OOS**: switching model wins, ratio %.2f.\n",
                post_ratio)
    else
        @printf(io, "- **Remaining OOS**: ROM1 wins, switching ratio %.2f.\n",
                post_ratio)
    end
    println(io, "")
    println(io, "## Artefacts")
    println(io, "")
    println(io, "- LaTeX table: `$tex_path`")
    println(io, "- Markdown per-obs table: `$md_path`")
    println(io, "- Raw innovations: `.local_artifacts/oos_forecast/oos_innovations_$(tag).jls`")
    println(io, "- Linear chain: `$linear_chain`")
    println(io, "- Surrogate chain: `$surrogate_chain`")
    println(io, "")
    println(io, "## Setup")
    println(io, "")
    println(io, "- Estimation sample: first $t_pre periods, warm-started from the full-sample posterior mean.")
    println(io, "- Forecast method: 1-step-ahead conditional expectation with shocks set to zero.")
    println(io, "  * Linear: Durbin–Koopman Kalman filter, innovation v_t = y_t − C·μ_{t|t-1}.")
    println(io, "  * Switching: ROM1 inversion filter up to t-1 with gate-conditional NN correction; " *
                "predict obs_t at state_{t-1} with ε_t=0 via ROM1 (+ NN on gate periods).")
    println(io, "- Chain lengths: linear n=$(lin_res["n_samples"]), surrogate n=$(sur_res["n_samples"]).")
    @printf(io, "- Gate share over full OOS: hard %.1f%%; soft mean %.1f%%.\n",
            100 * gate_share_full,
            isfinite(soft_gate_share_full) ? 100 * soft_gate_share_full : NaN)
    @printf(io, "- Gate share over %s window: hard %.1f%%; soft mean %.1f%%.\n",
            window_label,
            100 * gate_share_covid,
            isfinite(soft_gate_share_covid) ? 100 * soft_gate_share_covid : NaN)
    println(io, "- Gate padding: `k_pre=$gate_k_pre`, `k_post=$gate_k_post`, `min_len=$gate_min_len`.")
    println(io, "")
    println(io, "## Caveats")
    println(io, "")
    println(io, "- Chains are warm-started from the full-sample posterior mean; effective sample size is limited by draw count.")
    println(io, "- Gate thresholds `tau_eps`, `tau_y` are inherited from the stored calibration payload supplied to the evaluator; this is a diagnostic gate-design check, not a separately optimized forecasting rule.")
    println(io, "- Point forecasts use the posterior mean. Predictive-density scoring (log-score, CRPS) would be a natural next step.")
    println(io, "- RMSE is computed on model-space innovations in the same units as the observables (mostly quarter-on-quarter growth rates, inflation, and annualised rates per HLT convention).")
end

println("\n" * "=" ^ 72)
println("OOS EVALUATION COMPLETE")
println("Finished: $(now())")
println("=" ^ 72)
println("  Summary:    $summary_path")
println("  LaTeX:      $tex_path")
println("  Markdown:   $md_path")
println()
println("  HEADLINE:")
@printf("  Full OOS aggregate RMSE ratio (switch/linear): %.3f\n", full_ratio)
@printf("  %s window aggregate RMSE ratio: %.3f\n", window_label, covid_ratio)
@printf("  Remaining OOS aggregate RMSE ratio: %.3f\n", post_ratio)
