#!/usr/bin/env julia
# Debug script to decompose surrogate inversion filter LL
using Serialization, Statistics, LinearAlgebra
using MacroModelling, AxisKeys

repo_root = normpath(joinpath(@__DIR__, ".."))
cd(repo_root)

# Load data
payload = deserialize(".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls")
obs_data = payload["obs_data"]
obs_sigma_base = payload["obs_sigma"]
s0 = payload["s0"]
shock_sigmas = payload["shock_sigmas"]
theta_names = payload["theta_names"]
observables = payload["observables"]
state_names = payload["state_names"]

# Load Kalman posterior as init
d = deserialize(".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_chain_2000_seed99.jls")
θ_kalman = Float64.(vec(mean(d["chain"], dims=1)))
d_obs, T_obs = size(obs_data)

# Load model
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)

theta_param_idx = Int.(indexin(theta_names, mm_model.parameters))
base_params = copy(mm_model.parameter_values)
obs_idx = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names, mm_model.var))

# 1. Kalman LL at posterior mean
obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)
ll_kalman = MacroModelling.linear_model_loglik_per_period(
    mm_model, obs_data_ka, θ_kalman, theta_names;
    model_parameter_names=mm_model.parameters, base_parameters=base_params,
    theta_idx=theta_param_idx, algorithm=:first_order, filter=:kalman,
    on_failure_loglikelihood=-1e12, presample_periods=0,
    initial_covariance=:theoretical, verbose=false)
println("Kalman LL: $(round(sum(ll_kalman), digits=1))")
println("  Per-period: mean=$(round(mean(ll_kalman), digits=2)), range=[$(round(minimum(ll_kalman), digits=1)), $(round(maximum(ll_kalman), digits=1))]")

# 2. Load surrogate + build predict fns
surr = MacroModelling.load_hlt_surrogate_bundle(
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls")
frozen = surr.frozen
val_rmse = surr.payload["validation_rmse"]
obs_sigma = max.(obs_sigma_base, val_rmse[1:d_obs] .* 2.0)
obs_sigma = max.(obs_sigma, 0.1)
println("\nobs_sigma_base: $(round.(obs_sigma_base, sigdigits=3))")
println("val_rmse[1:7]:  $(round.(val_rmse[1:d_obs], sigdigits=3))")
println("obs_sigma:      $(round.(obs_sigma, sigdigits=3))")

rom_pred = RomPredictor(mm_model, 1, :baseline, false, Int[], base_params,
                        nothing, nothing, state_idx, obs_idx)
Main.ensure_rom_cache!(rom_pred, Float64[])

rom_full = (s, e, t) -> Main.rom_predict(rom_pred, s, e, t)
sur_resid_full = (s, e, t) -> predict_frozen(frozen, vcat(s, e, t))
sur_step = (s, e, t) -> MacroModelling.predict_additive_residual(
    rom_full, sur_resid_full, s, e, t, d_obs; allow_full_residual=true)
rom_only = (s, e, t) -> MacroModelling.predict_from_full(rom_full, s, e, t, d_obs)

# FIX: obs-only correction (no state drift)
sur_resid_obs = (s, e, t) -> predict_frozen(frozen, vcat(s, e, t))[1:d_obs]
sur_step_obs = (s, e, t) -> MacroModelling.predict_additive_residual(
    rom_full, sur_resid_obs, s, e, t, d_obs; allow_full_residual=false)

# 3. ROM1-only inversion LL (without surrogate correction)
println("\n--- ROM1-only inversion filter ---")
t0 = time()
ll_rom, shocks_rom = MacroModelling.inversion_loglik_per_period(
    rom_only, s0, θ_kalman, obs_data, obs_sigma, shock_sigmas;
    maxit=10, tol=1e-6, lambda=1e-4)
t1 = time() - t0
println("ROM1 inversion LL: $(round(sum(ll_rom), digits=1)) ($(round(t1*1000, digits=0)) ms)")
println("  Per-period: mean=$(round(mean(ll_rom), digits=1)), range=[$(round(minimum(ll_rom), digits=1)), $(round(maximum(ll_rom), digits=1))]")

# 4. Surrogate inversion LL
println("\n--- Surrogate inversion filter ---")
t0 = time()
ll_surr, shocks_surr = MacroModelling.inversion_loglik_per_period(
    rom_only, s0, θ_kalman, obs_data, obs_sigma, shock_sigmas;
    eval_predict_fn=sur_step, maxit=10, tol=1e-6, lambda=1e-4)
t1 = time() - t0
println("Surrogate inversion LL: $(round(sum(ll_surr), digits=1)) ($(round(t1*1000, digits=0)) ms)")
println("  Per-period: mean=$(round(mean(ll_surr), digits=1)), range=[$(round(minimum(ll_surr), digits=1)), $(round(maximum(ll_surr), digits=1))]")
println("  Worst 5 periods: $(round.(sort(ll_surr)[1:5], digits=1))")

# Check which periods are worst
worst_idx = sortperm(ll_surr)[1:5]
println("  Worst period indices: $worst_idx")
for t in worst_idx
    println("    t=$t: surr_ll=$(round(ll_surr[t], digits=1)), rom_ll=$(round(ll_rom[t], digits=1)), kalman_ll=$(round(ll_kalman[t], digits=1))")
end

# 5. Gate mask
gate_calib = deserialize(".local_artifacts/hlt_18param_realdata/gate_calibration.jls")
e_stat = Float64.(gate_calib["e_stats"])
f_stat = Float64.(gate_calib["f_stats"])
tau_eps = gate_calib["tau_eps"]
tau_y = gate_calib["tau_y"]
base_mask = (e_stat .> tau_eps) .| (f_stat .> tau_y)
gate_mask = MacroModelling.apply_gate_padding(base_mask, 4, 8, 4)
n_gate = count(gate_mask)
not_gate = .!gate_mask
println("\nGate: $n_gate/$T_obs periods use surrogate")

# 6. Combined
surr_gate_ll = sum(ll_surr[gate_mask])
kalman_rest_ll = sum(ll_kalman[not_gate])
println("\nCombined LL:")
println("  Surrogate (gate periods):   $(round(surr_gate_ll, digits=1))")
println("  Kalman (non-gate periods):  $(round(kalman_rest_ll, digits=1))")
println("  Total:                      $(round(surr_gate_ll + kalman_rest_ll, digits=1))")
println("  Pure Kalman (all periods):  $(round(sum(ll_kalman), digits=1))")

# 7. OBS-ONLY surrogate correction (FIX for state drift)
println("\n\n=== OBS-ONLY SURROGATE CORRECTION (FIX) ===")
println("--- At Kalman posterior mean ---")
ll_fix, _ = MacroModelling.inversion_loglik_per_period(
    rom_only, s0, θ_kalman, obs_data, obs_sigma, shock_sigmas;
    eval_predict_fn=sur_step_obs, maxit=10, tol=1e-6, lambda=1e-4)
println("Surrogate LL (obs-only): $(round(sum(ll_fix), digits=1))")
println("  Per-period: mean=$(round(mean(ll_fix), digits=1)), range=[$(round(minimum(ll_fix), digits=1)), $(round(maximum(ll_fix), digits=1))]")
println("  Worst 5: $(round.(sort(ll_fix)[1:5], digits=1))")

fix_gate_ll = sum(ll_fix[gate_mask])
fix_combined = fix_gate_ll + sum(ll_kalman[not_gate])
println("\nCombined (obs-only fix):")
println("  Surrogate (gate):  $(round(fix_gate_ll, digits=1))")
println("  Kalman (rest):     $(round(sum(ll_kalman[not_gate]), digits=1))")
println("  Total:             $(round(fix_combined, digits=1))")
println("  Pure Kalman:       $(round(sum(ll_kalman), digits=1))")

# 8. Also try at calibrated theta
println("\n--- At calibrated theta ---")
θ_calib = base_params[theta_param_idx]
ll_kalman_c = MacroModelling.linear_model_loglik_per_period(
    mm_model, obs_data_ka, θ_calib, theta_names;
    model_parameter_names=mm_model.parameters, base_parameters=base_params,
    theta_idx=theta_param_idx, algorithm=:first_order, filter=:kalman,
    on_failure_loglikelihood=-1e12, presample_periods=0,
    initial_covariance=:theoretical, verbose=false)
println("Kalman LL (calib): $(round(sum(ll_kalman_c), digits=1))")

ll_fix_c, _ = MacroModelling.inversion_loglik_per_period(
    rom_only, s0, θ_calib, obs_data, obs_sigma, shock_sigmas;
    eval_predict_fn=sur_step_obs, maxit=10, tol=1e-6, lambda=1e-4)
println("Surrogate LL (obs-only, calib): $(round(sum(ll_fix_c), digits=1))")
println("  Per-period: mean=$(round(mean(ll_fix_c), digits=1)), range=[$(round(minimum(ll_fix_c), digits=1)), $(round(maximum(ll_fix_c), digits=1))]")

fix_gate_c = sum(ll_fix_c[gate_mask])
fix_combined_c = fix_gate_c + sum(ll_kalman_c[not_gate])
println("Combined (obs-only, calib): $(round(fix_combined_c, digits=1))")

surr_gate_c = sum(ll_surr_c[gate_mask])
kalman_rest_c = sum(ll_kalman_c[not_gate])
println("Combined (calib): $(round(surr_gate_c + kalman_rest_c, digits=1))")
