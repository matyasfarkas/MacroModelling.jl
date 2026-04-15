#!/usr/bin/env julia
# Bootstrap particle filter degeneracy demo — runs BPF on HLT model with real US data.
# Demonstrates ESS collapse during crisis episodes.
#
# Usage:
#   julia --project=. scripts/pf_degeneracy_demo.jl \
#       --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls \
#       [--gate-calibration=.local_artifacts/hlt_18param_realdata/gate_calibration.jls] \
#       [--seed=42]

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, quantile
using Printf, Dates, MacroModelling

function parse_kv_string(args, key, default)
    for arg in args
        startswith(arg, "$key=") && return split(arg, "=", limit=2)[2]
    end
    return default
end
parse_kv_int(args, key, default) = parse(Int, parse_kv_string(args, key, string(default)))

data_path = parse_kv_string(ARGS, "--data", "")
gate_path = parse_kv_string(ARGS, "--gate-calibration", "")
seed      = parse_kv_int(ARGS, "--seed", 42)
data_path == "" && error("Usage: julia pf_degeneracy_demo.jl --data=<payload.jls> [--gate-calibration=<gate.jls>]")

# --- Load data payload + model ---
println("=" ^ 72)
println("PARTICLE FILTER DEGENERACY DEMO — $(now())")
println("=" ^ 72)

payload = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data     = payload["obs_data"]
obs_sigma    = payload["obs_sigma"]
s0           = payload["s0"]
shock_sigmas = payload["shock_sigmas"]
theta_names  = payload["theta_names"]
observables  = payload["observables"]
state_names  = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
d_obs, T_obs = size(obs_data)
println("  Obs: $d_obs, T: $T_obs, Structural shocks: $(count(shock_sigmas .> 0))/$(length(shock_sigmas))")

repo_root = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))

mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
obs_idx   = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names, mm_model.var))

rom_predictor = RomPredictor(mm_model, 1, :baseline, false, Int[],
                              copy(mm_model.parameter_values), nothing, nothing,
                              state_idx, obs_idx)
ensure_rom_cache!(rom_predictor, Float64[])

predict_fn = (state, shock, theta) -> MacroModelling.predict_from_full(
    (s, e, th) -> rom_predict(rom_predictor, s, e, th), state, shock, theta, d_obs)

include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
baseline_vals = get_phase1_18param_baseline()
theta_calib = Float64[get(baseline_vals, n, NaN) for n in theta_names]

obs_test, _ = predict_fn(Float64.(s0), zeros(length(shock_sigmas)), theta_calib)
println("  ROM1 sanity: obs[1:3] = $(round.(obs_test[1:min(3,d_obs)], sigdigits=5))")

# --- Run bootstrap PF at multiple particle counts ---
println("\n--- Running bootstrap particle filter ---")
particle_counts = [100, 500, 1000]
results = Dict{Int, NamedTuple}()

for Np in particle_counts
    t0 = time()
    res = MacroModelling.bootstrap_particle_filter(
        predict_fn, Float64.(s0), theta_calib, obs_data, obs_sigma, shock_sigmas;
        n_particles=Np, seed=seed, resample_threshold=0.5)
    elapsed = time() - t0
    results[Np] = res
    @printf("  N=%4d  LL=%9.1f  mean_ESS=%6.1f  min_ESS=%5.1f  time=%.1fs\n",
            Np, res.ll_total, mean(res.ess_per_period), minimum(res.ess_per_period), elapsed)
end

# --- Summary table ---
println("\n" * "=" ^ 72)
println("RESULTS SUMMARY")
@printf("  %6s  %10s  %8s  %8s  %8s  %12s\n",
        "N_p", "Total LL", "Mean ESS", "Min ESS", "Med ESS", "ESS<5 periods")
println("  " * "-" ^ 62)
for Np in particle_counts
    r = results[Np]
    @printf("  %6d  %10.1f  %8.1f  %8.1f  %8.1f  %8d/%d\n",
            Np, r.ll_total, mean(r.ess_per_period), minimum(r.ess_per_period),
            quantile(r.ess_per_period, 0.5), count(r.ess_per_period .< 5.0), T_obs)
end

# --- Worst ESS episodes for largest particle count ---
r_max = results[maximum(particle_counts)]
worst_idx = sortperm(r_max.ess_per_period)[1:min(15, T_obs)]
println("\nWORST ESS EPISODES (N=$(maximum(particle_counts)))")
@printf("  %6s  %8s  %10s  %10s\n", "Period", "ESS", "Max Weight", "LL_t")
println("  " * "-" ^ 40)
for t in worst_idx
    @printf("  %6d  %8.1f  %10.4f  %10.2f\n",
            t, r_max.ess_per_period[t], r_max.max_weight_per_period[t], r_max.ll_per_period[t])
end

# --- Gate vs non-gate comparison ---
if gate_path != "" && isfile(gate_path)
    println("\nGATE vs NON-GATE ESS COMPARISON")
    gate_calib = MacroModelling.load_hlt_gate_calibration(gate_path)
    if haskey(gate_calib, "e_stats") && haskey(gate_calib, "f_stats")
        e_stat = vec(Float64.(gate_calib["e_stats"]))
        f_stat = vec(Float64.(gate_calib["f_stats"]))
        base_mask = (e_stat .> gate_calib["tau_eps"]) .| (f_stat .> gate_calib["tau_y"])
        gate_mask = MacroModelling.apply_gate_padding(base_mask, 4, 8, 4)
        for Np in particle_counts
            r = results[Np]
            ess_g = r.ess_per_period[gate_mask]
            ess_c = r.ess_per_period[.!gate_mask]
            @printf("  N=%4d  Gate(%3d): mean_ESS=%6.1f min=%5.1f | Calm(%3d): mean_ESS=%6.1f min=%5.1f\n",
                    Np, count(gate_mask), mean(ess_g), minimum(ess_g),
                    count(.!gate_mask), mean(ess_c), length(ess_c) > 0 ? minimum(ess_c) : NaN)
        end
    else
        println("  Gate calibration missing e_stats/f_stats -- skipping")
    end
else
    println("\n  (No gate calibration -- skipping gate/non-gate comparison)")
end

println("\nDONE — $(now())")
