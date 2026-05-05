#!/usr/bin/env julia
# ============================================================================
# OOS FORECAST — Truncated-Sample Estimation
# ============================================================================
#
# Truncates the extended 18-parameter real-data payload to the first T_pre
# quarters and re-runs the linear Kalman HMC chain or surrogate inversion-filter
# HMC chain on this restricted sample. Default T_pre=244 gives 1959Q1..2019Q4.
# For the quiet-sample OOS exercise, use --t-pre=144 --tag=quiet_1994
# (1959Q1..1994Q4) and evaluate through 2007Q4 with oos_forecast_evaluate.jl.
#
# Usage:
#   julia --project=. scripts/oos_forecast_run.jl --mode=linear --samples=500 \
#         [--adapt=150]
#   julia --project=. scripts/oos_forecast_run.jl --mode=surrogate --samples=300 \
#         [--adapt=150]
# ============================================================================

using Serialization, Random, Dates, Statistics, Printf

repo_root = normpath(joinpath(@__DIR__, ".."))
cd(repo_root)

# --- CLI ---
function parse_kv(args, key, default)
    for a in args
        if startswith(a, "$key=")
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

mode      = parse_kv(ARGS, "--mode", "")
n_samples = parse(Int, parse_kv(ARGS, "--samples", "500"))
n_adapt   = parse(Int, parse_kv(ARGS, "--adapt", "150"))
t_pre     = parse(Int, parse_kv(ARGS, "--t-pre", "244"))  # 1959Q1..2019Q4
seed      = parse(Int, parse_kv(ARGS, "--seed", "42"))
tag       = parse_kv(ARGS, "--tag", t_pre == 244 ? "preCOVID" : "T$(t_pre)")

mode in ("linear", "surrogate") ||
    error("Usage: --mode=linear|surrogate [--samples=500] [--adapt=150] [--t-pre=244]")

# --- Paths ---
payload_full = ".local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls"
payload_pre  = ".local_artifacts/oos_forecast/hlt_real_data_payload_$(tag)_18p.jls"
gate_full    = ".local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p.jls"
gate_pre     = ".local_artifacts/oos_forecast/gate_calibration_$(tag)_18p.jls"
linear_init  = ".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls"
surrogate_init = ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls"
surrogate_bundle = ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls"

mkpath(".local_artifacts/oos_forecast")

# --- Truncate payload (once) ---
if !isfile(payload_pre)
    println("[oos_forecast_run] Building truncated payload (tag=$tag, T_pre=$t_pre) ...")
    p = deserialize(payload_full)
    p_pre = deepcopy(p)
    T_full = size(p["obs_data"], 2)
    t_pre <= T_full || error("t_pre=$t_pre exceeds T_full=$T_full")
    p_pre["obs_data"] = p["obs_data"][:, 1:t_pre]
    if haskey(p, "shocks")
        p_pre["shocks"] = p["shocks"][:, 1:t_pre]
    end
    p_pre["sample_idx"] = p["sample_idx"][1:t_pre]
    p_pre["sample_idx_requested"] = p["sample_idx_requested"][1:t_pre]
    p_pre["generated_periods"] = t_pre
    p_pre["requested_periods"] = t_pre
    p_pre["oos_split_note"] = "Truncated to first $t_pre periods for OOS forecast tag=$tag"
    serialize(payload_pre, p_pre)
    println("[oos_forecast_run] Wrote $payload_pre with obs_data size=$(size(p_pre["obs_data"]))")
else
    println("[oos_forecast_run] Truncated payload already exists: $payload_pre")
end

# --- Truncate gate calibration (surrogate mode only) ---
if mode == "surrogate" && !isfile(gate_pre)
    println("[oos_forecast_run] Building truncated gate calibration (tag=$tag, T_pre=$t_pre) ...")
    g = deserialize(gate_full)
    g_pre = deepcopy(g)
    # Truncate per-period statistics
    for key in ("e_stats", "f_stats")
        if haskey(g, key) && length(g[key]) >= t_pre
            g_pre[key] = g[key][1:t_pre]
        end
    end
    # Keep tau_eps / tau_y from full-sample (reflects the researcher's calibration
    # as of the data cutoff — we freeze gate thresholds for the OOS exercise).
    g_pre["oos_split_note"] = "e_stats/f_stats truncated to first $t_pre periods for tag=$tag; tau thresholds preserved from full-sample calibration"
    serialize(gate_pre, g_pre)
    println("[oos_forecast_run] Wrote $gate_pre")
end

# --- Run the requested estimation as a subprocess (isolates its CLI parsing) ---
out_path = if mode == "linear"
    ".local_artifacts/oos_forecast/hlt_linear_hmc_$(tag)_$(n_samples).jls"
else
    ".local_artifacts/oos_forecast/hlt_surrogate_hmc_$(tag)_$(n_samples).jls"
end

cmd = if mode == "linear"
    `julia --project=. scripts/run_linear_hmc_advancedhmc.jl
        --data=$payload_pre
        --out=$out_path
        --samples=$n_samples
        --adapt=$n_adapt
        --seed=$seed
        --init-from=$linear_init`
else
    `julia --project=. scripts/run_surrogate_hmc_advancedhmc.jl
        --surrogate=$surrogate_bundle
        --data=$payload_pre
        --out=$out_path
        --samples=$n_samples
        --adapt=$n_adapt
        --seed=$seed
        --init-from=$surrogate_init
        --gate-calibration=$gate_pre
        --gate-mode=soft`
end

println("\n[oos_forecast_run] Launching $mode HMC chain:")
println("  Command: $cmd")
println("  Started: $(now())")

t0 = time()
run(cmd)
elapsed_h = (time() - t0) / 3600

println("\n[oos_forecast_run] $mode chain complete")
println("  Elapsed: $(round(elapsed_h, digits=2)) hours")
println("  Output:  $out_path")
println("  Finished: $(now())")
