#!/usr/bin/env julia
# Post-process B3 SEP posterior validation results
# Run after sep_posterior_validation.jl completes
#
# Usage: julia --project=. scripts/process_b3_results.jl [path_to_results.jls]

using Serialization
import Statistics: mean, std, cor, quantile
using Printf

results_path = length(ARGS) >= 1 ? ARGS[1] :
    ".local_artifacts/hlt_18param_realdata/sep_posterior_validation_20draws.jls"

if !isfile(results_path)
    println("Results file not found: $results_path")
    println("The B3 validation run may still be in progress.")
    exit(1)
end

r = deserialize(results_path)

println("=" ^ 72)
println("B3 SEP POSTERIOR VALIDATION — POST-PROCESSING")
println("=" ^ 72)
println("  Source: $results_path")
println("  Timestamp: $(r["timestamp"])")
println("  Elapsed: $(round(r["elapsed_seconds"] / 3600, digits=2)) hours")
println("  Valid: $(r["n_valid"]) / $(r["n_valid"] + r["n_failed"])")

println("\n--- Key Results ---")
@printf("  ROM1 joint LL (fixed):  %.1f\n", r["ll_rom1_total"])
@printf("  Shock penalty:          %.1f\n", r["shock_penalty"])

valid = .!isnan.(r["ll_sep"])
if any(valid)
    @printf("  SEP LL (mean ± std):    %.1f ± %.1f\n", mean(r["ll_sep"][valid]), std(r["ll_sep"][valid]))
    @printf("  Kalman LL (mean ± std): %.1f ± %.1f\n", mean(r["ll_kalman"][valid]), std(r["ll_kalman"][valid]))

    println("\n--- Per-Variable RMSE: ROM1 vs SEP ---")
    obs_names = ["dy", "dc", "dinve", "labobs", "pinfobs", "dwobs", "robs"]
    obs_sigma = r["obs_sigma"]
    d_obs = length(obs_sigma)
    for j in 1:min(d_obs, length(obs_names))
        r1 = mean(r["rmse_rom1_sep"][j, valid])
        ratio = r1 / obs_sigma[j]
        @printf("  %-12s  RMSE=%.4f   RMSE/σ_y=%.2f\n", obs_names[j], r1, ratio)
    end
    avg_r1 = mean(r["rmse_rom1_sep"][:, valid])
    @printf("  %-12s  RMSE=%.4f\n", "AVERAGE", avg_r1)

    gap_nn_sep = abs.(r["ll_nn"][valid] .- r["ll_sep"][valid])
    println("\n--- LL Gaps ---")
    @printf("  |LL_ROM1+NN - LL_SEP| mean:   %.1f nats\n", mean(gap_nn_sep))
    @printf("  ROM1-SEP obs gap (per period): %.2f nats\n",
        (mean(r["ll_sep"][valid]) - r["ll_rom1_total"]) / size(r["rmse_rom1_sep"], 2) * -1)
end

println("\n" * "=" ^ 72)
println("FOR PAPER UPDATE:")
println("  Replace '1.03' with '$(round(mean(r["rmse_rom1_sep"][:, valid]), sigdigits=3))' if different")
println("  Replace '2.91' (labor) with '$(round(mean(r["rmse_rom1_sep"][4, valid]), sigdigits=3))' if different")
println("  Replace '0.92' (robs) with '$(round(mean(r["rmse_rom1_sep"][7, valid]), sigdigits=3))' if different")
println("=" ^ 72)
