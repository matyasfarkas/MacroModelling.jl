#!/usr/bin/env julia
# ============================================================================
# MERGE MONTE CARLO COVERAGE RESULTS — Parallel Batch Aggregation
# ============================================================================
#
# Merges mc_coverage_checkpoint.jls files from multiple parallel batch
# directories into a single aggregated result with coverage statistics,
# bias/RMSE diagnostics, and a LaTeX table.
#
# Usage:
#   julia --project=. scripts/merge_mc_coverage.jl \
#       --dirs=".local_artifacts/mc_cov_b1,.local_artifacts/mc_cov_b2" \
#       --out=.local_artifacts/monte_carlo_coverage_merged/ \
#       [--ci-level=0.90]
# ============================================================================

using Serialization, Statistics, Printf, Dates
import Distributions

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

dirs_str = parse_kv_string(ARGS, "--dirs", "")
out_dir  = parse_kv_string(ARGS, "--out", ".local_artifacts/monte_carlo_coverage_merged")
ci_level = parse(Float64, parse_kv_string(ARGS, "--ci-level", "0.90"))

if isempty(dirs_str)
    println("ERROR: --dirs is required. Provide comma-separated batch directories.")
    println("Usage: julia --project=. scripts/merge_mc_coverage.jl \\")
    println("    --dirs=\"dir1,dir2,dir3\" --out=output_dir [--ci-level=0.90]")
    exit(1)
end

dirs = filter(!isempty, split(dirs_str, ","))
alpha_lo = (1.0 - ci_level) / 2.0
alpha_hi = 1.0 - alpha_lo

println("=" ^ 78)
println("MERGE MONTE CARLO COVERAGE RESULTS")
println("Started: $(now())")
println("=" ^ 78)
println("  Batch directories: $(length(dirs))")
for d in dirs
    println("    - $d")
end
println("  Output directory:  $out_dir")
println("  CI level:          $(Int(ci_level * 100))%")

mkpath(out_dir)

# ============================================================================
# Step 1: Load and merge checkpoint files
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 1: Loading checkpoint files")
println("-" ^ 78)

merged = Dict{Int,Any}()
n_loaded = 0
n_duplicates = 0

for d in dirs
    global merged, n_loaded, n_duplicates
    ckpt_path = joinpath(d, "mc_coverage_checkpoint.jls")
    if !isfile(ckpt_path)
        @warn "Checkpoint not found: $ckpt_path — skipping."
        continue
    end
    local batch::Dict{Int,Any}
    try
        batch = deserialize(ckpt_path)
    catch e
        @warn "Failed to deserialize $ckpt_path: $e — skipping."
        continue
    end
    n_batch = length(batch)
    n_dup_batch = 0
    for (rep_id, result) in batch
        if haskey(merged, rep_id)
            n_duplicates += 1
            n_dup_batch += 1
            @warn "Duplicate replication ID $rep_id found in $d — skipping duplicate."
        else
            merged[rep_id] = result
        end
    end
    n_loaded += 1
    println("  Loaded $ckpt_path: $n_batch replications" *
            (n_dup_batch > 0 ? " ($n_dup_batch duplicates skipped)" : ""))
end

println("\n  Total checkpoints loaded: $n_loaded / $(length(dirs))")
println("  Total replications:       $(length(merged))")
if n_duplicates > 0
    println("  WARNING: $n_duplicates duplicate replication IDs encountered.")
end

if isempty(merged)
    println("ERROR: No replications loaded. Nothing to merge.")
    exit(1)
end

# ============================================================================
# Step 2: Filter successful replications and aggregate
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 2: Aggregating results")
println("-" ^ 78)

success_results = [merged[k] for k in sort(collect(keys(merged)))
                   if get(merged[k], "status", "") == "success"]
N_total  = length(merged)
N_success = length(success_results)

println("  Successful: $N_success / $N_total")
n_failed = N_total - N_success
if n_failed > 0
    println("  Failed:     $n_failed")
end

if N_success < 2
    println("ERROR: fewer than 2 successful replications. Cannot compute statistics.")
    exit(1)
end

# Infer parameter names and count from first successful result
first_res = success_results[1]
n_theta = length(first_res["theta_true"])

# The main MC script does not store theta_names in per-rep checkpoints,
# so use the canonical 18-parameter ordering from the estimation config.
# Fall back to generic names if the count does not match.
canonical_names = [:crhoa, :crhob, :crhog, :crhoqs, :crhopinf, :crhow, :crhoms,
                   :z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em,
                   :cprobp, :cindp, :curvp, :cprobw]

if n_theta == length(canonical_names)
    theta_names = canonical_names
else
    theta_names = [Symbol("theta_$i") for i in 1:n_theta]
    @warn "Parameter count ($n_theta) does not match canonical 18; using generic names."
end

println("  Parameters: $n_theta")
println("  Replication IDs: $(minimum(keys(merged))) .. $(maximum(keys(merged)))")

# Collect arrays
all_true    = hcat([r["theta_true"] for r in success_results]...)'     # (N_success, n_theta)
all_mean    = hcat([r["post_mean"]  for r in success_results]...)'
all_std     = hcat([r["post_std"]   for r in success_results]...)'
all_covered = hcat([r["covered"]    for r in success_results]...)'     # (N_success, n_theta) Bool
all_ess     = hcat([r["ess"]        for r in success_results]...)'
all_diverge = [r["n_divergent"]  for r in success_results]
all_accept  = [r["mean_accept"]  for r in success_results]

# Per-parameter statistics
coverage_rates = vec(mean(all_covered, dims=1))
mean_bias      = vec(mean(all_mean .- all_true, dims=1))
median_bias    = [median(all_mean[:, i] .- all_true[:, i]) for i in 1:n_theta]
rmse           = [sqrt(mean((all_mean[:, i] .- all_true[:, i]).^2)) for i in 1:n_theta]
mean_post_std  = vec(mean(all_std, dims=1))
mean_ess       = vec(mean(all_ess, dims=1))

# Relative bias
true_abs_mean = vec(mean(abs.(all_true), dims=1))
rel_bias      = mean_bias ./ max.(true_abs_mean, 1e-6)

# ============================================================================
# Step 3: Clopper-Pearson confidence intervals
# ============================================================================

function clopper_pearson_ci(k::Int, n::Int, alpha::Float64=0.05)
    if k == 0
        lo = 0.0
    else
        lo = quantile(Distributions.Beta(k, n - k + 1), alpha / 2)
    end
    if k == n
        hi = 1.0
    else
        hi = quantile(Distributions.Beta(k + 1, n - k), 1.0 - alpha / 2)
    end
    return lo, hi
end

# ============================================================================
# Step 4: Print console summary
# ============================================================================

println("\n" * "=" ^ 90)
@printf("  MERGED COVERAGE SUMMARY  (%d%% CI, %d successful replications)\n",
        Int(ci_level * 100), N_success)
println("=" ^ 90)
@printf("  %-12s %8s %8s %8s %8s %8s %8s %12s\n",
        "Parameter", "Cover", "CI_lo", "CI_hi", "Bias", "RMSE", "Post SD", "Mean ESS")
println("  " * "-" ^ 86)

for i in 1:n_theta
    k_covered = sum(all_covered[:, i])
    ci_lo, ci_hi = clopper_pearson_ci(k_covered, N_success)
    @printf("  %-12s %7.1f%% %7.1f%% %7.1f%% %+8.4f %8.4f %8.4f %8.0f\n",
            theta_names[i],
            100 * coverage_rates[i],
            100 * ci_lo,
            100 * ci_hi,
            mean_bias[i],
            rmse[i],
            mean_post_std[i],
            mean_ess[i])
end
println("  " * "-" ^ 86)
@printf("  %-12s %7.1f%%\n", "AVERAGE", 100 * mean(coverage_rates))
println("  " * "-" ^ 86)
println("  Total divergences: $(sum(all_diverge)) across $N_success reps ($(round(mean(all_diverge), digits=1)) per rep)")
println("  Mean acceptance:   $(round(mean(all_accept), digits=3))")

# ============================================================================
# Step 5: Generate LaTeX table
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 5: Generating LaTeX table")
println("-" ^ 78)

param_display = Dict(
    :crhoa => raw"\rho_a", :crhob => raw"\rho_b", :crhog => raw"\rho_g",
    :crhoqs => raw"\rho_{qs}", :crhopinf => raw"\rho_\pi", :crhow => raw"\rho_w",
    :crhoms => raw"\rho_{ms}",
    :z_ea => raw"\sigma_a", :z_eb => raw"\sigma_b", :z_eg => raw"\sigma_g",
    :z_eqs => raw"\sigma_{qs}", :z_epinf => raw"\sigma_\pi", :z_ew => raw"\sigma_w",
    :z_em => raw"\sigma_{ms}",
    :cprobp => raw"\xi_p", :cindp => raw"\iota_p",
    :curvp => raw"\varepsilon_p", :cprobw => raw"\xi_w",
)

# Parameter groups in display order
persistence_params = [:crhoa, :crhob, :crhog, :crhoqs, :crhopinf, :crhow, :crhoms]
volatility_params  = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em]
structural_params  = [:cprobp, :cindp, :curvp, :cprobw]

function write_param_rows(io, syms, theta_names, coverage_rates, all_covered,
                          mean_bias, rmse, mean_post_std, mean_ess, N_success,
                          param_display)
    for sym in syms
        i = findfirst(==(sym), theta_names)
        i === nothing && continue
        k_covered = sum(all_covered[:, i])
        ci_lo, ci_hi = clopper_pearson_ci(k_covered, N_success)
        dname = get(param_display, sym, string(sym))
        @printf(io, "\$%s\$ & %.1f & [%.1f, %.1f] & %+.4f & %.4f & %.4f & %.0f \\\\\n",
                dname, 100 * coverage_rates[i], 100 * ci_lo, 100 * ci_hi,
                mean_bias[i], rmse[i], mean_post_std[i], mean_ess[i])
    end
end

latex_path = joinpath(out_dir, "mc_coverage_merged_table.tex")

open(latex_path, "w") do io
    println(io, "% Merged Monte Carlo coverage: $(Int(ci_level*100))% CI, $N_success replications")
    println(io, "% Merged from $(length(dirs)) batch directories")
    println(io, "% Generated: $(now())")
    println(io, raw"\begin{table}[htbp]")
    println(io, raw"\centering")
    println(io, "\\caption{Monte Carlo coverage study: $(Int(ci_level*100))\\% credible intervals ($N_success replications, merged)}")
    println(io, raw"\label{tab:mc_coverage_merged}")
    println(io, raw"\begin{tabular}{lcccccc}")
    println(io, raw"\hline\hline")
    println(io, raw"Parameter & Coverage (\%) & 95\% CI & Bias & RMSE & Post.\ SD & ESS \\")
    println(io, raw"\hline")

    # Persistence
    println(io, raw"\multicolumn{7}{l}{\textit{Shock persistence}} \\")
    write_param_rows(io, persistence_params, theta_names, coverage_rates,
                     all_covered, mean_bias, rmse, mean_post_std, mean_ess,
                     N_success, param_display)

    # Volatility
    println(io, raw"\hline")
    println(io, raw"\multicolumn{7}{l}{\textit{Shock volatility}} \\")
    write_param_rows(io, volatility_params, theta_names, coverage_rates,
                     all_covered, mean_bias, rmse, mean_post_std, mean_ess,
                     N_success, param_display)

    # Structural
    println(io, raw"\hline")
    println(io, raw"\multicolumn{7}{l}{\textit{Structural}} \\")
    write_param_rows(io, structural_params, theta_names, coverage_rates,
                     all_covered, mean_bias, rmse, mean_post_std, mean_ess,
                     N_success, param_display)

    println(io, raw"\hline")
    @printf(io, "Average & %.1f & & & & & \\\\\n", 100 * mean(coverage_rates))
    println(io, raw"\hline\hline")
    println(io, raw"\end{tabular}")
    println(io, raw"\begin{minipage}{0.95\textwidth}")
    println(io, "\\footnotesize\\textit{Notes:} Coverage rates for $(Int(ci_level*100))\\% highest posterior density intervals from $N_success Monte Carlo replications ",
            "(merged from $(length(dirs)) parallel batches). ",
            "``95\\% CI'' gives the Clopper--Pearson confidence interval for the coverage rate. ",
            "``Bias'' is the average posterior mean minus the true value. ",
            "``RMSE'' is the root mean squared error of the posterior mean. ",
            "``Post.\\ SD'' is the average posterior standard deviation. ",
            "``ESS'' is the effective sample size (batch means estimator).")
    println(io, raw"\end{minipage}")
    println(io, raw"\end{table}")
end

println("  LaTeX table saved: $latex_path")

# ============================================================================
# Step 6: Save merged results
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 6: Saving merged results")
println("-" ^ 78)

final_results = Dict{String,Any}(
    "N_total"          => N_total,
    "N_success"        => N_success,
    "n_batches"        => length(dirs),
    "batch_dirs"       => dirs,
    "ci_level"         => ci_level,
    "theta_names"      => theta_names,
    "coverage_rates"   => coverage_rates,
    "mean_bias"        => mean_bias,
    "median_bias"      => median_bias,
    "rmse"             => rmse,
    "rel_bias"         => rel_bias,
    "mean_post_std"    => mean_post_std,
    "mean_ess"         => mean_ess,
    "all_true"         => all_true,
    "all_mean"         => all_mean,
    "all_std"          => all_std,
    "all_covered"      => all_covered,
    "all_diverge"      => all_diverge,
    "all_accept"       => all_accept,
    "per_rep_results"  => merged,
    "timestamp"        => string(now()),
)

results_path = joinpath(out_dir, "mc_coverage_merged_results.jls")
serialize(results_path, final_results)
println("  Results:  $results_path")

# Human-readable summary
summary_path = joinpath(out_dir, "mc_coverage_merged_summary.txt")
open(summary_path, "w") do io
    println(io, "MONTE CARLO COVERAGE — MERGED SUMMARY")
    println(io, "=" ^ 78)
    println(io, "Date:            $(now())")
    println(io, "Batches merged:  $(length(dirs))")
    for d in dirs
        println(io, "  - $d")
    end
    println(io, "Replications:    $N_total ($N_success successful)")
    println(io, "CI level:        $(Int(ci_level * 100))%")
    println(io, "")
    @printf(io, "%-12s %8s %8s %8s %8s %8s\n",
            "Parameter", "Cover%", "Bias", "RMSE", "PostSD", "ESS")
    println(io, "-" ^ 60)
    for i in 1:n_theta
        @printf(io, "%-12s %7.1f%% %+8.4f %8.4f %8.4f %8.0f\n",
                theta_names[i],
                100 * coverage_rates[i],
                mean_bias[i],
                rmse[i],
                mean_post_std[i],
                mean_ess[i])
    end
    println(io, "-" ^ 60)
    @printf(io, "%-12s %7.1f%%\n", "AVERAGE", 100 * mean(coverage_rates))
    println(io, "")
    println(io, "Total divergences: $(sum(all_diverge))")
    println(io, "Mean acceptance:   $(round(mean(all_accept), digits=3))")
end

println("  Summary:  $summary_path")

# ============================================================================
# Done
# ============================================================================

println("\n" * "=" ^ 78)
println("MERGE COMPLETE")
println("Finished: $(now())")
println("Replications: $N_success successful (from $(length(dirs)) batches)")
println("Mean coverage: $(round(100 * mean(coverage_rates), digits=1))%")
println("Output: $out_dir")
println("=" ^ 78)
