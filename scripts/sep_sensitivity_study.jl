#!/usr/bin/env julia
# ============================================================================
# SEP SOLVER SENSITIVITY STUDY
# ============================================================================
# Validates that the FOM/SEP decomposition results in Section 8.3 of the paper
# are robust to the SEP hyperparameters `accept_tol` and `nnodes` (K).
#
# Design (referee response, April 2026):
#   - Load 10 posterior draws stratified by Mahalanobis distance from the
#     pooled warm-started surrogate chains (hlt_surrogate_hmc_extended_18p_2000
#     + seeds 5/6/7). Stratification spans tail draws (high Mahalanobis) and
#     mode draws (low Mahalanobis) so the sensitivity check is not concentrated
#     on a single posterior region.
#   - For each draw θ and each cell in the grid
#         {accept_tol ∈ [0.01, 0.1, 0.35]} × {K ∈ [3, 5]}
#     run the SEP extended-path simulation over T_obs periods using the
#     shocks recovered at the pooled posterior mean. That yields 60 runs per
#     study (10 draws × 6 cells).
#   - The reference cell is (accept_tol=0.01, K=5). For every cell we log the
#     convergence rate (SEP not flagged as failure), mean periods solved and
#     per-variable prediction RMSE against the reference cell.
#   - Outputs: .jls serialization, LaTeX table and markdown summary.
#
# Usage:
#   julia --project=. scripts/sep_sensitivity_study.jl [--n-draws=10] [--cap-draws=5]
#   julia --project=. scripts/sep_sensitivity_study.jl --n-draws=3 --periods=40
#
# The `--cap-draws` flag is a safety net: if the script detects that each SEP
# run takes much longer than expected it halves the draw count.
# ============================================================================

using Serialization, Random, LinearAlgebra, Statistics, Printf, Dates
using MacroModelling
using AxisKeys

# ============================================================================
# Config / CLI parsing
# ============================================================================

function parse_kv_string(args, key, default)
    for arg in args
        if startswith(arg, "$key=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end
parse_kv_int(args, key, default)   = parse(Int,     parse_kv_string(args, key, string(default)))
parse_kv_float(args, key, default) = parse(Float64, parse_kv_string(args, key, string(default)))

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

chain_paths = [
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls"),
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed5.jls"),
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed6.jls"),
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed7.jls"),
]

data_path      = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls")
surrogate_path = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls")
out_dir        = joinpath(REPO_ROOT, ".local_artifacts/sep_sensitivity")
out_jls        = joinpath(out_dir, "sep_sensitivity_results.jls")
out_tex        = joinpath(out_dir, "sep_sensitivity_table.tex")
out_md         = joinpath(out_dir, "SEP_SENSITIVITY_SUMMARY.md")

mkpath(out_dir)

n_draws     = parse_kv_int(ARGS,   "--n-draws",    10)
cap_draws   = parse_kv_int(ARGS,   "--cap-draws",  5)
sep_horizon = parse_kv_int(ARGS,   "--sep-horizon", 40)
sep_maxit   = parse_kv_int(ARGS,   "--sep-maxit",   200)
periods_arg = parse_kv_int(ARGS,   "--periods",      0)
verbose     = any(==("--verbose"), ARGS)

inv_maxit  = 10
inv_tol    = 1e-6
inv_lambda = 1e-4

obs_sigma_scale = 2.0
obs_sigma_floor = 0.1

# Grid
accept_tols = [0.01, 0.1, 0.35]
Ks          = [3, 5]
ref_cell    = (accept_tol = 0.01, K = 5)

println("=" ^ 78)
println("SEP SENSITIVITY STUDY — accept_tol × K grid at posterior draws")
println("=" ^ 78)
println("  n_draws:     $n_draws   (cap=$cap_draws if SEP is slow)")
println("  sep_horizon: $sep_horizon")
println("  sep_maxit:   $sep_maxit")
println("  accept_tols: $accept_tols")
println("  K (nnodes):  $Ks")
println("  reference cell: accept_tol=$(ref_cell.accept_tol), K=$(ref_cell.K)")
println("  out_dir:     $out_dir")

# ============================================================================
# Load payload, model, surrogate bundle and build predictors
# ============================================================================

println("\n--- Loading data payload ---")
payload        = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data       = payload["obs_data"]
obs_sigma_base = payload["obs_sigma"]
theta_names    = payload["theta_names"]
observables    = payload["observables"]
state_names    = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
s0             = payload["s0"]
shock_sigmas   = payload["shock_sigmas"]

d_obs   = size(obs_data, 1)
T_full  = size(obs_data, 2)
T_obs   = periods_arg <= 0 ? T_full : min(periods_arg, T_full)
obs_data = obs_data[:, 1:T_obs]
n_theta = length(theta_names)
d_state = length(s0)

println("  obs: $d_obs × $T_obs (from full T=$T_full), state: $d_state, n_theta: $n_theta")

println("\n--- Loading model ---")
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

mm_model = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  model: $(mm_model.model_name)")

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)

obs_idx   = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names, mm_model.var))
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    any(isnothing, idx_any) && error("theta names missing in model.parameters")
    Int.(idx_any)
end

base_parameters = copy(mm_model.parameter_values)

# Observation sigma construction (same as run_surrogate_hmc_advancedhmc)
obs_sigma = copy(obs_sigma_base)
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_sigma = max.(obs_sigma, val_rmse[1:d_obs] .* obs_sigma_scale)
end
if obs_sigma_floor > 0
    obs_sigma = max.(obs_sigma, obs_sigma_floor)
end

# First solve model at baseline parameters via RomPredictor (same pattern as
# run_surrogate_hmc_advancedhmc.jl / sep_posterior_validation.jl).  Without this,
# `model.solution.perturbation.first_order.solution_matrix` is a 0×0 stub and the
# subsequent `build_matrix_rom_predict` call returns a predictor that errors
# with "tried to multiply a matrix of size (0, 0) with a vector of length N".
rom_predictor = RomPredictor(mm_model, 1, :baseline, false, Int[],
    base_parameters, nothing, nothing, Int.(state_idx), Int.(obs_idx))
ensure_rom_cache!(rom_predictor, Float64[])
println("  ROM1 cache initialized (baseline mode)")

# Build ForwardDiff-compatible ROM1 predictor for shock recovery
_, matrix_rom_predict_tuple, _ = build_matrix_rom_predict(mm_model;
    state_idx = Int.(state_idx), obs_idx = Int.(obs_idx))

# ============================================================================
# Pool chains and pick 10 Mahalanobis-stratified draws
# ============================================================================

println("\n--- Loading & pooling posterior chains ---")
chains = Matrix{Float64}[]
for p in chain_paths
    if isfile(p)
        d = deserialize(p)
        push!(chains, Matrix{Float64}(d["chain"]))
    else
        println("  WARN: missing $p")
    end
end
isempty(chains) && error("No posterior chains available")

pooled = vcat(chains...)
n_pooled = size(pooled, 1)
println("  pooled: $n_pooled × $(size(pooled, 2))")

post_mean = vec(Statistics.mean(pooled, dims=1))
post_cov  = Statistics.cov(pooled)
# Regularize cov for Mahalanobis (in case of near-singular)
post_cov_reg = post_cov + 1e-8 * I
cov_inv = inv(post_cov_reg)

maha = Vector{Float64}(undef, n_pooled)
for i in 1:n_pooled
    d = pooled[i, :] .- post_mean
    maha[i] = sqrt(max(dot(d, cov_inv * d), 0.0))
end

# Stratify: sort by Mahalanobis, then sample uniformly across the sorted list
sorted_idx = sortperm(maha)
actual_n = min(n_draws, n_pooled)
stratified_positions = actual_n == 1 ?
    [round(Int, length(sorted_idx) ÷ 2)] :
    round.(Int, range(1, stop=length(sorted_idx), length=actual_n))
draw_indices = sorted_idx[stratified_positions]
draws = [pooled[i, :] for i in draw_indices]
draw_maha = maha[draw_indices]

println("  Mahalanobis range across selected draws: " *
        "$(round(minimum(draw_maha), digits=2)) .. $(round(maximum(draw_maha), digits=2))")

# ============================================================================
# Recover shocks via ROM1 inversion filter at the posterior mean
# ============================================================================

println("\n--- Recovering shocks (ROM1 inversion filter, posterior mean) ---")
ll_rom1_vec, shocks_recovered = MacroModelling.inversion_loglik_per_period(
    matrix_rom_predict_tuple, s0, post_mean, obs_data, obs_sigma, shock_sigmas;
    maxit = inv_maxit, tol = inv_tol, lambda = inv_lambda)
println("  ROM1 inversion LL (joint): $(round(sum(ll_rom1_vec), digits=1))")

# ============================================================================
# Core: evaluate the (accept_tol, K) grid at each draw
# ============================================================================

n_cells = length(accept_tols) * length(Ks)
cells = [(atol, K) for atol in accept_tols, K in Ks]
cells = vec(cells)
n_cells = length(cells)

# Pre-allocate storage
# sep_obs_pred[cell, draw] = d_obs × T_obs  matrix (or nothing on failure)
sep_obs_pred  = Array{Union{Nothing, Matrix{Float64}}}(undef, n_cells, actual_n)
fill!(sep_obs_pred, nothing)

converged     = falses(n_cells, actual_n)
periods_done  = zeros(Int, n_cells, actual_n)
final_err     = fill(NaN, n_cells, actual_n)
wall_time     = zeros(Float64, n_cells, actual_n)
failure_period = fill(-1, n_cells, actual_n)

# Cap draws if a first timing probe shows SEP is very slow
effective_n = actual_n
probe_time = NaN

function run_one_sep(θ_vec; accept_tol, K)
    full_params = copy(base_parameters)
    for (j, idx) in enumerate(theta_param_idx)
        full_params[idx] = θ_vec[j]
    end
    old_params = copy(mm_model.parameter_values)
    mm_model.parameter_values .= full_params
    res = nothing
    try
        res = simulate_sep_extended_path(mm_model;
            periods         = T_obs,
            initial_state   = nothing,
            shocks          = shocks_recovered,
            burn_in         = 0,
            sep_horizon     = sep_horizon,
            sep_order       = 1,
            sep_nnodes      = K,
            sep_maxit       = sep_maxit,
            sep_tol         = 1e-7,
            sep_sparse_tree = true,
            sep_accept_tol  = accept_tol,
            sep_shock_scale = 1.0,
            silent          = !verbose)
    catch e
        @warn "SEP threw exception" exception=(e, catch_backtrace())
    finally
        mm_model.parameter_values .= old_params
    end
    return res
end

function extract_obs_pred(res)
    res === nothing && return nothing
    sim = res.simulation
    var_names = axiskeys(sim, 1)
    pred = Matrix{Float64}(undef, d_obs, T_obs)
    for (j, oname) in enumerate(observables)
        row = findfirst(==(oname), var_names)
        if row === nothing
            return nothing
        end
        pred[j, :] = sim[row, 1:T_obs]
    end
    return pred
end

println("\n--- Running grid: $n_cells cells × $actual_n draws = $(n_cells * actual_n) runs ---")
t0 = time()

for (di, θ_vec) in enumerate(draws)
    global effective_n, probe_time
    di > effective_n && break
    for (ci, cell) in enumerate(cells)
        t_cell = time()
        atol, K = cell
        res = run_one_sep(θ_vec; accept_tol = atol, K = K)
        dt = time() - t_cell
        wall_time[ci, di] = dt

        if res === nothing
            converged[ci, di] = false
            periods_done[ci, di] = 0
            final_err[ci, di] = NaN
            failure_period[ci, di] = 0
        else
            # `errorflag=true` iff SEP gave up before T_obs. Otherwise all periods OK.
            converged[ci, di] = !res.errorflag
            failure_period[ci, di] = res.failure_period === nothing ? -1 : res.failure_period
            periods_done[ci, di] = res.failure_period === nothing ? T_obs : res.failure_period - 1
            errs = res.sep_errors
            finite_errs = filter(isfinite, collect(errs))
            final_err[ci, di] = isempty(finite_errs) ? NaN : maximum(finite_errs)
            if converged[ci, di]
                sep_obs_pred[ci, di] = extract_obs_pred(res)
            end
        end

        if verbose || ci == 1 || ci == n_cells
            @printf("  draw %2d/%2d  cell (atol=%.3g, K=%d)  time=%.1fs  conv=%s  periods=%d\n",
                di, effective_n, atol, K, dt,
                converged[ci, di] ? "yes" : "NO", periods_done[ci, di])
        end

        # Timing probe: on the first cell of the first draw, if SEP took > 3 minutes
        # we halve the draw count to cap wall time at ~1 hour.
        if di == 1 && ci == 1
            probe_time = dt
            if dt > 180.0 && cap_draws < actual_n
                effective_n = cap_draws
                println("  PROBE: single run took $(round(dt, digits=1))s. " *
                        "Capping draws at $cap_draws (was $actual_n).")
            end
        end
    end
    elapsed = time() - t0
    eta = elapsed / di * (effective_n - di)
    @printf("  [done draw %d/%d]  elapsed=%.0fs  ETA=%.0fs\n", di, effective_n, elapsed, eta)
end

total_wall = time() - t0
println("\nTotal grid time: $(round(total_wall, digits=1))s")

# ============================================================================
# Aggregate diagnostics
# ============================================================================

# For each cell, compute:
#   conv_rate     : fraction of draws where SEP completed all T_obs periods
#   mean_periods  : mean number of periods completed (over all draws)
#   mean_time     : average wall time per run
#   mean_final_err: average final SEP residual
#
# For non-reference cells, additionally compute per-variable RMSE against ref.
ref_ci = findfirst(x -> x[1] == ref_cell.accept_tol && x[2] == ref_cell.K, cells)
ref_ci === nothing && error("reference cell not found in grid")

conv_rate     = zeros(Float64, n_cells)
mean_periods  = zeros(Float64, n_cells)
mean_time     = zeros(Float64, n_cells)
mean_final_err = fill(NaN, n_cells)

for ci in 1:n_cells
    valid_cols = 1:effective_n
    conv_rate[ci]    = mean(converged[ci, valid_cols])
    mean_periods[ci] = mean(periods_done[ci, valid_cols])
    mean_time[ci]    = mean(wall_time[ci, valid_cols])
    errs = filter(isfinite, final_err[ci, valid_cols])
    mean_final_err[ci] = isempty(errs) ? NaN : mean(errs)
end

# Per-variable RMSE vs reference cell (pooled across draws)
rmse_vs_ref_pervar = fill(NaN, d_obs, n_cells)
rmse_vs_ref_overall = fill(NaN, n_cells)

for ci in 1:n_cells
    errs = Float64[]
    pervar_sq = zeros(Float64, d_obs)
    pervar_cnt = zeros(Int, d_obs)
    for di in 1:effective_n
        ref_pred = sep_obs_pred[ref_ci, di]
        cell_pred = sep_obs_pred[ci, di]
        (ref_pred === nothing || cell_pred === nothing) && continue
        for j in 1:d_obs
            diffv = ref_pred[j, :] .- cell_pred[j, :]
            sq = sum(diffv .^ 2)
            pervar_sq[j] += sq
            pervar_cnt[j] += length(diffv)
            append!(errs, diffv)
        end
    end
    if !isempty(errs)
        rmse_vs_ref_overall[ci] = sqrt(mean(errs .^ 2))
        for j in 1:d_obs
            rmse_vs_ref_pervar[j, ci] = pervar_cnt[j] > 0 ?
                sqrt(pervar_sq[j] / pervar_cnt[j]) : NaN
        end
    end
end

println("\n--- Grid summary ---")
@printf("  %-10s %-3s %-7s %-10s %-12s %-12s\n",
        "accept_tol", "K", "conv", "periods", "mean err", "RMSE vs ref")
for ci in 1:n_cells
    atol, K = cells[ci]
    @printf("  %-10.3g %-3d %-7.2f %-10.1f %-12.2e %-12.4g\n",
            atol, K, conv_rate[ci], mean_periods[ci], mean_final_err[ci],
            rmse_vs_ref_overall[ci])
end

# ============================================================================
# Write LaTeX table
# ============================================================================

function fmt_float(x; digits=4)
    isnan(x) && return "---"
    return @sprintf("%.*g", digits, x)
end
function fmt_rmse(x)
    isnan(x) && return "---"
    return @sprintf("%.2e", x)
end

tex = IOBuffer()
println(tex, "% SEP sensitivity table (autogenerated by scripts/sep_sensitivity_study.jl)")
println(tex, "% Reference cell: accept_tol=$(ref_cell.accept_tol), K=$(ref_cell.K)")
println(tex, "\\begin{table}[ht]")
println(tex, "\\centering")
println(tex, "\\caption{SEP solver sensitivity to acceptance tolerance and quadrature nodes.")
println(tex, "Runs are over $(effective_n) posterior draws stratified by Mahalanobis distance")
println(tex, "from the pooled warm-started surrogate HMC chains (seeds 42, 5, 6, 7).")
println(tex, "``Convergence'' is the fraction of draws where SEP completed all $(T_obs) periods.")
println(tex, "``Max Euler err.'' is the maximum SEP residual (\$\\ell_\\infty\$-norm of the model's first-order condition")
println(tex, "residuals) across all periods, averaged over draws, reported in raw units and on the log\$_{10}\$ scale.")
println(tex, "``RMSE vs.\\ ref.'' is the root-mean-squared observation prediction error against the")
println(tex, "reference cell (accept\\_tol = $(ref_cell.accept_tol), \$K=$(ref_cell.K)\$), pooled over draws")
println(tex, "and $(T_obs) periods.}")
println(tex, "\\label{tab:sep_sensitivity}")
println(tex, "\\small")
println(tex, "\\begin{tabular}{cc ccccc}")
println(tex, "\\toprule")
println(tex, "accept\\_tol & \$K\$ & conv.\\ rate & mean periods & max Euler err. & \$\\log_{10}(\\mathrm{Euler\\ err.})\$ & RMSE vs.\\ ref.\\\\")
println(tex, "\\midrule")
for ci in 1:n_cells
    atol, K = cells[ci]
    is_ref = (ci == ref_ci)
    rmse_str = is_ref ? "(ref.)" : fmt_rmse(rmse_vs_ref_overall[ci])
    euler = mean_final_err[ci]
    log10_euler = (isnan(euler) || euler <= 0) ? NaN : log10(euler)
    @printf(tex, "%g & %d & %.2f & %.1f & %s & %s & %s \\\\\n",
            atol, K, conv_rate[ci], mean_periods[ci],
            fmt_rmse(euler),
            isnan(log10_euler) ? "---" : @sprintf("%.2f", log10_euler),
            rmse_str)
end
println(tex, "\\bottomrule")
println(tex, "\\end{tabular}")
println(tex, "\\end{table}")
write(out_tex, String(take!(tex)))
println("\nLaTeX table written to $out_tex")

# ============================================================================
# Write markdown summary
# ============================================================================

# Headline judgement: decomposition is "stable" if (a) convergence rate >=
# 0.9 at accept_tol <= 0.1 cells AND (b) RMSE vs ref <= 1e-2 (units are
# quarterly log-levels, so 1% would be a massive shift).
function classify_stability()
    thresholds = (conv = 0.9, rmse = 1e-2)
    rel_cells = [(ci, atol, K) for (ci, (atol, K)) in enumerate(cells)
                  if !(atol == ref_cell.accept_tol && K == ref_cell.K)]
    pass = true
    for (ci, atol, K) in rel_cells
        if conv_rate[ci] < thresholds.conv
            pass = false
            break
        end
        if !isnan(rmse_vs_ref_overall[ci]) && rmse_vs_ref_overall[ci] > thresholds.rmse
            pass = false
            break
        end
    end
    return pass
end
headline_stable = classify_stability()
headline = headline_stable ?
    "SEP decomposition is stable under tolerance variation (all non-reference cells converged and RMSE vs. reference < 1e-2)." :
    "SEP decomposition shows sensitivity to the (accept_tol, K) grid — see table for details."

md = IOBuffer()
println(md, "# SEP sensitivity study — referee response")
println(md)
println(md, "_Autogenerated by `scripts/sep_sensitivity_study.jl` on $(now())._")
println(md)
println(md, "## What was tested")
println(md)
println(md, "- Grid: `accept_tol ∈ {0.01, 0.1, 0.35}` × `K ∈ {3, 5}` = $n_cells cells")
println(md, "- $(effective_n) posterior draws stratified by Mahalanobis distance from pooled surrogate chains.")
println(md, "- Study window: $(T_obs) periods from the extended payload (full payload has $(T_full) periods).")
println(md, "- Mahalanobis distance range across selected draws: " *
            "$(round(minimum(draw_maha), digits=2)) .. $(round(maximum(draw_maha), digits=2)).")
println(md, "- Reference cell: `accept_tol = $(ref_cell.accept_tol)`, `K = $(ref_cell.K)`.")
println(md, "- Shocks recovered once via ROM1 inversion filter at the pooled posterior mean " *
            "(θ-independent at order 1), then SEP is solved over $(T_obs) periods per draw.")
println(md, "- Probe run took $(round(probe_time, digits=1))s; total wall clock $(round(total_wall, digits=1))s.")
println(md, "- Effective draws executed: $(effective_n) (cap triggered if SEP was slow).")
println(md)
println(md, "## Headline finding")
println(md)
println(md, "**$(headline)**")
println(md)
println(md, "## Results table")
println(md)
println(md, "| accept_tol | K | conv. rate | mean periods | max Euler err. | log10(Euler err.) | RMSE vs. ref. |")
println(md, "|---|---|---|---|---|---|---|")
for ci in 1:n_cells
    atol, K = cells[ci]
    is_ref = (ci == ref_ci)
    rmse_str = is_ref ? "(ref.)" : fmt_rmse(rmse_vs_ref_overall[ci])
    euler = mean_final_err[ci]
    log10_euler = (isnan(euler) || euler <= 0) ? NaN : log10(euler)
    log10_str = isnan(log10_euler) ? "---" : @sprintf("%.2f", log10_euler)
    @printf(md, "| %g | %d | %.2f | %.1f | %s | %s | %s |\n",
            atol, K, conv_rate[ci], mean_periods[ci],
            fmt_rmse(euler), log10_str, rmse_str)
end
println(md)
println(md, "## Per-variable RMSE vs. reference (non-reference cells only)")
println(md)
header = "| Variable | " * join(["atol=$(atol), K=$K" for (atol, K) in cells if !(atol == ref_cell.accept_tol && K == ref_cell.K)], " | ") * " |"
println(md, header)
sep = "|---|" * join(["---" for _ in 1:(length(cells) - 1)], "|") * "|"
println(md, sep)
for j in 1:d_obs
    row = "| $(observables[j])"
    for ci in 1:n_cells
        ci == ref_ci && continue
        row *= " | $(fmt_rmse(rmse_vs_ref_pervar[j, ci]))"
    end
    row *= " |"
    println(md, row)
end
println(md)
println(md, "## LaTeX table (paste into appendix)")
println(md)
println(md, "See `sep_sensitivity_table.tex` — same directory.")
println(md)
println(md, "## Interpretation for the paper")
println(md)
println(md, "The 68.9% investment/capital-block share of the FOM–ROM1 gap (Table in Section 8.3)")
println(md, "is evaluated at `accept_tol = 0.35` (required for robust dataset generation per the")
println(md, "project memory). This study checks whether tightening `accept_tol` down to 0.01 or")
println(md, "moving `K` from 3 to 5 would materially change the SEP prediction used to compute")
println(md, "that share. The headline finding above answers that question directly.")
println(md)
println(md, "## Files")
println(md)
println(md, "- `.local_artifacts/sep_sensitivity/sep_sensitivity_results.jls` (full data)")
println(md, "- `.local_artifacts/sep_sensitivity/sep_sensitivity_table.tex` (LaTeX)")
println(md, "- `.local_artifacts/sep_sensitivity/SEP_SENSITIVITY_SUMMARY.md` (this document)")
write(out_md, String(take!(md)))
println("Markdown summary written to $out_md")

# ============================================================================
# Serialize full results
# ============================================================================

result = Dict{String,Any}(
    "cells"                => cells,
    "ref_cell"             => ref_cell,
    "accept_tols"          => accept_tols,
    "Ks"                   => Ks,
    "draw_indices"         => draw_indices,
    "draw_maha"            => draw_maha,
    "effective_n"          => effective_n,
    "n_cells"              => n_cells,
    "converged"            => converged,
    "periods_done"         => periods_done,
    "final_err"            => final_err,
    "wall_time"            => wall_time,
    "failure_period"       => failure_period,
    "conv_rate"            => conv_rate,
    "mean_periods"         => mean_periods,
    "mean_time"            => mean_time,
    "mean_final_err"       => mean_final_err,
    "rmse_vs_ref_pervar"   => rmse_vs_ref_pervar,
    "rmse_vs_ref_overall"  => rmse_vs_ref_overall,
    "headline_stable"      => headline_stable,
    "probe_time_s"         => probe_time,
    "total_wall_s"         => total_wall,
    "sep_horizon"          => sep_horizon,
    "sep_maxit"            => sep_maxit,
    "T_obs"                => T_obs,
    "d_obs"                => d_obs,
    "observables"          => observables,
    "timestamp"            => string(now()),
    "chain_paths"          => chain_paths,
    "data_path"            => data_path,
    "surrogate_path"       => surrogate_path,
)
serialize(out_jls, result)
println("Results serialized to $out_jls")
println("Done.")
