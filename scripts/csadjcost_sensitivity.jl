#!/usr/bin/env julia
# ============================================================================
# S''(1) SENSITIVITY ANALYSIS — Investment Adjustment Cost Curvature
# ============================================================================
#
# Tests the robustness of the 69% investment dominance finding to
# alternative calibrations of csadjcost (= S''(1)). At each value,
# runs SEP simulations and computes the equation-block decomposition
# of the FOM-ROM1 gap.
#
# The invisibility property S(1) = S'(1) = 0 holds regardless of S''(1),
# so the *relative* investment share should be stable even as the absolute
# gap magnitude scales with curvature.
#
# Usage:
#   julia --project=. scripts/csadjcost_sensitivity.jl
#   julia --project=. scripts/csadjcost_sensitivity.jl --out=.local_artifacts/csadjcost_sensitivity/
#   julia --project=. scripts/csadjcost_sensitivity.jl --n-thetas=10 --sim-periods=40
#
# ============================================================================

using MacroModelling
using Random
using Serialization
using Dates
import Statistics: mean, median, std, quantile
using Distributions
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

# ============================================================================
# CLI Arguments
# ============================================================================

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

n_thetas       = parse_arg_int(ARGS, "--n-thetas", 25)
sep_horizon    = parse_arg_int(ARGS, "--sep-horizon", 10)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", 0.35)
sep_maxit      = parse_arg_int(ARGS, "--sep-maxit", 80)
sep_tol        = parse_arg_float(ARGS, "--sep-tol", 1e-5)
burn_in        = parse_arg_int(ARGS, "--burn-in", 20)
sim_periods    = parse_arg_int(ARGS, "--sim-periods", 40)
seed0          = parse_arg_int(ARGS, "--seed", 42)
chain_path     = parse_arg_string(ARGS, "--chain", "")
output_dir     = parse_arg_string(ARGS, "--out",
                    joinpath(REPO_ROOT, ".local_artifacts", "csadjcost_sensitivity"))
verbose        = "--verbose" in ARGS

csadjcost_values_str = parse_arg_string(ARGS, "--csadjcost-values", "")
if csadjcost_values_str != ""
    csadjcost_values = parse.(Float64, split(csadjcost_values_str, ","))
else
    csadjcost_values = [2.0, 4.0, 6.0144, 8.0, 10.0]
end

mkpath(output_dir)
rng = MersenneTwister(seed0)

println("=" ^ 78)
println("S''(1) SENSITIVITY ANALYSIS — INVESTMENT ADJUSTMENT COST CURVATURE")
println("Started: $(now())")
println("=" ^ 78)
println("  csadjcost values: $csadjcost_values")
println("  Thetas:           $n_thetas")
println("  Sim periods:      $sim_periods + $burn_in burn-in")
println("  Output:           $output_dir")
flush(stdout)

# ============================================================================
# Load Model
# ============================================================================

println("\nLoading HLT OBC model...")
const HLT = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
MacroModelling.solve!(HLT, silent = true)
println("  Model: $(HLT.model_name), $(length(HLT.var)) variables, $(length(HLT.exo)) shocks")

# Observable and state indices
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = Int.(indexin(observables, HLT.var))
@assert all(!isnothing, obs_idx)

state_idx = sort(unique(vcat(HLT.timings.past_not_future_and_mixed_idx,
                             HLT.timings.future_not_past_and_mixed_idx)))

d_obs = length(obs_idx)
d_state = length(state_idx)
d_eps = length(HLT.exo)
println("  d_obs=$d_obs, d_state=$d_state, d_eps=$d_eps")

# csadjcost parameter index
csadjcost_pidx = findfirst(==(:csadjcost), HLT.parameters)
csadjcost_pidx === nothing && error("csadjcost not found in model parameters")
println("  csadjcost param index: $csadjcost_pidx (baseline = $(HLT.parameter_values[csadjcost_pidx]))")

# ============================================================================
# Load Chain & Select Thetas
# ============================================================================

specs = get_phase1_18param_specs()
theta_names = [s.name for s in specs]

if chain_path == ""
    # Try surrogate chain first, fall back to linear
    chain_path = joinpath(REPO_ROOT,
        ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls")
    if !isfile(chain_path)
        chain_path = joinpath(REPO_ROOT,
            ".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls")
    end
end

println("\nLoading chain: $chain_path")
chain_data = Serialization.deserialize(chain_path)
chain_matrix = chain_data["chain"]
chain_theta_names = haskey(chain_data, "theta_names") ? Symbol.(chain_data["theta_names"]) : theta_names

# Align parameter ordering
if chain_theta_names != theta_names
    println("  Reindexing parameters...")
    reindex = [findfirst(==(n), chain_theta_names) for n in theta_names]
    @assert all(!isnothing, reindex)
    chain_matrix = chain_matrix[:, reindex]
end

n_draws_total = size(chain_matrix, 1)
println("  Chain: $(n_draws_total) draws × $(size(chain_matrix, 2)) parameters")

# Stratified sampling by Mahalanobis distance
param_means = vec(mean(chain_matrix, dims=1))
param_stds  = vec(std(chain_matrix, dims=1))
distances = zeros(n_draws_total)
for i in 1:n_draws_total
    d = 0.0
    for j in 1:size(chain_matrix, 2)
        d += ((chain_matrix[i, j] - param_means[j]) / max(param_stds[j], 1e-12))^2
    end
    distances[i] = sqrt(d / size(chain_matrix, 2))
end

sorted_idx = sortperm(distances)
stride = max(1, n_draws_total ÷ n_thetas)
selected_idx = sorted_idx[1:stride:min(n_draws_total, stride*n_thetas)]
selected_idx = selected_idx[1:min(n_thetas, length(selected_idx))]
theta_grid = chain_matrix[selected_idx, :]
actual_n_thetas = size(theta_grid, 1)
println("  Selected $actual_n_thetas representative thetas")
flush(stdout)

# ============================================================================
# Shock Drawing
# ============================================================================

function draw_shocks(rng::AbstractRNG, model, total_periods::Int, shock_scale::Float64)
    shock_names_local = model.exo
    nshocks = length(shock_names_local)
    shocks = zeros(nshocks, total_periods)
    obc_mask = contains.(string.(shock_names_local), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    isempty(structural_idx) && return shocks
    sigmas = fill(shock_scale, length(structural_idx))
    shocks[structural_idx, :] .= Diagonal(sigmas) * randn(rng, length(structural_idx), total_periods)
    return shocks
end

# ============================================================================
# Equation Block Definitions
# ============================================================================

obs_block_map = Dict(
    "Output/Resource"     => [:dy],
    "Consumption/Euler"   => [:dc],
    "Investment/Capital"  => [:dinve],
    "Labor Market"        => [:labobs],
    "Price Phillips"      => [:pinfobs],
    "Wage Phillips"       => [:dwobs],
    "Taylor Rule"         => [:robs],
)

block_names = ["Output/Resource", "Consumption/Euler", "Investment/Capital",
               "Labor Market", "Price Phillips", "Wage Phillips", "Taylor Rule"]

function compute_block_shares(delta_obs::Matrix{Float64}, obs_syms::Vector{Symbol})
    n_samples = size(delta_obs, 2)
    n_samples == 0 && return Dict(bn => 0.0 for bn in block_names)

    total_sq = mean(sum(delta_obs.^2, dims=1))
    total_sq < 1e-30 && return Dict(bn => 0.0 for bn in block_names)

    shares = Dict{String, Float64}()
    for bname in block_names
        vars = obs_block_map[bname]
        positions = [findfirst(==(v), obs_syms) for v in vars]
        filter!(!isnothing, positions)
        if isempty(positions)
            shares[bname] = 0.0
            continue
        end
        block_sq = mean(sum(delta_obs[positions, :].^2, dims=1))
        shares[bname] = block_sq / total_sq
    end
    return shares
end

# ============================================================================
# Main Loop: Simulate + Decompose at Each csadjcost Value
# ============================================================================

println("\n" * "=" ^ 78)
println("SIMULATING ACROSS csadjcost VALUES")
println("=" ^ 78)

n_values = length(csadjcost_values)
csadjcost_results = Dict{Float64, Dict{String, Any}}()

for (vi, csadj) in enumerate(csadjcost_values)
    println("\n--- csadjcost = $csadj ($vi/$n_values) ---")
    flush(stdout)

    # Pre-allocate for this value
    max_samples = actual_n_thetas * sim_periods
    delta_obs_matrix = zeros(d_obs, max_samples)
    cursor = 0
    n_converged = 0

    for (ti, theta_row) in enumerate(eachrow(theta_grid))
        theta = collect(Float64, theta_row)

        # Set parameters including the csadjcost override
        params = Float64.(HLT.parameter_values)
        for (j, tname) in enumerate(theta_names)
            pidx = findfirst(==(tname), HLT.parameters)
            if pidx !== nothing
                params[pidx] = theta[j]
            end
        end
        params[csadjcost_pidx] = csadj

        # Build ROM1 cache for this theta + csadjcost
        local rom_cache
        try
            rom_cache = build_rom_cache(HLT, 1; params=params, use_obc=true)
        catch e
            verbose && println("  [theta $ti] ROM build failed: $e")
            continue
        end

        # Draw shocks (deterministic per theta/csadjcost for reproducibility)
        trajectory_seed = seed0 * 1000 + vi * 100 + ti
        total_periods = sim_periods + burn_in
        shocks = draw_shocks(MersenneTwister(trajectory_seed), HLT, total_periods, 1.0)

        # Write parameters to model for SEP solve
        try
            MacroModelling.write_parameters_input!(HLT, params, verbose = false)
        catch e
            verbose && println("  [theta $ti] Parameter write failed: $e")
            continue
        end

        # Run SEP simulation
        local res
        try
            res = MacroModelling.simulate_sep_extended_path(
                HLT;
                periods          = sim_periods,
                burn_in          = burn_in,
                sep_horizon      = sep_horizon,
                sep_order        = 1,
                sep_nnodes       = 3,
                sep_maxit        = sep_maxit,
                sep_tol          = sep_tol,
                sep_sparse_tree  = true,
                sep_linear_solver = :normal_equations,
                sep_stall_iters  = 25,
                sep_stall_rel_tol = 1e-4,
                sep_stall_abs_tol = 1e-10,
                sep_line_search  = true,
                sep_line_search_maxit = 6,
                sep_line_search_factor = 0.5,
                sep_line_search_min_alpha = 1e-4,
                sep_lm_lambda    = 1e-8,
                sep_lm_lambda_scale = 10.0,
                sep_lm_lambda_min = 1e-12,
                sep_lm_lambda_max = 1e4,
                sep_shock_scale  = 1.0,
                sep_accept_tol   = sep_accept_tol,
                shock_scaling    = :none,
                shocks           = shocks,
                random_seed      = trajectory_seed,
                silent           = true,
            )
        catch e
            if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                verbose && println("  [theta $ti] SEP failed (singular)")
                continue
            else
                rethrow()
            end
        end

        if res.errorflag
            sep_errors = hasproperty(res, :sep_errors) ? res.sep_errors : Float64[]
            valid_errors = filter(isfinite, sep_errors)
            if isempty(valid_errors) || !all(e -> e <= sep_accept_tol, valid_errors)
                verbose && println("  [theta $ti] SEP diverged")
                continue
            end
        end

        n_converged += 1
        sim = Array(res.simulation)
        sim_shocks = res.shocks

        # Compute ROM1 for each period and record delta
        T_avail = min(sim_periods, size(sim, 2) - 1)
        for t in 1:T_avail
            fom_obs = sim[obs_idx, t + 1]
            local rom_next
            try
                rom_next = rom_step_full(rom_cache, sim[:, t], sim_shocks[:, t])
            catch
                continue
            end
            rom1_obs = rom_next[obs_idx]
            delta = fom_obs .- rom1_obs

            cursor += 1
            if cursor > size(delta_obs_matrix, 2)
                delta_obs_matrix = hcat(delta_obs_matrix, zeros(d_obs, max_samples))
            end
            delta_obs_matrix[:, cursor] = delta
        end
    end

    # Trim to actual samples
    delta_obs = delta_obs_matrix[:, 1:cursor]
    n_samples = cursor

    # Compute decomposition
    shares = compute_block_shares(delta_obs, observables)

    # Mean absolute gap
    mean_gap = n_samples > 0 ? mean(sqrt.(sum(delta_obs.^2, dims=1))) : 0.0

    # Per-observable RMSE
    obs_rmse = n_samples > 0 ? [sqrt(mean(delta_obs[i, :].^2)) for i in 1:d_obs] : zeros(d_obs)

    csadjcost_results[csadj] = Dict(
        "shares" => shares,
        "n_samples" => n_samples,
        "n_converged" => n_converged,
        "mean_gap" => mean_gap,
        "obs_rmse" => obs_rmse,
    )

    @printf("  Samples: %d, Converged thetas: %d/%d\n", n_samples, n_converged, actual_n_thetas)
    @printf("  Mean |delta|: %.6f\n", mean_gap)
    for bn in block_names
        @printf("    %-22s  %5.1f%%\n", bn, shares[bn]*100)
    end
    flush(stdout)
end

# ============================================================================
# Save Results
# ============================================================================

results_path = joinpath(output_dir, "csadjcost_sensitivity.jls")
Serialization.serialize(results_path, Dict(
    "csadjcost_results" => csadjcost_results,
    "csadjcost_values" => csadjcost_values,
    "block_names" => block_names,
    "n_thetas" => actual_n_thetas,
    "sim_periods" => sim_periods,
    "chain_path" => chain_path,
    "timestamp" => now(),
))
println("\nResults saved: $results_path")

# ============================================================================
# Generate LaTeX Table
# ============================================================================

println("\n" * "=" ^ 78)
println("GENERATING LaTeX TABLE")
println("=" ^ 78)

latex_path = joinpath(output_dir, "csadjcost_sensitivity_table.tex")
open(latex_path, "w") do io
    println(io, "% S''(1) sensitivity analysis — equation-block decomposition")
    println(io, "% Generated: $(now())")
    println(io, raw"\begin{table}[htbp]")
    println(io, raw"\centering")
    println(io, raw"\caption{Equation-Block Decomposition: Sensitivity to $S''(1)$}")
    println(io, raw"\label{tab:csadjcost_sensitivity}")
    println(io, raw"\begin{tabular}{lcccccccc}")
    println(io, raw"\hline\hline")
    @printf(io, "\$S''(1)\$ & Invest & Cons & Output & Labor & Price & Wage & Taylor & Gap \\\\\n")
    println(io, raw"\hline")

    for csadj in sort(csadjcost_values)
        r = csadjcost_results[csadj]
        s = r["shares"]
        # Bold the baseline value
        prefix = csadj ≈ 6.0144 ? raw"\textbf{" : ""
        suffix = csadj ≈ 6.0144 ? "}" : ""
        @printf(io, "%s%.1f%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.4f \\\\\n",
            prefix, csadj, suffix,
            s["Investment/Capital"]*100,
            s["Consumption/Euler"]*100,
            s["Output/Resource"]*100,
            s["Labor Market"]*100,
            s["Price Phillips"]*100,
            s["Wage Phillips"]*100,
            s["Taylor Rule"]*100,
            r["mean_gap"],
        )
    end

    println(io, raw"\hline\hline")
    println(io, raw"\end{tabular}")
    println(io, raw"\begin{minipage}{0.92\textwidth}")
    println(io, raw"\footnotesize\textit{Notes:} Each column reports the share of the mean squared FOM-ROM1 gap attributable to the corresponding equation block, expressed as a percentage. ``Gap'' reports the root-mean-square observable gap in model units. The baseline calibration $S''(1) = 6.0$ is shown in bold. Shares sum to 100\% by construction (orthogonal decomposition). ", @sprintf("%d", actual_n_thetas), " representative parameter vectors drawn from the posterior; ", @sprintf("%d", sim_periods), " periods per trajectory at unit shock scale.")
    println(io, raw"\end{minipage}")
    println(io, raw"\end{table}")
end
println("  LaTeX table saved: $latex_path")

# ============================================================================
# Summary Table (console)
# ============================================================================

println("\n" * "=" ^ 78)
println("SUMMARY TABLE")
println("=" ^ 78)
@printf("%-10s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s\n",
    "S''(1)", "N", "Conv",
    "Invest", "Euler", "Output", "Labor", "Price", "Wage", "Taylor")
println("-" ^ 100)

for csadj in sort(csadjcost_values)
    r = csadjcost_results[csadj]
    s = r["shares"]
    @printf("%-10.4f  %6d  %4d/%d  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%\n",
        csadj, r["n_samples"], r["n_converged"], actual_n_thetas,
        s["Investment/Capital"]*100,
        s["Consumption/Euler"]*100,
        s["Output/Resource"]*100,
        s["Labor Market"]*100,
        s["Price Phillips"]*100,
        s["Wage Phillips"]*100,
        s["Taylor Rule"]*100,
    )
end
@printf("\nMean gap magnitude across S''(1) values:\n")
for csadj in sort(csadjcost_values)
    r = csadjcost_results[csadj]
    @printf("  S''(1) = %6.1f  →  mean |delta| = %.6f\n", csadj, r["mean_gap"])
end

println("\n" * "=" ^ 78)
println("DONE: $(now())")
println("=" ^ 78)
