#!/usr/bin/env julia
# ============================================================================
# SYSTEMATIC SHOCK-SCALE GRID SIMULATION FOR HLT MODEL
# ============================================================================
#
# Maps SEP convergence frontier across shock scales and posterior parameter
# draws to determine the maximum reliable shock scale for surrogate training.
#
# Phase 1: Coarse convergence grid (12 shock scales × 30 thetas × 5 runs)
# Phase 2: Frontier refinement (fine grid near convergence cliff)
# Phase 3: Production dataset at optimal scale (optional)
#
# Usage:
#   julia --project=. scripts/shock_scale_grid_simulation.jl
#   julia --project=. scripts/shock_scale_grid_simulation.jl --phase=1
#   julia --project=. scripts/shock_scale_grid_simulation.jl --resume
#
# ============================================================================

using MacroModelling
using Random
using Serialization
using Dates
import Statistics: mean, median, std, quantile, var
using Distributions
using LinearAlgebra
using Printf
using DataFrames

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

# ============================================================================
# CLI Arguments
# ============================================================================

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

phase_max         = parse_arg_int(ARGS, "--phase", 2)
n_thetas          = parse_arg_int(ARGS, "--n-thetas", 10)
n_trajectories    = parse_arg_int(ARGS, "--n-trajectories", 2)
sep_horizon       = parse_arg_int(ARGS, "--sep-horizon", 10)
sep_accept_tol    = parse_arg_float(ARGS, "--sep-accept-tol", 0.35)
sep_maxit         = parse_arg_int(ARGS, "--sep-maxit", 80)
sep_tol           = parse_arg_float(ARGS, "--sep-tol", 1e-5)
burn_in           = parse_arg_int(ARGS, "--burn-in", 20)
sim_periods       = parse_arg_int(ARGS, "--sim-periods", 40)
seed0             = parse_arg_int(ARGS, "--seed", 42)
convergence_threshold = parse_arg_float(ARGS, "--convergence-threshold", 0.80)
chain_path        = parse_arg_string(ARGS, "--chain", "")
output_dir        = parse_arg_string(ARGS, "--output-dir",
                        joinpath(script_repo_root(), ".local_artifacts", "shock_scale_grid"))
do_resume         = "--resume" in ARGS
verbose           = "--verbose" in ARGS

# Phase 1 shock scales — configurable via CLI
shock_scales_str = parse_arg_string(ARGS, "--shock-scales", "")
if shock_scales_str != ""
    shock_scales_phase1 = parse.(Float64, split(shock_scales_str, ","))
else
    shock_scales_phase1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
end

# ============================================================================
# Setup
# ============================================================================

mkpath(output_dir)
rng = MersenneTwister(seed0)

println("=" ^ 78)
println("SHOCK-SCALE GRID SIMULATION")
println("Started: $(now())")
println("=" ^ 78)
println("  Phases:           1..$(phase_max)")
println("  Shock scales:     $(shock_scales_phase1)")
println("  Thetas:           $(n_thetas)")
println("  Trajectories:     $(n_trajectories) per (scale, theta)")
println("  SEP horizon:      $(sep_horizon)")
println("  SEP accept_tol:   $(sep_accept_tol)")
println("  Burn-in:          $(burn_in)")
println("  Sim periods:      $(sim_periods)")
println("  Conv. threshold:  $(convergence_threshold)")
println("  Output:           $(output_dir)")
println("  Resume:           $(do_resume)")
flush(stdout)

# ============================================================================
# Step 1: Load Model
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 1: Loading HLT OBC model")
println("-" ^ 78)

model = load_hlt_model(script_repo_root(), "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
println("  Model: $(model.model_name)")

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
hlt_param_names = model.parameters
base_values = copy(model.parameter_values)

# Parameter configuration
theta_names = get_parameter_names(:phase1_18params_narrow)
theta_idx = Int.(indexin(theta_names, hlt_param_names))
@assert all(!isnothing, theta_idx) "Missing parameters in model"
d_theta = length(theta_names)

shock_names = model.exo
obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
structural_idx = findall(!, obc_mask)
println("  Parameters: $(d_theta)")
println("  Structural shocks: $(length(structural_idx))")

# ============================================================================
# Step 2: Load Posterior Chain and Sample Thetas
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 2: Loading posterior chain and sampling thetas")
println("-" ^ 78)

# Find chain file
if chain_path == ""
    artifacts_dir = joinpath(script_repo_root(), ".local_artifacts", "hlt_18param_realdata")
    # Prefer HMC chains over MH chains
    for candidate in [
        "hlt_linear_hmc_chain_2000_seed99.jls",
        "hlt_linear_hmc_chain_1000.jls",
        "hlt_kalman_mh_chain_50k.jls",
        "hlt_kalman_mh_chain_500k.jls",
    ]
        p = joinpath(artifacts_dir, candidate)
        if isfile(p)
            global chain_path = p
            break
        end
    end
    if chain_path == ""
        error("No posterior chain found in $(artifacts_dir). Provide --chain=<path>.")
    end
end

println("  Chain file: $(basename(chain_path))")
chain_data = Serialization.deserialize(chain_path)

# Extract chain matrix: (n_draws, n_theta)
chain_matrix = if chain_data isa Dict
    chain_data["chain"]
else
    error("Unsupported chain format. Expected Dict with 'chain' key.")
end

chain_theta_names = if chain_data isa Dict && haskey(chain_data, "theta_names")
    Symbol.(chain_data["theta_names"])
else
    theta_names
end

n_draws_total = size(chain_matrix, 1)
println("  Chain draws: $(n_draws_total)")
println("  Chain params: $(chain_theta_names)")

# Verify parameter alignment
if chain_theta_names != theta_names
    println("  WARNING: Chain theta_names differ from config. Reindexing...")
    chain_reindex = Int[]
    for name in theta_names
        ci = findfirst(==(name), chain_theta_names)
        if ci === nothing
            error("Parameter $name not found in chain.")
        end
        push!(chain_reindex, ci)
    end
    chain_matrix = chain_matrix[:, chain_reindex]
end

# Stratified sampling of thetas: center (mode), 1σ boundary, 2σ boundary
function sample_stratified_thetas(chain::Matrix{Float64}, n_total::Int, rng::AbstractRNG)
    n_draws, n_params = size(chain)

    # Compute per-parameter quantiles
    param_means = vec(mean(chain, dims=1))
    param_stds  = vec(std(chain, dims=1))

    # Compute Mahalanobis-like distance for each draw from the posterior mean
    distances = zeros(n_draws)
    for i in 1:n_draws
        d = 0.0
        for j in 1:n_params
            d += ((chain[i, j] - param_means[j]) / max(param_stds[j], 1e-12))^2
        end
        distances[i] = sqrt(d / n_params)  # normalized distance
    end

    # Split into 3 strata
    n_center = n_total ÷ 3
    n_1sigma = n_total ÷ 3
    n_2sigma = n_total - n_center - n_1sigma

    # Center: draws with distance < 0.5 (near mode)
    center_mask = distances .< 0.5
    # 1σ: draws with 0.5 ≤ distance < 1.5
    sigma1_mask = (distances .>= 0.5) .& (distances .< 1.5)
    # 2σ: draws with distance ≥ 1.5
    sigma2_mask = distances .>= 1.5

    center_idx = findall(center_mask)
    sigma1_idx = findall(sigma1_mask)
    sigma2_idx = findall(sigma2_mask)

    println("  Strata sizes: center=$(length(center_idx)), 1σ=$(length(sigma1_idx)), 2σ=$(length(sigma2_idx))")

    selected = Int[]

    # Sample from each stratum (with fallback to full chain if stratum is too small)
    function sample_stratum(pool, n_needed)
        if length(pool) >= n_needed
            return pool[randperm(rng, length(pool))[1:n_needed]]
        else
            return pool  # take all available
        end
    end

    append!(selected, sample_stratum(center_idx, n_center))
    append!(selected, sample_stratum(sigma1_idx, n_1sigma))
    append!(selected, sample_stratum(sigma2_idx, n_2sigma))

    # Fill remainder from full chain if needed
    deficit = n_total - length(selected)
    if deficit > 0
        remaining = setdiff(1:n_draws, selected)
        extra = remaining[randperm(rng, length(remaining))[1:min(deficit, length(remaining))]]
        append!(selected, extra)
    end

    return chain[selected[1:min(n_total, length(selected))], :]
end

theta_grid = sample_stratified_thetas(chain_matrix, n_thetas, rng)
actual_n_thetas = size(theta_grid, 1)
println("  Selected $(actual_n_thetas) representative thetas")
flush(stdout)

# ============================================================================
# Step 3: Shock Drawing Utility
# ============================================================================

function draw_shocks(rng::AbstractRNG, model, total_periods::Int,
                     shock_scale::Float64)
    shock_names_local = model.exo
    nshocks = length(shock_names_local)
    shocks = zeros(nshocks, total_periods)
    obc_mask_local = contains.(string.(shock_names_local), "ᵒᵇᶜ")
    structural_idx_local = findall(!, obc_mask_local)
    if isempty(structural_idx_local)
        return shocks
    end
    sigmas = ones(length(structural_idx_local))
    for (i, idx) in enumerate(structural_idx_local)
        sigmas[i] = MacroModelling.sep_irf_shock_std(model, shock_names_local[idx])
    end
    sigmas .*= shock_scale
    shocks[structural_idx_local, :] .= Diagonal(sigmas) * randn(rng, length(structural_idx_local), total_periods)
    return shocks
end

# ============================================================================
# Step 4: Single Trajectory Runner
# ============================================================================

function run_single_trajectory(model, params::Vector{Float64}, shock_scale::Float64,
                               trajectory_seed::Int;
                               sep_horizon, sep_accept_tol, sep_maxit, sep_tol,
                               burn_in, sim_periods, verbose)
    total_periods = sim_periods + burn_in
    shocks = draw_shocks(MersenneTwister(trajectory_seed), model, total_periods, shock_scale)

    converged = false
    errorflag = true
    final_residual = NaN
    n_iterations = 0
    zlb_binding = false
    failure_period = 0
    sep_errors = Float64[]

    try
        res = MacroModelling.simulate_sep_extended_path(
            model;
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
            shock_scaling    = :parameter,
            shocks           = shocks,
            random_seed      = trajectory_seed,
            silent           = !verbose,
        )

        errorflag = res.errorflag
        failure_period = res.failure_period === nothing ? 0 : Int(res.failure_period)
        zlb_binding = hasproperty(res, :zlb_periods) && res.zlb_periods > 0

        if hasproperty(res, :sep_errors)
            sep_errors = res.sep_errors
            final_residual = isempty(sep_errors) ? NaN : maximum(filter(isfinite, sep_errors))
        end

        # Converged if no error, or if all residuals within accept_tol
        converged = !errorflag
        if !converged && !isempty(sep_errors)
            valid_errors = filter(isfinite, sep_errors)
            if !isempty(valid_errors) && all(e -> e <= sep_accept_tol, valid_errors)
                converged = true
            end
        end
    catch e
        if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
            converged = false
            errorflag = true
            failure_period = 1
        else
            rethrow()
        end
    end

    return (
        converged       = converged,
        errorflag       = errorflag,
        final_residual  = final_residual,
        failure_period  = failure_period,
        zlb_binding     = zlb_binding,
        sep_errors      = sep_errors,
    )
end

# ============================================================================
# Phase 1: Coarse Convergence Grid
# ============================================================================

println("\n" * "=" ^ 78)
println("PHASE 1: Coarse Convergence Grid")
println("=" ^ 78)

checkpoint_path_p1 = joinpath(output_dir, "convergence_grid.jls")

# Result storage
n_scales = length(shock_scales_phase1)
convergence_matrix = fill(NaN, n_scales, actual_n_thetas)      # convergence rate per (scale, theta)
residual_matrix = fill(NaN, n_scales, actual_n_thetas)         # median residual per (scale, theta)
max_residual_matrix = fill(NaN, n_scales, actual_n_thetas)     # max residual per (scale, theta)
zlb_matrix = fill(false, n_scales, actual_n_thetas)            # ZLB ever binding

# Per-trajectory detail records
detail_records = DataFrame(
    scale_idx      = Int[],
    theta_idx      = Int[],
    trajectory     = Int[],
    shock_scale    = Float64[],
    converged      = Bool[],
    errorflag      = Bool[],
    final_residual = Float64[],
    failure_period = Int[],
    zlb_binding    = Bool[],
)

start_scale_idx = 1
start_theta_idx = 1

# Resume from checkpoint
if do_resume && isfile(checkpoint_path_p1)
    println("  Resuming from checkpoint: $(checkpoint_path_p1)")
    chk = Serialization.deserialize(checkpoint_path_p1)
    global convergence_matrix = chk["convergence_matrix"]
    global residual_matrix    = chk["residual_matrix"]
    global max_residual_matrix = chk["max_residual_matrix"]
    global zlb_matrix         = chk["zlb_matrix"]
    global detail_records     = chk["detail_records"]
    global start_scale_idx    = chk["last_scale_idx"]
    global start_theta_idx    = get(chk, "last_theta_idx", 1) + 1
    if start_theta_idx > actual_n_thetas
        global start_scale_idx += 1
        global start_theta_idx = 1
    end
    println("  Resuming from scale_idx=$(start_scale_idx), theta_idx=$(start_theta_idx)")
end

phase1_start = time()

for si in start_scale_idx:n_scales
    scale = shock_scales_phase1[si]
    scale_start = time()

    for ti in (si == start_scale_idx ? start_theta_idx : 1):actual_n_thetas
        theta = theta_grid[ti, :]

        # Set model parameters
        params = copy(base_values)
        params[theta_idx] = theta
        MacroModelling.write_parameters_input!(model, params, verbose = false)

        n_converged = 0
        residuals_this = Float64[]
        zlb_any = false

        for traj in 1:n_trajectories
            traj_seed = seed0 + si * 10000 + ti * 100 + traj
            @printf("    [%d/%d] theta=%d/%d traj=%d/%d ...", si, n_scales, ti, actual_n_thetas, traj, n_trajectories)
            flush(stdout)
            t_traj = time()
            result = run_single_trajectory(
                model, params, scale, traj_seed;
                sep_horizon=sep_horizon, sep_accept_tol=sep_accept_tol,
                sep_maxit=sep_maxit, sep_tol=sep_tol,
                burn_in=burn_in, sim_periods=sim_periods, verbose=verbose)

            @printf(" %s (%.1fs, res=%.2e)\n",
                    result.converged ? "OK" : "FAIL",
                    time() - t_traj,
                    isfinite(result.final_residual) ? result.final_residual : NaN)
            flush(stdout)
            if result.converged
                n_converged += 1
            end
            if isfinite(result.final_residual)
                push!(residuals_this, result.final_residual)
            end
            zlb_any |= result.zlb_binding

            push!(detail_records, (
                scale_idx      = si,
                theta_idx      = ti,
                trajectory     = traj,
                shock_scale    = scale,
                converged      = result.converged,
                errorflag      = result.errorflag,
                final_residual = isfinite(result.final_residual) ? result.final_residual : -1.0,
                failure_period = result.failure_period,
                zlb_binding    = result.zlb_binding,
            ))
        end

        convergence_matrix[si, ti] = n_converged / n_trajectories
        residual_matrix[si, ti] = isempty(residuals_this) ? NaN : median(residuals_this)
        max_residual_matrix[si, ti] = isempty(residuals_this) ? NaN : maximum(residuals_this)
        zlb_matrix[si, ti] = zlb_any
    end

    # Per-scale summary
    scale_elapsed = time() - scale_start
    mean_conv = mean(filter(!isnan, convergence_matrix[si, :]))
    zlb_frac = sum(zlb_matrix[si, :]) / actual_n_thetas
    @printf("  scale=%.2f  conv=%.1f%%  ZLB=%.1f%%  (%.1f sec)\n",
            scale, 100*mean_conv, 100*zlb_frac, scale_elapsed)
    flush(stdout)

    # Checkpoint after each scale
    Serialization.serialize(checkpoint_path_p1, Dict(
        "convergence_matrix"  => convergence_matrix,
        "residual_matrix"     => residual_matrix,
        "max_residual_matrix" => max_residual_matrix,
        "zlb_matrix"          => zlb_matrix,
        "detail_records"      => detail_records,
        "shock_scales"        => shock_scales_phase1,
        "theta_grid"          => theta_grid,
        "theta_names"         => theta_names,
        "n_trajectories"      => n_trajectories,
        "last_scale_idx"      => si,
        "last_theta_idx"      => actual_n_thetas,
        "settings"            => Dict(
            "sep_horizon"   => sep_horizon,
            "sep_accept_tol" => sep_accept_tol,
            "sep_maxit"     => sep_maxit,
            "sep_tol"       => sep_tol,
            "burn_in"       => burn_in,
            "sim_periods"   => sim_periods,
            "seed0"         => seed0,
            "n_thetas"      => actual_n_thetas,
            "n_trajectories" => n_trajectories,
            "convergence_threshold" => convergence_threshold,
            "chain_path"    => chain_path,
        ),
        "completed_phase1"    => (si == n_scales),
    ))
end

phase1_elapsed = time() - phase1_start
println("\nPhase 1 complete in $(round(phase1_elapsed/60, digits=1)) minutes")

# Phase 1 summary
println("\n  Convergence Rate Summary (across $(actual_n_thetas) thetas):")
println("  " * "-" ^ 52)
@printf("  %8s  %6s  %6s  %6s  %6s  %5s\n", "Scale", "Mean", "P10", "P90", "Min", "ZLB%")
println("  " * "-" ^ 52)
for si in 1:n_scales
    conv_rates = filter(!isnan, convergence_matrix[si, :])
    if isempty(conv_rates)
        continue
    end
    m   = mean(conv_rates)
    p10 = quantile(conv_rates, 0.10)
    p90 = quantile(conv_rates, 0.90)
    mn  = minimum(conv_rates)
    zf  = 100 * sum(zlb_matrix[si, :]) / actual_n_thetas
    @printf("  %8.2f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %4.1f%%\n",
            shock_scales_phase1[si], 100*m, 100*p10, 100*p90, 100*mn, zf)
end
println("  " * "-" ^ 52)

# ============================================================================
# Phase 2: Frontier Refinement
# ============================================================================

if phase_max < 2
    println("\nSkipping Phase 2 (--phase=1)")
    println("\nDone.")
    exit(0)
end

println("\n" * "=" ^ 78)
println("PHASE 2: Frontier Refinement")
println("=" ^ 78)

# Find the convergence cliff: where mean convergence drops below threshold
mean_conv_rates = [mean(filter(!isnan, convergence_matrix[si, :])) for si in 1:n_scales]

cliff_idx = findfirst(r -> r < convergence_threshold, mean_conv_rates)
shock_scale_max = NaN
refinement_scales = Float64[]
if cliff_idx === nothing
    println("  All shock scales have ≥$(round(100*convergence_threshold))% convergence!")
    println("  shock_scale_max = $(shock_scales_phase1[end]) (upper bound of grid)")
    shock_scale_max = shock_scales_phase1[end]
else
    # Cliff is between shock_scales_phase1[cliff_idx-1] and shock_scales_phase1[cliff_idx]
    lo = cliff_idx == 1 ? 0.05 : shock_scales_phase1[cliff_idx - 1]
    hi = shock_scales_phase1[min(cliff_idx + 1, n_scales)]
    println("  Convergence cliff detected between $(lo) and $(hi)")
    println("  Mean convergence: $([@sprintf("%.1f%%", 100*r) for r in mean_conv_rates])")

    # Generate 4 refinement points
    step = (hi - lo) / 5
    refinement_scales = [lo + step * k for k in 1:4]
    # Remove any that duplicate Phase 1 scales
    refinement_scales = filter(s -> !any(abs(s - ps) < 0.001 for ps in shock_scales_phase1), refinement_scales)

    println("  Refinement scales: $(refinement_scales)")
end

# Run refinement
refinement_conv_rates = Dict{Float64, Float64}()
refinement_detail_records = DataFrame(
    scale_idx      = Int[],
    theta_idx      = Int[],
    trajectory     = Int[],
    shock_scale    = Float64[],
    converged      = Bool[],
    errorflag      = Bool[],
    final_residual = Float64[],
    failure_period = Int[],
    zlb_binding    = Bool[],
)

for (ri, scale) in enumerate(refinement_scales)
    scale_start = time()
    n_converged_total = 0
    n_total = 0

    for ti in 1:actual_n_thetas
        theta = theta_grid[ti, :]
        params = copy(base_values)
        params[theta_idx] = theta
        MacroModelling.write_parameters_input!(model, params, verbose = false)

        for traj in 1:n_trajectories
            traj_seed = seed0 + (n_scales + ri) * 10000 + ti * 100 + traj
            result = run_single_trajectory(
                model, params, scale, traj_seed;
                sep_horizon=sep_horizon, sep_accept_tol=sep_accept_tol,
                sep_maxit=sep_maxit, sep_tol=sep_tol,
                burn_in=burn_in, sim_periods=sim_periods, verbose=verbose)

            n_total += 1
            if result.converged
                n_converged_total += 1
            end

            push!(refinement_detail_records, (
                scale_idx      = n_scales + ri,
                theta_idx      = ti,
                trajectory     = traj,
                shock_scale    = scale,
                converged      = result.converged,
                errorflag      = result.errorflag,
                final_residual = isfinite(result.final_residual) ? result.final_residual : -1.0,
                failure_period = result.failure_period,
                zlb_binding    = result.zlb_binding,
            ))
        end
    end

    rate = n_converged_total / n_total
    refinement_conv_rates[scale] = rate
    scale_elapsed = time() - scale_start
    @printf("  scale=%.3f  conv=%.1f%%  (%.1f sec)\n", scale, 100*rate, scale_elapsed)
end

# Combine Phase 1 and refinement to determine shock_scale_max
all_scales_and_rates = Dict{Float64, Float64}()
for si in 1:n_scales
    all_scales_and_rates[shock_scales_phase1[si]] = mean_conv_rates[si]
end
merge!(all_scales_and_rates, refinement_conv_rates)

sorted_scales = sort(collect(keys(all_scales_and_rates)))
shock_scale_max = sorted_scales[1]  # fallback to smallest
for s in sorted_scales
    if all_scales_and_rates[s] >= convergence_threshold
        global shock_scale_max = s
    else
        break
    end
end

println("\n  RESULT: shock_scale_max = $(shock_scale_max)")
println("  Convergence at max: $(round(100*all_scales_and_rates[shock_scale_max], digits=1))%")

# Save Phase 2 results
Serialization.serialize(joinpath(output_dir, "frontier_refinement.jls"), Dict(
    "refinement_scales"      => refinement_scales,
    "refinement_conv_rates"  => refinement_conv_rates,
    "refinement_details"     => refinement_detail_records,
    "all_scales_and_rates"   => all_scales_and_rates,
    "shock_scale_max"        => shock_scale_max,
    "convergence_threshold"  => convergence_threshold,
    "mean_conv_rates_phase1" => mean_conv_rates,
))

# Also update Phase 1 checkpoint with final results
p1_data = Serialization.deserialize(checkpoint_path_p1)
p1_data["shock_scale_max"] = shock_scale_max
p1_data["all_scales_and_rates"] = all_scales_and_rates
Serialization.serialize(checkpoint_path_p1, p1_data)

println("\n  Results saved to:")
println("    $(checkpoint_path_p1)")
println("    $(joinpath(output_dir, "frontier_refinement.jls"))")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 78)
println("FINAL SUMMARY")
println("=" ^ 78)
println("  shock_scale_max = $(shock_scale_max) (≥$(round(100*convergence_threshold))% convergence)")
println("  " * "-" ^ 50)
@printf("  %8s  %8s\n", "Scale", "Conv%")
println("  " * "-" ^ 50)
for s in sorted_scales
    marker = s == shock_scale_max ? " <-- MAX" : ""
    @printf("  %8.3f  %7.1f%%%s\n", s, 100*all_scales_and_rates[s], marker)
end
println("  " * "-" ^ 50)

total_elapsed = time() - phase1_start
println("\n  Total elapsed: $(round(total_elapsed/60, digits=1)) minutes")
println("  Done: $(now())")
