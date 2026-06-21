#!/usr/bin/env julia
# ============================================================================
# NONLINEARITY DECOMPOSITION ACROSS SHOCK SCALES
# ============================================================================
#
# Re-simulates SEP + ROM1 at each shock scale and computes the equation-block
# decomposition of the FOM-ROM1 gap.  Traces how the investment-dominance
# share evolves from small perturbations (0.1) through ZLB-binding territory
# (≥0.5) and beyond (1.5).
#
# Usage:
#   julia --project=. scripts/shock_scale_decomposition.jl
#   julia --project=. scripts/shock_scale_decomposition.jl --chain=path/to/chain.jls
#   julia --project=. scripts/shock_scale_decomposition.jl --n-thetas=5 --sim-periods=40
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
using Plots

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

# ============================================================================
# CLI Arguments
# ============================================================================

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

n_thetas       = parse_arg_int(ARGS, "--n-thetas", 10)
sep_horizon    = parse_arg_int(ARGS, "--sep-horizon", 10)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", 0.35)
sep_maxit      = parse_arg_int(ARGS, "--sep-maxit", 80)
sep_tol        = parse_arg_float(ARGS, "--sep-tol", 1e-5)
burn_in        = parse_arg_int(ARGS, "--burn-in", 20)
sim_periods    = parse_arg_int(ARGS, "--sim-periods", 40)
seed0          = parse_arg_int(ARGS, "--seed", 42)
chain_path     = parse_arg_string(ARGS, "--chain", "")
output_dir     = parse_arg_string(ARGS, "--output-dir",
                    joinpath(REPO_ROOT, ".local_artifacts", "shock_scale_decomposition"))
verbose        = "--verbose" in ARGS

shock_scales_str = parse_arg_string(ARGS, "--shock-scales", "")
if shock_scales_str != ""
    shock_scales = parse.(Float64, split(shock_scales_str, ","))
else
    shock_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
end

mkpath(output_dir)
rng = MersenneTwister(seed0)

println("=" ^ 78)
println("NONLINEARITY DECOMPOSITION ACROSS SHOCK SCALES")
println("Started: $(now())")
println("=" ^ 78)
println("  Shock scales: $shock_scales")
println("  Thetas:       $n_thetas")
println("  Sim periods:  $sim_periods + $burn_in burn-in")
println("  Output:       $output_dir")
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

# ============================================================================
# Load Chain & Select Thetas
# ============================================================================

specs = get_phase1_18param_specs()
theta_names = [s.name for s in specs]

if chain_path == ""
    chain_path = joinpath(REPO_ROOT,
        ".local_artifacts/hlt_18param_realdata/hlt_switching_synthetic_chain_2000.jls")
    if !isfile(chain_path)
        chain_path = joinpath(REPO_ROOT,
            ".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_chain_2000_seed99.jls")
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

# Select thetas: stratified by distance
sorted_idx = sortperm(distances)
stride = max(1, n_draws_total ÷ n_thetas)
selected_idx = sorted_idx[1:stride:min(n_draws_total, stride*n_thetas)]
selected_idx = selected_idx[1:min(n_thetas, length(selected_idx))]
theta_grid = chain_matrix[selected_idx, :]
actual_n_thetas = size(theta_grid, 1)
println("  Selected $actual_n_thetas representative thetas")
flush(stdout)

# ============================================================================
# Shock Drawing (same as grid simulation)
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
# Main Loop: Simulate + Decompose at Each Scale
# ============================================================================

println("\n" * "=" ^ 78)
println("SIMULATING ACROSS SHOCK SCALES")
println("=" ^ 78)

n_scales = length(shock_scales)

# Results storage
scale_results = Dict{Float64, Dict{String, Any}}()

for (si, scale) in enumerate(shock_scales)
    println("\n--- Shock scale $scale ($si/$n_scales) ---")
    flush(stdout)

    # Collect all FOM and ROM1 observable outputs for this scale
    all_delta_obs = Float64[]
    n_total_samples = 0
    n_converged = 0
    n_zlb_binding = 0

    # Pre-allocate for this scale
    max_samples = actual_n_thetas * sim_periods
    delta_obs_matrix = zeros(d_obs, max_samples)
    zlb_flags = falses(max_samples)
    cursor = 0

    for (ti, theta_row) in enumerate(eachrow(theta_grid))
        theta = collect(Float64, theta_row)

        # Set parameters
        params = Float64.(HLT.parameter_values)
        for (j, tname) in enumerate(theta_names)
            pidx = findfirst(==(tname), HLT.parameters)
            if pidx !== nothing
                params[pidx] = theta[j]
            end
        end

        # Build ROM1 cache for this theta
        local rom_cache
        try
            rom_cache = build_rom_cache(HLT, 1; params=params, use_obc=true)
        catch e
            println("  [theta $ti] ROM build failed: $e")
            continue
        end

        # Draw shocks (deterministic per theta/scale for reproducibility)
        trajectory_seed = seed0 * 1000 + si * 100 + ti
        total_periods = sim_periods + burn_in
        shocks = draw_shocks(MersenneTwister(trajectory_seed), HLT, total_periods, scale)

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
                println("  [theta $ti] SEP failed (singular)")
                continue
            else
                rethrow()
            end
        end

        if res.errorflag
            # Accept partial prefix if available
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
            # FOM: next-period state from SEP simulation
            fom_obs = sim[obs_idx, t + 1]

            # ROM1: linear prediction from current state
            local rom_next
            try
                rom_next = rom_step_full(rom_cache, sim[:, t], sim_shocks[:, t])
            catch
                continue
            end
            rom1_obs = rom_next[obs_idx]

            delta = fom_obs .- rom1_obs

            cursor += 1
            if cursor > max_samples
                # Extend arrays
                delta_obs_matrix = hcat(delta_obs_matrix, zeros(d_obs, max_samples))
                zlb_flags = vcat(zlb_flags, falses(max_samples))
                max_samples *= 2
            end
            delta_obs_matrix[:, cursor] = delta

            # Detect ZLB binding (robs near zero → constrained)
            robs_pos = findfirst(==(:robs), observables)
            robs_val = sim[obs_idx[robs_pos], t + 1]
            # robs is 100*(R-1), ZLB binding when very low
            zlb_flags[cursor] = robs_val < 0.05  # ~near zero net rate
        end
    end

    # Trim to actual samples
    delta_obs = delta_obs_matrix[:, 1:cursor]
    zlb = zlb_flags[1:cursor]
    n_samples = cursor
    n_zlb = sum(zlb)

    # Compute decomposition
    shares_all = compute_block_shares(delta_obs, observables)

    # Split by ZLB status
    shares_binding = Dict(bn => 0.0 for bn in block_names)
    shares_nonbinding = Dict(bn => 0.0 for bn in block_names)
    if n_zlb > 0 && n_zlb < n_samples
        shares_binding = compute_block_shares(delta_obs[:, zlb], observables)
        shares_nonbinding = compute_block_shares(delta_obs[:, .!zlb], observables)
    elseif n_zlb == 0
        shares_nonbinding = shares_all
    else
        shares_binding = shares_all
    end

    # Mean absolute gap
    mean_gap = n_samples > 0 ? mean(sqrt.(sum(delta_obs.^2, dims=1))) : 0.0

    scale_results[scale] = Dict(
        "shares_all" => shares_all,
        "shares_binding" => shares_binding,
        "shares_nonbinding" => shares_nonbinding,
        "n_samples" => n_samples,
        "n_converged" => n_converged,
        "n_zlb" => n_zlb,
        "zlb_pct" => n_samples > 0 ? 100.0 * n_zlb / n_samples : 0.0,
        "mean_gap" => mean_gap,
    )

    @printf("  Samples: %d, Converged thetas: %d/%d, ZLB: %d (%.1f%%)\n",
            n_samples, n_converged, actual_n_thetas, n_zlb,
            n_samples > 0 ? 100.0 * n_zlb / n_samples : 0.0)
    @printf("  Mean |delta|: %.6f\n", mean_gap)
    for bn in block_names
        @printf("    %-22s  All: %5.1f%%  Binding: %5.1f%%  Non-binding: %5.1f%%\n",
                bn, shares_all[bn]*100, shares_binding[bn]*100, shares_nonbinding[bn]*100)
    end
    flush(stdout)
end

# ============================================================================
# Save Results
# ============================================================================

results_path = joinpath(output_dir, "decomposition_by_scale.jls")
Serialization.serialize(results_path, Dict(
    "scale_results" => scale_results,
    "shock_scales" => shock_scales,
    "block_names" => block_names,
    "n_thetas" => actual_n_thetas,
    "sim_periods" => sim_periods,
    "chain_path" => chain_path,
    "timestamp" => now(),
))
println("\nResults saved: $results_path")

# ============================================================================
# Figures
# ============================================================================

const FIG_DIR = joinpath(REPO_ROOT, "docs", "paper", "figures")
mkpath(FIG_DIR)

gr()
default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
        tickfontsize=9, legendfontsize=8, linewidth=2.0, dpi=300)

# Colors for each block
block_colors = Dict(
    "Investment/Capital"  => :red,
    "Consumption/Euler"   => :blue,
    "Wage Phillips"       => :green,
    "Output/Resource"     => :orange,
    "Labor Market"        => :purple,
    "Price Phillips"      => :cyan,
    "Taylor Rule"         => :gray,
)

# --- Figure 1: Block shares vs shock scale (all samples) ---
sorted_scales = sort(shock_scales)
fig1 = plot(size=(700, 450), margin=5Plots.mm, bottom_margin=8Plots.mm)

for bn in block_names
    vals = [get(scale_results[s]["shares_all"], bn, 0.0) * 100 for s in sorted_scales]
    plot!(fig1, sorted_scales, vals, label=bn, color=block_colors[bn],
          marker=:circle, markersize=4)
end

plot!(fig1,
    xlabel="Shock scale",
    ylabel="Share of FOM-ROM1 gap (%)",
    title="Equation-block decomposition across shock scales",
    legend=:right,
    ylims=(-2, 102),
)

savefig(fig1, joinpath(FIG_DIR, "decomposition_vs_shock_scale.pdf"))
println("Saved: decomposition_vs_shock_scale.pdf")

# --- Figure 2: Mean gap magnitude vs shock scale ---
fig2 = plot(size=(600, 400), margin=5Plots.mm)
gaps = [scale_results[s]["mean_gap"] for s in sorted_scales]
zlb_pcts = [scale_results[s]["zlb_pct"] for s in sorted_scales]

plot!(fig2, sorted_scales, gaps,
    label="Mean |FOM-ROM1| gap",
    color=:black, marker=:circle, markersize=5,
    xlabel="Shock scale",
    ylabel="Mean observable gap (model units)",
    title="Nonlinearity magnitude vs shock scale",
)

# Secondary axis: ZLB binding percentage
fig2b = twinx(fig2)
plot!(fig2b, sorted_scales, zlb_pcts,
    label="ZLB binding (%)",
    color=:red, linestyle=:dash, marker=:square, markersize=4,
    ylabel="ZLB binding (%)",
    legend=:topleft,
)

savefig(fig2, joinpath(FIG_DIR, "gap_magnitude_vs_shock_scale.pdf"))
println("Saved: gap_magnitude_vs_shock_scale.pdf")

# --- Figure 3: Binding vs non-binding shares (stacked area-like) ---
# Only for scales where ZLB actually binds
binding_scales = [s for s in sorted_scales if scale_results[s]["n_zlb"] > 0]

if length(binding_scales) >= 2
    fig3 = plot(layout=(1, 2), size=(1100, 450), margin=5Plots.mm, bottom_margin=8Plots.mm)

    # Panel (a): Non-binding shares
    for bn in block_names
        vals = [scale_results[s]["shares_nonbinding"][bn] * 100 for s in binding_scales]
        plot!(fig3[1], binding_scales, vals, label=bn, color=block_colors[bn],
              marker=:circle, markersize=3)
    end
    plot!(fig3[1], xlabel="Shock scale", ylabel="Share (%)",
          title="(a) Non-binding periods", legend=:right, ylims=(-2, 102))

    # Panel (b): Binding shares
    for bn in block_names
        vals = [scale_results[s]["shares_binding"][bn] * 100 for s in binding_scales]
        plot!(fig3[2], binding_scales, vals, label=bn, color=block_colors[bn],
              marker=:circle, markersize=3)
    end
    plot!(fig3[2], xlabel="Shock scale", ylabel="Share (%)",
          title="(b) ZLB-binding periods", legend=:right, ylims=(-2, 102))

    savefig(fig3, joinpath(FIG_DIR, "decomposition_binding_vs_nonbinding.pdf"))
    println("Saved: decomposition_binding_vs_nonbinding.pdf")
end

# ============================================================================
# Summary Table
# ============================================================================

println("\n" * "=" ^ 78)
println("SUMMARY TABLE")
println("=" ^ 78)
@printf("%-8s  %6s  %6s  %5s  %6s  %6s  %6s  %6s  %6s  %6s  %6s\n",
    "Scale", "N", "ZLB%", "Gap",
    "Invest", "Euler", "Wage", "Output", "Labor", "Price", "Taylor")
println("-" ^ 100)

for s in sorted_scales
    r = scale_results[s]
    sa = r["shares_all"]
    @printf("%-8.2f  %6d  %5.1f  %5.4f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%\n",
        s, r["n_samples"], r["zlb_pct"], r["mean_gap"],
        sa["Investment/Capital"]*100,
        sa["Consumption/Euler"]*100,
        sa["Wage Phillips"]*100,
        sa["Output/Resource"]*100,
        sa["Labor Market"]*100,
        sa["Price Phillips"]*100,
        sa["Taylor Rule"]*100,
    )
end

println("\n" * "=" ^ 78)
println("DONE: $(now())")
println("=" ^ 78)
