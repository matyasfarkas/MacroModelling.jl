#!/usr/bin/env julia
# ============================================================================
# ARRS PRICING ROBUSTNESS: BLOCK DECOMPOSITION ACROSS PRICING SPECIFICATIONS
# ============================================================================
#
# Tests whether the 0.2% pricing-block share is robust to:
#   (1) Golosov-Lucas slope (kappa ≈ 1.71, ARRS QJE 2024)
#   (2) Nakamura-Steinsson slope (kappa ≈ 0.47, ARRS QJE 2024)
#   (3) Rotemberg quadratic adjustment-cost pricing (explicit pi^2 nonlinearity)
#
# Usage:
#   julia --project=. scripts/arrs_pricing_robustness.jl
#   julia --project=. scripts/arrs_pricing_robustness.jl --models=baseline,GL,NS,Rot
#   julia --project=. scripts/arrs_pricing_robustness.jl --shock-scales=0.1,0.5,1.0
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
                    joinpath(REPO_ROOT, ".local_artifacts", "arrs_pricing_robustness"))
verbose        = "--verbose" in ARGS

# Model selection
models_str = parse_arg_string(ARGS, "--models", "baseline,GL,NS,Rot")
model_keys = strip.(split(models_str, ","))

# Map short names to full model names
model_name_map = Dict(
    "baseline" => "Smets_Wouters_2007_HLT_obc",
    "GL"       => "Smets_Wouters_2007_HLT_obc_GL",
    "NS"       => "Smets_Wouters_2007_HLT_obc_NS",
    "Rot"      => "Smets_Wouters_2007_HLT_obc_Rot",
)

model_label_map = Dict(
    "baseline" => "Baseline Calvo (ξ=0.60)",
    "GL"       => "GL slope (ξ=0.034, κ≈1.71)",
    "NS"       => "NS slope (ξ=0.17, κ≈0.47)",
    "Rot"      => "Rotemberg (φ_R≈5.7)",
)

shock_scales_str = parse_arg_string(ARGS, "--shock-scales", "")
if shock_scales_str != ""
    shock_scales = parse.(Float64, split(shock_scales_str, ","))
else
    shock_scales = [0.1, 0.5, 1.0]
end

mkpath(output_dir)
rng = MersenneTwister(seed0)

println("=" ^ 78)
println("ARRS PRICING ROBUSTNESS: BLOCK DECOMPOSITION")
println("Started: $(now())")
println("=" ^ 78)
println("  Models:       $model_keys")
println("  Shock scales: $shock_scales")
println("  Thetas:       $n_thetas")
println("  Sim periods:  $sim_periods + $burn_in burn-in")
println("  Output:       $output_dir")
flush(stdout)

# ============================================================================
# Observables & Block Definitions
# ============================================================================

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]

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

if chain_theta_names != theta_names
    println("  Reindexing parameters...")
    reindex = [findfirst(==(n), chain_theta_names) for n in theta_names]
    @assert all(!isnothing, reindex)
    chain_matrix = chain_matrix[:, reindex]
end

n_draws_total = size(chain_matrix, 1)
println("  Chain: $(n_draws_total) draws × $(size(chain_matrix, 2)) parameters")

# Stratified sampling
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
# Preload All Models (each at top-level to avoid world-age issues)
# ============================================================================

println("\nPreloading models...")
const preloaded_models = Dict{String, Any}()
for mkey in model_keys
    full_name = get(model_name_map, mkey, mkey)
    try
        m = load_hlt_model(REPO_ROOT, full_name; mod = @__MODULE__)
        Base.invokelatest(MacroModelling.solve!, m, silent = true)
        preloaded_models[mkey] = m
        println("  $mkey: $(length(m.var)) vars, $(length(m.exo)) shocks — OK")
    catch e
        println("  $mkey: LOAD ERROR — $e")
    end
end
flush(stdout)

# ============================================================================
# Main Loop: Model × Scale
# ============================================================================

# Results: model_key => scale => Dict of shares/stats
all_results = Dict{String, Dict{Float64, Dict{String, Any}}}()

for mkey in model_keys
    if !haskey(preloaded_models, mkey)
        println("  Skipping $mkey (not loaded)")
        continue
    end
    model = preloaded_models[mkey]
    full_name = get(model_name_map, mkey, mkey)
    label = get(model_label_map, mkey, mkey)
    println("\n" * "=" ^ 78)
    println("MODEL: $label ($full_name)")
    println("=" ^ 78)
    flush(stdout)
    println("  Loaded: $(model.model_name), $(length(model.var)) vars, $(length(model.exo)) shocks")

    obs_idx = Int.(indexin(observables, model.var))
    if any(isnothing, obs_idx)
        missing_obs = observables[isnothing.(obs_idx)]
        println("  ERROR: missing observables $missing_obs")
        continue
    end

    d_obs = length(obs_idx)
    d_eps = length(model.exo)

    scale_results = Dict{Float64, Dict{String, Any}}()

    for (si, scale) in enumerate(shock_scales)
        println("\n  --- Scale $scale ($si/$(length(shock_scales))) ---")
        flush(stdout)

        max_samples = actual_n_thetas * sim_periods
        delta_obs_matrix = zeros(d_obs, max_samples)
        zlb_flags = falses(max_samples)
        cursor = 0
        n_converged = 0

        for (ti, theta_row) in enumerate(eachrow(theta_grid))
            theta = collect(Float64, theta_row)

            # Set parameters — map from baseline 18-param names to this model's params
            params = Float64.(model.parameter_values)
            for (j, tname) in enumerate(theta_names)
                pidx = findfirst(==(tname), model.parameters)
                if pidx !== nothing
                    params[pidx] = theta[j]
                end
            end

            # Build ROM1 cache
            local rom_cache
            try
                rom_cache = build_rom_cache(model, 1; params=params, use_obc=true)
            catch e
                verbose && println("    [theta $ti] ROM build failed: $e")
                continue
            end

            # Draw shocks
            trajectory_seed = seed0 * 1000 + si * 100 + ti
            total_periods = sim_periods + burn_in
            shocks = draw_shocks(MersenneTwister(trajectory_seed), model, total_periods, scale)

            # Run SEP simulation
            local res
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
                    shock_scaling    = :none,
                    shocks           = shocks,
                    random_seed      = trajectory_seed,
                    silent           = true,
                )
            catch e
                if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                    verbose && println("    [theta $ti] SEP failed (singular)")
                    continue
                else
                    rethrow()
                end
            end

            if res.errorflag
                sep_errors = hasproperty(res, :sep_errors) ? res.sep_errors : Float64[]
                valid_errors = filter(isfinite, sep_errors)
                if isempty(valid_errors) || !all(e -> e <= sep_accept_tol, valid_errors)
                    verbose && println("    [theta $ti] SEP diverged")
                    continue
                end
            end

            n_converged += 1
            sim = Array(res.simulation)
            sim_shocks = res.shocks

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
                    zlb_flags = vcat(zlb_flags, falses(max_samples))
                end
                delta_obs_matrix[:, cursor] = delta

                robs_pos = findfirst(==(:robs), observables)
                robs_val = sim[obs_idx[robs_pos], t + 1]
                zlb_flags[cursor] = robs_val < 0.05
            end
        end

        delta_obs = delta_obs_matrix[:, 1:cursor]
        zlb = zlb_flags[1:cursor]
        n_samples = cursor
        n_zlb = sum(zlb)

        shares_all = compute_block_shares(delta_obs, observables)
        mean_gap = n_samples > 0 ? mean(sqrt.(sum(delta_obs.^2, dims=1))) : 0.0

        scale_results[scale] = Dict(
            "shares_all" => shares_all,
            "n_samples" => n_samples,
            "n_converged" => n_converged,
            "n_zlb" => n_zlb,
            "zlb_pct" => n_samples > 0 ? 100.0 * n_zlb / n_samples : 0.0,
            "mean_gap" => mean_gap,
        )

        @printf("    Samples: %d, Converged: %d/%d, ZLB: %d (%.1f%%)\n",
                n_samples, n_converged, actual_n_thetas, n_zlb,
                n_samples > 0 ? 100.0 * n_zlb / n_samples : 0.0)
        @printf("    Mean |delta|: %.6f\n", mean_gap)
        for bn in block_names
            @printf("      %-22s  %5.1f%%\n", bn, shares_all[bn]*100)
        end
        flush(stdout)
    end

    all_results[mkey] = scale_results
end

# ============================================================================
# Save Results
# ============================================================================

results_path = joinpath(output_dir, "arrs_pricing_robustness.jls")
Serialization.serialize(results_path, Dict(
    "all_results" => all_results,
    "model_keys" => model_keys,
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

model_colors = Dict("baseline" => :black, "GL" => :red, "NS" => :blue, "Rot" => :green)
model_markers = Dict("baseline" => :circle, "GL" => :diamond, "NS" => :square, "Rot" => :utriangle)

# --- Figure: Investment/Capital share across pricing specs ---
sorted_scales = sort(shock_scales)
fig1 = plot(size=(700, 450), margin=5Plots.mm, bottom_margin=8Plots.mm)

for mkey in model_keys
    haskey(all_results, mkey) || continue
    label = get(model_label_map, mkey, mkey)
    vals = [get(all_results[mkey][s]["shares_all"], "Investment/Capital", 0.0) * 100
            for s in sorted_scales]
    plot!(fig1, sorted_scales, vals, label=label,
          color=get(model_colors, mkey, :gray),
          marker=get(model_markers, mkey, :circle), markersize=5)
end

plot!(fig1,
    xlabel="Shock scale",
    ylabel="Investment/Capital block share (%)",
    title="Investment dominance across pricing specifications",
    legend=:bottomright,
    ylims=(0, 100),
)

savefig(fig1, joinpath(FIG_DIR, "arrs_investment_share.pdf"))
println("Saved: arrs_investment_share.pdf")

# --- Figure: Price Phillips share across pricing specs ---
fig2 = plot(size=(700, 450), margin=5Plots.mm, bottom_margin=8Plots.mm)

for mkey in model_keys
    haskey(all_results, mkey) || continue
    label = get(model_label_map, mkey, mkey)
    vals = [get(all_results[mkey][s]["shares_all"], "Price Phillips", 0.0) * 100
            for s in sorted_scales]
    plot!(fig2, sorted_scales, vals, label=label,
          color=get(model_colors, mkey, :gray),
          marker=get(model_markers, mkey, :circle), markersize=5)
end

plot!(fig2,
    xlabel="Shock scale",
    ylabel="Price Phillips block share (%)",
    title="Pricing block share across pricing specifications",
    legend=:topright,
    ylims=(-1, 20),
)

savefig(fig2, joinpath(FIG_DIR, "arrs_pricing_share.pdf"))
println("Saved: arrs_pricing_share.pdf")

# --- Figure: All blocks grouped bar chart at scale 1.0 ---
ref_scale = 1.0
if ref_scale in shock_scales
    fig3 = plot(size=(900, 500), margin=5Plots.mm, bottom_margin=12Plots.mm)
    n_models = length([k for k in model_keys if haskey(all_results, k)])
    n_blocks = length(block_names)
    bar_width = 0.8 / n_models

    for (mi, mkey) in enumerate(model_keys)
        haskey(all_results, mkey) || continue
        haskey(all_results[mkey], ref_scale) || continue
        label = get(model_label_map, mkey, mkey)
        vals = [get(all_results[mkey][ref_scale]["shares_all"], bn, 0.0) * 100
                for bn in block_names]
        x_pos = (1:n_blocks) .- 0.4 .+ (mi - 0.5) * bar_width
        bar!(fig3, x_pos, vals, bar_width=bar_width, label=label,
             color=get(model_colors, mkey, :gray), alpha=0.85)
    end

    plot!(fig3,
        xticks=(1:n_blocks, block_names),
        xrotation=25,
        ylabel="Share of FOM-ROM1 gap (%)",
        title="Block decomposition across pricing specs (scale=$(ref_scale))",
        legend=:topright,
        ylims=(-2, 100),
    )

    savefig(fig3, joinpath(FIG_DIR, "arrs_block_comparison.pdf"))
    println("Saved: arrs_block_comparison.pdf")
end

# ============================================================================
# Summary Table
# ============================================================================

println("\n" * "=" ^ 78)
println("SUMMARY TABLE: INVESTMENT/CAPITAL AND PRICE PHILLIPS SHARES")
println("=" ^ 78)
@printf("%-12s  %-8s  %6s  %8s  %8s  %8s\n",
    "Model", "Scale", "N", "Invest%", "Price%", "MeanGap")
println("-" ^ 60)

for mkey in model_keys
    haskey(all_results, mkey) || continue
    for s in sorted_scales
        haskey(all_results[mkey], s) || continue
        r = all_results[mkey][s]
        sa = r["shares_all"]
        @printf("%-12s  %-8.2f  %6d  %7.1f%%  %7.1f%%  %8.5f\n",
            mkey, s, r["n_samples"],
            sa["Investment/Capital"]*100,
            sa["Price Phillips"]*100,
            r["mean_gap"])
    end
end

println("\n" * "=" ^ 78)
println("DONE: $(now())")
println("=" ^ 78)
