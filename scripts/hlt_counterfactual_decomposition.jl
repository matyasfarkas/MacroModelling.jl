#!/usr/bin/env julia
# ============================================================================
# HLT INVESTMENT-BLOCK COUNTERFACTUAL DECOMPOSITION
# ============================================================================
#
# Isolates the contribution of each investment-block nonlinearity to the
# FOM-ROM1 gap by running SEP on model variants where specific nonlinearities
# are linearized:
#
#   Baseline:     All nonlinearities active (original HLT OBC model)
#   noS:          S(x) = 0, S'(x) = 0  (quadratic investment adj cost off)
#   lina:         a(z) = rk_ss*(z-1)    (exponential utilization cost linearized)
#   noS_lina:     Both S and a linearized (residual = Tobin's q + Kimball)
#
# Methodology:
#   For each (theta, shock_scale), run SEP on each variant with IDENTICAL shocks.
#   Build ROM1 for each variant and compute delta_V = FOM_V - ROM1_V.
#   Share explained by removing X = 1 - ||delta_noX||^2 / ||delta_baseline||^2
#
# Usage:
#   julia --project=. scripts/hlt_counterfactual_decomposition.jl
#   julia --project=. scripts/hlt_counterfactual_decomposition.jl --n-thetas=3
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

# ============================================================================
# CLI Arguments
# ============================================================================

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

n_thetas       = parse_arg_int(ARGS, "--n-thetas", 5)
sep_horizon    = parse_arg_int(ARGS, "--sep-horizon", 10)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", 0.35)
sep_maxit      = parse_arg_int(ARGS, "--sep-maxit", 80)
sep_tol        = parse_arg_float(ARGS, "--sep-tol", 1e-5)
burn_in        = parse_arg_int(ARGS, "--burn-in", 10)
sim_periods    = parse_arg_int(ARGS, "--sim-periods", 20)
seed0          = parse_arg_int(ARGS, "--seed", 42)
chain_path     = parse_arg_string(ARGS, "--chain", "")
output_dir     = parse_arg_string(ARGS, "--output-dir",
                    joinpath(REPO_ROOT, ".local_artifacts", "counterfactual_decomposition"))

shock_scales_str = parse_arg_string(ARGS, "--shock-scales", "")
if shock_scales_str != ""
    shock_scales = parse.(Float64, split(shock_scales_str, ","))
else
    shock_scales = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
end

mkpath(output_dir)
rng = MersenneTwister(seed0)

println("=" ^ 78)
println("HLT INVESTMENT-BLOCK COUNTERFACTUAL DECOMPOSITION")
println("Started: $(now())")
println("=" ^ 78)
println("  Shock scales: $shock_scales")
println("  Thetas:       $n_thetas")
println("  Sim periods:  $sim_periods + $burn_in burn-in")
println("  Output:       $output_dir")
flush(stdout)

# ============================================================================
# Load 4 Model Variants
# ============================================================================

model_specs = [
    ("baseline",  "Smets_Wouters_2007_HLT_obc.jl",           :Smets_Wouters_2007_HLT_obc),
    ("noS",       "Smets_Wouters_2007_HLT_obc_noS.jl",       :Smets_Wouters_2007_HLT_obc_noS),
    ("lina",      "Smets_Wouters_2007_HLT_obc_lina.jl",      :Smets_Wouters_2007_HLT_obc_lina),
    ("noS_lina",  "Smets_Wouters_2007_HLT_obc_noS_lina.jl",  :Smets_Wouters_2007_HLT_obc_noS_lina),
]

models = Dict{String, Any}()
for (label, filename, sym) in model_specs
    t0 = time()
    println("\nLoading model: $label ($filename)...")
    flush(stdout)
    include(joinpath(REPO_ROOT, "models", filename))
    mdl = Base.invokelatest(getfield, @__MODULE__, sym)
    Base.invokelatest(MacroModelling.solve!, mdl; silent=true)
    models[label] = mdl
    dt = time() - t0
    @printf("  Loaded + solved in %.1f seconds. %d variables, %d shocks\n",
            dt, length(mdl.var), length(mdl.exo))
    flush(stdout)
end

const BASELINE = models["baseline"]

# Observable definition (same across all models)
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
d_obs = length(observables)

# Build observable index for each model (in case orderings differ)
function get_obs_idx(mdl, obs_list)
    idx = Int[]
    for v in obs_list
        pos = findfirst(==(v), mdl.var)
        pos === nothing && error("Variable $v not found in model $(mdl.model_name)")
        push!(idx, pos)
    end
    return idx
end

obs_indices = Dict(label => get_obs_idx(models[label], observables)
                   for (label, _, _) in model_specs)

# Shock name mapping across models
function get_shock_reindex(src_model, dst_model)
    src_names = src_model.exo
    dst_names = dst_model.exo
    reindex = Int[]
    for sname in dst_names
        pos = findfirst(==(sname), src_names)
        if pos === nothing
            # OBC shock may differ; try partial match
            push!(reindex, 0)
        else
            push!(reindex, pos)
        end
    end
    return reindex
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
# Shock Drawing (identical across models)
# ============================================================================

function draw_shocks_for_model(rng_local::AbstractRNG, model, total_periods::Int, shock_scale::Float64)
    shock_names_local = model.exo
    nshocks = length(shock_names_local)
    shocks = zeros(nshocks, total_periods)
    obc_mask = contains.(string.(shock_names_local), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    isempty(structural_idx) && return shocks
    sigmas = ones(length(structural_idx))
    for (i, idx) in enumerate(structural_idx)
        sigmas[i] = MacroModelling.sep_irf_shock_std(model, shock_names_local[idx])
    end
    sigmas .*= shock_scale
    shocks[structural_idx, :] .= Diagonal(sigmas) * randn(rng_local, length(structural_idx), total_periods)
    return shocks
end

# Draw shocks from baseline model and reindex for each variant
function draw_and_reindex_shocks(seed_val::Int, scale::Float64, total_periods::Int)
    shocks_base = draw_shocks_for_model(MersenneTwister(seed_val), BASELINE, total_periods, scale)
    result = Dict{String, Matrix{Float64}}()
    result["baseline"] = shocks_base

    for (label, _, _) in model_specs
        label == "baseline" && continue
        mdl = models[label]
        reindex = get_shock_reindex(BASELINE, mdl)
        shocks_v = zeros(length(mdl.exo), total_periods)
        for (di, si) in enumerate(reindex)
            si > 0 && (shocks_v[di, :] .= shocks_base[si, :])
        end
        result[label] = shocks_v
    end
    return result
end

# ============================================================================
# SEP Runner
# ============================================================================

function run_sep_on_model(mdl, shocks_matrix, sim_periods, burn_in, seed_val;
                          sep_horizon=10, sep_maxit=80, sep_tol=1e-5, sep_accept_tol=0.35)
    try
        res = MacroModelling.simulate_sep_extended_path(
            mdl;
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
            shocks           = shocks_matrix,
            random_seed      = seed_val,
            silent           = true,
        )
        return res
    catch e
        if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
            return nothing
        else
            rethrow()
        end
    end
end

# ============================================================================
# Main Loop: Counterfactual Decomposition
# ============================================================================

println("\n" * "=" ^ 78)
println("COUNTERFACTUAL SIMULATIONS")
println("=" ^ 78)

variant_labels = ["baseline", "noS", "lina", "noS_lina"]

# Results storage: scale → Dict with variant L² norms and share explained
all_results = Dict{Float64, Dict{String, Any}}()

for (si, scale) in enumerate(shock_scales)
    println("\n--- Shock scale $scale ($si/$(length(shock_scales))) ---")
    flush(stdout)

    # Accumulate squared deltas per variant
    sq_norms = Dict(v => 0.0 for v in variant_labels)
    n_samples = Dict(v => 0 for v in variant_labels)
    n_converged_theta = 0

    for (ti, theta_row) in enumerate(eachrow(theta_grid))
        theta = collect(Float64, theta_row)

        # Set parameters on all models
        param_vecs = Dict{String, Vector{Float64}}()
        rom_caches = Dict{String, Any}()
        all_ok = true

        for label in variant_labels
            mdl = models[label]
            params = Float64.(mdl.parameter_values)
            for (j, tname) in enumerate(theta_names)
                pidx = findfirst(==(tname), mdl.parameters)
                pidx !== nothing && (params[pidx] = theta[j])
            end
            param_vecs[label] = params

            try
                rom_caches[label] = build_rom_cache(mdl, 1; params=params, use_obc=true)
            catch e
                @printf("  [theta %d] ROM build failed for %s: %s\n", ti, label, e)
                all_ok = false
                break
            end
        end
        !all_ok && continue

        # Draw shocks (same seed for all models at this theta/scale)
        trajectory_seed = seed0 * 1000 + si * 100 + ti
        total_periods = sim_periods + burn_in
        shock_sets = draw_and_reindex_shocks(trajectory_seed, scale, total_periods)

        # Run SEP on each variant
        sep_results = Dict{String, Any}()
        for label in variant_labels
            mdl = models[label]
            # Need to set parameters before SEP
            MacroModelling.write_parameters_input!(mdl, param_vecs[label], verbose=false)
            MacroModelling.solve!(mdl; algorithm=:first_order, dynamics=true, obc=true, silent=true)

            res = run_sep_on_model(mdl, shock_sets[label], sim_periods, burn_in, trajectory_seed;
                                   sep_horizon=sep_horizon, sep_maxit=sep_maxit,
                                   sep_tol=sep_tol, sep_accept_tol=sep_accept_tol)
            if res === nothing || res.errorflag
                if res !== nothing
                    sep_errors = hasproperty(res, :sep_errors) ? res.sep_errors : Float64[]
                    valid_errors = filter(isfinite, sep_errors)
                    if isempty(valid_errors) || !all(e -> e <= sep_accept_tol, valid_errors)
                        @printf("  [theta %d] SEP diverged for %s\n", ti, label)
                        sep_results[label] = nothing
                        continue
                    end
                else
                    @printf("  [theta %d] SEP failed for %s\n", ti, label)
                    sep_results[label] = nothing
                    continue
                end
            end
            sep_results[label] = res
        end

        # Skip theta if baseline didn't converge
        if sep_results["baseline"] === nothing
            continue
        end
        n_converged_theta += 1

        # Compute FOM-ROM1 deltas for each variant
        for label in variant_labels
            sep_results[label] === nothing && continue

            mdl = models[label]
            oidx = obs_indices[label]
            rc = rom_caches[label]
            sim = Array(sep_results[label].simulation)
            sim_shocks = sep_results[label].shocks

            T_avail = min(sim_periods, size(sim, 2) - 1)
            for t in 1:T_avail
                fom_obs = sim[oidx, t + 1]
                local rom_next
                try
                    rom_next = rom_step_full(rc, sim[:, t], sim_shocks[:, t])
                catch
                    continue
                end
                rom1_obs = rom_next[oidx]
                delta = fom_obs .- rom1_obs
                sq_norms[label] += sum(delta.^2)
                n_samples[label] += 1
            end
        end
    end

    # Compute share explained
    base_sq = n_samples["baseline"] > 0 ? sq_norms["baseline"] / n_samples["baseline"] : 0.0
    share_explained = Dict{String, Float64}()

    for label in variant_labels
        if label == "baseline"
            share_explained[label] = 0.0
            continue
        end
        if base_sq < 1e-30 || n_samples[label] == 0
            share_explained[label] = 0.0
        else
            variant_sq = sq_norms[label] / n_samples[label]
            share_explained[label] = 1.0 - variant_sq / base_sq
        end
    end

    # Derived contributions
    S_share = share_explained["noS"]           # fraction explained by S(x)
    a_share = share_explained["lina"]          # fraction explained by a(z)
    joint_share = share_explained["noS_lina"]  # fraction explained by S+a jointly
    interaction = S_share + a_share - joint_share
    residual = 1.0 - joint_share               # Tobin's q + Kimball + other

    all_results[scale] = Dict(
        "sq_norms" => sq_norms,
        "n_samples" => n_samples,
        "n_converged_theta" => n_converged_theta,
        "share_explained" => share_explained,
        "S_share" => S_share,
        "a_share" => a_share,
        "joint_share" => joint_share,
        "interaction" => interaction,
        "residual" => residual,
        "mean_baseline_gap" => base_sq > 0 ? sqrt(base_sq) : 0.0,
    )

    @printf("  Converged: %d/%d thetas\n", n_converged_theta, actual_n_thetas)
    @printf("  Baseline mean ||delta||²: %.6e  (n=%d)\n", base_sq, n_samples["baseline"])
    @printf("  Share explained:\n")
    @printf("    S(x) only:          %6.1f%%\n", S_share * 100)
    @printf("    a(z) only:          %6.1f%%\n", a_share * 100)
    @printf("    S(x) + a(z) joint:  %6.1f%%\n", joint_share * 100)
    @printf("    Interaction:        %6.1f%%\n", interaction * 100)
    @printf("    Residual (Tobin q): %6.1f%%\n", residual * 100)
    flush(stdout)
end

# ============================================================================
# Save Results
# ============================================================================

results_path = joinpath(output_dir, "counterfactual_results.jls")
Serialization.serialize(results_path, Dict(
    "all_results" => all_results,
    "shock_scales" => shock_scales,
    "variant_labels" => variant_labels,
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

sorted_scales = sort(shock_scales)

# --- Figure 1: Bar chart of nonlinearity shares ---
fig1 = plot(size=(700, 450), margin=5Plots.mm, bottom_margin=12Plots.mm)

S_vals = [all_results[s]["S_share"] * 100 for s in sorted_scales]
a_vals = [all_results[s]["a_share"] * 100 for s in sorted_scales]
joint_vals = [all_results[s]["joint_share"] * 100 for s in sorted_scales]
resid_vals = [all_results[s]["residual"] * 100 for s in sorted_scales]

plot!(fig1, sorted_scales, S_vals,
    label="S(x) quadratic IAC", color=:red, marker=:circle, markersize=5)
plot!(fig1, sorted_scales, a_vals,
    label="a(z) exponential util.", color=:blue, marker=:diamond, markersize=5)
plot!(fig1, sorted_scales, joint_vals,
    label="S(x) + a(z) joint", color=:purple, marker=:square, markersize=5)
plot!(fig1, sorted_scales, resid_vals,
    label="Residual (Tobin q, Kimball)", color=:gray, marker=:triangle, markersize=5,
    linestyle=:dash)

plot!(fig1,
    xlabel="Shock scale",
    ylabel="Share of FOM-ROM1 gap explained (%)",
    title="Investment-block nonlinearity decomposition",
    legend=:right,
    ylims=(-5, 105),
)

savefig(fig1, joinpath(FIG_DIR, "counterfactual_decomposition.pdf"))
println("Saved: counterfactual_decomposition.pdf")

# --- Figure 2: Stacked area chart ---
fig2 = plot(size=(700, 450), margin=5Plots.mm, bottom_margin=12Plots.mm)

# Stacked: S(x) alone, a(z) alone, interaction, residual
interaction_vals = [all_results[s]["interaction"] * 100 for s in sorted_scales]
# Ensure non-negative for stacking (interaction can be negative)
S_pure = S_vals .- interaction_vals  # S exclusive of interaction
a_pure = a_vals .- interaction_vals  # a exclusive of interaction

areaplot!(fig2, sorted_scales,
    hcat(max.(S_pure, 0), max.(interaction_vals, 0), max.(a_pure, 0), max.(resid_vals, 0)),
    labels=["S(x) exclusive" "S×a interaction" "a(z) exclusive" "Residual"],
    fillcolor=[:red :orange :blue :gray],
    fillalpha=0.6,
    xlabel="Shock scale",
    ylabel="Share of FOM-ROM1 gap (%)",
    title="Decomposition of investment-block nonlinearity",
)

savefig(fig2, joinpath(FIG_DIR, "counterfactual_stacked.pdf"))
println("Saved: counterfactual_stacked.pdf")

# ============================================================================
# Summary Table
# ============================================================================

println("\n" * "=" ^ 78)
println("SUMMARY TABLE")
println("=" ^ 78)
@printf("%-8s  %6s  %6s  %7s  %7s  %7s  %7s  %7s\n",
    "Scale", "N_base", "Conv", "S(x)%", "a(z)%", "Joint%", "Inter%", "Resid%")
println("-" ^ 75)

for s in sorted_scales
    r = all_results[s]
    @printf("%-8.2f  %6d  %5d  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f\n",
        s, r["n_samples"]["baseline"], r["n_converged_theta"],
        r["S_share"]*100, r["a_share"]*100, r["joint_share"]*100,
        r["interaction"]*100, r["residual"]*100)
end

println("\n" * "=" ^ 78)
println("DONE: $(now())")
println("=" ^ 78)
