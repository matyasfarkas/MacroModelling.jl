#!/usr/bin/env julia
# ============================================================================
# HLT KIMBALL CURVATURE SENSITIVITY
# ============================================================================
#
# Re-runs the SEP-vs-ROM1 equation-block decomposition in the high-Kimball
# region emphasized by Harding, Linde, and Trabandt (JME 2022). The maintained
# repo model sets curvp=10 by default, while the HLT/JME region and our HLT
# posterior discussion put the price Kimball curvature around 75--95.
#
# By default, the diagnostic overrides curvp, and optionally cprobp and cfc,
# while keeping the rest of each posterior draw fixed. Passing
# --no-chain-theta and --param-overrides=... instead evaluates a fixed
# calibration, which is useful for HLT/JME neighborhood checks.
#
# Usage:
#   julia --project=. scripts/kimball_curvature_sensitivity.jl --n-thetas=3 --sim-periods=12
#   julia --project=. scripts/kimball_curvature_sensitivity.jl --curvp-values=10,50.1,64.5,77.3,84.2,94.3
#   julia --project=. scripts/kimball_curvature_sensitivity.jl --curvp-values=77.3 --cprobp-values=0.667 --cfc-values=1.2
#   julia --project=. scripts/kimball_curvature_sensitivity.jl --no-chain-theta --param-overrides=ctou=0.025,crr=0.73
# ============================================================================

using MacroModelling
using Random
using Serialization
using Dates
import Statistics: mean, std
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

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
output_dir     = parse_arg_string(ARGS, "--out",
                    joinpath(REPO_ROOT, ".local_artifacts", "kimball_curvature_sensitivity"))
verbose        = "--verbose" in ARGS
use_chain_theta = !("--no-chain-theta" in ARGS)

function parse_param_overrides(raw::String)
    overrides = Dict{Symbol, Float64}()
    isempty(strip(raw)) && return overrides
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        parts = split(s, "="; limit=2)
        length(parts) == 2 || error("Malformed parameter override '$s'. Use name=value.")
        name = Symbol(strip(parts[1]))
        value = parse(Float64, strip(parts[2]))
        overrides[name] = value
    end
    return overrides
end

function format_param_overrides(overrides::Dict{Symbol, Float64})
    isempty(overrides) && return "none"
    pairs = sort(collect(overrides); by = p -> String(p[1]))
    return join(["$(p[1])=$(p[2])" for p in pairs], ", ")
end

function parse_optional_values(flag::String)
    raw = parse_arg_string(ARGS, flag, "")
    isempty(strip(raw)) && return Union{Nothing, Float64}[nothing]
    vals = Float64[]
    for s in split(raw, ",")
        ss = strip(s)
        isempty(ss) && continue
        push!(vals, parse(Float64, ss))
    end
    return Union{Nothing, Float64}[v for v in vals]
end

curvp_values_raw = parse_arg_string(ARGS, "--curvp-values", "")
curvp_values = isempty(strip(curvp_values_raw)) ?
    [10.0, 50.1, 64.5, 72.2, 77.3, 84.2, 94.3] :
    parse.(Float64, split(curvp_values_raw, ","))

cprobp_values = parse_optional_values("--cprobp-values")
cfc_values = parse_optional_values("--cfc-values")
param_overrides = parse_param_overrides(parse_arg_string(ARGS, "--param-overrides", ""))

optional_values_label(vals, fallback) = vals == Union{Nothing, Float64}[nothing] ? fallback : string(vals)

mkpath(output_dir)

println("=" ^ 78)
println("HLT KIMBALL CURVATURE SENSITIVITY")
println("Started: $(now())")
println("=" ^ 78)
println("  curvp values:   $curvp_values")
println("  cprobp values:  $(optional_values_label(cprobp_values, "posterior draw"))")
println("  cfc values:     $(optional_values_label(cfc_values, "model/draw value"))")
println("  Theta source:   $(use_chain_theta ? "posterior chain" : "fixed model calibration")")
println("  Param overrides: $(format_param_overrides(param_overrides))")
println("  Thetas:         $n_thetas")
println("  Sim periods:    $sim_periods + $burn_in burn-in")
println("  Output:         $output_dir")
flush(stdout)

println("\nLoading HLT OBC model...")
const HLT = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
MacroModelling.solve!(HLT, silent = true)
println("  Model: $(HLT.model_name), $(length(HLT.var)) variables, $(length(HLT.exo)) shocks")

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = Int.(indexin(observables, HLT.var))
@assert all(!isnothing, obs_idx)
d_obs = length(obs_idx)
d_eps = length(HLT.exo)

curvp_pidx = findfirst(==(:curvp), HLT.parameters)
cprobp_pidx = findfirst(==(:cprobp), HLT.parameters)
cfc_pidx = findfirst(==(:cfc), HLT.parameters)
curvp_pidx === nothing && error("curvp not found in model parameters")
cprobp_pidx === nothing && error("cprobp not found in model parameters")
cfc_pidx === nothing && error("cfc not found in model parameters")

param_override_indices = Dict{Int, Float64}()
for (pname, pval) in param_overrides
    pidx = findfirst(==(pname), HLT.parameters)
    pidx === nothing && error("Override parameter '$pname' not found in model parameters")
    param_override_indices[pidx] = pval
end

println("  Baseline curvp=$(HLT.parameter_values[curvp_pidx]), cprobp=$(HLT.parameter_values[cprobp_pidx]), cfc=$(HLT.parameter_values[cfc_pidx])")

specs = get_phase1_18param_specs()
theta_names = [s.name for s in specs]

if use_chain_theta
    if chain_path == ""
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

    if chain_theta_names != theta_names
        println("  Reindexing parameters...")
        reindex = [findfirst(==(n), chain_theta_names) for n in theta_names]
        @assert all(!isnothing, reindex)
        chain_matrix = chain_matrix[:, reindex]
    end

    n_draws_total = size(chain_matrix, 1)
    println("  Chain: $(n_draws_total) draws x $(size(chain_matrix, 2)) parameters")

    param_means = vec(mean(chain_matrix, dims=1))
    param_stds = vec(std(chain_matrix, dims=1))
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
else
    chain_path = "(fixed model calibration; --no-chain-theta)"
    theta_grid = zeros(Float64, n_thetas, 0)
    actual_n_thetas = n_thetas
    println("\nUsing fixed model calibration with $actual_n_thetas shock replications")
end

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
    total_sq = mean(vec(sum(delta_obs.^2, dims=1)))
    total_sq < 1e-30 && return Dict(bn => 0.0 for bn in block_names)

    shares = Dict{String, Float64}()
    for bname in block_names
        positions = [findfirst(==(v), obs_syms) for v in obs_block_map[bname]]
        filter!(!isnothing, positions)
        block_sq = isempty(positions) ? 0.0 : mean(vec(sum(delta_obs[positions, :].^2, dims=1)))
        shares[bname] = block_sq / total_sq
    end
    return shares
end

function draw_shocks(rng::AbstractRNG, model, total_periods::Int)
    shock_names_local = model.exo
    shocks = zeros(length(shock_names_local), total_periods)
    obc_mask = contains.(string.(shock_names_local), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    isempty(structural_idx) && return shocks
    sigmas = ones(length(structural_idx))
    shocks[structural_idx, :] .= Diagonal(sigmas) * randn(rng, length(structural_idx), total_periods)
    return shocks
end

function setting_label(curvp, cprobp, cfc)
    cp = cprobp === nothing ? "draw" : @sprintf("%.3f", cprobp)
    cf = cfc === nothing ? "model" : @sprintf("%.3f", cfc)
    return @sprintf("curvp=%.1f,cprobp=%s,cfc=%s", curvp, cp, cf)
end

results = Dict{String, Dict{String, Any}}()
settings = Tuple{Float64, Union{Nothing, Float64}, Union{Nothing, Float64}}[]
for curvp in curvp_values, cprobp in cprobp_values, cfc in cfc_values
    push!(settings, (curvp, cprobp, cfc))
end

println("\n" * "=" ^ 78)
println("SIMULATING KIMBALL SETTINGS")
println("=" ^ 78)

for (si, (curvp, cprobp, cfc)) in enumerate(settings)
    label = setting_label(curvp, cprobp, cfc)
    println("\n--- $label ($si/$(length(settings))) ---")
    flush(stdout)

    max_samples = actual_n_thetas * sim_periods
    delta_obs_matrix = zeros(d_obs, max_samples)
    cursor = 0
    n_converged = 0
    n_rom_fail = 0
    n_sep_fail = 0

    for (ti, theta_row) in enumerate(eachrow(theta_grid))
        params = Float64.(HLT.parameter_values)
        if use_chain_theta
            for (j, tname) in enumerate(theta_names)
                pidx = findfirst(==(tname), HLT.parameters)
                pidx !== nothing && (params[pidx] = Float64(theta_row[j]))
            end
        end
        for (pidx, pval) in param_override_indices
            params[pidx] = pval
        end
        params[curvp_pidx] = curvp
        cprobp !== nothing && (params[cprobp_pidx] = cprobp)
        cfc !== nothing && (params[cfc_pidx] = cfc)

        local rom_cache
        try
            rom_cache = build_rom_cache(HLT, 1; params=params, use_obc=true)
        catch e
            n_rom_fail += 1
            verbose && println("  [theta $ti] ROM build failed: $e")
            continue
        end

        trajectory_seed = seed0 * 100_000 + si * 1_000 + ti
        total_periods = sim_periods + burn_in
        shocks = draw_shocks(MersenneTwister(trajectory_seed), HLT, total_periods)

        try
            MacroModelling.write_parameters_input!(HLT, params, verbose = false)
        catch e
            n_sep_fail += 1
            verbose && println("  [theta $ti] Parameter write failed: $e")
            continue
        end

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
                n_sep_fail += 1
                verbose && println("  [theta $ti] SEP failed: $e")
                continue
            else
                rethrow()
            end
        end

        if res.errorflag
            sep_errors = hasproperty(res, :sep_errors) ? res.sep_errors : Float64[]
            valid_errors = filter(isfinite, sep_errors)
            if isempty(valid_errors) || !all(e -> e <= sep_accept_tol, valid_errors)
                n_sep_fail += 1
                verbose && println("  [theta $ti] SEP diverged")
                continue
            end
        end

        n_converged += 1
        sim = Array(res.simulation)
        sim_shocks = res.shocks
        T_avail = min(sim_periods, size(sim, 2) - 1)

        for t in 1:T_avail
            local rom_next
            try
                rom_next = rom_step_full(rom_cache, sim[:, t], sim_shocks[:, t])
            catch
                continue
            end
            cursor += 1
            cursor > size(delta_obs_matrix, 2) && (delta_obs_matrix = hcat(delta_obs_matrix, zeros(d_obs, max_samples)))
            delta_obs_matrix[:, cursor] = sim[obs_idx, t + 1] .- rom_next[obs_idx]
        end
    end

    delta_obs = delta_obs_matrix[:, 1:cursor]
    shares = compute_block_shares(delta_obs, observables)
    mean_gap = cursor > 0 ? mean(sqrt.(vec(sum(delta_obs.^2, dims=1)))) : 0.0
    obs_rmse = cursor > 0 ? [sqrt(mean(delta_obs[i, :].^2)) for i in 1:d_obs] : zeros(d_obs)

    results[label] = Dict(
        "curvp" => curvp,
        "cprobp" => cprobp,
        "cfc" => cfc,
        "shares" => shares,
        "n_samples" => cursor,
        "n_converged" => n_converged,
        "n_rom_fail" => n_rom_fail,
        "n_sep_fail" => n_sep_fail,
        "mean_gap" => mean_gap,
        "obs_rmse" => obs_rmse,
    )

    @printf("  Samples: %d, converged thetas: %d/%d, ROM fails: %d, SEP fails: %d\n",
            cursor, n_converged, actual_n_thetas, n_rom_fail, n_sep_fail)
    @printf("  Mean |delta|: %.6f\n", mean_gap)
    for bn in block_names
        @printf("    %-22s %6.2f%%\n", bn, shares[bn] * 100)
    end
    flush(stdout)
end

results_path = joinpath(output_dir, "kimball_curvature_sensitivity.jls")
Serialization.serialize(results_path, Dict(
    "results" => results,
    "block_names" => block_names,
    "curvp_values" => curvp_values,
    "cprobp_values" => cprobp_values,
    "cfc_values" => cfc_values,
    "n_thetas" => actual_n_thetas,
    "sim_periods" => sim_periods,
    "burn_in" => burn_in,
    "chain_path" => chain_path,
    "use_chain_theta" => use_chain_theta,
    "param_overrides" => Dict(String(k) => v for (k, v) in param_overrides),
    "timestamp" => now(),
    "source_note" => "Harding, Linde, and Trabandt (JME 2022) report high Kimball curvature around epsilon_p=77.3 at gross markup 1.20. Optional --param-overrides can map their reported calibration into the maintained HLT OBC model.",
))
println("\nResults saved: $results_path")

csv_path = joinpath(output_dir, "kimball_curvature_sensitivity.csv")
open(csv_path, "w") do io
    println(io, "setting,curvp,cprobp,cfc,n_samples,n_converged,n_rom_fail,n_sep_fail,mean_gap,investment,consumption,output,labor,price,wage,taylor")
    for label in sort(collect(keys(results)))
        r = results[label]
        s = r["shares"]
        @printf(io, "\"%s\",%.6f,%s,%s,%d,%d,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
            label,
            r["curvp"],
            r["cprobp"] === nothing ? "" : @sprintf("%.6f", r["cprobp"]),
            r["cfc"] === nothing ? "" : @sprintf("%.6f", r["cfc"]),
            r["n_samples"],
            r["n_converged"],
            r["n_rom_fail"],
            r["n_sep_fail"],
            r["mean_gap"],
            s["Investment/Capital"] * 100,
            s["Consumption/Euler"] * 100,
            s["Output/Resource"] * 100,
            s["Labor Market"] * 100,
            s["Price Phillips"] * 100,
            s["Wage Phillips"] * 100,
            s["Taylor Rule"] * 100,
        )
    end
end
println("CSV saved: $csv_path")

md_path = joinpath(output_dir, "KIMBALL_CURVATURE_SENSITIVITY_SUMMARY.md")
open(md_path, "w") do io
    println(io, "# Kimball Curvature Sensitivity")
    println(io)
    println(io, "**Generated:** $(now())")
    println(io)
    println(io, "This diagnostic reruns the SEP-vs-ROM1 equation-block decomposition around the high-Kimball region emphasized by Harding, Linde, and Trabandt (JME 2022).")
    println(io)
    println(io, "- Parameter source: `$(use_chain_theta ? "posterior chain" : "fixed model calibration")`")
    println(io, "- Chain: `$chain_path`")
    println(io, "- Parameter overrides: `$(format_param_overrides(param_overrides))`")
    println(io, "- Theta/shock replications: `$actual_n_thetas`")
    println(io, "- Periods per theta: `$sim_periods` after `$burn_in` burn-in")
    println(io, "- SEP: horizon `$sep_horizon`, maxit `$sep_maxit`, tol `$sep_tol`, accept_tol `$sep_accept_tol`")
    println(io)
    println(io, "| Setting | N | Conv. | Mean gap | Invest | Cons. | Output | Labor | Price | Wage | Taylor |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in sort(collect(keys(results)))
        r = results[label]
        s = r["shares"]
        @printf(io, "| `%s` | %d | %d/%d | %.4f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
            label,
            r["n_samples"],
            r["n_converged"],
            actual_n_thetas,
            r["mean_gap"],
            s["Investment/Capital"] * 100,
            s["Consumption/Euler"] * 100,
            s["Output/Resource"] * 100,
            s["Labor Market"] * 100,
            s["Price Phillips"] * 100,
            s["Wage Phillips"] * 100,
            s["Taylor Rule"] * 100,
        )
    end
    println(io)
    println(io, "Shares are percentages of the mean squared observable FOM-ROM1 gap.")
end
println("Summary saved: $md_path")

println("\n" * "=" ^ 78)
println("DONE: $(now())")
println("=" ^ 78)
