#!/usr/bin/env julia
# ============================================================================
# HLT MECHANISM ABLATION DECOMPOSITION
# ============================================================================
#
# Isolates the contribution of candidate nonlinear mechanisms to the FOM-ROM1
# gap by running SEP and ROM1 on model variants where specific nonlinearities
# are switched off or linearized:
#
#   Baseline:     All nonlinearities active (original HLT OBC model)
#   noS:          S(x) = 0, S'(x) = 0  (quadratic investment adj cost off)
#   lina:         a(z) = rk_ss*(z-1)    (exponential utilization cost linearized)
#   noS_lina:     Both S and a linearized (residual = Tobin's q + Kimball)
#   price_ces:    price Kimball curvature set to CES value curvp=1
#   wage_ces:     wage Kimball curvature set to CES value curvw=1
#   pw_ces:       price and wage Kimball curvatures set to CES values
#   elb_off:      rate floor set far below the relevant state space
#
# Methodology:
#   For each (theta, shock_scale), run SEP on each variant with identical shock
#   seeds and the same maintained parameter overrides.
#   Build ROM1 for each variant and compute delta_V = FOM_V - ROM1_V.
#   Share explained by removing X = 1 - ||delta_noX||^2 / ||delta_baseline||^2
#
# Usage:
#   julia --project=. scripts/hlt_counterfactual_decomposition.jl
#   julia --project=. scripts/hlt_counterfactual_decomposition.jl --n-thetas=3
#   julia --project=. scripts/hlt_counterfactual_decomposition.jl --no-chain-theta --variant-suite=all
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
variant_suite  = parse_arg_string(ARGS, "--variant-suite", "investment")
use_chain_theta = !("--no-chain-theta" in ARGS)
verbose        = "--verbose" in ARGS

function parse_float_list(raw::String)
    vals = Float64[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        push!(vals, parse(Float64, s))
    end
    return vals
end

function parse_int_list(raw::String)
    vals = Int[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    return vals
end

function parse_symbol_or_nothing(raw::String)
    s = lowercase(strip(raw))
    s in ("", "none", "nothing", "null") && return nothing
    return Symbol(s)
end

function parse_param_overrides(raw::String)
    overrides = Dict{Symbol, Float64}()
    isempty(strip(raw)) && return overrides
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        parts = split(s, "="; limit=2)
        length(parts) == 2 || error("Malformed parameter override '$s'. Use name=value.")
        overrides[Symbol(strip(parts[1]))] = parse(Float64, strip(parts[2]))
    end
    return overrides
end

function format_param_overrides(overrides::Dict{Symbol, Float64})
    isempty(overrides) && return "none"
    pairs = sort(collect(overrides); by = p -> String(p[1]))
    return join(["$(p[1])=$(p[2])" for p in pairs], ", ")
end

param_overrides = parse_param_overrides(parse_arg_string(ARGS, "--param-overrides", ""))

shock_scales_str = parse_arg_string(ARGS, "--shock-scales", "")
if shock_scales_str != ""
    shock_scales = parse_float_list(shock_scales_str)
else
    shock_scales = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
end

sep_recovery = parse_arg_bool(ARGS, "--sep-recovery", true)
sep_recovery_scales = parse_float_list(parse_arg_string(
    ARGS, "--sep-recovery-scales", "0.0,0.05,0.1,0.25,0.5,0.75,1.0"))
sep_fallback_solver = parse_symbol_or_nothing(parse_arg_string(ARGS, "--sep-fallback-solver", "qr"))
sep_retry = parse_arg_bool(ARGS, "--sep-retry", true)
retry_maxit_multipliers = parse_float_list(parse_arg_string(ARGS, "--sep-retry-maxit-multipliers", "1,2,4"))
retry_horizons_raw = parse_arg_string(ARGS, "--sep-retry-horizons", "")
retry_horizons = isempty(strip(retry_horizons_raw)) ?
    unique([sep_horizon, max(sep_horizon, 6), max(sep_horizon, 8)]) :
    parse_int_list(retry_horizons_raw)
checkpoint_every = parse_arg_int(ARGS, "--checkpoint-every", 1)

mkpath(output_dir)
rng = MersenneTwister(seed0)

println("=" ^ 78)
println("HLT MECHANISM ABLATION DECOMPOSITION")
println("Started: $(now())")
println("=" ^ 78)
println("  Shock scales: $shock_scales")
println("  Thetas:       $n_thetas")
println("  Theta source: $(use_chain_theta ? "posterior chain" : "fixed model calibration")")
println("  Variant suite: $variant_suite")
println("  Param overrides: $(format_param_overrides(param_overrides))")
println("  Sim periods:  $sim_periods + $burn_in burn-in")
println("  SEP retry:    $(sep_retry), recovery=$(sep_recovery), fallback=$(isnothing(sep_fallback_solver) ? "none" : sep_fallback_solver)")
println("  Output:       $output_dir")
flush(stdout)

struct VariantSpec
    label::String
    model_label::String
    overrides::Dict{Symbol, Float64}
    description::String
end

function local_override(p::Symbol, fallback::Float64, mult::Float64)
    return Dict(p => get(param_overrides, p, fallback) * mult)
end

function build_variant_specs(suite::String)
    investment = VariantSpec[
        VariantSpec("baseline", "baseline", Dict{Symbol, Float64}(), "all maintained nonlinearities active"),
        VariantSpec("noS", "noS", Dict{Symbol, Float64}(), "level investment adjustment-cost function set to zero"),
        VariantSpec("lina", "lina", Dict{Symbol, Float64}(), "capital-utilization cost linearized around steady state"),
        VariantSpec("noS_lina", "noS_lina", Dict{Symbol, Float64}(), "investment adjustment cost off and utilization cost linearized"),
    ]
    pricing = VariantSpec[
        VariantSpec("price_ces", "baseline", Dict(:curvp => 1.0), "price Kimball curvature switched to CES value curvp=1"),
        VariantSpec("wage_ces", "baseline", Dict(:curvw => 1.0), "wage Kimball curvature switched to CES value curvw=1"),
        VariantSpec("pw_ces", "baseline", Dict(:curvp => 1.0, :curvw => 1.0), "price and wage Kimball curvature switched to CES values"),
    ]
    policy = VariantSpec[
        VariantSpec("elb_off", "baseline", Dict(:R_bar => -100.0), "effective lower bound moved far below the relevant state space"),
    ]
    local10 = VariantSpec[
        VariantSpec("baseline", "baseline", Dict{Symbol, Float64}(), "all maintained nonlinearities active"),
        VariantSpec("csadjcost_m10", "baseline", local_override(:csadjcost, 4.89, 0.9), "investment adjustment-cost parameter reduced by 10%"),
        VariantSpec("csadjcost_p10", "baseline", local_override(:csadjcost, 4.89, 1.1), "investment adjustment-cost parameter increased by 10%"),
        VariantSpec("czcap_m10", "baseline", local_override(:czcap, 0.431818, 0.9), "capital-utilization curvature parameter reduced by 10%"),
        VariantSpec("czcap_p10", "baseline", local_override(:czcap, 0.431818, 1.1), "capital-utilization curvature parameter increased by 10%"),
        VariantSpec("curvp_m10", "baseline", local_override(:curvp, 77.3, 0.9), "price Kimball curvature reduced by 10%"),
        VariantSpec("curvp_p10", "baseline", local_override(:curvp, 77.3, 1.1), "price Kimball curvature increased by 10%"),
        VariantSpec("curvw_m10", "baseline", local_override(:curvw, 8.31, 0.9), "wage Kimball curvature reduced by 10%"),
        VariantSpec("curvw_p10", "baseline", local_override(:curvw, 8.31, 1.1), "wage Kimball curvature increased by 10%"),
    ]
    if suite == "investment"
        return investment
    elseif suite == "baseline"
        return investment[1:1]
    elseif suite == "local10" || suite == "stability"
        return local10
    elseif suite == "pricing"
        return vcat(investment[1:1], pricing)
    elseif suite == "policy"
        return vcat(investment[1:1], policy)
    elseif suite == "all"
        return vcat(investment, pricing, policy)
    else
        error("Unknown --variant-suite=$suite. Use baseline, investment, local10, stability, pricing, policy, or all.")
    end
end

variant_specs = build_variant_specs(variant_suite)
variant_labels = [v.label for v in variant_specs]
variant_by_label = Dict(v.label => v for v in variant_specs)

# ============================================================================
# Load Required Model Variants
# ============================================================================

all_model_specs = [
    ("baseline",  "Smets_Wouters_2007_HLT_obc.jl",           :Smets_Wouters_2007_HLT_obc),
    ("noS",       "Smets_Wouters_2007_HLT_obc_noS.jl",       :Smets_Wouters_2007_HLT_obc_noS),
    ("lina",      "Smets_Wouters_2007_HLT_obc_lina.jl",      :Smets_Wouters_2007_HLT_obc_lina),
    ("noS_lina",  "Smets_Wouters_2007_HLT_obc_noS_lina.jl",  :Smets_Wouters_2007_HLT_obc_noS_lina),
]

required_model_labels = Set(v.model_label for v in variant_specs)
push!(required_model_labels, "baseline")
model_specs = [spec for spec in all_model_specs if spec[1] in required_model_labels]

models = Dict{String, Any}()
println("\nRequired model files: $(join(first.(model_specs), ", "))")
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

manifest_path = joinpath(output_dir, "RUN_MANIFEST.md")
open(manifest_path, "w") do io
    println(io, "# HLT Mechanism Ablation Production Run")
    println(io)
    println(io, "**Started:** $(now())")
    println(io)
    println(io, "## Command")
    println(io)
    println(io, "```")
    println(io, join(Base.julia_cmd().exec, " ") * " " * join(ARGS, " "))
    println(io, "```")
    println(io)
    println(io, "## Design")
    println(io)
    println(io, "- Variant suite: `$variant_suite`")
    println(io, "- Shock scales: `$(join(shock_scales, ", "))`")
    println(io, "- Theta/shock replications: `$n_thetas`")
    println(io, "- Parameter source: `$(use_chain_theta ? "posterior chain" : "fixed model calibration")`")
    println(io, "- Maintained parameter overrides: `$(format_param_overrides(param_overrides))`")
    println(io, "- Periods: `$sim_periods` after `$burn_in` burn-in")
    println(io, "- SEP base horizon/maxit/tol/accept_tol: `$sep_horizon` / `$sep_maxit` / `$sep_tol` / `$sep_accept_tol`")
    println(io, "- Retry enabled: `$sep_retry`; retry horizons: `$(join(retry_horizons, ", "))`; maxit multipliers: `$(join(retry_maxit_multipliers, ", "))`")
    println(io, "- SEP recovery: `$sep_recovery`; recovery scales: `$(join(sep_recovery_scales, ", "))`; fallback solver: `$(isnothing(sep_fallback_solver) ? "none" : sep_fallback_solver)`")
    println(io)
    println(io, "## Variant Definitions")
    println(io)
    println(io, "| Variant | Model | Switch |")
    println(io, "|---|---|---|")
    for v in variant_specs
        println(io, "| `$(v.label)` | `$(v.model_label)` | $(v.description) |")
    end
end
println("Run manifest: $manifest_path")

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

if use_chain_theta && chain_path == ""
    chain_path = joinpath(REPO_ROOT,
        ".local_artifacts/hlt_18param_realdata/hlt_switching_synthetic_chain_2000.jls")
    if !isfile(chain_path)
        chain_path = joinpath(REPO_ROOT,
            ".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_chain_2000_seed99.jls")
    end
end

if use_chain_theta
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
    theta_source_indices = selected_idx
    actual_n_thetas = size(theta_grid, 1)
    println("  Selected $actual_n_thetas representative thetas")
else
    chain_path = "(fixed model calibration; --no-chain-theta)"
    theta_grid = zeros(Float64, n_thetas, 0)
    theta_source_indices = collect(1:n_thetas)
    actual_n_thetas = n_thetas
    println("\nUsing fixed model calibration with $actual_n_thetas shock replications")
end
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
    sigmas = fill(shock_scale, length(structural_idx))
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

function finite_simulation(res)
    res === nothing && return false
    try
        sim = Array(res.simulation)
        return !isempty(sim) && all(isfinite, sim)
    catch
        return false
    end
end

function max_sep_error(res)
    res === nothing && return Inf
    if hasproperty(res, :sep_errors)
        vals = Float64[]
        for e in res.sep_errors
            isfinite(e) && push!(vals, abs(Float64(e)))
        end
        !isempty(vals) && return maximum(vals)
    end
    return getproperty(res, :errorflag) ? Inf : 0.0
end

function sep_solution_accepted(res, accept_tol::Float64)
    finite_simulation(res) || return false
    # MacroModelling sets errorflag when Newton does not meet sep_tol. We keep
    # mildly flagged paths only when all retained residuals satisfy the explicit
    # production acceptance tolerance.
    return !res.errorflag || max_sep_error(res) <= accept_tol
end

function recovery_count(res)
    res === nothing && return 0
    hasproperty(res, :sep_recovery_log) || return 0
    return length(res.sep_recovery_log)
end

function build_sep_attempts(sep_horizon::Int, sep_maxit::Int;
                            sep_retry::Bool,
                            retry_horizons::Vector{Int},
                            retry_maxit_multipliers::Vector{Float64},
                            sep_recovery::Bool)
    attempts = NamedTuple[]
    horizons = sep_retry ? retry_horizons : [sep_horizon]
    multipliers = sep_retry ? retry_maxit_multipliers : [1.0]
    for h in horizons
        for mult in multipliers
            maxit_i = max(1, ceil(Int, sep_maxit * mult))
            push!(attempts, (
                label = "h$(h)_m$(maxit_i)_primary",
                horizon = h,
                maxit = maxit_i,
                lm_lambda = 1e-8,
                line_search_maxit = 6,
                recovery = false,
            ))
        end
    end
    if sep_recovery
        h = maximum(horizons)
        maxit_i = max(1, ceil(Int, sep_maxit * maximum(multipliers)))
        push!(attempts, (
            label = "h$(h)_m$(maxit_i)_recovery",
            horizon = h,
            maxit = maxit_i,
            lm_lambda = 1e-6,
            line_search_maxit = 10,
            recovery = true,
        ))
    end
    return attempts
end

function run_sep_on_model(mdl, shocks_matrix, sim_periods, burn_in, seed_val;
                          sep_horizon=10, sep_maxit=80, sep_tol=1e-5,
                          sep_accept_tol=0.35, sep_retry=true,
                          retry_horizons=[sep_horizon], retry_maxit_multipliers=[1.0],
                          sep_recovery=true, sep_recovery_scales=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                          sep_fallback_solver=nothing)
    attempts = build_sep_attempts(sep_horizon, sep_maxit;
        sep_retry=sep_retry,
        retry_horizons=retry_horizons,
        retry_maxit_multipliers=retry_maxit_multipliers,
        sep_recovery=sep_recovery)
    best = (res = nothing, accepted = false, attempt = "none", max_error = Inf,
            recovery_count = 0, status = "not_run", elapsed = 0.0)

    for attempt in attempts
        t0 = time()
        local res
        try
            res = MacroModelling.simulate_sep_extended_path(
                mdl;
                periods          = sim_periods,
                burn_in          = burn_in,
                sep_horizon      = attempt.horizon,
                sep_order        = 1,
                sep_nnodes       = 3,
                sep_maxit        = attempt.maxit,
                sep_tol          = sep_tol,
                sep_sparse_tree  = true,
                sep_linear_solver = :normal_equations,
                sep_fallback_solver = sep_fallback_solver,
                sep_stall_iters  = 25,
                sep_stall_rel_tol = 1e-4,
                sep_stall_abs_tol = 1e-10,
                sep_line_search  = true,
                sep_line_search_maxit = attempt.line_search_maxit,
                sep_line_search_factor = 0.5,
                sep_line_search_min_alpha = 1e-4,
                sep_lm_lambda    = attempt.lm_lambda,
                sep_lm_lambda_scale = 10.0,
                sep_lm_lambda_min = 1e-12,
                sep_lm_lambda_max = 1e4,
                sep_shock_scale  = 1.0,
                sep_accept_tol   = sep_accept_tol,
                sep_recovery     = attempt.recovery,
                sep_recovery_scales = sep_recovery_scales,
                shock_scaling    = :none,
                shocks           = shocks_matrix,
                random_seed      = seed_val,
                silent           = true,
            )
        catch e
            if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException || e isa DomainError || e isa ArgumentError
                elapsed = time() - t0
                status = "$(typeof(e))"
                if verbose
                    @printf("      attempt %-24s exception %s after %.1fs\n", attempt.label, status, elapsed)
                end
                continue
            else
                rethrow()
            end
        end

        elapsed = time() - t0
        err = max_sep_error(res)
        accepted = sep_solution_accepted(res, sep_accept_tol)
        status = accepted ? "accepted" : "rejected"
        rec_ct = recovery_count(res)
        if err < best.max_error || accepted
            best = (res = res, accepted = accepted, attempt = attempt.label,
                    max_error = err, recovery_count = rec_ct, status = status,
                    elapsed = elapsed)
        end
        if verbose
            @printf("      attempt %-24s %-8s err=%.3e rec=%d time=%.1fs\n",
                    attempt.label, status, err, rec_ct, elapsed)
        end
        accepted && return best
    end
    return best
end

const LOCAL_SENSITIVITY_PAIRS = [
    (parameter = :csadjcost, minus = "csadjcost_m10", plus = "csadjcost_p10",
     description = "investment adjustment-cost curvature"),
    (parameter = :czcap, minus = "czcap_m10", plus = "czcap_p10",
     description = "capital-utilization curvature"),
    (parameter = :curvp, minus = "curvp_m10", plus = "curvp_p10",
     description = "price Kimball curvature"),
    (parameter = :curvw, minus = "curvw_m10", plus = "curvw_p10",
     description = "wage Kimball curvature"),
]

function active_local_sensitivity_pairs(labels::Vector{String})
    label_set = Set(labels)
    return [p for p in LOCAL_SENSITIVITY_PAIRS if p.minus in label_set && p.plus in label_set]
end

# ============================================================================
# Main Loop: Counterfactual Decomposition
# ============================================================================

println("\n" * "=" ^ 78)
println("COUNTERFACTUAL SIMULATIONS")
println("=" ^ 78)

# Results storage: scale → Dict with variant L² norms and share explained
all_results = Dict{Float64, Dict{String, Any}}()

for (si, scale) in enumerate(shock_scales)
    println("\n--- Shock scale $scale ($si/$(length(shock_scales))) ---")
    flush(stdout)

    # Accumulate squared deltas per variant
    sq_norms = Dict(v => 0.0 for v in variant_labels)
    n_samples = Dict(v => 0 for v in variant_labels)
    n_accepted_paths = Dict(v => 0 for v in variant_labels)
    n_failed_paths = Dict(v => 0 for v in variant_labels)
    max_errors_by_variant = Dict(v => Float64[] for v in variant_labels)
    attempts_by_variant = Dict(v => String[] for v in variant_labels)
    recovery_counts_by_variant = Dict(v => Int[] for v in variant_labels)
    n_converged_theta = 0
    theta_records = Vector{Dict{String, Any}}()

    for (ti, theta_row) in enumerate(eachrow(theta_grid))
        theta = collect(Float64, theta_row)

        # Set parameters on all models
        param_vecs = Dict{String, Vector{Float64}}()
        rom_caches = Dict{String, Any}()
        all_ok = true

        for label in variant_labels
            vspec = variant_by_label[label]
            mdl = models[vspec.model_label]
            params = Float64.(mdl.parameter_values)
            if use_chain_theta
                for (j, tname) in enumerate(theta_names)
                    pidx = findfirst(==(tname), mdl.parameters)
                    pidx !== nothing && (params[pidx] = theta[j])
                end
            end
            for (pname, pval) in param_overrides
                pidx = findfirst(==(pname), mdl.parameters)
                pidx !== nothing && (params[pidx] = pval)
            end
            for (pname, pval) in vspec.overrides
                pidx = findfirst(==(pname), mdl.parameters)
                pidx !== nothing || error("Variant $(vspec.label) override parameter '$pname' not found in model $(mdl.model_name)")
                params[pidx] = pval
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
            vspec = variant_by_label[label]
            mdl = models[vspec.model_label]
            # Need to set parameters before SEP
            MacroModelling.write_parameters_input!(mdl, param_vecs[label], verbose=false)
            MacroModelling.solve!(mdl; algorithm=:first_order, dynamics=true, obc=true, silent=true)

            res = run_sep_on_model(mdl, shock_sets[vspec.model_label], sim_periods, burn_in, trajectory_seed;
                                   sep_horizon=sep_horizon, sep_maxit=sep_maxit,
                                   sep_tol=sep_tol, sep_accept_tol=sep_accept_tol,
                                   sep_retry=sep_retry,
                                   retry_horizons=retry_horizons,
                                   retry_maxit_multipliers=retry_maxit_multipliers,
                                   sep_recovery=sep_recovery,
                                   sep_recovery_scales=sep_recovery_scales,
                                   sep_fallback_solver=sep_fallback_solver)
            push!(max_errors_by_variant[label], res.max_error)
            push!(attempts_by_variant[label], res.attempt)
            push!(recovery_counts_by_variant[label], res.recovery_count)
            if !res.accepted
                n_failed_paths[label] += 1
                @printf("  [scale %.3g theta %d] SEP failed for %-12s best=%s err=%.3e\n",
                        scale, ti, label, res.attempt, res.max_error)
                sep_results[label] = nothing
                continue
            end
            n_accepted_paths[label] += 1
            sep_results[label] = res.res
        end

        # Skip theta if baseline didn't converge
        if sep_results["baseline"] === nothing
            continue
        end
        n_converged_theta += 1

        # Compute FOM-ROM1 deltas for each variant
        theta_sq_norms = Dict(v => 0.0 for v in variant_labels)
        theta_n_samples = Dict(v => 0 for v in variant_labels)
        for label in variant_labels
            sep_results[label] === nothing && continue

            vspec = variant_by_label[label]
            mdl = models[vspec.model_label]
            oidx = obs_indices[vspec.model_label]
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
                theta_sq_norms[label] += sum(delta.^2)
                theta_n_samples[label] += 1
            end
        end

        theta_variant_mse = Dict(label =>
            theta_n_samples[label] > 0 ? theta_sq_norms[label] / theta_n_samples[label] : NaN
            for label in variant_labels)
        theta_variant_rmse = Dict(label =>
            isfinite(theta_variant_mse[label]) && theta_variant_mse[label] >= 0 ?
                sqrt(theta_variant_mse[label]) : NaN
            for label in variant_labels)
        theta_base_mse = theta_variant_mse["baseline"]
        local_elasticities = Dict{String, Float64}()
        local_abs_elasticities = Dict{String, Float64}()
        for pair in active_local_sensitivity_pairs(variant_labels)
            mse_minus = theta_variant_mse[pair.minus]
            mse_plus = theta_variant_mse[pair.plus]
            vminus = variant_by_label[pair.minus]
            vplus = variant_by_label[pair.plus]
            value_minus = vminus.overrides[pair.parameter]
            value_plus = vplus.overrides[pair.parameter]
            elasticity = (isfinite(mse_minus) && isfinite(mse_plus) &&
                          mse_minus > 0 && mse_plus > 0 &&
                          value_minus > 0 && value_plus > 0) ?
                (log(mse_plus) - log(mse_minus)) / (log(value_plus) - log(value_minus)) : NaN
            local_elasticities[String(pair.parameter)] = elasticity
            local_abs_elasticities[String(pair.parameter)] = isfinite(elasticity) ? abs(elasticity) : NaN
        end
        finite_abs = [(k, v) for (k, v) in local_abs_elasticities if isfinite(v)]
        largest_parameter = isempty(finite_abs) ? "" : sort(finite_abs; by = p -> (-p[2], p[1]))[1][1]
        push!(theta_records, Dict(
            "theta_index" => ti,
            "chain_draw_index" => theta_source_indices[ti],
            "variant_mse" => theta_variant_mse,
            "variant_rmse" => theta_variant_rmse,
            "variant_n_samples" => theta_n_samples,
            "baseline_mse" => theta_base_mse,
            "baseline_rmse" => theta_variant_rmse["baseline"],
            "local_elasticities" => local_elasticities,
            "local_abs_elasticities" => local_abs_elasticities,
            "largest_abs_elasticity_parameter" => largest_parameter,
        ))
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

    has_investment_joint = all(haskey(share_explained, k) for k in ("noS", "lina", "noS_lina"))
    S_share = has_investment_joint ? share_explained["noS"] : NaN
    a_share = has_investment_joint ? share_explained["lina"] : NaN
    joint_share = has_investment_joint ? share_explained["noS_lina"] : NaN
    interaction = has_investment_joint ? S_share + a_share - joint_share : NaN
    residual = has_investment_joint ? 1.0 - joint_share : NaN

    all_results[scale] = Dict(
        "sq_norms" => sq_norms,
        "n_samples" => n_samples,
        "n_accepted_paths" => n_accepted_paths,
        "n_failed_paths" => n_failed_paths,
        "n_attempted_paths" => Dict(label => n_accepted_paths[label] + n_failed_paths[label]
                                    for label in variant_labels),
        "max_errors_by_variant" => max_errors_by_variant,
        "attempts_by_variant" => attempts_by_variant,
        "recovery_counts_by_variant" => recovery_counts_by_variant,
        "n_converged_theta" => n_converged_theta,
        "theta_records" => theta_records,
        "share_explained" => share_explained,
        "variant_rmse" => Dict(label => n_samples[label] > 0 ? sqrt(sq_norms[label] / n_samples[label]) : NaN
                                for label in variant_labels),
        "variant_mse" => Dict(label => n_samples[label] > 0 ? sq_norms[label] / n_samples[label] : NaN
                              for label in variant_labels),
        "S_share" => S_share,
        "a_share" => a_share,
        "joint_share" => joint_share,
        "interaction" => interaction,
        "residual" => residual,
        "mean_baseline_gap" => base_sq > 0 ? sqrt(base_sq) : 0.0,
    )

    @printf("  Converged: %d/%d thetas\n", n_converged_theta, actual_n_thetas)
    @printf("  Baseline mean ||delta||²: %.6e  (n=%d)\n", base_sq, n_samples["baseline"])
    @printf("  Variant ablations:\n")
    for label in variant_labels
        label == "baseline" && continue
        variant_sq = n_samples[label] > 0 ? sq_norms[label] / n_samples[label] : NaN
        ratio = isfinite(variant_sq) && base_sq > 0 ? sqrt(variant_sq / base_sq) : NaN
        attempted = n_accepted_paths[label] + n_failed_paths[label]
        pass_rate = attempted > 0 ? n_accepted_paths[label] / attempted : NaN
        @printf("    %-12s RMSE ratio=%8.4f  MSE reduction=%8.1f%%  n=%d  paths=%d/%d (%.1f%%)\n",
                label, ratio, share_explained[label] * 100, n_samples[label],
                n_accepted_paths[label], attempted, pass_rate * 100)
    end
    if has_investment_joint
        @printf("  Investment interaction summary:\n")
        @printf("    S(x) only:          %6.1f%%\n", S_share * 100)
        @printf("    a(z) only:          %6.1f%%\n", a_share * 100)
        @printf("    S(x) + a(z) joint:  %6.1f%%\n", joint_share * 100)
        @printf("    Interaction:        %6.1f%%\n", interaction * 100)
        @printf("    Residual:           %6.1f%%\n", residual * 100)
    end
    flush(stdout)

    if checkpoint_every > 0 && (si % checkpoint_every == 0)
        checkpoint_path = joinpath(output_dir, "counterfactual_checkpoint.jls")
        Serialization.serialize(checkpoint_path, Dict(
            "all_results" => all_results,
            "completed_scales" => sort(collect(keys(all_results))),
            "shock_scales" => shock_scales,
            "variant_labels" => variant_labels,
            "timestamp" => now(),
            "incomplete" => si < length(shock_scales),
        ))
        println("  Checkpoint saved: $checkpoint_path")
        flush(stdout)
    end
end

# ============================================================================
# Save Results
# ============================================================================

function finite_values(xs)
    vals = Float64[]
    for x in xs
        isfinite(x) && push!(vals, Float64(x))
    end
    return vals
end

function finite_quantile(xs, q::Float64)
    vals = finite_values(xs)
    isempty(vals) && return NaN
    return quantile(vals, q)
end

function compact_attempt_summary(attempts::Vector{String})
    isempty(attempts) && return "none"
    counts = Dict{String, Int}()
    for a in attempts
        counts[a] = get(counts, a, 0) + 1
    end
    ordered = sort(collect(counts); by = p -> (-p[2], p[1]))
    return join(["$(p[1]):$(p[2])" for p in ordered], "; ")
end

function local_sensitivity_stats(r::Dict{String, Any}, pair, variant_by_label)
    base_mse = r["variant_mse"]["baseline"]
    mse_minus = r["variant_mse"][pair.minus]
    mse_plus = r["variant_mse"][pair.plus]
    rmse_minus = r["variant_rmse"][pair.minus]
    rmse_plus = r["variant_rmse"][pair.plus]
    base_rmse = r["variant_rmse"]["baseline"]
    vminus = variant_by_label[pair.minus]
    vplus = variant_by_label[pair.plus]
    value_minus = vminus.overrides[pair.parameter]
    value_plus = vplus.overrides[pair.parameter]
    elasticity = (isfinite(mse_minus) && isfinite(mse_plus) &&
                  mse_minus > 0 && mse_plus > 0 &&
                  value_minus > 0 && value_plus > 0) ?
        (log(mse_plus) - log(mse_minus)) / (log(value_plus) - log(value_minus)) : NaN
    reduction_minus = (isfinite(base_mse) && base_mse > 0 && isfinite(mse_minus)) ?
        100 * (1 - mse_minus / base_mse) : NaN
    reduction_plus = (isfinite(base_mse) && base_mse > 0 && isfinite(mse_plus)) ?
        100 * (1 - mse_plus / base_mse) : NaN
    return (
        value_minus = value_minus,
        value_plus = value_plus,
        mse_minus = mse_minus,
        mse_plus = mse_plus,
        rmse_ratio_minus = (isfinite(rmse_minus) && isfinite(base_rmse) && base_rmse > 0) ? rmse_minus / base_rmse : NaN,
        rmse_ratio_plus = (isfinite(rmse_plus) && isfinite(base_rmse) && base_rmse > 0) ? rmse_plus / base_rmse : NaN,
        reduction_minus = reduction_minus,
        reduction_plus = reduction_plus,
        elasticity = elasticity,
    )
end

results_path = joinpath(output_dir, "counterfactual_results.jls")
Serialization.serialize(results_path, Dict(
    "all_results" => all_results,
    "shock_scales" => shock_scales,
    "variant_labels" => variant_labels,
    "variant_specs" => [Dict(
        "label" => v.label,
        "model_label" => v.model_label,
        "overrides" => Dict(String(k) => val for (k, val) in v.overrides),
        "description" => v.description,
    ) for v in variant_specs],
    "n_thetas" => actual_n_thetas,
    "theta_source_indices" => collect(theta_source_indices),
    "sim_periods" => sim_periods,
    "burn_in" => burn_in,
    "sep_horizon" => sep_horizon,
    "sep_maxit" => sep_maxit,
    "sep_tol" => sep_tol,
    "sep_accept_tol" => sep_accept_tol,
    "sep_retry" => sep_retry,
    "retry_horizons" => retry_horizons,
    "retry_maxit_multipliers" => retry_maxit_multipliers,
    "sep_recovery" => sep_recovery,
    "sep_recovery_scales" => sep_recovery_scales,
    "sep_fallback_solver" => isnothing(sep_fallback_solver) ? "none" : String(sep_fallback_solver),
    "chain_path" => chain_path,
    "use_chain_theta" => use_chain_theta,
    "param_overrides" => Dict(String(k) => v for (k, v) in param_overrides),
    "timestamp" => now(),
))
println("\nResults saved: $results_path")

csv_path = joinpath(output_dir, "counterfactual_ablation_summary.csv")
open(csv_path, "w") do io
    println(io, "shock_scale,variant,model,description,n_samples,n_converged_theta,path_accepts,path_attempts,path_pass_rate,max_sep_error_p50,max_sep_error_p90,max_sep_error_max,recovery_uses,attempt_summary,rmse,mse,rmse_ratio_to_baseline,mse_reduction_pct,overrides")
    for scale in sort(shock_scales)
        r = all_results[scale]
        base_rmse = r["variant_rmse"]["baseline"]
        for vspec in variant_specs
            label = vspec.label
            rmse = r["variant_rmse"][label]
            mse = r["variant_mse"][label]
            ratio = isfinite(rmse) && isfinite(base_rmse) && base_rmse > 0 ? rmse / base_rmse : NaN
            reduction = r["share_explained"][label] * 100
            overrides_txt = format_param_overrides(vspec.overrides)
            accepts = r["n_accepted_paths"][label]
            attempts = r["n_attempted_paths"][label]
            pass_rate = attempts > 0 ? accepts / attempts : NaN
            err_p50 = finite_quantile(r["max_errors_by_variant"][label], 0.5)
            err_p90 = finite_quantile(r["max_errors_by_variant"][label], 0.9)
            err_max = finite_quantile(r["max_errors_by_variant"][label], 1.0)
            recovery_uses = sum(>(0), r["recovery_counts_by_variant"][label])
            attempt_summary = compact_attempt_summary(r["attempts_by_variant"][label])
            @printf(io, "%.8f,%s,%s,\"%s\",%d,%d,%d,%d,%.8f,%.12g,%.12g,%.12g,%d,\"%s\",%.12g,%.12g,%.12g,%.8f,\"%s\"\n",
                    scale, label, vspec.model_label, vspec.description,
                    r["n_samples"][label], r["n_converged_theta"],
                    accepts, attempts, pass_rate,
                    err_p50, err_p90, err_max, recovery_uses, attempt_summary,
                    rmse, mse, ratio, reduction, overrides_txt)
        end
    end
end
println("CSV saved: $csv_path")

local_pairs = active_local_sensitivity_pairs(variant_labels)
if !isempty(local_pairs)
    local_csv_path = joinpath(output_dir, "counterfactual_local_sensitivity.csv")
    open(local_csv_path, "w") do io
        println(io, "shock_scale,parameter,description,value_minus,value_plus,mse_minus,mse_plus,rmse_ratio_minus,rmse_ratio_plus,mse_reduction_minus_pct,mse_reduction_plus_pct,log_mse_elasticity")
        for scale in sort(shock_scales)
            r = all_results[scale]
            for pair in local_pairs
                stats = local_sensitivity_stats(r, pair, variant_by_label)
                @printf(io, "%.8f,%s,\"%s\",%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.8f,%.8f,%.12g\n",
                        scale, String(pair.parameter), pair.description,
                        stats.value_minus, stats.value_plus,
                        stats.mse_minus, stats.mse_plus,
                        stats.rmse_ratio_minus, stats.rmse_ratio_plus,
                        stats.reduction_minus, stats.reduction_plus,
                        stats.elasticity)
            end
        end
    end
    println("Local sensitivity CSV saved: $local_csv_path")

    bytheta_csv_path = joinpath(output_dir, "counterfactual_local_sensitivity_by_theta.csv")
    open(bytheta_csv_path, "w") do io
        println(io, "shock_scale,theta_index,chain_draw_index,parameter,description,value_minus,value_plus,baseline_mse,mse_minus,mse_plus,rmse_ratio_minus,rmse_ratio_plus,log_mse_elasticity,abs_log_mse_elasticity,is_largest_abs")
        for scale in sort(shock_scales)
            r = all_results[scale]
            for rec in r["theta_records"]
                largest = get(rec, "largest_abs_elasticity_parameter", "")
                base_mse = rec["baseline_mse"]
                for pair in local_pairs
                    pname = String(pair.parameter)
                    mse_minus = rec["variant_mse"][pair.minus]
                    mse_plus = rec["variant_mse"][pair.plus]
                    rmse_minus = rec["variant_rmse"][pair.minus]
                    rmse_plus = rec["variant_rmse"][pair.plus]
                    base_rmse = rec["baseline_rmse"]
                    vminus = variant_by_label[pair.minus]
                    vplus = variant_by_label[pair.plus]
                    value_minus = vminus.overrides[pair.parameter]
                    value_plus = vplus.overrides[pair.parameter]
                    elasticity = rec["local_elasticities"][pname]
                    abs_elasticity = rec["local_abs_elasticities"][pname]
                    ratio_minus = (isfinite(rmse_minus) && isfinite(base_rmse) && base_rmse > 0) ? rmse_minus / base_rmse : NaN
                    ratio_plus = (isfinite(rmse_plus) && isfinite(base_rmse) && base_rmse > 0) ? rmse_plus / base_rmse : NaN
                    @printf(io, "%.8f,%d,%d,%s,\"%s\",%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%s\n",
                            scale, rec["theta_index"], rec["chain_draw_index"], pname,
                            pair.description, value_minus, value_plus, base_mse,
                            mse_minus, mse_plus, ratio_minus, ratio_plus,
                            elasticity, abs_elasticity, largest == pname)
                end
            end
        end
    end
    println("Theta-level local sensitivity CSV saved: $bytheta_csv_path")

    dominance_csv_path = joinpath(output_dir, "counterfactual_local_dominance.csv")
    parameter_names = [String(p.parameter) for p in local_pairs]
    open(dominance_csv_path, "w") do io
        header = "shock_scale,n_theta_records,n_finite_records," *
                 join(["largest_$(p)_count,largest_$(p)_share" for p in parameter_names], ",") *
                 ",kimball_largest_count,kimball_largest_share,real_side_largest_count,real_side_largest_share," *
                 join(["median_abs_elasticity_$(p)" for p in parameter_names], ",")
        println(io, header)
        for scale in sort(shock_scales)
            r = all_results[scale]
            records = r["theta_records"]
            counts = Dict(p => 0 for p in parameter_names)
            finite_records = 0
            abs_by_param = Dict(p => Float64[] for p in parameter_names)
            for rec in records
                largest = get(rec, "largest_abs_elasticity_parameter", "")
                if largest != ""
                    finite_records += 1
                    haskey(counts, largest) && (counts[largest] += 1)
                end
                for p in parameter_names
                    v = rec["local_abs_elasticities"][p]
                    isfinite(v) && push!(abs_by_param[p], v)
                end
            end
            kimball_count = get(counts, "curvp", 0) + get(counts, "curvw", 0)
            real_side_count = get(counts, "csadjcost", 0) + get(counts, "czcap", 0)
            fields = Any[scale, length(records), finite_records]
            for p in parameter_names
                push!(fields, counts[p])
                push!(fields, finite_records > 0 ? counts[p] / finite_records : NaN)
            end
            push!(fields, kimball_count)
            push!(fields, finite_records > 0 ? kimball_count / finite_records : NaN)
            push!(fields, real_side_count)
            push!(fields, finite_records > 0 ? real_side_count / finite_records : NaN)
            for p in parameter_names
                vals = abs_by_param[p]
                push!(fields, isempty(vals) ? NaN : median(vals))
            end
            println(io, join(fields, ","))
        end
    end
    println("Local dominance CSV saved: $dominance_csv_path")
end

md_path = joinpath(output_dir, "COUNTERFACTUAL_ABLATION_SUMMARY.md")
open(md_path, "w") do io
    println(io, "# HLT Mechanism Ablation: SEP--ROM1 Gap")
    println(io)
    println(io, "**Generated:** $(now())")
    println(io)
    println(io, "This exercise compares the ROM1 error against direct SEP across structural variants. For each variant, the script rebuilds ROM1 and reruns SEP under the same shock seeds. The reported reduction is")
    println(io)
    println(io, "\\[1 - \\mathrm{MSE}_{variant}/\\mathrm{MSE}_{baseline}.\\]")
    println(io)
    println(io, "Positive values mean the ablated mechanism reduces the nonlinear SEP--ROM1 gap. Negative values mean the variant makes the gap larger. Shares are not additive because mechanisms interact and each variant has its own equilibrium path.")
    println(io)
    println(io, "- Variant suite: `$variant_suite`")
    println(io, "- Parameter source: `$(use_chain_theta ? "posterior chain" : "fixed model calibration")`")
    println(io, "- Chain: `$chain_path`")
    println(io, "- Maintained parameter overrides: `$(format_param_overrides(param_overrides))`")
    println(io, "- Theta/shock replications: `$actual_n_thetas`")
    println(io, "- Periods per theta: `$sim_periods` after `$burn_in` burn-in")
    println(io, "- SEP: horizon `$sep_horizon`, maxit `$sep_maxit`, tol `$sep_tol`, accept_tol `$sep_accept_tol`")
    println(io, "- SEP retry/recovery: retry `$sep_retry`, horizons `$(join(retry_horizons, ", "))`, maxit multipliers `$(join(retry_maxit_multipliers, ", "))`, recovery `$sep_recovery`, fallback `$(isnothing(sep_fallback_solver) ? "none" : sep_fallback_solver)`")
    println(io)
    println(io, "## Variant Definitions")
    println(io)
    println(io, "| Variant | Model | Switch |")
    println(io, "|---|---|---|")
    for v in variant_specs
        println(io, "| `$(v.label)` | `$(v.model_label)` | $(v.description) |")
    end
    println(io)
    for scale in sort(shock_scales)
        r = all_results[scale]
        base_rmse = r["variant_rmse"]["baseline"]
        println(io, "## Shock Scale $(scale)")
        println(io)
        @printf(io, "- Baseline RMSE: `%.6g` with `%d` samples and `%d/%d` converged theta paths.\n",
                base_rmse, r["n_samples"]["baseline"], r["n_converged_theta"], actual_n_thetas)
        println(io)
        println(io, "| Variant | N | RMSE | RMSE ratio | MSE reduction |")
        println(io, "|---|---:|---:|---:|---:|")
        for v in variant_specs
            label = v.label
            rmse = r["variant_rmse"][label]
            ratio = isfinite(rmse) && isfinite(base_rmse) && base_rmse > 0 ? rmse / base_rmse : NaN
            reduction = r["share_explained"][label] * 100
            @printf(io, "| `%s` | %d | %.6g | %.4f | %.1f%% |\n",
                    label, r["n_samples"][label], rmse, ratio, reduction)
        end
        println(io)
        println(io, "### Solver Coverage")
        println(io)
        println(io, "| Variant | Accepted paths | Attempted paths | Pass rate | Max SEP error p50 | Max SEP error p90 | Max SEP error max | Recovery uses | Attempt summary |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for v in variant_specs
            label = v.label
            accepts = r["n_accepted_paths"][label]
            attempts = r["n_attempted_paths"][label]
            pass_rate = attempts > 0 ? accepts / attempts : NaN
            err_p50 = finite_quantile(r["max_errors_by_variant"][label], 0.5)
            err_p90 = finite_quantile(r["max_errors_by_variant"][label], 0.9)
            err_max = finite_quantile(r["max_errors_by_variant"][label], 1.0)
            recovery_uses = sum(>(0), r["recovery_counts_by_variant"][label])
            attempt_summary = compact_attempt_summary(r["attempts_by_variant"][label])
            @printf(io, "| `%s` | %d | %d | %.1f%% | %.3g | %.3g | %.3g | %d | `%s` |\n",
                    label, accepts, attempts, pass_rate * 100, err_p50, err_p90,
                    err_max, recovery_uses, attempt_summary)
        end
        if !isempty(local_pairs)
            println(io)
            println(io, "### Local +/-10% Sensitivity")
            println(io)
            println(io, "The elasticity is the centered log change in the SEP--ROM1 MSE divided by the centered log change in the parameter. Positive values mean that increasing the parameter raises the local nonlinear gap.")
            println(io)
            println(io, "| Parameter | Description | RMSE ratio (-10%) | RMSE ratio (+10%) | MSE reduction (-10%) | MSE reduction (+10%) | Log-MSE elasticity |")
            println(io, "|---|---|---:|---:|---:|---:|---:|")
            for pair in local_pairs
                stats = local_sensitivity_stats(r, pair, variant_by_label)
                @printf(io, "| `%s` | %s | %.4f | %.4f | %.1f%% | %.1f%% | %.3f |\n",
                        String(pair.parameter), pair.description,
                        stats.rmse_ratio_minus, stats.rmse_ratio_plus,
                        stats.reduction_minus, stats.reduction_plus,
                        stats.elasticity)
            end
            records = r["theta_records"]
            parameter_names = [String(p.parameter) for p in local_pairs]
            counts = Dict(p => 0 for p in parameter_names)
            finite_records = 0
            abs_by_param = Dict(p => Float64[] for p in parameter_names)
            for rec in records
                largest = get(rec, "largest_abs_elasticity_parameter", "")
                if largest != ""
                    finite_records += 1
                    haskey(counts, largest) && (counts[largest] += 1)
                end
                for p in parameter_names
                    v = rec["local_abs_elasticities"][p]
                    isfinite(v) && push!(abs_by_param[p], v)
                end
            end
            kimball_count = get(counts, "curvp", 0) + get(counts, "curvw", 0)
            real_side_count = get(counts, "csadjcost", 0) + get(counts, "czcap", 0)
            println(io)
            println(io, "### Theta-Level Dominance")
            println(io)
            println(io, "For each theta path, dominance is the parameter with the largest absolute local log-MSE elasticity. This is the statistic used to ask whether Kimball curvature is usually the largest local source of the SEP--ROM1 gap in the sampled parameter region.")
            println(io)
            @printf(io, "- Finite theta-level dominance records: `%d/%d`.\n", finite_records, length(records))
            if finite_records > 0
                @printf(io, "- Kimball curvature largest: `%d/%d` (`%.1f%%`).\n",
                        kimball_count, finite_records, 100 * kimball_count / finite_records)
                @printf(io, "- Real-side curvature largest: `%d/%d` (`%.1f%%`).\n",
                        real_side_count, finite_records, 100 * real_side_count / finite_records)
            end
            println(io)
            println(io, "| Parameter | Largest-count share | Median abs. elasticity |")
            println(io, "|---|---:|---:|")
            for p in parameter_names
                vals = abs_by_param[p]
                med_abs = isempty(vals) ? NaN : median(vals)
                share = finite_records > 0 ? counts[p] / finite_records : NaN
                @printf(io, "| `%s` | %.1f%% | %.3f |\n", p, 100 * share, med_abs)
            end
        end
        if all(haskey(r["share_explained"], k) for k in ("noS", "lina", "noS_lina"))
            println(io)
            @printf(io, "Investment joint summary: `noS=%.1f%%`, `lina=%.1f%%`, `noS_lina=%.1f%%`, interaction `%.1f%%`.\n",
                    r["S_share"]*100, r["a_share"]*100, r["joint_share"]*100, r["interaction"]*100)
        end
        println(io)
    end
end
println("Summary saved: $md_path")

# ============================================================================
# Figures
# ============================================================================

const FIG_DIR = joinpath(output_dir, "figures")
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
    label="Residual (Tobin q, Kimball)", color=:gray, marker=:utriangle, markersize=5,
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
