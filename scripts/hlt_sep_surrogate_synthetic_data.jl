#!/usr/bin/env julia
using MacroModelling
using Random
using Serialization
using Dates
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function parse_shock_list(arg::AbstractString, model, structural_idx::Vector{Int})
    if arg == "all"
        return structural_idx
    end
    names = split(arg, ",")
    idx = Int[]
    for name in names
        sym = Meta.parse(strip(name))
        pos = findfirst(==(sym), model.exo)
        if pos === nothing
            error("Shock $sym not found in model.")
        end
        if !(pos in structural_idx)
            error("Shock $sym is not structural (OBC shocks not supported here).")
        end
        push!(idx, pos)
    end
    return idx
end

function shock_sigmas_for(model, shock_scaling::Symbol, shock_scale::Float64)
    shock_names = model.exo
    sigmas = zeros(length(shock_names))
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    for (i, name) in enumerate(shock_names)
        if !obc_mask[i]
            sigmas[i] = shock_scaling == :parameter ?
                MacroModelling.sep_irf_shock_std(model, name) : 1.0
        end
    end
    sigmas .*= shock_scale
    return sigmas
end

@inline function attempt_seed(seed0::Int, attempt::Int, stride::Int)
    return seed0 + (attempt - 1) * stride
end

@inline function attempt_shock_scale(base_scale::Float64, attempt::Int, backoff::Float64)
    return base_scale * (backoff ^ (attempt - 1))
end

@inline function available_periods(sim, shocks)
    sim_periods = max(size(sim, 2) - 1, 0)
    shock_periods = ndims(shocks) == 2 ? size(shocks, 2) : 0
    return min(sim_periods, shock_periods)
end

@inline function retry_threshold_period(burn_in::Int, stable_min_periods::Int)
    return max(1, burn_in + max(stable_min_periods, 1))
end

function should_retry(res_errorflag::Bool, failure_period::Int, T_available::Int;
                      stable_prefix::Bool, stable_min_periods::Int, burn_in::Int,
                      retry_on_early_failure::Bool, attempt::Int, max_attempts::Int)
    attempt < max_attempts || return false
    retry_on_early_failure || return true
    if stable_prefix && stable_min_periods > 0 && T_available > 0 && T_available < stable_min_periods
        return true
    end
    res_errorflag || return false
    return failure_period <= retry_threshold_period(burn_in, stable_min_periods)
end

burn_in = parse_arg_int(ARGS, "--burn-in", 100)
sample_start = parse_arg_int(ARGS, "--sample-start", 47)
sample_length = parse_arg_int(ARGS, "--sample-length", 0)
sep_horizon = parse_arg_int(ARGS, "--sep-horizon", 20)
sep_order = parse_arg_int(ARGS, "--sep-order", 1)
sep_nnodes = parse_arg_int(ARGS, "--sep-nnodes", 3)
sep_maxit = parse_arg_int(ARGS, "--sep-maxit", 80)
sep_tol = parse_arg_float(ARGS, "--sep-tol", 1e-5)
sep_linear_solver = parse_arg_symbol(ARGS, "--sep-linear-solver", :normal_equations)
sep_fallback_solver_arg = parse_arg_symbol(ARGS, "--sep-fallback-solver", :none)
sep_fallback_solver = sep_fallback_solver_arg == :none ? nothing : sep_fallback_solver_arg
sep_stall_iters = parse_arg_int(ARGS, "--sep-stall-iters", 25)
sep_stall_rel_tol = parse_arg_float(ARGS, "--sep-stall-rel-tol", 1e-4)
sep_stall_abs_tol = parse_arg_float(ARGS, "--sep-stall-abs-tol", 1e-10)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", NaN)
sep_accept_tol = isnan(sep_accept_tol) ? nothing : sep_accept_tol
sep_shock_scale = parse_arg_float(ARGS, "--sep-shock-scale", 1.0)
stable_prefix = "--stable-prefix" in ARGS
stable_min_periods = parse_arg_int(ARGS, "--stable-min-periods", 0)
attempts = parse_arg_int(ARGS, "--attempts", stable_prefix ? 3 : 1)
seed_attempt_stride = parse_arg_int(ARGS, "--seed-attempt-stride", 10_000)
retry_on_early_failure = parse_arg_bool(ARGS, "--retry-on-early-failure", stable_prefix)
retry_shock_scale_backoff = parse_arg_float(ARGS, "--retry-shock-scale-backoff", 1.0)
allow_empty = "--allow-empty" in ARGS
min_generated_periods = parse_arg_int(ARGS, "--min-generated-periods", stable_min_periods > 0 ? stable_min_periods : 1)
obs_sigma_val = parse_arg_float(ARGS, "--obs-sigma", NaN)
seed = parse_arg_int(ARGS, "--seed", 123)
shock_scaling = parse_arg_symbol(ARGS, "--shock-scaling", :parameter)
shock_scale = parse_arg_float(ARGS, "--shock-scale", 0.25)
sep_sparse_tree = !("--no-sparse-tree" in ARGS)
force_obc = "--use-obc" in ARGS
force_no_obc = "--no-obc" in ARGS
if force_obc && force_no_obc
    error("Specify only one of --use-obc or --no-obc.")
end
use_obc = force_obc
if force_no_obc
    use_obc = false
end
attempts >= 1 || error("--attempts must be >= 1.")
seed_attempt_stride >= 1 || error("--seed-attempt-stride must be >= 1.")
(0.0 < retry_shock_scale_backoff <= 1.0) || error("--retry-shock-scale-backoff must be in (0, 1].")
min_generated_periods >= 0 || error("--min-generated-periods must be >= 0.")
vol_start = parse_arg_int(ARGS, "--vol-start", 0)
vol_end = parse_arg_int(ARGS, "--vol-end", 0)
vol_mult = parse_arg_float(ARGS, "--vol-mult", 1.0)
vol_shocks = parse_arg_string(ARGS, "--vol-shocks", "all")
vol_min_ratio = parse_arg_float(ARGS, "--vol-min-ratio", NaN)
output_root_arg = parse_arg_string(ARGS, "--output-root", "")
output_dir_arg = parse_arg_string(ARGS, "--output-dir", "")
output_root = output_root_arg == "" ? joinpath(@__DIR__, "..", "data") : output_root_arg
mkpath(output_root)
output_dir = output_dir_arg == "" ?
    joinpath(output_root, "hlt_sep_surrogate_synth_$(Dates.format(now(), "yyyymmdd_HHMMSS"))") :
    output_dir_arg
mkpath(output_dir)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
sample_idx = sample_length > 0 ? (sample_start:(sample_start + sample_length - 1)) : (47:230)
T_obs = length(sample_idx)

model_name = use_obc ? "Smets_Wouters_2007_HLT_obc" : "Smets_Wouters_2007_HLT"
hlt_model_file_and_symbol(model_name)  # fail fast on unsupported names
model = load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)
hlt_param_names = model.parameters
base_values = copy(model.parameter_values)

param_set_name = parse_arg_symbol(ARGS, "--param-set", :legacy_3params)
if param_set_name == :legacy_3params
    theta_names = [:cprobp, :cindp, :curvp]
else
    theta_names = get_parameter_names(param_set_name)
end
theta_idx = indexin(theta_names, hlt_param_names)
@assert all(!isnothing, theta_idx) "Missing parameters in $(model.model_name): $(theta_names[isnothing.(theta_idx)])"
theta_idx = Int.(theta_idx)
theta_true = base_values[theta_idx]

MacroModelling.write_parameters_input!(model, base_values, verbose = false)

println("Generating synthetic HLT data")
println("Model: $(model.model_name) (use_obc=$use_obc)")
println("Sample length: $T_obs, burn-in: $burn_in, seed: $seed")
if attempts > 1 || stable_prefix
    println("Retry policy: attempts=$attempts seed_stride=$seed_attempt_stride retry_on_early_failure=$retry_on_early_failure shock_backoff=$retry_shock_scale_backoff")
end
if stable_prefix
    println("Stable-prefix mode: min_periods=$stable_min_periods")
end

total_periods = T_obs + burn_in
shock_sigmas_base = shock_sigmas_for(model, shock_scaling, shock_scale)
structural_idx = findall(shock_sigmas_base .> 0)
has_vol_episode = vol_start > 0 || vol_end > 0 || vol_mult != 1.0
vol_start_total = vol_start
vol_end_total = vol_end
if has_vol_episode
    println("High-volatility episode (observed periods): $vol_start:$vol_end mult=$vol_mult shocks=$vol_shocks")
    if vol_start <= 0 || vol_end <= 0
        error("Specify --vol-start and --vol-end (>=1) for a volatility episode.")
    end
    if vol_end < vol_start
        error("vol-end must be >= vol-start.")
    end
    if vol_mult <= 0
        error("vol-mult must be positive.")
    end
    if vol_end > T_obs
        error("vol-end ($vol_end) exceeds observed sample length ($T_obs).")
    end
    vol_start_total = vol_start + burn_in
    vol_end_total = vol_end + burn_in
    if vol_end_total > total_periods
        error("vol-end ($vol_end) with burn-in ($burn_in) exceeds total periods ($total_periods).")
    end
    println("High-volatility episode (total periods): $vol_start_total:$vol_end_total")
    if isnan(vol_min_ratio)
        vol_min_ratio = vol_mult > 1.0 ? vol_mult : 0.0
    end
end
vol_idx = has_vol_episode ? parse_shock_list(vol_shocks, model, structural_idx) : Int[]
use_override = shock_scale != 1.0 || has_vol_episode || retry_shock_scale_backoff != 1.0

function build_shocks_override(model;
                               seed::Int,
                               total_periods::Int,
                               structural_idx::Vector{Int},
                               shock_sigmas::AbstractVector,
                               has_vol_episode::Bool,
                               vol_idx::Vector{Int},
                               vol_start_total::Int,
                               vol_end_total::Int,
                               vol_mult::Float64,
                               vol_min_ratio::Float64,
                               burn_in::Int)
    rng = MersenneTwister(seed)
    shock_names = model.exo
    nshocks = length(shock_names)
    shocks = zeros(nshocks, total_periods)
    if length(structural_idx) == 1
        shocks[structural_idx[1], :] .= randn(rng, total_periods) .* shock_sigmas[structural_idx[1]]
    elseif !isempty(structural_idx)
        shocks[structural_idx, :] .= Diagonal(shock_sigmas[structural_idx]) *
            randn(rng, length(structural_idx), total_periods)
    end
    if has_vol_episode && !isempty(vol_idx)
        shocks[vol_idx, vol_start_total:vol_end_total] .*= vol_mult
    end
    if has_vol_episode && !isempty(vol_idx) && vol_min_ratio > 0
        sample_periods = (burn_in + 1):total_periods
        outside_periods = setdiff(sample_periods, vol_start_total:vol_end_total)
        if isempty(outside_periods)
            println("Warning: volatility window covers the full sample; skipping realized ratio enforcement.")
        else
            inside_norm = mean(vec(sqrt.(sum(shocks[vol_idx, vol_start_total:vol_end_total] .^ 2, dims = 1))))
            outside_norm = mean(vec(sqrt.(sum(shocks[vol_idx, outside_periods] .^ 2, dims = 1))))
            if outside_norm > 0
                ratio = inside_norm / outside_norm
                if ratio < vol_min_ratio
                    scale = (vol_min_ratio * outside_norm) / max(inside_norm, eps())
                    shocks[vol_idx, vol_start_total:vol_end_total] .*= scale
                    new_norm = mean(vec(sqrt.(sum(shocks[vol_idx, vol_start_total:vol_end_total] .^ 2, dims = 1))))
                    new_ratio = new_norm / outside_norm
                    println("Enforced high-vol shocks: ratio $(round(ratio, digits = 3)) -> $(round(new_ratio, digits = 3)) (scale=$(round(scale, digits = 3))).")
                else
                    println("High-vol shocks already satisfy ratio $(round(ratio, digits = 3)) >= $(round(vol_min_ratio, digits = 3)).")
                end
            else
                println("Warning: outside shock norm is zero; skipping realized ratio enforcement.")
            end
        end
    end
    return shocks
end

res = nothing
sim = zeros(0, 0)
shocks = zeros(0, 0)
shock_sigmas = copy(shock_sigmas_base)
seed_used = seed
attempt_used = 0
shock_scale_used = shock_scale
sep_failure_period = 0
sep_partial_prefix_accepted = false
generated_periods = 0

for attempt in 1:attempts
    global attempt_used = attempt
    seed_try = attempt_seed(seed, attempt, seed_attempt_stride)
    shock_scale_eff = attempt_shock_scale(shock_scale, attempt, retry_shock_scale_backoff)
    shock_sigmas_try = shock_sigmas_for(model, shock_scaling, shock_scale_eff)
    structural_idx_try = findall(shock_sigmas_try .> 0)
    vol_idx_try = has_vol_episode ? parse_shock_list(vol_shocks, model, structural_idx_try) : Int[]
    shocks_override_try = use_override ? build_shocks_override(model;
                                                               seed = seed_try,
                                                               total_periods = total_periods,
                                                               structural_idx = structural_idx_try,
                                                               shock_sigmas = shock_sigmas_try,
                                                               has_vol_episode = has_vol_episode,
                                                               vol_idx = vol_idx_try,
                                                               vol_start_total = vol_start_total,
                                                               vol_end_total = vol_end_total,
                                                               vol_mult = vol_mult,
                                                               vol_min_ratio = vol_min_ratio,
                                                               burn_in = burn_in) : nothing

    res_try = nothing
    try
        res_try = MacroModelling.simulate_sep_extended_path(
            model;
            periods = T_obs,
            burn_in = burn_in,
            sep_horizon = sep_horizon,
            sep_order = sep_order,
            sep_nnodes = sep_nnodes,
            sep_maxit = sep_maxit,
            sep_tol = sep_tol,
            sep_sparse_tree = sep_sparse_tree,
            sep_linear_solver = sep_linear_solver,
            sep_fallback_solver = sep_fallback_solver,
            sep_stall_iters = sep_stall_iters,
            sep_stall_rel_tol = sep_stall_rel_tol,
            sep_stall_abs_tol = sep_stall_abs_tol,
            sep_accept_tol = sep_accept_tol,
            sep_shock_scale = sep_shock_scale,
            shock_scaling = shock_scaling,
            shocks = shocks_override_try,
            random_seed = seed_try,
            silent = true,
        )
    catch e
        if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
            if attempt < attempts
                println("Synthetic SEP singular on attempt $attempt/$attempts; retrying (seed=$seed_try, shock_scale=$(round(shock_scale_eff, digits=4))).")
                continue
            end
            rethrow()
        else
            rethrow()
        end
    end

    sim_try = Array(res_try.simulation)
    shocks_try = res_try.shocks
    T_available = available_periods(sim_try, shocks_try)
    failure_period = res_try.failure_period === nothing ? 0 : Int(res_try.failure_period)

    if res_try.errorflag
        @warn "SEP failed during synthetic data generation" attempt failure_period seed_try shock_scale_eff
    end

    accept_full = !res_try.errorflag && T_available >= T_obs
    accept_partial = stable_prefix && T_available > 0 && (stable_min_periods <= 0 || T_available >= stable_min_periods)

    if accept_full || accept_partial || (allow_empty && T_available == 0)
        global res = res_try
        global sim = sim_try
        global shocks = shocks_try
        global shock_sigmas = shock_sigmas_try
        global seed_used = seed_try
        global shock_scale_used = shock_scale_eff
        global sep_failure_period = failure_period
        global generated_periods = T_available
        global sep_partial_prefix_accepted = res_try.errorflag && accept_partial
        if sep_partial_prefix_accepted
            println("Accepting partial synthetic SEP prefix: generated_periods=$generated_periods failure_period=$failure_period (attempt $attempt/$attempts).")
        elseif attempt > 1
            println("Recovered synthetic SEP generation on attempt $attempt/$attempts (seed=$seed_try, shock_scale=$(round(shock_scale_eff, digits=4))).")
        end
        break
    end

    if should_retry(res_try.errorflag, failure_period, T_available;
                    stable_prefix = stable_prefix,
                    stable_min_periods = stable_min_periods,
                    burn_in = burn_in,
                    retry_on_early_failure = retry_on_early_failure,
                    attempt = attempt,
                    max_attempts = attempts)
        println("Retrying synthetic SEP (attempt $attempt/$attempts): failure_period=$failure_period, generated_periods=$T_available.")
        continue
    end

    global res = res_try
    global sim = sim_try
    global shocks = shocks_try
    global shock_sigmas = shock_sigmas_try
    global seed_used = seed_try
    global shock_scale_used = shock_scale_eff
    global sep_failure_period = failure_period
    global generated_periods = T_available
    break
end

if res === nothing
    error("Synthetic SEP generation failed before producing a result object.")
end

if generated_periods < min_generated_periods && !allow_empty
    error("Synthetic SEP generated $generated_periods periods (< min-generated-periods=$min_generated_periods). " *
          "Consider increasing --attempts, enabling --stable-prefix, lowering --shock-scale, or raising --sep-accept-tol.")
end

if res.errorflag && !(stable_prefix && generated_periods > 0 && (stable_min_periods <= 0 || generated_periods >= stable_min_periods))
    error("SEP failed during synthetic data generation (failure_period=$(res.failure_period), generated_periods=$generated_periods).")
end

state_idx = sort(unique(vcat(model.timings.past_not_future_and_mixed_idx,
                             model.timings.future_not_past_and_mixed_idx)))
state_names = model.var[state_idx]
obs_idx = indexin(observables, model.var)
@assert all(!isnothing, obs_idx) "Observable indices not found in $(model.model_name)."
obs_idx = Int.(obs_idx)

s0 = sim[state_idx, 1]
requested_periods = T_obs
generated_periods = min(generated_periods, max(size(sim, 2) - 1, 0))
generated_periods = min(generated_periods, length(sample_idx))

if size(sim, 2) < generated_periods + 1
    error("Synthetic simulation length mismatch: generated_periods=$generated_periods but simulation has $(size(sim, 2)) columns.")
end

sample_idx_requested = collect(sample_idx)
sample_idx_used = generated_periods > 0 ? sample_idx_requested[1:generated_periods] : Int[]
obs_data = generated_periods > 0 ? sim[obs_idx, 2:(generated_periods + 1)] : zeros(length(obs_idx), 0)

shock_cols_used = min(size(shocks, 2), burn_in + generated_periods)
shocks = shock_cols_used > 0 ? shocks[:, 1:shock_cols_used] : zeros(size(shocks, 1), 0)

size(obs_data, 2) == length(sample_idx_used) || error("obs_data/sample_idx length mismatch: obs=$(size(obs_data, 2)) sample_idx=$(length(sample_idx_used)).")
all(isfinite, obs_data) || error("Synthetic observables contain non-finite values.")
if !isempty(shocks)
    all(isfinite, shocks) || error("Synthetic shocks contain non-finite values.")
end

if isnan(obs_sigma_val)
    obs_sigma = fill(sep_tol, size(obs_data, 1))
else
    obs_sigma = fill(obs_sigma_val, size(obs_data, 1))
end

payload = Dict(
    "model" => string(model.model_name),
    "observables" => observables,
    "sample_idx" => sample_idx_used,
    "sample_idx_requested" => sample_idx_requested,
    "state_names" => state_names,
    "state_idx" => state_idx,
    "state_definition" => "past_not_future_and_mixed + future_not_past_and_mixed",
    "obs_idx" => obs_idx,
    "theta_names" => theta_names,
    "theta_true" => theta_true,
    "s0" => s0,
    "shocks" => shocks,
    "shock_sigmas" => shock_sigmas,
    "obs_data" => obs_data,
    "obs_sigma" => obs_sigma,
    "requested_periods" => requested_periods,
    "generated_periods" => generated_periods,
    "burn_in" => burn_in,
    "sep_horizon" => sep_horizon,
    "sep_order" => sep_order,
    "sep_nnodes" => sep_nnodes,
    "sep_maxit" => sep_maxit,
    "sep_tol" => sep_tol,
    "sep_sparse_tree" => sep_sparse_tree,
    "sep_shock_scale" => sep_shock_scale,
    "shock_scaling" => shock_scaling,
    "shock_scale" => shock_scale,
    "shock_scale_used" => shock_scale_used,
    "seed" => seed,
    "seed_used" => seed_used,
    "attempts" => attempts,
    "attempt_used" => attempt_used,
    "seed_attempt_stride" => seed_attempt_stride,
    "retry_on_early_failure" => retry_on_early_failure,
    "retry_shock_scale_backoff" => retry_shock_scale_backoff,
    "stable_prefix" => stable_prefix,
    "stable_min_periods" => stable_min_periods,
    "min_generated_periods" => min_generated_periods,
    "allow_empty" => allow_empty,
    "sep_accept_tol" => sep_accept_tol,
    "sep_failure_period" => sep_failure_period,
    "sep_partial_prefix_accepted" => sep_partial_prefix_accepted,
    "vol_start" => vol_start,
    "vol_end" => vol_end,
    "vol_start_total" => vol_start_total,
    "vol_end_total" => vol_end_total,
    "vol_mult" => vol_mult,
    "vol_shocks" => vol_shocks,
    "vol_min_ratio" => vol_min_ratio,
)

out_path = joinpath(output_dir, "hlt_sep_synth_data.jls")
serialize(out_path, payload)

println("Saved synthetic data: $out_path")
println("Synthetic summary: requested_periods=$requested_periods generated_periods=$generated_periods attempt_used=$attempt_used seed_used=$seed_used partial_prefix=$sep_partial_prefix_accepted")
