#!/usr/bin/env julia
using MacroModelling
using Serialization
using Statistics
using LinearAlgebra
using Dates
using AxisKeys

ENV["GKSwstype"] = "100"
using StatsPlots

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

script_repo_root() = normpath(joinpath(@__DIR__, ".."))

function parse_symbol_list(arg::String)
    if isempty(arg)
        return Symbol[]
    end
    return [Symbol(strip(name)) for name in split(arg, ",")]
end

function apply_paper_calibration!(model)
    base_values = copy(model.parameter_values)
    MacroModelling.write_parameters_input!(model, base_values, verbose = false)
    return base_values
end

function shock_value_for(model, shock::Symbol, shock_size::Float64; shock_scaling::Symbol)
    if shock_scaling == :none
        return shock_size
    elseif shock_scaling == :parameter
        return shock_size * MacroModelling.sep_irf_shock_std(model, shock; warn_missing = false)
    end
    error("Unknown shock_scaling=$shock_scaling. Use :none or :parameter.")
end

function surrogate_rollout(frozen::FrozenMLP,
                           s0::Vector{Float64},
                           shocks::AbstractMatrix,
                           θ::Vector{Float64},
                           d_obs::Int;
                           rom_predictor::Union{Nothing,RomPredictor} = nothing,
                           check_finite::Bool = false,
                           rom_fallback_order::Union{Nothing,Int} = nothing)
    state_full = nothing
    rom_fallback_cache = nothing
    predict_surrogate = function (state, shock_t, θ_local)
        x = vcat(state, shock_t, θ_local)
        y_resid = predict_frozen(frozen, x)
        rom_out = nothing
        if rom_predictor === nothing
            y = y_resid
        else
            ensure_rom_cache!(rom_predictor, θ_local)
            cache = rom_predictor.cache
            cache === nothing && error("ROM cache not initialized.")
            if state_full === nothing
                state_full = copy(cache.nsss)
            end
            state_full[rom_predictor.state_idx] = state
            rom_next_full = rom_step_full(cache, state_full, shock_t)
            if rom_fallback_order !== nothing && !all(isfinite, rom_next_full)
                if rom_fallback_cache === nothing
                    rom_fallback_cache = build_rom_cache(rom_predictor.model, rom_fallback_order;
                                                        params = rom_predictor.base_params,
                                                        use_obc = rom_predictor.use_obc)
                end
                rom_next_full = rom_step_full(rom_fallback_cache, state_full, shock_t)
            end
            rom_out = vcat(rom_next_full[rom_predictor.obs_idx], rom_next_full[rom_predictor.state_idx])
            y = rom_out .+ y_resid
            state_full = rom_next_full
            state_full[rom_predictor.state_idx] = y[(d_obs + 1):end]
        end
        if check_finite && !all(isfinite, y)
            bad = findall(.!isfinite.(y))
            resid_ok = all(isfinite, y_resid)
            rom_ok = rom_predictor === nothing ? true : (rom_out !== nothing && all(isfinite, rom_out))
            state_ok = rom_predictor === nothing ? true : (state_full !== nothing && all(isfinite, state_full))
            println("Surrogate rollout non-finite")
            println("  resid_ok=$resid_ok rom_ok=$rom_ok state_ok=$state_ok")
            error("Surrogate rollout produced non-finite values (bad indices=$(bad[1:min(end, 5)])).")
        end
        return y[1:d_obs], y[(d_obs + 1):end]
    end

    return MacroModelling.rollout_observations(
        predict_surrogate,
        s0,
        shocks,
        θ;
        check_finite = false,
    )
end

function rom_rollout(cache::RomCache,
                     s0::Vector{Float64},
                     shocks::AbstractMatrix,
                     obs_idx::Vector{Int},
                     state_idx::Vector{Int})
    T = size(shocks, 2)
    obs = zeros(length(obs_idx), T)
    states = zeros(length(state_idx), T)
    state_full = copy(cache.nsss)
    state_full[state_idx] = s0
    for t in 1:T
        next_full = rom_step_full(cache, state_full, shocks[:, t])
        obs[:, t] = next_full[obs_idx]
        states[:, t] = next_full[state_idx]
        state_full = next_full
    end
    return obs, states
end

function surrogate_residual_obs(frozen::FrozenMLP,
                                rom_states::AbstractMatrix,
                                shocks::AbstractMatrix,
                                θ::Vector{Float64},
                                rom_obs::AbstractMatrix,
                                d_obs::Int;
                                check_finite::Bool = false)
    T = size(shocks, 2)
    obs = copy(rom_obs)
    for t in 1:T
        x = vcat(rom_states[:, t], shocks[:, t], θ)
        y_resid = predict_frozen(frozen, x)
        length(y_resid) < d_obs && error("Residual output size mismatch: expected at least $d_obs, got $(length(y_resid)).")
        if check_finite && !all(isfinite, y_resid)
            bad = findall(.!isfinite.(y_resid))
            error("Residual surrogate produced non-finite values at t=$t (bad indices=$(bad[1:min(end, 5)])).")
        end
        obs[:, t] .+= y_resid[1:d_obs]
    end
    return obs
end

function sep_irf_extended_path(model,
                               shock::Symbol,
                               shock_size::Float64;
                               periods::Int,
                               variables::Vector{Symbol},
                               sep_horizon::Int,
                               sep_order::Int,
                               sep_nnodes::Int,
                               sep_maxit::Int,
                               sep_tol::Float64,
                               sep_sparse_tree::Bool,
                               sep_shock_scale::Float64,
                               shock_scaling::Symbol)
    shock_idx = findfirst(==(shock), model.exo)
    shock_idx === nothing && error("Shock $shock not found in model.")
    nshocks = length(model.exo)
    shock_value = MacroModelling.sep_irf_shock_scale(model, shock, shock_size;
                                                     shock_scaling = shock_scaling,
                                                     negative_shock = false)
    shocks_zero = zeros(nshocks, periods)
    shocks_shock = zeros(nshocks, periods)
    shocks_shock[shock_idx, 1] = shock_value

    base = simulate_sep_extended_path(model;
                                      periods = periods,
                                      burn_in = 0,
                                      shocks = shocks_zero,
                                      sep_horizon = sep_horizon,
                                      sep_order = sep_order,
                                      sep_nnodes = sep_nnodes,
                                      sep_maxit = sep_maxit,
                                      sep_tol = sep_tol,
                                      sep_sparse_tree = sep_sparse_tree,
                                      sep_shock_scale = sep_shock_scale,
                                      shock_scaling = shock_scaling,
                                      silent = true)
    base.errorflag && error("SEP baseline failed at period $(base.failure_period).")

    shocked = simulate_sep_extended_path(model;
                                         periods = periods,
                                         burn_in = 0,
                                         shocks = shocks_shock,
                                         sep_horizon = sep_horizon,
                                         sep_order = sep_order,
                                         sep_nnodes = sep_nnodes,
                                         sep_maxit = sep_maxit,
                                         sep_tol = sep_tol,
                                         sep_sparse_tree = sep_sparse_tree,
                                         sep_shock_scale = sep_shock_scale,
                                         shock_scaling = shock_scaling,
                                         silent = true)
    shocked.errorflag && error("SEP shocked path failed at period $(shocked.failure_period).")

    var_idx = [findfirst(==(v), axiskeys(base.simulation, 1)) for v in variables]
    if any(isnothing, var_idx)
        missing = variables[findall(isnothing, var_idx)]
        error("Variables not found in SEP simulation: $(missing)")
    end
    var_idx = Int.(var_idx)
    sim_base = Array(base.simulation)
    sim_shock = Array(shocked.simulation)
    return sim_shock[var_idx, 2:end] .- sim_base[var_idx, 2:end]
end

synthetic_path = parse_arg_string(ARGS, "--synthetic", "")
if synthetic_path == ""
    synthetic_path = first_positional_arg(ARGS)
end
if synthetic_path === nothing || synthetic_path == ""
    error("Usage: julia hlt_regime_switching_illustration.jl <synthetic_path> [--surrogate=PATH]")
end

surrogate_path = parse_arg_string(ARGS, "--surrogate", "")
out_dir = parse_arg_string(ARGS, "--out-dir", "")
irf_periods = parse_arg_int(ARGS, "--irf-periods", 20)
irf_small = parse_arg_float(ARGS, "--irf-small", 0.5)
irf_large = parse_arg_float(ARGS, "--irf-large", 2.0)
irf_shock_scale_arg = parse_arg_float(ARGS, "--irf-shock-scale", NaN)
irf_method = parse_arg_symbol(ARGS, "--irf-method", :funnel)
irf_burn_in = parse_arg_int(ARGS, "--irf-burn-in", 50)
irf_sep_periods = parse_arg_int(ARGS, "--irf-sep-periods", 0)
irf_sep_nnodes = parse_arg_int(ARGS, "--irf-sep-nnodes", 0)
irf_sep_maxit = parse_arg_int(ARGS, "--irf-sep-maxit", 0)
irf_sep_shock_scale_arg = parse_arg_float(ARGS, "--irf-sep-shock-scale", NaN)
irf_vars_arg = parse_arg_string(ARGS, "--irf-vars", "")
plot_vars_arg = parse_arg_string(ARGS, "--plot-vars", "")
irf_shock_arg = parse_arg_string(ARGS, "--irf-shock", "")
disable_rom_residual = "--surrogate-no-rom" in ARGS
debug_surrogate = "--surrogate-debug" in ARGS
rom_fallback_order = parse_arg_int(ARGS, "--rom-fallback-order", 1)
disable_rom_fallback = "--no-rom-fallback" in ARGS
skip_irf = parse_arg_bool(ARGS, "--skip-irf", false)

force_obc = "--use-obc" in ARGS
force_no_obc = "--no-obc" in ARGS
if force_obc && force_no_obc
    error("Specify only one of --use-obc or --no-obc.")
end

synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
synthetic_model_name = get(synthetic, "model", "Smets_Wouters_2007_HLT")
if force_obc
    use_obc = true
elseif force_no_obc
    use_obc = false
else
    use_obc = false
    if synthetic_model_name == "Smets_Wouters_2007_HLT_obc"
        println("Warning: synthetic model is OBC; defaulting to non-OBC HLT. Use --use-obc to override.")
    elseif synthetic_model_name != "Smets_Wouters_2007_HLT"
        println("Warning: unknown synthetic model '$synthetic_model_name'; defaulting to non-OBC HLT. Use --use-obc to override.")
    end
end

model_name = use_obc ? "Smets_Wouters_2007_HLT_obc" : "Smets_Wouters_2007_HLT"
mm_model = load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)
base_values = apply_paper_calibration!(mm_model)

observables = haskey(synthetic, "observables") ? Symbol.(synthetic["observables"]) : [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_data = Array(synthetic["obs_data"])
shocks = Array(synthetic["shocks"])
s0 = Float64.(synthetic["s0"])
theta_true = Float64.(synthetic["theta_true"])
state_idx = Int.(synthetic["state_idx"])
shock_scaling = Symbol(get(synthetic, "shock_scaling", :parameter))
vol_start = get(synthetic, "vol_start", 0)
vol_end = get(synthetic, "vol_end", 0)
vol_mult = get(synthetic, "vol_mult", 1.0)
vol_shocks = get(synthetic, "vol_shocks", "all")

if out_dir == ""
    out_dir = dirname(synthetic_path)
end
mkpath(out_dir)

if surrogate_path == ""
    surrogate_path = joinpath(dirname(synthetic_path), "hlt_sep_surrogate_trained.jls")
end
if !isfile(surrogate_path)
    error("Surrogate not found at $surrogate_path")
end

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
surrogate = surrogate_bundle.payload
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
sur_shock_scale = get(sur_meta, "shock_scale", NaN)
sur_shock_scaling = Symbol(get(sur_meta, "shock_scaling", shock_scaling))
sur_sep_shock_scale = get(sur_meta, "sep_shock_scale", NaN)
if sur_shock_scaling != shock_scaling
    println("Warning: surrogate shock_scaling=$(sur_shock_scaling) differs from synthetic shock_scaling=$(shock_scaling).")
end
irf_shock_scale = isnan(irf_shock_scale_arg) ? (isnan(sur_shock_scale) ? 1.0 : sur_shock_scale) : irf_shock_scale_arg
irf_sep_shock_scale = isnan(irf_sep_shock_scale_arg) ? (isnan(sur_sep_shock_scale) ? 1.0 : sur_sep_shock_scale) : irf_sep_shock_scale_arg
if !isnan(sur_shock_scale) && abs(irf_shock_scale - sur_shock_scale) > 1e-8
    println("Warning: IRF shock scale ($irf_shock_scale) differs from surrogate training shock_scale ($sur_shock_scale).")
end

d_obs = size(obs_data, 1)
d_state = length(state_idx)
d_eps = size(shocks, 1)
d_theta = length(theta_true)
expected_in = d_state + d_eps + d_theta
expected_out = d_obs + d_state
obs_only_surrogate = false
if frozen.d_in != expected_in
    error("Surrogate input size mismatch: expected $expected_in, got $(frozen.d_in).")
end
if frozen.d_out == d_obs
    obs_only_surrogate = true
elseif frozen.d_out != expected_out
    error("Surrogate output size mismatch: expected $expected_out or $d_obs, got $(frozen.d_out).")
end

obs_idx = indexin(observables, mm_model.var)
if any(isnothing, obs_idx)
    error("Observable names not found in $(mm_model.model_name).")
end
obs_idx = Int.(obs_idx)

rom_residual = get(sur_meta, "rom_residual", false)
rom_order = Int(get(sur_meta, "rom_residual_order", 0))
rom_mode_raw = get(sur_meta, "rom_mode", :baseline)
rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)
rom_predictor = nothing
if rom_residual && rom_order != 0 && !disable_rom_residual
    theta_names = get(sur_meta, "theta_names", Symbol[])
    theta_idx = Int[]
    if rom_mode == :theta
        theta_idx = indexin(theta_names, mm_model.parameters)
        if any(isnothing, theta_idx)
            error("Theta names not found in $(mm_model.model_name) parameters for rom_mode=:theta.")
        end
        println("Warning: rom_mode=:theta will rebuild ROM in the illustration.")
    end
    rom_predictor = RomPredictor(mm_model,
                                 rom_order,
                                 rom_mode,
                                 use_obc,
                                 Int.(theta_idx),
                                 base_values,
                                 nothing,
                                 nothing,
                                 state_idx,
                                 obs_idx)
end
if disable_rom_residual
    println("Surrogate ROM residual disabled (--surrogate-no-rom).")
end
rom_fallback = (rom_residual && rom_order > 1 && !disable_rom_residual && !disable_rom_fallback) ? rom_fallback_order : nothing
if rom_fallback !== nothing
    println("ROM fallback enabled (order=$rom_fallback).")
end
if obs_only_surrogate && rom_predictor === nothing
    error("Obs-only surrogate requires ROM residual mode. Remove --surrogate-no-rom or provide a ROM residual surrogate.")
end

SS_result = get_steady_state(mm_model, derivatives = false)
yss = [Float64(SS_result(var)) for var in mm_model.var]
initial_state = copy(yss)
initial_state[state_idx] = s0

T = size(obs_data, 2)
rom_cache = nothing
rom_obs = nothing
rom_state = nothing
if rom_predictor !== nothing
    ensure_rom_cache!(rom_predictor, theta_true)
    rom_cache = rom_predictor.cache
    rom_cache === nothing && error("ROM cache not initialized.")
    rom_obs, rom_state = rom_rollout(rom_cache, s0, shocks, obs_idx, state_idx)
else
    rom_irf = get_irf(mm_model;
                      algorithm = :first_order,
                      shocks = shocks,
                      periods = T,
                      initial_state = initial_state,
                      levels = true,
                      variables = observables,
                      ignore_obc = false)
    rom_obs_full = Array(rom_irf)[:, :, 1]
    rom_obs = rom_obs_full[:, 1:T]
end
sep_obs = obs_data
if rom_state !== nothing
    sur_obs_full = surrogate_residual_obs(frozen, rom_state, shocks, theta_true, rom_obs, d_obs;
                                          check_finite = debug_surrogate)
else
    sur_obs_full = surrogate_rollout(frozen, s0, shocks, theta_true, d_obs;
                                     rom_predictor = rom_predictor,
                                     check_finite = debug_surrogate,
                                     rom_fallback_order = rom_fallback)
end

window_start = max(1, vol_start)
window_end = min(T, vol_end)
has_vol = vol_start > 0 && vol_end >= vol_start && window_start <= window_end
if has_vol
    in_window = window_start:window_end
    out_window = setdiff(1:T, in_window)
    sur_obs = copy(rom_obs)
    sur_obs[:, in_window] .= sur_obs_full[:, in_window]
    println("Windowed surrogate active: using ROM1 outside [$window_start, $window_end].")
else
    in_window = 1:T
    out_window = Int[]
    sur_obs = sur_obs_full
end

err_rom = [norm(sep_obs[:, t] - rom_obs[:, t]) for t in 1:T]
err_sur = [norm(sep_obs[:, t] - sur_obs[:, t]) for t in 1:T]

if has_vol
    println("Error stats (ROM vs SEP): inside=$(mean(err_rom[in_window])) outside=$(mean(err_rom[out_window]))")
    println("Error stats (ROM+Surrogate vs SEP): inside=$(mean(err_sur[in_window])) outside=$(mean(err_sur[out_window]))")
    if mean(err_rom[in_window]) > 0
        gain = 1.0 - mean(err_sur[in_window]) / mean(err_rom[in_window])
        println("Windowed gain vs ROM1: $(round(100 * gain, digits = 2))%")
    end
else
    println("Error stats (ROM vs SEP): mean=$(mean(err_rom))")
    println("Error stats (ROM+Surrogate vs SEP): mean=$(mean(err_sur))")
end

obc_mask = contains.(string.(mm_model.exo), "ᵒᵇᶜ")
structural_idx = findall(!, obc_mask)
shock_norm = vec(sqrt.(sum(shocks[structural_idx, :].^2, dims = 1)))

plot_vars = isempty(plot_vars_arg) ? observables[1:min(3, length(observables))] : parse_symbol_list(plot_vars_arg)
irf_vars = isempty(irf_vars_arg) ? observables : parse_symbol_list(irf_vars_arg)

function add_vol_span!(p)
    if has_vol
        vspan!(p, [window_start, window_end], color = :orange, alpha = 0.15, label = "")
    end
end

default(show = false)

x_axis = collect(1.0:1.0:T)
n_rows = length(plot_vars) + 1
p_series = plot(layout = (n_rows, 1), size = (1200, 250 * n_rows),
                plot_title = "HLT: series and shocks (high-vol window x$(vol_mult))")
@assert length(vec(shock_norm)) == T
plot!(p_series[1], x_axis, vec(shock_norm), color = :black, label = "shock norm")
add_vol_span!(p_series[1])
plot!(p_series[1], ylabel = "shock norm")

for (i, var) in enumerate(plot_vars)
    idx = findfirst(==(var), observables)
    if idx === nothing
        error("Plot variable $var not found in observables.")
    end
    @assert length(vec(sep_obs[idx, :])) == T
    @assert length(vec(rom_obs[idx, :])) == T
    @assert length(vec(sur_obs[idx, :])) == T
    show_legend = i == 1
    plot!(p_series[i + 1], x_axis, vec(sep_obs[idx, :]), color = :black, label = show_legend ? "SEP" : "")
    plot!(p_series[i + 1], x_axis, vec(rom_obs[idx, :]), color = :blue, linestyle = :dash, label = show_legend ? "ROM1" : "")
    plot!(p_series[i + 1], x_axis, vec(sur_obs[idx, :]), color = :red, linestyle = :dot,
          label = show_legend ? (has_vol ? "ROM1+Surrogate" : "Surrogate") : "")
    add_vol_span!(p_series[i + 1])
    plot!(p_series[i + 1], ylabel = string(var))
end

series_pdf = joinpath(out_dir, "hlt_regime_switching_series.pdf")
savefig(p_series, series_pdf)
println("Saved: $series_pdf")

p_err = plot(size = (1200, 300),
             title = "ROM vs Surrogate error vs SEP",
             xlabel = "t",
             ylabel = "L2 error")
plot!(p_err, x_axis, vec(err_rom), color = :blue, label = "ROM1 error")
plot!(p_err, x_axis, vec(err_sur), color = :red, label = has_vol ? "ROM1+Surrogate error" : "Surrogate error")
add_vol_span!(p_err)

errors_pdf = joinpath(out_dir, "hlt_regime_switching_errors.pdf")
savefig(p_err, errors_pdf)
println("Saved: $errors_pdf")

if skip_irf
    println("Skipping IRF diagnostics (--skip-irf=true).")
    exit(0)
end

default_shock = :epmu
if !(default_shock in mm_model.exo)
    default_shock = :epinf
end
if !(default_shock in mm_model.exo)
    default_shock = mm_model.exo[structural_idx[1]]
end
irf_shock = irf_shock_arg == "" ? default_shock : Symbol(irf_shock_arg)
if findfirst(==(irf_shock), mm_model.exo) === nothing
    error("IRF shock $irf_shock not found in model.")
end
for var in irf_vars
    if !(var in observables)
        error("IRF variable $var must be in observables for surrogate comparison.")
    end
end

sep_periods_default = max(irf_periods, get(synthetic, "sep_horizon", 40), 40)
sep_periods = irf_sep_periods > 0 ? max(irf_periods, irf_sep_periods) : sep_periods_default
sep_horizon = irf_sep_periods > 0 ? irf_sep_periods : get(synthetic, "sep_horizon", 40)
sep_order = get(synthetic, "sep_order", 1)
sep_nnodes_default = get(synthetic, "sep_nnodes", 3)
sep_nnodes = irf_sep_nnodes > 0 ? irf_sep_nnodes : sep_nnodes_default
sep_tol = get(synthetic, "sep_tol", 1e-5)
sep_sparse_tree = get(synthetic, "sep_sparse_tree", true)
sep_maxit_default = get(synthetic, "sep_maxit", 80)
sep_maxit = irf_sep_maxit > 0 ? irf_sep_maxit : sep_maxit_default

if irf_method == :simulation
    solve!(mm_model;
           algorithm = :stochastic_extended_path,
           sep_periods = sep_periods,
           sep_order = sep_order,
           sep_nnodes = sep_nnodes,
           sep_maxit = sep_maxit,
           sep_tol = sep_tol,
           sep_sparse_tree = sep_sparse_tree,
           silent = true)
end

irf_small_eff = irf_small * irf_shock_scale
irf_large_eff = irf_large * irf_shock_scale
println("IRF shock scaling: base=$(irf_shock_scale), small=$(irf_small_eff), large=$(irf_large_eff)")

rom1_small = get_irf(mm_model;
                     algorithm = :first_order,
                     shocks = irf_shock,
                     shock_size = shock_value_for(mm_model, irf_shock, irf_small_eff; shock_scaling = shock_scaling),
                     periods = irf_periods,
                     variables = irf_vars,
                     ignore_obc = false)
rom1_large = get_irf(mm_model;
                     algorithm = :first_order,
                     shocks = irf_shock,
                     shock_size = shock_value_for(mm_model, irf_shock, irf_large_eff; shock_scaling = shock_scaling),
                     periods = irf_periods,
                     variables = irf_vars,
                     ignore_obc = false)

if irf_method == :extended_path
    sep_small = sep_irf_extended_path(mm_model, irf_shock, irf_small_eff;
                                      variables = irf_vars,
                                      periods = irf_periods,
                                      sep_horizon = sep_horizon,
                                      sep_order = sep_order,
                                      sep_nnodes = sep_nnodes,
                                      sep_maxit = sep_maxit,
                                      sep_tol = sep_tol,
                                      sep_sparse_tree = sep_sparse_tree,
                                      sep_shock_scale = irf_sep_shock_scale,
                                      shock_scaling = shock_scaling)
    sep_large = sep_irf_extended_path(mm_model, irf_shock, irf_large_eff;
                                      variables = irf_vars,
                                      periods = irf_periods,
                                      sep_horizon = sep_horizon,
                                      sep_order = sep_order,
                                      sep_nnodes = sep_nnodes,
                                      sep_maxit = sep_maxit,
                                      sep_tol = sep_tol,
                                      sep_sparse_tree = sep_sparse_tree,
                                      sep_shock_scale = irf_sep_shock_scale,
                                      shock_scaling = shock_scaling)
else
    sep_small = get_sep_irf(mm_model, irf_shock, irf_small_eff;
                            variables = irf_vars,
                            periods = irf_periods,
                            method = irf_method,
                            baseline = :zero_shock,
                            burn_in = irf_burn_in,
                            shock_scaling = shock_scaling,
                            sep_periods = sep_periods,
                            sep_order = sep_order,
                            sep_nnodes = sep_nnodes,
                            sep_maxit = sep_maxit,
                            sep_tol = sep_tol,
                            sep_sparse_tree = sep_sparse_tree,
                            silent = true)
    sep_large = get_sep_irf(mm_model, irf_shock, irf_large_eff;
                            variables = irf_vars,
                            periods = irf_periods,
                            method = irf_method,
                            baseline = :zero_shock,
                            burn_in = irf_burn_in,
                            shock_scaling = shock_scaling,
                            sep_periods = sep_periods,
                            sep_order = sep_order,
                            sep_nnodes = sep_nnodes,
                            sep_maxit = sep_maxit,
                            sep_tol = sep_tol,
                            sep_sparse_tree = sep_sparse_tree,
                            silent = true)
end

shock_idx = findfirst(==(irf_shock), mm_model.exo)
shock_small_value = MacroModelling.sep_irf_shock_scale(mm_model, irf_shock, irf_small_eff;
                                                      shock_scaling = shock_scaling,
                                                      negative_shock = false)
shock_large_value = MacroModelling.sep_irf_shock_scale(mm_model, irf_shock, irf_large_eff;
                                                      shock_scaling = shock_scaling,
                                                      negative_shock = false)
shock_small_matrix = zeros(length(mm_model.exo), irf_periods)
shock_large_matrix = zeros(length(mm_model.exo), irf_periods)
shock_small_matrix[shock_idx, 1] = shock_small_value
shock_large_matrix[shock_idx, 1] = shock_large_value

state0 = yss[state_idx]
if rom_cache !== nothing
    rom_obs_small, rom_state_small = rom_rollout(rom_cache, state0, shock_small_matrix, obs_idx, state_idx)
    rom_obs_large, rom_state_large = rom_rollout(rom_cache, state0, shock_large_matrix, obs_idx, state_idx)
    rom_obs_zero, rom_state_zero = rom_rollout(rom_cache, state0, zeros(length(mm_model.exo), irf_periods), obs_idx, state_idx)
    sur_small = surrogate_residual_obs(frozen, rom_state_small, shock_small_matrix, theta_true, rom_obs_small, d_obs)
    sur_large = surrogate_residual_obs(frozen, rom_state_large, shock_large_matrix, theta_true, rom_obs_large, d_obs)
    sur_zero = surrogate_residual_obs(frozen, rom_state_zero, zeros(length(mm_model.exo), irf_periods), theta_true, rom_obs_zero, d_obs)
else
    sur_small = surrogate_rollout(frozen, state0, shock_small_matrix, theta_true, d_obs;
                                  rom_predictor = rom_predictor,
                                  rom_fallback_order = rom_fallback)
    sur_large = surrogate_rollout(frozen, state0, shock_large_matrix, theta_true, d_obs;
                                  rom_predictor = rom_predictor,
                                  rom_fallback_order = rom_fallback)
    sur_zero = surrogate_rollout(frozen, state0, zeros(length(mm_model.exo), irf_periods), theta_true, d_obs;
                                 rom_predictor = rom_predictor,
                                 rom_fallback_order = rom_fallback)
end
sur_small_irf = sur_small .- sur_zero
sur_large_irf = sur_large .- sur_zero

function extract_series(irf, var::Symbol, periods::Int)
    if irf isa AbstractMatrix
        return irf[findfirst(==(var), irf_vars), 1:periods]
    end
    idx = findfirst(==(var), axiskeys(irf, 1))
    if idx === nothing
        return nothing
    end
    series = ndims(irf) == 3 ? Float64.(irf[idx, :, 1]) : Float64.(irf[idx, :])
    return length(series) == periods + 1 ? series[2:end] : series[1:periods]
end

p_irf = plot(layout = (length(irf_vars), 2),
             size = (1200, 300 * length(irf_vars)),
             plot_title = "IRF comparison (small vs large shocks)")

rmse(v, ref) = sqrt(mean((v .- ref) .^ 2))
rom1_err_small = Float64[]
rom1_err_large = Float64[]
sur_err_small = Float64[]
sur_err_large = Float64[]

for (i, var) in enumerate(irf_vars)
    rom1_s = extract_series(rom1_small, var, irf_periods)
    rom1_l = extract_series(rom1_large, var, irf_periods)
    sep_s = extract_series(sep_small, var, irf_periods)
    sep_l = extract_series(sep_large, var, irf_periods)
    obs_pos = findfirst(==(var), observables)
    sur_s = sur_small_irf[obs_pos, :]
    sur_l = sur_large_irf[obs_pos, :]

    push!(rom1_err_small, rmse(rom1_s, sep_s))
    push!(rom1_err_large, rmse(rom1_l, sep_l))
    push!(sur_err_small, rmse(sur_s, sep_s))
    push!(sur_err_large, rmse(sur_l, sep_l))

    show_legend = i == 1
    plot!(p_irf[2 * (i - 1) + 1], 1:irf_periods, rom1_s,
          label = show_legend ? "ROM1" : "", color = :blue, linestyle = :dash)
    plot!(p_irf[2 * (i - 1) + 1], 1:irf_periods, sep_s,
          label = show_legend ? "SEP" : "", color = :black)
    plot!(p_irf[2 * (i - 1) + 1], 1:irf_periods, sur_s,
          label = show_legend ? "Surrogate" : "", color = :red, linestyle = :dot)
    hline!(p_irf[2 * (i - 1) + 1], [0.0], color = :gray, linestyle = :dot, label = "")
    plot!(p_irf[2 * (i - 1) + 1], title = string(var) * " (small)")

    plot!(p_irf[2 * (i - 1) + 2], 1:irf_periods, rom1_l,
          label = show_legend ? "ROM1" : "", color = :blue, linestyle = :dash)
    plot!(p_irf[2 * (i - 1) + 2], 1:irf_periods, sep_l,
          label = show_legend ? "SEP" : "", color = :black)
    plot!(p_irf[2 * (i - 1) + 2], 1:irf_periods, sur_l,
          label = show_legend ? "Surrogate" : "", color = :red, linestyle = :dot)
    hline!(p_irf[2 * (i - 1) + 2], [0.0], color = :gray, linestyle = :dot, label = "")
    plot!(p_irf[2 * (i - 1) + 2], title = string(var) * " (large)")
end

irf_pdf = joinpath(out_dir, "hlt_regime_switching_irf_comparison.pdf")
savefig(p_irf, irf_pdf)
println("Saved: $irf_pdf")
println("IRF RMSE (small shocks): ROM1=$(round(mean(rom1_err_small), digits=5)) Surrogate=$(round(mean(sur_err_small), digits=5))")
println("IRF RMSE (large shocks): ROM1=$(round(mean(rom1_err_large), digits=5)) Surrogate=$(round(mean(sur_err_large), digits=5))")
