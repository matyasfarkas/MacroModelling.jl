#!/usr/bin/env julia
using Serialization
using Statistics
using LinearAlgebra
using AxisKeys
using Printf
using MCMCChains

ENV["GKSwstype"] = "100"
using StatsPlots
using MacroModelling

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

script_repo_root() = normpath(joinpath(@__DIR__, ".."))

function first_two_positional_args(args::Vector{String})
    out = String[]
    for arg in args
        if !startswith(arg, "--")
            push!(out, arg)
            if length(out) == 2
                break
            end
        end
    end
    return out
end

positional = first_two_positional_args(ARGS)
surrogate_path = length(positional) >= 1 ? positional[1] : nothing
synthetic_path = length(positional) >= 2 ? positional[2] : nothing
if surrogate_path === nothing || synthetic_path === nothing
    error("Usage: julia hlt_sep_surrogate_forecast_errors.jl <trained_surrogate.jls> <synthetic_data.jls> [--chain-in=PATH]")
end

chain_in = parse_arg_string(ARGS, "--chain-in", "")
out_dir = parse_arg_string(ARGS, "--out-dir", "")
theta_mode = parse_arg_symbol(ARGS, "--theta", :synthetic)
shock_source = parse_arg_symbol(ARGS, "--shock-source", :synthetic)
gate_threshold = parse_arg_float(ARGS, "--gate-threshold", 0.5)
if !(theta_mode in (:synthetic, :baseline, :posterior))
    error("Unknown --theta=$theta_mode. Use :synthetic, :baseline, or :posterior.")
end
if !(shock_source in (:synthetic, :kalman, :inversion))
    error("Unknown --shock-source=$shock_source. Use :synthetic, :kalman, or :inversion.")
end
if !(gate_threshold > 0 && gate_threshold < 1)
    error("Invalid --gate-threshold=$gate_threshold. Use a value between 0 and 1.")
end

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
surrogate_data = surrogate_bundle.payload
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
rom_residual = get(sur_meta, "rom_residual", false)
rom_order = Int(get(sur_meta, "rom_residual_order", 0))
rom_mode_raw = get(sur_meta, "rom_mode", :baseline)
rom_mode = rom_mode_raw isa Symbol ? rom_mode_raw : Symbol(rom_mode_raw)

synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
obs_data = synthetic["obs_data"]
obs_sigma = synthetic["obs_sigma"]
s0 = synthetic["s0"]
shocks = synthetic["shocks"]
theta_true = synthetic["theta_true"]
theta_names = haskey(synthetic, "theta_names") ? synthetic["theta_names"] : Symbol[]
observables = haskey(synthetic, "observables") ? synthetic["observables"] : Symbol[]
state_names = haskey(synthetic, "state_names") ? synthetic["state_names"] : Symbol[]
shock_sigmas = haskey(synthetic, "shock_sigmas") ? synthetic["shock_sigmas"] : vec(std(shocks, dims = 2))
model_name = get(synthetic, "model", "Smets_Wouters_2007_HLT")
use_obc = model_name == "Smets_Wouters_2007_HLT_obc"
mm_model = load_hlt_model(script_repo_root(), String(model_name); mod = @__MODULE__)
using Turing
chain_payload = chain_in == "" ? nothing : MacroModelling.load_hlt_chain_payload(chain_in)

if !rom_residual || rom_order != 1
    error("This diagnostic expects a ROM1 residual surrogate (rom_residual=true, rom_order=1).")
end
if isempty(observables) || isempty(state_names)
    error("Synthetic data must contain observables and state_names.")
end

function filtered_shocks_from_theta(obs_data::AbstractMatrix,
                                    observables::Vector{Symbol},
                                    shock_sigmas::AbstractVector,
                                    theta_vec::AbstractVector,
                                    theta_names::Vector{Symbol};
                                    filter::Symbol)
    params = MacroModelling.override_named_parameters(
        mm_model.parameter_values,
        mm_model.parameters,
        theta_names,
        theta_vec;
        label = "Theta names",
    )
    shocks = MacroModelling.estimate_observed_shocks_matrix(
        mm_model,
        obs_data,
        observables;
        parameters = params,
        filter = filter,
        expected_rows = length(shock_sigmas),
        expected_cols = size(obs_data, 2),
        label = "Filtered shocks",
    )
    zero_idx = findall(shock_sigmas .== 0)
    if !isempty(zero_idx)
        shocks[zero_idx, :] .= 0.0
    end
    return shocks
end

theta = if theta_mode == :synthetic
    theta_true
elseif theta_mode == :baseline
    MacroModelling.extract_named_parameters(
        mm_model.parameter_values,
        mm_model.parameters,
        theta_names;
        label = "Theta names",
    )
else
    chain_in == "" && error("--theta=posterior requires --chain-in.")
    if haskey(chain_payload, "post_mean_theta")
        chain_payload["post_mean_theta"]
    else
        error("Chain payload missing post_mean_theta; re-run estimation to store it.")
    end
end

if shock_source == :kalman || shock_source == :inversion
    shocks = filtered_shocks_from_theta(obs_data, observables, shock_sigmas, theta, theta_names; filter = shock_source)
end

obs_idx = indexin(observables, mm_model.var)
state_idx = indexin(state_names, mm_model.var)
if any(isnothing, obs_idx) || any(isnothing, state_idx)
    error("Observable/state names not found in $(mm_model.model_name).")
end
rom_predictor = RomPredictor(mm_model,
                             rom_order,
                             rom_mode,
                             use_obc,
                             Int[],
                             copy(mm_model.parameter_values),
                             nothing,
                             nothing,
                             Int.(state_idx),
                             Int.(obs_idx))
ensure_rom_cache!(rom_predictor, theta)

T = size(obs_data, 2)
d_obs = size(obs_data, 1)
predict_rom_obs = (state, shock_t, θ_local) -> begin
    rom_full = rom_predict(rom_predictor, state, shock_t, θ_local)
    return rom_full[1:d_obs], rom_full[d_obs + 1:end]
end
predict_sur_obs = (state, shock_t, θ_local) -> begin
    rom_full = rom_predict(rom_predictor, state, shock_t, θ_local)
    y_resid = predict_frozen(frozen, vcat(state, shock_t, θ_local))
    if length(y_resid) == d_obs
        obs_sur = rom_full[1:d_obs] .+ y_resid
        state_sur = rom_full[d_obs + 1:end]
    elseif length(y_resid) == length(rom_full)
        y = rom_full .+ y_resid
        obs_sur = y[1:d_obs]
        state_sur = y[d_obs + 1:end]
    else
        error("Residual output size mismatch: got $(length(y_resid)), expected $d_obs or $(length(rom_full)).")
    end
    return obs_sur, state_sur
end
obs_rom = MacroModelling.rollout_observations(predict_rom_obs, s0, shocks, theta)
obs_sur = MacroModelling.rollout_observations(predict_sur_obs, s0, shocks, theta)
err_rom = obs_data .- obs_rom
err_sur = obs_data .- obs_sur

rmse_rom = vec(sqrt.(mean(err_rom .^ 2, dims = 1)))
rmse_sur = vec(sqrt.(mean(err_sur .^ 2, dims = 1)))
mae_rom = vec(mean(abs.(err_rom), dims = 1))
mae_sur = vec(mean(abs.(err_sur), dims = 1))
gain_rmse = rmse_rom .- rmse_sur
pred_diff = err_rom .- err_sur
pred_diff_rmse = vec(sqrt.(mean(pred_diff .^ 2, dims = 1)))
pred_diff_mae = vec(mean(abs.(pred_diff), dims = 1))
share_improve = mean(rmse_sur .< rmse_rom)
mean_gain = mean(gain_rmse)
rmse_mix = copy(rmse_rom)

gate_mask = nothing
soft_window = nothing
gate_probs = nothing
if chain_payload !== nothing
    if haskey(chain_payload, "gate_mask")
        gate_mask = Bool.(chain_payload["gate_mask"])
        if length(gate_mask) != T
            gate_mask = nothing
        end
    end
    if haskey(chain_payload, "gate_info") && haskey(chain_payload["gate_info"], "soft_window")
        soft_window = Bool.(chain_payload["gate_info"]["soft_window"])
        if length(soft_window) != T
            soft_window = nothing
        end
    elseif haskey(chain_payload, "gate_stats") && haskey(chain_payload["gate_stats"], "soft_window")
        soft_window = Bool.(chain_payload["gate_stats"]["soft_window"])
        if length(soft_window) != T
            soft_window = nothing
        end
    end
    if haskey(chain_payload, "gate_probs")
        gate_probs = chain_payload["gate_probs"]
        if gate_probs === nothing
            gate_probs = nothing
        elseif length(gate_probs) != T
            gate_probs = nothing
        end
    end
end

gate_plot_mask = gate_probs !== nothing ? gate_probs .>= gate_threshold :
                 (soft_window !== nothing ? soft_window : gate_mask)
if gate_plot_mask !== nothing
    rmse_mix[gate_plot_mask] .= rmse_sur[gate_plot_mask]
end

if out_dir == ""
    out_dir = dirname(synthetic_path)
end
mkpath(out_dir)

csv_path = joinpath(out_dir, "hlt_surrogate_forecast_errors.csv")
open(csv_path, "w") do io
    println(io, "t,rmse_rom,rmse_sur,rmse_mix,mae_rom,mae_sur,rmse_gain,pred_diff_rmse,pred_diff_mae,gate")
    for t in 1:T
        gate_val = gate_mask === nothing ? 0 : (gate_mask[t] ? 1 : 0)
        @printf(io, "%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n",
                t, rmse_rom[t], rmse_sur[t], rmse_mix[t], mae_rom[t], mae_sur[t], gain_rmse[t],
                pred_diff_rmse[t], pred_diff_mae[t], gate_val)
    end
end

p = plot(1:T, rmse_rom, label = "ROM1 RMSE", color = :blue, linewidth = 2)
plot!(p, 1:T, rmse_sur, label = "Surrogate RMSE", color = :red, linewidth = 2)
if gate_plot_mask !== nothing
    let gate_label = true, start = nothing
        for t in 1:T
            if gate_plot_mask[t] && start === nothing
                start = t
            elseif !gate_plot_mask[t] && start !== nothing
                vspan!(p, start, t - 1, color = :orange, alpha = 0.15, label = gate_label ? "Gate" : "")
                gate_label = false
                start = nothing
            end
        end
        if start !== nothing
            vspan!(p, start, T, color = :orange, alpha = 0.15, label = gate_label ? "Gate" : "")
        end
    end
end
plot!(p, xlabel = "Period", ylabel = "RMSE (obs)", title = "Per-period Forecast Errors")

pdf_path = joinpath(out_dir, "hlt_surrogate_forecast_errors.pdf")
savefig(p, pdf_path)

summary_path = joinpath(out_dir, "hlt_surrogate_forecast_errors_summary.txt")
open(summary_path, "w") do io
    println(io, "theta_mode=$theta_mode shock_source=$shock_source")
    println(io, "mean_rmse_rom=$(mean(rmse_rom)) mean_rmse_sur=$(mean(rmse_sur))")
    println(io, "mean_gain=$(mean_gain) share_improve=$(share_improve)")
    println(io, "mean_pred_diff_rmse=$(mean(pred_diff_rmse)) mean_pred_diff_mae=$(mean(pred_diff_mae))")
    println(io, "mean_rmse_mix=$(mean(rmse_mix))")
    if gate_plot_mask !== nothing
        inside = gate_plot_mask
        outside = .!gate_plot_mask
        println(io, "gate_threshold=$gate_threshold")
        println(io, "gate_share=$(mean(gate_plot_mask))")
        println(io, "mean_rmse_rom_gate=$(mean(rmse_rom[inside])) mean_rmse_sur_gate=$(mean(rmse_sur[inside]))")
        println(io, "mean_rmse_rom_out=$(mean(rmse_rom[outside])) mean_rmse_sur_out=$(mean(rmse_sur[outside]))")
        println(io, "mean_rmse_mix_gate=$(mean(rmse_mix[inside])) mean_rmse_mix_out=$(mean(rmse_mix[outside]))")
        println(io, "share_improve_gate=$(mean(rmse_sur[inside] .< rmse_rom[inside]))")
        println(io, "share_improve_out=$(mean(rmse_sur[outside] .< rmse_rom[outside]))")
        println(io, "mean_gain_gate=$(mean(gain_rmse[inside])) mean_gain_out=$(mean(gain_rmse[outside]))")
        println(io, "mean_pred_diff_rmse_gate=$(mean(pred_diff_rmse[inside])) mean_pred_diff_rmse_out=$(mean(pred_diff_rmse[outside]))")
    end
end

println("Saved: $csv_path")
println("Saved: $pdf_path")
println("Saved: $summary_path")
