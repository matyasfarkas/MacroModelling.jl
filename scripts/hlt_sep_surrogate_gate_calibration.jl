#!/usr/bin/env julia
using Serialization
using Statistics
using LinearAlgebra
using Printf
using MacroModelling
using AxisKeys

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function select_model(model_name::String)
    return load_hlt_model(script_repo_root(), model_name; mod = @__MODULE__)
end

synthetic_path = first_positional_arg(ARGS)
if synthetic_path == ""
    error("Usage: julia hlt_sep_surrogate_gate_calibration.jl <synthetic_data.jls> [--out=path] [--target-share=0.1] [--shock-norm=l2] [--error-norm=l2] [--quantile=0.95] [--tau-eps=1.95] [--tau-y=...] [--periods=1] [--model=Smets_Wouters_2007_HLT_obc] [--use-obc|--no-obc]")
end

out_path = parse_arg_string(ARGS, "--out", "")
target_share = parse_arg_float(ARGS, "--target-share", 0.1)
min_achieved_share = parse_arg_float(ARGS, "--min-achieved-share", 0.0)
max_achieved_share = parse_arg_float(ARGS, "--max-achieved-share", 1.0)
max_target_share_error = parse_arg_float(ARGS, "--max-target-share-error", Inf)
fail_unreachable = parse_arg_bool(ARGS, "--fail-unreachable", false)
shock_norm = parse_arg_symbol(ARGS, "--shock-norm", :l2)
error_norm = parse_arg_symbol(ARGS, "--error-norm", :l2)
quantile_arg = parse_arg_float(ARGS, "--quantile", NaN)
tau_eps_arg = parse_arg_float(ARGS, "--tau-eps", 1.95)
tau_y_arg = parse_arg_float(ARGS, "--tau-y", NaN)
use_eps = parse_arg_bool(ARGS, "--use-eps", true)
use_y = parse_arg_bool(ARGS, "--use-y", true)
shock_source = parse_arg_symbol(ARGS, "--shock-source", :synthetic)
shock_filter = parse_arg_symbol(ARGS, "--shock-filter", :kalman)
periods = parse_arg_int(ARGS, "--periods", 1)
model_override = parse_arg_string(ARGS, "--model", "")
force_obc = "--use-obc" in ARGS
force_no_obc = "--no-obc" in ARGS
if force_obc && force_no_obc
    error("Specify only one of --use-obc or --no-obc.")
end
if !use_eps && !use_y
    error("At least one of --use-eps or --use-y must be true.")
end
if !(0.0 <= min_achieved_share < max_achieved_share <= 1.0)
    error("Invalid achieved share bounds: [$min_achieved_share, $max_achieved_share].")
end
if !(shock_source in (:synthetic, :filtered))
    error("Unknown --shock-source=$(shock_source). Use :synthetic or :filtered.")
end

synthetic = MacroModelling.load_hlt_synthetic_scenario(synthetic_path)
model_name = model_override != "" ? model_override : get(synthetic, "model", "Smets_Wouters_2007_HLT")
if force_obc
    model_name = "Smets_Wouters_2007_HLT_obc"
elseif force_no_obc
    model_name = "Smets_Wouters_2007_HLT"
end
if model_name != get(synthetic, "model", model_name)
    println("Warning: calibration model $model_name differs from synthetic model $(get(synthetic, "model", "unknown")).")
end
hlt_model_file_and_symbol(model_name)  # fail fast on unsupported names
model = select_model(model_name)
if !use_eps
    tau_eps_arg = Inf
end
if !use_y
    tau_y_arg = Inf
end

obs_data = synthetic["obs_data"]
obs_sigma = synthetic["obs_sigma"]
shocks = synthetic["shocks"]
shock_sigmas = haskey(synthetic, "shock_sigmas") ? synthetic["shock_sigmas"] : vec(std(shocks, dims = 2))
observables = synthetic["observables"]

T = size(obs_data, 2)
if shock_source == :filtered
    shocks = MacroModelling.estimate_observed_shocks_matrix(
        model,
        obs_data,
        observables;
        parameters = model.parameter_values,
        algorithm = :first_order,
        filter = shock_filter,
        data_in_levels = true,
        smooth = false,
        verbose = false,
        expected_cols = T,
        label = "Filtered shock matrix",
    )
end
if size(shocks, 2) < T
    error("Shock matrix shorter than obs_data (shocks T=$(size(shocks,2)), obs T=$T)")
elseif size(shocks, 2) > T
    shocks = shocks[:, 1:T]
end

lin_obs, e_stat, f_stat = MacroModelling.compute_linear_gate_stats_from_shocks(
    model,
    obs_data,
    observables,
    shocks,
    obs_sigma,
    shock_sigmas;
    periods = periods,
    shock_norm = shock_norm,
    error_norm = error_norm,
    ignore_obc = false,
    label = "Linear simulation",
)
structural_idx = findall(shock_sigmas .> 0)

target_reachable = true
reachability_reason = ""
share_floor_fixed_tau_eps = use_eps && !isnan(tau_eps_arg) && isfinite(tau_eps_arg) ? mean(e_stat .> tau_eps_arg) : NaN
share_floor_fixed_tau_y = use_y && !isnan(tau_y_arg) && isfinite(tau_y_arg) ? mean(f_stat .> tau_y_arg) : NaN

if !isnan(tau_eps_arg) && !isnan(tau_y_arg)
    q = NaN
    tau_e = tau_eps_arg
    tau_f = tau_y_arg
    share = MacroModelling.gate_share(e_stat, f_stat, tau_e, tau_f; use_eps = use_eps, use_y = use_y)
elseif !isnan(tau_eps_arg)
    q = NaN
    tau_e = tau_eps_arg
    share_floor = mean(use_eps ? (e_stat .> tau_e) : falses(length(e_stat)))
    if share_floor > target_share
        tau_f = maximum(f_stat)
        share = share_floor
        target_reachable = false
        reachability_reason = "fixed_tau_eps"
        msg = "target_share=$(target_share) cannot be reached with tau_eps=$(tau_e). Minimum share is $(round(share_floor, digits=4))."
        println("Warning: $msg")
        fail_unreachable && error(msg)
    else
        tau_f, share = MacroModelling.calibrate_tau_y(e_stat, f_stat, tau_e, target_share)
    end
elseif !isnan(tau_y_arg)
    q = NaN
    tau_f = tau_y_arg
    share_floor = mean(use_y ? (f_stat .> tau_f) : falses(length(f_stat)))
    if share_floor > target_share
        tau_e = maximum(e_stat)
        share = share_floor
        target_reachable = false
        reachability_reason = "fixed_tau_y"
        msg = "target_share=$(target_share) cannot be reached with tau_y=$(tau_f). Minimum share is $(round(share_floor, digits=4))."
        println("Warning: $msg")
        fail_unreachable && error(msg)
    else
        tau_e, share = MacroModelling.calibrate_tau_eps(e_stat, f_stat, tau_f, target_share)
    end
elseif isnan(quantile_arg)
    gate_result = MacroModelling.calibrate_gate(
        e_stat, f_stat;
        config = MacroModelling.GateCalibrationConfig(
            target_share = target_share,
            use_eps = use_eps,
            use_y = use_y,
        ),
    )
    q = gate_result.quantile
    tau_e = gate_result.tau_eps
    tau_f = gate_result.tau_y
    share = gate_result.achieved_share
else
    q = quantile_arg
    tau_e = quantile(e_stat, q)
    tau_f = quantile(f_stat, q)
    share = MacroModelling.gate_share(e_stat, f_stat, tau_e, tau_f; use_eps = use_eps, use_y = use_y)
end

isfinite(share) || error("Gate calibration produced non-finite achieved share.")
share_error = abs(share - target_share)
if share < min_achieved_share || share > max_achieved_share
    error("Achieved gate share $(round(share, digits=4)) is outside requested bounds [$min_achieved_share, $max_achieved_share].")
end
if isfinite(max_target_share_error) && share_error > max_target_share_error
    error("Gate share error $(round(share_error, digits=4)) exceeds --max-target-share-error=$max_target_share_error.")
end

if out_path == ""
    out_path = joinpath(dirname(synthetic_path), "gate_calibration.jls")
end

payload = Dict(
    "model" => model_name,
    "synthetic_path" => synthetic_path,
    "target_share" => target_share,
    "quantile" => q,
    "achieved_share" => share,
    "share_error" => share_error,
    "target_reachable" => target_reachable,
    "reachability_reason" => reachability_reason,
    "min_achieved_share" => min_achieved_share,
    "max_achieved_share" => max_achieved_share,
    "max_target_share_error" => max_target_share_error,
    "fail_unreachable" => fail_unreachable,
    "share_floor_fixed_tau_eps" => share_floor_fixed_tau_eps,
    "share_floor_fixed_tau_y" => share_floor_fixed_tau_y,
    "tau_eps" => tau_e,
    "tau_y" => tau_f,
    "periods" => periods,
    "shock_norm" => String(shock_norm),
    "error_norm" => String(error_norm),
    "use_eps" => use_eps,
    "use_y" => use_y,
    "shock_source" => String(shock_source),
    "shock_filter" => String(shock_filter),
    "shock_sigmas" => shock_sigmas,
    "obs_sigma" => obs_sigma,
    "structural_idx" => structural_idx,
    "e_stats" => e_stat,
    "f_stats" => f_stat,
)

serialize(out_path, payload)

println("Gate calibration")
println("  Model: $model_name")
println("  Target share: $target_share")
println("  Quantile: $(round(q, digits=4))")
println("  Achieved share: $(round(share, digits=4))")
println("  Share error: $(round(share_error, digits=4))")
reachability_suffix = isempty(reachability_reason) ? "" : " ($reachability_reason)"
println("  Target reachable: $target_reachable$reachability_suffix")
println("  tau_eps: $(round(tau_e, digits=4))")
println("  tau_y: $(round(tau_f, digits=4))")
println("  Saved: $out_path")
