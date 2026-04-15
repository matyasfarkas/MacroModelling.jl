#!/usr/bin/env julia
#
# Verify high-residual ZLB samples against the smooth OBC (softplus) approximation.
#
# For each of the N worst-residual samples in the checkpoint:
#   1. Extract state, shocks, and theta from X_ref
#   2. Set model parameters to that theta
#   3. Run a short (5-period) SEP solve from that state with those shocks
#   4. Compare: new residual vs checkpoint residual, and check if ZLB is binding
#
# Usage:
#   julia --project=. scripts/verify_zlb_high_residual_samples.jl [--n-samples=10]

using Serialization, Random, Statistics, LinearAlgebra
using MacroModelling

const REPO_ROOT = dirname(@__DIR__)
cd(REPO_ROOT)

include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "parameter_config.jl"))

function parse_kv_int(args, key, default)
    for a in args
        startswith(a, "$key=") && return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

n_samples = parse_kv_int(ARGS, "--n-samples", 10)

# ── Load checkpoint ──
println("Loading checkpoint...")
cp = deserialize(joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls"))
settings = cp["settings"]
cursor = cp["cursor"]
X_ref = cp["X"]
Y_ref = cp["Y"]
theta_ids_ref = cp["theta_ids"]
sep_residuals = cp["sep_residuals"]
theta_grid = cp["theta_grid"]

# ── Load model ──
model_name = String(settings["model_name"])
println("Loading model: $model_name")
model = load_hlt_model(REPO_ROOT, model_name; mod = @__MODULE__)
MacroModelling.solve!(model, silent = true)

# ── Parameter / index setup ──
param_set = Symbol(settings["param_set"])
theta_names = get_parameter_names(param_set)
theta_idx = Int.(indexin(theta_names, model.parameters))
base_values = copy(model.parameter_values)

state_idx = sort(unique(vcat(model.timings.past_not_future_and_mixed_idx,
                             model.timings.future_not_past_and_mixed_idx)))
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = Int.(indexin(observables, model.var))

d_state = length(state_idx)
d_eps = length(model.exo)
d_theta = length(theta_names)

r_idx = findfirst(==(:r), model.var)
r_tilde_idx = findfirst(==(:r_tilde), model.var)

println("d_state=$d_state, d_eps=$d_eps, d_theta=$d_theta")
println("r=$r_idx, r_tilde=$r_tilde_idx")
println("X columns: state[1:$d_state] | shocks[$(d_state+1):$(d_state+d_eps)] | theta[$(d_state+d_eps+1):end]")

# ── Find worst-residual samples ──
res_vec = sep_residuals[1:cursor]
worst_idx = sortperm(res_vec, rev=true)[1:min(n_samples, length(res_vec))]

println("\n" * "="^70)
println("RE-SOLVING $(length(worst_idx)) WORST-RESIDUAL SAMPLES")
println("="^70)

# Free memory
cp_light = nothing
GC.gc()

function resolve_sample(model, sample_idx, X_ref, Y_ref, theta_ids_ref, sep_residuals,
                        theta_grid, theta_idx, base_values, state_idx, obs_idx,
                        d_state, d_eps, r_idx, r_tilde_idx, settings)
    orig_resid = sep_residuals[sample_idx]
    theta_i = theta_ids_ref[sample_idx]
    x = X_ref[:, sample_idx]
    y_ref = Y_ref[:, sample_idx]

    # Decompose X = [state, shocks, theta]
    state_vals = x[1:d_state]
    shock_vals = x[d_state+1:d_state+d_eps]
    theta_vals = x[d_state+d_eps+1:end]

    # Set model parameters
    params = copy(base_values)
    params[theta_idx] = theta_vals
    MacroModelling.write_parameters_input!(model, params, verbose = false)

    # Build full state vector from state_idx subset
    nvars = length(model.var)
    # Get steady state the same way simulate_sep_extended_path does (line 403-404 of sep_simulation.jl)
    SS_result = MacroModelling.get_steady_state(model, derivatives=false)
    yss = [Float64(SS_result(var)) for var in model.var]

    init_state = copy(yss)
    for (i, si) in enumerate(state_idx)
        init_state[si] = state_vals[i]
    end

    # Build shock matrix: 5 periods, first period = actual shocks, rest = zeros
    nshocks = length(model.exo)
    shock_mat = zeros(nshocks, 5)
    shock_mat[:, 1] = shock_vals

    # Run short SEP simulation
    res = MacroModelling.simulate_sep_extended_path(
        model;
        periods = 5,
        burn_in = 0,
        initial_state = init_state,
        shocks = shock_mat,
        sep_horizon = Int(settings["sep_horizon"]),
        sep_order = Int(settings["sep_order"]),
        sep_nnodes = Int(settings["sep_nnodes"]),
        sep_maxit = Int(settings["sep_maxit"]),
        sep_tol = Float64(settings["sep_tol"]),
        sep_sparse_tree = Bool(settings["sep_sparse_tree"]),
        sep_linear_solver = Symbol(settings["sep_linear_solver"]),
        sep_fallback_solver = nothing,
        sep_stall_iters = Int(settings["sep_stall_iters"]),
        sep_stall_rel_tol = Float64(settings["sep_stall_rel_tol"]),
        sep_stall_abs_tol = Float64(settings["sep_stall_abs_tol"]),
        sep_line_search = Bool(settings["sep_line_search"]),
        sep_line_search_maxit = Int(settings["sep_line_search_maxit"]),
        sep_line_search_factor = Float64(settings["sep_line_search_factor"]),
        sep_line_search_min_alpha = Float64(settings["sep_line_search_min_alpha"]),
        sep_lm_lambda = Float64(settings["sep_lm_lambda"]),
        sep_lm_lambda_scale = Float64(settings["sep_lm_lambda_scale"]),
        sep_lm_lambda_min = Float64(settings["sep_lm_lambda_min"]),
        sep_lm_lambda_max = Float64(settings["sep_lm_lambda_max"]),
        sep_shock_scale = Float64(settings["sep_shock_scale"]),
        sep_accept_tol = Float64(settings["sep_accept_tol"]),
        shock_scaling = :none,  # shocks already scaled in X
        silent = true,
    )

    sim = Array(res.simulation)
    new_sep_err = (hasproperty(res, :sep_errors) && res.sep_errors !== nothing && length(res.sep_errors) >= 1) ? res.sep_errors[1] : NaN

    # Extract Y_new for period 1 (same layout as checkpoint)
    y_new = vcat(sim[obs_idx, 2], sim[state_idx, 2])
    y_diff = maximum(abs.(y_new .- y_ref))

    # ZLB check
    r_val = r_idx !== nothing ? sim[r_idx, 2] : NaN
    rt_val = r_tilde_idx !== nothing ? sim[r_tilde_idx, 2] : NaN
    zlb_binding = rt_val < 1.0

    return (;
        sample_idx, theta_i, orig_resid, new_sep_err,
        y_diff, r_val, rt_val, zlb_binding,
        errorflag = res.errorflag
    )
end

# Run all samples
results = []
for (vi, si) in enumerate(worst_idx)
    print("Sample $si ($(vi)/$(length(worst_idx))): theta=$(theta_ids_ref[si]), orig_resid=$(round(sep_residuals[si], sigdigits=4))... ")
    flush(stdout)

    try
        r = resolve_sample(model, si, X_ref, Y_ref, theta_ids_ref, sep_residuals,
                          theta_grid, theta_idx, base_values, state_idx, obs_idx,
                          d_state, d_eps, r_idx, r_tilde_idx, settings)
        push!(results, r)
        println("new_resid=$(round(r.new_sep_err, sigdigits=4)), |Y_diff|=$(round(r.y_diff, sigdigits=4)), " *
                "r=$(round(r.r_val, digits=5)), r̃=$(round(r.rt_val, digits=5)), ZLB=$(r.zlb_binding)")
    catch e
        println("ERROR: $e")
        if !(e isa AssertionError || e isa TypeError)
            showerror(stdout, e, catch_backtrace())
            println()
        end
    end
    flush(stdout)
end

# ── Summary ──
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println()
println("  Sample  | Theta | Orig Resid | New Resid  | |Y diff|   | r       | r̃       | ZLB")
println("  " * "-"^90)
for r in results
    zlb_str = r.zlb_binding ? "YES" : "no"
    println("  $(lpad(r.sample_idx, 6)) | $(lpad(r.theta_i, 5)) | " *
            "$(lpad(round(r.orig_resid, sigdigits=4), 10)) | " *
            "$(lpad(round(r.new_sep_err, sigdigits=4), 10)) | " *
            "$(lpad(round(r.y_diff, sigdigits=4), 10)) | " *
            "$(lpad(round(r.r_val, digits=5), 7)) | " *
            "$(lpad(round(r.rt_val, digits=5), 7)) | $zlb_str")
end

improved = count(r -> r.new_sep_err < r.orig_resid * 0.5, results)
worse = count(r -> r.new_sep_err > r.orig_resid * 2.0, results)
zlb_count = count(r -> r.zlb_binding, results)
large_ydiff = count(r -> r.y_diff > 1.0, results)

println()
println("Residual improved (>2x better): $improved / $(length(results))")
println("Residual worsened (>2x worse):  $worse / $(length(results))")
println("ZLB binding in re-solve:        $zlb_count / $(length(results))")
println("Large Y difference (>1.0):      $large_ydiff / $(length(results))")
println()

if large_ydiff > length(results) ÷ 2
    println("⚠️  CONCERN: Majority of high-residual samples produce different Y values.")
    println("   The smooth OBC solutions at these states are sensitive to solver path.")
else
    println("✓ Most high-residual samples produce stable Y values under re-solve.")
end
println("="^70)
