#!/usr/bin/env julia
#
# Verify ZLB dataset checkpoint reproducibility
#
# Picks N completed thetas from the checkpoint, re-runs them with the
# current code, and compares:
#   1. X/Y matrix outputs (bit-exact reproducibility)
#   2. Whether the softplus ZLB actually binds (r ≈ 1.0, r_tilde < 1.0)
#   3. SEP residual magnitudes
#
# Usage:
#   julia --project=. scripts/verify_zlb_checkpoint_thetas.jl [--n-thetas=3]

using Serialization, Random, Statistics, LinearAlgebra, Dates

# ── project root ──
const REPO_ROOT = dirname(@__DIR__)
cd(REPO_ROOT)

# ── MacroModelling ──
using MacroModelling

# ── helpers from dataset gen ──
include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(REPO_ROOT, "scripts", "hlt_surrogate", "parameter_config.jl"))

# ── CLI ──
function parse_kv_int(args, key, default)
    for a in args
        startswith(a, "$key=") && return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

n_verify = parse_kv_int(ARGS, "--n-thetas", 3)

# ── Load checkpoint ──
checkpoint_path = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls")
println("Loading checkpoint: $checkpoint_path")
cp = deserialize(checkpoint_path)

settings = cp["settings"]
theta_last = cp["theta_last"]
cursor = cp["cursor"]
theta_grid = cp["theta_grid"]
theta_seeds = cp["theta_seeds"]
X_ref = cp["X"]
Y_ref = cp["Y"]
theta_ids_ref = cp["theta_ids"]
sep_residuals_ref = cp["sep_residuals"]
theta_full_success = cp["theta_full_success"]

println("Checkpoint: $theta_last / $(length(theta_grid)) thetas, cursor=$cursor")
println("Model: $(settings["model_name"])")
println()

# ── Load model ──
model_name = String(settings["model_name"])
model = load_hlt_model(REPO_ROOT, model_name; mod = @__MODULE__)
MacroModelling.solve!(model, silent = true)

# ── Parameter setup ──
param_set = Symbol(settings["param_set"])
theta_names = get_parameter_names(param_set)
hlt_param_names = model.parameters
theta_idx = Int.(indexin(theta_names, hlt_param_names))
base_values = copy(model.parameter_values)

# ── Indices (must match dataset gen) ──
state_idx = sort(unique(vcat(model.timings.past_not_future_and_mixed_idx,
                             model.timings.future_not_past_and_mixed_idx)))
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = Int.(indexin(observables, model.var))

# Find r and r_tilde indices for ZLB analysis
r_idx = findfirst(==(:r), model.var)
r_tilde_idx = findfirst(==(:r_tilde), model.var)
robs_idx = findfirst(==(:robs), model.var)
robs_tilde_idx = findfirst(==(:robs_tilde), model.var)
println("Variable indices: r=$r_idx, r_tilde=$r_tilde_idx, robs=$robs_idx, robs_tilde=$robs_tilde_idx")

# ── SEP settings from checkpoint ──
T_obs = Int(settings["sample_length"])
burn_in = Int(settings["burn_in"])
sep_horizon = Int(settings["sep_horizon"])
sep_order = Int(settings["sep_order"])
sep_nnodes = Int(settings["sep_nnodes"])
sep_maxit = Int(settings["sep_maxit"])
sep_tol = Float64(settings["sep_tol"])
sep_sparse_tree = Bool(settings["sep_sparse_tree"])
sep_linear_solver = Symbol(settings["sep_linear_solver"])
sep_fallback_solver = haskey(settings, "sep_fallback_solver") && settings["sep_fallback_solver"] !== nothing ? Symbol(settings["sep_fallback_solver"]) : nothing
sep_stall_iters = Int(settings["sep_stall_iters"])
sep_stall_rel_tol = Float64(settings["sep_stall_rel_tol"])
sep_stall_abs_tol = Float64(settings["sep_stall_abs_tol"])
sep_line_search = Bool(settings["sep_line_search"])
sep_line_search_maxit = Int(settings["sep_line_search_maxit"])
sep_line_search_factor = Float64(settings["sep_line_search_factor"])
sep_line_search_min_alpha = Float64(settings["sep_line_search_min_alpha"])
sep_lm_lambda = Float64(settings["sep_lm_lambda"])
sep_lm_lambda_scale = Float64(settings["sep_lm_lambda_scale"])
sep_lm_lambda_min = Float64(settings["sep_lm_lambda_min"])
sep_lm_lambda_max = Float64(settings["sep_lm_lambda_max"])
sep_shock_scale = Float64(settings["sep_shock_scale"])
sep_accept_tol = Float64(settings["sep_accept_tol"])
shock_scaling = Symbol(settings["shock_scaling"])
shock_scale = Float64(settings["shock_scale"])
seed0 = Int(settings["seed0"])

# ── draw_shocks (copied from dataset gen) ──
function draw_shocks(rng::AbstractRNG, model, total_periods::Int, shock_scaling::Symbol, shock_scale::Float64)
    shock_names = model.exo
    nshocks = length(shock_names)
    shocks = zeros(nshocks, total_periods)
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    isempty(structural_idx) && return shocks
    for t in 1:total_periods
        for i in structural_idx
            shocks[i, t] = shock_scale * randn(rng)
        end
    end
    return shocks
end

function attempt_seed(seed0::Int, theta_i::Int, attempt::Int, seed_attempt_stride::Int)
    return seed0 + theta_i - 1 + (attempt - 1) * seed_attempt_stride
end

# ── Pick thetas to verify ──
# Choose from successful thetas spread across the range
successful = findall(theta_full_success[1:theta_last])
step = max(1, length(successful) ÷ (n_verify + 1))
verify_indices = successful[step:step:min(step*n_verify, length(successful))]
if length(verify_indices) < n_verify
    verify_indices = successful[1:min(n_verify, length(successful))]
end

println("\n" * "="^70)
println("VERIFICATION: Re-running $(length(verify_indices)) thetas with current code")
println("="^70)

total_zlb_periods = 0
total_periods_checked = 0
all_match = true

for (vi, theta_i) in enumerate(verify_indices)
    println("\n--- Theta $theta_i ($(vi)/$(length(verify_indices))) ---")

    theta = theta_grid[theta_i]
    seed = theta_seeds[theta_i]

    # Set parameters
    params = copy(base_values)
    params[theta_idx] = theta
    MacroModelling.write_parameters_input!(model, params, verbose = false)

    # Draw shocks exactly as the dataset gen does
    total_periods = T_obs + burn_in
    shocks_override = shock_scale == 1.0 ? nothing :
        draw_shocks(MersenneTwister(seed), model, total_periods, shock_scaling, shock_scale)

    # Run SEP
    t0 = time()
    res = MacroModelling.simulate_sep_extended_path(
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
        sep_line_search = sep_line_search,
        sep_line_search_maxit = sep_line_search_maxit,
        sep_line_search_factor = sep_line_search_factor,
        sep_line_search_min_alpha = sep_line_search_min_alpha,
        sep_lm_lambda = sep_lm_lambda,
        sep_lm_lambda_scale = sep_lm_lambda_scale,
        sep_lm_lambda_min = sep_lm_lambda_min,
        sep_lm_lambda_max = sep_lm_lambda_max,
        sep_shock_scale = sep_shock_scale,
        sep_accept_tol = sep_accept_tol,
        shock_scaling = shock_scaling,
        shocks = shocks_override,
        random_seed = seed,
        silent = true,
    )
    elapsed = time() - t0

    sim = Array(res.simulation)
    shocks_out = res.shocks
    T_available = min(max(size(sim, 2) - 1, 0), ndims(shocks_out) == 2 ? size(shocks_out, 2) : 0)

    println("  SEP completed in $(round(elapsed, digits=1))s, errorflag=$(res.errorflag), T_available=$T_available")

    # ── ZLB analysis ──
    if r_idx !== nothing && r_tilde_idx !== nothing
        r_vals = sim[r_idx, 2:T_available+1]     # skip initial condition
        rt_vals = sim[r_tilde_idx, 2:T_available+1]

        # ZLB binds when r_tilde < 1.0 (shadow rate below zero)
        zlb_binding = rt_vals .< 1.0
        n_binding = sum(zlb_binding)
        pct_binding = round(100 * n_binding / T_available, digits=1)

        # Softplus gap: how much does r deviate from r_tilde?
        gap = r_vals .- rt_vals

        println("  ZLB analysis:")
        println("    r range:       [$(round(minimum(r_vals), digits=6)), $(round(maximum(r_vals), digits=6))]")
        println("    r_tilde range: [$(round(minimum(rt_vals), digits=6)), $(round(maximum(rt_vals), digits=6))]")
        println("    ZLB binding periods: $n_binding / $T_available ($pct_binding%)")
        if n_binding > 0
            println("    r at ZLB:      min=$(round(minimum(r_vals[zlb_binding]), digits=6))")
            println("    r_tilde at ZLB: min=$(round(minimum(rt_vals[zlb_binding]), digits=6))")
            println("    Softplus gap at ZLB: max=$(round(maximum(gap[zlb_binding]), digits=6))")
        end
        println("    Softplus gap (all periods): mean=$(round(mean(gap), digits=6)), max=$(round(maximum(gap), digits=6))")

        total_zlb_periods += n_binding
        total_periods_checked += T_available
    end

    # ── Compare with checkpoint data ──
    # Find samples in checkpoint belonging to this theta
    ref_mask = theta_ids_ref[1:cursor] .== theta_i
    n_ref = sum(ref_mask)
    ref_cols = findall(ref_mask)

    if n_ref > 0
        println("  Reproducibility check ($n_ref samples in checkpoint):")

        # Reconstruct X/Y for the same periods
        max_diffs_X = Float64[]
        max_diffs_Y = Float64[]

        for (ci, col) in enumerate(ref_cols)
            # The checkpoint stores X = [state(t), shocks(t), theta]
            # and Y = [obs(t+1), state(t+1)]
            # We need to figure out which period t this corresponds to
            # Since samples_per_theta == T_obs for this run, t = ci
            t = ci
            if t > T_available
                println("    WARNING: sample $ci maps to period $t > T_available=$T_available")
                continue
            end

            x_new = vcat(sim[state_idx, t], shocks_out[:, t], theta)
            y_new = vcat(sim[obs_idx, t + 1], sim[state_idx, t + 1])
            x_ref = X_ref[:, col]
            y_ref = Y_ref[:, col]

            dx = maximum(abs.(x_new .- x_ref))
            dy = maximum(abs.(y_new .- y_ref))
            push!(max_diffs_X, dx)
            push!(max_diffs_Y, dy)
        end

        if !isempty(max_diffs_X)
            mx = maximum(max_diffs_X)
            my = maximum(max_diffs_Y)
            println("    Max |X_new - X_ref|: $(mx)")
            println("    Max |Y_new - Y_ref|: $(my)")

            if mx > 1e-10 || my > 1e-10
                println("    ⚠️  MISMATCH DETECTED — results differ from checkpoint!")
                all_match = false

                # Find which samples differ most
                worst_x = argmax(max_diffs_X)
                worst_y = argmax(max_diffs_Y)
                println("    Worst X diff at sample $worst_x: $(max_diffs_X[worst_x])")
                println("    Worst Y diff at sample $worst_y: $(max_diffs_Y[worst_y])")
            else
                println("    ✓ EXACT MATCH (within floating-point tolerance)")
            end
        end
    else
        println("  No reference samples found for theta $theta_i in checkpoint")
    end

    # ── SEP residuals ──
    if hasproperty(res, :sep_errors) && res.sep_errors !== nothing
        errs = res.sep_errors
        println("  SEP residuals: mean=$(round(mean(errs), digits=6)), max=$(round(maximum(errs), digits=6))")
        n_high = sum(errs .> 0.1)
        if n_high > 0
            println("    Periods with residual > 0.1: $n_high / $(length(errs))")
        end
    end
end

# ── Summary ──
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("Thetas verified: $(length(verify_indices))")
println("Total ZLB-binding periods: $total_zlb_periods / $total_periods_checked ($(round(100*total_zlb_periods/max(1,total_periods_checked), digits=1))%)")
println("Reproducibility: $(all_match ? "✓ ALL MATCH" : "⚠️  MISMATCHES DETECTED")")
println()
if total_zlb_periods == 0
    println("⚠️  WARNING: ZLB never binds in any verified theta!")
    println("   This suggests shock_scale=0.4 may still be too small for the softplus ZLB")
    println("   to activate. Consider: (a) increasing shock_scale, or (b) checking kappa_zlb.")
else
    println("✓ ZLB binding confirmed — smoothed OBC treatment is active.")
end
println("="^70)
