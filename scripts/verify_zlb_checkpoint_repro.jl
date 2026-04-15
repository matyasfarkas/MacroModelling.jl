#!/usr/bin/env julia
#
# Verify ZLB checkpoint reproducibility — lean version
# Re-runs 1 theta and compares X/Y with checkpoint, plus ZLB analysis.
#
# Usage:
#   julia --project=. scripts/verify_zlb_checkpoint_repro.jl [--theta=27]

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

target_theta = parse_kv_int(ARGS, "--theta", 27)

# ── Load checkpoint ──
checkpoint_path = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls")
println("Loading checkpoint...")
cp = deserialize(checkpoint_path)
settings = cp["settings"]
theta_grid = cp["theta_grid"]
theta_seeds = cp["theta_seeds"]
X_ref = cp["X"]
Y_ref = cp["Y"]
theta_ids_ref = cp["theta_ids"]
cursor = cp["cursor"]
println("Checkpoint: $(cp["theta_last"])/$(length(theta_grid)) thetas, cursor=$cursor")

# ── Load model ──
model_name = String(settings["model_name"])
println("Loading model: $model_name")
model = load_hlt_model(REPO_ROOT, model_name; mod = @__MODULE__)
MacroModelling.solve!(model, silent = true)

# ── Parameter setup ──
param_set = Symbol(settings["param_set"])
theta_names = get_parameter_names(param_set)
theta_idx = Int.(indexin(theta_names, model.parameters))
base_values = copy(model.parameter_values)

state_idx = sort(unique(vcat(model.timings.past_not_future_and_mixed_idx,
                             model.timings.future_not_past_and_mixed_idx)))
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = Int.(indexin(observables, model.var))

r_idx = findfirst(==(:r), model.var)
r_tilde_idx = findfirst(==(:r_tilde), model.var)
println("r=$r_idx, r_tilde=$r_tilde_idx")

# ── Settings ──
T_obs = Int(settings["sample_length"])
burn_in = Int(settings["burn_in"])
shock_scale = Float64(settings["shock_scale"])
shock_scaling = Symbol(settings["shock_scaling"])
seed0 = Int(settings["seed0"])

function draw_shocks(rng, model, total_periods, shock_scaling, shock_scale)
    shock_names = model.exo
    nshocks = length(shock_names)
    shocks = zeros(nshocks, total_periods)
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    for t in 1:total_periods, i in structural_idx
        shocks[i, t] = shock_scale * randn(rng)
    end
    return shocks
end

seed_fn(s0, ti, att, stride) = s0 + ti - 1 + (att - 1) * stride

# ── Free checkpoint memory we don't need ──
Y_rom1 = nothing; Y_rom2 = nothing
cp_keys_to_keep = Set(["X", "Y", "theta_ids", "settings", "theta_grid", "theta_seeds", "cursor"])
for k in collect(keys(cp))
    k in cp_keys_to_keep || delete!(cp, k)
end
GC.gc()

# ── Run SEP for target theta ──
theta_i = target_theta
theta = theta_grid[theta_i]
seed = theta_seeds[theta_i]
println("\n" * "="^60)
println("Re-running theta $theta_i (seed=$seed)")
println("="^60)

params = copy(base_values)
params[theta_idx] = theta
MacroModelling.write_parameters_input!(model, params, verbose = false)

total_periods = T_obs + burn_in
shocks_override = shock_scale == 1.0 ? nothing :
    draw_shocks(MersenneTwister(seed), model, total_periods, shock_scaling, shock_scale)

flush(stdout)
t0 = time()
res = MacroModelling.simulate_sep_extended_path(
    model;
    periods = T_obs,
    burn_in = burn_in,
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
    shock_scaling = shock_scaling,
    shocks = shocks_override,
    random_seed = seed,
    silent = true,
)
elapsed = time() - t0

sim = Array(res.simulation)
shocks_out = res.shocks
T_available = min(max(size(sim, 2) - 1, 0), ndims(shocks_out) == 2 ? size(shocks_out, 2) : 0)

println("SEP completed in $(round(elapsed/60, digits=1)) min, errorflag=$(res.errorflag), T_available=$T_available")
flush(stdout)

# ── ZLB analysis ──
if r_idx !== nothing && r_tilde_idx !== nothing
    r_vals = sim[r_idx, 2:T_available+1]
    rt_vals = sim[r_tilde_idx, 2:T_available+1]
    zlb_binding = rt_vals .< 1.0
    n_binding = sum(zlb_binding)
    pct = round(100 * n_binding / T_available, digits=1)

    println("\nZLB ANALYSIS:")
    println("  r range:       [$(round(minimum(r_vals), digits=6)), $(round(maximum(r_vals), digits=6))]")
    println("  r_tilde range: [$(round(minimum(rt_vals), digits=6)), $(round(maximum(rt_vals), digits=6))]")
    println("  ZLB binding:   $n_binding / $T_available ($pct%)")
    if n_binding > 0
        gap = r_vals .- rt_vals
        println("  r at ZLB min:     $(round(minimum(r_vals[zlb_binding]), digits=6))")
        println("  r_tilde at ZLB min: $(round(minimum(rt_vals[zlb_binding]), digits=6))")
        println("  Softplus gap at ZLB max: $(round(maximum(gap[zlb_binding]), digits=6))")
    end
end
flush(stdout)

# ── Reproducibility check ──
function check_reproducibility(sim, shocks_out, T_available, theta, state_idx, obs_idx,
                               theta_ids_ref, cursor, theta_i, X_ref, Y_ref, model)
    println("\nREPRODUCIBILITY CHECK:")
    ref_mask = theta_ids_ref[1:cursor] .== theta_i
    n_ref = sum(ref_mask)
    ref_cols = findall(ref_mask)
    println("  Samples in checkpoint for theta $theta_i: $n_ref")
    n_ref == 0 && return

    max_dx = 0.0
    max_dy = 0.0
    worst_t_x = 0
    worst_t_y = 0

    for (ci, col) in enumerate(ref_cols)
        t = ci
        t > T_available && break

        x_new = vcat(sim[state_idx, t], shocks_out[:, t], theta)
        y_new = vcat(sim[obs_idx, t + 1], sim[state_idx, t + 1])
        x_ref = X_ref[:, col]
        y_ref = Y_ref[:, col]

        dx = maximum(abs.(x_new .- x_ref))
        dy = maximum(abs.(y_new .- y_ref))

        if dx > max_dx
            max_dx = dx
            worst_t_x = t
        end
        if dy > max_dy
            max_dy = dy
            worst_t_y = t
        end
    end

    println("  Max |X_new - X_ref|: $max_dx (period $worst_t_x)")
    println("  Max |Y_new - Y_ref|: $max_dy (period $worst_t_y)")

    if max_dx > 1e-10 || max_dy > 1e-10
        println("\n  ⚠️  MISMATCH — current code produces DIFFERENT results!")
        println("  This means the OBC treatment has effectively changed.")

        # Show which dimensions differ most
        t = worst_t_y
        if t > 0 && t <= T_available
            col = ref_cols[t]
            y_new = vcat(sim[obs_idx, t + 1], sim[state_idx, t + 1])
            y_ref = Y_ref[:, col]
            diffs = abs.(y_new .- y_ref)
            observables_sym = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
            all_names = vcat(String.(observables_sym), String.(model.var[state_idx]))
            top5 = sortperm(diffs, rev=true)[1:min(5, length(diffs))]
            println("  Top-5 Y dimension diffs at period $t:")
            for i in top5
                println("    $(all_names[i]): |$(round(y_new[i], digits=8)) - $(round(y_ref[i], digits=8))| = $(round(diffs[i], sigdigits=3))")
            end
        end
    else
        println("\n  ✓ EXACT MATCH — current code reproduces checkpoint data.")
    end
end

check_reproducibility(sim, shocks_out, T_available, theta, state_idx, obs_idx,
                      theta_ids_ref, cursor, theta_i, X_ref, Y_ref, model)

println("\n" * "="^60)
println("DONE")
println("="^60)
flush(stdout)
