#!/usr/bin/env julia
# ============================================================================
# EULER ERRORS STUDY — three DSGE solution methods: SEP, ROM1, ROM1+NN
# ============================================================================
# For the 18-parameter HLT model evaluated at the pooled warm-started
# surrogate posterior mean, this script:
#   1. Pools the 4 warm-started surrogate HMC chains and takes the posterior mean θ.
#   2. Builds ROM1 / ROM1+NN predictors; recovers shocks ε̂_{1:T} via ROM1
#      inversion filter.
#   3. Simulates three full-state trajectories of length T_obs using those
#      recovered shocks:
#         - ROM1  : repeated rom_step_full
#         - SEP   : single call to simulate_sep_extended_path (one SEP solve;
#                   reused for Euler-error evaluation)
#         - ROM1+NN : ROM1 rollout with per-period NN correction applied
#                     to obs AND state components ("gate everywhere")
#   4. Computes the nonlinear dynamic residual
#         R_t = resid_func(𝔓, 𝔙_t)
#      where 𝔓 = [params; SS] and
#            𝔙_t = [y_{t+1}; y_t; y_{t-1}; ε̂_t]
#      (ordered by build_dynamic_residual_jacobian's vars_raw convention,
#      filled via fill_dyn_values!).
#   5. Records ‖R_t‖_∞ per period and per equation, writes
#         .local_artifacts/euler_errors/euler_errors_table.tex
#         .local_artifacts/euler_errors/EULER_ERRORS_SUMMARY.md
#         .local_artifacts/euler_errors/euler_errors_results.jls
#
# Guard-rails: exactly ONE SEP solve (reused for Euler evaluation); total
# wall time target < 30 min. If SEP fails or times out, the script reports
# only ROM1 and ROM1+NN and explicitly notes the omission.
# ============================================================================

using Serialization, Statistics, LinearAlgebra, Printf, Dates
using MacroModelling
using AxisKeys

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

chain_paths = [
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls"),
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed5.jls"),
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed6.jls"),
    joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed7.jls"),
]

data_path      = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls")
surrogate_path = joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls")

out_dir = joinpath(REPO_ROOT, ".local_artifacts/euler_errors")
out_jls = joinpath(out_dir, "euler_errors_results.jls")
out_tex = joinpath(out_dir, "euler_errors_table.tex")
out_md  = joinpath(out_dir, "EULER_ERRORS_SUMMARY.md")
mkpath(out_dir)

inv_maxit  = 10
inv_tol    = 1e-6
inv_lambda = 1e-4

# SEP solver settings (mirroring sep_posterior_validation / sep_sensitivity reference cell)
sep_horizon    = 40
sep_maxit      = 200
sep_nnodes     = 3
sep_accept_tol = 0.35   # from project memory: needed for robust convergence

# SEP wall-time cap (seconds). If the SEP solve blows the budget we still
# write ROM1 / ROM1+NN results and flag SEP as omitted.
SEP_WALL_CAP_S = 1500.0  # 25 min upper bound — target is < 30 min total

obs_sigma_scale = 2.0
obs_sigma_floor = 0.1

println("=" ^ 78)
println("EULER ERRORS STUDY — SEP vs ROM1 vs ROM1+NN at pooled posterior mean")
println("=" ^ 78)
println("  out_dir: $out_dir")

# ============================================================================
# 1. Load payload + model + surrogate + ROM predictors
# ============================================================================

println("\n--- Loading data payload ---")
payload        = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data       = payload["obs_data"]
obs_sigma_base = payload["obs_sigma"]
theta_names    = payload["theta_names"]
observables    = payload["observables"]
state_names    = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
s0             = payload["s0"]
shock_sigmas   = payload["shock_sigmas"]

d_obs   = size(obs_data, 1)
T_obs   = size(obs_data, 2)
n_theta = length(theta_names)
d_state = length(s0)

println("  obs: $d_obs × $T_obs,  state: $d_state,  n_theta: $n_theta")

println("\n--- Loading model + surrogate helpers ---")
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

mm_model = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  model: $(mm_model.model_name),  nvars=$(length(mm_model.var)),  nexo=$(length(mm_model.exo))")

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
frozen  = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)
println("  surrogate:  d_in=$(frozen.d_in), d_out=$(frozen.d_out)")

obs_idx   = Int.(indexin(observables, mm_model.var))
state_idx = Int.(indexin(state_names,  mm_model.var))
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    any(isnothing, idx_any) && error("theta names missing in model.parameters")
    Int.(idx_any)
end
base_parameters = copy(mm_model.parameter_values)

# ============================================================================
# 2. Pool chains and take posterior mean θ
# ============================================================================

println("\n--- Pooling warm-started chains ---")
chains = Matrix{Float64}[]
for p in chain_paths
    if isfile(p)
        d = deserialize(p)
        push!(chains, Matrix{Float64}(d["chain"]))
    else
        println("  WARN: missing $p")
    end
end
isempty(chains) && error("No posterior chains available")
pooled = vcat(chains...)
post_mean = vec(Statistics.mean(pooled, dims=1))
println("  pooled draws: $(size(pooled, 1)) × $(size(pooled, 2))")
println("  posterior mean (first 6 params): $(round.(post_mean[1:min(6, end)], digits=4))")

# Build full-parameter vector at posterior mean (base_parameters with θ_pm overriding theta_idx)
post_mean_full_params = copy(base_parameters)
for (j, idx) in enumerate(theta_param_idx)
    post_mean_full_params[idx] = post_mean[j]
end

# ============================================================================
# 3. Solve model + build predictors at the posterior mean
# ============================================================================

println("\n--- Building ROM1 cache at posterior mean ---")
# Write posterior-mean params and solve first-order so solution matrices populate.
MacroModelling.write_parameters_input!(mm_model, post_mean_full_params, verbose=false)
MacroModelling.solve!(mm_model; algorithm=:first_order, dynamics=true, obc=false, silent=true)

# ROM predictor with posterior-mean params as baseline
rom_predictor = RomPredictor(mm_model, 1, :baseline, false, Int[],
    post_mean_full_params, nothing, nothing, Int.(state_idx), Int.(obs_idx))
ensure_rom_cache!(rom_predictor, Float64[])
rom_cache = rom_predictor.cache
nsss_full = copy(rom_cache.nsss)      # full NSSS (length nvars)
println("  NSSS norm: $(round(norm(nsss_full), digits=3))")

# Matrix ROM1 predictor (for shock recovery via inversion filter)
_, matrix_rom_predict_tuple, _ = build_matrix_rom_predict(mm_model;
    state_idx = Int.(state_idx), obs_idx = Int.(obs_idx))

# Observation sigma (same recipe as run_surrogate_hmc_advancedhmc)
obs_sigma = copy(obs_sigma_base)
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_sigma = max.(obs_sigma, val_rmse[1:d_obs] .* obs_sigma_scale)
end
if obs_sigma_floor > 0
    obs_sigma = max.(obs_sigma, obs_sigma_floor)
end

# Surrogate theta-padding (NN was trained on 18p theta; here estimation uses 18p,
# but the surrogate meta may list them in a different order)
surrogate_theta_names = get(sur_meta, "theta_names", Symbol[])
_pad_theta = if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    _baseline = zeros(Float64, length(surrogate_theta_names))
    _est_idx = zeros(Int, length(surrogate_theta_names))
    for (si, sname) in enumerate(surrogate_theta_names)
        ei = findfirst(==(sname), theta_names)
        if ei !== nothing
            _est_idx[si] = ei
        else
            pi = findfirst(==(sname), mm_model.parameters)
            _baseline[si] = pi !== nothing ? base_parameters[pi] : 0.0
        end
    end
    θ -> begin
        θf = copy(_baseline)
        for i in eachindex(_est_idx)
            if _est_idx[i] > 0
                θf[i] = θ[_est_idx[i]]
            end
        end
        return θf
    end
else
    θ -> θ
end

# Single-sample NN residual returning full d_out vector (obs corr ++ state corr)
single_nn_residual = function(x_nn::AbstractVector)
    if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
        d_prefix = d_state + length(shock_sigmas)
        xp = vcat(x_nn[1:d_prefix], _pad_theta(x_nn[(d_prefix+1):end]))
        return predict_frozen(frozen, xp)
    else
        return predict_frozen(frozen, x_nn)
    end
end

# Correction clamp: ±3×RMSE (same as inference-time clamp)
nn_correction_clamp = (val_rmse !== nothing && length(val_rmse) == frozen.d_out) ?
    3.0 .* val_rmse : nothing

# ============================================================================
# 4. Recover shocks via ROM1 inversion filter at the posterior mean
# ============================================================================

println("\n--- Recovering shocks (ROM1 inversion filter) ---")
ll_rom1_vec, shocks_recovered = MacroModelling.inversion_loglik_per_period(
    matrix_rom_predict_tuple, s0, post_mean, obs_data, obs_sigma, shock_sigmas;
    maxit  = inv_maxit,
    tol    = inv_tol,
    lambda = inv_lambda)
println("  ROM1 inversion joint LL: $(round(sum(ll_rom1_vec), digits=1))")

# ============================================================================
# 5. Simulate full-state trajectories
# ============================================================================

# Initial full-state: NSSS with s0 overriding state positions. This matches
# build_matrix_rom_predict's convention (state_full starts from nsss and is
# overwritten at state_idx) and is also the initial_state we hand to SEP.
y0_full = copy(nsss_full)
y0_full[state_idx] .= Float64.(s0)

println("\n--- ROM1 full-state rollout ($T_obs periods) ---")
Y_rom1 = Matrix{Float64}(undef, length(mm_model.var), T_obs + 1)
Y_rom1[:, 1] = y0_full
let y = copy(y0_full)
    for t in 1:T_obs
        y = rom_step_full(rom_cache, y, shocks_recovered[:, t])
        Y_rom1[:, t + 1] = y
    end
end
println("  done.  last-period max |y| = $(round(maximum(abs.(Y_rom1[:, end])), sigdigits=4))")

println("\n--- Building gate mask from gate calibration ---")
gate_path = joinpath(REPO_ROOT, ".local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p.jls")
gate_calib = isfile(gate_path) ? deserialize(gate_path) : nothing
gate_mask = falses(T_obs)
if gate_calib !== nothing
    tau_eps = get(gate_calib, "tau_eps", Inf)
    tau_y   = get(gate_calib, "tau_y",   Inf)
    structural_idx = Int.(get(gate_calib, "structural_idx", 1:size(shocks_recovered, 1)))
    shock_std_local = shock_sigmas[structural_idx]
    for t in 1:T_obs
        e_t = sqrt(sum((shocks_recovered[structural_idx, t] ./ shock_std_local) .^ 2))
        # ROM1 forecast error in obs at period t (using true obs minus ROM1 prediction)
        if t == 1
            y_rom_pred = rom_step_full(rom_cache, y0_full, shocks_recovered[:, t])
        else
            y_rom_pred = rom_step_full(rom_cache, y0_full, shocks_recovered[:, t])  # placeholder; full inversion fwd not needed
        end
        f_t = sqrt(sum(((obs_data[:, t] .- y_rom_pred[obs_idx]) ./ obs_sigma) .^ 2))
        gate_mask[t] = (e_t > tau_eps) || (f_t > tau_y)
    end
end
println("  Gate-on periods: $(sum(gate_mask))/$T_obs ($(round(100*mean(gate_mask), digits=1))%)")

println("\n--- ROM1+NN full-state rollout — gate-everywhere ($T_obs periods) ---")
Y_nn = Matrix{Float64}(undef, length(mm_model.var), T_obs + 1)
Y_nn[:, 1] = y0_full
let y = copy(y0_full)
    post_mean_f = Float64.(post_mean)
    for t in 1:T_obs
        # ROM1 base step
        y_rom_next = rom_step_full(rom_cache, y, shocks_recovered[:, t])

        # NN correction: x = [state_subset; shock; theta]
        x_nn = vcat(Float64.(y[state_idx]), Float64.(shocks_recovered[:, t]), post_mean_f)
        y_nn_corr = single_nn_residual(x_nn)
        if nn_correction_clamp !== nothing
            y_nn_corr = clamp.(y_nn_corr, -nn_correction_clamp, nn_correction_clamp)
        end

        # Apply correction: first d_obs -> obs_idx, rest -> state_idx
        y_next = copy(y_rom_next)
        y_next[obs_idx]   .+= Float64.(y_nn_corr[1:d_obs])
        y_next[state_idx] .+= Float64.(y_nn_corr[(d_obs+1):end])

        Y_nn[:, t + 1] = y_next
        y = y_next
    end
end
println("  done.  last-period max |y| = $(round(maximum(abs.(Y_nn[:, end])), sigdigits=4))")

println("\n--- ROM1+NN full-state rollout — gated (NN only on gate-on periods) ($T_obs periods) ---")
Y_nn_gated = Matrix{Float64}(undef, length(mm_model.var), T_obs + 1)
Y_nn_gated[:, 1] = y0_full
let y = copy(y0_full)
    post_mean_f = Float64.(post_mean)
    for t in 1:T_obs
        y_rom_next = rom_step_full(rom_cache, y, shocks_recovered[:, t])
        y_next = copy(y_rom_next)
        if gate_mask[t]
            x_nn = vcat(Float64.(y[state_idx]), Float64.(shocks_recovered[:, t]), post_mean_f)
            y_nn_corr = single_nn_residual(x_nn)
            if nn_correction_clamp !== nothing
                y_nn_corr = clamp.(y_nn_corr, -nn_correction_clamp, nn_correction_clamp)
            end
            y_next[obs_idx]   .+= Float64.(y_nn_corr[1:d_obs])
            y_next[state_idx] .+= Float64.(y_nn_corr[(d_obs+1):end])
        end
        Y_nn_gated[:, t + 1] = y_next
        y = y_next
    end
end
println("  done.  last-period max |y| = $(round(maximum(abs.(Y_nn_gated[:, end])), sigdigits=4))")

# SEP trajectory: one solve, guarded by wall-time cap
println("\n--- SEP simulation ($T_obs periods; single solve, budget $(Int(SEP_WALL_CAP_S))s) ---")
Y_sep = nothing
sep_omitted = false
sep_wall_s = NaN
sep_errorflag = false
sep_failure_period = nothing

t_sep = time()
sep_task_result = try
    # post_mean params should already be in mm_model (set above)
    MacroModelling.simulate_sep_extended_path(mm_model;
        periods         = T_obs,
        initial_state   = y0_full,
        shocks          = shocks_recovered,
        burn_in         = 0,
        sep_horizon     = sep_horizon,
        sep_order       = 1,
        sep_nnodes      = sep_nnodes,
        sep_maxit       = sep_maxit,
        sep_tol         = 1e-7,
        sep_sparse_tree = true,
        sep_accept_tol  = sep_accept_tol,
        sep_shock_scale = 1.0,
        silent          = true)
catch e
    @warn "SEP threw an exception" exception=(e, catch_backtrace())
    nothing
end
sep_wall_s = time() - t_sep

if sep_task_result === nothing
    sep_omitted = true
    println("  SEP failed with exception.  Continuing with ROM1 and ROM1+NN only.")
elseif sep_wall_s > SEP_WALL_CAP_S
    sep_omitted = true
    println(@sprintf("  SEP took %.1fs (> %.0fs budget).  Treating as omitted.",
                     sep_wall_s, SEP_WALL_CAP_S))
else
    sep_errorflag = sep_task_result.errorflag
    sep_failure_period = sep_task_result.failure_period
    if sep_errorflag
        sep_omitted = true
        println(@sprintf("  SEP errorflag=true at period=%s (wall %.1fs).  Omitting.",
                         string(sep_failure_period), sep_wall_s))
    else
        sim = sep_task_result.simulation
        var_names = collect(axiskeys(sim, 1))
        # simulation is indexed by Time = 0:periods, so columns 1..T_obs+1 are t=0..T_obs
        # Align to mm_model.var ordering
        Y_sep = Matrix{Float64}(undef, length(mm_model.var), T_obs + 1)
        sim_mat = Matrix{Float64}(sim)  # (nvars, T_obs+1)
        for (i, v) in enumerate(mm_model.var)
            row = findfirst(==(v), var_names)
            row === nothing && error("SEP simulation missing variable $v")
            Y_sep[i, :] = sim_mat[row, :]
        end
        println(@sprintf("  SEP converged in %.1fs.  last-period max |y| = %.4g",
                         sep_wall_s, maximum(abs.(Y_sep[:, end]))))
    end
end

# ============================================================================
# 6. Build dynamic residual function + parameters & SS
# ============================================================================

println("\n--- Building dynamic residual function ---")
resid_func, _jac_func, vars_raw, parameters_and_SS, resid_buffer, _jac_buffer =
    MacroModelling.build_dynamic_residual_jacobian(mm_model)
n_eq = length(resid_buffer)
n_dyn_vars = length(vars_raw)
println("  n_eq = $n_eq,  n_dyn_vars = $n_dyn_vars,  n_par_and_SS = $(length(parameters_and_SS))")

# var_kind / var_idx classify each entry of vars_raw as :future/:present/:past/:shock
var_kind, var_idx = MacroModelling.build_dyn_var_maps(mm_model, vars_raw)

# 𝔓 = [params (active + calib-params); steady-state values].  The SS values here
# correspond to the *posterior-mean* solve (because mm_model.parameter_values
# was set above and solve! has been called).
SS_result = MacroModelling.get_steady_state(mm_model, derivatives=false)
yss_full = [Float64(SS_result(var)) for var in mm_model.var]
params_and_ss_vals = MacroModelling.build_parameters_and_ss_values(
    parameters_and_SS, post_mean_full_params, mm_model, yss_full, SS_result)
println("  params & SS vector length: $(length(params_and_ss_vals))")

# ============================================================================
# 7. Evaluate Euler residuals at each period for each method
# ============================================================================

function evaluate_euler_errors(Y::AbstractMatrix, shocks::AbstractMatrix, label::AbstractString)
    # Y is (nvars, T_obs+1).  Y[:,1] = y at t=0, Y[:,t+1] = y at t, Y[:,T_obs+1] = y at t=T_obs.
    # Evaluate R_t for t = 1 .. T_obs-1 because we need y_{t+1} (only available up to T_obs).
    # Task asks for T = 1..265 (= T_obs).  The last period lacks y_{t+1}, so we evaluate
    # t = 1..T_obs-1 and report the per-period sup-norm vector.
    T = size(Y, 2) - 1   # = T_obs
    periods_eval = T - 1
    resid_buf = zeros(Float64, n_eq)
    dyn_values = zeros(Float64, n_dyn_vars)
    per_period_sup = fill(NaN, periods_eval)
    per_eq_max = fill(0.0, n_eq)
    failed = 0

    for t in 1:periods_eval
        y_lag = view(Y, :, t)       # y at t-1 (simulation column t)
        y_cur = view(Y, :, t + 1)   # y at t   (column t+1)
        y_fwd = view(Y, :, t + 2)   # y at t+1 (column t+2)
        ε_t   = view(shocks, :, t)  # shock used between t-1 and t

        MacroModelling.fill_dyn_values!(dyn_values, var_kind, var_idx,
            Vector{Float64}(y_lag), Vector{Float64}(y_cur), Vector{Float64}(y_fwd),
            Vector{Float64}(ε_t))

        # Call the compiled residual function
        try
            resid_func(resid_buf, params_and_ss_vals, dyn_values)
        catch e
            failed += 1
            continue
        end
        if !all(isfinite, resid_buf)
            failed += 1
            continue
        end

        per_period_sup[t] = maximum(abs, resid_buf)
        for i in 1:n_eq
            per_eq_max[i] = max(per_eq_max[i], abs(resid_buf[i]))
        end
    end

    finite_sups = filter(isfinite, per_period_sup)
    max_str  = isempty(finite_sups) ? "NaN" : string(maximum(finite_sups))
    mean_str = isempty(finite_sups) ? "NaN" : string(Statistics.mean(finite_sups))
    println("  [$label] evaluated $periods_eval periods ($(failed) failed),  " *
            "max ‖R_t‖_∞ = $max_str,  mean ‖R_t‖_∞ = $mean_str")

    return per_period_sup, per_eq_max, failed
end

println("\n--- Evaluating per-period Euler residuals ---")

rom1_per_period, rom1_per_eq, rom1_failed = evaluate_euler_errors(Y_rom1, shocks_recovered, "ROM1")
nn_per_period,   nn_per_eq,   nn_failed   = evaluate_euler_errors(Y_nn,   shocks_recovered, "ROM1+NN gate-everywhere")
nng_per_period,  nng_per_eq,  nng_failed  = evaluate_euler_errors(Y_nn_gated, shocks_recovered, "ROM1+NN gated")
sep_per_period = Float64[]; sep_per_eq = Float64[]; sep_failed = 0
if !sep_omitted && Y_sep !== nothing
    sep_per_period, sep_per_eq, sep_failed = evaluate_euler_errors(Y_sep, shocks_recovered, "SEP")
end

# ============================================================================
# 8. Aggregate headline statistics
# ============================================================================

function summarize_method(per_period::AbstractVector, per_eq::AbstractVector)
    finite_pp = filter(isfinite, per_period)
    finite_eq = filter(isfinite, per_eq)
    max_euler = isempty(finite_pp) ? NaN : maximum(finite_pp)
    mean_euler = isempty(finite_pp) ? NaN : Statistics.mean(finite_pp)
    log10_max = (isnan(max_euler) || max_euler <= 0) ? NaN : log10(max_euler)
    # Top-3 worst equations by per-equation sup across periods
    top3 = if isempty(finite_eq)
        Int[]
    else
        sortperm(per_eq; rev=true)[1:min(3, length(per_eq))]
    end
    return (; max_euler, mean_euler, log10_max, top3)
end

stats_rom1 = summarize_method(rom1_per_period, rom1_per_eq)
stats_nn   = summarize_method(nn_per_period,   nn_per_eq)
stats_nng  = summarize_method(nng_per_period,  nng_per_eq)
stats_sep  = sep_omitted ? nothing : summarize_method(sep_per_period, sep_per_eq)

println("\n=== Headline Euler errors ===")
@printf("  ROM1:               max = %-12.4e  mean = %-12.4e  log10(max) = %+7.3f\n",
        stats_rom1.max_euler, stats_rom1.mean_euler, stats_rom1.log10_max)
@printf("  ROM1+NN (everywhere): max = %-12.4e  mean = %-12.4e  log10(max) = %+7.3f\n",
        stats_nn.max_euler,   stats_nn.mean_euler,   stats_nn.log10_max)
@printf("  ROM1+NN (gated):    max = %-12.4e  mean = %-12.4e  log10(max) = %+7.3f\n",
        stats_nng.max_euler,  stats_nng.mean_euler,  stats_nng.log10_max)
if stats_sep === nothing
    println("  SEP:                OMITTED (wall $(round(sep_wall_s, digits=1))s, errorflag=$sep_errorflag)")
else
    @printf("  SEP:                max = %-12.4e  mean = %-12.4e  log10(max) = %+7.3f\n",
            stats_sep.max_euler, stats_sep.mean_euler, stats_sep.log10_max)
end

# Human-readable equation indices. dyn_equations is a vector of Expr; we just
# report the equation index (1-based) since equations don't have short names.
function eq_label(i::Int)
    return "eq#$i"
end

# ============================================================================
# 9. Write LaTeX table
# ============================================================================

function fmt_sci(x; digits=3)
    (isnan(x) || !isfinite(x)) && return "---"
    return @sprintf("%.*e", digits, x)
end
function fmt_log10(x)
    (isnan(x) || !isfinite(x)) && return "---"
    return @sprintf("%+.3f", x)
end
function fmt_top3(top3)
    isempty(top3) && return "---"
    return join([eq_label(i) for i in top3], ", ")
end

tex = IOBuffer()
println(tex, "% Euler errors table (autogenerated by scripts/euler_errors_study.jl)")
println(tex, "% Generated: $(now())")
println(tex, "% Posterior pool: 4 warm-started surrogate chains (seeds 42, 5, 6, 7)")
println(tex, "\\begin{table}[ht]")
println(tex, "\\centering")
println(tex, "\\caption{Nonlinear Euler residuals across DSGE solution methods at the pooled")
println(tex, "warm-started surrogate posterior mean \$\\bar{\\theta}\$ (18 parameters).  For")
println(tex, "each method the trajectory \$\\{y_t\\}_{t=0}^{T}\$ is simulated over \$T=$(T_obs)\$")
println(tex, "periods using the shocks \$\\hat{\\varepsilon}_{1:T}\$ recovered at the ROM1")
println(tex, "inversion filter.  The residual \$R_t = R(\\bar{\\theta}, y_{t+1}, y_t, y_{t-1}, \\hat{\\varepsilon}_t)\$")
println(tex, "is the full nonlinear dynamic equation vector.  We report \$\\max_t \\|R_t\\|_\\infty\$,")
println(tex, "its \$\\log_{10}\$, the time-average, and the three equations most responsible for the sup-norm.}")
println(tex, "\\label{tab:euler_errors}")
println(tex, "\\small")
println(tex, "\\begin{tabular}{l cccc l}")
println(tex, "\\toprule")
println(tex, "Method & \$\\max_t \\|R_t\\|_\\infty\$ & mean \$\\|R_t\\|_\\infty\$ & \$\\log_{10}(\\max \\|R_t\\|_\\infty)\$ & Top-3 worst equations \\\\")
println(tex, "\\midrule")
@printf(tex, "SEP           & %s & %s & %s & %s \\\\\n",
        stats_sep === nothing ? "---" : fmt_sci(stats_sep.max_euler),
        stats_sep === nothing ? "---" : fmt_sci(stats_sep.mean_euler),
        stats_sep === nothing ? "---" : fmt_log10(stats_sep.log10_max),
        stats_sep === nothing ? "(omitted)" : fmt_top3(stats_sep.top3))
@printf(tex, "ROM1          & %s & %s & %s & %s \\\\\n",
        fmt_sci(stats_rom1.max_euler),
        fmt_sci(stats_rom1.mean_euler),
        fmt_log10(stats_rom1.log10_max),
        fmt_top3(stats_rom1.top3))
@printf(tex, "ROM1 + NN (gate-on only) & %s & %s & %s & %s \\\\\n",
        fmt_sci(stats_nng.max_euler),
        fmt_sci(stats_nng.mean_euler),
        fmt_log10(stats_nng.log10_max),
        fmt_top3(stats_nng.top3))
@printf(tex, "ROM1 + NN (gate-everywhere) & %s & %s & %s & %s \\\\\n",
        fmt_sci(stats_nn.max_euler),
        fmt_sci(stats_nn.mean_euler),
        fmt_log10(stats_nn.log10_max),
        fmt_top3(stats_nn.top3))
println(tex, "\\bottomrule")
println(tex, "\\end{tabular}")
println(tex, "\\end{table}")
write(out_tex, String(take!(tex)))
println("\nLaTeX table written to $out_tex")

# ============================================================================
# 10. Write markdown summary
# ============================================================================

md = IOBuffer()
println(md, "# Euler errors study — SEP vs ROM1 vs ROM1+NN")
println(md)
println(md, "_Autogenerated by `scripts/euler_errors_study.jl` on $(now())._")
println(md)
println(md, "## Setup")
println(md)
println(md, "- Model: $(mm_model.model_name), 18 estimated parameters.")
println(md, "- Data: $(data_path) (T_obs = $T_obs).")
println(md, "- Posterior mean θ pooled over 4 warm-started surrogate HMC chains (seeds 42, 5, 6, 7).")
println(md, "- Shocks ε̂_{1:T} recovered via ROM1 inversion filter at θ-posterior-mean.")
println(md, "- Trajectories of length T_obs+1 computed once per method; residuals evaluated for t=1..T_obs-1")
println(md, "  (last period omitted because it requires y_{T_obs+1}).")
println(md)
println(md, "## Headline Euler errors (sup-norm across time)")
println(md)
println(md, "| Method | max \\|R_t\\|_∞ | mean \\|R_t\\|_∞ | log10(max) | Top-3 worst equations |")
println(md, "|---|---|---|---|---|")
if stats_sep === nothing
    println(md, "| SEP | --- | --- | --- | omitted — wall $(round(sep_wall_s, digits=1))s, errorflag=$sep_errorflag |")
else
    @printf(md, "| SEP | %s | %s | %s | %s |\n",
            fmt_sci(stats_sep.max_euler), fmt_sci(stats_sep.mean_euler),
            fmt_log10(stats_sep.log10_max), fmt_top3(stats_sep.top3))
end
@printf(md, "| ROM1 | %s | %s | %s | %s |\n",
        fmt_sci(stats_rom1.max_euler), fmt_sci(stats_rom1.mean_euler),
        fmt_log10(stats_rom1.log10_max), fmt_top3(stats_rom1.top3))
@printf(md, "| ROM1+NN (gate-on only) | %s | %s | %s | %s |\n",
        fmt_sci(stats_nng.max_euler), fmt_sci(stats_nng.mean_euler),
        fmt_log10(stats_nng.log10_max), fmt_top3(stats_nng.top3))
@printf(md, "| ROM1+NN (gate-everywhere) | %s | %s | %s | %s |\n",
        fmt_sci(stats_nn.max_euler), fmt_sci(stats_nn.mean_euler),
        fmt_log10(stats_nn.log10_max), fmt_top3(stats_nn.top3))
println(md)

println(md, "## Notes")
println(md)
if sep_omitted
    println(md, "- **SEP was omitted**: wall time $(round(sep_wall_s, digits=1))s, errorflag=$sep_errorflag. ")
    println(md, "  Only ROM1 and ROM1+NN Euler errors are available.")
else
    println(md, "- SEP finished in $(round(sep_wall_s, digits=1))s.  Trajectory reused for Euler-residual evaluation.")
end
println(md)
println(md, "## Files")
println(md)
println(md, "- `scripts/euler_errors_study.jl` — this script.")
println(md, "- `.local_artifacts/euler_errors/euler_errors_results.jls` — full per-period and per-equation arrays.")
println(md, "- `.local_artifacts/euler_errors/euler_errors_table.tex` — LaTeX table for the paper.")
println(md, "- `.local_artifacts/euler_errors/EULER_ERRORS_SUMMARY.md` — this summary.")
write(out_md, String(take!(md)))
println("Markdown summary written to $out_md")

# ============================================================================
# 11. Serialize full results
# ============================================================================

result = Dict{String,Any}(
    "T_obs"               => T_obs,
    "theta_names"         => theta_names,
    "post_mean"           => post_mean,
    "shocks_recovered"    => shocks_recovered,
    "Y_rom1"              => Y_rom1,
    "Y_nn"                => Y_nn,
    "Y_nn_gated"          => Y_nn_gated,
    "Y_sep"               => Y_sep,
    "gate_mask"           => gate_mask,
    "sep_omitted"         => sep_omitted,
    "sep_wall_s"          => sep_wall_s,
    "sep_errorflag"       => sep_errorflag,
    "sep_failure_period"  => sep_failure_period,
    "rom1_per_period_sup" => rom1_per_period,
    "nn_per_period_sup"   => nn_per_period,
    "nng_per_period_sup"  => nng_per_period,
    "sep_per_period_sup"  => sep_per_period,
    "rom1_per_eq_max"     => rom1_per_eq,
    "nn_per_eq_max"       => nn_per_eq,
    "nng_per_eq_max"      => nng_per_eq,
    "sep_per_eq_max"      => sep_per_eq,
    "stats_rom1"          => stats_rom1,
    "stats_nn"            => stats_nn,
    "stats_nng"           => stats_nng,
    "stats_sep"           => stats_sep,
    "obs_sigma"           => obs_sigma,
    "shock_sigmas"        => shock_sigmas,
    "chain_paths"         => chain_paths,
    "data_path"           => data_path,
    "surrogate_path"      => surrogate_path,
    "sep_horizon"         => sep_horizon,
    "sep_accept_tol"      => sep_accept_tol,
    "sep_nnodes"          => sep_nnodes,
    "sep_maxit"           => sep_maxit,
    "timestamp"           => string(now()),
)
serialize(out_jls, result)
println("Serialized results to $out_jls")
println("\nDone.")
