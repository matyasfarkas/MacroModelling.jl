#!/usr/bin/env julia
# ============================================================================
# COVID SHOCK DECOMPOSITION — Linear vs Nonlinear Inversion Filter
# ============================================================================
#
# The key exercise: run the inversion filter at each posterior mean to recover
# the shocks that rationalize the SAME observed data. The linear model uses
# the standard (ROM1) inversion filter; the surrogate model uses the
# NN-corrected inversion filter that accounts for nonlinear propagation.
#
# If the nonlinear model generates endogenous amplification through the
# investment channel, it should need SMALLER exogenous shocks to explain
# the same observations — especially during COVID.
#
# Three-way decomposition:
#   (A) Linear inversion @ θ_L  — baseline (linear model, linear posterior)
#   (B) Linear inversion @ θ_S  — parameter effect only (same model, different θ)
#   (C) Surrogate inversion @ θ_S — full nonlinear effect (different model + θ)
#
# The gap (B)→(C) isolates the MODEL channel: how much does the NN correction
# reduce the shocks needed to rationalize the data, holding parameters fixed?
#
# Output:
#   - figures/covid_shock_decomposition.pdf  (shock norm time series)
#   - figures/covid_shock_bar_decomposition.pdf (per-shock bar chart)
#   - figures/covid_ll_decomposition.pdf     (per-period LL contribution)
# ============================================================================

using Serialization, Random, LinearAlgebra
import Statistics: mean, std, quantile
using Printf, Dates
using MacroModelling
using AxisKeys

# ============================================================================
# Paths
# ============================================================================

repo_root = normpath(joinpath(@__DIR__, ".."))
artifact_dir = joinpath(repo_root, ".local_artifacts", "hlt_18param_realdata")
fig_dir = joinpath(repo_root, "docs", "paper", "figures")
mkpath(fig_dir)

data_path       = joinpath(artifact_dir, "hlt_real_data_payload_extended_18p.jls")
linear_chain    = joinpath(artifact_dir, "hlt_linear_hmc_extended_18p_2000.jls")
surrogate_chain = joinpath(artifact_dir, "hlt_surrogate_hmc_extended_18p_2000.jls")
surrogate_path  = joinpath(repo_root, ".local_artifacts",
                           "hlt_18param_validation_v2_combined",
                           "hlt_sep_surrogate_trained_with_zlb.jls")
gate_path       = joinpath(artifact_dir, "gate_calibration_extended_18p.jls")

println("=" ^ 72)
println("COVID SHOCK DECOMPOSITION — Linear vs Surrogate Inversion Filter")
println("Started: $(now())")
println("=" ^ 72)

# ============================================================================
# Step 1: Load data payload
# ============================================================================

println("\n--- Step 1: Loading data payload ---")

payload = MacroModelling.load_hlt_synthetic_scenario(data_path)
obs_data       = payload["obs_data"]
obs_sigma_base = payload["obs_sigma"]
theta_names    = payload["theta_names"]
observables    = payload["observables"]
state_names    = haskey(payload, "state_names") ? payload["state_names"] : Symbol[]
s0             = payload["s0"]
shock_sigmas   = payload["shock_sigmas"]
d_obs = size(obs_data, 1)
T_obs = size(obs_data, 2)
n_theta = length(theta_names)
d_state = length(s0)

println("  Observables: $observables ($d_obs × $T_obs)")
println("  Shock dims:  $(length(shock_sigmas)) ($(count(shock_sigmas .> 0)) structural)")

# ============================================================================
# Step 2: Load model + surrogate
# ============================================================================

println("\n--- Step 2: Loading HLT model + surrogate ---")

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

mm_model = load_hlt_model(repo_root, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  Model: $(mm_model.model_name), $(length(mm_model.var)) vars, $(length(mm_model.timings.exo)) shocks")

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
frozen = surrogate_bundle.frozen
sur_meta = surrogate_bundle.meta
val_rmse = get(surrogate_bundle.payload, "validation_rmse", nothing)
println("  Surrogate: d_in=$(frozen.d_in), d_out=$(frozen.d_out)")

# ============================================================================
# Step 3: Build predict functions (same as run_surrogate_hmc_advancedhmc.jl)
# ============================================================================

println("\n--- Step 3: Building predict functions ---")

obs_idx = indexin(observables, mm_model.var)
state_idx = indexin(state_names, mm_model.var)
theta_param_idx = Int.(indexin(theta_names, mm_model.parameters))
base_parameters = copy(mm_model.parameter_values)

rom_predictor = RomPredictor(mm_model, 1, :theta_dependent, false, theta_param_idx,
                             base_parameters, nothing, nothing,
                             Int.(state_idx), Int.(obs_idx))
# Initialize cache at calibration (will re-solve when θ changes)
ensure_rom_cache!(rom_predictor, Float64.(base_parameters[theta_param_idx]))

rom_full_predict = (state, shock_t, θ_local) -> rom_predict(rom_predictor, state, shock_t, θ_local)

# Surrogate theta padding
surrogate_theta_names = get(sur_meta, "theta_names", Symbol[])
if !isempty(surrogate_theta_names) && length(surrogate_theta_names) != length(theta_names)
    _sur_theta_baseline = zeros(Float64, length(surrogate_theta_names))
    _sur_theta_est_idx = zeros(Int, length(surrogate_theta_names))
    for (si, sname) in enumerate(surrogate_theta_names)
        ei = findfirst(==(sname), theta_names)
        if ei !== nothing
            _sur_theta_est_idx[si] = ei
        else
            pi = findfirst(==(sname), mm_model.parameters)
            _sur_theta_baseline[si] = pi !== nothing ? base_parameters[pi] : 0.0
        end
    end
    function _pad_theta(θ_local)
        θ_full = copy(_sur_theta_baseline)
        for i in eachindex(_sur_theta_est_idx)
            if _sur_theta_est_idx[i] > 0
                θ_full[i] = θ_local[_sur_theta_est_idx[i]]
            end
        end
        return θ_full
    end
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, _pad_theta(θ_local)))[1:d_obs]
    batch_nn_residual = function(X_nn::AbstractMatrix)
        d_prefix = d_state + length(shock_sigmas)
        T_batch = size(X_nn, 2)
        X_padded = Matrix{eltype(X_nn)}(undef, frozen.d_in, T_batch)
        X_padded[1:d_prefix, :] .= X_nn[1:d_prefix, :]
        for t in 1:T_batch
            X_padded[(d_prefix+1):end, t] .= _pad_theta(X_nn[(d_prefix+1):end, t])
        end
        return predict_frozen_batch(frozen, X_padded)
    end
    single_nn_residual = function(x_nn)
        d_prefix = d_state + length(shock_sigmas)
        x_padded = vcat(x_nn[1:d_prefix], _pad_theta(x_nn[(d_prefix+1):end]))
        return predict_frozen(frozen, x_padded)
    end
else
    surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, θ_local))[1:d_obs]
    batch_nn_residual = (X_nn) -> predict_frozen_batch(frozen, X_nn)
    single_nn_residual = (x_nn) -> predict_frozen(frozen, x_nn)
end

rom_only_predict = (state, shock_t, θ_local) -> MacroModelling.predict_from_full(
    rom_full_predict, state, shock_t, θ_local, d_obs)

# Obs sigma (matching surrogate HMC script)
obs_sigma_scale = 2.0
obs_sigma_floor = 0.1
obs_sigma = copy(obs_sigma_base)
if val_rmse !== nothing && length(val_rmse) >= d_obs
    obs_sigma = max.(obs_sigma, val_rmse[1:d_obs] .* obs_sigma_scale)
end
obs_sigma = max.(obs_sigma, obs_sigma_floor)
println("  obs_sigma: $(round.(obs_sigma, sigdigits=3))")

# Correction clamp
nn_correction_clamp = nothing
if val_rmse !== nothing && length(val_rmse) == frozen.d_out
    nn_correction_clamp = 3.0 .* val_rmse
end

# ============================================================================
# Step 4: Load gate calibration
# ============================================================================

println("\n--- Step 4: Loading gate calibration ---")

gate_probs = nothing
gate_mask  = trues(T_obs)

if isfile(gate_path)
    gate_calib = MacroModelling.load_hlt_gate_calibration(gate_path)
    tau_eps = gate_calib["tau_eps"]
    tau_y   = gate_calib["tau_y"]
    gate_use_eps = get(gate_calib, "use_eps", true)
    gate_use_y   = get(gate_calib, "use_y", true)

    if haskey(gate_calib, "e_stats") && haskey(gate_calib, "f_stats") &&
       length(gate_calib["e_stats"]) == T_obs && length(gate_calib["f_stats"]) == T_obs
        e_stat = vec(Float64.(gate_calib["e_stats"]))
        f_stat = vec(Float64.(gate_calib["f_stats"]))

        eps_mask = gate_use_eps ? (e_stat .> tau_eps) : falses(T_obs)
        y_mask   = gate_use_y   ? (f_stat .> tau_y)   : falses(T_obs)
        base_mask = eps_mask .| y_mask
        gate_mask = MacroModelling.apply_gate_padding(base_mask, 4, 8, 4)

        # Soft gate probabilities
        gate_prob_floor_val = 1e-4
        gate_prob_ceil_val  = 1.0 - 1e-4
        target_share = mean(base_mask)
        prior_probs = clamp.(fill(target_share, T_obs), gate_prob_floor_val, gate_prob_ceil_val)
        prior_logit = MacroModelling.logit.(prior_probs)
        eps_scale = max(tau_eps, eps(Float64))
        y_scale   = max(tau_y, eps(Float64))
        eps_score = gate_use_eps ? ((e_stat .- tau_eps) ./ eps_scale) : zeros(T_obs)
        y_score   = gate_use_y ? ((f_stat .- tau_y) ./ y_scale) : zeros(T_obs)
        score = eps_score .+ y_score
        gate_bias = MacroModelling.calibrate_gate_bias(score .+ prior_logit, target_share)
        gate_probs = clamp.(MacroModelling.logistic.(gate_bias .+ score .+ prior_logit),
                            gate_prob_floor_val, gate_prob_ceil_val)

        println("  Gate: tau_eps=$(round(tau_eps, sigdigits=3)), tau_y=$(round(tau_y, sigdigits=3))")
        println("  Gate periods: $(count(gate_mask))/$T_obs, soft mean=$(round(mean(gate_probs), sigdigits=3))")
    else
        println("  WARNING: gate calibration missing stats, using all-surrogate")
    end
else
    println("  No gate calibration file found — using surrogate on ALL periods")
end

use_regime_switching = isfile(gate_path) && !all(gate_mask)

# ============================================================================
# Step 5: Load posterior chains
# ============================================================================

println("\n--- Step 5: Loading posterior chains ---")

function load_chain_mean(path, burnin=500)
    raw = deserialize(path)
    chain_payload = raw isa Dict ? Dict{Any,Any}(raw) : Dict{Any,Any}("chain" => raw)
    chain_mat = chain_payload["chain"]  # (n_samples, n_params)
    n_samples = size(chain_mat, 1)
    burn_in = min(div(n_samples, 4), burnin)
    theta_mean = vec(mean(chain_mat[burn_in+1:end, :], dims=1))
    ll_stored = get(chain_payload, "ll_post_mean", NaN)
    println("  Loaded: $path ($n_samples samples, burn-in=$burn_in, LL=$(round(ll_stored, digits=1)))")
    return theta_mean
end

theta_linear    = load_chain_mean(linear_chain)
theta_surrogate = load_chain_mean(surrogate_chain)

println("  Linear posterior mean:    $(round.(theta_linear, sigdigits=4))")
println("  Surrogate posterior mean: $(round.(theta_surrogate, sigdigits=4))")

# ============================================================================
# Step 6: Run inversion filter — LINEAR model at linear posterior mean
# ============================================================================

println("\n--- Step 6: Linear inversion filter at linear posterior mean ---")

t0 = time()
ll_linear, shocks_linear = MacroModelling.inversion_loglik_per_period(
    rom_only_predict,
    s0,
    theta_linear,
    obs_data,
    obs_sigma,
    shock_sigmas;
    maxit = 10,
    tol = 1e-6,
    lambda = 1e-4,
)
dt_linear = time() - t0

println("  LL (linear inversion): $(round(sum(ll_linear), digits=2))  [$(@sprintf("%.1f", dt_linear))s]")
println("  Shocks matrix: $(size(shocks_linear))")

# ============================================================================
# Step 7: Run inversion filter — SURROGATE model at surrogate posterior mean
# ============================================================================

println("\n--- Step 7: Surrogate inversion filter at surrogate posterior mean ---")

t0 = time()
if use_regime_switching
    ll_surrogate, shocks_surrogate = MacroModelling.inversion_loglik_per_period(
        rom_only_predict,
        s0,
        theta_surrogate,
        obs_data,
        obs_sigma,
        shock_sigmas;
        batch_eval_residual_fn = batch_nn_residual,
        single_eval_residual_fn = single_nn_residual,
        gate_mask = BitVector(gate_mask),
        correction_clamp = nn_correction_clamp,
        maxit = 10,
        tol = 1e-6,
        lambda = 1e-4,
    )
else
    ll_surrogate, shocks_surrogate = MacroModelling.inversion_loglik_per_period(
        rom_only_predict,
        s0,
        theta_surrogate,
        obs_data,
        obs_sigma,
        shock_sigmas;
        batch_eval_residual_fn = batch_nn_residual,
        maxit = 10,
        tol = 1e-6,
        lambda = 1e-4,
    )
end
dt_surrogate = time() - t0

println("  LL (surrogate inversion): $(round(sum(ll_surrogate), digits=2))  [$(@sprintf("%.1f", dt_surrogate))s]")
println("  Shocks matrix: $(size(shocks_surrogate))")

# ============================================================================
# Step 8: Also run LINEAR inversion at surrogate posterior (for decomposition)
# ============================================================================

println("\n--- Step 8: Linear inversion at surrogate posterior mean (decomposition) ---")

t0 = time()
ll_linear_at_surr, shocks_linear_at_surr = MacroModelling.inversion_loglik_per_period(
    rom_only_predict,
    s0,
    theta_surrogate,
    obs_data,
    obs_sigma,
    shock_sigmas;
    maxit = 10,
    tol = 1e-6,
    lambda = 1e-4,
)
dt_linsurr = time() - t0
println("  LL (linear at surrogate θ): $(round(sum(ll_linear_at_surr), digits=2))  [$(@sprintf("%.1f", dt_linsurr))s]")

# ============================================================================
# Step 9: Compute shock statistics
# ============================================================================

println("\n--- Step 9: Shock decomposition statistics ---")

structural_idx = findall(shock_sigmas .> 0)
shock_std = shock_sigmas[structural_idx]
shock_names = mm_model.timings.exo

# Convert to σ-units
shocks_linear_sigma    = shocks_linear[structural_idx, :] ./ shock_std
shocks_surrogate_sigma = shocks_surrogate[structural_idx, :] ./ shock_std
shocks_linsurr_sigma   = shocks_linear_at_surr[structural_idx, :] ./ shock_std

# Per-period shock norm (L2 in σ-units)
norm_linear    = [sqrt(sum(shocks_linear_sigma[:, t].^2)) for t in 1:T_obs]
norm_surrogate = [sqrt(sum(shocks_surrogate_sigma[:, t].^2)) for t in 1:T_obs]
norm_linsurr   = [sqrt(sum(shocks_linsurr_sigma[:, t].^2)) for t in 1:T_obs]

# COVID window: 2020Q1-2021Q2 = quarters 245-250 in 1959Q1-2025Q1 sample (T=265)
covid_start = 1 + 4 * (2020 - 1959)  # = 245
covid_end   = covid_start + 5          # = 250 (2021Q2)

function quarter_label(t)
    year = 1959 + (t - 1) ÷ 4
    q = 1 + (t - 1) % 4
    return "$(year)Q$(q)"
end

println("\n" * "=" ^ 72)
println("SHOCK DECOMPOSITION: COVID WINDOW ($(quarter_label(covid_start))–$(quarter_label(covid_end)))")
println("=" ^ 72)

println("\n  Per-shock mean |ε/σ| during COVID:")
println(@sprintf("  %-15s  %-10s  %-10s  %-10s  %-10s", "Shock", "Linear@θ_L", "Linear@θ_S", "Surrog@θ_S", "Reduction"))
println("  " * "-" ^ 65)
for (i, sname) in enumerate(shock_names)
    mean_lin  = mean(abs.(shocks_linear_sigma[i, covid_start:covid_end]))
    mean_lsur = mean(abs.(shocks_linsurr_sigma[i, covid_start:covid_end]))
    mean_sur  = mean(abs.(shocks_surrogate_sigma[i, covid_start:covid_end]))
    reduction = (mean_lsur > 0) ? (mean_lsur - mean_sur) / mean_lsur * 100 : 0.0
    println(@sprintf("  %-15s  %8.2fσ   %8.2fσ   %8.2fσ   %+6.1f%%",
                     string(sname), mean_lin, mean_lsur, mean_sur, -reduction))
end

println("\n  Shock norm ||ε/σ||₂ during COVID:")
println(@sprintf("  %-10s  %-12s  %-12s  %-12s  %-10s", "Period", "Linear@θ_L", "Linear@θ_S", "Surrog@θ_S", "Reduction"))
println("  " * "-" ^ 65)
for t in covid_start:covid_end
    reduction = (norm_linsurr[t] > 0) ? (norm_linsurr[t] - norm_surrogate[t]) / norm_linsurr[t] * 100 : 0.0
    println(@sprintf("  %-10s  %10.2fσ   %10.2fσ   %10.2fσ   %+6.1f%%",
                     quarter_label(t), norm_linear[t], norm_linsurr[t], norm_surrogate[t], -reduction))
end

# Summary statistics
mean_norm_lin_covid = mean(norm_linear[covid_start:covid_end])
mean_norm_sur_covid = mean(norm_surrogate[covid_start:covid_end])
mean_norm_linsurr_covid = mean(norm_linsurr[covid_start:covid_end])
total_shock_energy_lin = sum(shocks_linear_sigma[:, covid_start:covid_end].^2)
total_shock_energy_sur = sum(shocks_surrogate_sigma[:, covid_start:covid_end].^2)
total_shock_energy_linsurr = sum(shocks_linsurr_sigma[:, covid_start:covid_end].^2)

println("\n  Summary:")
println(@sprintf("  Mean shock norm (COVID): Linear@θ_L=%.2fσ, Surrog@θ_S=%.2fσ",
                 mean_norm_lin_covid, mean_norm_sur_covid))
println(@sprintf("  Total shock energy (Σε²/σ²): Linear@θ_L=%.1f, Linear@θ_S=%.1f, Surrog@θ_S=%.1f",
                 total_shock_energy_lin, total_shock_energy_linsurr, total_shock_energy_sur))
if total_shock_energy_linsurr > 0
    println(@sprintf("  Energy reduction (θ_S: linear→surrogate): %.1f%%",
                     100 * (1 - total_shock_energy_sur / total_shock_energy_linsurr)))
end
println(@sprintf("  LL: Linear@θ_L=%.1f, Linear@θ_S=%.1f, Surrog@θ_S=%.1f",
                 sum(ll_linear), sum(ll_linear_at_surr), sum(ll_surrogate)))

# Per-period LL gap
ll_gap = ll_surrogate .- ll_linear
covid_ll_gap = sum(ll_gap[covid_start:covid_end])
noncovid_ll_gap = sum(ll_gap) - covid_ll_gap
total_gap = sum(ll_gap)
println(@sprintf("\n  LL gap (total): %.1f nats", total_gap))
if total_gap != 0
    println(@sprintf("  LL gap from COVID periods: %.1f nats (%.0f%% of total)", covid_ll_gap,
                     100 * abs(covid_ll_gap / total_gap)))
end
println(@sprintf("  LL gap from non-COVID:     %.1f nats", noncovid_ll_gap))

# Three-way LL decomposition
model_channel = sum(ll_surrogate) - sum(ll_linear_at_surr)
param_channel = sum(ll_linear_at_surr) - sum(ll_linear)
println("\n  LL gap decomposition:")
println(@sprintf("    Total:     %+.1f nats", total_gap))
println(@sprintf("    Model:     %+.1f nats (surrogate vs linear, same θ_S)", model_channel))
println(@sprintf("    Parameter: %+.1f nats (θ_S vs θ_L, same linear model)", param_channel))

# ============================================================================
# Step 10: Generate figures
# ============================================================================

println("\n--- Step 10: Generating figures ---")

using Plots
gr()

# Plot window: 2018Q1–2025Q1 (last 29 quarters)
plot_start = max(1, T_obs - 28)
plot_range = plot_start:T_obs
tick_positions = [t for t in plot_range if (t - 1) % 4 == 0]
tick_labels = [quarter_label(t) for t in tick_positions]

# --- Figure 1: Shock norm time series ---

fig1 = plot(size=(780, 420), margin=5Plots.mm, bottom_margin=14Plots.mm, left_margin=8Plots.mm)

vspan!(fig1, [covid_start - 0.5, covid_end + 0.5]; color=:gray90, label="", alpha=0.5)

plot!(fig1, plot_range, norm_linear[plot_range];
      color=RGB(0.2, 0.4, 0.8), linewidth=2.0, linestyle=:dash,
      label="Linear @ θ_L")
plot!(fig1, plot_range, norm_linsurr[plot_range];
      color=RGB(0.6, 0.4, 0.8), linewidth=1.5, linestyle=:dot,
      label="Linear @ θ_S")
plot!(fig1, plot_range, norm_surrogate[plot_range];
      color=RGB(0.8, 0.2, 0.2), linewidth=2.0,
      label="Surrogate @ θ_S")

hline!(fig1, [1.0]; color=:gray60, linestyle=:dot, linewidth=0.8, label="")

plot!(fig1,
      xlabel = "", ylabel = "Shock norm ||ε/σ||₂",
      title = "Recovered Shock Magnitudes: Linear vs Surrogate Inversion",
      legend = :topleft,
      xticks = (tick_positions, tick_labels),
      xrotation = 45)

figpath1 = joinpath(fig_dir, "covid_shock_decomposition.pdf")
savefig(fig1, figpath1)
println("  Saved: $figpath1")

# --- Figure 2: Per-shock comparison during COVID (grouped bars) ---

fig2 = plot(size=(780, 420), margin=5Plots.mm, left_margin=10Plots.mm)

shock_labels = string.(shock_names)
n_shocks = length(shock_names)
mean_abs_lin    = [mean(abs.(shocks_linear_sigma[i, covid_start:covid_end])) for i in 1:n_shocks]
mean_abs_linsur = [mean(abs.(shocks_linsurr_sigma[i, covid_start:covid_end])) for i in 1:n_shocks]
mean_abs_sur    = [mean(abs.(shocks_surrogate_sigma[i, covid_start:covid_end])) for i in 1:n_shocks]

x_pos = 1:n_shocks
bar_width = 0.25

bar!(fig2, x_pos .- bar_width, mean_abs_lin; bar_width=bar_width,
     color=RGB(0.2, 0.4, 0.8), alpha=0.8, label="Linear @ θ_L")
bar!(fig2, x_pos, mean_abs_linsur; bar_width=bar_width,
     color=RGB(0.6, 0.4, 0.8), alpha=0.8, label="Linear @ θ_S")
bar!(fig2, x_pos .+ bar_width, mean_abs_sur; bar_width=bar_width,
     color=RGB(0.8, 0.2, 0.2), alpha=0.8, label="Surrogate @ θ_S")

hline!(fig2, [1.0]; color=:gray60, linestyle=:dot, linewidth=0.8, label="")

plot!(fig2,
      xlabel = "", ylabel = "Mean |ε/σ| during COVID (2020Q1–2021Q2)",
      title = "Per-Shock Decomposition: Exogenous vs Endogenous Attribution",
      legend = :topright,
      xticks = (x_pos, shock_labels),
      xrotation = 0)

figpath2 = joinpath(fig_dir, "covid_shock_bar_decomposition.pdf")
savefig(fig2, figpath2)
println("  Saved: $figpath2")

# --- Figure 3: Per-period LL decomposition ---

fig3 = plot(size=(780, 420), margin=5Plots.mm, bottom_margin=14Plots.mm, left_margin=8Plots.mm)

vspan!(fig3, [covid_start - 0.5, covid_end + 0.5]; color=:gray90, label="", alpha=0.5)

plot!(fig3, plot_range, ll_linear[plot_range];
      color=RGB(0.2, 0.4, 0.8), linewidth=1.5, linestyle=:dash,
      label="Linear LL (per period)")
plot!(fig3, plot_range, ll_surrogate[plot_range];
      color=RGB(0.8, 0.2, 0.2), linewidth=1.5,
      label="Surrogate LL (per period)")

# LL gap
plot!(fig3, plot_range, ll_gap[plot_range];
      color=RGB(0.2, 0.7, 0.3), linewidth=2.0, fillalpha=0.15, fill=0,
      label="LL gap (Surrogate − Linear)")

hline!(fig3, [0.0]; color=:gray60, linestyle=:dot, linewidth=0.8, label="")

plot!(fig3,
      xlabel = "", ylabel = "Log-likelihood (per period)",
      title = "Per-Period Likelihood: Where the Surrogate Gains",
      legend = :bottomleft,
      xticks = (tick_positions, tick_labels),
      xrotation = 45)

figpath3 = joinpath(fig_dir, "covid_ll_decomposition.pdf")
savefig(fig3, figpath3)
println("  Saved: $figpath3")

# ============================================================================
# Step 11: Save results
# ============================================================================

println("\n--- Step 11: Saving results ---")

results = Dict(
    "shocks_linear" => shocks_linear,
    "shocks_surrogate" => shocks_surrogate,
    "shocks_linear_at_surr" => shocks_linear_at_surr,
    "ll_linear" => ll_linear,
    "ll_surrogate" => ll_surrogate,
    "ll_linear_at_surr" => ll_linear_at_surr,
    "theta_linear" => theta_linear,
    "theta_surrogate" => theta_surrogate,
    "shock_names" => shock_names,
    "shock_sigmas" => shock_sigmas,
    "structural_idx" => structural_idx,
    "covid_start" => covid_start,
    "covid_end" => covid_end,
    "T_obs" => T_obs,
)

out_dir = joinpath(repo_root, ".local_artifacts", "shock_decomposition")
mkpath(out_dir)
outpath = joinpath(out_dir, "covid_shock_decomposition_results.jls")
serialize(outpath, results)
println("  Saved: $outpath")

println("\n" * "=" ^ 72)
println("SHOCK DECOMPOSITION COMPLETE: $(now())")
println("=" ^ 72)
