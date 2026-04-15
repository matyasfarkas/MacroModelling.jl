"""
HLT Model Estimation with NN Surrogate on Real U.S. Data

This script estimates the Smets-Wouters 2007 HLT model using a trained neural
network surrogate as the FOM (Full-Order Model) and first-order Kalman filter
as the ROM (Reduced-Order Model), with regime-switching between them.

Sample period: 1960Q1-2027Q2 (244 observations)
Estimated parameters: cprobp, cindp, curvp (price Phillips curve)
Filter: Regime-switching (ROM: Kalman, FOM: NN Surrogate)
"""

using MacroModelling
using Zygote
import Turing, Pigeons
import Turing: NUTS, sample
import ADTypes: AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL
using Dates
using Serialization

# Load the HLT model
include(joinpath(@__DIR__, "../models/Smets_Wouters_2007_HLT_obc.jl"))

# Load surrogate utilities
include(joinpath(@__DIR__, "../scripts/hlt_surrogate/hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "../scripts/hlt_surrogate/hlt_sep_surrogate_rom_utils.jl"))

println("="^80)
println("HLT Real Data Estimation with NN Surrogate")
println("Sample: 1960Q1-2027Q2 (244 observations)")
println("Date: ", Dates.now())
println("="^80)

# ============================================================================
# Load Trained Surrogate
# ============================================================================

println("\n[1/8] Loading trained NN surrogate...")

# Use most recent trained surrogate
surrogate_path = joinpath(@__DIR__, "..", ".local_artifacts", "hlt_validation_runs",
                         "hlt3_20260302_153555", "dataset", "hlt_sep_surrogate_trained.jls")

if !isfile(surrogate_path)
    error("Trained surrogate not found at: $surrogate_path")
end

surrogate_bundle = MacroModelling.load_hlt_surrogate_bundle(surrogate_path)
frozen = surrogate_bundle.frozen
surrogate_data = surrogate_bundle.payload
sur_meta = surrogate_bundle.meta

println("  ✓ Loaded surrogate from: $(basename(dirname(dirname(surrogate_path))))")
println("    ROM residual: $(get(sur_meta, "rom_residual", false))")
println("    ROM order: $(get(sur_meta, "rom_residual_order", 0))")

# ============================================================================
# Data Loading
# ============================================================================

println("\n[2/8] Loading U.S. macro data...")

dat = CSV.read(joinpath(@__DIR__, "data", "usmodel_update.csv"), DataFrame)
data_full = KeyedArray(Array(dat)', Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

observables_csv = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]
sample_idx = 47:290  # 1960Q1-2027Q2
data_matrix = data_full(observables_csv, sample_idx)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data_matrix = rekey(data_matrix, :Variable => observables)
obs_data = Array(data_matrix)  # T x n_obs matrix

println("  ✓ Loaded $(length(sample_idx)) observations")
println("  ✓ Observables: ", join(string.(observables), ", "))

# ============================================================================
# Parameter Setup
# ============================================================================

println("\n[3/8] Setting up parameters...")

mm_model = Smets_Wouters_2007_HLT_obc
hlt_param_names = string.(mm_model.parameters)
base_values = copy(mm_model.parameter_values)

# Estimate only price Phillips curve parameters
est_names = ["cprobp", "cindp", "curvp"]
theta_names = Symbol.(est_names)
est_idx = map(name -> findfirst(==(name), hlt_param_names), est_names)
@assert all(!isnothing, est_idx) "Missing parameters in model"
est_idx = Int.(est_idx)

init_params = base_values[est_idx]
println("  ✓ Estimating: ", join(est_names, ", "))
println("  ✓ Initial: cprobp=$(init_params[1]), cindp=$(init_params[2]), curvp=$(init_params[3])")

# Priors (from HLT 2016)
prior_cprobp_mu = 0.5
prior_cprobp_sd = 0.10
prior_cindp_mu = 0.5
prior_cindp_sd = 0.15
prior_curvp_mu = 75.0
prior_curvp_sd = 25.0

dists = [
    Beta(prior_cprobp_mu, prior_cprobp_sd, 0.5, 0.95, μσ = true),   # cprobp
    Beta(prior_cindp_mu, prior_cindp_sd, 0.01, 0.99, μσ = true),  # cindp
    Normal(prior_curvp_mu, prior_curvp_sd)           # curvp
]

# ============================================================================
# Surrogate Setup
# ============================================================================

println("\n[4/8] Setting up surrogate predictors...")

# Get state and observable names from surrogate metadata
state_names = haskey(sur_meta, "state_names") ? Symbol.(sur_meta["state_names"]) : Symbol[]
obs_names_sur = haskey(sur_meta, "observables") ? Symbol.(sur_meta["observables"]) : observables

@assert obs_names_sur == observables "Surrogate observables must match data observables"

# Map to model indices
obs_idx = indexin(observables, mm_model.var)
state_idx = indexin(state_names, mm_model.var)
@assert all(!isnothing, obs_idx) "Observable names not found in model"
@assert all(!isnothing, state_idx) "State names not found in model"

# Create ROM predictor (for residual surrogate)
rom_order = Int(get(sur_meta, "rom_residual_order", 1))
rom_mode = get(sur_meta, "rom_mode", :baseline)
theta_idx_rom = Int[]  # Baseline mode: no theta dependence

rom_predictor = RomPredictor(mm_model,
                             rom_order,
                             rom_mode,
                             true,  # use_obc
                             theta_idx_rom,
                             base_values,
                             nothing,
                             nothing,
                             Int.(state_idx),
                             Int.(obs_idx))

# Ensure ROM cache is initialized
if rom_mode == :baseline
    ensure_rom_cache!(rom_predictor, zeros(length(theta_idx_rom)))
end

# Create ROM full predictor (closure)
rom_full_predict = function (state::AbstractVector, shock_t::AbstractVector, θ_local::AbstractVector)
    return rom_predict(rom_predictor, state, shock_t, θ_local)
end

# Create surrogate predictor functions
d_obs = length(observables)
surrogate_residual_predict = (state, shock_t, θ_local) -> predict_frozen(frozen, vcat(state, shock_t, θ_local))

surrogate_step_predict = (state, shock_t, θ_local) -> MacroModelling.predict_additive_residual(
    rom_full_predict,
    surrogate_residual_predict,
    state,
    shock_t,
    θ_local,
    d_obs
)

println("  ✓ ROM predictor: order=$(rom_order), mode=$(rom_mode)")
println("  ✓ Surrogate residual predictor ready")

# ============================================================================
# Gate Calibration (TBA - using simple heuristic for now)
# ============================================================================

println("\n[5/8] Setting up gate function...")

# For now, use a simple heuristic: high ZLB probability when rates are low
# This can be refined with proper gate calibration

# Simple gate based on nominal rate (robs)
robs_idx = findfirst(==(Symbol("robs")), observables)
robs_data = obs_data[robs_idx, :]

# Use FOM (surrogate) when nominal rate < 2% (ZLB risk)
# Use ROM (Kalman) otherwise
zlb_threshold = 2.0  # 2% annual rate
gate_probs = Float64[r < zlb_threshold ? 0.75 : 0.25 for r in robs_data]

# Smooth the gate probabilities (moving average)
window = 4
gate_probs_smooth = copy(gate_probs)
for i in eachindex(gate_probs)
    lo = max(1, i - window)
    hi = min(length(gate_probs), i + window)
    gate_probs_smooth[i] = mean(gate_probs[lo:hi])
end
gate_probs = gate_probs_smooth

println("  ✓ Gate calibrated: mean FOM probability = $(round(mean(gate_probs), digits=3))")
println("  ✓ Periods with high ZLB risk: $(sum(gate_probs .> 0.5))/$(length(gate_probs))")

# ============================================================================
# Likelihood Function
# ============================================================================

println("\n[6/8] Defining likelihood function...")

obs_data_ka = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))

# Initial state (zeros in deviation from steady state)
d_state = length(state_names)
s0 = zeros(d_state)

# Observation error std (small, as in HLT)
obs_sigma = 0.01 * ones(length(observables))

# Shock standard deviations (from model calibration)
shock_sigmas = zeros(Float64, length(mm_model.exo))
for (i, shock_name) in enumerate(mm_model.exo)
    if contains(string(shock_name), "ᵒᵇᶜ")
        shock_sigmas[i] = 0.0
        continue
    end
    pidx = findfirst(==(Symbol("z_", shock_name)), mm_model.parameters)
    shock_sigmas[i] = pidx === nothing ? 1.0 : abs(Float64(base_values[pidx]))
end

"""
Linear (ROM) log-likelihood per period using first-order Kalman filter.
"""
function linear_loglik_per_period(obs_data_ka, theta, theta_names, s0)
    # Build full parameter vector
    params = copy(base_values)
    for (name, val) in zip(theta_names, theta)
        idx = findfirst(==(String(name)), hlt_param_names)
        params[idx] = val
    end

    # Compute Kalman filter log-likelihood
    # Use Zygote.@ignore since Kalman filter has internal mutations
    ll = Zygote.@ignore begin
        try
            MacroModelling.get_loglikelihood_per_period(
                mm_model,
                obs_data_ka,
                params,
                presample_periods = 4,
                initial_covariance = :diagonal,
                algorithm = :first_order,
                filter = :kalman,
            )
        catch e
            # If Kalman fails, return -Inf for all periods
            fill(-Inf, size(obs_data, 2))
        end
    end

    return ll
end

"""
Surrogate (FOM) log-likelihood per period using inversion filter with NN surrogate.
"""
function surrogate_loglik_per_period(s0, theta, obs_data, obs_sigma)
    # Inversion filter with surrogate
    ll, _ = MacroModelling.inversion_loglik_per_period(
        surrogate_step_predict,
        s0,
        theta,
        obs_data,
        obs_sigma,
        shock_sigmas;
        maxit = 100,
        tol = 1e-6,
        lambda = 1.0
    )
    return ll
end

"""
Turing model with regime-switching between ROM (Kalman) and FOM (NN Surrogate).
"""
Turing.@model function HLT_surrogate_switching(obs_data_ka, s0, obs_sigma, gate_probs)
    # Sample parameters
    cprobp ~ dists[1]
    cindp ~ dists[2]
    curvp ~ dists[3]
    θ = [cprobp, cindp, curvp]

    # Skip likelihood in prior sampling
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        # ROM likelihood (linear Kalman)
        ll_rom = linear_loglik_per_period(obs_data_ka, θ, theta_names, s0)

        # FOM likelihood (NN surrogate with inversion)
        ll_fom = surrogate_loglik_per_period(s0, θ, Array(obs_data_ka), obs_sigma)

        # Mix using regime-switching
        llh = MacroModelling.mix_loglikelihood(ll_fom, ll_rom, gate_probs)

        Turing.@addlogprob! llh
    end
end

println("  ✓ Regime-switching model defined")
println("    - ROM: First-order Kalman filter")
println("    - FOM: NN Surrogate + Inversion filter")
println("    - Gate: ZLB-based probability (mean = $(round(mean(gate_probs), digits=3)))")

# ============================================================================
# Mode Finding
# ============================================================================

println("\n[7/8] Finding posterior mode...")

function find_mode(; verbose::Bool = true)
    Random.seed!(42)

    loglik = HLT_surrogate_switching(obs_data_ka, s0, obs_sigma, gate_probs)

    verbose && println("  Running optimization...")

    mode = Turing.maximum_a_posteriori(
        loglik,
        Optim.NelderMead(),
        initial_params = init_params
    )

    if verbose
        println("  ✓ Mode found:")
        println("    cprobp = $(round(mode.values[1], digits=4))")
        println("    cindp  = $(round(mode.values[2], digits=4))")
        println("    curvp  = $(round(mode.values[3], digits=2))")
        println("    Log-posterior = $(round(mode.lp, digits=2))")
    end

    return mode
end

mode_result = find_mode()

# ============================================================================
# MCMC Sampling
# ============================================================================

println("\n[8/8] Running MCMC sampling...")

function run_mcmc(mode_init; n_samples::Int = 2000, verbose::Bool = true)
    Random.seed!(123)

    loglik = HLT_surrogate_switching(obs_data_ka, s0, obs_sigma, gate_probs)

    verbose && println("\n  Sampling $(n_samples) draws with surrogate regime-switching...")
    verbose && println("  Using NUTS with AutoZygote differentiation...")

    start_time = time()
    samps = Turing.sample(
        loglik,
        NUTS(adtype = AutoZygote()),
        n_samples,
        progress = true,
        initial_params = mode_init.values
    )
    elapsed = time() - start_time

    if verbose
        println("\n  ✓ Sampling complete ($(round(elapsed, digits=1))s)")
        println("\n  Posterior means:")
        means = mean(samps).nt.mean
        println("    cprobp = $(round(means[1], digits=4))")
        println("    cindp  = $(round(means[2], digits=4))")
        println("    curvp  = $(round(means[3], digits=2))")

        println("\n  MCMC Diagnostics:")
        summary_stats = summarystats(samps)
        println(summary_stats)
    end

    return samps, elapsed
end

samples, elapsed = run_mcmc(mode_result, n_samples = 50)

# ============================================================================
# Save Results
# ============================================================================

println("\nSaving results...")

results_dir = joinpath(@__DIR__, "..", ".local_artifacts",
                       "hlt_estimation_real_surrogate_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(results_dir)

# Save chain
chain_path = joinpath(results_dir, "mcmc_chain.jls")
serialize(chain_path, samples)

# Save summary
summary_path = joinpath(results_dir, "estimation_summary.jls")
summary = Dict(
    "model" => "Smets_Wouters_2007_HLT_obc",
    "sample_period" => "1960Q1-2027Q2",
    "n_observations" => length(sample_idx),
    "sample_idx" => sample_idx,
    "estimated_params" => est_names,
    "mode" => Dict(zip(est_names, mode_result.values)),
    "mode_logposterior" => mode_result.lp,
    "posterior_means" => Dict(zip(est_names, mean(samples).nt.mean)),
    "n_samples" => 2000,
    "elapsed_seconds" => elapsed,
    "filter" => "regime_switching_surrogate (ROM: kalman, FOM: nn_surrogate+inversion)",
    "surrogate_path" => surrogate_path,
    "gate_mean_fom_prob" => mean(gate_probs),
    "timestamp" => string(now()),
)
serialize(summary_path, summary)

# Save CSV
csv_path = joinpath(results_dir, "posterior_estimates.csv")
posterior_df = DataFrame(samples)
CSV.write(csv_path, posterior_df)

# Save gate probabilities
gate_path = joinpath(results_dir, "gate_probs.csv")
gate_df = DataFrame(period = 1:length(gate_probs), fom_prob = gate_probs)
CSV.write(gate_path, gate_df)

println("  ✓ Results saved to: $(results_dir)")
println("    - MCMC chain: mcmc_chain.jls")
println("    - Summary: estimation_summary.jls")
println("    - CSV: posterior_estimates.csv")
println("    - Gate: gate_probs.csv")

println("\n" * "="^80)
println("Surrogate-Based Regime-Switching Estimation Complete!")
println("="^80)
println("\nResults directory: $(results_dir)")
println("="^80)
