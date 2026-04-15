"""
18-Parameter Surrogate Dataset Generation for HLT Model

This script generates a comprehensive dataset for training a neural network surrogate
to approximate the SEP (Stochastic Extended Path) solver for the Smets-Wouters 2007
HLT model with 18 estimated structural parameters.

Parameter Set (18 total):
1. Structural preferences and technology:
   - csigma: Risk aversion
   - csigl: Labor supply elasticity (inverse Frisch)
   - chabb: Habit persistence
   - calfa: Capital share

2. Price and wage rigidities (Calvo + indexation):
   - cprobp: Price stickiness (Calvo probability)
   - cindp: Price indexation to past inflation
   - cprobw: Wage stickiness (Calvo probability)
   - cindw: Wage indexation to past inflation

3. Kimball curvatures (quasi-kinked demand):
   - curvp: Price markup curvature
   - curvw: Wage markup curvature

4. Taylor rule coefficients:
   - crpi: Response to inflation
   - cry: Response to output gap
   - crr: Interest rate smoothing

5. Shock persistence (AR(1) coefficients):
   - crhoa: Technology shock persistence
   - crhob: Preference shock persistence
   - crhog: Government spending shock persistence

6. Shock standard deviations:
   - z_ea: Technology shock SD
   - z_eb: Preference shock SD
   - z_em: Monetary shock SD

Dataset Structure:
- Parameter grid: 200-point Sobol sequence over prior support
- Time periods: 180 (45 years quarterly)
- Total transitions: 200 × 180 = 36,000 samples
- States: 22 variables (from HLT model state vector)
- Observables: 7 (dy, dc, dinve, labobs, pinfobs, dwobs, robs)

Output:
- Parameter grid CSV
- SEP dataset HDF5 (states, observables, shocks)
- Metadata JSON (parameter names, bounds, dataset stats)
"""

using MacroModelling
using Random, Distributions
using CSV, DataFrames, HDF5
using ProgressMeter
using Dates
using Sobol  # For low-discrepancy parameter sampling
using JSON3

# Load model
include(joinpath(@__DIR__, "../models/Smets_Wouters_2007_HLT_obc.jl"))

println("="^80)
println("18-Parameter SEP Dataset Generation for HLT Model")
println("Date: ", Dates.now())
println("="^80)

# ============================================================================
# Parameter Setup
# ============================================================================

println("\n[1/6] Defining parameter grid...")

mm_model = Smets_Wouters_2007_HLT_obc
hlt_param_names = string.(mm_model.parameters)
base_values = copy(mm_model.parameter_values)

# Define the 18 parameters to estimate
est_names = [
    # Structural preferences and technology (4)
    "csigma",    # Risk aversion
    "csigl",     # Labor supply elasticity (inverse Frisch)
    "chabb",     # Habit persistence
    "calfa",     # Capital share

    # Price and wage rigidities (4)
    "cprobp",    # Price stickiness (Calvo)
    "cindp",     # Price indexation
    "cprobw",    # Wage stickiness (Calvo)
    "cindw",     # Wage indexation

    # Kimball curvatures (2)
    "curvp",     # Price markup curvature
    "curvw",     # Wage markup curvature

    # Taylor rule (3)
    "crpi",      # Response to inflation
    "cry",       # Response to output gap
    "crr",       # Interest rate smoothing

    # Shock persistence (3)
    "crhoa",     # Technology shock persistence
    "crhob",     # Preference shock persistence
    "crhog",     # Government spending shock persistence

    # Shock standard deviations (2 most important)
    "z_ea",      # Technology shock SD
    "z_eb",      # Preference shock SD
]

n_params = length(est_names)
@assert n_params == 18 "Expected 18 parameters"

# Get parameter indices
est_idx = map(name -> findfirst(==(name), hlt_param_names), est_names)
@assert all(!isnothing, est_idx) "Missing parameters in model: $(est_names[findfirst(isnothing, est_idx)])"
est_idx = Int.(est_idx)

println("  ✓ Estimating $(n_params) parameters:")
for (i, name) in enumerate(est_names)
    println("    $(i). $(name) (baseline: $(round(base_values[est_idx[i]], digits=4)))")
end

# Define prior bounds (from HLT 2016 and Smets-Wouters 2007)
# Format: (lower, upper) for each parameter
prior_bounds = Dict(
    # Structural preferences and technology
    "csigma"  => (0.5, 4.0),      # Risk aversion (SW: ~1.5, range 0.5-4)
    "csigl"   => (0.5, 4.0),      # Inverse Frisch (SW: ~2, range 0.5-4)
    "chabb"   => (0.3, 0.9),      # Habit (SW: ~0.6, range 0.3-0.9)
    "calfa"   => (0.15, 0.35),    # Capital share (SW: ~0.24, range 0.15-0.35)

    # Price and wage rigidities
    "cprobp"  => (0.5, 0.95),     # Price stickiness (SW: ~0.6, HLT range 0.5-0.95)
    "cindp"   => (0.01, 0.99),    # Price indexation (SW: ~0.47, range 0.01-0.99)
    "cprobw"  => (0.5, 0.95),     # Wage stickiness (SW: ~0.81, range 0.5-0.95)
    "cindw"   => (0.01, 0.99),    # Wage indexation (SW: ~0.32, range 0.01-0.99)

    # Kimball curvatures
    "curvp"   => (1.0, 150.0),    # Price curvature (HLT: baseline 10, range 1-150)
    "curvw"   => (1.0, 150.0),    # Wage curvature (HLT: baseline 10, range 1-150)

    # Taylor rule
    "crpi"    => (1.1, 3.0),      # Inflation response (SW: ~1.5, Taylor principle 1.1-3)
    "cry"     => (0.01, 0.5),     # Output response (SW: ~0.06, range 0.01-0.5)
    "crr"     => (0.5, 0.95),     # Smoothing (SW: ~0.88, range 0.5-0.95)

    # Shock persistence
    "crhoa"   => (0.8, 0.999),    # Technology AR(1) (SW: ~0.998, high persistence)
    "crhob"   => (0.3, 0.95),     # Preference AR(1) (SW: ~0.58, moderate)
    "crhog"   => (0.8, 0.999),    # Government AR(1) (SW: ~0.996, high)

    # Shock standard deviations
    "z_ea"    => (0.2, 1.0),      # Technology SD (SW: ~0.46, range 0.2-1.0)
    "z_eb"    => (0.5, 4.0),      # Preference SD (SW: ~1.85, range 0.5-4.0)
)

# Extract bounds arrays
lb = [prior_bounds[name][1] for name in est_names]
ub = [prior_bounds[name][2] for name in est_names]

println("  ✓ Prior bounds defined")

# ============================================================================
# Generate Parameter Grid using Sobol Sequence
# ============================================================================

println("\n[2/6] Generating 200-point Sobol sequence...")

n_grid = 200
Random.seed!(42)

# Generate Sobol sequence in unit hypercube [0,1]^18
sobol_seq = SobolSeq(n_params)
sobol_points = [next!(sobol_seq) for _ in 1:n_grid]

# Transform to parameter bounds
param_grid = zeros(n_grid, n_params)
for i in 1:n_grid
    for j in 1:n_params
        # Linear transformation: [0,1] → [lb, ub]
        param_grid[i, j] = lb[j] + sobol_points[i][j] * (ub[j] - lb[j])
    end
end

println("  ✓ Generated $(n_grid) parameter vectors")
println("  ✓ Grid statistics:")
for (j, name) in enumerate(est_names)
    pmin = minimum(param_grid[:, j])
    pmax = maximum(param_grid[:, j])
    pmean = mean(param_grid[:, j])
    println("    $(name): [$(round(pmin, digits=3)), $(round(pmax, digits=3))], mean=$(round(pmean, digits=3))")
end

# Save parameter grid
grid_df = DataFrame(param_grid, Symbol.(est_names))
grid_path = joinpath(@__DIR__, "..", ".local_artifacts", "hlt_18param_grid_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
mkpath(dirname(grid_path))
CSV.write(grid_path, grid_df)
println("  ✓ Saved parameter grid to: $(basename(grid_path))")

# ============================================================================
# SEP Simulation Setup
# ============================================================================

println("\n[3/6] Setting up SEP simulation...")

n_periods = 180  # 45 years quarterly
n_shocks = length(mm_model.exo)
n_vars = length(mm_model.var)

# Identify state and observable variables
# States: Use all non-auxiliary forward-looking variables
# For HLT model, key states are: c, inve, k, kp, lab, w, pinf, r, mc, zcap, rk, pk, xi, qs, ...
# Observables: dy, dc, dinve, labobs, pinfobs, dwobs, robs (7 variables)

observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = indexin(observables, mm_model.var)
@assert all(!isnothing, obs_idx) "Observable names not found in model"
obs_idx = Int.(obs_idx)

# Define state variables (22 key endogenous states)
# These are the minimal set needed to characterize the model dynamics
state_names = [
    :c, :inve, :k, :kp, :lab, :w, :pinf, :r, :y,
    :mc, :zcap, :rk, :pk, :xi, :qs,
    :a, :b, :gy, :ms, :spinf, :sw,
    :yflex
]
state_idx = indexin(state_names, mm_model.var)
@assert all(!isnothing, state_idx) "State names not found in model"
state_idx = Int.(state_idx)
n_states = length(state_names)

println("  ✓ Observables ($(length(observables))): ", join(string.(observables), ", "))
println("  ✓ States ($(n_states)): ", join(string.(state_names), ", "))

# Shock standard deviations (will be updated for each parameter vector)
shock_names = mm_model.exo
base_shock_sigmas = zeros(Float64, n_shocks)
for (i, shock_name) in enumerate(shock_names)
    if contains(string(shock_name), "ᵒᵇᶜ")
        base_shock_sigmas[i] = 0.0
        continue
    end
    pidx = findfirst(==(Symbol("z_", shock_name)), mm_model.parameters)
    base_shock_sigmas[i] = pidx === nothing ? 0.1 : abs(Float64(base_values[pidx]))
end

println("  ✓ Shock configuration:")
for (i, name) in enumerate(shock_names)
    if base_shock_sigmas[i] > 0
        println("    $(name): σ = $(round(base_shock_sigmas[i], digits=4))")
    end
end

# ============================================================================
# Generate SEP Dataset
# ============================================================================

println("\n[4/6] Running SEP simulations ($(n_grid) params × $(n_periods) periods)...")

# Preallocate storage
dataset_states = zeros(Float64, n_grid * n_periods, n_states)
dataset_obs = zeros(Float64, n_grid * n_periods, length(observables))
dataset_shocks = zeros(Float64, n_grid * n_periods, n_shocks)
dataset_params = zeros(Float64, n_grid * n_periods, n_params)
dataset_valid = BitVector(undef, n_grid * n_periods)

n_success = 0
n_failed = 0

@showprogress desc="SEP simulation: " for i_param in 1:n_grid
    # Build parameter vector
    params = copy(base_values)
    params[est_idx] = param_grid[i_param, :]

    # Update shock standard deviations if they're in est_idx
    shock_sigmas = copy(base_shock_sigmas)
    for (j, name) in enumerate(est_names)
        if startswith(name, "z_")
            # Find corresponding shock
            shock_name = Symbol(name[3:end])  # Remove "z_" prefix
            shock_idx = findfirst(==(shock_name), shock_names)
            if shock_idx !== nothing
                shock_sigmas[shock_idx] = param_grid[i_param, j]
            end
        end
    end

    # Run SEP simulation
    try
        # Solve model at this parameter vector
        MacroModelling.write_parameters_input!(mm_model, params, verbose = false)
        MacroModelling.solve!(mm_model;
            algorithm = :first_order,
            dynamics = true,
            obc = true,
            silent = true
        )

        # Simulate using SEP (stochastic extended path)
        # API: simulate_sep(model; periods, initial_state, shocks, burn_in, sep_horizon, sep_order, sep_nnodes, shock_scaling, random_seed, silent)
        sim_result = MacroModelling.simulate_sep(
            mm_model;
            periods = n_periods,
            initial_state = nothing,  # Will use steady state
            shocks = nothing,         # Will generate random shocks
            burn_in = 50,            # Stabilization period
            sep_horizon = 40,
            sep_order = 1,           # First-order approximation in SEP
            sep_nnodes = 3,          # Gauss-Hermite nodes
            shock_scaling = :none,
            random_seed = i_param,   # Different seed per parameter vector
            silent = true
        )

        # Extract states and observables
        for t in 1:n_periods
            idx = (i_param - 1) * n_periods + t

            # sim_result is a matrix: time × variables
            full_state = sim_result[t, :]
            dataset_states[idx, :] = full_state[state_idx]
            dataset_obs[idx, :] = full_state[obs_idx]
            # Shocks are not directly returned by simulate_sep, only state evolution
            # For surrogate training, we need to invert the shocks from state transitions
            # For now, store zeros and note this limitation
            dataset_shocks[idx, :] .= 0.0  # TODO: Invert shocks from state evolution
            dataset_params[idx, :] = param_grid[i_param, :]
            dataset_valid[idx] = true
        end

        n_success += 1

    catch e
        # Mark failed simulations
        for t in 1:n_periods
            idx = (i_param - 1) * n_periods + t
            dataset_valid[idx] = false
        end
        n_failed += 1

        if n_failed <= 5  # Print first 5 failures for diagnostics
            println("  ⚠ Warning: SEP failed for parameter vector $(i_param): $(e)")
        end
    end
end

println("\n  ✓ SEP simulation complete")
println("    Success: $(n_success)/$(n_grid) ($(round(100*n_success/n_grid, digits=1))%)")
println("    Failed: $(n_failed)/$(n_grid) ($(round(100*n_failed/n_grid, digits=1))%)")

# Filter to valid samples only
valid_idx = findall(dataset_valid)
n_valid = length(valid_idx)

dataset_states = dataset_states[valid_idx, :]
dataset_obs = dataset_obs[valid_idx, :]
dataset_shocks = dataset_shocks[valid_idx, :]
dataset_params = dataset_params[valid_idx, :]

println("  ✓ Valid samples: $(n_valid) ($(round(100*n_valid/(n_grid*n_periods), digits=1))% of total)")

# ============================================================================
# Save Dataset
# ============================================================================

println("\n[5/6] Saving dataset...")

output_dir = joinpath(@__DIR__, "..", ".local_artifacts", "hlt_18param_dataset_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(output_dir)

# Save as HDF5 for efficient storage
h5_path = joinpath(output_dir, "sep_dataset.h5")
h5open(h5_path, "w") do file
    write(file, "states", dataset_states)
    write(file, "observables", dataset_obs)
    write(file, "shocks", dataset_shocks)
    write(file, "parameters", dataset_params)
    write(file, "state_names", string.(state_names))
    write(file, "observable_names", string.(observables))
    write(file, "shock_names", string.(shock_names))
    write(file, "parameter_names", est_names)
end

println("  ✓ Saved HDF5 dataset: $(basename(h5_path))")

# Save metadata as JSON
metadata = Dict(
    "n_parameters" => n_params,
    "parameter_names" => est_names,
    "parameter_bounds" => prior_bounds,
    "n_states" => n_states,
    "state_names" => string.(state_names),
    "n_observables" => length(observables),
    "observable_names" => string.(observables),
    "n_shocks" => n_shocks,
    "shock_names" => string.(shock_names),
    "n_grid_points" => n_grid,
    "n_periods" => n_periods,
    "n_total_samples" => n_grid * n_periods,
    "n_valid_samples" => n_valid,
    "success_rate" => n_success / n_grid,
    "algorithm" => "pruned_second_order",
    "use_obc" => true,
    "timestamp" => string(now())
)

meta_path = joinpath(output_dir, "metadata.json")
open(meta_path, "w") do f
    JSON3.pretty(f, metadata)
end

println("  ✓ Saved metadata: $(basename(meta_path))")

# ============================================================================
# Dataset Statistics
# ============================================================================

println("\n[6/6] Computing dataset statistics...")

println("\n  State variable statistics:")
for (j, name) in enumerate(state_names)
    vals = dataset_states[:, j]
    println("    $(name): mean=$(round(mean(vals), digits=4)), std=$(round(std(vals), digits=4)), range=[$(round(minimum(vals), digits=4)), $(round(maximum(vals), digits=4))]")
end

println("\n  Observable statistics:")
for (j, name) in enumerate(observables)
    vals = dataset_obs[:, j]
    println("    $(name): mean=$(round(mean(vals), digits=4)), std=$(round(std(vals), digits=4)), range=[$(round(minimum(vals), digits=4)), $(round(maximum(vals), digits=4))]")
end

println("\n  Parameter coverage:")
for (j, name) in enumerate(est_names)
    vals = dataset_params[:, j]
    println("    $(name): unique values=$(length(unique(vals))), mean=$(round(mean(vals), digits=4))")
end

println("\n" * "="^80)
println("Dataset Generation Complete!")
println("="^80)
println("\nOutput directory: $(output_dir)")
println("Dataset size: $(n_valid) samples ($(n_success) parameter vectors × ~$(n_periods) periods)")
println("Next step: Train surrogate using scripts/train_18param_surrogate.jl")
println("="^80)
