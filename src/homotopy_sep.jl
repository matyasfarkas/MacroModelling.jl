"""
Homotopy Continuation Method for Stochastic Extended Path

This module implements σ-perturbation continuation: gradually increase shock
variance from deterministic (σ=0) to full stochastic (σ=1), using each solution
as warm start for the next step.

Physics motivation: Adiabatic evolution - slowly morph Hamiltonian so system
stays on solution manifold, never jumping to local minima at ZLB kinks.

Author: Based on user's σ-perturbation insight
Date: 2026-02-06
"""

using LinearAlgebra
using Statistics

"""
    solve_sep_at_noise_level(model; σ::Float64, initial_state, sep_params...)

Solve SEP with scaled shock covariance: Σ_effective = σ² * Σ_original

# Arguments
- `model`: MacroModelling model
- `σ`: Noise level in [0, 1]. σ=0 is deterministic, σ=1 is full stochastic
- `initial_state`: Starting state for simulation
- `sep_params...`: All standard SEP parameters (horizon, order, tol, etc.)

# Returns
- Result object from simulate_sep_extended_path
"""
function solve_sep_at_noise_level(
    model;
    σ::Float64 = 1.0,
    periods::Int = 1,
    initial_state = nothing,
    shocks = nothing,
    sep_horizon::Int = 20,
    sep_order::Int = 1,
    sep_nnodes::Int = 3,
    sep_tol::Float64 = 1e-2,
    sep_maxit::Int = 500,
    sep_sparse_tree::Bool = true,
    sep_yss = nothing,
    sep_lm_lambda::Float64 = 1.0,
    sep_accept_tol::Float64 = 0.25,
    burn_in::Int = 0,
    silent::Bool = true,
    verbose::Bool = false
)

    if verbose
        println("  Solving at σ = $(round(σ, digits=3))")
    end

    # Scale shocks if provided
    scaled_shocks = if shocks !== nothing
        σ * shocks
    else
        nothing
    end

    # For deterministic case (σ=0), force perfect foresight
    effective_order = (σ < 1e-10) ? 0 : sep_order

    # Call existing SEP solver
    res = simulate_sep_extended_path(
        model;
        periods = periods,
        burn_in = burn_in,
        initial_state = initial_state,
        shocks = scaled_shocks,
        sep_horizon = sep_horizon,
        sep_order = effective_order,
        sep_nnodes = sep_nnodes,
        sep_tol = sep_tol,
        sep_maxit = sep_maxit,
        sep_sparse_tree = sep_sparse_tree,
        sep_yss = sep_yss,
        sep_lm_lambda = sep_lm_lambda,
        sep_accept_tol = sep_accept_tol,
        silent = silent
    )

    return res
end


"""
    homotopy_sep(model; n_steps=10, adaptive=true, ...)

Homotopy continuation from deterministic (σ=0) to stochastic (σ=1) SEP solution.

# Algorithm
1. Solve deterministic SEP at σ=0 (perfect foresight - always works)
2. For σ ∈ {0.1, 0.2, ..., 1.0}:
   a. Use previous solution as initial guess (warm start)
   b. Solve SEP with scaled shocks: ε ~ N(0, σ²Σ)
   c. If failed and adaptive=true, subdivide interval and retry
3. Return final solution at σ=1

# Arguments
- `model`: MacroModelling model
- `n_steps`: Number of continuation steps (default 10)
- `adaptive`: Enable adaptive step subdivision on failure (default true)
- `max_retries`: Maximum subdivisions per step (default 3)
- `periods`: Simulation length (default 1 for single-period Markov transitions)
- `initial_state`: Starting state (default model steady state)
- `shocks`: Shock realizations to use (will be scaled by σ)
- `verbose`: Print detailed progress (default false)
- `sep_...`: All standard SEP parameters

# Returns
- `(success, final_result, σ_path_actual)`:
  - `success`: Boolean indicating convergence at σ=1
  - `final_result`: Result object at σ=1 (or last converged σ)
  - `σ_path_actual`: Actual σ values used (may differ from uniform if adaptive)
"""
function homotopy_sep(
    model;
    n_steps::Int = 10,
    adaptive::Bool = true,
    max_retries::Int = 3,
    periods::Int = 1,
    initial_state = nothing,
    shocks = nothing,
    sep_horizon::Int = 20,
    sep_order::Int = 1,
    sep_nnodes::Int = 3,
    sep_tol::Float64 = 1e-2,
    sep_maxit::Int = 500,
    sep_sparse_tree::Bool = true,
    sep_yss = nothing,
    sep_lm_lambda::Float64 = 1.0,
    sep_accept_tol::Float64 = 0.25,
    burn_in::Int = 0,
    verbose::Bool = false,
    silent::Bool = true
)

    # Generate initial σ path (uniform steps)
    σ_path = collect(LinRange(0.0, 1.0, n_steps + 1))
    σ_actual = Float64[0.0]  # Track actual path taken

    if verbose
        println("\n" * "="^80)
        println("HOMOTOPY CONTINUATION SEP")
        println("="^80)
        println("Steps: $n_steps (σ: 0 → 1)")
        println("Periods: $periods")
        println("Adaptive: $adaptive (max retries: $max_retries)")
        println("SEP: horizon=$sep_horizon, order=$sep_order, tol=$sep_tol")
        println("="^80 * "\n")
    end

    # Step 1: Solve deterministic case (σ=0)
    if verbose
        println("Step 0: Solving deterministic case (σ=0, perfect foresight)")
    end

    res_prev = solve_sep_at_noise_level(
        model;
        σ = 0.0,
        periods = periods,
        initial_state = initial_state,
        shocks = shocks,
        sep_horizon = sep_horizon,
        sep_order = 0,  # Force perfect foresight
        sep_nnodes = sep_nnodes,
        sep_tol = sep_tol,
        sep_maxit = sep_maxit,
        sep_sparse_tree = sep_sparse_tree,
        sep_yss = sep_yss,
        sep_lm_lambda = sep_lm_lambda,
        sep_accept_tol = sep_accept_tol,
        burn_in = burn_in,
        silent = silent,
        verbose = verbose
    )

    if res_prev.errorflag
        @warn "Homotopy failed at σ=0 (deterministic case). Cannot proceed."
        return (false, res_prev, σ_actual)
    end

    if verbose
        println("  ✓ Converged at σ=0.000\n")
    end

    # Step 2: Continuation loop
    i = 1
    while i < length(σ_path)
        σ_target = σ_path[i + 1]
        σ_prev = σ_actual[end]

        if verbose
            println("Step $i: σ = $(round(σ_prev, digits=3)) → $(round(σ_target, digits=3))")
        end

        # Try to solve at target σ
        success = false
        retry_count = 0
        σ_current = σ_target

        while !success && retry_count <= max_retries
            # Use last converged solution as initial state
            sim_prev = Array(res_prev.simulation)
            warm_start = if size(sim_prev, 2) >= 2
                sim_prev[:, 2]  # State at t=1
            else
                sim_prev[:, 1]  # Fallback to t=0
            end

            res_current = solve_sep_at_noise_level(
                model;
                σ = σ_current,
                periods = periods,
                initial_state = warm_start,  # Warm start from previous σ
                shocks = shocks,
                sep_horizon = sep_horizon,
                sep_order = sep_order,
                sep_nnodes = sep_nnodes,
                sep_tol = sep_tol,
                sep_maxit = sep_maxit,
                sep_sparse_tree = sep_sparse_tree,
                sep_yss = sep_yss,
                sep_lm_lambda = sep_lm_lambda,
                sep_accept_tol = sep_accept_tol,
                burn_in = burn_in,
                silent = silent,
                verbose = verbose
            )

            if !res_current.errorflag
                # Success!
                success = true
                res_prev = res_current
                push!(σ_actual, σ_current)

                if verbose
                    println("  ✓ Converged at σ=$(round(σ_current, digits=3))")
                end

                # If we subdivided, update path to continue from here
                if retry_count > 0 && adaptive
                    # We're at intermediate σ, need to reach original target
                    # Insert intermediate steps
                    if σ_current < σ_target
                        if verbose
                            println("    Subdivided: will continue toward σ=$(round(σ_target, digits=3))")
                        end
                        # Don't increment i - retry same target with new starting point
                        i -= 1
                    end
                end

            else
                # Failed - try adaptive subdivision
                if adaptive && retry_count < max_retries
                    σ_current = 0.5 * (σ_prev + σ_current)
                    retry_count += 1

                    if verbose
                        println("  ✗ Failed at σ=$(round(σ_path[i + 1], digits=3)), subdividing → σ=$(round(σ_current, digits=3)) (retry $retry_count/$max_retries)")
                    end
                else
                    # Give up
                    if verbose
                        println("  ✗ Failed at σ=$(round(σ_current, digits=3)) after $retry_count retries")
                        println("\n" * "="^80)
                        println("HOMOTOPY TERMINATED")
                        println("="^80)
                        println("Last converged σ: $(round(σ_actual[end], digits=3))")
                        println("Target σ: 1.0")
                        println("Success: $(σ_actual[end] >= 0.999 ? "✓" : "✗")")
                        println("="^80)
                    end

                    return (false, res_prev, σ_actual)
                end
            end
        end

        i += 1
    end

    # Success - reached σ=1
    if verbose
        println("\n" * "="^80)
        println("HOMOTOPY COMPLETED")
        println("="^80)
        println("Final σ: $(round(σ_actual[end], digits=3))")
        println("Total steps: $(length(σ_actual) - 1) (planned: $n_steps)")
        println("Status: ✓ SUCCESS")
        println("="^80)
    end

    return (true, res_prev, σ_actual)
end


"""
    homotopy_chained_trajectory(model, yss, T, rng; homotopy_params...)

Generate full trajectory by solving each period with homotopy continuation.

This combines the chained/shooting approach with homotopy robustness:
- For each period t, solve one-step transition with homotopy
- Use σ-continuation to handle difficult periods near ZLB

# Arguments
- `model`: MacroModelling model
- `yss`: Steady state vector
- `T`: Number of periods to simulate
- `rng`: Random number generator for shocks
- `homotopy_params...`: Parameters for homotopy_sep (n_steps, adaptive, etc.)

# Returns
- `(trajectory, shocks, success, periods_completed)`
"""
function homotopy_chained_trajectory(
    model,
    yss,
    T::Int,
    rng;
    n_steps::Int = 10,
    adaptive::Bool = true,
    sep_horizon::Int = 20,
    sep_order::Int = 1,
    sep_tol::Float64 = 1e-2,
    sep_maxit::Int = 500,
    sep_yss = nothing,
    verbose::Bool = false
)
    nvars = length(model.var)
    nexo = length(model.exo)

    # Storage
    trajectory = zeros(nvars, T + 1)
    shocks = zeros(nexo, T)

    trajectory[:, 1] = yss  # Start from steady state

    for t in 1:T
        # Draw shock for this period
        shock_t = randn(rng, nexo)
        shocks[:, t] = shock_t

        if verbose
            println("\n" * "-"^80)
            println("Period $t / $T")
            println("-"^80)
        end

        # Solve one period with homotopy
        current_state = trajectory[:, t]
        success, res, σ_path = homotopy_sep(
            model;
            n_steps = n_steps,
            adaptive = adaptive,
            periods = 1,
            initial_state = current_state,
            shocks = reshape(shock_t, :, 1),
            sep_horizon = sep_horizon,
            sep_order = sep_order,
            sep_tol = sep_tol,
            sep_maxit = sep_maxit,
            sep_yss = sep_yss,
            verbose = verbose
        )

        if !success
            if verbose
                println("✗ Trajectory generation failed at period $t")
            end
            return (trajectory, shocks, false, t)
        end

        # Extract next state
        sim = Array(res.simulation)
        trajectory[:, t + 1] = sim[:, 2]
    end

    if verbose
        println("\n" * "="^80)
        println("TRAJECTORY GENERATION COMPLETE")
        println("="^80)
        println("Periods: $T / $T ✓")
        println("="^80)
    end

    return (trajectory, shocks, true, T)
end

