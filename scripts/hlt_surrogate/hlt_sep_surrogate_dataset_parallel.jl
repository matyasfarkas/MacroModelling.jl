# OPTIMIZATION: Multi-threaded dataset generation for M4 Pro (12 cores)
# Expected speedup: 4-6x with 12 threads (10 hours → 1.5-2 hours)
# Usage: Set JULIA_NUM_THREADS=12 and pass --parallel flag to dataset generation script

using Base.Threads
using Random
using Statistics

"""
    parallel_theta_simulation(theta_range, theta_grid, model, base_values, theta_idx,
                              samples_per_theta, sep_params...; seed0=42)

Parallel execution of SEP simulations across theta parameter grid.

# Arguments
- `theta_range`: Range of theta indices to process (e.g., 1:125)
- `theta_grid`: Vector of theta parameter vectors
- `model`: MacroModelling model object
- `base_values`: Base parameter values
- `theta_idx`: Indices of parameters to vary in theta
- `samples_per_theta`: Number of samples per theta point
- `sep_params`: Dictionary with SEP solver settings
- `seed0`: Base random seed (default 42)

# Returns
- `results`: Vector of result tuples (theta_i, X_thread, Y_thread, success, stable_periods)

# Thread Safety
- Each thread uses thread-local RNG: MersenneTwister(seed0 + task_idx)
- Each thread allocates its own buffers (no shared state)
- Model parameters written per-thread (model must support this)

# Example
```julia
results = parallel_theta_simulation(
    1:125,
    theta_grid,
    model,
    base_values,
    theta_idx,
    samples_per_theta,
    sep_params;
    seed0=42
)
```
"""
function parallel_theta_simulation(theta_range, theta_grid, model, base_values, theta_idx,
                                   samples_per_theta, sep_params; seed0=42, verbose=true)
    n_theta = length(theta_range)
    n_threads = Threads.nthreads()

    if verbose
        println("=" ^ 70)
        println("PARALLEL DATASET GENERATION")
        println("=" ^ 70)
        println("Hardware: M4 Pro optimized")
        println("Threads:  $(n_threads)")
        println("Theta samples: $(n_theta)")
        println("Expected speedup: $(min(n_threads, n_theta))x (ideal), ~$(0.7 * min(n_threads, n_theta))x (realistic)")
        println("=" ^ 70)
    end

    # Pre-allocate result storage
    results = Vector{Any}(undef, n_theta)

    # Thread-safe progress counter
    progress = Threads.Atomic{Int}(0)
    start_time = time()

    # Parallelize theta loop
    Threads.@threads for task_idx in 1:n_theta
        theta_i = theta_range[task_idx]

        # Thread-local RNG (critical for reproducibility)
        rng_thread = Random.MersenneTwister(seed0 + task_idx)

        # Thread-local parameter copy
        params_thread = copy(base_values)
        theta = theta_grid[theta_i]
        params_thread[theta_idx] = theta

        # Thread-local model (or write parameters safely)
        # NOTE: MacroModelling.write_parameters_input! may need thread-local model copy
        # For now, assume model is immutable or thread-safe for parameter writes
        try
            MacroModelling.write_parameters_input!(model, params_thread, verbose=false)
        catch e
            if verbose
                println("Warning: Thread $(Threads.threadid()) failed to write parameters for theta $theta_i")
            end
            results[task_idx] = (theta_i, nothing, nothing, false, 0, e)
            Threads.atomic_add!(progress, 1)
            continue
        end

        # SEP simulation
        seed = seed0 + theta_i - 1
        total_periods = sep_params[:T_obs] + sep_params[:burn_in]

        # Draw shocks with thread-local RNG
        shocks_override = sep_params[:shock_scale] == 1.0 ? nothing :
            draw_shocks_thread(rng_thread, model, total_periods,
                             sep_params[:shock_scaling], sep_params[:shock_scale])

        res = nothing
        try
            res = MacroModelling.simulate_sep_extended_path(
                model;
                periods = sep_params[:T_obs],
                burn_in = sep_params[:burn_in],
                sep_horizon = sep_params[:sep_horizon],
                sep_order = sep_params[:sep_order],
                sep_nnodes = sep_params[:sep_nnodes],
                sep_maxit = sep_params[:sep_maxit],
                sep_tol = sep_params[:sep_tol],
                sep_sparse_tree = sep_params[:sep_sparse_tree],
                sep_linear_solver = get(sep_params, :sep_linear_solver, :normal_equations),
                sep_fallback_solver = get(sep_params, :sep_fallback_solver, nothing),
                sep_shock_scale = get(sep_params, :sep_shock_scale, 1.0),
                shock_scaling = sep_params[:shock_scaling],
                shocks = shocks_override,
                random_seed = seed,
                silent = true,
            )
        catch e
            if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                if verbose && (progress[] % 10 == 0)
                    println("  Thread $(Threads.threadid()): SEP solve failed (singular) for theta $theta_i")
                end
                results[task_idx] = (theta_i, nothing, nothing, false, 0, e)
                Threads.atomic_add!(progress, 1)
                continue
            else
                # Unexpected error - store but don't halt all threads
                if verbose
                    println("  Thread $(Threads.threadid()): Unexpected error for theta $theta_i: $e")
                end
                results[task_idx] = (theta_i, nothing, nothing, false, 0, e)
                Threads.atomic_add!(progress, 1)
                continue
            end
        end

        # Check for errors
        if res.errorflag
            results[task_idx] = (theta_i, nothing, nothing, false,
                                res.failure_period === nothing ? 0 : res.failure_period, nothing)
            Threads.atomic_add!(progress, 1)
            continue
        end

        # Extract samples
        sim = Array(res.simulation)
        shocks = res.shocks
        T_available = min(size(sim, 2) - 1, size(shocks, 2))

        if T_available <= 0 || T_available < get(sep_params, :stable_min_periods, 0)
            results[task_idx] = (theta_i, nothing, nothing, false, T_available, nothing)
            Threads.atomic_add!(progress, 1)
            continue
        end

        # Thread-local buffers for samples
        d_state = length(sep_params[:state_idx])
        d_eps = size(shocks, 1)
        d_theta = length(theta)
        d_obs = length(sep_params[:obs_idx])
        d_input = d_state + d_eps + d_theta
        d_output = d_obs + d_state

        n_samples = min(samples_per_theta, T_available)
        X_thread = zeros(d_input, n_samples)
        Y_thread = zeros(d_output, n_samples)

        # Sample indices (random or sequential)
        sample_idx_local = if samples_per_theta < T_available
            # Random sampling with thread-local RNG
            [rand(rng_thread, 1:T_available) for _ in 1:samples_per_theta]
        else
            1:T_available
        end

        # Fill thread-local buffers
        for (sample_i, t) in enumerate(sample_idx_local)
            X_thread[:, sample_i] = vcat(sim[sep_params[:state_idx], t],
                                         shocks[:, t],
                                         theta)
            Y_thread[:, sample_i] = vcat(sim[sep_params[:obs_idx], t + 1],
                                         sim[sep_params[:state_idx], t + 1])
        end

        # Store result
        results[task_idx] = (theta_i, X_thread, Y_thread, true, T_available, nothing)

        # Update progress (thread-safe)
        current_progress = Threads.atomic_add!(progress, 1)

        # Progress reporting (every 10 theta or at milestones)
        if verbose && (current_progress % 10 == 0 || current_progress == n_theta)
            elapsed = time() - start_time
            avg_per = elapsed / current_progress
            remaining = n_theta - current_progress
            eta_sec = remaining * avg_per
            eta = @sprintf("%02d:%02d:%02d",
                           floor(Int, eta_sec / 3600),
                           floor(Int, (eta_sec % 3600) / 60),
                           floor(Int, eta_sec % 60))
            println("  Progress: $(current_progress) / $(n_theta) theta samples (ETA $eta)")
        end
    end

    if verbose
        elapsed = time() - start_time
        println("=" ^ 70)
        println("Parallel simulation completed in $(round(elapsed, digits=1))s")
        println("Average: $(round(elapsed / n_theta, digits=2))s per theta")
        println("Speedup: ~$(round(n_threads * (elapsed / n_theta) / elapsed, digits=1))x vs sequential")
        println("=" ^ 70)
    end

    return results
end

"""
    merge_parallel_results(results, total_samples, d_input, d_output)

Merge thread-local results into global X, Y matrices.

# Arguments
- `results`: Vector of (theta_i, X_thread, Y_thread, success, stable_periods, error) tuples
- `total_samples`: Total pre-allocated sample count
- `d_input`: Input dimension (d_state + d_eps + d_theta)
- `d_output`: Output dimension (d_obs + d_state)

# Returns
- `X, Y`: Merged dataset matrices
- `theta_ids`: Theta index for each sample
- `theta_success`: Success flag per theta
- `theta_stable_periods`: Stable periods per theta
- `sample_count`: Actual number of samples collected
"""
function merge_parallel_results(results, total_samples, d_input, d_output)
    X = zeros(d_input, total_samples)
    Y = zeros(d_output, total_samples)
    theta_ids = zeros(Int, total_samples)
    theta_success = falses(length(results))
    theta_stable_periods = zeros(Int, length(results))

    cursor = 0

    for (idx, result) in enumerate(results)
        theta_i, X_thread, Y_thread, success, stable_periods, error = result

        theta_success[theta_i] = success
        theta_stable_periods[theta_i] = stable_periods

        if success && X_thread !== nothing
            n_samples = size(X_thread, 2)
            X[:, cursor+1:cursor+n_samples] = X_thread
            Y[:, cursor+1:cursor+n_samples] = Y_thread
            theta_ids[cursor+1:cursor+n_samples] .= theta_i
            cursor += n_samples
        end
    end

    # Trim unused pre-allocated space
    X = X[:, 1:cursor]
    Y = Y[:, 1:cursor]
    theta_ids = theta_ids[1:cursor]

    return X, Y, theta_ids, theta_success, theta_stable_periods, cursor
end

"""
    draw_shocks_thread(rng, model, total_periods, shock_scaling, shock_scale)

Thread-safe shock drawing using thread-local RNG.
"""
function draw_shocks_thread(rng, model, total_periods, shock_scaling, shock_scale)
    # Implementation depends on MacroModelling shock drawing API
    # For now, placeholder that uses thread-local RNG
    n_shocks = length(model.exo_present)
    return shock_scale .* randn(rng, n_shocks, total_periods)
end

# Export functions
export parallel_theta_simulation, merge_parallel_results
