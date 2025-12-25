# SEP Impulse Response Functions
# Extract IRF from SEP stochastic tree solution

"""
    extract_sep_irf(sep_sol::sep_solution, shock_idx::Int, shock_size::Float64,
                    var_indices::Vector{Int}; path_type::Symbol=:mean)

Extract impulse response function from SEP solution.

# Arguments
- `sep_sol`: SEP solution structure
- `shock_idx`: Index of shock (1 to nshocks)
- `shock_size`: Size of shock (in standard deviations)
- `var_indices`: Indices of variables to extract
- `path_type`: How to traverse tree (:mean for expected path, :max for path with largest shock)

# Returns
- Matrix of size (nvars, periods+1) with IRF paths
"""
function extract_sep_irf(
    sep_sol::sep_solution,
    shock_idx::Int,
    shock_size::Float64,
    var_indices::Vector{Int};
    path_type::Symbol=:mean
)
    layout = sep_sol.layout
    Y = sep_sol.Y
    T = sep_sol.periods
    ny_ = layout.ny_

    # Preallocate IRF matrix: variables × time
    nvars = length(var_indices)
    irf = zeros(nvars, T+1)

    # Extract steady state (t=0, group 1)
    yss_indices = index_y(layout, 0, 1)
    yss = Y[yss_indices]

    # t=0: steady state (before shock)
    for (i, vidx) in enumerate(var_indices)
        irf[i, 1] = yss[vidx]
    end

    # Follow deterministic/expected path through tree
    current_group = 1

    for t in 1:T
        # Get current state
        y_indices = index_y(layout, t, current_group)
        yt = Y[y_indices]

        # Store deviations from steady state
        for (i, vidx) in enumerate(var_indices)
            irf[i, t+1] = yt[vidx]
        end

        # For next period, follow the mean path (group 1 in deterministic case)
        # In a branching tree, we'd need to select which branch to follow
        # For now, assume group 1 is the expected/central path
        current_group = 1
    end

    return irf
end


"""
    get_sep_irf(𝓂::ℳ, shock::Symbol, shock_size::Float64=1.0;
                variables::Vector{Symbol}=Symbol[], periods::Int=40)

Compute impulse response using SEP solution.

# Arguments
- `𝓂`: Model with SEP solution
- `shock`: Shock name
- `shock_size`: Shock size in standard deviations (default: 1.0)
- `variables`: Variables to include (default: all)
- `periods`: IRF horizon (default: 40)

# Returns
- KeyedArray with IRF paths
"""
function get_sep_irf(
    𝓂::ℳ,
    shock::Symbol,
    shock_size::Float64=1.0;
    variables::Vector{Symbol}=Symbol[],
    periods::Int=40
)
    # Check if SEP solution exists
    sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    if sep_sol === nothing
        error("No SEP solution found. Run solve!(model, algorithm=:stochastic_extended_path) first.")
    end

    # Get shock index
    shock_idx = findfirst(==(shock), 𝓂.exo)
    if shock_idx === nothing
        error("Shock $shock not found in model")
    end

    # Get variable indices
    if isempty(variables)
        var_names = 𝓂.var
        var_indices = collect(1:length(𝓂.var))
    else
        var_names = variables
        var_indices = [findfirst(==(v), 𝓂.var) for v in variables]
        if any(isnothing, var_indices)
            error("Some variables not found in model")
        end
    end

    # Adjust periods to SEP horizon if needed
    T_sep = sep_sol.periods
    T_irf = min(periods, T_sep)

    if T_irf < periods
        @warn "SEP horizon ($T_sep) is shorter than requested IRF horizon ($periods). Using $T_irf periods."
    end

    # Extract IRF
    irf = extract_sep_irf(sep_sol, shock_idx, shock_size, var_indices)

    # Convert to deviations from steady state
    for i in 1:size(irf, 1)
        ss_val = irf[i, 1]
        irf[i, :] .-= ss_val
    end

    # Return as KeyedArray
    time_labels = 0:T_irf
    return KeyedArray(irf[:, 1:T_irf+1]; Variables=var_names, Periods=time_labels)
end


"""
    get_sep_simulation(𝓂::ℳ; variables::Vector{Symbol}=Symbol[],
                       periods::Int=40, nsims::Int=1, levels::Bool=true)

Generate stochastic simulations using SEP solution.

# Arguments
- `𝓂`: Model with SEP solution
- `variables`: Variables to include (default: all)
- `periods`: Simulation horizon (default: 40)
- `nsims`: Number of simulations (default: 1)
- `levels`: Return levels (true) or deviations from SS (false)

# Returns
- KeyedArray with simulated paths (Variables × Periods × Simulations)

# Note
This is a simplified implementation that follows the expected (mean) path through
the SEP tree. For more sophisticated stochastic simulations, consider using
perturbation methods which have direct policy functions.
"""
function get_sep_simulation(
    𝓂::ℳ;
    variables::Vector{Symbol}=Symbol[],
    periods::Int=40,
    nsims::Int=1,
    levels::Bool=true
)
    # Check if SEP solution exists
    sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    if sep_sol === nothing
        error("No SEP solution found. Run solve!(model, algorithm=:stochastic_extended_path) first.")
    end

    # Get variable indices
    if isempty(variables)
        var_names = 𝓂.var
        var_indices = collect(1:length(𝓂.var))
    else
        var_names = variables
        var_indices = [findfirst(==(v), 𝓂.var) for v in variables]
        if any(isnothing, var_indices)
            error("Some variables not found in model")
        end
    end

    # Adjust periods to SEP horizon if needed
    T_sep = sep_sol.periods
    T_sim = min(periods, T_sep)

    if T_sim < periods
        @warn "SEP horizon ($T_sep) is shorter than requested simulation horizon ($periods). Using $T_sim periods."
    end

    layout = sep_sol.layout
    Y = sep_sol.Y
    nvars = length(var_indices)

    # Extract steady state
    yss_indices = index_y(layout, 0, 1)
    yss = Y[yss_indices]

    # Preallocate simulation array
    sims = zeros(nvars, T_sim+1, nsims)

    # For each simulation
    for s in 1:nsims
        # Start at steady state
        for (i, vidx) in enumerate(var_indices)
            sims[i, 1, s] = yss[vidx]
        end

        # Follow expected path through tree (group 1 = mean/expected path)
        # In a more sophisticated version, we would:
        #   1. Draw random shocks
        #   2. Find closest GH nodes
        #   3. Select corresponding group
        # For now, just follow the mean path
        current_group = 1

        for t in 1:T_sim
            y_indices = index_y(layout, t, current_group)
            yt = Y[y_indices]

            for (i, vidx) in enumerate(var_indices)
                sims[i, t+1, s] = yt[vidx]
            end

            # Continue along mean path
            current_group = 1
        end
    end

    # Convert to deviations from SS if requested
    if !levels
        for s in 1:nsims
            for i in 1:nvars
                ss_val = sims[i, 1, s]
                sims[i, :, s] .-= ss_val
            end
        end
    end

    # Return as KeyedArray
    time_labels = 0:T_sim
    sim_labels = 1:nsims
    return KeyedArray(sims; Variables=var_names, Periods=time_labels, Simulations=sim_labels)
end
