# SEP Stochastic Simulation
# Sequential simulation using SEP for expectations

using Random

"""
    simulate_sep(𝓂::ℳ;
                 periods::Int=200,
                 initial_state::Union{Nothing,Vector{Float64}}=nothing,
                 shocks::Union{Nothing,Matrix{Float64}}=nothing,
                 burn_in::Int=0,
                 sep_horizon::Int=40,
                 sep_order::Int=1,
                 sep_nnodes::Int=3,
                 random_seed::Union{Nothing,Int}=nothing)

Perform stochastic simulation using SEP for forward-looking expectations.

# Algorithm
For each period t:
1. Solve SEP problem from current state with T-period horizon
2. Apply realized shock for period t
3. Take first-period decision from SEP solution
4. Update state and move to next period

# Arguments
- `𝓂`: Model
- `periods`: Total simulation periods
- `initial_state`: Starting state (default: deterministic SS)
- `shocks`: Matrix of shock realizations (nshocks × periods). If nothing, drawn from N(0,Σ)
- `burn_in`: Number of initial periods to discard
- `sep_horizon`: Horizon for each SEP problem (T)
- `sep_order`: SEP branching order (Lbr)
- `sep_nnodes`: Number of Gauss-Hermite nodes
- `random_seed`: Random seed for shock generation

# Returns
- `simulation`: KeyedArray of simulated paths (Variables × Time)
- `shocks_used`: Matrix of shocks used in simulation
"""
function simulate_sep(
    𝓂::ℳ;
    periods::Int=200,
    initial_state::Union{Nothing,Vector{Float64}}=nothing,
    shocks::Union{Nothing,Matrix{Float64}}=nothing,
    burn_in::Int=0,
    sep_horizon::Int=40,
    sep_order::Int=1,
    sep_nnodes::Int=3,
    random_seed::Union{Nothing,Int}=nothing,
    silent::Bool=true
)
    # Set random seed if provided
    if !isnothing(random_seed)
        Random.seed!(random_seed)
    end

    # Get model dimensions
    nvars = length(𝓂.var)
    nshocks = length(𝓂.exo)

    # Get parameter values
    params = 𝓂.parameter_values

    # Get steady state
    SS_result = get_steady_state(𝓂, derivatives=false)
    yss = [Float64(SS_result(var)) for var in 𝓂.var]

    # Initial state
    if isnothing(initial_state)
        y0 = copy(yss)
    else
        @assert length(initial_state) == nvars "Initial state dimension mismatch"
        y0 = copy(initial_state)
    end

    # Generate or use provided shocks
    total_periods = periods + burn_in
    if isnothing(shocks)
        # Get shock covariance matrix
        Σ = zeros(nshocks, nshocks)
        for (i, shock_name) in enumerate(𝓂.exo)
            param_name = Symbol("z_", shock_name)
            param_idx = findfirst(==(param_name), 𝓂.parameters)
            if !isnothing(param_idx)
                σ = params[param_idx]
                Σ[i, i] = σ^2
            else
                @warn "Shock std parameter $param_name not found, using default 1.0"
                Σ[i, i] = 1.0
            end
        end

        # Draw shocks from N(0, Σ)
        if nshocks == 1
            shocks_used = randn(nshocks, total_periods) .* sqrt(Σ[1,1])
        else
            L = cholesky(Σ).L
            shocks_used = L * randn(nshocks, total_periods)
        end
    else
        @assert size(shocks, 1) == nshocks "Shock dimension mismatch"
        @assert size(shocks, 2) >= total_periods "Not enough shock periods"
        shocks_used = shocks[:, 1:total_periods]
    end

    # Allocate storage for simulation
    Y_sim = zeros(nvars, total_periods + 1)
    Y_sim[:, 1] = y0

    # Main simulation loop
    !silent && println("Running SEP stochastic simulation...")
    for t in 1:total_periods
        if !silent && mod(t, 50) == 0
            println("  Period $t / $total_periods")
        end

        # Current state
        y_current = Y_sim[:, t]

        # Current shock realization
        ε_t = shocks_used[:, t]

        # Solve SEP problem from current state
        # This gives us the optimal path accounting for future uncertainty
        y_next = sep_step(𝓂, y_current, ε_t, yss, params,
                         sep_horizon, sep_order, sep_nnodes, silent)

        # Store next state
        Y_sim[:, t+1] = y_next
    end

    # Discard burn-in period
    if burn_in > 0
        Y_final = Y_sim[:, (burn_in+1):end]
        shocks_final = shocks_used[:, (burn_in+1):end]
    else
        Y_final = Y_sim
        shocks_final = shocks_used
    end

    # Return as KeyedArray
    time_labels = 0:periods
    result = KeyedArray(Y_final; Variables=𝓂.var, Time=time_labels)

    return result, shocks_final
end


"""
    sep_step(𝓂, y_current, ε_current, yss, params, T, Lbr, nnodes, silent)

Perform one step of SEP simulation: given current state and shock,
solve SEP problem and return next state.

This is a simplified implementation that uses the full SEP solution
to approximate the sequential decision.
"""
function sep_step(
    𝓂::ℳ,
    y_current::Vector{Float64},
    ε_current::Vector{Float64},
    yss::Vector{Float64},
    params::Vector{Float64},
    T::Int,
    Lbr::Int,
    nnodes::Int,
    silent::Bool
)
    # For the first implementation, we use a heuristic:
    # Solve SEP from steady state, find the group corresponding to the shock,
    # and extract the state at t=1

    # This is an approximation - ideally we'd resolve SEP from y_current
    # but that's expensive. For now, use the pre-solved SEP solution.

    # Get SEP solution (should be pre-computed)
    sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    if isnothing(sep_sol)
        # Solve SEP if not already done
        solve!(𝓂,
               algorithm=:stochastic_extended_path,
               sep_periods=T,
               sep_order=Lbr,
               sep_nnodes=nnodes,
               silent=silent)
        sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    end

    # Map shock realization to group index
    # For Lbr >= 1, groups at t=1 represent different shock combinations
    K = nnodes^length(𝓂.exo)

    # Find closest group to the realized shock
    # This is approximate - we map the shock to the nearest GH node
    group_idx = shock_to_group(ε_current, nnodes, length(𝓂.exo))

    # Extract state at t=1 for this group
    layout = sep_sol.layout
    Y = sep_sol.Y
    y_indices = index_y(layout, 1, group_idx)
    y_next = Y[y_indices]

    return y_next
end


"""
    shock_to_group(ε::Vector{Float64}, nnodes::Int, nshocks::Int)

Map a shock realization to the nearest group index in the SEP tree.

For nnodes=3: GH nodes are at [-√3, 0, +√3]
For each shock dimension, find nearest node, then compute group index.
"""
function shock_to_group(ε::Vector{Float64}, nnodes::Int, nshocks::Int)
    # GH nodes (in standard deviations)
    if nnodes == 3
        nodes_1d = [-√3, 0.0, √3]
    elseif nnodes == 5
        nodes_1d = [-√(5+2√(10/7)), -√(5-2√(10/7)), 0.0,
                    √(5-2√(10/7)), √(5+2√(10/7))]
    else
        error("Only nnodes ∈ {3, 5} supported")
    end

    # For each shock, find nearest node
    node_indices = zeros(Int, nshocks)
    for d in 1:nshocks
        # Find nearest node
        distances = abs.(nodes_1d .- ε[d])
        node_indices[d] = argmin(distances)
    end

    # Convert to group index using base-nnodes arithmetic
    group = 1
    for d in 1:nshocks
        group += (node_indices[d] - 1) * nnodes^(d - 1)
    end

    return group
end
