# SEP Impulse Response Functions
# Extract IRF from SEP stochastic tree solution

using Statistics

function sep_irf_shock_std(𝓂::ℳ, shock::Symbol; warn_missing::Bool=true)
    param_name = Symbol("z_", shock)
    param_idx = findfirst(==(param_name), 𝓂.parameters)
    if param_idx === nothing
        if warn_missing
            @warn "Shock std parameter $param_name not found, using default 1.0"
        end
        return 1.0
    end
    return 𝓂.parameter_values[param_idx]
end

function sep_irf_shock_scale(
    𝓂::ℳ,
    shock::Symbol,
    shock_size::Float64;
    shock_scaling::Symbol=:none,
    negative_shock::Bool=false
)
    shock_scale = negative_shock ? -abs(shock_size) : shock_size
    if shock_scaling == :none
        return shock_scale
    elseif shock_scaling == :parameter
        return shock_scale * sep_irf_shock_std(𝓂, shock)
    end
    error("Unknown shock_scaling=$shock_scaling. Use :none or :parameter.")
end

function sep_irf_stochastic_state(𝓂::ℳ, algorithm::Symbol; silent::Bool=true)
    solve!(𝓂, algorithm = algorithm, dynamics = true, silent = silent)
    if algorithm == :first_order
        nss = 𝓂.solution.non_stochastic_steady_state
        return nss[1:length(𝓂.var)]
    elseif algorithm == :third_order
        return copy(𝓂.solution.perturbation.third_order.stochastic_steady_state)
    elseif algorithm == :pruned_third_order
        return copy(𝓂.solution.perturbation.pruned_third_order.stochastic_steady_state)
    elseif algorithm == :pruned_second_order
        return copy(𝓂.solution.perturbation.pruned_second_order.stochastic_steady_state)
    else
        return copy(𝓂.solution.perturbation.second_order.stochastic_steady_state)
    end
end

function sep_irf_extract_path(
    sep_sol::sep_solution,
    var_indices::Vector{Int},
    periods::Int
)
    layout = sep_sol.layout
    ny_ = layout.ny_
    nvars = length(var_indices)
    path = zeros(nvars, periods + 1)

    for t in 0:periods
        y_t = sep_sol.Y[layout.voff[t + 1] .+ (1:ny_)]
        for (i, vidx) in enumerate(var_indices)
            path[i, t + 1] = y_t[vidx]
        end
    end

    return path
end

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
    nnodes = sep_sol.nnodes
    order = sep_sol.order

    # Preallocate IRF matrix: variables × time
    nvars = length(var_indices)
    irf = zeros(nvars, T+1)

    # Extract steady state (t=0, group 1 = no shock)
    yss_indices = index_y(layout, 0, 1)
    yss = Y[yss_indices]

    # t=0: steady state (before shock)
    for (i, vidx) in enumerate(var_indices)
        irf[i, 1] = yss[vidx]
    end

    # SEP uses tensor product of GH nodes across all shocks
    # Groups are indexed using base-nnodes arithmetic:
    # k = 1 + sum_{d=0}^{nshocks-1} (node_d - 1) * nnodes^d
    # where node_d is the node index (1 to nnodes) for shock d

    # For nnodes=3, the GH nodes are [-√3, 0, +√3]
    # Middle node (index 2) represents zero shock
    zero_node_idx = (nnodes + 1) ÷ 2

    # Determine target node for the shocked variable
    if shock_size > 0
        target_node_idx = nnodes  # Highest GH node (most positive)
    elseif shock_size < 0
        target_node_idx = 1       # Lowest GH node (most negative)
    else
        target_node_idx = zero_node_idx  # Middle node (zero)
    end

    # Calculate group index where:
    # - shock shock_idx is at target_node_idx
    # - all other shocks are at zero_node_idx (middle node)
    nshocks = layout.dε  # Number of shocks

    # Build base-nnodes representation
    idx = 0
    for d in 1:nshocks
        if d == shock_idx
            # This is the shocked variable
            idx += (target_node_idx - 1) * nnodes^(d - 1)
        else
            # All other shocks at zero node
            idx += (zero_node_idx - 1) * nnodes^(d - 1)
        end
    end

    # Convert to 1-based group index
    shock_group_t1 = idx + 1

    # For nnodes=3, GH nodes are at [-√3, 0, √3] standard deviations
    # We need to scale the IRF to match the requested shock_size
    # Get the actual shock magnitude at the target node
    if nnodes == 3
        gh_nodes_1d = [-√3, 0.0, √3]
        actual_shock_std = abs(gh_nodes_1d[target_node_idx])
    elseif nnodes == 5
        gh_nodes_1d = [-√(5+2√(10/7)), -√(5-2√(10/7)), 0.0, √(5-2√(10/7)), √(5+2√(10/7))]
        actual_shock_std = abs(gh_nodes_1d[target_node_idx])
    else
        # Default: assume node spacing roughly √nnodes
        actual_shock_std = abs(shock_size)  # No scaling
    end

    # Scaling factor to convert from GH node magnitude to requested shock_size
    scale_factor = abs(shock_size) / max(actual_shock_std, 1e-10)

    # Diagnostic
    @info "SEP IRF Scaling" nnodes=nnodes shock_size=shock_size actual_shock_std=actual_shock_std scale_factor=scale_factor

    # Navigate through tree
    # For SEP with limited branching (Lbr), groups continue along the same path
    # after the branching period. We follow the shocked group throughout.
    current_group = shock_group_t1

    for t in 1:T
        # Get state at time t (following the shocked path)
        y_indices = index_y(layout, t, current_group)
        yt = Y[y_indices]

        # Store values
        for (i, vidx) in enumerate(var_indices)
            irf[i, t+1] = yt[vidx]
        end

        # For t < order (branching region), we would navigate to child groups
        # For t >= order, groups map to themselves (no branching)
        # With Lbr=1, we stay in the same shocked group for all t >= 1
        if t < order
            # In branching region: would need to choose which child to follow
            # For IRF, we continue with the shocked group
            # (child_groups returns K copies of the same group for t >= Lbr)
        end
    end

    return irf, scale_factor
end


"""
    get_sep_irf_tree(𝓂::ℳ, shock::Symbol, shock_size::Float64=1.0;
                     variables::Vector{Symbol}=Symbol[], periods::Int=40,
                     shock_scaling::Symbol=:none)

Compute impulse response by extracting path from pre-solved SEP tree.

NOTE: This is the old tree-extraction method. The new simulation-based method
is preferred and available via get_sep_irf().

# Arguments
- `𝓂`: Model with SEP solution
- `shock`: Shock name
- `shock_size`: Shock size in shock units (default: 1.0)
- `variables`: Variables to include (default: all)
- `periods`: IRF horizon (default: 40)
- `shock_scaling`: `:none` for shock units, `:parameter` to scale by `z_<shock>`

# Returns
- KeyedArray with IRF paths
"""
function get_sep_irf_tree(
    𝓂::ℳ,
    shock::Symbol,
    shock_size::Float64=1.0;
    variables::Vector{Symbol}=Symbol[],
    periods::Int=40,
    shock_scaling::Symbol=:none
)
    # Check if SEP solution exists
    sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    if sep_sol === nothing
        error("No SEP solution found. Run solve!(model, algorithm=:stochastic_extended_path) first.")
    end

    # Get shock index (full exo list)
    shock_idx = findfirst(==(shock), 𝓂.exo)
    if shock_idx === nothing
        error("Shock $shock not found in model")
    end
    stochastic_idx = findall(x -> !contains(x, "ᵒᵇᶜ"), string.(𝓂.exo))
    shock_idx_stoch = findfirst(==(shock_idx), stochastic_idx)
    if shock_idx_stoch === nothing
        error("Shock $shock is not a stochastic shock (OBC shocks are excluded from SEP trees).")
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

    shock_size_effective = shock_scaling == :parameter ?
        shock_size * sep_irf_shock_std(𝓂, shock) :
        shock_size

    # Extract IRF
    irf, scale_factor = extract_sep_irf(sep_sol, shock_idx_stoch, shock_size_effective, var_indices)

    # Convert to deviations from steady state and apply scaling
    for i in 1:size(irf, 1)
        ss_val = irf[i, 1]
        irf[i, :] .-= ss_val
        irf[i, :] .*= scale_factor  # Scale to match requested shock_size
    end

    # Return as KeyedArray
    time_labels = 0:T_irf
    return KeyedArray(irf[:, 1:T_irf+1]; Variables=var_names, Periods=time_labels)
end


"""
    get_sep_irf_funnel(𝓂::ℳ, shock::Symbol, shock_size::Float64=1.0;
                       variables::Vector{Symbol}=Symbol[],
                       periods::Int=40,
                       sep_periods::Union{Nothing,Int}=nothing,
                       sep_order::Union{Nothing,Int}=nothing,
                       sep_nnodes::Union{Nothing,Int}=nothing,
                       sep_maxit::Int=80,
                       sep_tol::Float64=1e-7,
                       sep_sparse_tree::Union{Nothing,Bool}=nothing,
                       initial_state::Union{Nothing,Vector{Float64}}=nothing,
                       sss_algorithm::Symbol=:second_order,
                       baseline::Symbol=:zero_shock,
                       shock_scaling::Symbol=:none,
                       negative_shock::Bool=false,
                       silent::Bool=true)

Compute SEP IRF using Dynare-style funnel baseline (tt - ts).

This method:
1. Solves SEP once at the full branching order for the shocked path (tt).
2. Builds the funnel baseline (ts) by decreasing the branching order and
   appending the first-period solution at each order, with a final
   deterministic continuation.
3. Returns IRF = tt - ts (levels), with t=0 included.

Set `baseline=:steady_state` to return tt deviations from the initial state
instead of the funnel baseline.

Set `baseline=:zero_shock` to return tt minus the SEP path computed with
the same settings and zero deterministic shocks (shocked minus unshocked).

`shock_scaling=:none` interprets `shock_size` in shock units (consistent with
`get_irf`); use `shock_scaling=:parameter` to multiply by `z_<shock>`.
"""
function get_sep_irf_funnel(
    𝓂::ℳ,
    shock::Symbol,
    shock_size::Float64=1.0;
    variables::Vector{Symbol}=Symbol[],
    periods::Int=40,
    sep_periods::Union{Nothing,Int}=nothing,
    sep_order::Union{Nothing,Int}=nothing,
    sep_nnodes::Union{Nothing,Int}=nothing,
    sep_maxit::Int=80,
    sep_tol::Float64=1e-7,
    sep_sparse_tree::Union{Nothing,Bool}=nothing,
    initial_state::Union{Nothing,Vector{Float64}}=nothing,
    sss_algorithm::Symbol=:second_order,
    baseline::Symbol=:zero_shock,
    shock_scaling::Symbol=:none,
    negative_shock::Bool=false,
    silent::Bool=true
)
    sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    if sep_sol === nothing && (sep_periods === nothing || sep_order === nothing || sep_nnodes === nothing || sep_sparse_tree === nothing)
        error("No SEP solution found. Run solve!(model, algorithm=:stochastic_extended_path) or provide sep_* options.")
    end

    if sep_periods === nothing
        sep_periods = sep_sol.periods
    end
    if sep_order === nothing
        sep_order = sep_sol.order
    end
    if sep_nnodes === nothing
        sep_nnodes = sep_sol.nnodes
    end
    if sep_sparse_tree === nothing
        sep_sparse_tree = hasproperty(sep_sol, :layout) ? sep_sol.layout.sparse : true
    end

    if sep_periods < periods
        @warn "SEP horizon ($sep_periods) is shorter than requested IRF horizon ($periods). Using $sep_periods periods."
        periods = sep_periods
    end

    shock_idx = findfirst(==(shock), 𝓂.exo)
    if shock_idx === nothing
        error("Shock $shock not found in model")
    end

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

    if isnothing(initial_state)
        initial_state = sep_irf_stochastic_state(𝓂, sss_algorithm; silent = silent)
    else
        @assert length(initial_state) == length(𝓂.var) "initial_state must have length $(length(𝓂.var))"
    end

    shock_value = sep_irf_shock_scale(𝓂, shock, shock_size;
                                      shock_scaling = shock_scaling,
                                      negative_shock = negative_shock)
    nshocks = length(𝓂.exo)

    shock_sequence = zeros(sep_periods, nshocks)
    shock_sequence[1, shock_idx] = shock_value

    solve!(𝓂,
           algorithm = :stochastic_extended_path,
           sep_periods = sep_periods,
           sep_order = sep_order,
           sep_nnodes = sep_nnodes,
           sep_maxit = sep_maxit,
           sep_tol = sep_tol,
           sep_sparse_tree = sep_sparse_tree,
           sep_initial_state = initial_state,
           sep_deterministic_shocks = shock_sequence,
           silent = silent)

    sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
    tt_path = sep_irf_extract_path(sep_sol, var_indices, periods)

    if baseline == :steady_state
        irf = tt_path .- initial_state[var_indices]
    elseif baseline == :zero_shock
        shock_sequence = zeros(sep_periods, nshocks)

        solve!(𝓂,
               algorithm = :stochastic_extended_path,
               sep_periods = sep_periods,
               sep_order = sep_order,
               sep_nnodes = sep_nnodes,
               sep_maxit = sep_maxit,
               sep_tol = sep_tol,
               sep_sparse_tree = sep_sparse_tree,
               sep_initial_state = initial_state,
               sep_deterministic_shocks = shock_sequence,
               silent = silent)

        sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
        zero_path = sep_irf_extract_path(sep_sol, var_indices, periods)
        irf = tt_path .- zero_path
    elseif baseline == :funnel
        ts_path = zeros(length(var_indices), periods + 1)
        ts_path[:, 1] = initial_state[var_indices]

        ny = length(𝓂.var)
        cursor = 2
        prev = copy(initial_state)

        for order in sep_order:-1:0
            shock_sequence = zeros(sep_periods, nshocks)
            if order == sep_order
                shock_sequence[1, shock_idx] = shock_value
                periods_step = 1
            elseif order == 0
                periods_step = periods
            else
                periods_step = 1
            end

            if order == 0
                initial_guess = repeat(initial_state, sep_periods + 1)
                initial_guess[1:ny] .= prev
            else
                initial_guess = nothing
            end

            solve!(𝓂,
                   algorithm = :stochastic_extended_path,
                   sep_periods = sep_periods,
                   sep_order = order,
                   sep_nnodes = sep_nnodes,
                   sep_maxit = sep_maxit,
                   sep_tol = sep_tol,
                   sep_sparse_tree = sep_sparse_tree,
                   sep_initial_guess = initial_guess,
                   sep_initial_state = prev,
                   sep_deterministic_shocks = shock_sequence,
                   silent = silent)

            sep_sol = 𝓂.solution.perturbation.stochastic_extended_path
            layout = sep_sol.layout

            if order == 0
                for t in 1:periods_step
                    if cursor > periods + 1
                        break
                    end
                    y_t = sep_sol.Y[layout.voff[t + 1] .+ (1:layout.ny_)]
                    ts_path[:, cursor] = y_t[var_indices]
                    prev = y_t
                    cursor += 1
                end
            else
                if cursor <= periods + 1
                    y_t = sep_sol.Y[layout.voff[2] .+ (1:layout.ny_)]
                    ts_path[:, cursor] = y_t[var_indices]
                    cursor += 1
                    prev = y_t
                end
            end

            if cursor > periods + 1
                break
            end
        end

        irf = tt_path .- ts_path
    else
        error("Unknown baseline $baseline. Use :funnel, :steady_state, or :zero_shock.")
    end

    time_labels = 0:periods
    return KeyedArray(irf; Variables=var_names, Periods=time_labels)
end


"""
    get_sep_irf(𝓂::ℳ, shock::Symbol, shock_size::Float64=1.0;
                variables::Vector{Symbol}=Symbol[],
                periods::Int=40,
                burn_in::Int=100,
                random_seed::Union{Nothing,Int}=nothing,
                method::Symbol=:simulation,
                baseline::Symbol=:zero_shock,
                sep_periods::Union{Nothing,Int}=nothing,
                sep_order::Union{Nothing,Int}=nothing,
                sep_nnodes::Union{Nothing,Int}=nothing,
                sep_maxit::Int=80,
                sep_tol::Float64=1e-7,
                sep_sparse_tree::Union{Nothing,Bool}=nothing,
                initial_state::Union{Nothing,Vector{Float64}}=nothing,
                sss_algorithm::Symbol=:second_order,
                shock_scaling::Symbol=:none,
                negative_shock::Bool=false,
                silent::Bool=true)

Compute SEP IRFs using either a Dynare-style funnel baseline or a stochastic
simulation baseline.

Use `method=:funnel` to compute IRF = tt − ts (funnel baseline).
Use `method=:simulation` to compute IRF = shocked path − baseline path
from stochastic simulations (legacy behavior).

# Arguments
- `𝓂`: Model with SEP solution
- `shock`: Shock name
- `shock_size`: Shock size in shock units (default: 1.0)
- `variables`: Variables to include (default: all)
- `periods`: IRF horizon (default: 40)
- `burn_in`: Burn-in periods to reach SSS (simulation method only)
- `random_seed`: Random seed for simulation reproducibility (simulation only)
- `method`: `:funnel` or `:simulation`
- `baseline`: For `method=:funnel`:
    - `:funnel` returns IRF = tt − ts (Dynare SEP funnel baseline)
    - `:steady_state` returns tt − initial_state (absolute deviation from SSS)
    - `:zero_shock` returns tt − zero_shock_path (shock vs unshocked SEP path)
- `sep_periods`: SEP horizon (T) for each SEP solve (funnel method)
- `sep_order`: Branching order (Lbr) for SEP (funnel method)
- `sep_nnodes`: Gauss–Hermite nodes per shock dimension (funnel method)
- `sep_maxit`: SEP max Newton iterations (funnel method)
- `sep_tol`: SEP convergence tolerance (funnel method)
- `sep_sparse_tree`: Use fishbone sparse tree (funnel method)
- `initial_state`: Initial state for SEP funnel baseline (defaults to SSS)
- `sss_algorithm`: Perturbation order for SSS (`:second_order`, `:third_order`, etc.)
- `shock_scaling`: `:none` for shock units, `:parameter` to scale by `z_<shock>`
- `negative_shock`: Apply shock with negative sign when true
- `silent`: Suppress progress messages

# Returns
- `KeyedArray` with IRF paths (Variables × Periods)

# Notes
- `shock_size` is interpreted in shock units (consistent with `get_irf`). Use
  `shock_scaling=:parameter` to multiply by `z_<shock>` for Dynare-style stderr scaling.
- When `method=:funnel`, `sep_periods` should be at least `periods` or the IRF will be truncated.
- `baseline=:zero_shock` is recommended when comparing to `get_irf`, which returns
  deviations from an unshocked baseline path.
"""
function get_sep_irf(
    𝓂::ℳ,
    shock::Symbol,
    shock_size::Float64=1.0;
    variables::Vector{Symbol}=Symbol[],
    periods::Int=40,
    burn_in::Int=100,
    random_seed::Union{Nothing,Int}=nothing,
    method::Symbol=:simulation,
    baseline::Symbol=:zero_shock,
    sep_periods::Union{Nothing,Int}=nothing,
    sep_order::Union{Nothing,Int}=nothing,
    sep_nnodes::Union{Nothing,Int}=nothing,
    sep_maxit::Int=80,
    sep_tol::Float64=1e-7,
    sep_sparse_tree::Union{Nothing,Bool}=nothing,
    initial_state::Union{Nothing,Vector{Float64}}=nothing,
    sss_algorithm::Symbol=:second_order,
    shock_scaling::Symbol=:none,
    negative_shock::Bool=false,
    silent::Bool=true
)
    if method == :funnel
        return get_sep_irf_funnel(
            𝓂, shock, shock_size;
            variables = variables,
            periods = periods,
            sep_periods = sep_periods,
            sep_order = sep_order,
            sep_nnodes = sep_nnodes,
            sep_maxit = sep_maxit,
            sep_tol = sep_tol,
            sep_sparse_tree = sep_sparse_tree,
            initial_state = initial_state,
            sss_algorithm = sss_algorithm,
            baseline = baseline,
            shock_scaling = shock_scaling,
            negative_shock = negative_shock,
            silent = silent
        )
    elseif method != :simulation
        error("Unknown method $method. Use :funnel or :simulation.")
    end

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

    # Get number of shocks
    nshocks = length(𝓂.exo)

    !silent && println("Computing SEP IRF using stochastic simulation...")
    !silent && println("  Burn-in: $burn_in periods")
    !silent && println("  IRF horizon: $periods periods")

    # Step 1: Run burn-in to reach stochastic steady state
    !silent && println("\n1. Running burn-in simulation to reach SSS...")
    sim_burnin, shocks_burnin = simulate_sep(𝓂,
                                               periods = periods,
                                               burn_in = burn_in,
                                               sep_horizon = sep_sol.periods,
                                               sep_order = sep_sol.order,
                                               sep_nnodes = sep_sol.nnodes,
                                               shock_scaling = shock_scaling,
                                               random_seed = random_seed,
                                               silent = silent)

    # Extract stochastic steady state from burn-in
    # Take mean over last portion of burn-in (e.g., last 20% of periods)
    # This gives a better estimate of the ergodic mean
    sample_periods = max(20, burn_in ÷ 5)  # Use last 20% of burn-in, minimum 20 periods
    sss_sample = sim_burnin[:, (end-sample_periods+1):end]
    # Extract underlying matrix before computing mean (sss_sample is KeyedArray)
    y_sss = vec(mean(parent(parent(sss_sample)), dims=2))

    !silent && println("  ✓ Reached stochastic steady state")

    # Step 2: From SSS, run baseline simulation (zero shocks)
    !silent && println("\n2. Running baseline simulation (zero shocks)...")
    shocks_baseline = zeros(nshocks, periods)
    sim_baseline, _ = simulate_sep(𝓂,
                                    periods = periods,
                                    initial_state = y_sss,
                                    shocks = shocks_baseline,
                                    burn_in = 0,
                                    sep_horizon = sep_sol.periods,
                                    sep_order = sep_sol.order,
                                    sep_nnodes = sep_sol.nnodes,
                                    shock_scaling = shock_scaling,
                                    silent = silent)

    # Step 3: From SSS, run shocked simulation (shock applied at t=1)
    !silent && println("\n3. Running shocked simulation...")

    # Create shock matrix: shock at t=1, zeros elsewhere
    shocks_shocked = zeros(nshocks, periods)
    shocks_shocked[shock_idx, 1] = sep_irf_shock_scale(𝓂, shock, shock_size;
                                                       shock_scaling = shock_scaling,
                                                       negative_shock = negative_shock)

    sim_shocked, _ = simulate_sep(𝓂,
                                   periods = periods,
                                   initial_state = y_sss,
                                   shocks = shocks_shocked,
                                   burn_in = 0,
                                   sep_horizon = sep_sol.periods,
                                   sep_order = sep_sol.order,
                                   sep_nnodes = sep_sol.nnodes,
                                   shock_scaling = shock_scaling,
                                   silent = silent)

    !silent && println("  ✓ Simulations complete")

    # Step 4: Compute IRF = shocked - baseline
    !silent && println("\n4. Computing IRF as difference...")
    irf = sim_shocked .- sim_baseline

    # Select variables
    if !isempty(variables)
        var_indices = [findfirst(==(v), axiskeys(irf, 1)) for v in variables]
        if any(isnothing, var_indices)
            error("Some variables not found in model")
        end
        irf = irf[var_indices, :]
        var_names = variables
    else
        var_names = collect(axiskeys(irf, 1))
    end

    !silent && println("  ✓ IRF computation complete\n")

    # Return as KeyedArray with proper labels
    # Extract underlying matrix since irf already has dimension names from simulate_sep
    time_labels = collect(axiskeys(irf, 2))
    irf_matrix = parent(parent(irf))  # Extract the underlying matrix
    return KeyedArray(irf_matrix; Variables=var_names, Periods=time_labels)
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
