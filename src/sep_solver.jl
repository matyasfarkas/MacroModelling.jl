# Stochastic Extended Path (SEP) Solver for MacroModelling.jl
# Implements global nonlinear solution method with Gauss-Hermite quadrature

using SparseArrays, LinearAlgebra, ForwardDiff

"""
Configuration options for SEP solver.
"""
struct SEPSolverOptions
    periods::Int          # Time horizon T
    order::Int            # Branching order L
    nnodes::Int          # GH nodes per shock dimension
    maxit::Int           # Max Newton iterations
    tol::Float64         # Convergence tolerance
    verbose::Bool        # Print iteration info
    shock_scale::Float64 # Scale factor for GH nodes
    sparse_tree::Bool    # Use fishbone sparse tree (default: false for full tree)
    deterministic_shocks::Union{Matrix{Float64}, Nothing}  # NEW: T×dε deterministic shock sequence or nothing

    function SEPSolverOptions(;
        periods=20,
        order=1,
        nnodes=3,
        maxit=80,
        tol=1e-7,
        verbose=true,
        shock_scale=1.0,
        sparse_tree=false,
        deterministic_shocks=nothing  # NEW
    )
        # Validation
        if !isnothing(deterministic_shocks)
            @assert size(deterministic_shocks, 1) == periods "Deterministic shock sequence must have $periods rows (got $(size(deterministic_shocks, 1)))"
            @assert size(deterministic_shocks, 2) >= 1 "Deterministic shock sequence must have at least 1 column"
        end
        new(periods, order, nnodes, maxit, tol, verbose, shock_scale, sparse_tree, deterministic_shocks)
    end
end

"""
Layout describing SEP branching tree structure.
"""
struct SEPLayout
    T::Int                    # Periods
    Lbr::Int                  # Branching order
    K::Int                    # Total GH nodes (full tree: nnodes^dε)
    G::Vector{Int}            # Groups at each time
    voff::Vector{Int}         # Variable offsets
    eoff::Vector{Int}         # Shock offsets
    ny_::Int                  # Number of variables
    dε::Int                   # Number of shocks
    sparse::Bool              # Whether to use fishbone sparse tree (default: false for full tree)
    m::Int                    # Number of GH nodes per dimension for sparse tree (typically m=3 or 5)
end

# Constructor for backward compatibility (default to full tree)
function SEPLayout(T::Int, Lbr::Int, K::Int, G::Vector{Int}, voff::Vector{Int},
                   eoff::Vector{Int}, ny_::Int, dε::Int)
    SEPLayout(T, Lbr, K, G, voff, eoff, ny_, dε, false, 0)
end

# Tree navigation functions
groups_at(layout::SEPLayout, t::Int) = layout.G[t+1]
index_y(layout::SEPLayout, t::Int, g::Int) = layout.voff[t+1] + (g-1)*layout.ny_ .+ (1:layout.ny_)
row_range(layout::SEPLayout, t::Int, g::Int) = (layout.ny_*(sum(layout.G[2:t])+g-1)+1):(layout.ny_*(sum(layout.G[2:t])+g))

function parent_group(layout::SEPLayout, t::Int, g::Int)
    (t == 0) && return 1
    (t > layout.Lbr) && return g
    return div(g-1, layout.K) + 1
end

function child_groups(layout::SEPLayout, t::Int, g::Int)
    if t == 0 return 1:layout.K end
    if t >= layout.Lbr
        # Return K copies of the same group to integrate over K shock realizations
        return [g for _ in 1:layout.K]
    end
    return (g-1)*layout.K .+ (1:layout.K)
end

"""
Fishbone sparse tree navigation functions.
For sparse tree: only the trunk (group 1 at each t) branches.
Side branches are deterministic continuations.
"""

function is_trunk_node(layout::SEPLayout, t::Int, g::Int)
    !layout.sparse && return true  # Full tree: all nodes can branch
    return g == 1  # Sparse tree: only trunk branches
end

function parent_group_sparse(layout::SEPLayout, t::Int, g::Int)
    !layout.sparse && return parent_group(layout, t, g)  # Fall back to full tree

    (t == 0) && return 1
    (t > layout.Lbr) && return g  # After branching period, stay in same group

    # In branching period: trunk stays trunk, side branches follow their parent
    if g == 1
        return 1  # Trunk's parent is always trunk
    else
        # Side branch: came from trunk at previous period
        # Group g at time t corresponds to shock dimension h and node index k
        # We need to maintain the mapping
        return 1  # All side branches come from trunk
    end
end

function child_groups_sparse(layout::SEPLayout, t::Int, g::Int)
    !layout.sparse && return child_groups(layout, t, g)  # Fall back to full tree

    if t == 0
        # t=0: trunk branches into trunk + H*(m-1) side branches
        # Group numbering: 1=trunk, 2...(1+H*(m-1))=side branches
        return 1:(1 + layout.dε * (layout.m - 1))
    end

    if t >= layout.Lbr
        # After branching: integrate over shocks but stay in same group
        # Return copies for integration
        return [g for _ in 1:layout.m]
    end

    # In branching period (1 ≤ t < Lbr):
    if g == 1
        # Trunk branches into trunk + side branches
        return 1:(1 + layout.dε * (layout.m - 1))
    else
        # Side branch: deterministic continuation (no branching)
        # Return same group repeated for shock integration
        return [g for _ in 1:layout.m]
    end
end

"""
Get shock dimension and node index from side branch group number.
For sparse tree, groups are organized as:
- g=1: trunk (zero-shock path)
- g=2 to 1+H*(m-1): side branches, one for each (shock_dim, non-zero node) pair

Returns: (shock_dim, node_index) or nothing for trunk
"""
function get_shock_from_group(layout::SEPLayout, g::Int)
    !layout.sparse && error("get_shock_from_group only valid for sparse tree")

    if g == 1
        return nothing  # Trunk
    else
        # Map group to (h, k)
        # Groups 2 to 1+H*(m-1) correspond to H dimensions × (m-1) non-zero nodes
        branch_idx = g - 1  # 1-indexed to 0-indexed (branch_idx = 1, 2, ..., H*(m-1))
        h = div(branch_idx - 1, layout.m - 1) + 1  # Shock dimension (1 to H)
        k_offset = mod(branch_idx - 1, layout.m - 1)  # Offset from zero node (0 to m-2)
        k = k_offset + 2  # Node index (2 to m), skipping k=1 which is zero/trunk
        return (h, k)
    end
end

"""
Get child group index for a given shock realization in sparse tree.
For trunk at time t, when shock dimension h realizes node k:
- If k=1 (zero node): child stays on trunk (group 1)
- If k≠1: child goes to side branch for that (h,k) pair
"""
function get_child_group_for_shock(layout::SEPLayout, h::Int, k::Int)
    !layout.sparse && error("get_child_group_for_shock only valid for sparse tree")

    if k == 1
        return 1  # Zero-shock node stays on trunk
    else
        # Side branch: group = 1 + (h-1)*(m-1) + (k-1)
        # This maps (h,k) with k∈{2,...,m} to groups {2,...,1+H*(m-1)}
        return 1 + (h-1)*(layout.m - 1) + (k - 1)
    end
end

"""
Get shock realization vector for a side branch group.
Side branches follow deterministic paths with one specific shock.
"""
function get_shock_for_branch(layout::SEPLayout, g::Int, X::Matrix{Float64})
    !layout.sparse && error("get_shock_for_branch only valid for sparse tree")

    shock_info = get_shock_from_group(layout, g)
    if isnothing(shock_info)
        # Trunk: zero shocks
        return zeros(layout.dε)
    else
        h, k = shock_info
        # Extract shock from sparse node matrix
        # For sparse tree, X has H*m columns
        # Column index = (h-1)*m + k
        shock_idx = (h-1)*layout.m + k
        return X[:, shock_idx]
    end
end

"""
Gauss-Hermite tensor product nodes and weights.
"""
function gh_tensor_nodes_weights(nnodes::Int, dim::Int)
    # 1D Gauss-Hermite nodes/weights
    if nnodes == 1
        x1d = [0.0]; w1d = [√π]
    elseif nnodes == 3
        x1d = [-√3, 0.0, √3]
        w1d = [π/6, 2π/3, π/6]
    elseif nnodes == 5
        x1d = [-√(5+2√(10/7)), -√(5-2√(10/7)), 0.0, √(5-2√(10/7)), √(5+2√(10/7))]
        w1d = [π/30*(322-13√70)/900, π/30*(322+13√70)/900, 128π/(225*30),
               π/30*(322+13√70)/900, π/30*(322-13√70)/900]
    else
        error("Only nnodes ∈ {1,3,5} supported")
    end

    # Tensor product
    K = nnodes^dim
    X = zeros(dim, K)
    W = zeros(K)

    for k in 1:K
        w_prod = 1.0
        idx = k - 1
        for d in 1:dim
            local_idx = mod(idx, nnodes) + 1
            X[d, k] = x1d[local_idx]
            w_prod *= w1d[local_idx]
            idx = div(idx, nnodes)
        end
        W[k] = w_prod
    end

    # Normalize weights
    W ./= sum(W)

    return X, W
end

"""
Transform GH nodes by shock covariance.
"""
function transform_nodes!(X::Matrix{Float64}, Σ::Matrix{Float64})
    L = cholesky(Σ).L
    X .= L * X
    nothing
end

"""
Get 1D Gauss-Hermite nodes and weights for sparse tree.
Returns nodes and weights for a single dimension.
"""
function gh_nodes_1d(nnodes::Int)
    if nnodes == 1
        return [0.0], [√π]
    elseif nnodes == 3
        return [-√3, 0.0, √3], [π/6, 2π/3, π/6]
    elseif nnodes == 5
        x = [-√(5+2√(10/7)), -√(5-2√(10/7)), 0.0, √(5-2√(10/7)), √(5+2√(10/7))]
        w = [π*(322-13√70)/900, π*(322+13√70)/900, 128π/225,
             π*(322+13√70)/900, π*(322-13√70)/900]
        return x, w
    else
        error("Unsupported nnodes=$nnodes for sparse tree. Use 1, 3, or 5.")
    end
end

"""
Build shock nodes and weights for sparse fishbone tree.
For sparse tree, we use monomial rule: H*m nodes instead of tensor product m^H.

Returns:
- X_sparse: (dε × H*m) matrix, each column is a shock realization
- W_sparse: (H*m,) vector of weights
- shock_map: Dict mapping (shock_dim h, node_index k) → column in X_sparse
"""
function build_sparse_shock_nodes(m::Int, dε::Int, Σ::Matrix{Float64})
    # Get 1D nodes and weights
    x1d, w1d = gh_nodes_1d(m)

    # For sparse tree: H dimensions × m nodes per dimension
    n_nodes = dε * m

    X_sparse = zeros(dε, n_nodes)
    W_sparse = zeros(n_nodes)

    # Build monomial rule: vary one shock at a time
    for h in 1:dε  # For each shock dimension
        for k in 1:m  # For each node in that dimension
            idx = (h-1)*m + k
            # All shocks at zero except dimension h
            X_sparse[h, idx] = x1d[k]
            # Weight is 1D weight normalized by number of dimensions
            W_sparse[idx] = w1d[k] / dε
        end
    end

    # Transform by shock covariance
    transform_nodes!(X_sparse, Σ)

    # Normalize weights
    W_sparse ./= sum(W_sparse)

    # Create mapping (h, k) → column index
    shock_map = Dict{Tuple{Int,Int}, Int}()
    for h in 1:dε, k in 1:m
        shock_map[(h,k)] = (h-1)*m + k
    end

    return X_sparse, W_sparse, shock_map
end

"""
Estimate non-zeros in sparse Jacobian (for pre-allocation).
"""
function estimate_nnz(layout::SEPLayout, T::Int, ny_::Int, K::Int, Lbr::Int)
    nnz_estimate = 0
    for t in 1:T
        Gt = groups_at(layout, t)
        if t <= Lbr
            nnz_per_group = ny_ * ny_ * (2 + K)
        else
            nnz_per_group = 3 * ny_ * ny_
        end
        nnz_estimate += Gt * nnz_per_group
    end
    return Int(ceil(1.5 * nnz_estimate))
end

"""
Deterministic perfect foresight path solver.

Solves for the deterministic path given a sequence of shocks. This is equivalent
to Dynare's extended_path with deterministic innovations.

# Arguments
- `𝓂`: MacroModelling model
- `parameters`: Vector of parameter values
- `opts`: SEPSolverOptions with deterministic_shocks specified
- `initial_guess`: Optional warm start vector
- `yss`: Steady state vector
- `SS_and_pars`: Combined steady state and calibration parameters for Jacobian

Returns: (flag, Y, layout, err) - same structure as stochastic SEP
"""
function solve_deterministic_path(
    𝓂::ℳ,
    parameters::Vector{Float64},
    opts::SEPSolverOptions,
    initial_guess::Union{Nothing,Vector{Float64}},
    yss::Vector{Float64},
    SS_and_pars::Vector{Float64}
)
    # Extract dimensions
    ny_ = length(𝓂.var)
    dε = length(𝓂.exo)
    T = opts.periods

    opts.verbose && @info "Deterministic path solver" T=T ny_=ny_ dε=dε

    # Compute Jacobian blocks
    J_compressed = calculate_jacobian(parameters, SS_and_pars, 𝓂)
    timings = 𝓂.timings

    # Extract Jacobian blocks [∇₊, ∇₀, ∇₋, ∇ₑ] for y_{t+1}, y_t, y_{t-1}, ε_t
    ∇₊ = zeros(Float64, ny_, ny_)
    ∇₊[:, timings.future_not_past_and_mixed_idx] = J_compressed[:, 1:timings.nFuture_not_past_and_mixed]

    ∇₀ = J_compressed[:, timings.nFuture_not_past_and_mixed .+ (1:timings.nVars)]

    ∇₋ = zeros(Float64, ny_, ny_)
    ∇₋[:, timings.past_not_future_and_mixed_idx] = J_compressed[:, timings.nFuture_not_past_and_mixed + timings.nVars .+ (1:timings.nPast_not_future_and_mixed)]

    ∇ₑ = J_compressed[:, timings.nFuture_not_past_and_mixed + timings.nVars + timings.nPast_not_future_and_mixed .+ (1:timings.nExo)]

    # Create deterministic layout (no branching - single path)
    G = ones(Int, T+2)  # One group at each period
    voff = Vector{Int}(undef, T+3)
    acc = 1
    for t in 0:T
        voff[t+1] = acc
        acc += ny_
    end
    voff[T+2] = acc
    voff[T+3] = acc

    eoff = Vector{Int}(undef, T+1)
    for t in 1:T
        eoff[t] = 1 + (t-1)*dε
    end

    layout = SEPLayout(T, 0, 1, G, voff, eoff, ny_, dε, false, 1)

    # Initialize solution vector Y: [y₀, y₁, ..., yT]
    nvars_total = ny_ * (T + 1)

    if !isnothing(initial_guess) && length(initial_guess) == nvars_total
        Y = copy(initial_guess)
        opts.verbose && @info "Using provided initial guess"
    else
        # Initialize at steady state
        Y = repeat(yss, T + 1)
        opts.verbose && @info "Initializing at steady state"
    end

    # Newton solver
    neq = ny_ * T  # T periods of equations (y₀ is initial condition, fixed)
    nnz_est = 3 * ny_ * ny_ * T  # Tridiagonal block structure
    rows = Vector{Int}(undef, nnz_est)
    cols = Vector{Int}(undef, nnz_est)
    vals = Vector{Float64}(undef, nnz_est)
    R = zeros(neq)  # Declare outside loop so it's available after loop ends
    err = Inf  # Initialize error

    for it in 1:opts.maxit
        fill!(R, 0.0)  # Reuse R vector
        nnz_count = 0

        # Build stacked system for t=1,...,T
        for t in 1:T
            # Get states at t-1, t, t+1
            y_lag = view(Y, (t-1)*ny_ .+ (1:ny_))
            y_cur = view(Y, t*ny_ .+ (1:ny_))
            y_fwd = (t < T) ? view(Y, (t+1)*ny_ .+ (1:ny_)) : yss  # Terminal: return to SS

            # Get deterministic shock at period t
            ε_t = view(opts.deterministic_shocks, t, :)

            # Deviations from steady state
            Δy_lag = y_lag - yss
            Δy_cur = y_cur - yss
            Δy_fwd = y_fwd - yss

            # Equilibrium condition (first-order approximation)
            r = ∇₊ * Δy_fwd + ∇₀ * Δy_cur + ∇₋ * Δy_lag + ∇ₑ * ε_t

            # Store residual
            rr = (t-1)*ny_ .+ (1:ny_)
            R[rr] .= r

            # Fill Jacobian blocks
            # Note: Column indices are for [y₁, ..., yT] (excluding fixed y₀)
            # So we map: y₁ → cols 1:ny_, y₂ → cols ny_+1:2*ny_, etc.

            # J[rr, y_{t-1}] = ∇₋ (for t>1; for t=1, y₀ is fixed so no derivative)
            if t > 1
                c_lag = (t-2)*ny_ .+ (1:ny_)  # y_{t-1} columns (already excludes y₀)
                for i in 1:ny_, j in 1:ny_
                    v = ∇₋[i, j]
                    if abs(v) > 1e-16
                        nnz_count += 1
                        rows[nnz_count] = rr[i]
                        cols[nnz_count] = c_lag[j]
                        vals[nnz_count] = v
                    end
                end
            end

            # J[rr, y_t] = ∇₀
            c_cur = (t-1)*ny_ .+ (1:ny_)  # y_t columns (t=1 maps to cols 1:ny_)
            for i in 1:ny_, j in 1:ny_
                v = ∇₀[i, j]
                if abs(v) > 1e-16
                    nnz_count += 1
                    rows[nnz_count] = rr[i]
                    cols[nnz_count] = c_cur[j]
                    vals[nnz_count] = v
                end
            end

            # J[rr, y_{t+1}] = ∇₊
            if t < T
                c_fwd = t*ny_ .+ (1:ny_)  # y_{t+1} columns
                for i in 1:ny_, j in 1:ny_
                    v = ∇₊[i, j]
                    if abs(v) > 1e-16
                        nnz_count += 1
                        rows[nnz_count] = rr[i]
                        cols[nnz_count] = c_fwd[j]
                        vals[nnz_count] = v
                    end
                end
            end
        end

        # Build sparse Jacobian
        # Jacobian maps: Δ[y₁, ..., yT] → R (residuals)
        # Dimensions: (ny_×T) × (ny_×T) - excluding y₀ which is fixed
        ncols = ny_ * T
        J = sparse(view(rows, 1:nnz_count), view(cols, 1:nnz_count),
                   view(vals, 1:nnz_count), neq, ncols)

        # Check convergence
        err = maximum(abs, R)

        if it % 5 == 0 || it == 1 || err < opts.tol
            opts.verbose && @info "Deterministic path it=$it/$opts.maxit" max_res=err
        end

        if err < opts.tol
            opts.verbose && @info "✓ Deterministic path converged" iterations=it err=err
            return (flag=0, Y=Y, layout=layout, err=err)
        end

        # Newton step with regularization
        λ = 1e-8
        Δ = (J'*J + λ*I) \ (J'*(-R))

        # Adaptive damping
        α = err > 1e-3 ? 0.5 : (err > 1e-5 ? 0.7 : 1.0)

        # Apply update to y₁, ..., yT (keep y₀ fixed at steady state)
        Y[ny_+1:end] .+= α * Δ
    end

    # Did not converge
    err = maximum(abs, R)
    @warn "Deterministic path did not converge" max_iterations=opts.maxit final_err=err
    return (flag=1, Y=Y, layout=layout, err=err)
end

"""
Main SEP solver implementation.

# Arguments
- `𝓂`: MacroModelling model
- `parameters`: Vector of parameter values
- `opts`: SEPSolverOptions configuration (optional)
- `initial_guess`: Vector{Float64} from previous solution for warm start (optional)
  When solving over parameter grids, passing the previous solution as initial_guess
  dramatically speeds up convergence. Get this from result.Y of previous solve.

Returns: (flag, Y, layout, err) where:
- flag: 0=success, 1=max iterations, 2=domain violation
- Y: Solution vector containing all states (use this for next warm start)
- layout: SEPLayout describing tree structure
- err: Final residual norm
"""
function sep_solve_mm!(
    𝓂::ℳ,
    parameters::Vector{Float64};
    opts::SEPSolverOptions=SEPSolverOptions(),
    initial_guess::Union{Nothing,Vector{Float64}}=nothing
)
    # Get model dimensions
    ny_ = length(𝓂.var)
    dε = length(𝓂.exo)

    # Get steady state
    SS_result = get_steady_state(𝓂; parameters=parameters, return_variables_only=true, derivatives=false)

    # Get available keys from SS_result (may not include auxiliary variables)
    ss_keys = try
        axiskeys(SS_result, 1)
    catch
        Symbol[]  # If SS_result is not a KeyedArray
    end

    # Build yss vector - use steady state for available vars, or their definitions for auxiliary
    yss = Float64[]
    for var in 𝓂.var
        if var ∈ ss_keys
            push!(yss, Float64(SS_result(var)))
        else
            # For auxiliary variables not in SS_result, initialize to zero
            # They will be computed from the equilibrium conditions
            push!(yss, 0.0)
        end
    end

    # Get Jacobian
    SS = 𝓂.solution.non_stochastic_steady_state
    calib_pars = Float64[]
    if length(𝓂.calibration_equations) > 0
        # ss_keys already computed above

        for param in 𝓂.calibration_equations_parameters
            # Only add parameter if it exists in SS_result
            if param ∈ ss_keys
                push!(calib_pars, Float64(SS_result(param)))
            else
                # Parameter not in steady state, skip it
                # This can happen for flex-price variables in sticky-price models
                @debug "Calibration parameter $param not found in steady state, skipping"
            end
        end
    end
    SS_and_pars = vcat(SS, calib_pars)

    # DETERMINISTIC MODE BRANCHING: If deterministic shocks provided, use perfect foresight solver
    if !isnothing(opts.deterministic_shocks)
        opts.verbose && @info "Deterministic mode detected - using perfect foresight solver"
        return solve_deterministic_path(𝓂, parameters, opts, initial_guess, yss, SS_and_pars)
    end

    # STOCHASTIC MODE: Continue with existing SEP solver
    J_compressed = calculate_jacobian(parameters, SS_and_pars, 𝓂)
    timings = 𝓂.timings

    # Reconstruct full Jacobian blocks from compressed format
    # J_compressed has structure: [future_dynamic, all_current, past_dynamic, shocks]
    # We need: [all_future, all_current, all_past, shocks]

    ∇₊ = zeros(Float64, ny_, ny_)
    ∇₊[:, timings.future_not_past_and_mixed_idx] = J_compressed[:, 1:timings.nFuture_not_past_and_mixed]

    ∇₀ = J_compressed[:, timings.nFuture_not_past_and_mixed .+ (1:timings.nVars)]

    ∇₋ = zeros(Float64, ny_, ny_)
    ∇₋[:, timings.past_not_future_and_mixed_idx] = J_compressed[:, timings.nFuture_not_past_and_mixed + timings.nVars .+ (1:timings.nPast_not_future_and_mixed)]

    ∇ₑ = J_compressed[:, timings.nFuture_not_past_and_mixed + timings.nVars + timings.nPast_not_future_and_mixed .+ (1:timings.nExo)]

    if opts.verbose
        @info "Shock Jacobian ∇ₑ stats:" size=size(∇ₑ) norm=norm(∇ₑ) max_abs=maximum(abs.(∇ₑ)) nnz=count(x -> abs(x) > 1e-12, ∇ₑ)
        # Print ALL non-zero values
        println("  Non-zero elements of ∇ₑ:")
        for i in 1:size(∇ₑ, 1)
            for j in 1:size(∇ₑ, 2)
                if abs(∇ₑ[i, j]) > 1e-12
                    shock_name = length(𝓂.exo) >= j ? 𝓂.exo[j] : "shock$j"
                    println("    ∇ₑ[eq $i, $shock_name] = $(∇ₑ[i,j])")
                end
            end
        end
    end

    # Get shock covariance from model parameters
    # MacroModelling stores shock std devs as parameters named "z_{shock_name}"
    Σ = zeros(dε, dε)
    for (i, shock_name) in enumerate(𝓂.exo)
        # Look for parameter z_{shock_name}
        param_name = Symbol("z_", shock_name)
        param_idx = findfirst(==(param_name), 𝓂.parameters)

        if param_idx !== nothing
            σ = parameters[param_idx]
            Σ[i, i] = σ^2  # Variance = std^2
            if opts.verbose
                @info "Shock $shock_name: σ = $σ (from parameter $param_name)"
            end
        else
            # Fallback to small default if parameter not found
            @warn "Shock std parameter $param_name not found, using default 0.01"
            Σ[i, i] = 0.01^2
        end
    end

    if opts.verbose
        @info "Shock covariance matrix Σ:" Σ
    end

    # Setup GH quadrature
    T = opts.periods
    Lbr = max(opts.order, 0)
    m = opts.nnodes  # Nodes per dimension (for sparse tree)

    # Build shock nodes depending on sparse_tree option
    if opts.sparse_tree
        # Sparse tree: monomial rule (H*m nodes instead of m^H)
        X, W, shock_map = build_sparse_shock_nodes(m, dε, Σ)
        K = size(X, 2)  # K = dε * m for sparse tree
    else
        # Full tree: tensor product (m^H nodes)
        X, W = gh_tensor_nodes_weights(m, dε)
        K = size(X, 2)  # K = m^dε for full tree
        shock_map = nothing
    end

    if opts.shock_scale != 1.0
        X .*= opts.shock_scale
    end
    # Note: X already transformed by Σ in build_sparse_shock_nodes for sparse tree
    # For full tree, apply transformation here
    if !opts.sparse_tree && dε > 0
        transform_nodes!(X, Σ)
    end

    # Build tree structure
    G = Vector{Int}(undef, T+2)
    if opts.sparse_tree
        # Sparse tree: only trunk branches
        # Groups per period: 1 trunk + H*(m-1) side branches
        G_branch = 1 + dε*(m-1)  # Total groups that can exist
        for t in 0:T+1
            if t == 0
                G[t+1] = 1  # Just steady state
            elseif t <= Lbr
                # In branching region: trunk + side branches from all previous periods
                # Actually, for fishbone: same number of groups at each branching period
                G[t+1] = G_branch
            else
                # After branching: maintain same groups (no new branching)
                G[t+1] = G_branch
            end
        end
    else
        # Full tree: exponential branching
        for t in 0:T+1
            if t == 0
                G[t+1] = 1
            elseif t <= Lbr
                G[t+1] = K^t
            else
                G[t+1] = K^Lbr
            end
        end
    end

    voff = Vector{Int}(undef, T+3)
    acc = 1
    for t in 0:T
        voff[t+1] = acc
        acc += ny_ * G[t+1]
    end
    voff[T+2] = acc
    voff[T+3] = acc

    eoff = Vector{Int}(undef, T+1)
    accE = 1
    for t in 1:T
        eoff[t] = accE
        accE += dε * G[t+1]
    end

    layout = SEPLayout(T, Lbr, K, G, voff, eoff, ny_, dε, opts.sparse_tree, m)

    # Initialize solution
    # voff[T+2] is the next free index after all variables, so we need voff[T+2]-1 elements
    # But index_y can return voff[T+2] as max index, so we allocate voff[T+2] elements
    nvars_total = voff[T+2]

    if !isnothing(initial_guess)
        # Use provided initial guess (from previous solution)
        # This dramatically speeds up convergence when solving over parameter grids
        if length(initial_guess) != nvars_total
            @warn "Initial guess dimension ($(length(initial_guess))) doesn't match expected ($nvars_total). Using steady state instead."
            Y = zeros(nvars_total)
            for t in 0:T, g in 1:G[t+1]
                Y[index_y(layout, t, g)] .= yss
            end
        else
            Y = copy(initial_guess)
            opts.verbose && @info "Using provided initial guess for warm start"
        end
    else
        # Default: initialize at steady state
        Y = zeros(nvars_total)
        for t in 0:T, g in 1:G[t+1]
            Y[index_y(layout, t, g)] .= yss
        end
    end

    # Pre-allocate workspace
    neq = ny_ * sum(G[2:end-1])
    nnz_est = estimate_nnz(layout, T, ny_, K, Lbr)
    rows = Vector{Int}(undef, nnz_est)
    cols = Vector{Int}(undef, nnz_est)
    vals = Vector{Float64}(undef, nnz_est)
    R = zeros(neq)

    # Cache child groups (use sparse functions if sparse_tree)
    cgs_cache = Dict{Tuple{Int,Int}, Vector{Int}}()
    for t in 1:T, g in 1:G[t+1]
        if layout.sparse
            cgs_cache[(t,g)] = collect(child_groups_sparse(layout, t, g))
        else
            cgs_cache[(t,g)] = collect(child_groups(layout, t, g))
        end
    end

    # Workspace buffers
    r_sum = zeros(ny_)
    J_lag = zeros(ny_, ny_)
    J_cur = zeros(ny_, ny_)

    # Helper to get state
    function get_y(t, g)
        (t >= T+1) ? yss : view(Y, index_y(layout, t, g))
    end

    # Newton iterations
    for it in 1:opts.maxit
        fill!(R, 0.0)
        nnz_count = 0

        # Diagnostic: Check Y variation at t=1 on first iteration
        if it == 1 && opts.verbose && T >= 1
            y_idx_test = 1  # Test first variable
            println("\n  Diagnostic - Y values at t=1 for different groups (variable $y_idx_test):")
            for g in [1, 100, 500, 1000, 1175, 1500, 2000, min(2187, G[2])]
                if g <= G[2]
                    y_test = Y[index_y(layout, 1, g)[y_idx_test]]
                    println("    Group $g: Y[$y_idx_test] = $y_test, dev=$(y_test - yss[y_idx_test])")
                end
            end
        end

        for t in 1:T
            Gt = G[t+1]
            for g in 1:Gt
                # Use sparse parent navigation if sparse tree
                pg = layout.sparse ? parent_group_sparse(layout, t, g) : parent_group(layout, t, g)
                yl = view(Y, index_y(layout, t-1, pg))
                yc = view(Y, index_y(layout, t, g))
                cgs = cgs_cache[(t,g)]

                # Determine if this node branches
                node_branches = if layout.sparse
                    # Sparse: only trunk nodes branch in branching period
                    (t <= Lbr) && (g == 1) && (dε > 0)
                else
                    # Full tree: all nodes branch in branching period
                    (t <= Lbr) && (dε > 0)
                end

                if node_branches
                    # Branching node: expectation over child groups
                    fill!(r_sum, 0.0)
                    fill!(J_lag, 0.0)
                    fill!(J_cur, 0.0)

                    # Extract shock for current group g
                    if layout.sparse
                        # Sparse tree: trunk at t-1 experienced zero shock
                        ε_curr = zeros(dε)
                    else
                        # Full tree: extract shock from tensor product
                        k_shock = mod(g - 1, K) + 1
                        ε_curr = view(X, :, k_shock)
                    end

                    for (kidx, cg) in enumerate(cgs)
                        yl1 = get_y(t+1, cg)

                        # Deviations from steady state
                        Δy_lag = yl - yss
                        Δy_cur = yc - yss
                        Δy_fwd = yl1 - yss

                        # For sparse tree, extract shock that leads to this child group
                        if layout.sparse
                            # Sparse tree trunk branches: iterate over shock dimensions × nodes
                            # kidx maps to (h, k) via linear indexing
                            # For sparse tree X has H*m columns organized as [(h=1,k=1)...(h=1,k=m), (h=2,k=1)...(h=2,k=m), ...]
                            shock_idx = kidx
                            ε_to_child = view(X, :, shock_idx)
                            wk = W[shock_idx]
                        else
                            # Full tree: shock for parent group g
                            k_shock = mod(g - 1, K) + 1
                            ε_to_child = view(X, :, k_shock)
                            wk = W[kidx]
                        end

                        # Linear approximation: r = J*(y - yss) + shock term
                        r = ∇₊ * Δy_fwd + ∇₀ * Δy_cur + ∇₋ * Δy_lag + ∇ₑ * ε_to_child

                        # Jacobian blocks
                        J_lag_local = ∇₋
                        J_cur_local = ∇₀
                        J_fwd = ∇₊
                        r_sum .+= wk .* r
                        J_lag .+= wk .* J_lag_local
                        J_cur .+= wk .* J_cur_local

                        # Lead Jacobian entries
                        if t+1 <= T
                            rrng = row_range(layout, t, g)
                            c_rng = index_y(layout, t+1, cg)
                            for irow in 1:ny_, jcol in 1:ny_
                                v = wk * J_fwd[irow, jcol]
                                if abs(v) > 1e-16
                                    nnz_count += 1
                                    rows[nnz_count] = first(rrng) + irow - 1
                                    cols[nnz_count] = first(c_rng) + jcol - 1
                                    vals[nnz_count] = v
                                end
                            end
                        end
                    end

                    # Store residual and lag/current Jacobians
                    rr = row_range(layout, t, g)
                    R[rr] .= r_sum
                    c_lag = index_y(layout, t-1, pg)
                    c_cur = index_y(layout, t, g)

                    for irow in 1:ny_
                        r_gl = first(rr) + irow - 1
                        for jcol in 1:ny_
                            v = J_lag[irow, jcol]
                            if abs(v) > 1e-16
                                nnz_count += 1
                                rows[nnz_count] = r_gl
                                cols[nnz_count] = c_lag.start + jcol - 1
                                vals[nnz_count] = v
                            end
                        end
                        for jcol in 1:ny_
                            v = J_cur[irow, jcol]
                            if abs(v) > 1e-16
                                nnz_count += 1
                                rows[nnz_count] = r_gl
                                cols[nnz_count] = c_cur.start + jcol - 1
                                vals[nnz_count] = v
                            end
                        end
                    end
                else
                    # Non-branching node: deterministic (or sparse tree side branch)
                    cg = first(cgs)
                    yl1 = get_y(t+1, cg)

                    Δy_lag = yl - yss
                    Δy_cur = yc - yss
                    Δy_fwd = yl1 - yss

                    # For sparse tree side branches, include shock term
                    if layout.sparse && g > 1
                        # Side branch: extract fixed shock for this branch
                        ε_branch = get_shock_for_branch(layout, g, X)
                        r = ∇₊ * Δy_fwd + ∇₀ * Δy_cur + ∇₋ * Δy_lag + ∇ₑ * ε_branch
                    else
                        # Full tree non-branching or sparse trunk after branching period
                        r = ∇₊ * Δy_fwd + ∇₀ * Δy_cur + ∇₋ * Δy_lag
                    end

                    rr = row_range(layout, t, g)
                    R[rr] .= r
                    c_lag = index_y(layout, t-1, pg)
                    c_cur = index_y(layout, t, g)

                    if t+1 <= T
                        c_lea = index_y(layout, t+1, cg)
                        for irow in 1:ny_, jcol in 1:ny_
                            v = ∇₊[irow, jcol]
                            if abs(v) > 1e-16
                                nnz_count += 1
                                rows[nnz_count] = first(rr) + irow - 1
                                cols[nnz_count] = c_lea.start + jcol - 1
                                vals[nnz_count] = v
                            end
                        end
                    end

                    for irow in 1:ny_
                        r_gl = first(rr) + irow - 1
                        for jcol in 1:ny_
                            v = ∇₋[irow, jcol]
                            if abs(v) > 1e-16
                                nnz_count += 1
                                rows[nnz_count] = r_gl
                                cols[nnz_count] = c_lag.start + jcol - 1
                                vals[nnz_count] = v
                            end
                        end
                        for jcol in 1:ny_
                            v = ∇₀[irow, jcol]
                            if abs(v) > 1e-16
                                nnz_count += 1
                                rows[nnz_count] = r_gl
                                cols[nnz_count] = c_cur.start + jcol - 1
                                vals[nnz_count] = v
                            end
                        end
                    end
                end
            end
        end

        # Build sparse Jacobian
        J = sparse(view(rows, 1:nnz_count), view(cols, 1:nnz_count),
                   view(vals, 1:nnz_count), neq, length(Y))

        # Check current residual
        err = maximum(abs, R)

        # Solve Newton step with regularization
        λ = 1e-8
        Δ = (J'*J + λ*I) \ (J'*(-R))

        # Adaptive damping based on residual norm
        # Start aggressive, reduce if residual is large
        α = err > 1e-3 ? 0.5 : (err > 1e-5 ? 0.7 : 1.0)

        # Apply update with adaptive step size
        Y .+= α * Δ
        if opts.verbose && (it % 5 == 0 || it == 1)
            @info "SEP it=$it/$opts.maxit  max|res|=$err  step_norm=$(norm(Δ))"
        end

        if err < opts.tol
            opts.verbose && @info "✓ SEP converged (err=$err)"
            return (flag=0, Y=Y, layout=layout, err=err)
        end
    end

    # Did not converge
    @warn "SEP did not converge in $(opts.maxit) iterations (err=$(maximum(abs, R)))"
    return (flag=1, Y=Y, layout=layout, err=maximum(abs, R))
end
