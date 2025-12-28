# Stochastic Extended Path (SEP) Solver for MacroModelling.jl
# Implements global nonlinear solution method with Gauss-Hermite quadrature

using SparseArrays, LinearAlgebra, ForwardDiff, MacroTools

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

function replace_symbols_local(exprs, remap::Dict{Symbol,<:Any})
    MacroTools.postwalk(node -> node isa Symbol && haskey(remap, node) ? remap[node] : node, exprs)
end

function build_dynamic_residual_jacobian(𝓂::ℳ)
    dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
    dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
    dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
    dyn_exo_list = collect(reduce(union, 𝓂.dyn_exo_list))
    dyn_ss_list = Symbol.(string.(collect(reduce(union, 𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

    future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")), string.(dyn_future_list))
    present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")), string.(dyn_present_list))
    past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")), string.(dyn_past_list))
    exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")), string.(dyn_exo_list))
    stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")), string.(dyn_ss_list))

    vars_raw = vcat(
        dyn_future_list[indexin(sort(future), future)],
        dyn_present_list[indexin(sort(present), present)],
        dyn_past_list[indexin(sort(past), past)],
        dyn_exo_list[indexin(sort(exo), exo)]
    )

    pars_ext = vcat(𝓂.parameters, 𝓂.calibration_equations_parameters)
    parameters_and_SS = vcat(pars_ext, dyn_ss_list[indexin(sort(stst), stst)])

    np = length(parameters_and_SS)
    nv = length(vars_raw)
    Symbolics.@variables 𝔓[1:np] 𝔙[1:nv]

    parameter_dict = Dict{Symbol, Symbol}()
    back_to_array_dict = Dict{Symbolics.Num, Symbolics.Num}()

    for (i, v) in enumerate(parameters_and_SS)
        push!(parameter_dict, v => :($(Symbol("𝔓_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔓_$i"))), @__MODULE__) => 𝔓[i])
    end

    for (i, v) in enumerate(vars_raw)
        push!(parameter_dict, v => :($(Symbol("𝔙_$i"))))
        push!(back_to_array_dict, Symbolics.parse_expr_to_symbolic(:($(Symbol("𝔙_$i"))), @__MODULE__) => 𝔙[i])
    end

    calib_vars = Symbol[]
    calib_expr = []
    for v in 𝓂.calibration_equations_no_var
        push!(calib_vars, v.args[1])
        push!(calib_expr, v.args[2])
    end

    calib_replacements = Dict{Symbol,Any}()
    for (i, x) in enumerate(calib_vars)
        replacement = Dict(x => calib_expr[i])
        for ii in i+1:length(calib_vars)
            calib_expr[ii] = replace_symbols_local(calib_expr[ii], replacement)
        end
        push!(calib_replacements, x => calib_expr[i])
    end

    dyn_equations_sub = 𝓂.dyn_equations |>
        x -> replace_symbols_local.(x, Ref(calib_replacements)) |>
        x -> replace_symbols_local.(x, Ref(parameter_dict)) |>
        x -> Symbolics.parse_expr_to_symbolic.(x, Ref(@__MODULE__)) |>
        x -> Symbolics.substitute.(x, Ref(back_to_array_dict))

    _, resid_func = Symbolics.build_function(dyn_equations_sub, 𝔓, 𝔙,
                                            cse = true,
                                            skipzeros = true,
                                            parallel = Symbolics.SerialForm(),
                                            expression_module = @__MODULE__,
                                            expression = Val(false))

    jac_sym = Symbolics.sparsejacobian(dyn_equations_sub, 𝔙)
    _, jac_func = Symbolics.build_function(jac_sym, 𝔓, 𝔙,
                                           cse = true,
                                           skipzeros = true,
                                           parallel = Symbolics.SerialForm(),
                                           expression_module = @__MODULE__,
                                           expression = Val(false))

    resid_buffer = zeros(Float64, length(dyn_equations_sub))
    jac_buffer = jac_sym isa SparseMatrixCSC ? similar(jac_sym, Float64) : zeros(Float64, size(jac_sym))
    if jac_sym isa SparseMatrixCSC
        jac_buffer.nzval .= 0
    end

    return resid_func, jac_func, vars_raw, parameters_and_SS, resid_buffer, jac_buffer
end

function build_parameters_and_ss_values(parameters_and_SS, parameters, 𝓂::ℳ, yss, SS_result)
    vals = zeros(Float64, length(parameters_and_SS))
    ss_lookup = Dict{Symbol,Float64}()
    if SS_result isa KeyedArray
        for key in axiskeys(SS_result, 1)
            ss_lookup[Symbol(key)] = Float64(SS_result(key))
        end
    end

    param_index = Dict{Symbol,Int}(p => i for (i, p) in enumerate(𝓂.parameters))
    var_index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(𝓂.var))

    for (i, sym) in enumerate(parameters_and_SS)
        if haskey(param_index, sym)
            vals[i] = parameters[param_index[sym]]
        elseif sym in 𝓂.calibration_equations_parameters
            if haskey(ss_lookup, sym)
                vals[i] = ss_lookup[sym]
            else
                vals[i] = 0.0
            end
        elseif occursin("₍ₛₛ₎", string(sym))
            base = Symbol(replace(string(sym), r"₍ₛₛ₎$" => ""))
            if haskey(var_index, base)
                vals[i] = yss[var_index[base]]
            else
                vals[i] = 0.0
            end
        else
            vals[i] = 0.0
        end
    end

    return vals
end

function fill_dyn_values!(
    dyn_values::Vector{Float64},
    var_kind::Vector{Symbol},
    var_idx::Vector{Int},
    y_lag::AbstractVector{Float64},
    y_cur::AbstractVector{Float64},
    y_fwd::AbstractVector{Float64},
    ε_curr::AbstractVector{Float64}
)
    @inbounds for i in 1:length(var_kind)
        kind = var_kind[i]
        idx = var_idx[i]
        if kind == :future
            dyn_values[i] = y_fwd[idx]
        elseif kind == :present
            dyn_values[i] = y_cur[idx]
        elseif kind == :past
            dyn_values[i] = y_lag[idx]
        elseif kind == :shock
            dyn_values[i] = ε_curr[idx]
        else
            dyn_values[i] = 0.0
        end
    end
    return
end

function build_dyn_var_maps(𝓂::ℳ, vars_raw::Vector{Symbol})
    var_index_map = Dict{Symbol,Int}(v => i for (i, v) in enumerate(𝓂.var))
    shock_index_map = Dict{Symbol,Int}(v => i for (i, v) in enumerate(𝓂.exo))
    var_kind = Vector{Symbol}(undef, length(vars_raw))
    var_idx = Vector{Int}(undef, length(vars_raw))

    for (i, v) in enumerate(vars_raw)
        vstr = string(v)
        if occursin("₍₁₎", vstr)
            base = Symbol(replace(vstr, r"₍₁₎$" => ""))
            var_kind[i] = :future
            var_idx[i] = var_index_map[base]
        elseif occursin("₍₀₎", vstr)
            base = Symbol(replace(vstr, r"₍₀₎$" => ""))
            var_kind[i] = :present
            var_idx[i] = var_index_map[base]
        elseif occursin("₍₋₁₎", vstr)
            base = Symbol(replace(vstr, r"₍₋₁₎$" => ""))
            var_kind[i] = :past
            var_idx[i] = var_index_map[base]
        elseif occursin("₍ₓ₎", vstr)
            base = Symbol(replace(vstr, r"₍ₓ₎$" => ""))
            var_kind[i] = :shock
            var_idx[i] = shock_index_map[base]
        else
            base = Symbol(replace(vstr, r"₍.*₎$" => ""))
            if haskey(var_index_map, base)
                var_kind[i] = :present
                var_idx[i] = var_index_map[base]
            else
                error("Unrecognized dynamic variable $v in SEP residual mapping.")
            end
        end
    end

    return var_kind, var_idx
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
For sparse tree: only the trunk (group 1) branches; side branches are created
over time and then follow deterministic continuations.
"""

function is_trunk_node(layout::SEPLayout, t::Int, g::Int)
    !layout.sparse && return true  # Full tree: all nodes can branch
    return g == 1  # Sparse tree: only trunk branches
end

function parent_group_sparse(layout::SEPLayout, t::Int, g::Int)
    !layout.sparse && return parent_group(layout, t, g)  # Fall back to full tree

    (t == 0) && return 1
    (g == 1) && return 1

    # Side branch: parent is trunk only at branch time, otherwise self
    info = branch_info_sparse(layout, g)
    if isnothing(info)
        return g
    end
    branch_time, _ = info
    return (t == branch_time) ? 1 : g
end

function child_groups_sparse(layout::SEPLayout, t::Int, g::Int)
    !layout.sparse && return child_groups(layout, t, g)  # Fall back to full tree

    # No branching after Lbr
    if t > layout.Lbr
        return [g]
    end

    # Side branches never branch
    if g != 1
        return [g]
    end

    # Trunk branches: nodes map to branches created at t+1 (Dynare indexing)
    if layout.K <= 1
        return [1]
    end

    branch_time = t + 1
    cgs = Vector{Int}(undef, layout.K)
    cgs[1] = 1  # zero node stays on trunk
    for k in 2:layout.K
        cgs[k] = branch_group_index(layout, branch_time, k)
    end
    return cgs
end

"""
Get branch time and node index from side branch group number.
Groups are organized as:
- g=1: trunk
- g>1: side branches created at times 2..Lbr+1, K-1 per time

Returns: (branch_time, node_index) or nothing for trunk
"""
function branch_info_sparse(layout::SEPLayout, g::Int)
    !layout.sparse && error("branch_info_sparse only valid for sparse tree")

    if g == 1 || layout.K <= 1
        return nothing
    end

    idx = g - 2  # 0-based index among side branches
    branch_offset = div(idx, layout.K - 1)
    node_offset = mod(idx, layout.K - 1)
    branch_time = 2 + branch_offset
    node_index = 2 + node_offset
    return (branch_time, node_index)
end

function get_shock_from_group(layout::SEPLayout, g::Int)
    return branch_info_sparse(layout, g)
end

"""
Get child group index for a given branch time and node index.
"""
function branch_group_index(layout::SEPLayout, branch_time::Int, node_index::Int)
    !layout.sparse && error("branch_group_index only valid for sparse tree")
    if node_index == 1
        return 1
    end
    return 1 + (branch_time - 2) * (layout.K - 1) + (node_index - 1)
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

    # Ensure zero node is first for Dynare-compatible ordering
    reorder_1d_zero_first!(x1d, w1d)

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
Ensure the zero-shock node is first (Dynare ordering).
"""
function reorder_nodes_zero_first!(X::Matrix{Float64}, W::Vector{Float64}; tol::Float64=1e-12)
    zero_idx = findfirst(i -> all(abs.(view(X, :, i)) .< tol), 1:size(X, 2))
    if zero_idx === nothing || zero_idx == 1
        return
    end
    X[:, [1, zero_idx]] = X[:, [zero_idx, 1]]
    W[[1, zero_idx]] = W[[zero_idx, 1]]
    return
end

"""
Ensure zero node is first for 1D node arrays.
"""
function reorder_1d_zero_first!(x::Vector{Float64}, w::Vector{Float64}; tol::Float64=1e-12)
    zero_idx = findfirst(v -> abs(v) < tol, x)
    if zero_idx === nothing || zero_idx == 1
        return
    end
    x[1], x[zero_idx] = x[zero_idx], x[1]
    w[1], w[zero_idx] = w[zero_idx], w[1]
    return
end

"""
Get 1D Gauss-Hermite nodes and weights for sparse tree.
Returns nodes and weights for a single dimension.
"""
function gh_nodes_1d(nnodes::Int)
    if nnodes == 1
        return [0.0], [√π]
    elseif nnodes == 3
        x = [-√3, 0.0, √3]
        w = [π/6, 2π/3, π/6]
        reorder_1d_zero_first!(x, w)
        return x, w
    elseif nnodes == 5
        x = [-√(5+2√(10/7)), -√(5-2√(10/7)), 0.0, √(5-2√(10/7)), √(5+2√(10/7))]
        w = [π*(322-13√70)/900, π*(322+13√70)/900, 128π/225,
             π*(322+13√70)/900, π*(322-13√70)/900]
        reorder_1d_zero_first!(x, w)
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
    SS_and_pars::Vector{Float64},
    SS_result,
    initial_state::Union{Nothing,Vector{Float64}}=nothing
)
    # Extract dimensions
    ny_ = length(𝓂.var)
    dε = length(𝓂.exo)
    T = opts.periods

    opts.verbose && @info "Deterministic path solver" T=T ny_=ny_ dε=dε

    # Build nonlinear residual and Jacobian for dynamic equations
    dyn_resid_func, dyn_jac_func, vars_raw, parameters_and_SS, resid_buffer, jac_buffer =
        build_dynamic_residual_jacobian(𝓂)

    # Map parameters and steady state values into parameters_and_SS ordering
    params_and_ss = build_parameters_and_ss_values(parameters_and_SS, parameters, 𝓂, yss, SS_result)

    # Precompute variable mapping from vars_raw to y/shock vectors
    var_kind, var_idx = build_dyn_var_maps(𝓂, vars_raw)

    dyn_values = zeros(Float64, length(vars_raw))

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
    # Indexing uses layout.voff starting at 1, so keep Y[1] as unused padding.
    nvars_total = ny_ * (T + 1) + 1

    if !isnothing(initial_guess) && length(initial_guess) == nvars_total
        Y = copy(initial_guess)
        opts.verbose && @info "Using provided initial guess"
    elseif !isnothing(initial_guess) && length(initial_guess) == nvars_total - 1
        Y = zeros(nvars_total)
        Y[2:end] .= initial_guess
        opts.verbose && @info "Using provided initial guess (padded)"
    else
        Y = zeros(nvars_total)
        Y[2:end] .= repeat(yss, T + 1)
        opts.verbose && @info "Initializing at steady state"
    end
    if !isnothing(initial_state)
        @assert length(initial_state) == ny_ "sep_initial_state must have length $ny_ (got $(length(initial_state)))"
        Y[2:ny_+1] .= initial_state
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
            # Get states at t-1, t, t+1 (offset by 1 due to padding)
            y_lag = view(Y, 1 .+ (t-1)*ny_ .+ (1:ny_))
            y_cur = view(Y, 1 .+ t*ny_ .+ (1:ny_))
            y_fwd = (t < T) ? view(Y, 1 .+ (t+1)*ny_ .+ (1:ny_)) : yss  # Terminal: return to SS

            # Get deterministic shock at period t
            ε_t = view(opts.deterministic_shocks, t, :)

            fill_dyn_values!(dyn_values, var_kind, var_idx, y_lag, y_cur, y_fwd, ε_t)

            # Nonlinear residual and Jacobian
            dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
            dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

            # Store residual
            rr = (t-1)*ny_ .+ (1:ny_)
            R[rr] .= resid_buffer

            # Fill Jacobian blocks
            # Note: Column indices are for [y₁, ..., yT] (excluding fixed y₀)
            # So we map: y₁ → cols 1:ny_, y₂ → cols ny_+1:2*ny_, etc.

            c_cur = (t-1)*ny_ .+ (1:ny_)  # y_t columns (t=1 maps to cols 1:ny_)
            c_lag = (t-2)*ny_ .+ (1:ny_)
            c_fwd = t*ny_ .+ (1:ny_)

            for i in 1:ny_
                r_row = rr[i]
                for j in 1:length(vars_raw)
                    v = jac_buffer[i, j]
                    if abs(v) > 1e-16
                        kind = var_kind[j]
                        idx = var_idx[j]
                        if kind == :present
                            nnz_count += 1
                            rows[nnz_count] = r_row
                            cols[nnz_count] = c_cur[idx]
                            vals[nnz_count] = v
                        elseif kind == :past
                            if t > 1
                                nnz_count += 1
                                rows[nnz_count] = r_row
                                cols[nnz_count] = c_lag[idx]
                                vals[nnz_count] = v
                            end
                        elseif kind == :future
                            if t < T
                                nnz_count += 1
                                rows[nnz_count] = r_row
                                cols[nnz_count] = c_fwd[idx]
                                vals[nnz_count] = v
                            end
                        end
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
        Y[(ny_+2):end] .+= α * Δ
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
    initial_guess::Union{Nothing,Vector{Float64}}=nothing,
    initial_state::Union{Nothing,Vector{Float64}}=nothing
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

    # DETERMINISTIC MODE BRANCHING: Only use perfect foresight when order == 0
    if !isnothing(opts.deterministic_shocks) && opts.order == 0
        opts.verbose && @info "Deterministic mode detected (order=0) - using perfect foresight solver"
        return solve_deterministic_path(𝓂, parameters, opts, initial_guess, yss, SS_and_pars, SS_result, initial_state)
    end

    # STOCHASTIC MODE: Continue with existing SEP solver
    use_nonlinear_residuals = true
    if use_nonlinear_residuals
        dyn_resid_func, dyn_jac_func, vars_raw, parameters_and_SS, resid_buffer, jac_buffer =
            build_dynamic_residual_jacobian(𝓂)
        params_and_ss = build_parameters_and_ss_values(parameters_and_SS, parameters, 𝓂, yss, SS_result)
        var_kind, var_idx = build_dyn_var_maps(𝓂, vars_raw)
        dyn_values = zeros(Float64, length(vars_raw))
    else
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
            # Fallback to unit variance if parameter not found (Dynare-style)
            @warn "Shock std parameter $param_name not found, using default 1.0"
            Σ[i, i] = 1.0
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
        # Ensure zero node is first (Dynare ordering)
        reorder_nodes_zero_first!(X, W)
    end

    # Build tree structure
    G = Vector{Int}(undef, T+2)
    if opts.sparse_tree
        # Sparse tree: trunk branches over time, adding (K-1) side branches per period
        for t in 0:T+1
            if t == 0 || K <= 1
                G[t+1] = 1
            else
                branch_levels = min(t - 1, Lbr)
                G[t+1] = 1 + (K - 1) * branch_levels
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

    # Fix initial conditions (y0) for stochastic SEP
    y0_idx = index_y(layout, 0, 1)
    if !isnothing(initial_state)
        @assert length(initial_state) == ny_ "sep_initial_state must have length $ny_ (got $(length(initial_state)))"
        Y[y0_idx] .= initial_state
    end
    y0_fixed = copy(Y[y0_idx])

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
    ε_det = isnothing(opts.deterministic_shocks) ? nothing : zeros(dε)
    ε_tmp = zeros(dε)

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
                    # Branching node: expectation over child groups (nonlinear residuals)
                    fill!(r_sum, 0.0)
                    fill!(J_lag, 0.0)
                    fill!(J_cur, 0.0)

                    rr = row_range(layout, t, g)

                    for (kidx, cg) in enumerate(cgs)
                        yl1 = get_y(t+1, cg)

                        if layout.sparse
                            shock_idx = kidx
                            ε_to_child = view(X, :, shock_idx)
                            wk = W[shock_idx]
                        else
                            k_shock = mod(g - 1, K) + 1
                            ε_to_child = view(X, :, k_shock)
                            wk = W[kidx]
                        end

                        # Dynare convention: at t=1 use deterministic shock only; at t>=2 use node shocks
                        if t == 1
                            if ε_det !== nothing
                                ε_det .= view(opts.deterministic_shocks, t, :)
                                ε_curr = ε_det
                            else
                                fill!(ε_tmp, 0.0)
                                ε_curr = ε_tmp
                            end
                        else
                            ε_curr = ε_to_child
                        end

                        fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, ε_curr)
                        dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                        dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

                        r_sum .+= wk .* resid_buffer

                        if t+1 <= T
                            c_rng = index_y(layout, t+1, cg)
                        end

                        for irow in 1:ny_
                            r_row = first(rr) + irow - 1
                            for j in 1:length(vars_raw)
                                v = jac_buffer[irow, j]
                                if abs(v) > 1e-16
                                    kind = var_kind[j]
                                    idx = var_idx[j]
                                    if kind == :present
                                        J_cur[irow, idx] += wk * v
                                    elseif kind == :past
                                        if t > 1
                                            J_lag[irow, idx] += wk * v
                                        end
                                    elseif kind == :future
                                        if t+1 <= T
                                            nnz_count += 1
                                            rows[nnz_count] = r_row
                                            cols[nnz_count] = c_rng.start + idx - 1
                                            vals[nnz_count] = wk * v
                                        end
                                    end
                                end
                            end
                        end
                    end

                    # Store residual and lag/current Jacobians
                    R[rr] .= r_sum
                    c_lag = index_y(layout, t-1, pg)
                    c_cur = index_y(layout, t, g)

                    for irow in 1:ny_
                        r_gl = first(rr) + irow - 1
                        if t > 1
                            for jcol in 1:ny_
                                v = J_lag[irow, jcol]
                                if abs(v) > 1e-16
                                    nnz_count += 1
                                    rows[nnz_count] = r_gl
                                    cols[nnz_count] = c_lag.start + jcol - 1
                                    vals[nnz_count] = v
                                end
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

                    # For sparse tree side branches, apply node shock only at branch time
                    if layout.sparse && g > 1
                        info = branch_info_sparse(layout, g)
                        if !isnothing(info) && t == info[1]
                            ε_curr = view(X, :, info[2])
                        else
                            if ε_det !== nothing
                                ε_det .= view(opts.deterministic_shocks, t, :)
                                ε_curr = ε_det
                            else
                                fill!(ε_tmp, 0.0)
                                ε_curr = ε_tmp
                            end
                        end
                    else
                        # Full tree non-branching or sparse trunk after branching period
                        if ε_det !== nothing
                            ε_det .= view(opts.deterministic_shocks, t, :)
                            ε_curr = ε_det
                        else
                            fill!(ε_tmp, 0.0)
                            ε_curr = ε_tmp
                        end
                    end

                    fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, ε_curr)
                    dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                    dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

                    rr = row_range(layout, t, g)
                    R[rr] .= resid_buffer
                    c_lag = index_y(layout, t-1, pg)
                    c_cur = index_y(layout, t, g)

                    if t+1 <= T
                        c_lea = index_y(layout, t+1, cg)
                    end

                    for irow in 1:ny_
                        r_gl = first(rr) + irow - 1
                        for j in 1:length(vars_raw)
                            v = jac_buffer[irow, j]
                            if abs(v) > 1e-16
                                kind = var_kind[j]
                                idx = var_idx[j]
                                if kind == :future
                                    if t+1 <= T
                                        nnz_count += 1
                                        rows[nnz_count] = r_gl
                                        cols[nnz_count] = c_lea.start + idx - 1
                                        vals[nnz_count] = v
                                    end
                                elseif kind == :past
                                    if t > 1
                                        nnz_count += 1
                                        rows[nnz_count] = r_gl
                                        cols[nnz_count] = c_lag.start + idx - 1
                                        vals[nnz_count] = v
                                    end
                                elseif kind == :present
                                    nnz_count += 1
                                    rows[nnz_count] = r_gl
                                    cols[nnz_count] = c_cur.start + idx - 1
                                    vals[nnz_count] = v
                                end
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

        # Apply update with adaptive step size (keep y0 fixed)
        Δ[y0_idx] .= 0.0
        Y .+= α * Δ
        Y[y0_idx] .= y0_fixed
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
