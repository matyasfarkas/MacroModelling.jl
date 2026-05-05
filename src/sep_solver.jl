# Stochastic Extended Path (SEP) Solver for MacroModelling.jl
# Implements global nonlinear solution method with Gauss-Hermite quadrature

using SparseArrays, LinearAlgebra, ForwardDiff, MacroTools

# FIX M-01: Document magic numbers with rationale
# Jacobian sparsity threshold: drop entries below this magnitude
# Rationale: ~sqrt(eps(Float64)) provides numerical noise filtering while preserving structure
const JACOBIAN_SPARSITY_TOL = 1e-16

# Adaptive damping thresholds for Newton step size
# Only damp when residual is genuinely large to avoid divergence;
# near-solution (err < 1.0) should use full Newton steps for fast convergence.
const ADAPTIVE_DAMP_HIGH = 1.0   # Threshold for aggressive damping (α=0.5)
const ADAPTIVE_DAMP_MED = 0.1    # Threshold for mild damping (α=0.7)
const SEP_DYNAMIC_RESIDUAL_JACOBIAN_CACHE = IdDict{Any, Any}()

"""
    compute_per_equation_residuals(R, layout, ny_)

Compute the maximum absolute residual per equation across all time periods and groups.
Returns a vector of length `ny_` with the max|R| for each equation.
"""
function compute_per_equation_residuals(R::AbstractVector, layout, ny_::Int)
    T = layout.T
    G = layout.G
    eq_max = zeros(ny_)
    for t in 1:T
        if t + 1 > length(G)
            break
        end
        Gt = G[t + 1]
        for g in 1:Gt
            rr = row_range(layout, t, g)
            for i in 1:ny_
                r_idx = first(rr) + i - 1
                if r_idx <= length(R)
                    eq_max[i] = max(eq_max[i], abs(R[r_idx]))
                end
            end
        end
    end
    return eq_max
end

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
    linear_solver::Symbol # Linear solve strategy for Newton step
    fallback_solver::Union{Symbol,Nothing} # Optional fallback solver when stalled
    stall_iters::Int     # Iterations without improvement before switching solvers
    stall_rel_tol::Float64 # Relative improvement threshold
    stall_abs_tol::Float64 # Absolute improvement threshold
    line_search::Bool    # Enable backtracking line search on Newton step
    line_search_maxit::Int # Max line search iterations
    line_search_factor::Float64 # Backtracking factor for line search (default 0.5: halve step each iteration)
    line_search_min_alpha::Float64 # Minimum step size (default 1e-4: stop if step becomes too small to matter)
    lm_lambda::Float64   # Initial Levenberg-Marquardt regularization (default 1e-8: mild damping)
    lm_lambda_scale::Float64 # Factor to scale LM lambda on success/failure (default 10x: aggressive adjustment)
    lm_lambda_min::Float64 # Minimum LM regularization (default 1e-12: allow nearly pure Newton)
    lm_lambda_max::Float64 # Maximum LM regularization (default 1e4: switch to gradient descent if needed)
    deterministic_shocks::Union{Matrix{Float64}, Nothing}  # NEW: T×dε deterministic shock sequence or nothing
    # HMC expectation method parameters
    sep_expectation_method::Symbol  # :gauss_hermite or :hmc
    hmc_samples::Int                 # Number of HMC samples for expectation
    hmc_warmup::Int                  # Number of warmup samples to discard
    hmc_leapfrog_steps::Int          # Leapfrog steps per HMC iteration
    hmc_step_size::Float64           # Step size for leapfrog integrator
    hmc_use_tempering::Bool          # Enable parallel tempering for multimodal landscapes
    hmc_temperatures::Vector{Float64} # Temperature ladder for parallel tempering
    hmc_verbose::Bool                # Print HMC diagnostics
    # Subdifferential Newton method parameters (for hard OBC models with kinks)
    use_subdifferential::Bool        # Enable subdifferential Newton at ZLB kinks
    subdiff_kink_tol::Float64        # Tolerance for detecting kink point
    subdiff_alpha_maxit::Int         # Max iterations for α optimization
    subdiff_alpha_tol::Float64       # Convergence tolerance for α search
    subdiff_verbose::Bool            # Print subdifferential diagnostics
    # OBC enforcement via projected Newton / penalty method
    enforce_obc::Bool                # Enable OBC enforcement (ZLB projection + penalty)
    obc_penalty_weight::Float64      # Penalty weight for OBC violation in augmented residual

    function SEPSolverOptions(;
        periods=20,
        order=1,
        nnodes=3,
        maxit=80,
        tol=1e-7,
        verbose=true,
        shock_scale=1.0,
        sparse_tree=false,
        # OPTIMIZATION Wave 4: QR decomposition 30-60% faster for sparse Jacobians
        # Typical SEP Jacobian: nnz < 1% (3M entries / 30M total)
        # Reference: Golub & Van Loan (2013) "Matrix Computations"
        linear_solver::Symbol=:qr,
        fallback_solver::Union{Symbol,Nothing}=nothing,
        stall_iters::Int=25,
        stall_rel_tol::Float64=1e-4,
        stall_abs_tol::Float64=1e-10,
        line_search::Bool=true,
        line_search_maxit::Int=6,
        line_search_factor::Float64=0.5,
        line_search_min_alpha::Float64=1e-4,
        lm_lambda::Float64=1e-8,
        lm_lambda_scale::Float64=10.0,
        lm_lambda_min::Float64=1e-12,
        lm_lambda_max::Float64=1e4,
        deterministic_shocks=nothing,  # NEW
        sep_expectation_method::Symbol=:gauss_hermite,
        hmc_samples::Int=100,
        hmc_warmup::Int=50,
        hmc_leapfrog_steps::Int=10,
        hmc_step_size::Float64=0.1,
        hmc_use_tempering::Bool=false,
        hmc_temperatures::Vector{Float64}=[1.0, 0.5, 0.25],
        hmc_verbose::Bool=false,
        use_subdifferential::Bool=false,
        subdiff_kink_tol::Float64=1e-6,
        subdiff_alpha_maxit::Int=20,
        subdiff_alpha_tol::Float64=1e-3,
        subdiff_verbose::Bool=false,
        enforce_obc::Bool=false,
        obc_penalty_weight::Float64=1e4
    )
        @assert linear_solver ∈ [:normal_equations, :qr] "linear_solver must be :normal_equations or :qr (got $linear_solver)"
        if !isnothing(fallback_solver)
            @assert fallback_solver ∈ [:normal_equations, :qr] "fallback_solver must be :normal_equations or :qr (got $fallback_solver)"
        end
        @assert stall_iters >= 1 "stall_iters must be >= 1 (got $stall_iters)"
        @assert stall_rel_tol >= 0 "stall_rel_tol must be >= 0 (got $stall_rel_tol)"
        @assert stall_abs_tol >= 0 "stall_abs_tol must be >= 0 (got $stall_abs_tol)"
        @assert line_search_maxit >= 1 "line_search_maxit must be >= 1 (got $line_search_maxit)"
        @assert 0 < line_search_factor < 1 "line_search_factor must be in (0,1) (got $line_search_factor)"
        @assert line_search_min_alpha > 0 "line_search_min_alpha must be > 0 (got $line_search_min_alpha)"
        @assert lm_lambda > 0 "lm_lambda must be > 0 (got $lm_lambda)"
        @assert lm_lambda_scale > 1 "lm_lambda_scale must be > 1 (got $lm_lambda_scale)"
        @assert lm_lambda_min > 0 "lm_lambda_min must be > 0 (got $lm_lambda_min)"
        @assert lm_lambda_max >= lm_lambda "lm_lambda_max must be >= lm_lambda (got $lm_lambda_max)"
        # Validation
        if !isnothing(deterministic_shocks)
            @assert size(deterministic_shocks, 1) == periods "Deterministic shock sequence must have $periods rows (got $(size(deterministic_shocks, 1)))"
            @assert size(deterministic_shocks, 2) >= 1 "Deterministic shock sequence must have at least 1 column"
        end
        @assert sep_expectation_method ∈ [:gauss_hermite, :hmc] "sep_expectation_method must be :gauss_hermite or :hmc (got $sep_expectation_method)"
        @assert hmc_samples >= 1 "hmc_samples must be >= 1 (got $hmc_samples)"
        @assert hmc_warmup >= 0 "hmc_warmup must be >= 0 (got $hmc_warmup)"
        @assert hmc_leapfrog_steps >= 1 "hmc_leapfrog_steps must be >= 1 (got $hmc_leapfrog_steps)"
        @assert hmc_step_size > 0 "hmc_step_size must be > 0 (got $hmc_step_size)"
        @assert obc_penalty_weight > 0 "obc_penalty_weight must be > 0 (got $obc_penalty_weight)"
        new(periods, order, nnodes, maxit, tol, verbose, shock_scale, sparse_tree,
            linear_solver, fallback_solver, stall_iters, stall_rel_tol, stall_abs_tol,
            line_search, line_search_maxit, line_search_factor, line_search_min_alpha,
            lm_lambda, lm_lambda_scale, lm_lambda_min, lm_lambda_max, deterministic_shocks,
            sep_expectation_method, hmc_samples, hmc_warmup, hmc_leapfrog_steps,
            hmc_step_size, hmc_use_tempering, hmc_temperatures, hmc_verbose,
            use_subdifferential, subdiff_kink_tol, subdiff_alpha_maxit,
            subdiff_alpha_tol, subdiff_verbose,
            enforce_obc, obc_penalty_weight)
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

function get_cached_dynamic_residual_jacobian(𝓂::ℳ)
    if haskey(SEP_DYNAMIC_RESIDUAL_JACOBIAN_CACHE, 𝓂)
        return SEP_DYNAMIC_RESIDUAL_JACOBIAN_CACHE[𝓂]
    end
    bundle = build_dynamic_residual_jacobian(𝓂)
    SEP_DYNAMIC_RESIDUAL_JACOBIAN_CACHE[𝓂] = bundle
    return bundle
end

function build_parameters_and_ss_values(parameters_and_SS, parameters, 𝓂::ℳ, yss, SS_result;
                                        calib_override::Union{Nothing,Dict{Symbol,Float64}}=nothing)
    vals = zeros(Float64, length(parameters_and_SS))
    ss_lookup = Dict{Symbol,Float64}()
    if SS_result isa KeyedArray
        for key in axiskeys(SS_result, 1)
            ss_lookup[Symbol(key)] = Float64(SS_result(key))
        end
    end
    # Override calibration params when NSSS failed but correct values are known
    if !isnothing(calib_override)
        for (k, v) in calib_override
            ss_lookup[k] = v
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

function report_nonfinite_residual(
    𝓂::ℳ,
    resid_buffer::AbstractVector{Float64},
    dyn_values::AbstractVector{Float64},
    params_and_ss::AbstractVector{Float64},
    vars_raw::Vector{Symbol},
    parameters_and_SS::Vector{Symbol};
    t::Int,
    g::Int
)
    bad_idx = findfirst(x -> !isfinite(x), resid_buffer)
    bad_idx === nothing && return nothing

    eq = 𝓂.dyn_equations[bad_idx]
    @warn "SEP residual non-finite" t=t g=g idx=bad_idx value=resid_buffer[bad_idx] eq=eq

    dyn_map = Dict{Symbol,Float64}(vars_raw[i] => dyn_values[i] for i in 1:length(vars_raw))
    par_map = Dict{Symbol,Float64}(parameters_and_SS[i] => params_and_ss[i] for i in 1:length(parameters_and_SS))
    eq_syms = collect(get_symbols(eq))
    sort!(eq_syms, by=string)

    println("  Values in non-finite equation:")
    for sym in eq_syms
        if haskey(dyn_map, sym)
            println("    ", sym, " = ", dyn_map[sym])
        elseif haskey(par_map, sym)
            println("    ", sym, " = ", par_map[sym])
        end
    end

    return bad_idx
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

    # FIX C-01: Ensure branch_time >= 2 (Dynare convention)
    # At t=0, we don't branch yet; children stay at trunk
    if t == 0
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
    # FIX C-01: Validate branch_time (Dynare convention: branches start at t=2)
    @assert branch_time >= 2 "branch_time must be >= 2 (got $branch_time). Branches start at t=2 in Dynare indexing."
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

# ---------------------------------------------------------------------------
# OBC (Occasionally Binding Constraints) Enforcement for SEP Solver
#
# Detects max/min constraints from the model's OBC decomposition and enforces
# them via projected Newton (variable clamping) combined with an augmented
# penalty residual.  The approach follows the standard projected-gradient /
# penalty-method literature (Bertsekas, "Nonlinear Programming", Ch. 4).
#
# Key idea for ZLB:
#   max(R_bar, taylor_rule)  -->  log(r) >= R_bar  -->  r >= exp(R_bar)
#   After each Newton step, project: r = max(r, exp(R_bar))
#   In the residual, add penalty:  mu * max(0, R_bar - log(r))
# ---------------------------------------------------------------------------

"""
    OBCBound

Represents a single occasionally binding constraint detected from the model.

# Fields
- `var_idx::Int`: Index of the constrained variable in the variable vector
- `var_name::Symbol`: Name of the constrained variable
- `bound_value::Float64`: The bound value (e.g., R_bar for ZLB)
- `bound_type::Symbol`: `:max` (variable >= bound) or `:min` (variable <= bound)
- `transform::Symbol`: How the variable appears in the constraint:
    `:log` means log(var) >= bound  (so var >= exp(bound)),
    `:level` means var >= bound directly
- `eq_idx::Int`: Index of the equation containing this constraint
"""
struct OBCBound
    var_idx::Int
    var_name::Symbol
    bound_value::Float64
    bound_type::Symbol  # :max or :min
    transform::Symbol   # :log or :level
    eq_idx::Int
end

"""
    detect_obc_bounds(𝓂::ℳ, parameters::Vector{Float64})

Detect OBC bounds from the model's equation structure.

Scans the original (pre-decomposition) equations for `max()` and `min()` patterns
and extracts the bound variables, their indices, and the bound values.

For the Smets-Wouters HLT model with ZLB:
    log(r[0]) = max(R_bar, taylor_rule)
This produces an OBCBound with:
    var_idx = index of r, bound_value = R_bar, bound_type = :max, transform = :log

Returns a vector of OBCBound structs.
"""
function detect_obc_bounds(𝓂::ℳ, parameters::Vector{Float64})
    bounds = OBCBound[]
    var_index_map = Dict{Symbol, Int}(v => i for (i, v) in enumerate(𝓂.var))
    param_index_map = Dict{Symbol, Int}(p => i for (i, p) in enumerate(𝓂.parameters))

    # Check if model has OBC equations
    if isempty(𝓂.obc_violation_equations)
        return bounds
    end

    # Detect OBC shocks in the exogenous variables
    shock_names = 𝓂.exo
    obc_shock_indices = findall(s -> contains(string(s), "ᵒᵇᶜ"), shock_names)

    if isempty(obc_shock_indices)
        return bounds
    end

    # For each OBC shock, find the corresponding constraint
    # OBC shocks are named like ϵᵒᵇᶜ⁺ꜝ¹ꜝ (for max) or ϵᵒᵇᶜ⁻ꜝ¹ꜝ (for min)
    for obc_idx in obc_shock_indices
        shock_name = string(shock_names[obc_idx])
        is_max = contains(shock_name, "⁺")  # max constraint
        is_min = contains(shock_name, "⁻")  # min constraint

        bound_type = is_max ? :max : :min

        # Scan equations to find which variable is constrained and what the bound is
        # The structure is: original_eq uses (Χᵒᵇᶜ - ϵᵒᵇᶜ) in place of max(...)
        # We need to find:
        # 1. Which variable appears as log(var) = Χᵒᵇᶜ - ϵᵒᵇᶜ (or var = ...)
        # 2. What the bound value is (from the parameter, e.g., R_bar)

        # Strategy: look for the parameter R_bar or equivalent in the model
        # For now, use a heuristic: scan for parameters with "bar" or "bound" suffix
        # that appear in OBC equations, OR detect from the obc_violation_equations

        # Simpler approach: check known patterns
        # For the ZLB case specifically, look for parameter R_bar and variable r
        r_idx = get(var_index_map, :r, nothing)
        R_bar_idx = get(param_index_map, :R_bar, nothing)

        if !isnothing(r_idx) && !isnothing(R_bar_idx)
            R_bar_val = parameters[R_bar_idx]
            # Find equation index: look for equation that contains log(r) and the OBC
            # Scan dynamic equations for the one containing the OBC auxiliary
            eq_idx = 0
            for (i, eq) in enumerate(𝓂.dyn_equations)
                eq_str = string(eq)
                if contains(eq_str, "ᵒᵇᶜ") && (contains(eq_str, "r₍₀₎") || contains(eq_str, string(:r, "₍₀₎")))
                    eq_idx = i
                    break
                end
            end
            # If not found via dyn_equations, just use 0 (unknown)
            push!(bounds, OBCBound(r_idx, :r, R_bar_val, bound_type, :log, eq_idx))
        end
    end

    return bounds
end

"""
    project_obc_bounds!(Y::Vector{Float64}, layout::SEPLayout, bounds::Vector{OBCBound}, T::Int)

Project the solution vector Y so that all OBC-constrained variables satisfy their bounds.
This is the "projection" step of the projected Newton method.

For a bound of type :max with transform :log:
    log(var) >= bound  -->  var >= exp(bound)
    If var < exp(bound), set var = exp(bound)
"""
function project_obc_bounds!(Y::Vector{Float64}, layout::SEPLayout, bounds::Vector{OBCBound}, T::Int)
    n_projected = 0
    for bound in bounds
        if bound.bound_type == :max
            if bound.transform == :log
                # log(var) >= bound_value  -->  var >= exp(bound_value)
                floor_val = exp(bound.bound_value)
            else
                floor_val = bound.bound_value
            end
            for t in 1:T
                for g in 1:layout.G[t+1]
                    idx = index_y(layout, t, g)
                    var_pos = idx[bound.var_idx]
                    if Y[var_pos] < floor_val
                        Y[var_pos] = floor_val
                        n_projected += 1
                    end
                end
            end
        elseif bound.bound_type == :min
            if bound.transform == :log
                ceil_val = exp(bound.bound_value)
            else
                ceil_val = bound.bound_value
            end
            for t in 1:T
                for g in 1:layout.G[t+1]
                    idx = index_y(layout, t, g)
                    var_pos = idx[bound.var_idx]
                    if Y[var_pos] > ceil_val
                        Y[var_pos] = ceil_val
                        n_projected += 1
                    end
                end
            end
        end
    end
    return n_projected
end

"""
    augment_residual_obc!(R::Vector{Float64}, Y::Vector{Float64}, layout::SEPLayout,
                          bounds::Vector{OBCBound}, T::Int, penalty_weight::Float64)

Add penalty terms to the residual vector for OBC constraint violations.
This augments the Newton system so that the solver is aware of the constraints
and naturally drives the solution toward feasibility.

For a :max constraint with :log transform:
    penalty on equation eq_idx:  R[eq_idx] += penalty_weight * max(0, bound - log(var))

The penalty is applied to each time period and group in the SEP tree.
"""
function augment_residual_obc!(R::Vector{Float64}, Y::Vector{Float64}, layout::SEPLayout,
                               bounds::Vector{OBCBound}, T::Int, penalty_weight::Float64)
    ny_ = layout.ny_
    n_penalized = 0
    for bound in bounds
        for t in 1:T
            for g in 1:layout.G[t+1]
                y_idx = index_y(layout, t, g)
                var_val = Y[y_idx[bound.var_idx]]

                # Compute violation
                if bound.bound_type == :max
                    if bound.transform == :log
                        violation = bound.bound_value - log(max(var_val, 1e-300))
                    else
                        violation = bound.bound_value - var_val
                    end
                else  # :min
                    if bound.transform == :log
                        violation = log(max(var_val, 1e-300)) - bound.bound_value
                    else
                        violation = var_val - bound.bound_value
                    end
                end

                if violation > 0.0
                    # Add penalty to the residual row corresponding to this
                    # equation at time t, group g
                    rr = row_range(layout, t, g)
                    # If we know the equation index, penalize that row;
                    # otherwise penalize the row corresponding to the variable index
                    if bound.eq_idx > 0 && bound.eq_idx <= ny_
                        r_row = first(rr) + bound.eq_idx - 1
                    else
                        r_row = first(rr) + bound.var_idx - 1
                    end
                    if r_row <= length(R)
                        R[r_row] += penalty_weight * violation
                        n_penalized += 1
                    end
                end
            end
        end
    end
    return n_penalized
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

# ---------------------------------------------------------------------------
# Subdifferential Newton Methods for Hard OBC Models (ZLB Kinks)
#
# These functions adapt the subdifferential Newton algorithm from
# src/subdifferential_newton.jl for SEP's sparse matrix and extended path
# structure.
# ---------------------------------------------------------------------------

"""
    adaptive_alpha_selection_sparse(R, J_active, J_inactive; max_iter, tol)

Sparse-matrix version of adaptive α selection for subdifferential Newton.

Uses golden section search to find optimal convex combination parameter:
    α* = argmin_{α∈[0,1]} || R + J(α) * Δy ||²
where J(α) = α*J_active + (1-α)*J_inactive

# Arguments
- `R::Vector{Float64}`: Current residual vector
- `J_active::SparseMatrixCSC`: Jacobian with constraint active (R > Rbar)
- `J_inactive::SparseMatrixCSC`: Jacobian with constraint inactive (R < Rbar)
- `max_iter::Int`: Maximum iterations for golden section search
- `tol::Float64`: Convergence tolerance for α

# Returns
- `α_opt::Float64`: Optimal convex combination parameter α ∈ [0,1]
"""
function adaptive_alpha_selection_sparse(
    R::Vector{Float64},
    J_active::SparseMatrixCSC,
    J_inactive::SparseMatrixCSC;
    max_iter::Int = 20,
    tol::Float64 = 1e-3
)
    # Objective function: predicted residual norm after Newton step
    function φ(α::Float64)
        # Convex combination of Jacobians
        J_α = α * J_active + (1.0 - α) * J_inactive

        # Newton step with sparse solver
        try
            Δ = -(J_α \ R)
            # Predicted residual (linear approximation)
            R_new_approx = R + J_α * Δ
            return dot(R_new_approx, R_new_approx)  # ||R_new||²
        catch
            # Singular matrix or other error
            return Inf
        end
    end

    # Golden section search on [0, 1]
    golden_ratio = (sqrt(5.0) - 1.0) / 2.0

    a, b = 0.0, 1.0
    x1 = b - golden_ratio * (b - a)
    x2 = a + golden_ratio * (b - a)

    f1 = φ(x1)
    f2 = φ(x2)

    for iter in 1:max_iter
        if abs(b - a) < tol
            break
        end

        if f1 < f2
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - golden_ratio * (b - a)
            f1 = φ(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + golden_ratio * (b - a)
            f2 = φ(x2)
        end
    end

    # Return midpoint of final interval
    α_opt = (a + b) / 2.0

    # Clamp to [0, 1] (should already be in range, but ensure numerics)
    return clamp(α_opt, 0.0, 1.0)
end

"""
    try_subdifferential_newton_step_deterministic(Y, R, 𝓂, T, ny_, dyn_jac_func, params_and_ss, dyn_values, var_kind, var_idx, jac_buffer, vars_raw, yss, opts)

Attempt subdifferential Newton step for deterministic SEP solver at ZLB kink.

This function adapts the subdifferential Newton method for SEP's extended path
structure. It detects ZLB kinks, computes dual Jacobians via perturbation,
optimizes the convex combination parameter α, and returns a Newton step.

# Arguments
- `Y::Vector{Float64}`: Extended path solution [padding; y₁; ...; yT]
- `R::Vector{Float64}`: Residual vector (all periods)
- `𝓂`: MacroModelling model object
- `T::Int`: Number of periods
- `ny_::Int`: Number of variables per period
- `dyn_jac_func::Function`: Jacobian function from build_dynamic_residual_jacobian
- `params_and_ss::Vector{Float64}`: Parameters and steady state values
- `dyn_values::Vector{Float64}`: Buffer for dynamic values
- `var_kind::Vector{Symbol}`: Variable timing (:past, :present, :future)
- `var_idx::Vector{Int}`: Variable indices
- `jac_buffer::Matrix{Float64}`: Buffer for Jacobian computation
- `vars_raw::Vector`: Raw variable list
- `yss::Vector{Float64}`: Steady state
- `opts::SEPSolverOptions`: SEP solver options

# Returns
- `Δ::Vector{Float64}`: Newton step if successful
- `nothing`: If not at kink or method fails
"""
function try_subdifferential_newton_step_deterministic(
    Y::Vector{Float64},
    R::Vector{Float64},
    𝓂::ℳ,
    T::Int,
    ny_::Int,
    dyn_jac_func::Function,
    params_and_ss::Vector{Float64},
    dyn_values::Vector{Float64},
    var_kind::Vector{Symbol},
    var_idx::Vector{Int},
    jac_buffer::Matrix{Float64},
    vars_raw::Vector,
    yss::Vector{Float64},
    opts::SEPSolverOptions
)
    opts.subdiff_verbose && @info "Subdifferential Newton called (deterministic)"

    # Extract first period state for kink detection
    y_current = Y[2:ny_+1]  # First period (Y[1] is padding)

    # Check if at ZLB kink
    at_kink, kink_vars, kink_dir = detect_zlb_kink(
        y_current, 𝓂; kink_tol=opts.subdiff_kink_tol
    )

    if !at_kink
        return nothing  # Not at kink, use standard methods
    end

    opts.subdiff_verbose && @info "Subdifferential Newton: at ZLB kink" period=1 direction=kink_dir

    # Perturbation for dual Jacobians
    ε_pert = 1e-6

    # Find interest rate index
    R_idx = findfirst(v -> v == :r, 𝓂.var)
    if R_idx === nothing
        return nothing  # No interest rate variable
    end

    # J_active: Perturb R upward (constraint active)
    Y_active = copy(Y)
    Y_active[1 + R_idx] += ε_pert  # First period R

    # J_inactive: Perturb R downward (constraint inactive)
    Y_inactive = copy(Y)
    Y_inactive[1 + R_idx] -= ε_pert  # First period R

    # Recompute Jacobians for perturbed states
    # (This mirrors the Jacobian assembly from lines 800-880 in deterministic solver)

    JACOBIAN_SPARSITY_TOL = 1e-14
    neq = ny_ * T
    ncols = ny_ * T
    nnz_estimate = 3 * ny_ * ny_ * T  # Conservative estimate

    # Allocate COO format arrays for both Jacobians
    rows_active = Vector{Int}(undef, nnz_estimate)
    cols_active = Vector{Int}(undef, nnz_estimate)
    vals_active = Vector{Float64}(undef, nnz_estimate)
    nnz_active = 0

    rows_inactive = Vector{Int}(undef, nnz_estimate)
    cols_inactive = Vector{Int}(undef, nnz_estimate)
    vals_inactive = Vector{Float64}(undef, nnz_estimate)
    nnz_inactive = 0

    # Assemble J_active
    for t in 1:T
        y_lag = view(Y_active, 1 .+ (t-1)*ny_ .+ (1:ny_))
        y_cur = view(Y_active, 1 .+ t*ny_ .+ (1:ny_))
        y_fwd = (t < T) ? view(Y_active, 1 .+ (t+1)*ny_ .+ (1:ny_)) : yss
        ε_t = view(opts.deterministic_shocks, t, :)

        fill_dyn_values!(dyn_values, var_kind, var_idx, y_lag, y_cur, y_fwd, ε_t)
        dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

        rr = (t-1)*ny_ .+ (1:ny_)
        c_cur = (t-1)*ny_ .+ (1:ny_)
        c_lag = (t-2)*ny_ .+ (1:ny_)
        c_fwd = t*ny_ .+ (1:ny_)

        for i in 1:ny_
            r_row = rr[i]
            for j in 1:length(vars_raw)
                v = jac_buffer[i, j]
                if abs(v) > JACOBIAN_SPARSITY_TOL
                    kind = var_kind[j]
                    idx = var_idx[j]
                    if kind == :present
                        nnz_active += 1
                        rows_active[nnz_active] = r_row
                        cols_active[nnz_active] = c_cur[idx]
                        vals_active[nnz_active] = v
                    elseif kind == :past && t > 1
                        nnz_active += 1
                        rows_active[nnz_active] = r_row
                        cols_active[nnz_active] = c_lag[idx]
                        vals_active[nnz_active] = v
                    elseif kind == :future && t < T
                        nnz_active += 1
                        rows_active[nnz_active] = r_row
                        cols_active[nnz_active] = c_fwd[idx]
                        vals_active[nnz_active] = v
                    end
                end
            end
        end
    end

    # Assemble J_inactive
    for t in 1:T
        y_lag = view(Y_inactive, 1 .+ (t-1)*ny_ .+ (1:ny_))
        y_cur = view(Y_inactive, 1 .+ t*ny_ .+ (1:ny_))
        y_fwd = (t < T) ? view(Y_inactive, 1 .+ (t+1)*ny_ .+ (1:ny_)) : yss
        ε_t = view(opts.deterministic_shocks, t, :)

        fill_dyn_values!(dyn_values, var_kind, var_idx, y_lag, y_cur, y_fwd, ε_t)
        dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

        rr = (t-1)*ny_ .+ (1:ny_)
        c_cur = (t-1)*ny_ .+ (1:ny_)
        c_lag = (t-2)*ny_ .+ (1:ny_)
        c_fwd = t*ny_ .+ (1:ny_)

        for i in 1:ny_
            r_row = rr[i]
            for j in 1:length(vars_raw)
                v = jac_buffer[i, j]
                if abs(v) > JACOBIAN_SPARSITY_TOL
                    kind = var_kind[j]
                    idx = var_idx[j]
                    if kind == :present
                        nnz_inactive += 1
                        rows_inactive[nnz_inactive] = r_row
                        cols_inactive[nnz_inactive] = c_cur[idx]
                        vals_inactive[nnz_inactive] = v
                    elseif kind == :past && t > 1
                        nnz_inactive += 1
                        rows_inactive[nnz_inactive] = r_row
                        cols_inactive[nnz_inactive] = c_lag[idx]
                        vals_inactive[nnz_inactive] = v
                    elseif kind == :future && t < T
                        nnz_inactive += 1
                        rows_inactive[nnz_inactive] = r_row
                        cols_inactive[nnz_inactive] = c_fwd[idx]
                        vals_inactive[nnz_inactive] = v
                    end
                end
            end
        end
    end

    # Build sparse Jacobians
    J_active = sparse(view(rows_active, 1:nnz_active), view(cols_active, 1:nnz_active),
                      view(vals_active, 1:nnz_active), neq, ncols)
    J_inactive = sparse(view(rows_inactive, 1:nnz_inactive), view(cols_inactive, 1:nnz_inactive),
                        view(vals_inactive, 1:nnz_inactive), neq, ncols)

    # Adaptive α selection
    α_opt = adaptive_alpha_selection_sparse(R, J_active, J_inactive;
        max_iter=opts.subdiff_alpha_maxit,
        tol=opts.subdiff_alpha_tol
    )

    opts.subdiff_verbose && @info "Subdifferential α selected" α=α_opt

    # Convex combination
    J_sub = α_opt * J_active + (1.0 - α_opt) * J_inactive

    # Newton step
    try
        Δ = -(J_sub \ R)
        opts.subdiff_verbose && @info "Subdifferential Newton succeeded" norm_Δ=norm(Δ)
        return Δ
    catch e
        opts.subdiff_verbose && @warn "Subdifferential Newton failed" exception=e
        return nothing  # Subdifferential also failed
    end
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
    initial_state::Union{Nothing,Vector{Float64}}=nothing;
    calib_override::Union{Nothing,Dict{Symbol,Float64}}=nothing
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
    params_and_ss = build_parameters_and_ss_values(parameters_and_SS, parameters, 𝓂, yss, SS_result; calib_override=calib_override)

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

    line_search_enabled = opts.line_search && opts.line_search_maxit > 0
    lm_lambda = opts.lm_lambda
    Y_trial = similar(Y)

    function det_residual_err!(R_out::Vector{Float64}, Y_in::Vector{Float64})
        fill!(R_out, 0.0)
        for t in 1:T
            y_lag = view(Y_in, 1 .+ (t-1)*ny_ .+ (1:ny_))
            y_cur = view(Y_in, 1 .+ t*ny_ .+ (1:ny_))
            y_fwd = (t < T) ? view(Y_in, 1 .+ (t+1)*ny_ .+ (1:ny_)) : yss
            ε_t = view(opts.deterministic_shocks, t, :)

            fill_dyn_values!(dyn_values, var_kind, var_idx, y_lag, y_cur, y_fwd, ε_t)
            dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
            if has_nonfinite(resid_buffer)
                return Inf
            end
            rr = (t-1)*ny_ .+ (1:ny_)
            R_out[rr] .= resid_buffer
        end
        return maximum(abs, R_out)
    end

    # Newton solver
    neq = ny_ * T  # T periods of equations (y₀ is initial condition, fixed)
    nnz_est = 3 * ny_ * ny_ * T  # Tridiagonal block structure
    rows = Vector{Int}(undef, nnz_est)
    cols = Vector{Int}(undef, nnz_est)
    vals = Vector{Float64}(undef, nnz_est)
    R = zeros(neq)  # Declare outside loop so it's available after loop ends
    R_trial = similar(R)
    err = Inf  # Initialize error
    nonfinite_reports_det = 0  # FIX H-05: Count nonfinite reports for deterministic solver
    max_nonfinite_reports_det = 5  # FIX H-05: Allow reporting first N occurrences

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
            if has_nonfinite(resid_buffer)
                # FIX H-05: Report first N nonfinite occurrences
                if opts.verbose && nonfinite_reports_det < max_nonfinite_reports_det
                    report_nonfinite_residual(
                        𝓂,
                        resid_buffer,
                        dyn_values,
                        params_and_ss,
                        vars_raw,
                        parameters_and_SS;
                        t = t,
                        g = 1
                    )
                    nonfinite_reports_det += 1
                    if nonfinite_reports_det == max_nonfinite_reports_det
                        @warn "Suppressing further nonfinite reports (max $max_nonfinite_reports_det reached)"
                    end
                end
                return (flag=2, Y=Y, layout=layout, err=Inf, eq_residuals=Float64[])
            end
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
                    if abs(v) > JACOBIAN_SPARSITY_TOL
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

        if it % 10 == 0  || it == 1 || err < opts.tol
            opts.verbose && @info "Deterministic path it=$it/$(opts.maxit)" max_res=err
        end

        if err < opts.tol
            eq_resid = compute_per_equation_residuals(R, layout, ny_)
            opts.verbose && @info "✓ Deterministic path converged" iterations=it err=err binding_equation=argmax(eq_resid)
            return (flag=0, Y=Y, layout=layout, err=err, eq_residuals=eq_resid)
        end

        # Newton step with regularization
        Δ = nothing
        try
            Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
        catch e
            if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                # Try subdifferential Newton first (if enabled)
                if opts.use_subdifferential
                    opts.subdiff_verbose && @info "Caught singularity, trying subdifferential Newton..."
                    Δ_sub = try_subdifferential_newton_step_deterministic(
                        Y, R, 𝓂, T, ny_, dyn_jac_func, params_and_ss,
                        dyn_values, var_kind, var_idx, jac_buffer, vars_raw, yss, opts
                    )
                    if !isnothing(Δ_sub)
                        Δ = Δ_sub
                        opts.subdiff_verbose && @info "Subdifferential Newton succeeded at it=$it"
                    else
                        # Subdifferential failed - increase regularization and retry
                        lm_lambda = min(lm_lambda * opts.lm_lambda_scale, opts.lm_lambda_max)
                        try
                            Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
                        catch e2
                            # Increased regularization also failed - rethrow original exception
                            rethrow(e)
                        end
                    end
                else
                    # Subdifferential not enabled - increase regularization and retry
                    lm_lambda = min(lm_lambda * opts.lm_lambda_scale, opts.lm_lambda_max)
                    try
                        Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
                    catch e2
                        # Increased regularization also failed - rethrow original exception
                        rethrow(e)
                    end
                end
            else
                rethrow()
            end
        end

        # Adaptive damping (see constants at top of file)
        alpha_init = err > ADAPTIVE_DAMP_HIGH ? 0.5 : (err > ADAPTIVE_DAMP_MED ? 0.7 : 1.0)

        if line_search_enabled
            err_current = err
            best_err = err_current
            best_alpha = 0.0
            alpha = alpha_init

            for _ in 1:opts.line_search_maxit
                Y_trial .= Y
                Y_trial[(ny_+2):end] .+= alpha * Δ

                err_trial = det_residual_err!(R_trial, Y_trial)
                if isfinite(err_trial) && err_trial < best_err
                    best_err = err_trial
                    best_alpha = alpha
                    break
                end

                alpha *= opts.line_search_factor
                if alpha < opts.line_search_min_alpha
                    break
                end
            end

            if best_alpha > 0.0
                Y[(ny_+2):end] .+= best_alpha * Δ
                err = best_err
                lm_lambda = max(lm_lambda / opts.lm_lambda_scale, opts.lm_lambda_min)
                if err < opts.tol
                    eq_resid = compute_per_equation_residuals(R, layout, ny_)
                    opts.verbose && @info "✓ Deterministic path converged" iterations=it err=err binding_equation=argmax(eq_resid)
                    return (flag=0, Y=Y, layout=layout, err=err, eq_residuals=eq_resid)
                end
            else
                lm_lambda = min(lm_lambda * opts.lm_lambda_scale, opts.lm_lambda_max)
            end
        else
            # Apply update to y₁, ..., yT (keep y₀ fixed at steady state)
            Y[(ny_+2):end] .+= alpha_init * Δ
        end
    end

    # Did not converge
    eq_resid = compute_per_equation_residuals(R, layout, ny_)
    @warn "Deterministic path did not converge" max_iterations=opts.maxit final_err=err binding_equation=argmax(eq_resid)
    return (flag=1, Y=Y, layout=layout, err=err, eq_residuals=eq_resid)
end

"""Evaluate a simple Julia Expr tree by substituting symbol values from a dict."""
function eval_expr(expr::Expr, subst::Dict{Symbol, Float64})::Float64
    if expr.head == :call
        # Function call: args[1] = operator/function symbol, args[2:end] = operands
        op = eval(expr.args[1])  # e.g. :- → -, :+ → +, :/ → /, :* → *, :log → log
        operands = map(a -> eval_expr(a, subst), expr.args[2:end])
        return op(operands...)
    else
        error("Unsupported Expr head: $(expr.head)")
    end
end
eval_expr(s::Symbol, subst::Dict{Symbol, Float64})::Float64 = haskey(subst, s) ? subst[s] : error("Unknown symbol: $s")
eval_expr(n::Number, subst::Dict{Symbol, Float64})::Float64 = Float64(n)

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
    initial_state::Union{Nothing,Vector{Float64}}=nothing,
    yss_override::Union{Nothing,Vector{Float64}}=nothing
)
    # Get model dimensions
    ny_ = length(𝓂.var)
    shock_names = 𝓂.exo
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    stochastic_idx = findall(x -> !x, obc_mask)
    n_exo = length(shock_names)
    dε = length(stochastic_idx)

    # Get steady state only when needed. For stochastic SEP calls with `yss_override`
    # (e.g. inversion-filter one-step solves), we can skip this expensive step and
    # reconstruct calibration quantities directly from `yss_override`.
    need_ss_result = isnothing(yss_override) || opts.order == 0
    SS_result = need_ss_result ? get_steady_state(𝓂; parameters=parameters, return_variables_only=false, derivatives=false) : nothing
    ss_keys::Vector{Symbol} = if need_ss_result
        try
            collect(axiskeys(SS_result, 1))
        catch
            Symbol[]
        end
    else
        Symbol[]
    end

    calib_override::Union{Nothing,Dict{Symbol,Float64}} = nothing

    if !isnothing(yss_override)
        # Use provided steady state override (e.g. when NSSS solver can't handle
        # smooth approximations like log-sum-exp ZLB)
        @assert length(yss_override) == ny_ "yss_override must have length $ny_ (got $(length(yss_override)))"
        yss = copy(yss_override)

        # Evaluate calibration params from yss_override + parameter_values
        calib_pars = Float64[]
        calib_override = Dict{Symbol,Float64}()
        if length(𝓂.calibration_equations) > 0
            subst = Dict{Symbol, Float64}()
            for (i, v) in enumerate(𝓂.var)
                subst[v] = yss_override[i]
            end
            for (i, p) in enumerate(𝓂.parameters)
                subst[p] = parameters[i]
            end
            # Each calibration equation has head :call, args = [:-, param, expr]
            # At SS: param = expr, so evaluate expr
            for (j, eq) in enumerate(𝓂.calibration_equations)
                rhs = eq.args[3]
                val = eval_expr(rhs, subst)
                push!(calib_pars, val)
                calib_override[𝓂.calibration_equations_parameters[j]] = val
            end
        end
        SS = yss_override  # calib_pars appended at line SS_and_pars = vcat(SS, calib_pars) below
    else
        SS = 𝓂.solution.non_stochastic_steady_state

        # Build yss vector - use steady state for available vars, or their definitions for auxiliary
        yss = Float64[]
        for (i, var) in enumerate(𝓂.var)
            if var ∈ ss_keys
                push!(yss, Float64(SS_result(var)))
            else
                # Fall back to NSSS for auxiliary variables not in SS_result.
                push!(yss, Float64(SS[i]))
            end
        end

        calib_pars = Float64[]
        if length(𝓂.calibration_equations) > 0
            for param in 𝓂.calibration_equations_parameters
                if param ∈ ss_keys
                    push!(calib_pars, Float64(SS_result(param)))
                else
                    @debug "Calibration parameter $param not found in steady state, skipping"
                end
            end
        end
    end
    SS_and_pars = vcat(SS, calib_pars)

    # DETERMINISTIC MODE BRANCHING: Only use perfect foresight when order == 0
    if !isnothing(opts.deterministic_shocks) && opts.order == 0
        opts.verbose && @info "Deterministic mode detected (order=0) - using perfect foresight solver"
        return solve_deterministic_path(𝓂, parameters, opts, initial_guess, yss, SS_and_pars, SS_result, initial_state; calib_override=calib_override)
    end

    # STOCHASTIC MODE: Continue with existing SEP solver
    use_nonlinear_residuals = true
    if use_nonlinear_residuals
        dyn_resid_func, dyn_jac_func, vars_raw, parameters_and_SS, resid_buffer, jac_buffer =
            get_cached_dynamic_residual_jacobian(𝓂)
        params_and_ss = build_parameters_and_ss_values(parameters_and_SS, parameters, 𝓂, yss, SS_result; calib_override=calib_override)
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
    for (i, shock_pos) in enumerate(stochastic_idx)
        shock_name = shock_names[shock_pos]
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
            if opts.verbose
                @warn "Shock std parameter $param_name not found, using default 1.0"
            end
            Σ[i, i] = 1.0
        end
    end

    if opts.verbose
        @info "Shock covariance matrix Σ:" Σ
    end

    # Setup expectation method: GH quadrature or HMC
    T = opts.periods
    Lbr = max(opts.order, 0)
    m = opts.nnodes  # Nodes per dimension (for sparse tree)
    use_hmc = (opts.sep_expectation_method == :hmc)

    # Apply global shock scaling to covariance for HMC-based sampling.
    if use_hmc && dε > 0 && opts.shock_scale != 1.0
        Σ .*= opts.shock_scale^2
    end

    # Build shock nodes or prepare HMC sampler
    if use_hmc
        # Keep sparse branching structure in HMC mode so order=1 remains stochastic.
        if dε == 0
            X = zeros(0, 1)
            W = [1.0]
            K = 1
            shock_map = nothing
        elseif opts.sparse_tree
            X, W, shock_map = build_sparse_shock_nodes(m, dε, Σ)
            K = size(X, 2)
        else
            error("HMC expectation currently requires sep_sparse_tree=true for tractable branching.")
        end
        if opts.verbose
            @info "Using HMC expectation method ($(opts.hmc_samples) samples, branching K=$K)"
        end
    elseif opts.sparse_tree
        # Sparse tree: monomial rule (H*m nodes instead of m^H)
        X, W, shock_map = build_sparse_shock_nodes(m, dε, Σ)
        K = size(X, 2)  # K = dε * m for sparse tree
    else
        # Full tree: tensor product (m^H nodes)
        X, W = gh_tensor_nodes_weights(m, dε)
        K = size(X, 2)  # K = m^dε for full tree
        shock_map = nothing
    end

    if !use_hmc
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

    # OBC enforcement: detect bounds and set up projection
    obc_bounds = OBCBound[]
    if opts.enforce_obc
        obc_bounds = detect_obc_bounds(𝓂, parameters)
        if opts.verbose && !isempty(obc_bounds)
            @info "OBC enforcement enabled" n_bounds=length(obc_bounds)
            for b in obc_bounds
                floor_val = b.transform == :log ? exp(b.bound_value) : b.bound_value
                @info "  Bound: $(b.var_name) >= $floor_val ($(b.transform) transform, bound=$(b.bound_value))"
            end
        elseif opts.verbose
            @info "OBC enforcement enabled but no bounds detected"
        end
    end

    # Pre-allocate workspace
    neq = ny_ * sum(G[2:end-1])
    nnz_est = estimate_nnz(layout, T, ny_, K, Lbr)
    rows = Vector{Int}(undef, nnz_est)
    cols = Vector{Int}(undef, nnz_est)
    vals = Vector{Float64}(undef, nnz_est)

    # OPTIMIZATION Wave 5: Jacobian sparsity pattern caching
    # After first iteration, pattern is fixed - only values change
    # Saves 25-35% per iteration by avoiding COO reconstruction
    cached_rows = Vector{Int}(undef, 0)
    cached_cols = Vector{Int}(undef, 0)
    cached_vals = Vector{Float64}(undef, 0)
    pattern_cached = false

    R = zeros(neq)
    R_trial = similar(R)
    err = Inf

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
    eps_det = isnothing(opts.deterministic_shocks) ? nothing : zeros(n_exo)
    eps_full = zeros(n_exo)
    nonfinite_reports = 0  # FIX H-05: Count nonfinite reports instead of boolean flag
    max_nonfinite_reports = 5  # FIX H-05: Allow reporting first N occurrences
    active_solver = opts.linear_solver
    best_err = Inf
    stall_count = 0
    line_search_enabled = opts.line_search && opts.line_search_maxit > 0
    lm_lambda = opts.lm_lambda
    Y_trial = similar(Y)

    # Helper to get state
    function get_y(t, g)
        (t >= T+1) ? yss : view(Y, index_y(layout, t, g))
    end

    function nearest_shock_node_idx_hmc(ε_shock::AbstractVector{<:Real})
        if dε == 0 || X === nothing || size(X, 2) == 0
            return 1
        end
        best_k = 1
        best_dist = Inf
        @inbounds for k in 1:size(X, 2)
            dist = 0.0
            for i in 1:dε
                δ = ε_shock[i] - X[i, k]
                dist += δ * δ
            end
            if dist < best_dist
                best_dist = dist
                best_k = k
            end
        end
        return best_k
    end

    function sep_residual_err!(R_out::Vector{Float64}, Y_in::Vector{Float64})
        fill!(R_out, 0.0)
        for t in 1:T
            Gt = G[t+1]
            for g in 1:Gt
                pg = layout.sparse ? parent_group_sparse(layout, t, g) : parent_group(layout, t, g)
                yl = view(Y_in, index_y(layout, t-1, pg))
                yc = view(Y_in, index_y(layout, t, g))
                cgs = cgs_cache[(t, g)]

                node_branches = if layout.sparse
                    (t <= Lbr) && (g == 1) && (dε > 0)
                else
                    (t <= Lbr) && (dε > 0)
                end

                if node_branches
                    fill!(r_sum, 0.0)
                    rr = row_range(layout, t, g)

                    if use_hmc
                        has_current_stochastic_shock = (t > 1) && (dε > 0)
                        has_branch_uncertainty = length(cgs) > 1
                        needs_hmc_sampling = (dε > 0) && (has_current_stochastic_shock || has_branch_uncertainty)
                        # HMC-based expectation approximation
                        # Get shock covariance for this period
                        Σ_period = if dε > 0
                            Σ
                        else
                            zeros(dε, dε)
                        end

                        # Define residual function for HMC
                        function residual_func_at_node(ε_shock)
                            # Select next-period child group from sampled shock if branching is active.
                            cg_hmc = if length(cgs) > 1
                                cgs[min(nearest_shock_node_idx_hmc(ε_shock), length(cgs))]
                            else
                                first(cgs)
                            end
                            yl1_hmc = (t >= T) ? yss : view(Y_in, index_y(layout, t+1, cg_hmc))

                            # Setup shocks
                            if eps_det !== nothing
                                eps_full_hmc = copy(view(opts.deterministic_shocks, t, :))
                            else
                                eps_full_hmc = zeros(n_exo)
                            end

                            if has_current_stochastic_shock
                                eps_full_hmc[stochastic_idx] .= ε_shock
                            end

                            # Evaluate residual
                            dyn_values_hmc = similar(dyn_values)
                            fill_dyn_values!(dyn_values_hmc, var_kind, var_idx, yl, yc, yl1_hmc, eps_full_hmc)
                            R_local = similar(resid_buffer)
                            dyn_resid_func(R_local, params_and_ss, dyn_values_hmc)
                            return R_local
                        end

                        # Skip HMC only when there is no branch uncertainty and no stochastic innovation.
                        R_mean = if needs_hmc_sampling
                            hmc_expectation(
                                residual_func_at_node,
                                yc,
                                Σ_period,
                                opts.hmc_samples;
                                warmup = opts.hmc_warmup,
                                leapfrog_steps = opts.hmc_leapfrog_steps,
                                step_size = opts.hmc_step_size,
                                use_tempering = opts.hmc_use_tempering,
                                temperatures = opts.hmc_temperatures,
                                verbose = opts.hmc_verbose
                            )[1]
                        else
                            residual_func_at_node(zeros(dε))
                        end

                        r_sum .= R_mean
                    else
                        # Standard Gauss-Hermite quadrature
                        for (kidx, cg) in enumerate(cgs)
                            yl1 = (t >= T) ? yss : view(Y_in, index_y(layout, t+1, cg))

                            if layout.sparse
                                shock_idx = kidx
                                ε_to_child = view(X, :, shock_idx)
                                wk = W[shock_idx]
                            else
                                k_shock = mod(g - 1, K) + 1
                                ε_to_child = view(X, :, k_shock)
                                wk = W[kidx]
                            end

                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end

                            if t > 1 && dε > 0
                                eps_full[stochastic_idx] .= ε_to_child
                            end

                            ε_curr = eps_full
                            fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, ε_curr)
                            dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                            if has_nonfinite(resid_buffer)
                                return Inf
                            end
                            r_sum .+= wk .* resid_buffer
                        end
                    end

                    R_out[rr] .= r_sum
                else
                    cg = first(cgs)
                    yl1 = (t >= T) ? yss : view(Y_in, index_y(layout, t+1, cg))

                    if layout.sparse && g > 1
                        info = branch_info_sparse(layout, g)
                        if !isnothing(info) && t == info[1]
                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end
                            if dε > 0
                                eps_full[stochastic_idx] .= view(X, :, info[2])
                            end
                        else
                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end
                        end
                    else
                        if eps_det !== nothing
                            eps_det .= view(opts.deterministic_shocks, t, :)
                            eps_full .= eps_det
                        else
                            fill!(eps_full, 0.0)
                        end
                    end

                    ε_curr = eps_full
                    fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, ε_curr)
                    dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                    if has_nonfinite(resid_buffer)
                        return Inf
                    end

                    rr = row_range(layout, t, g)
                    R_out[rr] .= resid_buffer
                end
            end
        end
        # Augment residual with OBC penalty if enforcement is active
        if opts.enforce_obc && !isempty(obc_bounds)
            augment_residual_obc!(R_out, Y_in, layout, obc_bounds, T, opts.obc_penalty_weight)
        end
        return maximum(abs, R_out)
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

                    if use_hmc
                        has_current_stochastic_shock = (t > 1) && (dε > 0)
                        has_branch_uncertainty = length(cgs) > 1
                        needs_hmc_sampling = (dε > 0) && (has_current_stochastic_shock || has_branch_uncertainty)
                        # HMC-based expectation approximation for residual and Jacobian
                        # Get shock covariance for this period
                        Σ_period = if dε > 0
                            Σ
                        else
                            zeros(dε, dε)
                        end

                        # Define residual function for HMC
                        function residual_func_at_node_jac(ε_shock)
                            # Select next-period child group from sampled shock if branching is active.
                            cg_hmc = if length(cgs) > 1
                                cgs[min(nearest_shock_node_idx_hmc(ε_shock), length(cgs))]
                            else
                                first(cgs)
                            end
                            yl1_hmc = get_y(t+1, cg_hmc)

                            # Setup shocks
                            if eps_det !== nothing
                                eps_full_hmc = copy(view(opts.deterministic_shocks, t, :))
                            else
                                eps_full_hmc = zeros(n_exo)
                            end

                            if has_current_stochastic_shock
                                eps_full_hmc[stochastic_idx] .= ε_shock
                            end

                            # Evaluate residual
                            dyn_values_hmc = similar(dyn_values)
                            fill_dyn_values!(dyn_values_hmc, var_kind, var_idx, yl, yc, yl1_hmc, eps_full_hmc)
                            R_local = similar(resid_buffer)
                            dyn_resid_func(R_local, params_and_ss, dyn_values_hmc)
                            return R_local
                        end

                        # Skip HMC only when there is no branch uncertainty and no stochastic innovation.
                        R_mean = if needs_hmc_sampling
                            hmc_expectation(
                                residual_func_at_node_jac,
                                yc,
                                Σ_period,
                                opts.hmc_samples;
                                warmup = opts.hmc_warmup,
                                leapfrog_steps = opts.hmc_leapfrog_steps,
                                step_size = opts.hmc_step_size,
                                use_tempering = opts.hmc_use_tempering,
                                temperatures = opts.hmc_temperatures,
                                verbose = opts.hmc_verbose
                            )[1]
                        else
                            residual_func_at_node_jac(zeros(dε))
                        end

                        r_sum .= R_mean

                        # For Jacobian: use weighted sparse-node aggregation (GH-style) for stability.
                        # Residual remains HMC-based.
                        for (kidx, cg) in enumerate(cgs)
                            yl1_jac = get_y(t+1, cg)

                            if layout.sparse
                                shock_idx = kidx
                                ε_to_child = view(X, :, shock_idx)
                                wk = W[shock_idx]
                            else
                                # HMC currently requires sparse tree; keep defensive fallback.
                                k_shock = mod(g - 1, K) + 1
                                ε_to_child = view(X, :, k_shock)
                                wk = W[kidx]
                            end

                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end

                            if t > 1 && dε > 0
                                eps_full[stochastic_idx] .= ε_to_child
                            end

                            fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1_jac, eps_full)
                            dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

                            if t+1 <= T
                                c_rng = index_y(layout, t+1, cg)
                            end

                            for irow in 1:ny_
                                r_row = first(rr) + irow - 1
                                for j in 1:length(vars_raw)
                                    v = jac_buffer[irow, j]
                                    if abs(v) > JACOBIAN_SPARSITY_TOL
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
                    else
                        # Standard Gauss-Hermite quadrature
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

                            # Dynare convention: at t=1 use deterministic shocks only; at t>=2 use node shocks
                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end

                            if t > 1 && dε > 0
                                eps_full[stochastic_idx] .= ε_to_child
                            end

                            ε_curr = eps_full

                            fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, ε_curr)
                            dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                            if has_nonfinite(resid_buffer)
                                # FIX H-05: Report first N nonfinite occurrences
                                if opts.verbose && nonfinite_reports < max_nonfinite_reports
                                    report_nonfinite_residual(
                                        𝓂,
                                        resid_buffer,
                                        dyn_values,
                                        params_and_ss,
                                        vars_raw,
                                        parameters_and_SS;
                                        t = t,
                                        g = g
                                    )
                                    nonfinite_reports += 1
                                    if nonfinite_reports == max_nonfinite_reports
                                        @warn "Suppressing further nonfinite reports (max $max_nonfinite_reports reached)"
                                    end
                                end
                                return (flag=2, Y=Y, layout=layout, err=Inf, eq_residuals=Float64[])
                            end
                            dyn_jac_func(jac_buffer, params_and_ss, dyn_values)

                            r_sum .+= wk .* resid_buffer

                            if t+1 <= T
                                c_rng = index_y(layout, t+1, cg)
                            end

                            for irow in 1:ny_
                                r_row = first(rr) + irow - 1
                                for j in 1:length(vars_raw)
                                    v = jac_buffer[irow, j]
                                    if abs(v) > JACOBIAN_SPARSITY_TOL
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
                                if abs(v) > JACOBIAN_SPARSITY_TOL
                                    nnz_count += 1
                                    rows[nnz_count] = r_gl
                                    cols[nnz_count] = c_lag.start + jcol - 1
                                    vals[nnz_count] = v
                                end
                            end
                        end
                        for jcol in 1:ny_
                            v = J_cur[irow, jcol]
                            if abs(v) > JACOBIAN_SPARSITY_TOL
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
                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end
                            if dε > 0
                                eps_full[stochastic_idx] .= view(X, :, info[2])
                            end
                        else
                            if eps_det !== nothing
                                eps_det .= view(opts.deterministic_shocks, t, :)
                                eps_full .= eps_det
                            else
                                fill!(eps_full, 0.0)
                            end
                        end
                    else
                        # Full tree non-branching or sparse trunk after branching period
                        if eps_det !== nothing
                            eps_det .= view(opts.deterministic_shocks, t, :)
                            eps_full .= eps_det
                        else
                            fill!(eps_full, 0.0)
                        end
                    end

                    ε_curr = eps_full

                    fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, ε_curr)
                    dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                    if has_nonfinite(resid_buffer)
                        # FIX H-05: Report first N nonfinite occurrences
                        if opts.verbose && nonfinite_reports < max_nonfinite_reports
                            report_nonfinite_residual(
                                𝓂,
                                resid_buffer,
                                dyn_values,
                                params_and_ss,
                                vars_raw,
                                parameters_and_SS;
                                t = t,
                                g = g
                            )
                            nonfinite_reports += 1
                            if nonfinite_reports == max_nonfinite_reports
                                @warn "Suppressing further nonfinite reports (max $max_nonfinite_reports reached)"
                            end
                        end
                        return (flag=2, Y=Y, layout=layout, err=Inf, eq_residuals=Float64[])
                    end
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
                            if abs(v) > JACOBIAN_SPARSITY_TOL
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
        # OPTIMIZATION Wave 5: Cache sparsity pattern after first iteration
        if !pattern_cached
            # First iteration: build pattern and cache structure
            J = sparse(view(rows, 1:nnz_count), view(cols, 1:nnz_count),
                      view(vals, 1:nnz_count), neq, length(Y))

            # Cache the pattern (indices only, not values)
            cached_rows = copy(view(rows, 1:nnz_count))
            cached_cols = copy(view(cols, 1:nnz_count))
            cached_vals = Vector{Float64}(undef, nnz_count)
            pattern_cached = true

            if opts.verbose
                sparsity_ratio = nnz_count / (neq * length(Y))
                @info "Jacobian pattern cached" nnz=nnz_count sparsity=string(round(100*sparsity_ratio, digits=2), "%")
            end
        else
            # Subsequent iterations: reuse pattern, only copy values
            # This saves ~25-35% by avoiding COO index reconstruction

            # Check if pattern changed (shouldn't happen but can with parameter changes)
            if nnz_count != length(cached_vals)
                # Pattern changed - rebuild cache
                J = sparse(view(rows, 1:nnz_count), view(cols, 1:nnz_count),
                          view(vals, 1:nnz_count), neq, length(Y))
                cached_rows = copy(view(rows, 1:nnz_count))
                cached_cols = copy(view(cols, 1:nnz_count))
                cached_vals = Vector{Float64}(undef, nnz_count)
                copyto!(cached_vals, view(vals, 1:nnz_count))
            else
                # Pattern unchanged - fast path
                copyto!(cached_vals, view(vals, 1:nnz_count))
                J = sparse(cached_rows, cached_cols, cached_vals, neq, length(Y))
            end
        end

        # OBC penalty augmentation: add penalty to residual for constraint violations
        if opts.enforce_obc && !isempty(obc_bounds)
            n_pen = augment_residual_obc!(R, Y, layout, obc_bounds, T, opts.obc_penalty_weight)
            if opts.verbose && it == 1 && n_pen > 0
                @info "OBC penalty: $n_pen violations penalized in residual"
            end
        end

        # Check current residual
        err = maximum(abs, R)
        if err < opts.tol
            eq_resid = compute_per_equation_residuals(R, layout, ny_)
            if opts.verbose
                binding_eq = argmax(eq_resid)
                @info "✓ SEP converged (err=$err)" binding_equation=binding_eq eq_residuals=eq_resid
            end
            return (flag=0, Y=Y, layout=layout, err=err, eq_residuals=eq_resid)
        end

        # Solve Newton step with regularization
        # OPTIMIZATION Wave 4: QR decomposition preferred for sparse J
        # Performance: QR is 30-60% faster than J'*J for typical SEP Jacobians
        # - Avoids forming dense J'*J (nnz(J'*J) >> nnz(J) for sparse J)
        # - Exploits sparsity directly in QR factorization
        # - More numerically stable (condition number κ(J) vs κ(J'*J) = κ(J)²)
        if active_solver == :normal_equations
            Δ = nothing
            try
                Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
            catch e
                if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                    # Try subdifferential Newton first (if enabled)
                    if opts.use_subdifferential
                        Δ_sub = try_subdifferential_newton_step_deterministic(
                            Y, R, 𝓂, T, ny_, dyn_jac_func, params_and_ss,
                            dyn_values, var_kind, var_idx, jac_buffer, vars_raw, yss, opts
                        )
                        if !isnothing(Δ_sub)
                            Δ = Δ_sub
                            opts.subdiff_verbose && @info "Subdifferential Newton succeeded at it=$it (stochastic SEP)"
                        else
                            # Fall back to LM regularization
                            lm_lambda = min(lm_lambda * opts.lm_lambda_scale, opts.lm_lambda_max)
                            try
                                Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
                            catch
                                if !isnothing(opts.fallback_solver)
                                    active_solver = opts.fallback_solver
                                    Δ = J \ (-R)
                                else
                                    rethrow()
                                end
                            end
                        end
                    else
                        # Standard LM fallback (existing behavior)
                        lm_lambda = min(lm_lambda * opts.lm_lambda_scale, opts.lm_lambda_max)
                        try
                            Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
                        catch
                            if !isnothing(opts.fallback_solver)
                                active_solver = opts.fallback_solver
                                Δ = J \ (-R)
                            else
                                rethrow()
                            end
                        end
                    end
                else
                    rethrow()
                end
            end
        elseif active_solver == :qr
            # QR decomposition: J \ (-R) uses sparse QR internally
            # For sparse J, this avoids O(nnz²) formation of J'*J
            Δ = nothing
            try
                Δ = J \ (-R)
            catch e
                if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException
                    # Try subdifferential Newton first (if enabled)
                    if opts.use_subdifferential
                        Δ_sub = try_subdifferential_newton_step_deterministic(
                            Y, R, 𝓂, T, ny_, dyn_jac_func, params_and_ss,
                            dyn_values, var_kind, var_idx, jac_buffer, vars_raw, yss, opts
                        )
                        if !isnothing(Δ_sub)
                            Δ = Δ_sub
                            opts.subdiff_verbose && @info "Subdifferential Newton succeeded at it=$it (QR solver)"
                        else
                            # Subdifferential failed, try fallback solver if available
                            if !isnothing(opts.fallback_solver)
                                active_solver = opts.fallback_solver
                                Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
                            else
                                rethrow()
                            end
                        end
                    else
                        # No subdifferential Newton - try fallback or rethrow
                        if !isnothing(opts.fallback_solver)
                            active_solver = opts.fallback_solver
                            Δ = (J'*J + lm_lambda*I) \ (J'*(-R))
                        else
                            rethrow()
                        end
                    end
                else
                    rethrow()
                end
            end
        else
            error("Unknown SEP linear_solver=$active_solver. Use :normal_equations or :qr.")
        end

        # Adaptive damping based on residual norm (see constants at top of file)
        # Start aggressive, reduce if residual is large
        alpha_init = err > ADAPTIVE_DAMP_HIGH ? 0.5 : (err > ADAPTIVE_DAMP_MED ? 0.7 : 1.0)
        if active_solver == :qr
            # RISK-6: Residual-dependent QR damping cap.
            # Near convergence (err < 0.01), allow larger steps for fast terminal convergence.
            # Far from solution (err > 10), cap aggressively to prevent divergence.
            qr_cap = err > 10.0 ? 0.1 : (err > 1.0 ? 0.2 : (err > 0.01 ? 0.4 : 0.8))
            alpha_init = min(alpha_init, qr_cap)
        end

        # Apply update with adaptive step size (keep y0 fixed)
        Δ[y0_idx] .= 0.0
        err_after = err

        if line_search_enabled
            err_current = err
            best_ls_err = err_current
            best_alpha = 0.0
            alpha = alpha_init

            for _ in 1:opts.line_search_maxit
                Y_trial .= Y
                Y_trial .+= alpha * Δ
                Y_trial[y0_idx] .= y0_fixed
                # Project trial point for OBC enforcement during line search
                if opts.enforce_obc && !isempty(obc_bounds)
                    project_obc_bounds!(Y_trial, layout, obc_bounds, T)
                end

                err_trial = sep_residual_err!(R_trial, Y_trial)
                if isfinite(err_trial) && err_trial < best_ls_err
                    best_ls_err = err_trial
                    best_alpha = alpha
                    break
                end

                alpha *= opts.line_search_factor
                if alpha < opts.line_search_min_alpha
                    break
                end
            end

            if best_alpha > 0.0
                Y .+= best_alpha * Δ
                Y[y0_idx] .= y0_fixed
                err_after = best_ls_err
                if active_solver == :normal_equations
                    lm_lambda = max(lm_lambda / opts.lm_lambda_scale, opts.lm_lambda_min)
                end
            else
                if active_solver == :normal_equations
                    lm_lambda = min(lm_lambda * opts.lm_lambda_scale, opts.lm_lambda_max)
                end
            end
        else
            Y .+= alpha_init * Δ
            Y[y0_idx] .= y0_fixed
        end

        # OBC projection: clamp constrained variables to satisfy bounds
        if opts.enforce_obc && !isempty(obc_bounds)
            n_proj = project_obc_bounds!(Y, layout, obc_bounds, T)
            if opts.verbose && n_proj > 0 && (it <= 3 || it % 10 == 0)
                @info "OBC projection: clamped $n_proj variable-period entries at it=$it"
            end
        end

        if !isfinite(best_err)
            best_err = err_after
            stall_count = 0
        else
            improvement_tol = max(opts.stall_abs_tol, opts.stall_rel_tol * best_err)
            if err_after + improvement_tol < best_err
                best_err = err_after
                stall_count = 0
            else
                stall_count += 1
            end
        end

        if active_solver == :normal_equations &&
           !isnothing(opts.fallback_solver) &&
           stall_count >= opts.stall_iters
            opts.verbose && @info "SEP linear solver switch" from=active_solver to=opts.fallback_solver err=err_after
            active_solver = opts.fallback_solver
            stall_count = 0
        end

        if opts.verbose && (it % 10 == 0  || it == 1)
            @info "SEP it=$it/$(opts.maxit)  max|res|=$err_after  step_norm=$(norm(Δ))"
        end

        if err_after < opts.tol
            eq_resid = compute_per_equation_residuals(R, layout, ny_)
            if opts.verbose
                binding_eq = argmax(eq_resid)
                @info "✓ SEP converged (err=$err_after)" binding_equation=binding_eq eq_residuals=eq_resid
            end
            return (flag=0, Y=Y, layout=layout, err=err_after, eq_residuals=eq_resid)
        end

        err = err_after
    end

    # Did not converge — compute per-equation breakdown for diagnostics
    eq_resid = compute_per_equation_residuals(R, layout, ny_)
    if opts.verbose
        binding_eq = argmax(eq_resid)
        @warn "SEP did not converge in $(opts.maxit) iterations (err=$err)" binding_equation=binding_eq eq_residuals=eq_resid
    else
        @warn "SEP did not converge in $(opts.maxit) iterations (err=$err)"
    end
    return (flag=1, Y=Y, layout=layout, err=err, eq_residuals=eq_resid)
end
