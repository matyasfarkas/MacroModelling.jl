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

    function SEPSolverOptions(;
        periods=20,
        order=1,
        nnodes=3,
        maxit=80,
        tol=1e-7,
        verbose=true,
        shock_scale=1.0
    )
        new(periods, order, nnodes, maxit, tol, verbose, shock_scale)
    end
end

"""
Layout describing SEP branching tree structure.
"""
struct SEPLayout
    T::Int                    # Periods
    Lbr::Int                  # Branching order
    K::Int                    # Total GH nodes
    G::Vector{Int}            # Groups at each time
    voff::Vector{Int}         # Variable offsets
    eoff::Vector{Int}         # Shock offsets
    ny_::Int                  # Number of variables
    dε::Int                   # Number of shocks
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
    yss = [Float64(SS_result(var)) for var in 𝓂.var]

    # Get Jacobian
    SS = 𝓂.solution.non_stochastic_steady_state
    calib_pars = Float64[]
    if length(𝓂.calibration_equations) > 0
        # Get available keys from SS_result
        ss_keys = try
            axiskeys(SS_result, 1)
        catch
            Symbol[]  # If SS_result is not a KeyedArray
        end

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
    X, W = gh_tensor_nodes_weights(opts.nnodes, dε)
    K = size(X, 2)

    if opts.shock_scale != 1.0
        X .*= opts.shock_scale
    end
    if dε > 0
        transform_nodes!(X, Σ)
    end

    # Build tree structure
    G = Vector{Int}(undef, T+2)
    for t in 0:T+1
        if t == 0
            G[t+1] = 1
        elseif t <= Lbr
            G[t+1] = K^t
        else
            G[t+1] = K^Lbr
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

    layout = SEPLayout(T, Lbr, K, G, voff, eoff, ny_, dε)

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

    # Cache child groups
    cgs_cache = Dict{Tuple{Int,Int}, Vector{Int}}()
    for t in 1:T, g in 1:G[t+1]
        cgs_cache[(t,g)] = collect(child_groups(layout, t, g))
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
                pg = parent_group(layout, t, g)
                yl = view(Y, index_y(layout, t-1, pg))
                yc = view(Y, index_y(layout, t, g))
                cgs = cgs_cache[(t,g)]

                if t <= Lbr && dε > 0
                    # Branching node: expectation over child groups
                    fill!(r_sum, 0.0)
                    fill!(J_lag, 0.0)
                    fill!(J_cur, 0.0)

                    # Extract shock for current group g (same for all children)
                    k_shock = mod(g - 1, K) + 1
                    ε_curr = view(X, :, k_shock)

                    for (kidx, cg) in enumerate(cgs)
                        yl1 = get_y(t+1, cg)

                        # Deviations from steady state
                        Δy_lag = yl - yss
                        Δy_cur = yc - yss
                        Δy_fwd = yl1 - yss

                        # Linear approximation: r = J*(y - yss)
                        r = ∇₊ * Δy_fwd + ∇₀ * Δy_cur + ∇₋ * Δy_lag + ∇ₑ * ε_curr

                        # Jacobian blocks
                        J_lag_local = ∇₋
                        J_cur_local = ∇₀
                        J_fwd = ∇₊

                        # Use weight for enumeration index (Gauss-Hermite weight)
                        wk = W[kidx]
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
                    # Non-branching node: deterministic
                    cg = first(cgs)
                    yl1 = get_y(t+1, cg)

                    Δy_lag = yl - yss
                    Δy_cur = yc - yss
                    Δy_fwd = yl1 - yss

                    r = ∇₊ * Δy_fwd + ∇₀ * Δy_cur + ∇₋ * Δy_lag

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
