"""
Subdifferential Newton Method for Hard OBC Models

Handles singular Jacobians at non-smooth kink points (e.g., ZLB constraint max(0, R-Rbar))
using Clarke subdifferential and adaptive convex combination.

Author: Claude Code + Matyas Farkas
Date: 2026-02-08
"""

using LinearAlgebra

"""
    detect_zlb_kink(y, model; kink_tol=1e-6) -> (at_kink::Bool, kink_vars::Vector{Int}, kink_direction::Float64)

Detect if solution is at ZLB kink where max(0, R-Rbar) creates non-smooth derivative.

# Arguments
- `y::Vector{Float64}`: Current solution vector
- `model`: MacroModelling model object
- `kink_tol::Float64=1e-6`: Tolerance for kink detection

# Returns
- `at_kink::Bool`: true if within tolerance of kink point
- `kink_vars::Vector{Int}`: Indices of variables affected by kink
- `kink_direction::Float64`: +1 if approaching from above, -1 from below, 0 if at kink
"""
function detect_zlb_kink(
    y::Vector{Float64},
    model;
    kink_tol::Float64 = 1e-6
)
    # Find interest rate variable index
    R_idx = findfirst(v -> v == :r, model.var)

    if R_idx === nothing
        # No interest rate variable, no ZLB kink
        return false, Int[], 0.0
    end

    # Get current interest rate value
    R_current = y[R_idx]

    # Get R_bar parameter
    R_bar_idx = findfirst(p -> p == :R_bar, model.parameters)
    R_bar = R_bar_idx !== nothing ? model.parameter_values[R_bar_idx] : 1.0

    # Support both common OBC formulations:
    # 1) level-rate bound:     r_t = max(R_bar, ...)
    # 2) log-rate bound (HLT): log(r_t) = max(R_bar, ...)
    # Choose the representation that is closer to the current point.
    R_gap_level = R_current - R_bar
    R_gap_log = (isfinite(R_current) && R_current > 0.0) ? (log(R_current) - R_bar) : Inf
    R_gap = abs(R_gap_log) < abs(R_gap_level) ? R_gap_log : R_gap_level

    # Check if at kink
    at_kink = abs(R_gap) < kink_tol

    if !at_kink
        return false, Int[], 0.0
    end

    # At kink - identify affected equations
    # For now, assume all equations potentially affected (conservative)
    # TODO: Parse model equations to find which contain max(0, R-Rbar)
    kink_vars = collect(1:length(model.var))

    # Direction: sign of gap (or 0 if exactly at kink)
    kink_direction = abs(R_gap) < 1e-12 ? 0.0 : sign(R_gap)

    return true, kink_vars, kink_direction
end


"""
    compute_subdifferential_jacobians(y, model, kink_vars, jacobian_func; ε_pert=1e-6)
        -> (J_active::Matrix{Float64}, J_inactive::Matrix{Float64})

Compute dual Jacobians for subdifferential at kink point.

# Strategy
- J_active: Jacobian with constraint binding (R slightly > R_bar)
- J_inactive: Jacobian with constraint slack (R slightly < R_bar)

# Arguments
- `y::Vector{Float64}`: Current solution at kink
- `model`: MacroModelling model object
- `kink_vars::Vector{Int}`: Variables affected by kink
- `jacobian_func::Function`: Function to compute Jacobian, signature J = jacobian_func(y)
- `ε_pert::Float64=1e-6`: Finite difference perturbation size

# Returns
- `J_active::Matrix{Float64}`: Jacobian assuming ZLB binds
- `J_inactive::Matrix{Float64}`: Jacobian assuming ZLB slack
"""
function compute_subdifferential_jacobians(
    y::Vector{Float64},
    model,
    kink_vars::Vector{Int},
    jacobian_func::Function;
    ε_pert::Float64 = 1e-6
)
    ny = length(y)

    # Find interest rate index
    R_idx = findfirst(v -> v == :r, model.var)

    if R_idx === nothing
        # No ZLB variable, return same Jacobian twice
        J = jacobian_func(y)
        return J, copy(J)
    end

    # --- J_active: ZLB binding (R slightly > R_bar) ---
    y_active = copy(y)
    y_active[R_idx] += ε_pert

    J_active = jacobian_func(y_active)

    # --- J_inactive: ZLB slack (R slightly < R_bar) ---
    y_inactive = copy(y)
    y_inactive[R_idx] -= ε_pert

    J_inactive = jacobian_func(y_inactive)

    return J_active, J_inactive
end


"""
    adaptive_alpha_selection(R, J_active, J_inactive; max_iter=20, tol=1e-3)
        -> α_opt::Float64

Adaptively select convex combination parameter α ∈ [0,1] to maximize descent.

Minimizes: φ(α) = || R + J(α) * Δy(α) ||²
where J(α) = α*J_active + (1-α)*J_inactive
      Δy(α) = -J(α)^{-1} * R

# Algorithm
Golden section search on [0, 1] interval.

# Arguments
- `R::Vector{Float64}`: Current residual vector
- `J_active::Matrix{Float64}`: Jacobian with constraint active
- `J_inactive::Matrix{Float64}`: Jacobian with constraint inactive
- `max_iter::Int=20`: Maximum iterations for golden section search
- `tol::Float64=1e-3`: Convergence tolerance for α

# Returns
- `α_opt::Float64`: Optimal convex combination parameter
"""
function adaptive_alpha_selection(
    R::Vector{Float64},
    J_active::Matrix{Float64},
    J_inactive::Matrix{Float64};
    max_iter::Int = 20,
    tol::Float64 = 1e-3
)
    # Objective function: predicted residual norm after Newton step
    function φ(α::Float64)
        # Convex combination of Jacobians
        J_α = α * J_active + (1.0 - α) * J_inactive

        # Check conditioning
        if cond(J_α) > 1e12
            return Inf  # Penalize singular combinations
        end

        # Newton step with J_α
        try
            Δy = -(J_α \ R)

            # Predicted residual (linear approximation)
            R_new_approx = R + J_α * Δy

            return dot(R_new_approx, R_new_approx)  # ||R_new||²
        catch e
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
    subdifferential_newton_step(y, R, model, jacobian_func; kink_tol=1e-6, verbose=false)
        -> (Δy::Vector{Float64}, at_kink::Bool, α_used::Float64)

Compute Newton step using subdifferential method if at kink, otherwise standard Newton.

# Arguments
- `y::Vector{Float64}`: Current solution
- `R::Vector{Float64}`: Current residual
- `model`: MacroModelling model object
- `jacobian_func::Function`: Function to compute Jacobian
- `kink_tol::Float64=1e-6`: Tolerance for kink detection
- `verbose::Bool=false`: Print diagnostic info

# Returns
- `Δy::Vector{Float64}`: Newton step direction
- `at_kink::Bool`: Whether subdifferential method was used
- `α_used::Float64`: Convex combination parameter (NaN if standard Newton)
"""
function subdifferential_newton_step(
    y::Vector{Float64},
    R::Vector{Float64},
    model,
    jacobian_func::Function;
    kink_tol::Float64 = 1e-6,
    verbose::Bool = false
)
    # 1. Check if at kink
    at_kink, kink_vars, kink_dir = detect_zlb_kink(y, model; kink_tol=kink_tol)

    if at_kink
        # --- SUBDIFFERENTIAL NEWTON ---

        if verbose
            @info "Subdifferential Newton: at ZLB kink" kink_direction=kink_dir n_affected=length(kink_vars)
        end

        # 2. Compute dual Jacobians
        J_active, J_inactive = compute_subdifferential_jacobians(
            y, model, kink_vars, jacobian_func
        )

        # 3. Adaptive α selection
        α_sub = adaptive_alpha_selection(R, J_active, J_inactive)

        if verbose
            @info "Subdifferential α selected" α=α_sub
        end

        # 4. Convex combination
        J = α_sub * J_active + (1.0 - α_sub) * J_inactive

        # 5. Newton step
        Δy = -(J \ R)

        return Δy, true, α_sub
    else
        # --- STANDARD NEWTON ---
        J = jacobian_func(y)
        Δy = -(J \ R)

        return Δy, false, NaN
    end
end
