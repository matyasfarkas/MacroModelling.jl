# SEP Stochastic Simulation
# Sequential simulation using SEP for expectations

using Random
using LinearAlgebra

# ── ZLB enforcement via OBC anticipated-shock optimization ──────────────
#
# When enforce_zlb=true, after each period's SEP solve we check whether
# the policy rate violates the ZLB.  If so, we compute OBC anticipated
# shocks using the same NLopt SLSQP approach as MacroModelling.jl's
# obc_state_update(), then re-solve the SEP with those shocks injected.
# This gives a nonlinearly-consistent solution where the constraint holds.
# ────────────────────────────────────────────────────────────────────────

"""
    _solve_obc_shocks_for_sep(m, current_state, present_shocks, obc_shock_idx,
                               state_update, algorithm; silent)

Compute the OBC anticipated shocks that enforce occasionally-binding
constraints, using the same NLopt SLSQP approach as the linear IRF
path's `obc_state_update()`.

The optimisation problem is:
  min  sum(x.^2)   (minimum energy OBC shocks)
  s.t. obc_violation_function(x, p) <= 0   (constraints satisfied)

Returns `(optimized_shocks, solved)` where `optimized_shocks` is the full
shock vector (length nExo) with OBC shocks filled in, and `solved` is a
Bool indicating whether all constraints are satisfied.
"""
function _solve_obc_shocks_for_sep(
    m_obj::MacroModelling.ℳ,
    current_state::Vector{Float64},
    present_shocks::Vector{Float64},
    obc_shock_idx::BitVector,
    state_update::Function,
    algorithm::Symbol;
    silent::Bool=true
)
    reference_ss = m_obj.solution.non_stochastic_steady_state
    unconditional_forecast_horizon = m_obj.max_obc_horizon
    periods_per_shock = m_obj.max_obc_horizon + 1
    num_obc_shocks = sum(obc_shock_idx) ÷ periods_per_shock

    p = (current_state, state_update, reference_ss, m_obj, algorithm,
         unconditional_forecast_horizon, present_shocks)

    # Check if constraints are violated with zero OBC shocks
    n_obc_vars = num_obc_shocks * periods_per_shock
    constraints_violated = any(
        m_obj.obc_violation_function(zeros(n_obc_vars), p) .> eps(Float32)
    )

    if !constraints_violated
        return present_shocks, true
    end

    # Set up NLopt SLSQP optimization: min sum(x^2) s.t. constraints
    opt = NLopt.Opt(NLopt.:LD_SLSQP, n_obc_vars)

    opt.min_objective = obc_objective_optim_fun

    opt.xtol_abs = eps(Float32)
    opt.ftol_abs = eps(Float32)
    opt.maxeval = 500

    upper_bounds = fill(eps(), 1 + 2 * max(n_obc_vars - 1, 1))

    NLopt.inequality_constraint!(
        opt,
        (res, x, jac) -> obc_constraint_optim_fun(res, x, jac, p),
        upper_bounds
    )

    (minf, x, ret) = NLopt.optimize(opt, zeros(n_obc_vars))

    # Apply optimized OBC shocks to the shock vector
    optimized_shocks = copy(present_shocks)
    optimized_shocks[obc_shock_idx] .= x

    # Verify constraint satisfaction
    constraints_violated = any(
        m_obj.obc_violation_function(x, p) .> eps(Float32)
    )
    solved = !constraints_violated

    if !silent
        if solved
            obc_energy = sum(abs2, x)
            println("    OBC optimization converged (energy=$(round(obc_energy, digits=6)), ret=$ret)")
        else
            @warn "OBC optimization did not fully satisfy constraints (ret=$ret)"
        end
    end

    return optimized_shocks, solved
end

"""
    _build_first_order_state_update(m)

Return the first-order state_update function stored in the perturbation
solution.  This function maps (state, shocks) -> next_state in deviations
from steady state, matching the signature expected by obc_violation_function.

The state_update is `S1 * [state[past_idx]; shock]` where S1 is the
first-order solution matrix (nVars x (nPast + nExo)).
"""
function _build_first_order_state_update(m_obj::MacroModelling.ℳ)
    # The perturbation_solution struct already stores the state_update
    # closure built during solve!(:first_order).  Re-use it directly.
    return m_obj.solution.perturbation.first_order.state_update
end

"""
    simulate_sep(m; ...)

Perform stochastic simulation using SEP for forward-looking expectations.

Note: This implementation uses a pre-solved SEP tree and maps shocks to the
nearest Gauss-Hermite node. It is fast but approximate. For a Dynare-style
extended-path loop that re-solves SEP each period, use
`simulate_sep_extended_path`.

# Algorithm
For each period t:
1. Solve SEP problem from current state with T-period horizon
2. Apply realized shock for period t
3. Take first-period decision from SEP solution
4. Update state and move to next period

# Arguments
- `m`: Model
- `periods`: Total simulation periods
- `initial_state`: Starting state (default: deterministic SS)
- `shocks`: Matrix of shock realizations (nshocks x periods). If nothing, drawn from N(0,S)
- `burn_in`: Number of initial periods to discard
- `sep_horizon`: Horizon for each SEP problem (T)
- `sep_order`: SEP branching order (Lbr)
- `sep_nnodes`: Number of Gauss-Hermite nodes
- `shock_scaling`: `:none` for unit shocks, `:parameter` to scale by `z_<shock>`
- `random_seed`: Random seed for shock generation

# Returns
- `simulation`: KeyedArray of simulated paths (Variables x Time)
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
    shock_scaling::Symbol=:none,
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
        if shock_scaling != :none && shock_scaling != :parameter
            error("Unknown shock_scaling=$shock_scaling. Use :none or :parameter.")
        end

        Σ = zeros(nshocks, nshocks)
        for (i, shock_name) in enumerate(𝓂.exo)
            if shock_scaling == :parameter
                param_name = Symbol("z_", shock_name)
                param_idx = findfirst(==(param_name), 𝓂.parameters)
                if !isnothing(param_idx)
                    σ = params[param_idx]
                    Σ[i, i] = σ^2
                else
                    @warn "Shock std parameter $param_name not found, using default 1.0"
                    Σ[i, i] = 1.0
                end
            else
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
    simulate_sep_extended_path(m; ...)

Run a Dynare-style extended-path simulation by re-solving SEP at each period.

# Arguments
- `periods`: Total simulation periods (after burn-in)
- `initial_state`: Starting state (default: deterministic SS)
- `shocks`: Shock matrix (nshocks x periods+burn_in). If `nothing`, draws N(0,1)
  shocks for non-OBC shocks. If provided with only non-OBC rows, OBC shocks are
  padded with zeros.
- `burn_in`: Number of initial periods to discard
- `sep_horizon`: SEP horizon (T)
- `sep_order`: SEP branching order (Lbr)
- `sep_nnodes`: Gauss-Hermite nodes per shock dimension
- `sep_maxit`: SEP max Newton iterations
- `sep_tol`: SEP convergence tolerance
- `sep_sparse_tree`: Use fishbone sparse tree
- `sep_accept_tol`: Accept SEP solution if error below this (even if not converged)
- `shock_scaling`: `:none` for unit shocks, `:parameter` to scale by `z_<shock>`
- `random_seed`: Random seed for shock generation
- `enforce_zlb`: If true, enforce ZLB after each SEP solve. For OBC models
  (shocks with "obc" in their name), this optimizes the OBC anticipated shocks
  via NLopt SLSQP (matching MacroModelling.jl's obc_state_update) and re-solves
  SEP for nonlinear consistency. For non-OBC models, clamps the variable.
- `zlb_floor`: Floor value for the ZLB variable (default: 1.0 for gross rate)
- `zlb_variable`: Symbol of the variable to check (default: `:r`)
- `zlb_obc_maxiter`: Maximum OBC re-solve attempts per period (default: 3)
- `silent`: Suppress progress messages

# Returns
- NamedTuple with fields:
  - `simulation`: KeyedArray (Variables x Time)
  - `shocks`: shock matrix used (nshocks x time)
  - `errorflag`: true if SEP failed in some period
  - `failure_period`: first period with failure (or `nothing`)
  - `zlb_periods`: number of periods where ZLB was binding
  - `sep_errors`: per-period SEP residuals
"""
function simulate_sep_extended_path(
    𝓂::ℳ;
    periods::Int=200,
    initial_state::Union{Nothing,Vector{Float64}}=nothing,
    shocks::Union{Nothing,Matrix{Float64}}=nothing,
    burn_in::Int=0,
    sep_horizon::Int=40,
    sep_order::Int=1,
    sep_nnodes::Int=3,
    sep_maxit::Int=80,
    sep_tol::Float64=1e-7,
    sep_sparse_tree::Bool=true,
    sep_linear_solver::Symbol=:normal_equations,
    sep_fallback_solver::Union{Symbol,Nothing}=nothing,
    sep_stall_iters::Int=25,
    sep_stall_rel_tol::Float64=1e-4,
    sep_stall_abs_tol::Float64=1e-10,
    sep_line_search::Bool=true,
    sep_line_search_maxit::Int=6,
    sep_line_search_factor::Float64=0.5,
    sep_line_search_min_alpha::Float64=1e-4,
    sep_lm_lambda::Float64=1e-8,
    sep_lm_lambda_scale::Float64=10.0,
    sep_lm_lambda_min::Float64=1e-12,
    sep_lm_lambda_max::Float64=1e4,
    sep_accept_tol::Union{Nothing,Float64}=nothing,
    sep_shock_scale::Float64=1.0,
    shock_scaling::Symbol=:none,
    random_seed::Union{Nothing,Int}=nothing,
    sep_yss::Union{Nothing,Vector{Float64}}=nothing,
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
    enforce_zlb::Bool=false,
    zlb_floor::Float64=1.0,
    zlb_variable::Symbol=:r,
    zlb_obc_maxiter::Int=3,
    silent::Bool=true
)
    if !isnothing(random_seed)
        Random.seed!(random_seed)
    end

    nvars = length(𝓂.var)
    shock_names = 𝓂.exo
    nshocks = length(shock_names)

    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    has_obc_shocks = any(obc_mask)

    total_periods = periods + burn_in

    # Build shock matrix (full, including OBC shocks)
    shocks_used = zeros(nshocks, total_periods)
    if isnothing(shocks)
        if !isempty(structural_idx)
            if shock_scaling != :none && shock_scaling != :parameter
                error("Unknown shock_scaling=$shock_scaling. Use :none or :parameter.")
            end

            sigmas = ones(length(structural_idx))
            if shock_scaling == :parameter
                for (i, idx) in enumerate(structural_idx)
                    sigmas[i] = sep_irf_shock_std(𝓂, shock_names[idx])
                end
            end

            if length(structural_idx) == 1
                shocks_used[structural_idx[1], :] .= randn(total_periods) .* sigmas[1]
            else
                L = Diagonal(sigmas)
                shocks_used[structural_idx, :] .= L * randn(length(structural_idx), total_periods)
            end
        end
    else
        @assert size(shocks, 2) >= total_periods "Not enough shock periods"
        if size(shocks, 1) == nshocks
            shocks_used .= shocks[:, 1:total_periods]
        elseif size(shocks, 1) == length(structural_idx)
            shocks_used[structural_idx, :] .= shocks[:, 1:total_periods]
        else
            error("Shock dimension mismatch: expected $nshocks (or $(length(structural_idx)) non-OBC shocks), got $(size(shocks, 1)).")
        end
    end

    # Initial state (deterministic SS by default)
    SS_result = get_steady_state(𝓂, derivatives=false)
    yss = [Float64(SS_result(var)) for var in 𝓂.var]
    y0 = isnothing(initial_state) ? copy(yss) : copy(initial_state)
    @assert length(y0) == nvars "Initial state dimension mismatch"

    # ── ZLB enforcement setup ──────────────────────────────────────────
    zlb_var_idx = nothing
    zlb_count = 0
    obc_linear_state_update = nothing

    if enforce_zlb
        zlb_var_idx = findfirst(==(zlb_variable), 𝓂.var)
        if isnothing(zlb_var_idx)
            error("enforce_zlb=true but variable :$(zlb_variable) not found in model variables. " *
                  "Available: $(𝓂.var)")
        end

        if has_obc_shocks
            # OBC model: prepare the linear state_update and obc_violation_function
            # needed by _solve_obc_shocks_for_sep.

            # Ensure first-order solution exists
            try
                solve!(𝓂, algorithm=:first_order, silent=true)
            catch e
                @warn "Could not solve first-order for OBC state_update: $e"
            end

            # Build state_update closure from first-order solution
            try
                obc_linear_state_update = _build_first_order_state_update(𝓂)
            catch e
                @warn "Could not build first-order state_update: $e. " *
                      "Falling back to clamp mode for ZLB enforcement."
            end

            # Ensure obc_violation_function is set up (field is typed Function, not nullable;
            # check via obc_violation_equations which is empty for non-OBC models)
            obc_violation_ready = !isempty(𝓂.obc_violation_equations)
            if !obc_violation_ready
                try
                    set_up_obc_violation_function!(𝓂)
                    obc_violation_ready = !isempty(𝓂.obc_violation_equations)
                catch e
                    @warn "Could not set up obc_violation_function: $e"
                end
            end

            if obc_linear_state_update !== nothing && obc_violation_ready
                !silent && println("ZLB enforcement (OBC mode): $(zlb_variable) >= $(zlb_floor) " *
                                   "(variable index $zlb_var_idx, $(sum(obc_mask)) OBC shocks, " *
                                   "max_obc_horizon=$(𝓂.max_obc_horizon), " *
                                   "maxiter=$zlb_obc_maxiter)")
            else
                !silent && println("ZLB enforcement (clamp fallback): OBC setup incomplete")
                obc_linear_state_update = nothing  # force clamp mode
            end
        else
            !silent && println("ZLB enforcement (clamp mode): $(zlb_variable) >= $(zlb_floor) " *
                               "(variable index $zlb_var_idx, SS value = $(yss[zlb_var_idx]))")
        end
    end

    # ── Simulation storage ─────────────────────────────────────────────
    Y_sim = zeros(nvars, total_periods + 1)
    Y_sim[:, 1] = y0

    shock_sequence = zeros(sep_horizon, nshocks)
    errorflag = false
    failure_period = nothing
    # RISK-1: Track per-period SEP residual for training data quality weighting
    sep_errors = fill(NaN, total_periods)

    # Clear any cached SEP solution to prevent warm-start contamination from
    # a previous failed simulation (e.g., during retry with a different seed).
    𝓂.solution.perturbation.stochastic_extended_path = nothing
    push!(𝓂.solution.outdated_algorithms, :stochastic_extended_path)

    # ── Helper: run one SEP solve with given shock_sequence ────────────
    function _run_sep_solve!(shock_seq, state_col)
        solve!(𝓂,
               algorithm = :stochastic_extended_path,
               sep_periods = sep_horizon,
               sep_order = sep_order,
               sep_nnodes = sep_nnodes,
               sep_maxit = sep_maxit,
               sep_tol = sep_tol,
               sep_sparse_tree = sep_sparse_tree,
               sep_linear_solver = sep_linear_solver,
               sep_fallback_solver = sep_fallback_solver,
               sep_stall_iters = sep_stall_iters,
               sep_stall_rel_tol = sep_stall_rel_tol,
               sep_stall_abs_tol = sep_stall_abs_tol,
               sep_line_search = sep_line_search,
               sep_line_search_maxit = sep_line_search_maxit,
               sep_line_search_factor = sep_line_search_factor,
               sep_line_search_min_alpha = sep_line_search_min_alpha,
               sep_lm_lambda = sep_lm_lambda,
               sep_lm_lambda_scale = sep_lm_lambda_scale,
               sep_lm_lambda_min = sep_lm_lambda_min,
               sep_lm_lambda_max = sep_lm_lambda_max,
               sep_shock_scale = sep_shock_scale,
               sep_initial_state = state_col,
               sep_deterministic_shocks = shock_seq,
               sep_expectation_method = sep_expectation_method,
               hmc_samples = hmc_samples,
               hmc_warmup = hmc_warmup,
               hmc_leapfrog_steps = hmc_leapfrog_steps,
               hmc_step_size = hmc_step_size,
               hmc_use_tempering = hmc_use_tempering,
               hmc_temperatures = hmc_temperatures,
               hmc_verbose = hmc_verbose,
               use_subdifferential = use_subdifferential,
               subdiff_kink_tol = subdiff_kink_tol,
               subdiff_alpha_maxit = subdiff_alpha_maxit,
               subdiff_alpha_tol = subdiff_alpha_tol,
               subdiff_verbose = subdiff_verbose,
               silent = silent)
        return 𝓂.solution.perturbation.stochastic_extended_path
    end

    # ── Main simulation loop ───────────────────────────────────────────
    !silent && println("Running SEP extended-path simulation...")
    for t in 1:total_periods
        if !silent && mod(t, 25) == 0
            println("  Period $t / $total_periods")
        end

        fill!(shock_sequence, 0.0)
        shock_sequence[1, :] .= shocks_used[:, t]

        sep_sol = _run_sep_solve!(shock_sequence, Y_sim[:, t])

        # RISK-1: Record per-period SEP solver residual
        sep_errors[t] = sep_sol !== nothing && isfinite(sep_sol.final_error) ? sep_sol.final_error : NaN

        accept_threshold = isnothing(sep_accept_tol) ? sep_tol : sep_accept_tol
        converged = sep_sol !== nothing && isfinite(sep_sol.final_error) &&
                    (sep_sol.convergence_flag == 0 || sep_sol.final_error <= accept_threshold)
        if !converged
            errorflag = true
            failure_period = t
            break
        end

        layout = sep_sol.layout
        Y_sim[:, t + 1] = sep_sol.Y[layout.voff[2] .+ (1:layout.ny_)]

        # ── ZLB enforcement ────────────────────────────────────────────
        if enforce_zlb && !isnothing(zlb_var_idx)
            r_val = Y_sim[zlb_var_idx, t + 1]

            if r_val < zlb_floor
                if has_obc_shocks && obc_linear_state_update !== nothing
                    # ── OBC mode: optimize anticipated shocks & re-solve SEP ──
                    #
                    # Step 1: Use the linear model's OBC solver to find the
                    #         anticipated shocks that enforce the constraint.
                    # Step 2: Inject those shocks into shock_sequence.
                    # Step 3: Re-solve SEP for nonlinear consistency.
                    # Step 4: Iterate up to zlb_obc_maxiter times.

                    # The OBC state_update works in deviations from SS.
                    nsss = 𝓂.solution.non_stochastic_steady_state
                    state_dev = Y_sim[:, t] .- nsss[1:nvars]

                    for obc_iter in 1:zlb_obc_maxiter
                        # Solve for OBC shocks using the linear approximation
                        optimized_shocks, obc_solved = _solve_obc_shocks_for_sep(
                            𝓂, state_dev, shocks_used[:, t], obc_mask,
                            obc_linear_state_update, :first_order;
                            silent=silent
                        )

                        if !obc_solved && !silent
                            @warn "OBC optimizer did not converge at t=$t, iter=$obc_iter"
                        end

                        # Inject optimized OBC shocks into shock_sequence
                        fill!(shock_sequence, 0.0)
                        shock_sequence[1, :] .= optimized_shocks

                        # Record the optimized shocks for output
                        shocks_used[:, t] .= optimized_shocks

                        # Re-solve SEP with OBC shocks
                        sep_sol = _run_sep_solve!(shock_sequence, Y_sim[:, t])

                        if sep_sol === nothing || !isfinite(sep_sol.final_error)
                            !silent && println("  OBC re-solve failed at t=$t iter=$obc_iter")
                            break
                        end

                        sep_errors[t] = sep_sol.final_error
                        layout = sep_sol.layout
                        Y_sim[:, t + 1] = sep_sol.Y[layout.voff[2] .+ (1:layout.ny_)]
                        r_val = Y_sim[zlb_var_idx, t + 1]

                        if r_val >= zlb_floor
                            !silent && println("  ZLB enforced at t=$t after $obc_iter OBC iteration(s): " *
                                               "r=$(round(r_val, digits=6))")
                            break
                        end
                    end

                    if r_val < zlb_floor
                        # OBC optimization could not fully enforce the constraint.
                        # Apply clamp as last resort.
                        !silent && println("  ZLB still violated at t=$t after $(zlb_obc_maxiter) OBC iterations: " *
                                           "r=$(round(r_val, digits=6)) -> clamped to $(zlb_floor)")
                        Y_sim[zlb_var_idx, t + 1] = zlb_floor
                    end
                    zlb_count += 1
                else
                    # ── Clamp mode (non-OBC model or OBC setup failed) ──
                    !silent && println("  ZLB binding at t=$t: $(zlb_variable) = $(round(r_val, digits=6)) " *
                                       "-> clamped to $(zlb_floor)")
                    Y_sim[zlb_var_idx, t + 1] = zlb_floor
                    zlb_count += 1
                end
            end
        end

        # Early termination if state becomes non-finite (NaN/Inf blow-up)
        if !all(isfinite, @view(Y_sim[:, t + 1]))
            !silent && println("  Non-finite state at period $t -- aborting simulation.")
            errorflag = true
            failure_period = t
            break
        end
    end

    # ── ZLB binding detection (passive scan when enforce_zlb is off) ───
    if !enforce_zlb
        detect_var = zlb_variable
        detect_idx = findfirst(==(detect_var), 𝓂.var)
        # Fallback: try :R if :r was not found (SW2003 naming convention)
        if isnothing(detect_idx) && detect_var == :r
            detect_idx = findfirst(==(:R), 𝓂.var)
        end
        if !isnothing(detect_idx)
            last_col = errorflag ? failure_period : size(Y_sim, 2)
            for col in 2:last_col
                rate_val = Y_sim[detect_idx, col]
                if rate_val > 0 && rate_val <= zlb_floor + 1e-6
                    zlb_count += 1
                end
            end
        end
    end

    # Report ZLB summary
    if !silent
        last_col = errorflag ? (isnothing(failure_period) ? size(Y_sim, 2) : failure_period) : size(Y_sim, 2)
        total_sim_periods = max(last_col - 1, 0)
        if total_sim_periods > 0 && zlb_count > 0
            zlb_pct = round(100.0 * zlb_count / total_sim_periods, digits=2)
            println("  ZLB binding: $zlb_count / $total_sim_periods periods ($zlb_pct%)")
        end
    end

    if errorflag
        last_ok = failure_period
        Y_sim = Y_sim[:, 1:last_ok]
        shocks_used = shocks_used[:, 1:max(last_ok - 1, 0)]
    end

    if burn_in > 0
        start_col = min(burn_in + 1, size(Y_sim, 2))
        Y_final = Y_sim[:, start_col:end]
        shocks_final = size(shocks_used, 2) >= burn_in ? shocks_used[:, (burn_in + 1):end] : shocks_used[:, 1:0]
        # RISK-1: Trim sep_errors to match (burn_in periods dropped)
        errors_trimmed = length(sep_errors) >= burn_in ? sep_errors[(burn_in + 1):end] : sep_errors
    else
        Y_final = Y_sim
        shocks_final = shocks_used
        errors_trimmed = sep_errors
    end

    # Truncate sep_errors if simulation was cut short
    if errorflag
        n_final = size(shocks_final, 2)
        errors_trimmed = length(errors_trimmed) > n_final ? errors_trimmed[1:n_final] : errors_trimmed
    end

    time_labels = 0:(size(Y_final, 2) - 1)
    result = KeyedArray(Y_final; Variables=𝓂.var, Time=time_labels)

    return (simulation = result,
            shocks = shocks_final,
            errorflag = errorflag,
            failure_period = failure_period,
            zlb_periods = zlb_count,
            sep_errors = errors_trimmed)
end


"""
    sep_step(m, y_current, e_current, yss, params, T, Lbr, nnodes, silent)

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
    shock_to_group(e, nnodes, nshocks)

Map a shock realization to the nearest group index in the SEP tree.

For nnodes=3: GH nodes are at [-sqrt(3), 0, +sqrt(3)]
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
        error("Only nnodes in {3, 5} supported")
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
