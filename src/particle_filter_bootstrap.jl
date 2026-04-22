"""
Bootstrap Particle Filter for DSGE Model Likelihood Evaluation
===============================================================

Provides a clean, efficient bootstrap particle filter (sequential importance
resampling) that operates directly on the first-order solution matrices
(A, B, C) of a linearized DSGE model. This yields a Monte Carlo estimate
of the marginal likelihood p(y_{1:T}) that converges to the exact Kalman
filter value as the number of particles grows.

The implementation follows the standard bootstrap particle filter of
Gordon, Salmond, and Smith (1993), with systematic resampling (Kitagawa 1996)
as the default resampling scheme.

State-space representation (deviation from steady state):
    s_{t+1} = A * s_t + B * eps_t,   eps_t ~ N(0, I)
    y_t     = C * s_t + eta_t,       eta_t ~ N(0, diag(sigma_me^2))

where s_t is the vector of (observables_and_states), A is the transition
matrix for the states that appear with a lag, B loads the shocks, and C
selects the observables. The measurement error eta_t is optional; when
absent, a small jitter is added to avoid weight degeneracy.
"""

using Random, LinearAlgebra, Printf

# ============================================================================
# Resampling Algorithms
# ============================================================================

"""
    systematic_resample!(indices, weights, rng)

Systematic resampling (Kitagawa 1996): draw a single uniform u ~ U(0, 1/N),
then select particles at cumulative positions u, u+1/N, u+2/N, ...
Lower variance than multinomial resampling.
"""
function _pf_systematic_resample!(indices::Vector{Int}, weights::Vector{Float64}, rng::AbstractRNG)
    N = length(weights)
    cumw = cumsum(weights)
    cumw[end] = 1.0  # guard against floating-point drift

    u = rand(rng) / N
    j = 1
    @inbounds for i in 1:N
        while cumw[j] < u
            j += 1
        end
        indices[i] = j
        u += 1.0 / N
    end
    return indices
end

"""
    multinomial_resample!(indices, weights, rng)

Standard multinomial resampling: each particle is drawn independently from
the categorical distribution defined by `weights`.
"""
function _pf_multinomial_resample!(indices::Vector{Int}, weights::Vector{Float64}, rng::AbstractRNG)
    N = length(weights)
    cumw = cumsum(weights)
    cumw[end] = 1.0
    @inbounds for i in 1:N
        u = rand(rng)
        indices[i] = clamp(searchsortedfirst(cumw, u), 1, N)
    end
    return indices
end

# ============================================================================
# Internal: Extract Solution Matrices from MacroModelling Model
# ============================================================================

"""
    _pf_extract_state_space(model, params, observables; verbose)

Solve the model at `params` and build the Kalman-compatible state-space
matrices (A, B, C), the steady-state observables vector, and the
initial covariance P0. Returns a NamedTuple or nothing on failure.
"""
function _pf_extract_state_space(model, params::Vector{Float64},
                                 observables::Vector{Symbol};
                                 initial_covariance::Symbol = :theoretical,
                                 verbose::Bool = false)

    TT, SS_and_pars, S_mat, _, solved = MacroModelling.get_relevant_steady_state_and_state_update(
        Val(:first_order), params, model, opts = MacroModelling.merge_calculation_options())

    if !solved
        if verbose println("  [PF] Model solution failed.") end
        return nothing
    end

    obs_sorted = sort(observables)
    obs_idx = convert(Vector{Int}, indexin(obs_sorted,
                sort(union(TT.aux, TT.var, TT.exo_present))))

    # Augmented state = union of lagged states and observables (Kalman convention)
    observables_and_states = sort(union(TT.past_not_future_and_mixed_idx, obs_idx))
    n_aug = length(observables_and_states)

    # Transition: A maps lagged states within the augmented state
    A = S_mat[observables_and_states, 1:TT.nPast_not_future_and_mixed] *
        diagm(ones(n_aug))[indexin(TT.past_not_future_and_mixed_idx, observables_and_states), :]

    # Shock loading: B maps iid N(0,1) shocks to augmented state
    B = S_mat[observables_and_states, TT.nPast_not_future_and_mixed+1:end]

    # Observation selection: C picks observables from augmented state
    C = diagm(ones(n_aug))[indexin(sort(obs_idx), observables_and_states), :]

    # Shock covariance in augmented space
    BB = B * B'

    # Steady-state values of observables
    NSSS_labels = [sort(union(model.exo_present, model.var))...,
                   model.calibration_equations_parameters...]
    ss_obs_idx = convert(Vector{Int}, indexin(obs_sorted, NSSS_labels))
    ss_obs = SS_and_pars[ss_obs_idx]

    # Initial covariance from Lyapunov equation or diagonal fallback
    if initial_covariance == :theoretical
        P0, _ = MacroModelling.solve_lyapunov_equation(A, BB,
                    lyapunov_algorithm = :doubling, tol = 1e-12,
                    acceptance_tol = 1e-8, verbose = false)
    else
        P0 = 10.0 * diagm(ones(n_aug))
    end

    return (A = A, B = B, C = C, BB = BB, P0 = P0,
            ss_obs = ss_obs, n_aug = n_aug,
            n_shocks = size(B, 2), n_obs = length(obs_sorted))
end

# ============================================================================
# Main Entry Point: particle_filter_loglik
# ============================================================================

"""
    particle_filter_loglik(model, obs_data, observables, params;
                           n_particles=1000, seed=42,
                           measurement_error=:auto,
                           resample_scheme=:systematic,
                           resample_threshold=0.5,
                           initial_covariance=:theoretical,
                           verbose=false)

Compute the log marginal likelihood of observed data `obs_data` under the
first-order linear solution of `model` at parameter vector `params`, using
a bootstrap particle filter.

For a linear Gaussian model, the PF log-likelihood converges to the Kalman
filter value as n_particles -> infinity (up to the contribution of the
measurement error jitter).

# Arguments
- `model`: A MacroModelling model object.
- `obs_data`: Matrix of observables (n_obs x T) in levels (not deviations).
- `observables`: Vector{Symbol} of observable names.
- `params`: Full parameter vector (length == number of model parameters).

# Keyword Arguments
- `n_particles::Int=1000`: Number of particles.
- `seed::Int=42`: Random seed for reproducibility.
- `measurement_error`: Controls observation noise.
  - `:auto` (default): 1% of one-step-ahead forecast std dev per observable.
  - `:none`: No measurement error. Weight degeneracy is likely for large T.
  - `Float64`: Scalar std dev applied to all observables.
  - `Vector{Float64}`: Per-observable std dev.
- `resample_scheme::Symbol=:systematic`: `:systematic` or `:multinomial`.
- `resample_threshold::Float64=0.5`: Resample when ESS < threshold * N.
- `initial_covariance::Symbol=:theoretical`: `:theoretical` (Lyapunov) or `:diagonal`.
- `verbose::Bool=false`: Print progress information.

# Returns
NamedTuple with fields:
- `ll_total::Float64`: Total log marginal likelihood estimate.
- `ll_per_period::Vector{Float64}`: Per-period incremental log-likelihood.
- `ess_per_period::Vector{Float64}`: Effective sample size per period.
- `n_resamples::Int`: Number of times resampling was triggered.
"""
function particle_filter_loglik(model, obs_data::AbstractMatrix{Float64},
                                observables::Vector{Symbol},
                                params::Vector{Float64};
                                n_particles::Int = 1000,
                                seed::Int = 42,
                                measurement_error::Union{Symbol,Float64,Vector{Float64}} = :auto,
                                resample_scheme::Symbol = :systematic,
                                resample_threshold::Float64 = 0.5,
                                initial_covariance::Symbol = :theoretical,
                                verbose::Bool = false)

    rng = MersenneTwister(seed)
    N = n_particles
    T_obs = size(obs_data, 2)
    n_obs_vars = size(obs_data, 1)

    @assert n_obs_vars == length(observables) "obs_data rows ($n_obs_vars) != length(observables) ($(length(observables)))"

    # ------------------------------------------------------------------
    # 1. Extract state-space matrices from model
    # ------------------------------------------------------------------
    ss = _pf_extract_state_space(model, params, observables;
                                 initial_covariance = initial_covariance,
                                 verbose = verbose)
    if ss === nothing
        return (ll_total = -Inf,
                ll_per_period = fill(-Inf, T_obs),
                ess_per_period = zeros(T_obs),
                n_resamples = 0)
    end

    A, B, C = ss.A, ss.B, ss.C
    n_aug, n_shocks = ss.n_aug, ss.n_shocks

    if verbose
        println("  [PF] State dim=$n_aug  Shock dim=$n_shocks  Obs dim=$n_obs_vars  Particles=$N")
    end

    # ------------------------------------------------------------------
    # 2. Data in deviations from steady state
    # ------------------------------------------------------------------
    data_dev = obs_data .- ss.ss_obs

    # ------------------------------------------------------------------
    # 3. Determine measurement error std dev per observable
    # ------------------------------------------------------------------
    if measurement_error == :auto
        F_diag = diag(C * ss.BB * C')
        sigma_me = sqrt.(max.(F_diag, 1e-20)) .* 0.01
        if verbose
            println("  [PF] Auto sigma_me: ", round.(sigma_me, digits=6))
        end
    elseif measurement_error == :none
        sigma_me = zeros(n_obs_vars)
    elseif measurement_error isa Float64
        sigma_me = fill(measurement_error, n_obs_vars)
    else
        sigma_me = Float64.(measurement_error)
    end

    sigma_me_sq = sigma_me .^ 2
    has_me = any(sigma_me .> 0)

    # Per-period normalization constant: -0.5 * n_obs * log(2pi) - 0.5 * sum(log(sigma^2))
    if has_me
        log_norm_const = -0.5 * n_obs_vars * log(2 * pi) - 0.5 * sum(log.(sigma_me_sq[sigma_me .> 0]))
    else
        # Without measurement error, use unit variance for numerical purposes
        log_norm_const = -0.5 * n_obs_vars * log(2 * pi)
    end

    # ------------------------------------------------------------------
    # 4. Initialize particles from stationary distribution N(0, P0)
    # ------------------------------------------------------------------
    chol_P0 = try
        cholesky(Hermitian(ss.P0)).L
    catch
        eig = eigen(Hermitian(ss.P0))
        vals = max.(eig.values, 0.0)
        eig.vectors * diagm(sqrt.(vals))
    end

    states = chol_P0 * randn(rng, n_aug, N)

    # ------------------------------------------------------------------
    # 5. Pre-allocate working arrays
    # ------------------------------------------------------------------
    ll_per_period  = zeros(Float64, T_obs)
    ess_per_period = zeros(Float64, T_obs)
    log_weights    = zeros(Float64, N)
    weights        = zeros(Float64, N)
    indices        = zeros(Int, N)
    n_resamples    = 0
    eps_buf        = zeros(Float64, n_shocks)
    new_states     = similar(states)
    y_pred         = zeros(n_obs_vars)
    C_dense        = Matrix(C)

    # ------------------------------------------------------------------
    # 6. Sequential filtering loop
    # ------------------------------------------------------------------
    for t in 1:T_obs
        y_t = @view data_dev[:, t]

        @inbounds for i in 1:N
            # Draw iid standard normal shocks
            randn!(rng, eps_buf)

            # State transition: s_{t+1} = A * s_t + B * eps_t
            s_old = @view states[:, i]
            mul!(@view(new_states[:, i]), A, s_old)
            mul!(@view(new_states[:, i]), B, eps_buf, 1.0, 1.0)

            # Observation: y_pred = C * s_{t+1}
            mul!(y_pred, C_dense, @view(new_states[:, i]))

            # Log importance weight: log p(y_t | s_t^i)
            lw = 0.0
            if has_me
                for j in 1:n_obs_vars
                    lw -= 0.5 * (y_t[j] - y_pred[j])^2 / sigma_me_sq[j]
                end
            else
                for j in 1:n_obs_vars
                    lw -= 0.5 * (y_t[j] - y_pred[j])^2
                end
            end
            lw += log_norm_const
            log_weights[i] = lw
        end

        states .= new_states

        # Marginal likelihood increment: log p(y_t | y_{1:t-1}) = log(1/N * sum(w_i))
        max_lw = maximum(log_weights)
        shifted = log_weights .- max_lw
        sum_w = sum(exp, shifted)
        ll_per_period[t] = max_lw + log(sum_w) - log(N)

        # Normalize weights for ESS and resampling
        @inbounds for i in 1:N
            weights[i] = exp(shifted[i])
        end
        weights ./= sum_w

        # Effective sample size
        ess = 1.0 / sum(abs2, weights)
        ess_per_period[t] = ess

        # Conditional resampling
        if ess < resample_threshold * N
            n_resamples += 1
            if resample_scheme == :systematic
                _pf_systematic_resample!(indices, weights, rng)
            else
                _pf_multinomial_resample!(indices, weights, rng)
            end
            new_states .= states[:, indices]
            states .= new_states
        end

        if verbose && (t % 50 == 0 || t == T_obs)
            @printf("  [PF] t=%d/%d  ll_inc=%.2f  ESS=%.0f/%d  resamples=%d\n",
                    t, T_obs, ll_per_period[t], ess, N, n_resamples)
        end
    end

    return (ll_total = sum(ll_per_period),
            ll_per_period = ll_per_period,
            ess_per_period = ess_per_period,
            n_resamples = n_resamples)
end

# ============================================================================
# Low-level variant operating on pre-extracted matrices
# ============================================================================

"""
    particle_filter_loglik_matrices(A, B, C, data_dev, sigma_me;
                                    n_particles=1000, seed=42,
                                    P0=nothing,
                                    resample_scheme=:systematic,
                                    resample_threshold=0.5,
                                    verbose=false)

Run the bootstrap particle filter directly on pre-extracted state-space
matrices. Useful when the caller has already solved the model and wants
to avoid re-solving.

# Arguments
- `A`: Transition matrix (n_aug x n_aug).
- `B`: Shock loading matrix (n_aug x n_shocks).
- `C`: Observation matrix (n_obs x n_aug).
- `data_dev`: Data in deviations from steady state (n_obs x T).
- `sigma_me`: Measurement error std dev per observable (n_obs,).

# Keyword Arguments
- `P0`: Initial covariance (n_aug x n_aug). If nothing, uses 10*I.
- Other keywords as in `particle_filter_loglik`.

# Returns
Same NamedTuple as `particle_filter_loglik`.
"""
function particle_filter_loglik_matrices(A::AbstractMatrix{Float64},
                                         B::AbstractMatrix{Float64},
                                         C::AbstractMatrix{Float64},
                                         data_dev::AbstractMatrix{Float64},
                                         sigma_me::AbstractVector{Float64};
                                         n_particles::Int = 1000,
                                         seed::Int = 42,
                                         P0::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                                         resample_scheme::Symbol = :systematic,
                                         resample_threshold::Float64 = 0.5,
                                         verbose::Bool = false)

    rng = MersenneTwister(seed)
    N = n_particles
    n_obs_vars, T_obs = size(data_dev)
    n_aug = size(A, 1)
    n_shocks = size(B, 2)

    @assert size(A) == (n_aug, n_aug)
    @assert size(B, 1) == n_aug
    @assert size(C) == (n_obs_vars, n_aug)
    @assert length(sigma_me) == n_obs_vars

    sigma_me_sq = sigma_me .^ 2
    has_me = any(sigma_me .> 0)

    if has_me
        log_norm_const = -0.5 * n_obs_vars * log(2 * pi) - 0.5 * sum(log.(sigma_me_sq[sigma_me .> 0]))
    else
        log_norm_const = -0.5 * n_obs_vars * log(2 * pi)
    end

    # Initialize particles
    if P0 === nothing
        P0_use = 10.0 * diagm(ones(n_aug))
    else
        P0_use = P0
    end

    chol_P0 = try
        cholesky(Hermitian(P0_use)).L
    catch
        eig = eigen(Hermitian(P0_use))
        vals = max.(eig.values, 0.0)
        eig.vectors * diagm(sqrt.(vals))
    end

    states = chol_P0 * randn(rng, n_aug, N)

    ll_per_period  = zeros(Float64, T_obs)
    ess_per_period = zeros(Float64, T_obs)
    log_weights    = zeros(Float64, N)
    weights        = zeros(Float64, N)
    indices        = zeros(Int, N)
    n_resamples    = 0
    eps_buf        = zeros(Float64, n_shocks)
    new_states     = similar(states)
    y_pred         = zeros(n_obs_vars)
    C_dense        = Matrix(C)

    for t in 1:T_obs
        y_t = @view data_dev[:, t]

        @inbounds for i in 1:N
            randn!(rng, eps_buf)

            s_old = @view states[:, i]
            mul!(@view(new_states[:, i]), A, s_old)
            mul!(@view(new_states[:, i]), B, eps_buf, 1.0, 1.0)

            mul!(y_pred, C_dense, @view(new_states[:, i]))

            lw = 0.0
            if has_me
                for j in 1:n_obs_vars
                    lw -= 0.5 * (y_t[j] - y_pred[j])^2 / sigma_me_sq[j]
                end
            else
                for j in 1:n_obs_vars
                    lw -= 0.5 * (y_t[j] - y_pred[j])^2
                end
            end
            lw += log_norm_const
            log_weights[i] = lw
        end

        states .= new_states

        max_lw = maximum(log_weights)
        shifted = log_weights .- max_lw
        sum_w = sum(exp, shifted)
        ll_per_period[t] = max_lw + log(sum_w) - log(N)

        @inbounds for i in 1:N
            weights[i] = exp(shifted[i])
        end
        weights ./= sum_w

        ess = 1.0 / sum(abs2, weights)
        ess_per_period[t] = ess

        if ess < resample_threshold * N
            n_resamples += 1
            if resample_scheme == :systematic
                _pf_systematic_resample!(indices, weights, rng)
            else
                _pf_multinomial_resample!(indices, weights, rng)
            end
            new_states .= states[:, indices]
            states .= new_states
        end

        if verbose && (t % 50 == 0 || t == T_obs)
            @printf("  [PF] t=%d/%d  ll_inc=%.2f  ESS=%.0f/%d  resamples=%d\n",
                    t, T_obs, ll_per_period[t], ess, N, n_resamples)
        end
    end

    return (ll_total = sum(ll_per_period),
            ll_per_period = ll_per_period,
            ess_per_period = ess_per_period,
            n_resamples = n_resamples)
end

# ============================================================================
# Conditionally-Optimal Particle Filter (COPF) — Matrix-level
# ============================================================================

"""
    particle_filter_copf_loglik_matrices(A, B, C, data_dev, sigma_me;
                                          n_particles=1000, seed=42,
                                          P0=nothing,
                                          resample_scheme=:systematic,
                                          resample_threshold=0.5,
                                          verbose=false)

Run the conditionally-optimal particle filter (COPF) directly on
pre-extracted state-space matrices. For a linear Gaussian state-space model
the optimal importance proposal is the Kalman-updated density
p(s_t | s_{t-1}, y_t) rather than the bootstrap prior p(s_t | s_{t-1}).
This eliminates weight degeneracy because the proposal incorporates the
current observation; the importance weights reduce to the predictive
observation density p(y_t | s_{t-1}^i), which is integrated analytically
over the state transition.

Because all matrices are time-invariant, the Kalman gain, posterior
covariance, and its Cholesky factor are pre-computed once and reused for
every particle and every period.

# State-space representation (deviation from steady state)
    s_{t+1} = A * s_t + B * eps_t,   eps_t ~ N(0, I)
    y_t     = C * s_t + eta_t,       eta_t ~ N(0, diag(sigma_me^2))

# Arguments
- `A`: Transition matrix (n_aug x n_aug).
- `B`: Shock loading matrix (n_aug x n_shocks).
- `C`: Observation matrix (n_obs x n_aug).
- `data_dev`: Data in deviations from steady state (n_obs x T).
- `sigma_me`: Measurement error std dev per observable (n_obs,).

# Keyword Arguments
- `P0`: Initial covariance (n_aug x n_aug). If nothing, uses 10*I.
- Other keywords as in `particle_filter_loglik`.

# Returns
Same NamedTuple as `particle_filter_loglik`:
- `ll_total::Float64`: Total log marginal likelihood estimate.
- `ll_per_period::Vector{Float64}`: Per-period incremental log-likelihood.
- `ess_per_period::Vector{Float64}`: Effective sample size per period.
- `n_resamples::Int`: Number of times resampling was triggered.
"""
function particle_filter_copf_loglik_matrices(A::AbstractMatrix{Float64},
                                               B::AbstractMatrix{Float64},
                                               C::AbstractMatrix{Float64},
                                               data_dev::AbstractMatrix{Float64},
                                               sigma_me::AbstractVector{Float64};
                                               n_particles::Int = 1000,
                                               seed::Int = 42,
                                               P0::Union{Nothing,AbstractMatrix{Float64}} = nothing,
                                               resample_scheme::Symbol = :systematic,
                                               resample_threshold::Float64 = 0.5,
                                               verbose::Bool = false)

    rng = MersenneTwister(seed)
    N = n_particles
    n_obs_vars, T_obs = size(data_dev)
    n_aug = size(A, 1)

    @assert size(A) == (n_aug, n_aug)
    @assert size(B, 1) == n_aug
    @assert size(C) == (n_obs_vars, n_aug)
    @assert length(sigma_me) == n_obs_vars

    # ------------------------------------------------------------------
    # 1. Pre-compute COPF quantities (constant for linear model)
    # ------------------------------------------------------------------
    BB = B * B'                                          # state noise covariance
    Sigma_me = diagm(sigma_me .^ 2)                     # measurement error covariance

    F  = C * BB * C' + Sigma_me                          # innovation covariance
    F  = Hermitian((F + F') / 2)
    K  = BB * C' * inv(F)                                # Kalman gain (n_aug x n_obs)

    Sigma_post = BB - K * C * BB                         # posterior state covariance
    Sigma_post = Hermitian((Sigma_post + Sigma_post') / 2)

    chol_post = try
        cholesky(Sigma_post).L
    catch
        # Fallback: clamp eigenvalues to zero, reconstruct
        eig = eigen(Sigma_post)
        vals = max.(eig.values, 0.0)
        eig.vectors * diagm(sqrt.(vals))
    end

    F_inv = inv(F)
    log_det_F = logdet(F)
    log_norm = -0.5 * n_obs_vars * log(2 * pi) - 0.5 * log_det_F

    if verbose
        println("  [COPF] Innovation cov log-det=", round(log_det_F, digits=4),
                "  Posterior state cov trace=", round(tr(Sigma_post), digits=6))
    end

    # ------------------------------------------------------------------
    # 2. Initialize particles from stationary distribution N(0, P0)
    # ------------------------------------------------------------------
    if P0 === nothing
        P0_use = 10.0 * diagm(ones(n_aug))
    else
        P0_use = P0
    end

    chol_P0 = try
        cholesky(Hermitian(P0_use)).L
    catch
        eig = eigen(Hermitian(P0_use))
        vals = max.(eig.values, 0.0)
        eig.vectors * diagm(sqrt.(vals))
    end

    states = chol_P0 * randn(rng, n_aug, N)

    # ------------------------------------------------------------------
    # 3. Pre-allocate working arrays
    # ------------------------------------------------------------------
    ll_per_period  = zeros(Float64, T_obs)
    ess_per_period = zeros(Float64, T_obs)
    log_weights    = zeros(Float64, N)
    weights        = zeros(Float64, N)
    indices        = zeros(Int, N)
    n_resamples    = 0
    new_states     = similar(states)
    s_pred         = zeros(Float64, n_aug)
    y_pred         = zeros(Float64, n_obs_vars)
    innov          = zeros(Float64, n_obs_vars)
    eta_buf        = zeros(Float64, n_aug)
    C_dense        = Matrix(C)
    A_dense        = Matrix(A)
    K_dense        = Matrix(K)
    chol_dense     = Matrix(chol_post)
    F_inv_dense    = Matrix(F_inv)

    # ------------------------------------------------------------------
    # 4. Sequential filtering loop
    # ------------------------------------------------------------------
    for t in 1:T_obs
        y_t = @view data_dev[:, t]

        @inbounds for i in 1:N
            # (a) Predict state: s_pred = A * s_{t-1}^i
            mul!(s_pred, A_dense, @view(states[:, i]))

            # (b) Predicted observation: y_pred = C * s_pred
            mul!(y_pred, C_dense, s_pred)

            # (c) Innovation
            for j in 1:n_obs_vars
                innov[j] = y_t[j] - y_pred[j]
            end

            # (d) Draw from optimal proposal:
            #     s_t^i = s_pred + K * innovation + chol_post * randn
            randn!(rng, eta_buf)
            # Start with s_pred
            copyto!(@view(new_states[:, i]), s_pred)
            # Add K * innovation
            mul!(@view(new_states[:, i]), K_dense, innov, 1.0, 1.0)
            # Add chol_post * eta
            mul!(@view(new_states[:, i]), chol_dense, eta_buf, 1.0, 1.0)

            # (e) Log weight = predictive marginal p(y_t | s_{t-1}^i)
            #     = N(y_t; C * A * s_{t-1}^i, F)
            #     log w = log_norm - 0.5 * innov' * F_inv * innov
            lw = log_norm
            # Compute innov' * F_inv * innov via manual loops
            tmp = 0.0
            for k in 1:n_obs_vars
                Finv_innov_k = 0.0
                for j in 1:n_obs_vars
                    Finv_innov_k += F_inv_dense[k, j] * innov[j]
                end
                tmp += innov[k] * Finv_innov_k
            end
            lw -= 0.5 * tmp
            log_weights[i] = lw
        end

        states .= new_states

        # Marginal likelihood increment: log p(y_t | y_{1:t-1})
        max_lw = maximum(log_weights)
        shifted = log_weights .- max_lw
        sum_w = sum(exp, shifted)
        ll_per_period[t] = max_lw + log(sum_w) - log(N)

        # Normalize weights for ESS and resampling
        @inbounds for i in 1:N
            weights[i] = exp(shifted[i])
        end
        weights ./= sum_w

        # Effective sample size
        ess = 1.0 / sum(abs2, weights)
        ess_per_period[t] = ess

        # Conditional resampling
        if ess < resample_threshold * N
            n_resamples += 1
            if resample_scheme == :systematic
                _pf_systematic_resample!(indices, weights, rng)
            else
                _pf_multinomial_resample!(indices, weights, rng)
            end
            new_states .= states[:, indices]
            states .= new_states
        end

        if verbose && (t % 50 == 0 || t == T_obs)
            @printf("  [COPF] t=%d/%d  ll_inc=%.2f  ESS=%.0f/%d  resamples=%d\n",
                    t, T_obs, ll_per_period[t], ess, N, n_resamples)
        end
    end

    return (ll_total = sum(ll_per_period),
            ll_per_period = ll_per_period,
            ess_per_period = ess_per_period,
            n_resamples = n_resamples)
end

# ============================================================================
# High-level COPF Entry Point: particle_filter_copf_loglik
# ============================================================================

"""
    particle_filter_copf_loglik(model, obs_data, observables, params;
                                 n_particles=1000, seed=42,
                                 measurement_error=:auto,
                                 resample_scheme=:systematic,
                                 resample_threshold=0.5,
                                 initial_covariance=:theoretical,
                                 verbose=false)

Compute the log marginal likelihood of observed data `obs_data` under the
first-order linear solution of `model` at parameter vector `params`, using
the conditionally-optimal particle filter (COPF).

The COPF uses the Kalman-updated density p(s_t | s_{t-1}, y_t) as the
importance proposal instead of the bootstrap prior p(s_t | s_{t-1}). For
a linear Gaussian model this is the variance-minimizing proposal: the
importance weights reduce to the predictive observation density
p(y_t | s_{t-1}^i) and ESS remains close to N throughout the sample. As
n_particles -> infinity the log-likelihood converges to the exact Kalman
filter value.

# Arguments
- `model`: A MacroModelling model object.
- `obs_data`: Matrix of observables (n_obs x T) in levels (not deviations).
- `observables`: Vector{Symbol} of observable names.
- `params`: Full parameter vector (length == number of model parameters).

# Keyword Arguments
- `n_particles::Int=1000`: Number of particles.
- `seed::Int=42`: Random seed for reproducibility.
- `measurement_error`: Controls observation noise.
  - `:auto` (default): 1% of one-step-ahead forecast std dev per observable.
  - `:none`: No measurement error.
  - `Float64`: Scalar std dev applied to all observables.
  - `Vector{Float64}`: Per-observable std dev.
- `resample_scheme::Symbol=:systematic`: `:systematic` or `:multinomial`.
- `resample_threshold::Float64=0.5`: Resample when ESS < threshold * N.
- `initial_covariance::Symbol=:theoretical`: `:theoretical` (Lyapunov) or `:diagonal`.
- `verbose::Bool=false`: Print progress information.

# Returns
NamedTuple with fields:
- `ll_total::Float64`: Total log marginal likelihood estimate.
- `ll_per_period::Vector{Float64}`: Per-period incremental log-likelihood.
- `ess_per_period::Vector{Float64}`: Effective sample size per period.
- `n_resamples::Int`: Number of times resampling was triggered.
"""
function particle_filter_copf_loglik(model, obs_data::AbstractMatrix{Float64},
                                      observables::Vector{Symbol},
                                      params::Vector{Float64};
                                      n_particles::Int = 1000,
                                      seed::Int = 42,
                                      measurement_error::Union{Symbol,Float64,Vector{Float64}} = :auto,
                                      resample_scheme::Symbol = :systematic,
                                      resample_threshold::Float64 = 0.5,
                                      initial_covariance::Symbol = :theoretical,
                                      verbose::Bool = false)

    N = n_particles
    T_obs = size(obs_data, 2)
    n_obs_vars = size(obs_data, 1)

    @assert n_obs_vars == length(observables) "obs_data rows ($n_obs_vars) != length(observables) ($(length(observables)))"

    # ------------------------------------------------------------------
    # 1. Extract state-space matrices from model
    # ------------------------------------------------------------------
    ss = _pf_extract_state_space(model, params, observables;
                                 initial_covariance = initial_covariance,
                                 verbose = verbose)
    if ss === nothing
        return (ll_total = -Inf,
                ll_per_period = fill(-Inf, T_obs),
                ess_per_period = zeros(T_obs),
                n_resamples = 0)
    end

    A, B, C = ss.A, ss.B, ss.C
    n_aug, n_shocks = ss.n_aug, ss.n_shocks

    if verbose
        println("  [COPF] State dim=$n_aug  Shock dim=$n_shocks  Obs dim=$n_obs_vars  Particles=$N")
    end

    # ------------------------------------------------------------------
    # 2. Data in deviations from steady state
    # ------------------------------------------------------------------
    data_dev = obs_data .- ss.ss_obs

    # ------------------------------------------------------------------
    # 3. Determine measurement error std dev per observable
    # ------------------------------------------------------------------
    if measurement_error == :auto
        F_diag = diag(C * ss.BB * C')
        sigma_me = sqrt.(max.(F_diag, 1e-20)) .* 0.01
        if verbose
            println("  [COPF] Auto sigma_me: ", round.(sigma_me, digits=6))
        end
    elseif measurement_error == :none
        sigma_me = zeros(n_obs_vars)
    elseif measurement_error isa Float64
        sigma_me = fill(measurement_error, n_obs_vars)
    else
        sigma_me = Float64.(measurement_error)
    end

    # ------------------------------------------------------------------
    # 4. Delegate to matrix-level COPF
    # ------------------------------------------------------------------
    return particle_filter_copf_loglik_matrices(A, B, C, data_dev, sigma_me;
                                                 n_particles = N,
                                                 seed = seed,
                                                 P0 = ss.P0,
                                                 resample_scheme = resample_scheme,
                                                 resample_threshold = resample_threshold,
                                                 verbose = verbose)
end
