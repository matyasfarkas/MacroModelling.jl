using Statistics
using Zygote

function _logaddexp(a::Real, b::Real)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function compute_switching_loglikelihood(ll_rom::AbstractVector,
                                         ll_fom::AbstractVector;
                                         hard_mask::Union{Nothing,AbstractVector{Bool}} = nothing,
                                         gate_probs::Union{Nothing,AbstractVector} = nothing,
                                         config::SwitchingLikelihoodConfig = SwitchingLikelihoodConfig())
    length(ll_rom) == length(ll_fom) || error("ll_rom and ll_fom length mismatch.")
    T = length(ll_rom)
    Tret = promote_type(eltype(ll_rom), eltype(ll_fom), Float64)
    rom = Tret.(ll_rom)
    fom = Tret.(ll_fom)

    if hard_mask === nothing && gate_probs === nothing
        error("Provide either hard_mask or gate_probs.")
    end

    if hard_mask !== nothing
        length(hard_mask) == T || error("hard_mask length mismatch.")
        mask = BitVector(hard_mask)
        probs = clamp.(Tret.(mask), Tret(config.prob_floor), Tret(config.prob_ceiling))
        per = [mask[t] ? fom[t] : rom[t] for t in 1:T]
        return SwitchingLikelihoodResult(sum(per), Tret.(per), mask, probs, rom, fom)
    end

    length(gate_probs) == T || error("gate_probs length mismatch.")
    probs = clamp.(Tret.(gate_probs), Tret(config.prob_floor), Tret(config.prob_ceiling))
    mask = BitVector(probs .>= Tret(config.hard_threshold))

    # Use non-mutating array comprehension for Zygote compatibility
    per = if config.soft_mixture == :linear
        [probs[t] * fom[t] + (1 - probs[t]) * rom[t] for t in 1:T]
    elseif config.soft_mixture == :logsumexp
        [_logaddexp(log(probs[t]) + fom[t], log1p(-probs[t]) + rom[t]) for t in 1:T]
    else
        error("Unknown soft_mixture=$(config.soft_mixture). Use :linear or :logsumexp.")
    end

    return SwitchingLikelihoodResult(sum(per), per, mask, probs, rom, fom)
end

function mix_loglikelihood(ll_fom::AbstractVector,
                           ll_rom::AbstractVector,
                           gate_probs::AbstractVector;
                           config::SwitchingLikelihoodConfig = SwitchingLikelihoodConfig(
                               gate_mode = :soft,
                               soft_mixture = :logsumexp,
                           ))
    result = compute_switching_loglikelihood(
        ll_rom,
        ll_fom;
        gate_probs = gate_probs,
        config = config,
    )
    return result.total
end

function evaluate_switching_vs_fom(ll_switching::AbstractVector,
                                   ll_fom::AbstractVector;
                                   runtime_switching::Union{Nothing,Real} = nothing,
                                   runtime_fom::Union{Nothing,Real} = nothing)
    length(ll_switching) == length(ll_fom) || error("Input length mismatch.")
    a = Float64.(ll_switching)
    b = Float64.(ll_fom)
    diffs = a .- b
    abs_diffs = abs.(diffs)
    denom = max(mean(abs.(b)), eps())
    speedup = (runtime_switching === nothing || runtime_fom === nothing || runtime_switching <= 0) ?
        nothing : (float(runtime_fom) / float(runtime_switching))

    return (
        n = length(a),
        switching_total = sum(a),
        fom_total = sum(b),
        total_diff = sum(a) - sum(b),
        mean_abs_diff = mean(abs_diffs),
        max_abs_diff = maximum(abs_diffs),
        rmse = sqrt(mean(abs2, diffs)),
        relative_mean_abs_diff = mean(abs_diffs) / denom,
        runtime_switching_s = runtime_switching === nothing ? nothing : float(runtime_switching),
        runtime_fom_s = runtime_fom === nothing ? nothing : float(runtime_fom),
        speedup = speedup,
    )
end

evaluate_switching_vs_fom(ll_switching::Real, ll_fom::Real; kwargs...) =
    evaluate_switching_vs_fom([float(ll_switching)], [float(ll_fom)]; kwargs...)

function _override_named_parameters_with_index(base_parameters::AbstractVector,
                                               theta_idx::AbstractVector{<:Integer},
                                               theta::AbstractVector;
                                               theta_label::AbstractString)
    length(theta) == length(theta_idx) ||
        error("$theta_label/value length mismatch: $(length(theta_idx)) indices vs $(length(theta)) values.")
    T = eltype(theta)
    params = T.(base_parameters)
    if !isempty(theta_idx)
        minimum(theta_idx) >= 1 ||
            error("$theta_label contains an index < 1.")
        maximum(theta_idx) <= length(params) ||
            error("$theta_label contains an index > number of model parameters ($(length(params))).")
    end
    for (i, idx) in enumerate(theta_idx)
        params[idx] = theta[i]
    end
    return params
end

function linear_model_loglik_per_period(model,
                                        obs_data,
                                        theta::AbstractVector,
                                        theta_names::AbstractVector;
                                        model_parameter_names = model.parameters,
                                        base_parameters = model.parameter_values,
                                        theta_idx::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                                        algorithm::Symbol = :first_order,
                                        filter::Symbol = :kalman,
                                        on_failure_loglikelihood::Real = -1e12,
                                        presample_periods::Int = 0,
                                        initial_covariance = :theoretical,
                                        verbose::Bool = false,
                                        theta_label::AbstractString = "Theta names")
    params = if theta_idx === nothing
        override_named_parameters(
            base_parameters,
            model_parameter_names,
            theta_names,
            theta;
            label = theta_label,
        )
    else
        _override_named_parameters_with_index(
            base_parameters,
            theta_idx,
            theta;
            theta_label = theta_label,
        )
    end
    return get_loglikelihood_per_period(model, obs_data, params;
                                        algorithm = algorithm,
                                        filter = filter,
                                        on_failure_loglikelihood = on_failure_loglikelihood,
                                        presample_periods = presample_periods,
                                        initial_covariance = initial_covariance,
                                        verbose = verbose)
end

function conditional_loglik_per_period(predict_fn::Function,
                                       s0::AbstractVector,
                                       shocks::AbstractMatrix,
                                       theta::AbstractVector,
                                       obs_data::AbstractMatrix,
                                       obs_sigma::AbstractVector)
    Ttheta = eltype(theta)
    state = Ttheta.(s0)
    sigma = Ttheta.(obs_sigma)
    d_obs = size(obs_data, 1)
    size(obs_data, 2) == size(shocks, 2) ||
        error("obs_data/shocks period mismatch: $(size(obs_data,2)) vs $(size(shocks,2)).")
    d_obs == length(obs_sigma) ||
        error("obs_sigma length mismatch: $(length(obs_sigma)) vs obs_data rows ($d_obs).")

    log_norm_sum = sum(log.(2 * Ttheta(pi) .* sigma .^ 2))
    T = size(shocks, 2)
    ll = zeros(Ttheta, T)
    resid = Vector{Ttheta}(undef, d_obs)
    for t in 1:T
        obs_pred, state_next = predict_fn(state, @view(shocks[:, t]), theta)
        length(obs_pred) == d_obs ||
            error("predict_fn observation length mismatch at t=$t: got $(length(obs_pred)), expected $d_obs.")
        @inbounds for i in 1:d_obs
            resid[i] = (obs_data[i, t] - obs_pred[i]) / sigma[i]
        end
        ll[t] = Ttheta(-0.5) * (sum(abs2, resid) + log_norm_sum)
        state = state_next
    end
    return ll
end

function split_observation_state(y_full::AbstractVector,
                                 d_obs::Integer;
                                 label::AbstractString = "predict output")
    d_obs > 0 || error("d_obs must be positive, got $d_obs.")
    length(y_full) >= d_obs ||
        error("$label length mismatch: got $(length(y_full)), expected at least d_obs=$d_obs.")
    return y_full[1:d_obs], y_full[d_obs + 1:end]
end

function predict_from_full(full_predict::Function,
                           state::AbstractVector,
                           shocks::AbstractVector,
                           theta::AbstractVector,
                           d_obs::Integer)
    y_full = full_predict(state, shocks, theta)
    y_full isa AbstractVector ||
        error("full_predict must return an AbstractVector, got $(typeof(y_full)).")
    return split_observation_state(y_full, d_obs; label = "full_predict output")
end

function predict_additive_residual(full_predict::Function,
                                   residual_predict::Function,
                                   state::AbstractVector,
                                   shocks::AbstractVector,
                                   theta::AbstractVector,
                                   d_obs::Integer;
                                   allow_full_residual::Bool = true)
    y_full = full_predict(state, shocks, theta)
    y_full isa AbstractVector ||
        error("full_predict must return an AbstractVector, got $(typeof(y_full)).")
    y_resid = residual_predict(state, shocks, theta)
    y_resid isa AbstractVector ||
        error("residual_predict must return an AbstractVector, got $(typeof(y_resid)).")

    if length(y_resid) == d_obs
        obs_base, state_next = split_observation_state(y_full, d_obs; label = "full_predict output")
        return obs_base .+ y_resid, state_next
    elseif allow_full_residual && length(y_resid) == length(y_full)
        y_aug = y_full .+ y_resid
        return split_observation_state(y_aug, d_obs; label = "residual-augmented output")
    end

    allow_msg = allow_full_residual ? " or $(length(y_full))" : ""
    error("Residual output size mismatch: got $(length(y_resid)), expected $d_obs$(allow_msg).")
end

"""
    predict_additive_residual_ood(full_predict, residual_predict, state, shocks, theta,
                                  d_obs, norm_stats; z_threshold=4.0, allow_full_residual=true)

OOD-safe variant of `predict_additive_residual`. When the NN input [state; shock; theta]
has any z-score > `z_threshold` relative to training `norm_stats`, the surrogate correction
is suppressed and only the ROM1 base prediction is returned. This prevents catastrophic
extrapolation when the filter visits states far from the training distribution.

Returns `(obs, state_next)` — same interface as `predict_additive_residual`.
"""
function predict_additive_residual_ood(full_predict::Function,
                                       residual_predict::Function,
                                       state::AbstractVector,
                                       shocks::AbstractVector,
                                       theta::AbstractVector,
                                       d_obs::Integer,
                                       norm_stats;  # NormStats from FrozenMLP/FrozenResNet
                                       z_threshold::Float64 = 4.0,
                                       allow_full_residual::Bool = true)
    # Check if input is OOD
    x = vcat(Float64.(state), Float64.(shocks), Float64.(theta))
    z_scores = abs.((x .- norm_stats.μX) ./ norm_stats.σX)
    is_ood = maximum(z_scores) > z_threshold

    y_full = full_predict(state, shocks, theta)
    y_full isa AbstractVector ||
        error("full_predict must return an AbstractVector, got $(typeof(y_full)).")

    if is_ood
        # OOD: fall back to ROM1 only — zero surrogate correction
        return split_observation_state(y_full, d_obs; label = "full_predict output (OOD fallback)")
    end

    # In-distribution: apply NN correction
    return predict_additive_residual(full_predict, residual_predict, state, shocks, theta,
                                      d_obs; allow_full_residual = allow_full_residual)
end

function additive_residual_loglik_per_period(full_predict::Function,
                                             residual_predict::Function,
                                             s0::AbstractVector,
                                             shocks::AbstractMatrix,
                                             theta::AbstractVector,
                                             obs_data::AbstractMatrix,
                                             obs_sigma::AbstractVector;
                                             d_obs::Integer = size(obs_data, 1),
                                             allow_full_residual::Bool = true)
    predict_fn = (state, shock_t, θ_local) -> predict_additive_residual(
        full_predict,
        residual_predict,
        state,
        shock_t,
        θ_local,
        d_obs;
        allow_full_residual = allow_full_residual,
    )
    return conditional_loglik_per_period(
        predict_fn,
        s0,
        shocks,
        theta,
        obs_data,
        obs_sigma,
    )
end

"""
    batched_additive_residual_loglik_per_period(full_predict, batch_residual_predict, ...)

Batched variant of `additive_residual_loglik_per_period` that evaluates the NN
residual correction for all T periods in a single BLAS-3 matrix multiply, then
computes log-likelihoods vectorized.

The base model (`full_predict`) is rolled out sequentially (state at t depends on t-1),
but the NN residual correction is batched into one call. This replaces T separate
matrix-vector multiplies with a single matrix-matrix multiply — typically 50-70%
faster for T > 50.

# Arguments
- `full_predict`: `(state, shock, theta) -> Vector` — base model returning `[obs; state]`
- `batch_residual_predict`: `(X::Matrix) -> Y::Matrix` — batched NN, X is `(d_in, T)`,
   Y is `(d_out, T)`. Maps directly to `predict_frozen_batch`.
- `s0`, `shocks`, `theta`, `obs_data`, `obs_sigma`: same as sequential version
- `d_obs`: number of observable dimensions (default: rows of obs_data)
- `allow_full_residual`: if true, allow NN to correct both obs and state
"""
function batched_additive_residual_loglik_per_period(
    full_predict::Function,
    batch_residual_predict::Function,
    s0::AbstractVector,
    shocks::AbstractMatrix,
    theta::AbstractVector,
    obs_data::AbstractMatrix,
    obs_sigma::AbstractVector;
    d_obs::Integer = size(obs_data, 1),
    allow_full_residual::Bool = true)

    Ttheta = eltype(theta)
    sigma = Ttheta.(obs_sigma)
    Tp = size(shocks, 2)
    d_state = length(s0)

    size(obs_data, 2) == Tp ||
        error("obs_data/shocks period mismatch: $(size(obs_data,2)) vs $Tp.")
    d_obs == length(obs_sigma) ||
        error("obs_sigma length mismatch: $(length(obs_sigma)) vs obs_data rows ($d_obs).")

    # Phase A: Sequential ROM1 rollout (cheap matrix-vector products)
    state = Ttheta.(s0)
    input_states = Matrix{Ttheta}(undef, d_state, Tp)
    rom_obs = Matrix{Ttheta}(undef, d_obs, Tp)

    for t in 1:Tp
        y_full = full_predict(state, shocks[:, t], theta)
        obs_t, state_next = split_observation_state(y_full, d_obs;
            label = "batched full_predict output")
        input_states[:, t] = state
        rom_obs[:, t] = obs_t
        state = state_next
    end

    # Phase B: Batched NN evaluation (single BLAS-3 matmul)
    theta_col = reshape(theta, :, 1)
    X_nn = vcat(input_states, Ttheta.(shocks), repeat(theta_col, 1, Tp))
    Y_nn = batch_residual_predict(X_nn)

    # Phase C: Combine ROM1 base + NN correction
    d_resid = size(Y_nn, 1)
    if d_resid == d_obs
        obs_pred = rom_obs .+ Y_nn
    elseif allow_full_residual && d_resid == d_obs + d_state
        obs_pred = rom_obs .+ Y_nn[1:d_obs, :]
    else
        allow_msg = allow_full_residual ? " or $(d_obs + d_state)" : ""
        error("Batch residual output size mismatch: got $d_resid, expected $d_obs$allow_msg.")
    end

    # Phase D: Vectorized LL computation
    log_norm_sum = sum(log.(2 * Ttheta(pi) .* sigma .^ 2))
    resid_scaled = (obs_data .- obs_pred) ./ sigma
    ll = Vector{Ttheta}(undef, Tp)
    @inbounds for t in 1:Tp
        ss = zero(Ttheta)
        for i in 1:d_obs
            ss += resid_scaled[i, t]^2
        end
        ll[t] = Ttheta(-0.5) * (ss + log_norm_sum)
    end
    return ll
end

function rollout_observations(predict_fn::Function,
                              s0::AbstractVector,
                              shocks::AbstractMatrix,
                              theta::AbstractVector;
                              check_finite::Bool = false)
    T = size(shocks, 2)
    Tθ = eltype(theta)
    if T == 0
        return Matrix{Tθ}(undef, 0, 0)
    end

    state = Tθ.(s0)
    obs_pred_1, state_next = predict_fn(state, @view(shocks[:, 1]), theta)
    d_obs = length(obs_pred_1)
    d_obs > 0 || error("predict_fn must return a non-empty observation vector.")
    Tret = promote_type(Tθ, eltype(obs_pred_1))
    obs = Matrix{Tret}(undef, d_obs, T)
    obs[:, 1] = Tret.(obs_pred_1)
    check_finite && !all(isfinite, obs[:, 1]) &&
        error("predict_fn produced non-finite observation values at t=1.")
    check_finite && !all(isfinite, state_next) &&
        error("predict_fn produced non-finite state values at t=1.")
    state = state_next

    for t in 2:T
        obs_pred_t, state_next = predict_fn(state, @view(shocks[:, t]), theta)
        length(obs_pred_t) == d_obs ||
            error("predict_fn observation length mismatch at t=$t: got $(length(obs_pred_t)), expected $d_obs.")
        obs[:, t] = Tret.(obs_pred_t)
        check_finite && !all(isfinite, obs[:, t]) &&
            error("predict_fn produced non-finite observation values at t=$t.")
        check_finite && !all(isfinite, state_next) &&
            error("predict_fn produced non-finite state values at t=$t.")
        state = state_next
    end

    return obs
end

function advance_state(predict_fn::Function,
                       s0::AbstractVector,
                       shocks::AbstractMatrix,
                       theta::AbstractVector,
                       steps::Integer)
    steps >= 0 || error("steps must be nonnegative, got $steps.")
    steps <= size(shocks, 2) ||
        error("steps ($steps) exceeds available shock periods ($(size(shocks,2))).")

    state = eltype(theta).(s0)
    if steps == 0
        return state
    end
    for t in 1:steps
        _, state_next = predict_fn(state, shocks[:, t], theta)
        state = state_next
    end
    return state
end

function linear_reference_loglik_per_period(theta::AbstractVector,
                                            s0::AbstractVector,
                                            shocks::AbstractMatrix,
                                            obs_data::AbstractMatrix,
                                            obs_sigma::AbstractVector,
                                            shock_sigmas::AbstractVector;
                                            shock_filter::Symbol,
                                            linear_filter::Symbol,
                                            predict_linear::Union{Nothing,Function} = nothing,
                                            kalman_linear_loglik::Union{Nothing,Function} = nothing,
                                            inversion_maxit::Int = 10,
                                            inversion_tol::Float64 = 1e-6,
                                            inversion_lambda::Float64 = 1e-4)
    T = size(obs_data, 2)
    size(shocks, 2) == T || error("obs_data/shocks period mismatch: $(T) vs $(size(shocks,2)).")
    if shock_filter == :sampling
        predict_linear === nothing &&
            error("predict_linear is required when shock_filter=:sampling.")
        return conditional_loglik_per_period(
            predict_linear,
            s0,
            shocks,
            theta,
            obs_data,
            obs_sigma,
        )
    end

    if linear_filter == :inversion
        predict_linear === nothing &&
            error("predict_linear is required when linear_filter=:inversion.")
        ll, _ = inversion_loglik_per_period(
            predict_linear,
            s0,
            theta,
            obs_data,
            obs_sigma,
            shock_sigmas;
            maxit = inversion_maxit,
            tol = inversion_tol,
            lambda = inversion_lambda,
        )
        return ll
    elseif linear_filter == :kalman
        kalman_linear_loglik === nothing &&
            error("kalman_linear_loglik is required when linear_filter=:kalman.")
        ll = kalman_linear_loglik(theta)
        ll isa AbstractVector ||
            error("kalman_linear_loglik must return a per-period vector, got $(typeof(ll)).")
        length(ll) == T ||
            error("kalman_linear_loglik length mismatch: $(length(ll)) vs expected $T.")
        return ll
    end

    error("Unsupported linear_filter=$linear_filter. Use :kalman or :inversion.")
end

function inversion_step(predict_fn::Function,
                        state::AbstractVector,
                        y_obs::AbstractVector,
                        theta::AbstractVector,
                        obs_sigma::AbstractVector,
                        shock_sigmas::AbstractVector,
                        structural_idx::AbstractVector{Int};
                        eps_init::Union{Nothing,AbstractVector} = nothing,
                        maxit::Int = 10,
                        tol::Float64 = 1e-6,
                        lambda::Float64 = 1e-4)
    maxit > 0 || error("maxit must be positive, got $maxit.")
    tol > 0 || error("tol must be positive, got $tol.")
    lambda >= 0 || error("lambda must be nonnegative, got $lambda.")
    length(y_obs) == length(obs_sigma) ||
        error("y_obs/obs_sigma length mismatch: $(length(y_obs)) vs $(length(obs_sigma)).")
    all(isfinite, obs_sigma) || error("obs_sigma contains non-finite values.")
    all(obs_sigma .> 0) || error("obs_sigma must be strictly positive.")
    all(isfinite, shock_sigmas) || error("shock_sigmas contains non-finite values.")
    all(shock_sigmas .>= 0) || error("shock_sigmas must be nonnegative.")

    d_eps = length(shock_sigmas)
    isempty(structural_idx) || begin
        minimum(structural_idx) >= 1 || error("structural_idx contains index < 1.")
        maximum(structural_idx) <= d_eps || error("structural_idx contains index > d_eps ($d_eps).")
    end
    n_struct = length(structural_idx)
    shock_std = n_struct > 0 ? shock_sigmas[structural_idx] : zeros(eltype(obs_sigma), 0)
    n_struct == 0 || all(shock_std .> 0) || error("shock_sigmas at structural_idx must be > 0.")
    eps_struct = eps_init === nothing ? zeros(eltype(obs_sigma), n_struct) : copy(eps_init)
    length(eps_struct) == n_struct ||
        error("eps_init length mismatch: $(length(eps_struct)) vs structural shock count $n_struct.")
    obs_log_norm_const = sum(log.(2 * pi .* obs_sigma .^ 2))

    if n_struct == 0
        eps_full = zeros(eltype(obs_sigma), d_eps)
        obs_pred, state_next = predict_fn(state, eps_full, theta)
        resid = (y_obs .- obs_pred) ./ obs_sigma
        ll = -0.5 * (sum(resid .^ 2) + obs_log_norm_const)
        return eps_full, state_next, ll
    end

    shock_log_norm_const = sum(log.(2 * pi .* shock_std .^ 2))
    lambda_eff = lambda  # RISK-4: adaptive regularization based on Jacobian conditioning
    for inv_iter in 1:maxit
        eps_full = zeros(eltype(eps_struct), d_eps)
        eps_full[structural_idx] .= eps_struct
        obs_pred, _ = predict_fn(state, eps_full, theta)
        # Guard: if predict returns NaN/Inf, abort iteration with current eps
        if !all(isfinite, obs_pred)
            break
        end
        resid = (y_obs .- obs_pred) ./ obs_sigma
        r = vcat(resid, eps_struct ./ shock_std)

        J = ℱ.jacobian(eps_s -> begin
            eps_full = zeros(eltype(eps_s), d_eps)
            eps_full[structural_idx] .= eps_s
            predict_fn(state, eps_full, theta)[1]
        end, eps_struct)
        # Guard: if Jacobian contains NaN/Inf (e.g. surrogate OOD), abort
        if !all(isfinite, J)
            break
        end
        J_obs = -(J ./ obs_sigma)
        J_prior = ℒ.Diagonal(1.0 ./ shock_std)
        J_aug = vcat(J_obs, J_prior)

        # RISK-4: Monitor Jacobian conditioning and increase regularization if ill-conditioned.
        # κ(J'J + λI) > 1e14 indicates near-singular Gauss–Newton system.
        # Use cheap diagonal-ratio heuristic for small systems (n_struct ≤ 20),
        # falling back to full cond() only when the heuristic flags concern.
        lhs = J_aug' * J_aug + lambda_eff * ℒ.I
        lhs_dense = Matrix(lhs)
        diag_vals = diag(lhs_dense)
        κ_approx = maximum(diag_vals) / max(minimum(diag_vals), eps())
        if κ_approx > 1e10
            # Diagonal ratio suggests ill-conditioning — compute exact condition number
            κ = ℒ.cond(lhs_dense)
            if κ > 1e14
                lambda_eff = max(lambda_eff * 10, 1e-2)
                lhs = J_aug' * J_aug + lambda_eff * ℒ.I
            end
        elseif κ_approx < 1e4 && lambda_eff > lambda && inv_iter > 1
            # Relax regularization when clearly well-conditioned
            lambda_eff = max(lambda_eff / 2, lambda)
        end

        rhs = -J_aug' * r
        step = try
            lhs \ rhs
        catch
            fill(NaN, length(rhs))
        end
        # Guard against NaN/Inf propagation from singular solves
        if !all(isfinite, step)
            break
        end
        eps_struct .+= step
        if ℒ.norm(step) <= tol * (1 + ℒ.norm(eps_struct))
            break
        end
    end

    eps_full = zeros(eltype(eps_struct), d_eps)
    eps_full[structural_idx] .= eps_struct
    obs_pred, state_next = predict_fn(state, eps_full, theta)
    # Guard: return -Inf LL if final prediction is non-finite
    if !all(isfinite, obs_pred) || !all(isfinite, state_next)
        return eps_full, state, -Inf
    end
    resid = (y_obs .- obs_pred) ./ obs_sigma
    ll = -0.5 * (sum(resid .^ 2) +
                 sum((eps_struct ./ shock_std) .^ 2) +
                 obs_log_norm_const +
                 shock_log_norm_const)
    return eps_full, state_next, ll
end

# FIX AD-03: Extract Float64 from any Real (including ForwardDiff.Dual)
_to_f64(x::Float64) = x
_to_f64(x::AbstractFloat) = Float64(x)
_to_f64(x::Integer) = Float64(x)
function _to_f64(x)
    if hasproperty(x, :value)
        return _to_f64(x.value)
    end
    return convert(Float64, x)
end

"""
    _rom_obs_jacobian(predict_fn, state, theta, d_eps, structural_idx)

Compute the ROM1 observation Jacobian ∂obs/∂ε_struct. For a first-order perturbation
model this is `Z·R` restricted to structural shock columns — constant w.r.t. shock
values (linear model), so it can be computed once and reused across refinement iterations.
"""
function _rom_obs_jacobian(predict_fn::Function,
                           state::AbstractVector{Float64},
                           theta::AbstractVector{Float64},
                           d_eps::Int,
                           structural_idx::AbstractVector{Int})
    n_struct = length(structural_idx)
    n_struct > 0 || error("_rom_obs_jacobian requires at least one structural shock.")
    J = ℱ.jacobian(eps_s -> begin
        eps_full = zeros(eltype(eps_s), d_eps)
        eps_full[structural_idx] .= eps_s
        predict_fn(state, eps_full, theta)[1]
    end, zeros(Float64, n_struct))
    return J
end

function inversion_loglik_per_period(predict_fn::Function,
                                     s0::AbstractVector,
                                     theta::AbstractVector,
                                     obs_data::AbstractMatrix,
                                     obs_sigma::AbstractVector,
                                     shock_sigmas::AbstractVector;
                                     eval_predict_fn::Union{Nothing,Function} = nothing,
                                     batch_eval_residual_fn::Union{Nothing,Function} = nothing,
                                     maxit::Int = 10,
                                     tol::Float64 = 1e-6,
                                     lambda::Float64 = 1e-4,
                                     refine_maxit::Int = 0,
                                     refine_tol::Float64 = 1e-4,
                                     single_eval_residual_fn::Union{Nothing,Function} = nothing,
                                     gate_mask::Union{Nothing,AbstractVector{Bool}} = nothing,
                                     correction_clamp::Union{Nothing,AbstractVector{<:Real}} = nothing)
    maxit > 0 || error("maxit must be positive, got $maxit.")
    tol > 0 || error("tol must be positive, got $tol.")
    lambda >= 0 || error("lambda must be nonnegative, got $lambda.")
    refine_maxit >= 0 || error("refine_maxit must be nonnegative, got $refine_maxit.")
    refine_tol > 0 || error("refine_tol must be positive, got $refine_tol.")
    size(obs_data, 1) == length(obs_sigma) ||
        error("obs_data/obs_sigma mismatch: obs rows=$(size(obs_data,1)) vs sigma=$(length(obs_sigma)).")

    # FIX AD-03: Run the inversion filter in Float64 to avoid nested ForwardDiff.
    # FIX AD-04: predict_fn = ROM1 (linear) for shock recovery,
    #            eval_predict_fn = ROM1 + surrogate (nonlinear) for evaluation & state propagation.
    #
    # INTERLEAVED DESIGN: When eval_predict_fn is provided, state propagation in Phase 1
    # uses eval_predict_fn (nonlinear) so that shock recovery at each period happens at
    # the actual nonlinear state, not the linear state. This prevents state drift.
    # When refine_maxit > 0, shocks are iteratively refined using Gauss-Newton corrections
    # to reduce the observation residual under the nonlinear model.
    theta_f64 = _to_f64.(theta)
    s0_f64 = Float64.(s0)
    d_eps = length(shock_sigmas)
    T = size(obs_data, 2)
    structural_idx = findall(shock_sigmas .> 0)
    n_struct = length(structural_idx)
    eps_init = zeros(Float64, n_struct)
    d_obs = size(obs_data, 1)

    if gate_mask !== nothing
        length(gate_mask) == T || error("gate_mask length mismatch: $(length(gate_mask)) vs $T periods.")
    end

    eval_fn = eval_predict_fn === nothing ? predict_fn : eval_predict_fn
    do_interleave = eval_predict_fn !== nothing
    do_refine = refine_maxit > 0 && n_struct > 0 && do_interleave

    shock_std = n_struct > 0 ? shock_sigmas[structural_idx] : Float64[]
    obs_log_norm_const = sum(log.(2 * pi .* obs_sigma .^ 2))
    shock_log_norm_const = n_struct > 0 ? sum(log.(2 * pi .* shock_std .^ 2)) : 0.0

    # Phase 1: Interleaved inversion in pure Float64 (no AD)
    ll_f64 = zeros(Float64, T)
    shocks_out = zeros(Float64, d_eps, T)
    state_inv = copy(s0_f64)
    eps_init_local = copy(eps_init)
    refine_iters = zeros(Int, T)

    for t in 1:T
        y_obs = obs_data[:, t]

        # (a) Recover shocks at current state using ROM1 (predict_fn)
        eps_full, _, _ = inversion_step(predict_fn,
                                        state_inv,
                                        y_obs,
                                        theta_f64,
                                        obs_sigma,
                                        shock_sigmas,
                                        structural_idx;
                                        eps_init = eps_init_local,
                                        maxit = maxit,
                                        tol = tol,
                                        lambda = lambda)

        # (b) Optionally refine shocks to reduce nonlinear observation residual
        if do_refine
            # ROM1 Jacobian ∂obs/∂ε is constant for first-order perturbation — compute once per period
            J_rom = _rom_obs_jacobian(predict_fn, state_inv, theta_f64, d_eps, structural_idx)
            J_scaled = J_rom ./ obs_sigma
            J_prior = ℒ.Diagonal(1.0 ./ shock_std)
            J_aug = vcat(J_scaled, J_prior)
            JtJ_reg = J_aug' * J_aug + lambda * ℒ.I

            for k in 1:refine_maxit
                obs_nl, _ = eval_fn(state_inv, eps_full, theta_f64)
                r_obs = y_obs .- obs_nl
                if ℒ.norm(r_obs ./ obs_sigma) <= refine_tol
                    refine_iters[t] = k
                    break
                end
                # Gauss-Newton correction with shock prior regularization
                eps_struct = eps_full[structural_idx]
                r_aug = vcat(r_obs ./ obs_sigma, -eps_struct ./ shock_std)
                delta = JtJ_reg \ (J_aug' * r_aug)
                eps_full[structural_idx] .+= delta
                refine_iters[t] = k
            end
        end

        # (c) Evaluate likelihood and propagate state
        # Gate-conditional: full NN correction on gate periods, base model on non-gate
        if gate_mask !== nothing && single_eval_residual_fn !== nothing && gate_mask[t]
            obs_rom, state_rom_next = predict_fn(state_inv, eps_full, theta_f64)
            x_nn_t = vcat(Float64.(state_inv), Float64.(eps_full), theta_f64)
            y_nn_t = single_eval_residual_fn(x_nn_t)
            if correction_clamp !== nothing
                y_nn_t = clamp.(y_nn_t, -correction_clamp, correction_clamp)
            end
            obs_pred = obs_rom .+ y_nn_t[1:d_obs]
            state_resid = y_nn_t[(d_obs + 1):end]
            state_next = isempty(state_resid) ? state_rom_next : state_rom_next .+ state_resid
        elseif do_interleave
            obs_pred, state_next = eval_fn(state_inv, eps_full, theta_f64)
        else
            obs_pred, state_next = predict_fn(state_inv, eps_full, theta_f64)
        end
        # Guard: if prediction is non-finite, assign very negative LL and keep current state
        if !all(isfinite, obs_pred) || !all(isfinite, state_next)
            ll_f64[t] = -1e10
            shocks_out[:, t] .= eps_full
            # Don't update state — keep previous valid state
            continue
        end
        resid = (y_obs .- obs_pred) ./ obs_sigma
        ll_t = -0.5 * (sum(resid .^ 2) + obs_log_norm_const)
        if n_struct > 0
            ll_t += -0.5 * (sum((eps_full[structural_idx] ./ shock_std) .^ 2) + shock_log_norm_const)
        end
        ll_f64[t] = ll_t
        shocks_out[:, t] .= eps_full
        state_inv = Float64.(state_next)
        eps_init_local = eps_full[structural_idx]
    end

    # Phase 2: Recompute the likelihood in a differentiable forward pass using the
    # recovered shocks. This allows ForwardDiff to track gradients through the
    # evaluation function w.r.t. theta, while the shocks are treated as constants.
    Ttheta = eltype(theta)
    if Ttheta === Float64 && eval_predict_fn === nothing && batch_eval_residual_fn === nothing
        # No AD active and no separate eval function — return Float64 results directly.
        # Gate-conditional correction (if active) is already applied in Phase 1.
        # Phase 2 is needed for ForwardDiff gradient tracking or batch residual
        # corrections; do not skip it when batch_eval_residual_fn is provided.
        return ll_f64, shocks_out
    end

    Tll = Ttheta === Float64 ? Float64 : Ttheta

    # Batched Phase 2: When batch_eval_residual_fn is provided, evaluate all NN
    # corrections in a single BLAS-3 call. The ROM1 predictor (predict_fn) rolls out
    # the state trajectory, then the NN residual corrections are applied in batch.
    # Assumes obs-only residual correction (state propagation via predict_fn).
    if batch_eval_residual_fn !== nothing
        d_state = length(s0)

        if gate_mask !== nothing && single_eval_residual_fn !== nothing && any(gate_mask)
            # Hybrid Phase 2: sequential for gate periods, batch for non-gate periods
            obs_pred = Matrix{Tll}(undef, d_obs, T)
            input_states = Matrix{Tll}(undef, d_state, T)
            state_rom = Tll.(s0)

            for t in 1:T
                input_states[:, t] = state_rom
                obs_t, state_next = predict_fn(state_rom, Tll.(shocks_out[:, t]), theta)

                if gate_mask[t]
                    # Gate period: single-sample NN correction to obs AND state
                    x_nn = vcat(state_rom, Tll.(shocks_out[:, t]), theta)
                    y_nn = single_eval_residual_fn(x_nn)
                    if correction_clamp !== nothing
                        y_nn = clamp.(y_nn, -correction_clamp, correction_clamp)
                    end
                    obs_pred[:, t] = obs_t .+ y_nn[1:d_obs]
                    state_resid = y_nn[(d_obs + 1):end]
                    state_rom = isempty(state_resid) ? state_next : state_next .+ state_resid
                else
                    obs_pred[:, t] = obs_t  # placeholder — batch-corrected below
                    state_rom = state_next  # ROM1 only
                end
            end

            # Batch NN obs-correction for non-gate periods (BLAS-3)
            non_gate_idx = findall(.!gate_mask)
            if !isempty(non_gate_idx)
                theta_col = reshape(theta, :, 1)
                X_nn_ng = vcat(input_states[:, non_gate_idx], Tll.(shocks_out[:, non_gate_idx]),
                               repeat(theta_col, 1, length(non_gate_idx)))
                Y_nn_ng = batch_eval_residual_fn(X_nn_ng)
                obs_pred[:, non_gate_idx] .+= Y_nn_ng[1:d_obs, :]
            end
        else
            # Original batch path: ROM1 state rollout + batch NN obs correction
            state_rom = Tll.(s0)
            input_states = Matrix{Tll}(undef, d_state, T)
            rom_obs = Matrix{Tll}(undef, d_obs, T)

            for t in 1:T
                input_states[:, t] = state_rom
                obs_t, state_next = predict_fn(state_rom, Tll.(shocks_out[:, t]), theta)
                rom_obs[:, t] = obs_t
                state_rom = state_next
            end

            theta_col = reshape(theta, :, 1)
            X_nn = vcat(input_states, Tll.(shocks_out), repeat(theta_col, 1, T))
            Y_nn = batch_eval_residual_fn(X_nn)
            obs_pred = rom_obs .+ Y_nn[1:d_obs, :]
        end

        # Vectorized LL with shock penalty
        log_norm_sum = sum(log.(2 * pi .* obs_sigma .^ 2))
        resid_scaled = (obs_data .- obs_pred) ./ obs_sigma
        ll_eval = Vector{Tll}(undef, T)
        @inbounds for t in 1:T
            ss = zero(Tll)
            for i in 1:d_obs
                ss += resid_scaled[i, t]^2
            end
            obs_ll = Tll(-0.5) * (ss + log_norm_sum)
            if n_struct > 0
                shock_penalty = -0.5 * (sum((shocks_out[structural_idx, t] ./ shock_sigmas[structural_idx]) .^ 2) + shock_log_norm_const)
                ll_eval[t] = obs_ll + shock_penalty
            else
                ll_eval[t] = obs_ll
            end
        end
        return ll_eval, shocks_out
    end

    # Sequential Phase 2 (fallback): replay with eval_fn and Dual-number theta
    ll_eval = zeros(Tll, T)
    state_eval = Tll.(s0)
    log_norm = log.(2 * pi .* obs_sigma .^ 2)
    for t in 1:T
        obs_pred, state_next = eval_fn(state_eval, Tll.(shocks_out[:, t]), theta)
        resid = obs_data[:, t] .- obs_pred
        obs_ll = -0.5 * sum((resid ./ obs_sigma) .^ 2 .+ log_norm)
        if n_struct > 0
            shock_penalty = -0.5 * (sum((shocks_out[structural_idx, t] ./ shock_sigmas[structural_idx]) .^ 2) + shock_log_norm_const)
            ll_eval[t] = obs_ll + shock_penalty
        else
            ll_eval[t] = obs_ll
        end
        state_eval = state_next
    end

    return ll_eval, shocks_out
end

function build_shocks_from_eps(eps_mean::AbstractMatrix,
                               shock_sigmas::AbstractVector,
                               shock_guided::Union{Nothing,AbstractMatrix};
                               sample_idx::Union{Nothing,AbstractVector{Int}} = nothing,
                               shocks_base::Union{Nothing,AbstractMatrix} = nothing,
                               T_full::Union{Nothing,Int} = nothing)
    d_eps = length(shock_sigmas)
    structural_idx = findall(shock_sigmas .> 0)
    size(eps_mean, 1) == length(structural_idx) ||
        error("eps_mean row mismatch: got $(size(eps_mean,1)), expected $(length(structural_idx)) structural shocks.")

    sample_idx_vec = sample_idx === nothing ? collect(1:size(eps_mean, 2)) : Int.(collect(sample_idx))
    size(eps_mean, 2) == length(sample_idx_vec) ||
        error("eps_mean column mismatch: got $(size(eps_mean,2)), expected $(length(sample_idx_vec)) sample periods.")
    if !isempty(sample_idx_vec)
        minimum(sample_idx_vec) >= 1 || error("sample_idx must be >= 1.")
    end

    base = shocks_base === nothing ? shock_guided : shocks_base
    inferred_T = if base === nothing
        isempty(sample_idx_vec) ? size(eps_mean, 2) : maximum(sample_idx_vec)
    else
        size(base, 2)
    end
    T = T_full === nothing ? inferred_T : T_full
    T >= 0 || error("T_full must be nonnegative, got $T.")
    if !isempty(sample_idx_vec)
        maximum(sample_idx_vec) <= T || error("sample_idx exceeds target length $T.")
    end

    if base === nothing
        base = zeros(eltype(eps_mean), d_eps, T)
    else
        size(base, 1) == d_eps || error("Base shock matrix row mismatch: got $(size(base,1)), expected $d_eps.")
        size(base, 2) == T || error("Base shock matrix length mismatch: got $(size(base,2)), expected $T.")
    end

    shocks = eltype(eps_mean).(base)
    for (j, idx) in enumerate(structural_idx)
        shocks[idx, sample_idx_vec] .= shocks[idx, sample_idx_vec] .+ eps_mean[j, :] .* shock_sigmas[idx]
    end
    return shocks
end
