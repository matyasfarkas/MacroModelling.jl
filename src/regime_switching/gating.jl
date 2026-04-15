using Statistics
using LinearAlgebra

function _validate_gate_series(e::AbstractVector, f::AbstractVector)
    length(e) == length(f) || error("Gate statistics length mismatch: $(length(e)) vs $(length(f)).")
    all(isfinite, e) || error("e-stat series contains non-finite values.")
    all(isfinite, f) || error("f-stat series contains non-finite values.")
    return nothing
end

function _gate_mask(e::AbstractVector, f::AbstractVector, tau_e::Real, tau_f::Real;
                    use_eps::Bool = true, use_y::Bool = true)
    _validate_gate_series(e, f)
    (use_eps || use_y) || error("At least one of use_eps/use_y must be true.")
    mask = falses(length(e))
    if use_eps
        mask .|= e .> tau_e
    end
    if use_y
        mask .|= f .> tau_f
    end
    return mask
end

function gate_share(e::AbstractVector, f::AbstractVector, tau_e::Real, tau_f::Real;
                    use_eps::Bool = true, use_y::Bool = true)
    return mean(_gate_mask(e, f, tau_e, tau_f; use_eps = use_eps, use_y = use_y))
end

function _gate_norm(x::AbstractVector, mode::Symbol)
    if mode == :l2
        return norm(x, 2)
    elseif mode == :linf
        return norm(x, Inf)
    end
    error("Unknown norm mode $mode. Use :l2 or :linf.")
end

function compute_gate_stat_series(obs_data::AbstractMatrix,
                                  lin_obs::AbstractMatrix,
                                  shocks::AbstractMatrix,
                                  obs_sigma::AbstractVector,
                                  shock_sigmas::AbstractVector;
                                  structural_idx::Union{Nothing,AbstractVector} = nothing,
                                  shock_norm::Symbol = :l2,
                                  error_norm::Symbol = :l2)
    size(obs_data) == size(lin_obs) || error("obs_data and lin_obs size mismatch: $(size(obs_data)) vs $(size(lin_obs)).")
    size(shocks, 2) == size(obs_data, 2) || error("shocks period mismatch: $(size(shocks, 2)) vs $(size(obs_data, 2)).")
    size(obs_data, 1) == length(obs_sigma) || error("obs_sigma length mismatch: $(length(obs_sigma)) vs $(size(obs_data, 1)).")
    size(shocks, 1) == length(shock_sigmas) || error("shock_sigmas length mismatch: $(length(shock_sigmas)) vs $(size(shocks, 1)).")
    all(isfinite, obs_sigma) || error("obs_sigma contains non-finite values.")
    all(obs_sigma .> 0) || error("obs_sigma must be strictly positive.")
    all(isfinite, shock_sigmas) || error("shock_sigmas contains non-finite values.")

    idx = structural_idx === nothing ? findall(shock_sigmas .> 0) : Int.(collect(structural_idx))
    if !isempty(idx)
        minimum(idx) >= 1 || error("structural_idx must be >= 1.")
        maximum(idx) <= size(shocks, 1) || error("structural_idx exceeds number of shocks ($(size(shocks, 1))).")
        all(shock_sigmas[idx] .> 0) || error("shock_sigmas at structural_idx must be > 0.")
    end

    T = size(obs_data, 2)
    f_stat = zeros(Float64, T)
    e_stat = zeros(Float64, T)
    for t in 1:T
        f_stat[t] = _gate_norm((obs_data[:, t] .- lin_obs[:, t]) ./ obs_sigma, error_norm)
        if isempty(idx)
            e_stat[t] = 0.0
        else
            e_stat[t] = _gate_norm(shocks[idx, t] ./ shock_sigmas[idx], shock_norm)
        end
    end

    return e_stat, f_stat
end

function estimate_observed_shocks_matrix(model,
                                         obs_data::AbstractMatrix,
                                         observables::AbstractVector;
                                         parameters = nothing,
                                         filter::Symbol = :kalman,
                                         algorithm::Symbol = :first_order,
                                         data_in_levels::Bool = true,
                                         smooth::Bool = false,
                                         verbose::Bool = false,
                                         expected_rows::Union{Nothing,Integer} = nothing,
                                         expected_cols::Union{Nothing,Integer} = nothing,
                                         label::AbstractString = "Estimated shocks")
    isempty(observables) && error("Observables missing; cannot estimate shocks.")
    size(obs_data, 1) == length(observables) ||
        error("obs_data row count ($(size(obs_data,1))) must match number of observables ($(length(observables))).")

    data = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))
    kwargs = (
        algorithm = algorithm,
        filter = filter,
        data_in_levels = data_in_levels,
        smooth = smooth,
        verbose = verbose,
    )
    if parameters !== nothing
        kwargs = (; kwargs..., parameters = parameters)
    end
    shocks_ka = get_estimated_shocks(model, data; kwargs...)
    shocks = Array(shocks_ka)

    if expected_rows !== nothing && size(shocks, 1) != Int(expected_rows)
        error("$label row mismatch: got $(size(shocks,1)) rows, expected $(Int(expected_rows))).")
    end
    if expected_cols !== nothing && size(shocks, 2) != Int(expected_cols)
        error("$label length mismatch: got $(size(shocks,2)) periods, expected $(Int(expected_cols))).")
    end

    return shocks
end

function estimate_observed_variables_matrix(model,
                                            obs_data::AbstractMatrix,
                                            observables::AbstractVector;
                                            parameters = nothing,
                                            filter::Symbol = :kalman,
                                            algorithm::Symbol = :first_order,
                                            data_in_levels::Bool = true,
                                            levels::Bool = true,
                                            smooth::Bool = false,
                                            verbose::Bool = false,
                                            expected_rows::Union{Nothing,Integer} = nothing,
                                            expected_cols::Union{Nothing,Integer} = nothing,
                                            label::AbstractString = "Estimated variables")
    isempty(observables) && error("Observables missing; cannot estimate variables.")
    size(obs_data, 1) == length(observables) ||
        error("obs_data row count ($(size(obs_data,1))) must match number of observables ($(length(observables))).")

    data = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))
    kwargs = (
        algorithm = algorithm,
        filter = filter,
        data_in_levels = data_in_levels,
        levels = levels,
        smooth = smooth,
        verbose = verbose,
    )
    if parameters !== nothing
        kwargs = (; kwargs..., parameters = parameters)
    end
    vars_ka = get_estimated_variables(model, data; kwargs...)
    vars = Array(vars_ka)
    var_names = collect(axiskeys(vars_ka, 1))

    if expected_rows !== nothing && size(vars, 1) != Int(expected_rows)
        error("$label row mismatch: got $(size(vars,1)) rows, expected $(Int(expected_rows))).")
    end
    if expected_cols !== nothing && size(vars, 2) != Int(expected_cols)
        error("$label length mismatch: got $(size(vars,2)) periods, expected $(Int(expected_cols))).")
    end
    length(var_names) == size(vars, 1) ||
        error("$label variable-name count mismatch: got $(length(var_names)) names for $(size(vars,1)) rows.")

    return vars, var_names
end

function _named_parameter_indices(model_parameter_names::AbstractVector,
                                  names::AbstractVector;
                                  label::AbstractString)
    idx = indexin(names, model_parameter_names)
    if any(isnothing, idx)
        error("$label not found in model parameters.")
    end
    return Int.(idx)
end

function extract_named_parameters(params::AbstractVector,
                                  model_parameter_names::AbstractVector,
                                  names::AbstractVector;
                                  label::AbstractString = "Parameter names")
    isempty(names) && return eltype(params)[]
    idx = _named_parameter_indices(model_parameter_names, names; label = label)
    return params[idx]
end

function override_named_parameters(base_params::AbstractVector,
                                   model_parameter_names::AbstractVector,
                                   names::AbstractVector,
                                   values::AbstractVector;
                                   label::AbstractString = "Parameter names")
    length(values) == length(names) ||
        error("$label/value length mismatch: $(length(names)) names vs $(length(values)) values.")
    T = eltype(values)
    params = T.(base_params)
    isempty(names) && return params
    idx = _named_parameter_indices(model_parameter_names, names; label = label)
    for (i, j) in enumerate(idx)
        params[j] = values[i]
    end
    return params
end

function parameters_with_theta_mode(base_params::AbstractVector,
                                    model_parameter_names::AbstractVector,
                                    theta_names::AbstractVector,
                                    theta_values::Union{Nothing,AbstractVector};
                                    theta_mode::Symbol,
                                    mode_label::AbstractString = "theta_mode",
                                    theta_label::AbstractString = "Theta names")
    if theta_mode == :baseline
        return copy(base_params)
    elseif theta_mode == :synthetic
        theta_values === nothing &&
            error("$mode_label=:synthetic requires theta values.")
        return override_named_parameters(
            base_params,
            model_parameter_names,
            theta_names,
            theta_values;
            label = theta_label,
        )
    end
    error("Unknown $mode_label=$theta_mode. Use :baseline or :synthetic.")
end

function align_linear_observable_path(lin_irf_output,
                                      observables::AbstractVector,
                                      model_var_names::AbstractVector;
                                      T::Int,
                                      periods::Int,
                                      model_name::AbstractString = "model",
                                      label::AbstractString = "Linear simulation")
    lin_obs = Array(lin_irf_output)
    if ndims(lin_obs) == 3 && size(lin_obs, 3) == 1
        lin_obs = lin_obs[:, :, 1]
    end

    var_idx = indexin(observables, model_var_names)
    if any(isnothing, var_idx)
        error("Observables not found in $(model_name) variables.")
    end
    sorted_idx = sort(Int.(var_idx))
    perm = indexin(Int.(var_idx), sorted_idx)
    if any(isnothing, perm)
        error("Failed to align linear observables with requested order.")
    end
    lin_obs = lin_obs[Int.(perm), :]

    if size(lin_obs, 2) == T + periods
        lin_obs = lin_obs[:, 1:T]
    elseif size(lin_obs, 2) != T
        error("$label length mismatch: got $(size(lin_obs, 2)) periods, expected $T.")
    end

    return lin_obs
end

function compute_linear_gate_stats_from_shocks(model,
                                               obs_data::AbstractMatrix,
                                               observables::AbstractVector,
                                               shocks::AbstractMatrix,
                                               obs_sigma::AbstractVector,
                                               shock_sigmas::AbstractVector;
                                               periods::Int,
                                               parameters = nothing,
                                               initial_state = nothing,
                                               shock_norm::Symbol = :l2,
                                               error_norm::Symbol = :l2,
                                               ignore_obc::Bool = false,
                                               label::AbstractString = "Linear simulation")
    isempty(observables) && error("Observables missing; cannot compute linear gate statistics.")
    periods > 0 || error("periods must be positive, got $periods")

    T = size(obs_data, 2)
    size(shocks, 2) == T || error("Shock matrix length mismatch: got $(size(shocks,2)) periods, expected $T.")
    size(shocks, 1) == length(shock_sigmas) ||
        error("Shock matrix row mismatch: got $(size(shocks,1)) shocks, expected $(length(shock_sigmas)).")

    irf_kwargs = (
        shocks = shocks,
        variables = observables,
        periods = periods,
        algorithm = :first_order,
        levels = true,
        ignore_obc = ignore_obc,
        verbose = false,
    )
    if parameters !== nothing
        irf_kwargs = (; irf_kwargs..., parameters = parameters)
    end
    if initial_state !== nothing
        irf_kwargs = (; irf_kwargs..., initial_state = initial_state)
    end

    lin_irf = get_irf(model; irf_kwargs...)
    lin_obs = align_linear_observable_path(
        lin_irf,
        observables,
        model.timings.var;
        T = T,
        periods = periods,
        model_name = model.model_name,
        label = label,
    )

    structural_idx = findall(shock_sigmas .> 0)
    e_stat, f_stat = compute_gate_stat_series(
        obs_data,
        lin_obs,
        shocks,
        obs_sigma,
        shock_sigmas;
        structural_idx = structural_idx,
        shock_norm = shock_norm,
        error_norm = error_norm,
    )

    return lin_obs, e_stat, f_stat
end

function linear_filter_initial_state(model,
                                     obs_data::AbstractMatrix,
                                     observables::AbstractVector,
                                     state_names::AbstractVector;
                                     parameters = nothing,
                                     filter::Symbol = :kalman,
                                     algorithm::Symbol = :first_order,
                                     label::AbstractString = "Linear filter variables")
    if filter != :kalman && filter != :inversion
        error("Unsupported filter=$filter. Use :kalman or :inversion.")
    end
    isempty(state_names) && error("state_names missing; cannot compute linear initial state.")
    vars, var_names = estimate_observed_variables_matrix(
        model,
        obs_data,
        observables;
        parameters = parameters,
        filter = filter,
        algorithm = algorithm,
        data_in_levels = true,
        levels = true,
        smooth = false,
        verbose = false,
        expected_cols = size(obs_data, 2),
        label = label,
    )
    state_idx = indexin(state_names, var_names)
    if any(isnothing, state_idx)
        error("state_names not found in linear filter output.")
    end
    return vars[Int.(state_idx), end]
end

function linear_filter_full_state_initial(model,
                                          obs_data::AbstractMatrix,
                                          observables::AbstractVector;
                                          parameters = nothing,
                                          filter::Symbol = :kalman,
                                          algorithm::Symbol = :first_order,
                                          label::AbstractString = "Linear filter variables")
    if filter != :kalman && filter != :inversion
        error("Unsupported filter=$filter. Use :kalman or :inversion.")
    end
    vars, _ = estimate_observed_variables_matrix(
        model,
        obs_data,
        observables;
        parameters = parameters,
        filter = filter,
        algorithm = algorithm,
        data_in_levels = true,
        levels = true,
        smooth = false,
        verbose = false,
        expected_cols = size(obs_data, 2),
        label = label,
    )
    return vars[:, 1]
end

function compute_linear_gate_stats_from_filter(model,
                                               obs_data::AbstractMatrix,
                                               observables::AbstractVector,
                                               obs_sigma::AbstractVector,
                                               shock_sigmas::AbstractVector,
                                               state_names::AbstractVector;
                                               periods::Int,
                                               parameters = nothing,
                                               filter::Symbol = :kalman,
                                               algorithm::Symbol = :first_order,
                                               shock_norm::Symbol = :l2,
                                               error_norm::Symbol = :l2,
                                               ignore_obc::Bool = false,
                                               label::AbstractString = "Linear gate stats")
    shocks = estimate_observed_shocks_matrix(
        model,
        obs_data,
        observables;
        parameters = parameters,
        filter = filter,
        algorithm = algorithm,
        data_in_levels = true,
        smooth = false,
        verbose = false,
        expected_rows = length(shock_sigmas),
        expected_cols = size(obs_data, 2),
        label = "$(label) shocks",
    )
    init_state = linear_filter_full_state_initial(
        model,
        obs_data,
        observables;
        parameters = parameters,
        filter = filter,
        algorithm = algorithm,
        label = "$(label) variables",
    )
    lin_obs, e_stat, f_stat = compute_linear_gate_stats_from_shocks(
        model,
        obs_data,
        observables,
        shocks,
        obs_sigma,
        shock_sigmas;
        periods = periods,
        parameters = parameters,
        initial_state = init_state,
        shock_norm = shock_norm,
        error_norm = error_norm,
        ignore_obc = ignore_obc,
        label = label,
    )
    return lin_obs, shocks, e_stat, f_stat
end

function calibrate_gate(e::AbstractVector, f::AbstractVector;
                        config::GateCalibrationConfig = GateCalibrationConfig())
    _validate_gate_series(e, f)
    (config.use_eps || config.use_y) || error("At least one of use_eps/use_y must be true.")
    0 < config.target_share < 1 || error("target_share must be in (0,1).")

    lo = 0.0
    hi = 1.0
    tau_e = config.use_eps ? quantile(e, 0.5) : Inf
    tau_f = config.use_y ? quantile(f, 0.5) : Inf
    share = mean(_gate_mask(e, f, tau_e, tau_f; use_eps = config.use_eps, use_y = config.use_y))

    for _ in 1:config.maxiter
        q = (lo + hi) / 2
        tau_e = config.use_eps ? quantile(e, q) : Inf
        tau_f = config.use_y ? quantile(f, q) : Inf
        share = mean(_gate_mask(e, f, tau_e, tau_f; use_eps = config.use_eps, use_y = config.use_y))
        if abs(share - config.target_share) < config.tol
            return GateCalibrationResult(q, float(tau_e), float(tau_f), float(share))
        end
        if share > config.target_share
            lo = q
        else
            hi = q
        end
    end

    q = (lo + hi) / 2
    tau_e = config.use_eps ? quantile(e, q) : Inf
    tau_f = config.use_y ? quantile(f, q) : Inf
    share = mean(_gate_mask(e, f, tau_e, tau_f; use_eps = config.use_eps, use_y = config.use_y))
    return GateCalibrationResult(q, float(tau_e), float(tau_f), float(share))
end

function calibrate_tau_y(e::AbstractVector, f::AbstractVector, tau_eps::Real, target_share::Float64;
                         tol::Float64 = 1e-4, maxiter::Int = 60)
    _validate_gate_series(e, f)
    0 < target_share < 1 || error("target_share must be in (0,1), got $target_share")
    maxiter > 0 || error("maxiter must be positive.")

    lo = minimum(f)
    hi = maximum(f)
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        share = gate_share(e, f, tau_eps, mid)
        if abs(share - target_share) < tol
            return float(mid), float(share)
        end
        if share > target_share
            lo = mid
        else
            hi = mid
        end
    end

    tau_y = (lo + hi) / 2
    share = gate_share(e, f, tau_eps, tau_y)
    if abs(share - target_share) >= tol
        @warn "calibrate_tau_y did not converge to tolerance $tol after $maxiter iterations. " *
              "Final share=$share, target=$target_share, diff=$(abs(share - target_share))"
    end
    return float(tau_y), float(share)
end

function calibrate_tau_eps(e::AbstractVector, f::AbstractVector, tau_y::Real, target_share::Float64;
                           tol::Float64 = 1e-4, maxiter::Int = 60)
    _validate_gate_series(e, f)
    0 < target_share < 1 || error("target_share must be in (0,1), got $target_share")
    maxiter > 0 || error("maxiter must be positive.")

    lo = minimum(e)
    hi = maximum(e)
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        share = gate_share(e, f, mid, tau_y)
        if abs(share - target_share) < tol
            return float(mid), float(share)
        end
        if share > target_share
            lo = mid
        else
            hi = mid
        end
    end

    tau_eps = (lo + hi) / 2
    share = gate_share(e, f, tau_eps, tau_y)
    if abs(share - target_share) >= tol
        @warn "calibrate_tau_eps did not converge to tolerance $tol after $maxiter iterations. " *
              "Final share=$share, target=$target_share, diff=$(abs(share - target_share))"
    end
    return float(tau_eps), float(share)
end

function apply_gate_padding(mask::AbstractVector{Bool},
                            k_pre::Int,
                            k_post::Int,
                            min_len::Int)
    T = length(mask)
    expanded = falses(T)
    for t in 1:T
        if mask[t]
            lo = max(1, t - k_pre)
            hi = min(T, t + k_post)
            expanded[lo:hi] .= true
        end
    end

    if min_len <= 1
        return expanded
    end

    segments = Tuple{Int,Int}[]
    t = 1
    while t <= T
        if expanded[t]
            start = t
            while t <= T && expanded[t]
                t += 1
            end
            stop = t - 1
            push!(segments, (start, stop))
        else
            t += 1
        end
    end

    adjusted = falses(T)
    for (start0, stop0) in segments
        start = start0
        stop = stop0
        len = stop - start + 1
        if len < min_len
            extra = min_len - len
            add_right = min(extra, T - stop)
            add_left = extra - add_right
            new_start = max(1, start - add_left)
            new_stop = min(T, stop + add_right)
            if new_stop - new_start + 1 < min_len
                remaining = min_len - (new_stop - new_start + 1)
                new_start = max(1, new_start - remaining)
            end
            start = new_start
            stop = new_stop
        end
        adjusted[start:stop] .= true
    end

    return adjusted
end

function assign_regimes(e::AbstractVector, f::AbstractVector, tau_e::Real, tau_f::Real;
                        use_eps::Bool = true,
                        use_y::Bool = true,
                        k_pre::Int = 0,
                        k_post::Int = 0,
                        min_len::Int = 1)
    base = _gate_mask(e, f, tau_e, tau_f; use_eps = use_eps, use_y = use_y)
    return apply_gate_padding(base, k_pre, k_post, min_len)
end

function assign_regimes(e::AbstractVector, f::AbstractVector, cfg::RegimeSwitchConfig)
    return assign_regimes(e, f, cfg.tau_eps, cfg.tau_y;
                          use_eps = cfg.use_eps,
                          use_y = cfg.use_y,
                          k_pre = cfg.k_pre,
                          k_post = cfg.k_post,
                          min_len = cfg.min_len)
end

function logistic(x::Real)
    if x >= 0
        z = exp(-x)
        return 1 / (1 + z)
    else
        z = exp(x)
        return z / (1 + z)
    end
end

function logit(p::Real)
    return log(p / (1 - p))
end

function calibrate_gate_bias(scores::AbstractVector, target_share::Float64)
    0 < target_share < 1 || error("target_share must be in (0,1).")
    lo = -20.0
    hi = 20.0
    for _ in 1:60
        mid = (lo + hi) / 2
        share = mean(logistic.(mid .+ scores))
        if share > target_share
            hi = mid
        else
            lo = mid
        end
    end
    return (lo + hi) / 2
end

function gate_probabilities(e::AbstractVector, f::AbstractVector, cfg::RegimeSwitchConfig)
    _validate_gate_series(e, f)
    if cfg.gate_mode == :hard
        hard = assign_regimes(e, f, cfg)
        probs = Float64.(hard)
        return clamp.(probs, cfg.prob_floor, cfg.prob_ceiling)
    elseif cfg.gate_mode == :soft
        scores = zeros(Float64, length(e))
        if cfg.use_eps
            scores .+= cfg.beta_eps .* (e .- cfg.tau_eps)
        end
        if cfg.use_y
            scores .+= cfg.beta_y .* (f .- cfg.tau_y)
        end
        probs = logistic.(cfg.bias .+ scores)
        probs = clamp.(probs, cfg.prob_floor, cfg.prob_ceiling)
        hard = probs .>= cfg.hard_threshold
        if cfg.k_pre > 0 || cfg.k_post > 0 || cfg.min_len > 1
            hard = apply_gate_padding(hard, cfg.k_pre, cfg.k_post, cfg.min_len)
        end
        return probs
    else
        error("Unknown gate_mode=$(cfg.gate_mode). Use :hard or :soft.")
    end
end
