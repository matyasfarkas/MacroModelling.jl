struct RomCache
    order::Int
    algorithm::Symbol
    use_obc::Bool
    nvars::Int
    nsss::Vector{Float64}
    state_update::Function
end

mutable struct RomPredictor
    model::Any
    order::Int
    mode::Symbol
    use_obc::Bool
    theta_idx::Vector{Int}
    base_params::Vector{Float64}
    last_theta::Union{Nothing,Vector{Float64}}
    cache::Union{Nothing,RomCache}
    state_idx::Vector{Int}
    obs_idx::Vector{Int}
    last_full_params::Union{Nothing,Vector{Float64}}  # FIX C-02: Track full parameter vector for cache invalidation
end

# FIX C-02: Constructor for backward compatibility
function RomPredictor(model, order::Int, mode::Symbol, use_obc::Bool,
                      theta_idx::Vector{Int}, base_params::Vector{Float64},
                      last_theta, cache, state_idx::Vector{Int}, obs_idx::Vector{Int})
    return RomPredictor(model, order, mode, use_obc, theta_idx, base_params,
                        last_theta, cache, state_idx, obs_idx, nothing)
end

function build_rom_cache(model::Any, order::Int;
                         params::Vector{Float64},
                         use_obc::Bool)
    if !isdefined(Main, :MacroModelling)
        error("MacroModelling not loaded; ROM baseline unavailable.")
    end
    MM = Main.MacroModelling

    # FIX H-04: Validate parameter count
    expected_nparams = length(model.parameters)
    if length(params) != expected_nparams
        error("Parameter vector has length $(length(params)), expected $expected_nparams for model $(model.model_name)")
    end

    # FIX H-04: Validate parameter values are finite
    if !all(isfinite, params)
        bad_idx = findfirst(!isfinite, params)
        error("Parameter[$bad_idx] is not finite (value: $(params[bad_idx]))")
    end

    algorithm = order == 1 ? :first_order :
                order == 2 ? :second_order :
                error("Unsupported ROM order: $order")

    MM.write_parameters_input!(model, params, verbose = false)

    # FIX H-04: Check solve success
    try
        MM.solve!(model; algorithm = algorithm, dynamics = true, obc = use_obc, silent = true)
    catch e
        error("ROM solve failed for order=$order, obc=$use_obc: $e")
    end

    state_update, _ = MM.parse_algorithm_to_state_update(algorithm, model, use_obc)
    _, NSSS, _ = MM.get_relevant_steady_states(model, algorithm)
    nvars = length(model.var)
    nsss = NSSS[1:nvars]

    # FIX H-04: Validate steady state is finite
    if !all(isfinite, nsss)
        bad_idx = findfirst(!isfinite, nsss)
        error("NSSS[$bad_idx] is not finite (value: $(nsss[bad_idx])) - ROM solve may have failed")
    end

    return RomCache(order, algorithm, use_obc, nvars, nsss, state_update)
end

function rom_step_full(cache::RomCache, state_full::AbstractVector, shock::AbstractVector)
    dev = state_full .- cache.nsss
    next_dev = cache.state_update(dev, shock)
    # FIX M-03: Provide informative error message with context
    if next_dev === nothing
        error("ROM state_update failed (likely explosive dynamics or OBC binding). " *
              "Algorithm=$(cache.algorithm), OBC=$(cache.use_obc), " *
              "State deviation norm: $(norm(dev)), Shock norm: $(norm(shock))")
    end
    return next_dev .+ cache.nsss
end

function rom_output_from_full(cache::RomCache,
                              state_full::AbstractVector,
                              shock::AbstractVector,
                              obs_idx::Vector{Int},
                              state_idx::Vector{Int})
    next_level = rom_step_full(cache, state_full, shock)
    return vcat(next_level[obs_idx], next_level[state_idx])
end

# Extract Float64 value from ForwardDiff.Dual or any Real number.
# Uses duck-typing to access .value field on Dual numbers without importing ForwardDiff.
_fval(x::Float64) = x
_fval(x::AbstractFloat) = Float64(x)
_fval(x::Integer) = Float64(x)
function _fval(x)
    # ForwardDiff.Dual has a .value field containing the primal
    if hasproperty(x, :value)
        return _fval(x.value)
    end
    return convert(Float64, x)
end

function ensure_rom_cache!(rp::RomPredictor, θ::AbstractVector)
    if rp.mode == :baseline
        if rp.cache === nothing
            rp.cache = build_rom_cache(rp.model, rp.order;
                                       params = rp.base_params,
                                       use_obc = rp.use_obc)
            rp.last_full_params = copy(rp.base_params)  # FIX C-02: Track full parameters
        end
        return
    end

    # FIX AD-01: Extract Float64 values from Dual numbers for cache check/rebuild.
    # The ROM linear solution is treated as a constant w.r.t. θ for AD purposes.
    # The surrogate NN correction provides the differentiable path for gradients.
    θ_f64 = _fval.(θ)

    # FIX C-02: Build full parameter vector and check if it changed
    params = copy(rp.base_params)
    params[rp.theta_idx] = θ_f64

    # Check if FULL parameter vector changed (not just θ subset)
    if rp.last_full_params !== nothing &&
       length(rp.last_full_params) == length(params) &&
       all(rp.last_full_params .== params)
        return  # Cache still valid
    end

    # Rebuild cache with Float64 parameters (MacroModelling solver requires Float64)
    rp.cache = build_rom_cache(rp.model, rp.order;
                               params = params,
                               use_obc = rp.use_obc)
    rp.last_full_params = copy(params)  # FIX C-02: Store full parameter vector
    rp.last_theta = copy(θ_f64)
end

function rom_predict(rp::RomPredictor,
                     state_subset::AbstractVector,
                     shock::AbstractVector,
                     θ::AbstractVector)
    ensure_rom_cache!(rp, θ)
    cache = rp.cache
    cache === nothing && error("ROM cache not initialized.")

    # FIX AD-02: Extract Float64 values for state_update (MacroModelling's compiled
    # transition function only supports Float64). The ROM linear prediction is treated
    # as non-differentiable; gradients flow through the surrogate NN correction instead.
    state_f64 = _fval.(state_subset)
    shock_f64 = _fval.(shock)

    # Build state_full without mutation (array comprehension)
    state_full = [i in rp.state_idx ? state_f64[findfirst(==(i), rp.state_idx)] : cache.nsss[i]
                  for i in eachindex(cache.nsss)]

    dev = state_full .- cache.nsss
    next_dev = cache.state_update(dev, shock_f64)
    if next_dev === nothing
        error("ROM state_update failed during prediction (likely explosive dynamics or OBC binding). " *
              "Algorithm=$(cache.algorithm), OBC=$(cache.use_obc), " *
              "State deviation norm: $(norm(dev)), Shock norm: $(norm(shock_f64))")
    end
    next_level = next_dev .+ cache.nsss

    # Return as the same eltype as the inputs so AD can track through downstream ops
    T = promote_type(eltype(state_subset), eltype(shock), eltype(θ))
    return T.(vcat(next_level[rp.obs_idx], next_level[rp.state_idx]))
end

"""
    build_matrix_rom_predict(model; state_idx, obs_idx)

Build a ForwardDiff-compatible ROM1 predictor directly from the model's first-order
solution matrix 𝐒. Unlike `rom_predict` (which strips ForwardDiff.Dual numbers via
`_fval` to pass through MacroModelling's compiled transition function), this predictor
uses the raw `𝐒` matrix and naturally propagates dual numbers through the linear
algebra. This is critical for `inversion_step`, which computes the Jacobian ∂obs/∂ε
via ForwardDiff — with `rom_predict`, those Jacobians are identically zero.

Returns `(predict_vec, predict_tuple, nsss)`:
- `predict_vec(state, shock, theta) -> Vector [obs; state]` for `predict_additive_residual`
- `predict_tuple(state, shock, theta) -> (obs, state)` for `conditional_loglik_per_period`
- `nsss`: steady-state vector (for diagnostics)

The model must already be solved at first order before calling this function.
"""
function build_matrix_rom_predict(model; state_idx::Vector{Int}, obs_idx::Vector{Int})
    MM = isdefined(Main, :MacroModelling) ? Main.MacroModelling :
        error("MacroModelling not loaded; cannot build matrix ROM predict.")

    𝐒 = model.solution.perturbation.first_order.solution_matrix
    TT = model.timings
    past_idx = TT.past_not_future_and_mixed_idx
    _, NSSS, _ = MM.get_relevant_steady_states(model, :first_order)
    nvars = length(model.var)
    nsss = Float64.(NSSS[1:nvars])
    d_obs = length(obs_idx)

    # Flat-vector variant: returns [obs; state]
    function predict_vec(state_subset::AbstractVector,
                         shock::AbstractVector,
                         _theta::AbstractVector)
        T_elem = promote_type(eltype(state_subset), eltype(shock))
        state_full = T_elem.(nsss)
        for (si, vi) in enumerate(state_idx)
            state_full[vi] = state_subset[si]
        end
        dev = state_full .- nsss
        input_vec = vcat(dev[past_idx], shock)
        next_dev = 𝐒 * input_vec
        next_full = next_dev .+ nsss
        return vcat(next_full[obs_idx], next_full[state_idx])
    end

    # Tuple variant: returns (obs, state)
    function predict_tuple(state_subset::AbstractVector,
                           shock::AbstractVector,
                           _theta::AbstractVector)
        y = predict_vec(state_subset, shock, _theta)
        return y[1:d_obs], y[d_obs+1:end]
    end

    return predict_vec, predict_tuple, nsss
end
