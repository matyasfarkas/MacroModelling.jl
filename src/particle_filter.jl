"""
    bootstrap_particle_filter(predict_fn, s0, theta, obs_data, obs_sigma, shock_sigmas;
                               n_particles=100, seed=42, resample_threshold=0.5)

Dynare-style bootstrap particle filter (sequential importance resampling).

Draws particles from the state transition (proposal = prior), weights by measurement
likelihood p(y_t | x_t), and resamples via multinomial resampling when ESS drops
below `resample_threshold * n_particles`.

The `predict_fn` signature must be: `(state, shocks, theta) -> (obs_pred, state_next)`
where `obs_pred` is a vector of predicted observables and `state_next` is the propagated
state vector. This matches the `predict_from_full` / `rom_only_predict` interface used
elsewhere in MacroModelling.

Returns a NamedTuple:
- `ll_total::Float64`: total log marginal likelihood
- `ll_per_period::Vector{Float64}`: per-period incremental log-likelihood
- `ess_per_period::Vector{Float64}`: effective sample size per period
- `max_weight_per_period::Vector{Float64}`: maximum normalised weight per period
"""
function bootstrap_particle_filter(predict_fn::Function,
                                    s0::AbstractVector,
                                    theta::AbstractVector,
                                    obs_data::AbstractMatrix,
                                    obs_sigma::AbstractVector,
                                    shock_sigmas::AbstractVector;
                                    n_particles::Int = 100,
                                    seed::Int = 42,
                                    resample_threshold::Float64 = 0.5)
    rng = Random.MersenneTwister(seed)
    d_obs, T = size(obs_data)
    d_state = length(s0)
    N = n_particles

    # Identify structural shock indices (where sigma > 0)
    struct_idx = findall(shock_sigmas .> 0)
    n_shocks_total = length(shock_sigmas)

    # Pre-allocate storage
    ll_per_period  = zeros(Float64, T)
    ess_per_period = zeros(Float64, T)
    max_w_per_period = zeros(Float64, T)

    # Particle states: each column is a particle
    states = repeat(Float64.(s0), 1, N)  # (d_state, N)

    log_weights = zeros(Float64, N)
    obs_sigma_sq = obs_sigma .^ 2

    for t in 1:T
        y_t = @view obs_data[:, t]

        # --- Propagate particles and compute log-weights ---
        for i in 1:N
            # Draw structural shocks
            shocks = zeros(Float64, n_shocks_total)
            for j in struct_idx
                shocks[j] = shock_sigmas[j] * randn(rng)
            end

            # Propagate state
            state_i = @view states[:, i]
            obs_pred, state_next = predict_fn(Vector{Float64}(state_i), shocks, Float64.(theta))
            states[:, i] .= state_next

            # Log-weight = log p(y_t | x_t) under Gaussian measurement error
            lw = 0.0
            for j in 1:d_obs
                lw -= 0.5 * (y_t[j] - obs_pred[j])^2 / obs_sigma_sq[j]
            end
            log_weights[i] = lw
        end

        # --- Log-sum-exp for numerical stability ---
        max_lw = maximum(log_weights)
        shifted = log_weights .- max_lw
        sum_w = sum(exp, shifted)
        ll_per_period[t] = max_lw + log(sum_w) - log(N)

        # --- Normalise weights ---
        w = exp.(shifted)
        w ./= sum_w

        # --- ESS and max weight ---
        ess_per_period[t] = 1.0 / sum(abs2, w)
        max_w_per_period[t] = maximum(w)

        # --- Multinomial resampling if ESS below threshold ---
        if ess_per_period[t] < resample_threshold * N
            cumw = cumsum(w)
            new_states = similar(states)
            for i in 1:N
                u = rand(rng)
                idx = searchsortedfirst(cumw, u)
                idx = clamp(idx, 1, N)
                new_states[:, i] .= states[:, idx]
            end
            states .= new_states
        end
    end

    return (ll_total = sum(ll_per_period),
            ll_per_period = ll_per_period,
            ess_per_period = ess_per_period,
            max_weight_per_period = max_w_per_period)
end
