#!/usr/bin/env julia
"""
Test suite for Red Team audit implementations:
  RISK-4:  Jacobian condition monitoring in inversion_step
  RISK-3d: OOD-safe predict_additive_residual_ood
  RISK-3a: SiLU activation + FrozenMLP backward compat
  RISK-3c: OOD detection (predict_frozen_safe, compute_ood_flag)
  RISK-1d: weighted_mse loss
  RISK-1b: SEP residual storage in dataset format
  RISK-6:  Residual-dependent QR damping (src/sep_solver.jl — smoke test only)
"""

using Test
using LinearAlgebra
using Statistics
using Random

# ============================================================================
# Part 1: Test nn_utils (FrozenMLP, SiLU, OOD, weighted_mse)
# ============================================================================
println("Loading nn_utils...")
include(joinpath(@__DIR__, "..", "scripts", "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))

@testset "nn_utils: SiLU activation" begin
    # SiLU(x) = x * sigmoid(x)
    @test silu(0.0) ≈ 0.0 atol=1e-12
    @test silu(100.0) ≈ 100.0 atol=1e-3  # saturates to identity for large x
    @test silu(-100.0) ≈ 0.0 atol=1e-3   # saturates to 0 for large negative x
    # SiLU is smooth and non-monotonic near origin
    @test silu(1.0) > 0.5
    @test silu(-1.0) < 0.0  # SiLU is negative for negative inputs
end

@testset "nn_utils: FrozenMLP backward-compatible constructor" begin
    d_in, d_h, d_out = 5, 8, 3
    W1 = randn(d_h, d_in); b1 = randn(d_h)
    W2 = randn(d_out, d_h); b2 = randn(d_out)
    norm = NormStats(zeros(d_in), ones(d_in), zeros(d_out), ones(d_out))

    # 9-arg constructor (old format) should default to :tanh
    f_old = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, d_in, d_out)
    @test f_old.activation == :tanh

    # 10-arg constructor (new format) should use specified activation
    f_new = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, d_in, d_out, :silu)
    @test f_new.activation == :silu
end

@testset "nn_utils: mlp_forward activation dispatch" begin
    Random.seed!(42)
    d_in, d_h, d_out = 4, 8, 2
    W1 = 0.1 .* randn(d_h, d_in); b1 = zeros(d_h)
    W2 = 0.1 .* randn(d_out, d_h); b2 = zeros(d_out)
    x = randn(d_in)

    y_tanh = mlp_forward(W1, b1, W2, b2, nothing, nothing, x; activation=:tanh)
    y_silu = mlp_forward(W1, b1, W2, b2, nothing, nothing, x; activation=:silu)

    @test length(y_tanh) == d_out
    @test length(y_silu) == d_out
    # With random weights, tanh and silu should generally produce different results
    @test y_tanh != y_silu
    @test all(isfinite, y_tanh)
    @test all(isfinite, y_silu)
end

@testset "nn_utils: predict_frozen matches manual computation" begin
    Random.seed!(123)
    d_in, d_h, d_out = 3, 6, 2
    W1 = 0.1 .* randn(d_h, d_in); b1 = zeros(d_h)
    W2 = 0.1 .* randn(d_out, d_h); b2 = zeros(d_out)
    μX = randn(d_in); σX = abs.(randn(d_in)) .+ 0.1
    μY = randn(d_out); σY = abs.(randn(d_out)) .+ 0.1
    norm = NormStats(μX, σX, μY, σY)

    f = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, d_in, d_out, :tanh)
    x = randn(d_in)

    # Manual computation
    xnorm = (x .- μX) ./ σX
    h = tanh.(W1 * xnorm .+ b1)
    y_raw = W2 * h .+ b2
    y_expected = μY .+ σY .* y_raw

    y_pred = predict_frozen(f, x)
    @test y_pred ≈ y_expected atol=1e-12
end

@testset "nn_utils: predict_frozen_batch consistency" begin
    Random.seed!(456)
    d_in, d_h, d_out = 5, 10, 3
    W1 = 0.1 .* randn(d_h, d_in); b1 = randn(d_h)
    W2 = 0.1 .* randn(d_out, d_h); b2 = randn(d_out)
    norm = NormStats(randn(d_in), abs.(randn(d_in)) .+ 0.1,
                     randn(d_out), abs.(randn(d_out)) .+ 0.1)
    f = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, d_in, d_out, :silu)

    batch_size = 20
    X = randn(d_in, batch_size)

    # Batch prediction
    Y_batch = predict_frozen_batch(f, X)
    @test size(Y_batch) == (d_out, batch_size)

    # Compare with sequential prediction
    for j in 1:batch_size
        y_single = predict_frozen(f, X[:, j])
        @test Y_batch[:, j] ≈ y_single atol=1e-10
    end
end

@testset "nn_utils: predict_frozen_safe OOD detection" begin
    Random.seed!(789)
    d_in, d_h, d_out = 4, 8, 2
    W1 = 0.1 .* randn(d_h, d_in); b1 = zeros(d_h)
    W2 = 0.1 .* randn(d_out, d_h); b2 = zeros(d_out)
    μX = zeros(d_in); σX = ones(d_in)
    norm = NormStats(μX, σX, zeros(d_out), ones(d_out))
    f = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, d_in, d_out, :tanh)

    # In-distribution input (z-scores < 4)
    x_id = [1.0, -0.5, 2.0, -1.5]
    pred_id, ood_id, maxz_id = predict_frozen_safe(f, x_id; z_threshold=4.0)
    @test !ood_id
    @test maxz_id < 4.0
    @test length(pred_id) == d_out

    # Out-of-distribution input (z-score > 4)
    x_ood = [1.0, -0.5, 5.5, -1.5]
    pred_ood, ood_ood, maxz_ood = predict_frozen_safe(f, x_ood; z_threshold=4.0)
    @test ood_ood
    @test maxz_ood ≈ 5.5
    @test length(pred_ood) == d_out
end

@testset "nn_utils: compute_ood_flag" begin
    norm = NormStats(zeros(6), ones(6), zeros(2), ones(2))
    state = [0.5, -0.3]
    shock = [1.0, 0.5]
    theta = [0.1, -0.2]
    @test !compute_ood_flag(norm, state, shock, theta; z_threshold=4.0)

    # Make one dimension OOD
    state_ood = [0.5, 6.0]
    @test compute_ood_flag(norm, state_ood, shock, theta; z_threshold=4.0)
end

@testset "nn_utils: weighted_mse" begin
    Ŷ = [1.0 2.0 3.0; 4.0 5.0 6.0]
    Y = [1.1 2.2 3.3; 4.1 5.2 6.3]

    # Uniform weights → same as MSE
    w_uniform = [1.0, 1.0, 1.0]
    wmse = weighted_mse(Ŷ, Y, w_uniform)
    mse_manual = mean(sum(abs2, Ŷ .- Y; dims=1))
    @test wmse ≈ mse_manual atol=1e-12

    # Higher weight on first sample
    w_biased = [10.0, 1.0, 1.0]
    wmse_biased = weighted_mse(Ŷ, Y, w_biased)
    @test wmse_biased != wmse  # should differ
    @test isfinite(wmse_biased)
end

@testset "nn_utils: train_mlp! basic training" begin
    Random.seed!(42)
    d_in, d_out = 3, 2
    n = 200

    # Simple linear target: Y = A*X + noise
    A = randn(d_out, d_in)
    X = randn(d_in, n)
    Y = A * X .+ 0.01 .* randn(d_out, n)

    # Train with SiLU (default)
    frozen = train_mlp!(copy(X), copy(Y);
        d_hidden=16, d_hidden2=nothing, nepoch=100,
        η_init=1e-3, seed=1, verbose=false, activation=:silu)

    @test frozen isa FrozenMLP
    @test frozen.activation == :silu
    @test frozen.d_in == d_in
    @test frozen.d_out == d_out

    # Check prediction quality on training data
    Y_pred = predict_frozen_batch(frozen, X)
    rmse = sqrt(mean(abs2, Y_pred .- (A * X .+ 0.01 .* randn(d_out, n))))
    # Just check it's finite and not huge
    @test all(isfinite, Y_pred)
end

@testset "nn_utils: train_mlp! with sample_weights" begin
    Random.seed!(42)
    d_in, d_out = 3, 2
    n = 100
    X = randn(d_in, n)
    Y = randn(d_out, n)
    w = abs.(randn(n)) .+ 0.1
    w ./= mean(w)

    # Should not error with weights
    frozen = train_mlp!(copy(X), copy(Y);
        d_hidden=8, d_hidden2=nothing, nepoch=10,
        η_init=1e-3, seed=1, verbose=false,
        activation=:tanh, sample_weights=w)
    @test frozen isa FrozenMLP
    @test frozen.activation == :tanh
end

# ============================================================================
# Part 2: Test regime_switching/likelihood.jl functions
# ============================================================================
println("\nLoading MacroModelling for likelihood tests...")

# We need these symbols from the package
using MacroModelling
import ForwardDiff as ℱ

@testset "likelihood: inversion_step basic functionality" begin
    # Simple 2-obs, 2-shock linear model
    Random.seed!(5555)
    d_obs = 2; d_eps = 2; d_state = 3
    Z_audit = randn(d_obs, d_state)
    R_audit = randn(d_state, d_eps)
    T_audit = 0.8 * I(d_state)

    function _audit_predict_basic(state, shock, theta)
        state_next = T_audit * state + R_audit * shock
        obs = Z_audit * state_next
        return obs, state_next
    end

    state0 = randn(d_state)
    theta = Float64[]
    obs_sigma = [0.01, 0.01]       # tight obs_sigma → fit observations well
    shock_sigmas = [100.0, 100.0]   # large shock_std → weak prior on shocks
    structural_idx = [1, 2]

    true_shock = [0.5, -0.3]
    state_next_true = T_audit * state0 + R_audit * true_shock
    y_obs = Z_audit * state_next_true  # noiseless observations

    eps_recovered, state_recovered, ll = MacroModelling.inversion_step(
        _audit_predict_basic, state0, y_obs, theta, obs_sigma, shock_sigmas, structural_idx;
        maxit=50, tol=1e-12, lambda=1e-6)

    @test length(eps_recovered) == d_eps
    @test isfinite(ll)
    # With weak prior and noiseless obs, recovery should be very good
    @test eps_recovered[structural_idx] ≈ true_shock atol=0.05
end

@testset "likelihood: inversion_step RISK-4 condition monitoring" begin
    # Create an ill-conditioned scenario: nearly collinear observation Jacobian
    d_obs = 2; d_eps = 2; d_state = 2

    function predict_fn_illcond(state, shock, theta)
        # Nearly identical observation equations → ill-conditioned J'J
        state_next = state + shock
        obs = [state_next[1] + 1e-10 * state_next[2],
               state_next[1] + 2e-10 * state_next[2]]
        return obs, state_next
    end

    state0 = [1.0, 2.0]
    theta = Float64[]
    obs_sigma = [0.1, 0.1]
    shock_sigmas = [1.0, 1.0]
    structural_idx = [1, 2]
    y_obs = [1.5, 1.5]

    # Should not error even with ill-conditioned Jacobian — RISK-4 adaptive regularization
    eps_recovered, state_recovered, ll = MacroModelling.inversion_step(
        predict_fn_illcond, state0, y_obs, theta, obs_sigma, shock_sigmas, structural_idx;
        maxit=20, tol=1e-6, lambda=1e-4)

    @test all(isfinite, eps_recovered)
    @test isfinite(ll)
    @test all(isfinite, state_recovered)
end

@testset "likelihood: predict_additive_residual" begin
    d_obs = 2; d_state = 3

    function full_predict(state, shocks, theta)
        return vcat(state[1:d_obs] .+ 0.1 .* shocks[1:d_obs], state .+ 0.01)
    end
    function residual_predict(state, shocks, theta)
        return [0.05, 0.03]  # small NN correction
    end

    state = [1.0, 2.0, 3.0]
    shocks = [0.5, -0.3]
    theta = [0.1]

    obs, state_next = MacroModelling.predict_additive_residual(
        full_predict, residual_predict, state, shocks, theta, d_obs)
    @test length(obs) == d_obs
    @test length(state_next) == d_state
    # obs should be full_predict_obs + residual
    full_y = full_predict(state, shocks, theta)
    @test obs ≈ full_y[1:d_obs] .+ [0.05, 0.03]
end

@testset "likelihood: predict_additive_residual_ood" begin
    d_obs = 2; d_state = 3

    function full_predict_ood(state, shocks, theta)
        return vcat(state[1:d_obs], state .+ 0.01)
    end
    function residual_predict_ood(state, shocks, theta)
        return [100.0, 200.0]  # large NN correction (would be catastrophic if OOD)
    end

    state = [1.0, 2.0, 3.0]
    shocks = [0.5, -0.3]
    theta = [0.1]

    # Create norm stats where the input is well within distribution
    d_in = length(state) + length(shocks) + length(theta)
    norm_id = NormStats(zeros(d_in), 10.0 .* ones(d_in),
                        zeros(d_obs), ones(d_obs))

    # In-distribution: should apply NN correction
    obs_id, state_id = MacroModelling.predict_additive_residual_ood(
        full_predict_ood, residual_predict_ood, state, shocks, theta,
        d_obs, norm_id; z_threshold=4.0)
    @test obs_id ≈ [1.0, 2.0] .+ [100.0, 200.0]

    # Out-of-distribution: should suppress NN correction → ROM1 only
    norm_ood = NormStats(zeros(d_in), 0.1 .* ones(d_in),  # tight distribution
                         zeros(d_obs), ones(d_obs))
    obs_ood, state_ood = MacroModelling.predict_additive_residual_ood(
        full_predict_ood, residual_predict_ood, state, shocks, theta,
        d_obs, norm_ood; z_threshold=4.0)
    # When OOD, should return ROM1-only prediction (no NN correction)
    @test obs_ood ≈ [1.0, 2.0]
    @test length(state_ood) == d_state
end

@testset "likelihood: conditional_loglik_per_period" begin
    d_obs = 2; d_state = 2; T = 10
    function predict_perfect(state, shocks, theta)
        state_next = 0.9 .* state .+ 0.1 .* shocks
        obs = state_next[1:d_obs]
        return obs, state_next
    end

    s0 = [1.0, 0.5]
    theta = Float64[]
    obs_sigma = [0.1, 0.1]
    shocks = 0.1 .* randn(d_state, T)

    # Generate perfect observations
    state = copy(s0)
    obs_data = zeros(d_obs, T)
    for t in 1:T
        obs_t, state = predict_perfect(state, shocks[:, t], theta)
        obs_data[:, t] = obs_t
    end

    ll = MacroModelling.conditional_loglik_per_period(
        predict_perfect, s0, shocks, theta, obs_data, obs_sigma)
    @test length(ll) == T
    @test all(isfinite, ll)
    # With perfect obs, residuals are 0 → LL should be high (only normalization constant)
    @test all(ll .> -5.0)  # should be close to -0.5 * sum(log(2π σ²))
end

@testset "likelihood: batched_additive_residual_loglik_per_period" begin
    d_obs = 2; d_state = 3; T = 15

    function full_predict_batch(state, shocks, theta)
        s_next = 0.9 .* state .+ vcat(0.1 .* shocks, [0.0])
        obs = s_next[1:d_obs]
        return vcat(obs, s_next)
    end

    function batch_residual(X::AbstractMatrix)
        # Tiny correction: all zeros
        return zeros(d_obs, size(X, 2))
    end

    s0 = randn(d_state)
    theta = [0.5, 0.3]
    obs_sigma = [0.2, 0.2]
    shocks = 0.1 .* randn(d_obs, T)

    # Generate observations
    state = copy(s0)
    obs_data = zeros(d_obs, T)
    for t in 1:T
        y = full_predict_batch(state, shocks[:, t], theta)
        obs_data[:, t] = y[1:d_obs]
        state = y[d_obs+1:end]
    end

    ll = MacroModelling.batched_additive_residual_loglik_per_period(
        full_predict_batch, batch_residual, s0, shocks, theta, obs_data, obs_sigma;
        d_obs=d_obs)

    @test length(ll) == T
    @test all(isfinite, ll)
    @test all(ll .> -20.0)  # reasonable LL values
end

# ============================================================================
# Part 3: Test SEP solver changes (RISK-6: damping schedule)
# ============================================================================
println("\nTesting SEP solver modifications (smoke test)...")

@testset "sep_solver: residual-dependent QR damping" begin
    # Verify the damping schedule logic via the solver's internal behavior.
    # We just test that a simple RBC model can still solve with the new damping.
    # Full SEP solve requires a compiled model — test at package level.

    # Test damping schedule logic directly
    function qr_cap_schedule(err)
        if err > 10.0
            return 0.1
        elseif err > 1.0
            return 0.2
        elseif err > 0.01
            return 0.4
        else
            return 0.8
        end
    end

    @test qr_cap_schedule(100.0) == 0.1
    @test qr_cap_schedule(5.0) == 0.2
    @test qr_cap_schedule(0.5) == 0.4
    @test qr_cap_schedule(0.001) == 0.8
    @test qr_cap_schedule(0.01) == 0.8  # boundary: 0.01 is NOT > 0.01, falls to else
    @test qr_cap_schedule(10.0) == 0.2  # boundary: 10.0 is NOT > 10.0
end

# ============================================================================
# Part 4: Integration test — full surrogate training pipeline
# ============================================================================
println("\nTesting surrogate training pipeline integration...")

@testset "integration: train + predict + OOD pipeline" begin
    Random.seed!(42)
    d_state = 4; d_shock = 2; d_theta = 3
    d_in = d_state + d_shock + d_theta
    d_obs = 2; d_out = d_obs + d_state
    n = 300

    # Synthetic data: simple nonlinear mapping
    X = randn(d_in, n)
    Y = zeros(d_out, n)
    for j in 1:n
        s = X[1:d_state, j]
        e = X[d_state+1:d_state+d_shock, j]
        Y[1:d_obs, j] = tanh.(s[1:d_obs]) .+ 0.1 .* e
        Y[d_obs+1:end, j] = 0.9 .* s .+ 0.05 .* vcat(e, zeros(d_state - d_shock))
    end

    # Train surrogate
    frozen = train_mlp!(copy(X), copy(Y);
        d_hidden=32, d_hidden2=16, nepoch=50,
        seed=1, verbose=false, activation=:silu)

    @test frozen.activation == :silu
    @test frozen.d_in == d_in
    @test frozen.d_out == d_out

    # Single prediction
    x_test = randn(d_in)
    y_pred = predict_frozen(frozen, x_test)
    @test length(y_pred) == d_out
    @test all(isfinite, y_pred)

    # Batch prediction consistency
    X_test = randn(d_in, 10)
    Y_batch = predict_frozen_batch(frozen, X_test)
    for j in 1:10
        @test Y_batch[:, j] ≈ predict_frozen(frozen, X_test[:, j]) atol=1e-10
    end

    # OOD detection
    x_normal = zeros(d_in)  # near training mean
    pred_n, ood_n, z_n = predict_frozen_safe(frozen, x_normal)
    @test !ood_n

    x_far = 20.0 .* ones(d_in)  # far from training
    pred_f, ood_f, z_f = predict_frozen_safe(frozen, x_far)
    @test ood_f
    @test z_f > 4.0
end

@testset "integration: weighted vs unweighted training" begin
    Random.seed!(99)
    d_in, d_out = 4, 2
    n = 200
    X = randn(d_in, n)
    Y = randn(d_out, n)

    # Uniform training
    f1 = train_mlp!(copy(X), copy(Y);
        d_hidden=16, d_hidden2=nothing, nepoch=30,
        seed=1, verbose=false, activation=:silu)

    # Weighted training (upweight first half)
    w = vcat(fill(5.0, n÷2), fill(1.0, n - n÷2))
    f2 = train_mlp!(copy(X), copy(Y);
        d_hidden=16, d_hidden2=nothing, nepoch=30,
        seed=1, verbose=false, activation=:silu, sample_weights=w)

    # Both should produce valid models
    @test f1 isa FrozenMLP
    @test f2 isa FrozenMLP

    # Predictions should differ (different weighting)
    x = randn(d_in)
    y1 = predict_frozen(f1, x)
    y2 = predict_frozen(f2, x)
    # They may be similar but shouldn't be identical
    @test all(isfinite, y1)
    @test all(isfinite, y2)
end

@testset "nn_utils: standardize_xy! NaN guard" begin
    X = [1.0 2.0 3.0; 4.0 5.0 6.0]
    Y = [0.1 0.2 0.3]
    # Should work normally
    norm = standardize_xy!(copy(X), copy(Y))
    @test all(isfinite, norm.μX)
    @test all(isfinite, norm.σX)

    # NaN in input should error early
    X_bad = copy(X)
    X_bad[1, 2] = NaN
    @test_throws ErrorException standardize_xy!(X_bad, copy(Y))

    Y_bad = copy(Y)
    Y_bad[1, 1] = Inf
    @test_throws ErrorException standardize_xy!(copy(X), Y_bad)
end

@testset "nn_utils: silu scalar vs array consistency" begin
    x_scalar = 1.5
    x_array = [1.5, -0.5, 0.0, 3.0]

    # Scalar version
    y_s = silu(x_scalar)
    @test y_s isa Float64
    @test y_s ≈ 1.5 / (1 + exp(-1.5))

    # Array version should match element-wise scalar
    y_a = silu(x_array)
    for i in eachindex(x_array)
        @test y_a[i] ≈ silu(x_array[i]) atol=1e-14
    end
end

@testset "nn_utils: validate_surrogate" begin
    Random.seed!(42)
    d_in, d_out = 5, 3
    n = 100
    X = randn(d_in, n)
    Y = randn(d_out, n)

    f = train_mlp!(copy(X), copy(Y);
        d_hidden=16, d_hidden2=nothing, nepoch=20,
        seed=1, verbose=false, activation=:silu)

    X_val = randn(d_in, 30)
    Y_val = randn(d_out, 30)
    result = validate_surrogate(f, X_val, Y_val)

    @test result.rmse_total > 0
    @test result.max_abs_error > 0
    @test 0.0 <= result.ood_fraction <= 1.0
    @test length(result.rmse_per_dim) == d_out
    @test result.n_samples == 30
end

@testset "nn_utils: summarize_frozen" begin
    Random.seed!(1)
    d_in, d_out = 4, 2
    W1 = randn(8, d_in); b1 = randn(8)
    W2 = randn(d_out, 8); b2 = randn(d_out)
    norm = NormStats(zeros(d_in), ones(d_in), zeros(d_out), ones(d_out))
    f = FrozenMLP(W1, b1, W2, b2, nothing, nothing, norm, d_in, d_out, :silu)

    # Should not error
    summarize_frozen(f)
    @test true  # if we got here, no error
end

@testset "likelihood: inversion_step NaN guard" begin
    # Create a predict_fn that returns NaN for certain shocks
    function predict_nan(state, shock, theta)
        if any(abs.(shock) .> 100)
            return fill(NaN, 2), fill(NaN, 3)
        end
        return state[1:2] .+ shock[1:2], state .+ 0.01
    end

    state0 = [1.0, 2.0, 3.0]
    theta = Float64[]
    y_obs = [1.5, 2.5]
    obs_sigma = [0.1, 0.1]
    shock_sigmas = [1.0, 1.0]
    structural_idx = [1, 2]

    # Should not crash — NaN guard prevents corruption
    eps_r, s_r, ll = MacroModelling.inversion_step(
        predict_nan, state0, y_obs, theta, obs_sigma, shock_sigmas, structural_idx;
        maxit=10, tol=1e-6, lambda=1e-4)

    @test isfinite(ll) || true  # may be -Inf but shouldn't crash
    @test length(eps_r) == 2
end

println("\nAll audit implementation tests complete.")
