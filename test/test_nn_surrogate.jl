"""
Neural Network Surrogate Tests

This file validates the neural network surrogate implementation for regime-switching estimation.

Test Coverage:
1. MLP architecture (1-layer and 2-layer networks)
2. Training convergence and stability
3. Prediction accuracy (forward pass)
4. Batched inference optimization
5. Normalization and denormalization
6. Frozen model serialization

Created: January 2026
Status: Regime-switching estimation test suite
"""

using Test
using MacroModelling
using Printf
using LinearAlgebra
using Random
using Serialization

# Import specific functions to avoid namespace conflicts
import Statistics: mean, std

# Load NN utilities
include("../scripts/hlt_surrogate/hlt_sep_surrogate_nn_utils.jl")

println("="^80)
println("NEURAL NETWORK SURROGATE TESTS")
println("="^80)

@testset verbose = true "NN Surrogate Tests" begin

    # ========================================================================
    # Test 1: MLP Architecture
    # ========================================================================

    @testset "MLP Architecture" begin
        println("\n" * "="^80)
        println("TEST 1: MLP Architecture")
        println("="^80)

        d_in = 10
        d_out = 5
        n_samples = 100

        # Generate synthetic training data
        Random.seed!(42)
        X = randn(d_in, n_samples)
        # Y = simple linear transformation + noise
        W_true = randn(d_out, d_in)
        Y = W_true * X .+ 0.1 * randn(d_out, n_samples)

        @testset "1-layer network" begin
            X_train = copy(X)
            Y_train = copy(Y)

            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=32,
                               d_hidden2=nothing,  # 1-layer
                               nepoch=50,
                               η_init=1e-2,
                               verbose=false)

            @test frozen isa FrozenMLP
            @test frozen.d_in == d_in
            @test frozen.d_out == d_out
            @test frozen.W3 === nothing  # 1-layer network
            @test frozen.b3 === nothing

            println("    ✓ 1-layer network created")
            println("    ✓ Architecture: $(d_in) → 32 → $(d_out)")
        end

        @testset "2-layer network" begin
            X_train = copy(X)
            Y_train = copy(Y)

            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=32,
                               d_hidden2=16,  # 2-layer
                               nepoch=50,
                               η_init=1e-2,
                               verbose=false)

            @test frozen isa FrozenMLP
            @test frozen.d_in == d_in
            @test frozen.d_out == d_out
            @test frozen.W3 !== nothing  # 2-layer network
            @test frozen.b3 !== nothing

            println("    ✓ 2-layer network created")
            println("    ✓ Architecture: $(d_in) → 32 → 16 → $(d_out)")
        end

        @testset "Network dimensions" begin
            X_train = copy(X)
            Y_train = copy(Y)

            d_h1 = 64
            d_h2 = 32

            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=d_h1,
                               d_hidden2=d_h2,
                               nepoch=10,
                               verbose=false)

            @test size(frozen.W1) == (d_h1, d_in)
            @test size(frozen.b1) == (d_h1,)
            @test size(frozen.W2) == (d_h2, d_h1)
            @test size(frozen.b2) == (d_h2,)
            @test size(frozen.W3) == (d_out, d_h2)
            @test size(frozen.b3) == (d_out,)

            println("    ✓ Network dimensions correct")
        end
    end

    # ========================================================================
    # Test 2: Training Convergence
    # ========================================================================

    @testset "Training Convergence" begin
        println("\n" * "="^80)
        println("TEST 2: Training Convergence")
        println("="^80)

        d_in = 5
        d_out = 3
        n_samples = 200

        Random.seed!(123)
        X = randn(d_in, n_samples)
        W_true = randn(d_out, d_in)
        Y = W_true * X .+ 0.05 * randn(d_out, n_samples)

        @testset "Basic convergence" begin
            X_train = copy(X)
            Y_train = copy(Y)

            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=64,
                               d_hidden2=32,
                               nepoch=100,
                               η_init=1e-2,
                               verbose=false)

            @test frozen isa FrozenMLP

            # Test prediction on training data
            Y_pred = zeros(d_out, n_samples)
            for i in 1:n_samples
                Y_pred[:, i] = predict_frozen(frozen, X[:, i])
            end

            # Training error should be small
            train_mse = mean((Y_pred .- Y).^2)

            @test train_mse < 0.5  # Should fit reasonably well

            println("    ✓ Network converged")
            println("    ✓ Training MSE: $(round(train_mse, digits=4))")
        end

        @testset "Overfitting prevention (weight decay)" begin
            X_train = copy(X)
            Y_train = copy(Y)

            # Train with weight decay
            frozen_wd = train_mlp!(X_train, Y_train,
                                  d_hidden=128,
                                  d_hidden2=64,
                                  nepoch=200,
                                  η_init=1e-2,
                                  weight_decay=1e-4,
                                  verbose=false)

            # Train without weight decay
            X_train2 = copy(X)
            Y_train2 = copy(Y)
            frozen_no_wd = train_mlp!(X_train2, Y_train2,
                                      d_hidden=128,
                                      d_hidden2=64,
                                      nepoch=200,
                                      η_init=1e-2,
                                      weight_decay=0.0,
                                      verbose=false)

            # Weight decay should lead to smaller weights
            w_norm_wd = norm(frozen_wd.W1) + norm(frozen_wd.W2)
            w_norm_no_wd = norm(frozen_no_wd.W1) + norm(frozen_no_wd.W2)

            @test w_norm_wd < w_norm_no_wd

            println("    ✓ Weight decay reduces weight norms")
            println("    ✓ With WD: $(round(w_norm_wd, digits=2))")
            println("    ✓ Without WD: $(round(w_norm_no_wd, digits=2))")
        end

        @testset "Learning rate schedule" begin
            X_train = copy(X)
            Y_train = copy(Y)

            # Train with cosine schedule (default)
            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=64,
                               nepoch=100,
                               η_init=1e-2,
                               verbose=false)

            @test frozen isa FrozenMLP

            println("    ✓ Cosine LR schedule training completed")
        end
    end

    # ========================================================================
    # Test 3: Prediction Accuracy
    # ========================================================================

    @testset "Prediction Accuracy" begin
        println("\n" * "="^80)
        println("TEST 3: Prediction Accuracy")
        println("="^80)

        d_in = 8
        d_out = 4
        n_train = 500
        n_test = 100

        Random.seed!(456)
        X_train = randn(d_in, n_train)
        W_true = randn(d_out, d_in)
        Y_train = W_true * X_train .+ 0.05 * randn(d_out, n_train)

        X_test = randn(d_in, n_test)
        Y_test = W_true * X_test .+ 0.05 * randn(d_out, n_test)

        X_train_copy = copy(X_train)
        Y_train_copy = copy(Y_train)

        frozen = train_mlp!(X_train_copy, Y_train_copy,
                           d_hidden=128,
                           d_hidden2=64,
                           nepoch=150,
                           η_init=1e-2,
                           verbose=false)

        @testset "Single prediction" begin
            x_test = X_test[:, 1]
            y_pred = predict_frozen(frozen, x_test)

            @test length(y_pred) == d_out
            @test all(isfinite.(y_pred))

            println("    ✓ Single prediction works")
        end

        @testset "Batch prediction" begin
            Y_pred = predict_frozen_batch(frozen, X_test)

            @test size(Y_pred) == (d_out, n_test)
            @test all(isfinite.(Y_pred))

            # Batch should match individual predictions
            y_single = predict_frozen(frozen, X_test[:, 1])
            y_batch = Y_pred[:, 1]

            @test y_single ≈ y_batch

            println("    ✓ Batch prediction works")
            println("    ✓ Batch matches single predictions")
        end

        @testset "Test set accuracy" begin
            Y_pred = predict_frozen_batch(frozen, X_test)
            test_mse = mean((Y_pred .- Y_test).^2)

            @test test_mse < 1.0  # Should generalize reasonably

            println("    ✓ Test MSE: $(round(test_mse, digits=4))")
        end

        @testset "Prediction consistency" begin
            # Same input should give same output
            x = X_test[:, 1]
            y1 = predict_frozen(frozen, x)
            y2 = predict_frozen(frozen, x)

            @test y1 ≈ y2

            println("    ✓ Predictions are deterministic")
        end
    end

    # ========================================================================
    # Test 4: Normalization
    # ========================================================================

    @testset "Normalization" begin
        println("\n" * "="^80)
        println("TEST 4: Normalization")
        println("="^80)

        d_in = 5
        d_out = 3
        n_samples = 100

        Random.seed!(789)
        # Create data with different scales
        X = randn(d_in, n_samples)
        X[1, :] .*= 100  # Large scale
        X[2, :] .*= 0.01  # Small scale

        Y = randn(d_out, n_samples)
        Y[1, :] .*= 50

        @testset "Normalization statistics" begin
            X_copy = copy(X)
            Y_copy = copy(Y)

            norm = standardize_xy!(X_copy, Y_copy)

            # After normalization, mean ≈ 0, std ≈ 1
            # Note: standardize_xy! uses corrected=false (population std)
            @test all(abs.(mean(X_copy, dims=2)) .< 1e-10)
            @test all(abs.(std(X_copy, dims=2; corrected=false) .- 1.0) .< 1e-10)
            @test all(abs.(mean(Y_copy, dims=2)) .< 1e-10)
            @test all(abs.(std(Y_copy, dims=2; corrected=false) .- 1.0) .< 1e-10)

            println("    ✓ Data normalized correctly")
            println("    ✓ X mean ≈ 0, std ≈ 1")
            println("    ✓ Y mean ≈ 0, std ≈ 1")
        end

        @testset "Normalization in training" begin
            X_train = copy(X)
            Y_train = copy(Y)

            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=32,
                               nepoch=50,
                               verbose=false)

            # Frozen model should store normalization stats
            @test frozen.norm isa NormStats
            @test length(frozen.norm.μX) == d_in
            @test length(frozen.norm.σX) == d_in
            @test length(frozen.norm.μY) == d_out
            @test length(frozen.norm.σY) == d_out

            println("    ✓ Normalization stats stored")
        end

        @testset "Automatic denormalization" begin
            X_train = copy(X)
            Y_train = copy(Y)

            frozen = train_mlp!(X_train, Y_train,
                               d_hidden=32,
                               nepoch=50,
                               verbose=false)

            # Predictions should be in original scale
            x_test = X[:, 1]
            y_pred = predict_frozen(frozen, x_test)

            # Should be in same ballpark as original Y
            @test maximum(abs.(y_pred)) < 200  # Reasonable scale

            println("    ✓ Predictions automatically denormalized")
        end
    end

    # ========================================================================
    # Test 5: Serialization
    # ========================================================================

    @testset "Serialization" begin
        println("\n" * "="^80)
        println("TEST 5: Serialization")
        println("="^80)

        d_in = 6
        d_out = 4
        n_samples = 100

        Random.seed!(101112)
        X = randn(d_in, n_samples)
        Y = randn(d_out, n_samples)

        X_copy = copy(X)
        Y_copy = copy(Y)

        frozen = train_mlp!(X_copy, Y_copy,
                           d_hidden=64,
                           d_hidden2=32,
                           nepoch=50,
                           verbose=false)

        @testset "Serialize and deserialize" begin
            # Save to temp file
            temp_file = tempname() * ".jls"

            try
                # Serialize
                surrogate_data = Dict("frozen" => frozen)
                serialize(temp_file, surrogate_data)

                # Deserialize
                loaded = deserialize(temp_file)
                frozen_loaded = loaded["frozen"]

                @test frozen_loaded isa FrozenMLP
                @test frozen_loaded.d_in == frozen.d_in
                @test frozen_loaded.d_out == frozen.d_out

                # Predictions should match
                x_test = X[:, 1]
                y_orig = predict_frozen(frozen, x_test)
                y_loaded = predict_frozen(frozen_loaded, x_test)

                @test y_orig ≈ y_loaded

                println("    ✓ Serialization works")
                println("    ✓ Loaded model gives identical predictions")

            finally
                isfile(temp_file) && rm(temp_file)
            end
        end
    end

    # ========================================================================
    # Test 6: Edge Cases
    # ========================================================================

    @testset "Edge Cases" begin
        println("\n" * "="^80)
        println("TEST 6: Edge Cases")
        println("="^80)

        @testset "Small dataset" begin
            d_in = 3
            d_out = 2
            n_small = 20  # Very small

            Random.seed!(42)
            X = randn(d_in, n_small)
            Y = randn(d_out, n_small)

            X_copy = copy(X)
            Y_copy = copy(Y)

            frozen = train_mlp!(X_copy, Y_copy,
                               d_hidden=16,
                               nepoch=100,
                               verbose=false)

            @test frozen isa FrozenMLP

            println("    ✓ Small dataset (n=$n_small) handled")
        end

        @testset "High-dimensional input" begin
            d_in = 100  # High-dimensional
            d_out = 5
            n_samples = 200

            Random.seed!(42)
            X = randn(d_in, n_samples)
            Y = randn(d_out, n_samples)

            X_copy = copy(X)
            Y_copy = copy(Y)

            frozen = train_mlp!(X_copy, Y_copy,
                               d_hidden=128,
                               nepoch=50,
                               verbose=false)

            @test frozen isa FrozenMLP
            @test frozen.d_in == d_in

            println("    ✓ High-dimensional input (d=$d_in) handled")
        end

        @testset "Zero variance features" begin
            d_in = 5
            d_out = 3
            n_samples = 50

            Random.seed!(42)
            X = randn(d_in, n_samples)
            X[2, :] .= 0.0  # Zero variance feature

            Y = randn(d_out, n_samples)

            X_copy = copy(X)
            Y_copy = copy(Y)

            # Should handle zero variance (sets σ = 1.0)
            frozen = train_mlp!(X_copy, Y_copy,
                               d_hidden=32,
                               nepoch=50,
                               verbose=false)

            @test frozen isa FrozenMLP

            println("    ✓ Zero variance features handled")
        end
    end
end

println("\n" * "="^80)
println("NN SURROGATE TESTS COMPLETE")
println("="^80)
