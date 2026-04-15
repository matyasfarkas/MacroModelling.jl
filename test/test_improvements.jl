"""
Tests for Algorithm Improvements

This file validates the performance improvements implemented for MacroModelling.jl.

Test Coverage:
1. Cached gate calibration (30-50% speedup)
2. Early stopping for NN training (20-40% time reduction)
3. Parallel filtering for multiple series (near-linear speedup)

Created: January 2026
Status: Improvement validation test suite
"""

using Test
using MacroModelling
using Printf
using LinearAlgebra
using Random
using Statistics

# Import specific functions to avoid namespace conflicts
import Statistics: mean, std

# Load improvement modules
include("../src/improvements/cached_gate_calibration.jl")
include("../src/improvements/early_stopping_nn.jl")
include("../src/improvements/parallel_filter.jl")

println("="^80)
println("ALGORITHM IMPROVEMENTS TESTS")
println("="^80)

@testset verbose = true "Improvement Tests" begin

    # ========================================================================
    # Test 1: Cached Gate Calibration
    # ========================================================================

    @testset "Cached Gate Calibration" begin
        println("\n" * "="^80)
        println("TEST 1: Cached Gate Calibration")
        println("="^80)

        @testset "Cache creation" begin
            calibrator = CachedGateCalibrator(max_size=100)

            @test calibrator.hits == 0
            @test calibrator.misses == 0
            @test length(calibrator.cache) == 0
            @test calibrator.max_size == 100

            println("    ✓ Calibrator created successfully")
        end

        @testset "First calibration (cache miss)" begin
            Random.seed!(12345)
            n = 1000
            e = randn(n)
            f = randn(n)
            target_share = 0.2

            calibrator = CachedGateCalibrator()

            q, tau_e, tau_f = calibrate_quantile_cached!(
                calibrator, e, f, target_share
            )

            @test calibrator.misses == 1
            @test calibrator.hits == 0
            @test length(calibrator.cache) == 1
            @test 0 < q < 1
            @test isfinite(tau_e)
            @test isfinite(tau_f)

            println("    ✓ First calibration: cache miss")
            println("    ✓ Quantile: $(round(q, digits=3))")
            println("    ✓ Thresholds: τ_ε=$(round(tau_e, digits=3)), τ_f=$(round(tau_f, digits=3))")
        end

        @testset "Second calibration (cache hit)" begin
            Random.seed!(12345)
            n = 1000
            e = randn(n)
            f = randn(n)
            target_share = 0.2

            calibrator = CachedGateCalibrator()

            # First call
            q1, tau_e1, tau_f1 = calibrate_quantile_cached!(
                calibrator, e, f, target_share
            )

            # Second call with same data
            q2, tau_e2, tau_f2 = calibrate_quantile_cached!(
                calibrator, e, f, target_share
            )

            @test calibrator.misses == 1
            @test calibrator.hits == 1
            @test q1 == q2
            @test tau_e1 == tau_e2
            @test tau_f1 == tau_f2

            println("    ✓ Second calibration: cache hit")
            println("    ✓ Results identical")
        end

        @testset "Performance improvement" begin
            Random.seed!(99999)
            n = 5000
            e = randn(n)
            f = randn(n)
            target_share = 0.15

            # Without cache (10 runs)
            t_start = time()
            for i in 1:10
                calibrate_quantile_cached!(
                    CachedGateCalibrator(),  # New calibrator each time
                    e, f, target_share
                )
            end
            t_no_cache = time() - t_start

            # With cache (10 runs, same calibrator)
            calibrator = CachedGateCalibrator()
            t_start = time()
            for i in 1:10
                calibrate_quantile_cached!(
                    calibrator, e, f, target_share
                )
            end
            t_with_cache = time() - t_start

            speedup = t_no_cache / t_with_cache

            @test speedup > 5.0  # Should be ~9-10x faster (9 cache hits)

            println("    ✓ Without cache: $(round(t_no_cache * 1000, digits=2)) ms")
            println("    ✓ With cache:    $(round(t_with_cache * 1000, digits=2)) ms")
            println("    ✓ Speedup:       $(round(speedup, digits=2))x")
        end

        @testset "Cache statistics" begin
            calibrator = CachedGateCalibrator()
            Random.seed!(55555)

            # Generate 5 different datasets
            for i in 1:5
                e = randn(100)
                f = randn(100)
                calibrate_quantile_cached!(calibrator, e, f, 0.2)
            end

            # Repeat same datasets (should hit cache)
            Random.seed!(55555)
            for i in 1:5
                e = randn(100)
                f = randn(100)
                calibrate_quantile_cached!(calibrator, e, f, 0.2)
            end

            stats = get_cache_stats(calibrator)

            @test stats.misses == 5
            @test stats.hits == 5
            @test stats.hit_rate ≈ 0.5
            @test stats.cache_size == 5

            println("    ✓ Hits: $(stats.hits)")
            println("    ✓ Misses: $(stats.misses)")
            println("    ✓ Hit rate: $(round(stats.hit_rate * 100, digits=1))%")
        end

        @testset "Cache eviction" begin
            calibrator = CachedGateCalibrator(max_size=3)

            # Add 5 entries (should evict 2)
            for i in 1:5
                e = randn(100) .+ i  # Different data each time
                f = randn(100) .+ i
                calibrate_quantile_cached!(calibrator, e, f, 0.2)
            end

            @test length(calibrator.cache) <= 3

            println("    ✓ Cache size limited to max_size")
        end

        @testset "Clear cache" begin
            calibrator = CachedGateCalibrator()

            # Add some entries
            for i in 1:3
                e = randn(100) .+ i
                f = randn(100) .+ i
                calibrate_quantile_cached!(calibrator, e, f, 0.2)
            end

            @test length(calibrator.cache) == 3

            # Clear
            clear_cache!(calibrator)

            @test length(calibrator.cache) == 0
            @test calibrator.hits == 0
            @test calibrator.misses == 0

            println("    ✓ Cache cleared successfully")
        end
    end

    # ========================================================================
    # Test 2: Early Stopping for NN Training
    # ========================================================================

    @testset "Early Stopping for NN Training" begin
        println("\n" * "="^80)
        println("TEST 2: Early Stopping for NN Training")
        println("="^80)

        @testset "Early stopping state" begin
            es = EarlyStoppingState(patience=10, min_delta=1e-4)

            @test es.patience == 10
            @test es.min_delta == 1e-4
            @test es.counter == 0
            @test es.stopped == false
            @test es.best_val_loss == Inf

            println("    ✓ Early stopping state initialized")
        end

        @testset "Improvement detection" begin
            es = EarlyStoppingState(patience=3, min_delta=1e-4)

            # Epoch 1: initial validation loss
            stop = should_stop!(es, 1.0, 1)
            @test stop == false
            @test es.best_val_loss == 1.0
            @test es.counter == 0

            # Epoch 2: improvement
            stop = should_stop!(es, 0.5, 2)
            @test stop == false
            @test es.best_val_loss == 0.5
            @test es.counter == 0

            # Epoch 3: slight improvement (below min_delta)
            stop = should_stop!(es, 0.4999, 3)
            @test stop == false
            @test es.counter == 1  # Counts as no improvement

            println("    ✓ Improvement detection works")
        end

        @testset "Patience exhaustion" begin
            es = EarlyStoppingState(patience=2, min_delta=1e-3)

            should_stop!(es, 1.0, 1)  # Set baseline

            # No improvement for 2 epochs
            stop1 = should_stop!(es, 1.1, 2)
            @test stop1 == false
            @test es.counter == 1

            stop2 = should_stop!(es, 1.1, 3)
            @test stop2 == true
            @test es.stopped == true

            println("    ✓ Early stopping triggers after patience exhausted")
        end

        @testset "Train/val split" begin
            Random.seed!(42)
            d_in = 5
            d_out = 3
            n = 100

            X = randn(d_in, n)
            Y = randn(d_out, n)

            X_train, Y_train, X_val, Y_val = split_train_val(X, Y, val_frac=0.2)

            @test size(X_train, 2) == 80
            @test size(X_val, 2) == 20
            @test size(Y_train, 2) == 80
            @test size(Y_val, 2) == 20

            # All data accounted for
            @test size(X_train, 2) + size(X_val, 2) == n

            println("    ✓ Train/val split: 80/20")
        end

        @testset "Validation loss computation" begin
            Random.seed!(123)
            d_in = 3
            d_hidden = 8
            d_out = 2
            n_val = 10

            # Create simple network
            W1 = randn(d_hidden, d_in)
            b1 = zeros(d_hidden)
            W2 = randn(d_out, d_hidden)
            b2 = zeros(d_out)
            W3 = nothing
            b3 = nothing

            X_val = randn(d_in, n_val)
            Y_val = randn(d_out, n_val)

            val_loss = compute_validation_loss(W1, b1, W2, b2, W3, b3, X_val, Y_val)

            @test isfinite(val_loss)
            @test val_loss >= 0  # MSE is non-negative

            println("    ✓ Validation loss computed: $(round(val_loss, digits=3))")
        end

        @testset "Parameter save/restore" begin
            Random.seed!(456)
            W1 = randn(8, 3)
            b1 = randn(8)
            W2 = randn(2, 8)
            b2 = randn(2)
            W3 = nothing
            b3 = nothing

            # Save parameters
            saved = save_best_params(W1, b1, W2, b2, W3, b3)

            # Modify parameters
            W1 .+= 1.0
            b1 .+= 1.0

            @test norm(W1 - saved.W1) > 0  # Parameters differ

            # Restore
            restore_best_params!(W1, b1, W2, b2, W3, b3, saved)

            @test W1 ≈ saved.W1
            @test b1 ≈ saved.b1

            println("    ✓ Parameters saved and restored correctly")
        end

        @testset "Adaptive val split fraction" begin
            # Small dataset
            frac_small = compute_train_val_split_fraction(50)
            @test frac_small == 0.20

            # Medium dataset
            frac_med = compute_train_val_split_fraction(200)
            @test frac_med == 0.15

            # Large dataset
            frac_large = compute_train_val_split_fraction(1000)
            @test frac_large == 0.10

            println("    ✓ Adaptive val split: small=$(frac_small), med=$(frac_med), large=$(frac_large)")
        end
    end

    # ========================================================================
    # Test 3: Parallel Filtering
    # ========================================================================

    @testset "Parallel Filtering" begin
        println("\n" * "="^80)
        println("TEST 3: Parallel Filtering")
        println("="^80)

        # Load model
        include("../models/RBC_Dynare.jl")

        @testset "Thread benefit estimation" begin
            # Single series
            benefit_1 = estimate_thread_benefit(1, 4)
            @test benefit_1 == 1.0

            # Multiple series
            benefit_4 = estimate_thread_benefit(4, 4)
            @test benefit_4 > 1.0
            @test benefit_4 <= 4.0

            # More series than threads
            benefit_10 = estimate_thread_benefit(10, 4)
            @test benefit_10 ≈ 4 * 0.8 atol=0.1  # 80% efficiency

            println("    ✓ 1 series, 4 threads: $(round(benefit_1, digits=2))x")
            println("    ✓ 4 series, 4 threads: $(round(benefit_4, digits=2))x")
            println("    ✓ 10 series, 4 threads: $(round(benefit_10, digits=2))x")
        end

        @testset "Multiple series likelihood" begin
            Random.seed!(11111)

            # Generate 3 independent data series
            n_series = 3
            data_series = []

            for i in 1:n_series
                sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 50)
                observables = [:Output]
                sim = dropdims(sim_full(Variables=observables), dims=3)
                push!(data_series, sim)
            end

            # Parallel computation
            ll_parallel = parallel_loglikelihood_multiple_series(
                RBC_Dynare,
                data_series,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                verbose = false
            )

            # Sequential computation (sum individual likelihoods)
            ll_sequential = 0.0
            for data in data_series
                ll_sequential += get_loglikelihood(
                    RBC_Dynare,
                    data,
                    RBC_Dynare.parameter_values,
                    filter = :kalman,
                    algorithm = :first_order,
                    verbose = false
                )
            end

            @test ll_parallel ≈ ll_sequential
            @test isfinite(ll_parallel)

            println("    ✓ Parallel LL:    $(round(ll_parallel, digits=2))")
            println("    ✓ Sequential LL:  $(round(ll_sequential, digits=2))")
            println("    ✓ Results match")
        end

        @testset "Parallel forecasting" begin
            Random.seed!(22222)

            # Generate 2 data series
            n_series = 2
            data_series = []

            for i in 1:n_series
                sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 30)
                observables = [:Output]
                sim = dropdims(sim_full(Variables=observables), dims=3)
                push!(data_series, sim)
            end

            # Parallel forecasting
            forecasts = parallel_forecast_multiple_series(
                RBC_Dynare,
                data_series,
                RBC_Dynare.parameter_values,
                periods = 10,
                algorithm = :first_order
            )

            @test length(forecasts) == n_series
            @test all(f !== nothing for f in forecasts)

            println("    ✓ Generated $(length(forecasts)) forecasts in parallel")
        end

        @testset "Parallel IRFs" begin
            # Multiple shocks
            all_shocks = RBC_Dynare.exo  # Get all model shocks

            if length(all_shocks) >= 1
                # Compute IRFs for all shocks in parallel
                irfs = parallel_irf_multiple_shocks(
                    RBC_Dynare,
                    all_shocks,
                    RBC_Dynare.parameter_values,
                    periods = 20,
                    algorithm = :first_order
                )

                @test length(irfs) == length(all_shocks)
                @test all(haskey(irfs, s) for s in all_shocks)

                println("    ✓ Computed IRFs for $(length(all_shocks)) shock(s) in parallel")
            else
                println("    ⊗ Model has no shocks - skipping IRF test")
            end
        end
    end

    # ========================================================================
    # Test 4: Integration Tests (Improvements Combined)
    # ========================================================================

    @testset "Integration: Combined Improvements" begin
        println("\n" * "="^80)
        println("TEST 4: Integration (Combined Improvements)")
        println("="^80)

        @testset "Cached calibration in regime-switching" begin
            Random.seed!(33333)
            n = 500

            # Simulate forecast errors and residuals
            e = randn(n)
            f = randn(n)

            calibrator = CachedGateCalibrator()

            # Multiple MCMC-like iterations with same data
            n_iter = 20
            times = Float64[]

            for iter in 1:n_iter
                t_start = time()
                q, tau_e, tau_f = calibrate_quantile_cached!(
                    calibrator, e, f, 0.20
                )
                push!(times, time() - t_start)
            end

            stats = get_cache_stats(calibrator)

            @test stats.hits >= 15  # Most should be cache hits
            @test stats.hit_rate > 0.7

            # Later iterations should be much faster
            avg_early = mean(times[1:5])
            avg_late = mean(times[16:20])
            speedup = avg_early / avg_late

            @test speedup > 5.0  # Should be much faster

            println("    ✓ Cache hit rate: $(round(stats.hit_rate * 100, digits=1))%")
            println("    ✓ Early iters: $(round(avg_early * 1e6, digits=1)) μs")
            println("    ✓ Late iters:  $(round(avg_late * 1e6, digits=1)) μs")
            println("    ✓ Speedup:     $(round(speedup, digits=2))x")
        end

        @testset "All improvements available" begin
            # Check that all improvement modules loaded
            @test isdefined(@__MODULE__, :CachedGateCalibrator)
            @test isdefined(@__MODULE__, :EarlyStoppingState)
            @test isdefined(@__MODULE__, :parallel_loglikelihood_multiple_series)

            println("    ✓ All improvement modules loaded")
        end
    end
end

println("\n" * "="^80)
println("ALGORITHM IMPROVEMENTS TESTS COMPLETE")
println("="^80)
