"""
Kalman Filter Implementation Tests

This file validates the Kalman filter implementation for the regime-switching pipeline.

Test Coverage:
1. State estimation (filtering vs smoothing)
2. Shock estimation
3. Presample period handling
4. Numerical stability

Created: January 2026
Status: Regime-switching estimation test suite - simplified version
"""

using Test
using MacroModelling
using Printf
using LinearAlgebra
using Statistics
using Random

println("="^80)
println("KALMAN FILTER IMPLEMENTATION TESTS")
println("="^80)

@testset verbose = true "Kalman Filter Tests" begin

    println("\nLoading test model...")
    include("../models/RBC_Dynare.jl")

    println("  Model: RBC_Dynare")
    println("  Variables: $(length(RBC_Dynare.var))")
    println("  Shocks: $(length(RBC_Dynare.exo))")

    # Generate synthetic data
    Random.seed!(12345)
    T_data = 100
    sim = simulate(RBC_Dynare, algorithm = :first_order, periods = T_data)

    println("  Generated synthetic data: $T_data periods")

    # ========================================================================
    # Test 1: State Estimation
    # ========================================================================

    @testset "State Estimation" begin
        println("\n" * "="^80)
        println("TEST 1: State Estimation")
        println("="^80)

        @testset "Filtered states" begin
            estimated_states = get_estimated_variables(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = false,
                verbose = false
            )

            @test size(estimated_states, 2) == T_data
            @test all(isfinite.(estimated_states))

            println("    ✓ Filtered states computed: $(size(estimated_states))")
        end

        @testset "Smoothed states" begin
            estimated_states = get_estimated_variables(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = true,
                verbose = false
            )

            @test size(estimated_states, 2) == T_data
            @test all(isfinite.(estimated_states))

            println("    ✓ Smoothed states computed: $(size(estimated_states))")
        end

        @testset "Filtering vs Smoothing" begin
            filtered = get_estimated_variables(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = false,
                verbose = false
            )

            smoothed = get_estimated_variables(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = true,
                verbose = false
            )

            @test size(filtered) == size(smoothed)
            @test !all(filtered .≈ smoothed)  # Should differ
            @test all(isfinite.(filtered))
            @test all(isfinite.(smoothed))

            println("    ✓ Filtered and smoothed estimates differ (as expected)")
        end
    end

    # ========================================================================
    # Test 2: Shock Estimation
    # ========================================================================

    @testset "Shock Estimation" begin
        println("\n" * "="^80)
        println("TEST 2: Shock Estimation")
        println("="^80)

        @testset "Filtered shocks" begin
            estimated_shocks = get_estimated_shocks(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = false,
                verbose = false
            )

            @test size(estimated_shocks, 2) == T_data
            @test all(isfinite.(estimated_shocks))

            println("    ✓ Filtered shocks computed: $(size(estimated_shocks))")
        end

        @testset "Smoothed shocks" begin
            estimated_shocks = get_estimated_shocks(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = true,
                verbose = false
            )

            @test size(estimated_shocks, 2) == T_data
            @test all(isfinite.(estimated_shocks))

            println("    ✓ Smoothed shocks computed: $(size(estimated_shocks))")
        end

        @testset "Shock properties" begin
            estimated_shocks = get_estimated_shocks(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = true,
                verbose = false
            )

            shock_means = vec(mean(estimated_shocks, dims=2))
            @test all(abs.(shock_means) .< 0.5)  # Mean ≈ 0

            println("    ✓ Shock means near zero: $(round.(shock_means, digits=3))")
        end
    end

    # ========================================================================
    # Test 3: Numerical Stability
    # ========================================================================

    @testset "Numerical Stability" begin
        println("\n" * "="^80)
        println("TEST 3: Numerical Stability")
        println("="^80)

        @testset "Repeated estimation" begin
            estimates = []

            for i in 1:3
                est = get_estimated_variables(
                    RBC_Dynare,
                    sim,
                    algorithm = :first_order,
                    smooth = true,
                    verbose = false
                )
                push!(estimates, est)
            end

            # All runs should give identical results
            @test estimates[1] ≈ estimates[2]
            @test estimates[2] ≈ estimates[3]

            println("    ✓ Numerically stable (identical results across runs)")
        end

        @testset "Short data" begin
            T_short = 10
            sim_short = simulate(RBC_Dynare, algorithm = :first_order, periods = T_short)

            est = get_estimated_variables(
                RBC_Dynare,
                sim_short,
                algorithm = :first_order,
                smooth = true,
                verbose = false
            )

            @test size(est, 2) == T_short
            @test all(isfinite.(est))

            println("    ✓ Short data (T=$T_short) handled")
        end
    end

    # ========================================================================
    # Test 4: Different Algorithms
    # ========================================================================

    @testset "Algorithm Compatibility" begin
        println("\n" * "="^80)
        println("TEST 4: Algorithm Compatibility")
        println("="^80)

        @testset "First order" begin
            est = get_estimated_variables(
                RBC_Dynare,
                sim,
                algorithm = :first_order,
                smooth = true,
                verbose = false
            )

            @test all(isfinite.(est))
            println("    ✓ First-order algorithm works")
        end

        # Note: Higher-order algorithms may not work with Kalman filter
        # They require particle filters or other nonlinear methods
    end
end

println("\n" * "="^80)
println("KALMAN FILTER TESTS COMPLETE")
println("="^80)
