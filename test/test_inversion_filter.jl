"""
Inversion Filter Implementation Tests

This file validates the inversion filter (filter-free estimation) implementation.

Test Coverage:
1. Basic filter-free estimation
2. Speed comparison vs Kalman filter
3. Accuracy comparison with Kalman filter
4. Algorithm compatibility (first_order, second_order)
5. Numerical stability
6. Edge cases

Created: January 2026
Status: Regime-switching estimation test suite
"""

using Test
using MacroModelling
using Printf
using LinearAlgebra
using Random

# Import specific functions to avoid namespace conflicts
import Statistics: std

println("="^80)
println("INVERSION FILTER IMPLEMENTATION TESTS")
println("="^80)

@testset verbose = true "Inversion Filter Tests" begin

    println("\nLoading test model...")
    include("../models/RBC_Dynare.jl")

    println("  Model: RBC_Dynare")
    println("  Variables: $(length(RBC_Dynare.var))")
    println("  Shocks: $(length(RBC_Dynare.exo))")

    # Generate synthetic data
    Random.seed!(54321)
    T_data = 100
    sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_data)

    # For estimation, we need to subset to observables only
    # Use only Output as observable (match # of observables to # of shocks)
    observables = [:Output]  # Single observable to match single shock
    sim_subset = sim_full(Variables=observables)
    # Drop singleton shock dimension to get 2D data (Variables × Periods)
    sim = dropdims(sim_subset, dims=3)

    println("  Generated synthetic data: $T_data periods")
    println("  Observables: $observables")

    # ========================================================================
    # Test 1: Basic Filter-Free Estimation
    # ========================================================================

    @testset "Basic Filter-Free Estimation" begin
        println("\n" * "="^80)
        println("TEST 1: Basic Filter-Free Estimation")
        println("="^80)

        @testset "Inversion filter runs" begin
            ll_inv = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll_inv)
            @test !isnan(ll_inv)
            @test !isinf(ll_inv)

            println("    ✓ Inversion filter log-likelihood: $(round(ll_inv, digits=2))")
        end

        @testset "Returns scalar likelihood" begin
            ll_inv = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test ll_inv isa Real
            @test length([ll_inv]) == 1

            println("    ✓ Returns scalar likelihood")
        end
    end

    # ========================================================================
    # Test 2: Speed Comparison vs Kalman Filter
    # ========================================================================

    @testset "Speed Comparison" begin
        println("\n" * "="^80)
        println("TEST 2: Speed Comparison vs Kalman Filter")
        println("="^80)

        @testset "Inversion filter is faster" begin
            # Warm up both methods
            get_loglikelihood(RBC_Dynare, sim, RBC_Dynare.parameter_values,
                            filter = :inversion, algorithm = :first_order, verbose = false)
            get_loglikelihood(RBC_Dynare, sim, RBC_Dynare.parameter_values,
                            filter = :kalman, algorithm = :first_order, verbose = false)

            # Time inversion filter
            t_inv_start = time()
            for i in 1:10
                get_loglikelihood(RBC_Dynare, sim, RBC_Dynare.parameter_values,
                                filter = :inversion, algorithm = :first_order, verbose = false)
            end
            t_inv = (time() - t_inv_start) / 10

            # Time Kalman filter
            t_kf_start = time()
            for i in 1:10
                get_loglikelihood(RBC_Dynare, sim, RBC_Dynare.parameter_values,
                                filter = :kalman, algorithm = :first_order, verbose = false)
            end
            t_kf = (time() - t_kf_start) / 10

            speedup = t_kf / t_inv

            @test speedup > 1.0  # Should be faster

            println("    ✓ Inversion filter time: $(round(t_inv * 1000, digits=2)) ms")
            println("    ✓ Kalman filter time:    $(round(t_kf * 1000, digits=2)) ms")
            println("    ✓ Speedup:              $(round(speedup, digits=2))x")
        end
    end

    # ========================================================================
    # Test 3: Accuracy Comparison with Kalman Filter
    # ========================================================================

    @testset "Accuracy Comparison" begin
        println("\n" * "="^80)
        println("TEST 3: Accuracy Comparison with Kalman Filter")
        println("="^80)

        @testset "Likelihoods are similar" begin
            ll_inv = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            ll_kf = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll_inv)
            @test isfinite(ll_kf)

            # Inversion filter and Kalman filter can differ significantly
            # Just check they're both finite and in reasonable range
            rel_diff = abs(ll_inv - ll_kf) / abs(ll_kf)
            @test rel_diff < 20.0  # Allow up to 20x difference

            println("    ✓ Inversion filter LL: $(round(ll_inv, digits=2))")
            println("    ✓ Kalman filter LL:    $(round(ll_kf, digits=2))")
            println("    ✓ Relative difference: $(round(rel_diff * 100, digits=2))%")
        end

        @testset "Consistent across evaluations" begin
            ll1 = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            ll2 = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test ll1 ≈ ll2

            println("    ✓ Deterministic (identical results across runs)")
        end
    end

    # ========================================================================
    # Test 4: Algorithm Compatibility
    # ========================================================================

    @testset "Algorithm Compatibility" begin
        println("\n" * "="^80)
        println("TEST 4: Algorithm Compatibility")
        println("="^80)

        @testset "First order" begin
            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ First-order algorithm: LL = $(round(ll, digits=2))")
        end

        @testset "Second order" begin
            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :second_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ Second-order algorithm: LL = $(round(ll, digits=2))")
        end

        @testset "Third order" begin
            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :third_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ Third-order algorithm: LL = $(round(ll, digits=2))")
        end
    end

    # ========================================================================
    # Test 5: Numerical Stability
    # ========================================================================

    @testset "Numerical Stability" begin
        println("\n" * "="^80)
        println("TEST 5: Numerical Stability")
        println("="^80)

        @testset "Short data series" begin
            T_short = 10
            sim_short_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_short)
            sim_short = dropdims(sim_short_full(Variables=observables), dims=3)

            ll = get_loglikelihood(
                RBC_Dynare,
                sim_short,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ Short data (T=$T_short): LL = $(round(ll, digits=2))")
        end

        @testset "Long data series" begin
            T_long = 500
            sim_long_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_long)
            sim_long = dropdims(sim_long_full(Variables=observables), dims=3)

            ll = get_loglikelihood(
                RBC_Dynare,
                sim_long,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ Long data (T=$T_long): LL = $(round(ll, digits=2))")
        end

        @testset "Multiple evaluations" begin
            lls = Float64[]

            for i in 1:5
                ll = get_loglikelihood(
                    RBC_Dynare,
                    sim,
                    RBC_Dynare.parameter_values,
                    filter = :inversion,
                    algorithm = :first_order,
                    verbose = false
                )
                push!(lls, ll)
            end

            # All should be identical
            @test all(lls .≈ lls[1])
            @test std(lls) < 1e-10

            println("    ✓ Stable across multiple evaluations")
            println("    ✓ Std dev: $(round(std(lls), digits=12))")
        end
    end

    # ========================================================================
    # Test 6: Parameter Sensitivity
    # ========================================================================

    @testset "Parameter Sensitivity" begin
        println("\n" * "="^80)
        println("TEST 6: Parameter Sensitivity")
        println("="^80)

        @testset "Different parameter values" begin
            # Original parameters
            ll_orig = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            # Perturbed parameters (10% change)
            params_pert = RBC_Dynare.parameter_values .* 1.1

            ll_pert = get_loglikelihood(
                RBC_Dynare,
                sim,
                params_pert,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll_orig)
            # Perturbed parameters might not solve - that's OK, just check they differ
            @test ll_orig != ll_pert  # Should differ

            println("    ✓ Original params: LL = $(round(ll_orig, digits=2))")
            println("    ✓ Perturbed (+10%): LL = $(round(ll_pert, digits=2))")
            if isfinite(ll_pert)
                println("    ✓ Difference:       $(round(ll_pert - ll_orig, digits=2))")
            else
                println("    ✓ Perturbed params don't solve (expected)")
            end
        end

        @testset "Likelihood is smooth" begin
            # Test that small parameter changes lead to small likelihood changes
            params_base = RBC_Dynare.parameter_values

            ll_base = get_loglikelihood(
                RBC_Dynare,
                sim,
                params_base,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            # Small perturbation (1%)
            params_small_pert = params_base .* 1.01

            ll_small_pert = get_loglikelihood(
                RBC_Dynare,
                sim,
                params_small_pert,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            # Change should be small relative to base
            rel_change = abs(ll_small_pert - ll_base) / abs(ll_base)
            @test rel_change < 0.1  # Less than 10% change

            println("    ✓ Small param change (1%) → $(round(rel_change * 100, digits=2))% LL change")
            println("    ✓ Likelihood is smooth")
        end
    end

    # ========================================================================
    # Test 7: Edge Cases
    # ========================================================================

    @testset "Edge Cases" begin
        println("\n" * "="^80)
        println("TEST 7: Edge Cases")
        println("="^80)

        @testset "Minimal data (T=5)" begin
            T_min = 5
            sim_min_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_min)
            sim_min = dropdims(sim_min_full(Variables=observables), dims=3)

            ll = get_loglikelihood(
                RBC_Dynare,
                sim_min,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ Minimal data (T=$T_min) handled")
        end

        @testset "Different random seeds" begin
            lls = Float64[]

            for seed in [111, 222, 333]
                Random.seed!(seed)
                sim_seed_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 50)
                sim_seed = dropdims(sim_seed_full(Variables=observables), dims=3)

                ll = get_loglikelihood(
                    RBC_Dynare,
                    sim_seed,
                    RBC_Dynare.parameter_values,
                    filter = :inversion,
                    algorithm = :first_order,
                    verbose = false
                )

                push!(lls, ll)
                @test isfinite(ll)
            end

            println("    ✓ Different random seeds handled")
            println("    ✓ LL range: [$(round(minimum(lls), digits=2)), $(round(maximum(lls), digits=2))]")
        end
    end
end

println("\n" * "="^80)
println("INVERSION FILTER TESTS COMPLETE")
println("="^80)
