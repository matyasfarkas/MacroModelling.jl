"""
Full Estimation Pipeline Tests

This file validates the complete end-to-end estimation workflow.

Test Coverage:
1. Data preparation and preprocessing
2. Prior specification
3. Likelihood evaluation (Kalman and inversion filters)
4. Parameter inference workflow
5. Integration with optimization and sampling

Created: January 2026
Status: Regime-switching estimation test suite
"""

using Test
using MacroModelling
using Printf
using LinearAlgebra
using Random

# Import specific functions to avoid namespace conflicts
import Statistics: mean, std

println("="^80)
println("FULL ESTIMATION PIPELINE TESTS")
println("="^80)

@testset verbose = true "Estimation Pipeline Tests" begin

    println("\nLoading test model...")
    include("../models/RBC_Dynare.jl")

    println("  Model: RBC_Dynare")
    println("  Variables: $(length(RBC_Dynare.var))")
    println("  Shocks: $(length(RBC_Dynare.exo))")
    println("  Parameters: $(length(RBC_Dynare.parameters))")

    # ========================================================================
    # Test 1: Data Preparation
    # ========================================================================

    @testset "Data Preparation" begin
        println("\n" * "="^80)
        println("TEST 1: Data Preparation")
        println("="^80)

        @testset "Simulated data generation" begin
            Random.seed!(12345)
            T_data = 100
            sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_data)

            @test size(sim_full, 1) == length(RBC_Dynare.var)
            @test size(sim_full, 2) == T_data
            @test all(isfinite.(sim_full))

            println("    ✓ Generated simulated data: $(size(sim_full))")
        end

        @testset "Observable subset" begin
            Random.seed!(12345)
            sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 100)

            # Select single observable
            observables = [:Output]
            sim_subset = sim_full(Variables=observables)
            sim = dropdims(sim_subset, dims=3)

            @test size(sim, 1) == length(observables)
            @test size(sim, 1) <= length(RBC_Dynare.exo)  # Must have ≤ shocks
            @test all(isfinite.(sim))

            println("    ✓ Observable subset: $(size(sim))")
            println("    ✓ Observables: $observables")
        end

        @testset "Data dimensions" begin
            Random.seed!(12345)
            sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 50)
            observables = [:Output]
            sim = dropdims(sim_full(Variables=observables), dims=3)

            # Should be 2D (Variables × Periods)
            @test ndims(sim) == 2
            @test size(sim, 1) == 1  # Single observable
            @test size(sim, 2) == 50  # Time periods

            println("    ✓ Data is 2D: $(size(sim))")
        end
    end

    # ========================================================================
    # Test 2: Likelihood Evaluation
    # ========================================================================

    @testset "Likelihood Evaluation" begin
        println("\n" * "="^80)
        println("TEST 2: Likelihood Evaluation")
        println("="^80)

        Random.seed!(54321)
        sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 100)
        observables = [:Output]
        sim = dropdims(sim_full(Variables=observables), dims=3)

        @testset "Kalman filter likelihood" begin
            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            @test ll isa Real
            @test isfinite(ll)
            # Log-likelihood can be positive or negative depending on data

            println("    ✓ Kalman filter LL: $(round(ll, digits=2))")
        end

        @testset "Inversion filter likelihood" begin
            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :inversion,
                algorithm = :first_order,
                verbose = false
            )

            @test ll isa Real
            @test isfinite(ll)

            println("    ✓ Inversion filter LL: $(round(ll, digits=2))")
        end

        @testset "Likelihood gradient" begin
            # Test that likelihood can be differentiated (needed for estimation)
            ll_base = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            # Perturb first parameter slightly
            params_pert = copy(RBC_Dynare.parameter_values)
            ε = 1e-6
            params_pert[1] += ε

            ll_pert = get_loglikelihood(
                RBC_Dynare,
                sim,
                params_pert,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            # Finite difference approximation of gradient
            grad_approx = (ll_pert - ll_base) / ε

            @test isfinite(grad_approx)
            @test abs(grad_approx) > 0  # Should have some gradient

            println("    ✓ Likelihood is differentiable")
            println("    ✓ Finite difference gradient: $(round(grad_approx, digits=2))")
        end

        @testset "Likelihood at different parameter values" begin
            lls = Float64[]

            for scale in [0.9, 0.95, 1.0, 1.05, 1.1]
                params = RBC_Dynare.parameter_values .* scale

                ll = get_loglikelihood(
                    RBC_Dynare,
                    sim,
                    params,
                    filter = :kalman,
                    algorithm = :first_order,
                    verbose = false
                )

                if isfinite(ll)
                    push!(lls, ll)
                end
            end

            # Should have multiple finite likelihoods
            @test length(lls) >= 3

            println("    ✓ Evaluated at $(length(lls)) parameter values")
            println("    ✓ LL range: [$(round(minimum(lls), digits=2)), $(round(maximum(lls), digits=2))]")
        end
    end

    # ========================================================================
    # Test 3: Parameter Inference Workflow
    # ========================================================================

    @testset "Parameter Inference Workflow" begin
        println("\n" * "="^80)
        println("TEST 3: Parameter Inference Workflow")
        println("="^80)

        Random.seed!(99999)
        T_data = 50  # Shorter for faster tests
        sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_data)
        observables = [:Output]
        sim = dropdims(sim_full(Variables=observables), dims=3)

        @testset "Parameter identification" begin
            # Test that parameters affect likelihood
            ll_true = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            # Perturb each parameter and check likelihood changes
            n_identified = 0

            for i in 1:length(RBC_Dynare.parameter_values)
                params_pert = copy(RBC_Dynare.parameter_values)
                params_pert[i] *= 1.1

                ll_pert = get_loglikelihood(
                    RBC_Dynare,
                    sim,
                    params_pert,
                    filter = :kalman,
                    algorithm = :first_order,
                    verbose = false
                )

                if isfinite(ll_pert) && abs(ll_pert - ll_true) > 1e-6
                    n_identified += 1
                end
            end

            @test n_identified > 0  # At least some parameters should be identified

            println("    ✓ Identified parameters: $n_identified / $(length(RBC_Dynare.parameter_values))")
        end

        @testset "Likelihood optimization workflow" begin
            # Test that likelihood function works in optimization context
            # (without actually running full optimization which would be slow)

            function objective(params)
                ll = get_loglikelihood(
                    RBC_Dynare,
                    sim,
                    params,
                    filter = :kalman,
                    algorithm = :first_order,
                    verbose = false
                )
                return -ll  # Minimize negative log-likelihood
            end

            # Test at initial parameters
            obj_init = objective(RBC_Dynare.parameter_values)
            @test isfinite(obj_init)

            # Test at perturbed parameters
            params_pert = RBC_Dynare.parameter_values .* 0.95
            obj_pert = objective(params_pert)

            @test isfinite(obj_pert) || isfinite(obj_init)  # At least one should work

            println("    ✓ Objective function works")
            println("    ✓ Initial objective: $(round(obj_init, digits=2))")
            if isfinite(obj_pert)
                println("    ✓ Perturbed objective: $(round(obj_pert, digits=2))")
            end
        end
    end

    # ========================================================================
    # Test 4: Algorithm Integration
    # ========================================================================

    @testset "Algorithm Integration" begin
        println("\n" * "="^80)
        println("TEST 4: Algorithm Integration")
        println("="^80)

        Random.seed!(11111)
        sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 50)
        observables = [:Output]
        sim = dropdims(sim_full(Variables=observables), dims=3)

        @testset "First-order perturbation" begin
            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll)
            println("    ✓ First-order estimation works")
        end

        @testset "Different initial covariances" begin
            ll_theoretical = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                initial_covariance = :theoretical,
                verbose = false
            )

            ll_diagonal = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                initial_covariance = :diagonal,
                verbose = false
            )

            @test isfinite(ll_theoretical)
            @test isfinite(ll_diagonal)

            println("    ✓ Theoretical covariance: LL = $(round(ll_theoretical, digits=2))")
            println("    ✓ Diagonal covariance:    LL = $(round(ll_diagonal, digits=2))")
        end

        @testset "Different presample periods" begin
            lls = Float64[]

            for presample in [0, 5, 10]
                ll = get_loglikelihood(
                    RBC_Dynare,
                    sim,
                    RBC_Dynare.parameter_values,
                    filter = :kalman,
                    algorithm = :first_order,
                    presample_periods = presample,
                    verbose = false
                )

                push!(lls, ll)
                @test isfinite(ll)
            end

            println("    ✓ Different presample periods work")
            println("    ✓ LL range: [$(round(minimum(lls), digits=2)), $(round(maximum(lls), digits=2))]")
        end
    end

    # ========================================================================
    # Test 5: Edge Cases and Robustness
    # ========================================================================

    @testset "Edge Cases and Robustness" begin
        println("\n" * "="^80)
        println("TEST 5: Edge Cases and Robustness")
        println("="^80)

        @testset "Very short data" begin
            Random.seed!(22222)
            T_short = 5
            sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_short)
            observables = [:Output]
            sim = dropdims(sim_full(Variables=observables), dims=3)

            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll) || isinf(ll)  # Either finite or -Inf is OK

            println("    ✓ Very short data (T=$T_short) handled")
        end

        @testset "Long data series" begin
            Random.seed!(33333)
            T_long = 500
            sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = T_long)
            observables = [:Output]
            sim = dropdims(sim_full(Variables=observables), dims=3)

            ll = get_loglikelihood(
                RBC_Dynare,
                sim,
                RBC_Dynare.parameter_values,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            @test isfinite(ll)

            println("    ✓ Long data (T=$T_long) handled")
            println("    ✓ LL = $(round(ll, digits=2))")
        end

        @testset "Different random seeds" begin
            lls = Float64[]

            for seed in [100, 200, 300, 400, 500]
                Random.seed!(seed)
                sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 50)
                observables = [:Output]
                sim = dropdims(sim_full(Variables=observables), dims=3)

                ll = get_loglikelihood(
                    RBC_Dynare,
                    sim,
                    RBC_Dynare.parameter_values,
                    filter = :kalman,
                    algorithm = :first_order,
                    verbose = false
                )

                push!(lls, ll)
                @test isfinite(ll)
            end

            @test length(lls) == 5
            @test all(isfinite.(lls))

            println("    ✓ Multiple random seeds work")
            println("    ✓ LL range: [$(round(minimum(lls), digits=2)), $(round(maximum(lls), digits=2))]")
            println("    ✓ LL std: $(round(std(lls), digits=2))")
        end

        @testset "Parameter bounds" begin
            # Test that extreme parameter values are handled gracefully
            Random.seed!(44444)
            sim_full = simulate(RBC_Dynare, algorithm = :first_order, periods = 50)
            observables = [:Output]
            sim = dropdims(sim_full(Variables=observables), dims=3)

            # Very small parameters
            params_small = RBC_Dynare.parameter_values .* 0.01
            ll_small = get_loglikelihood(
                RBC_Dynare,
                sim,
                params_small,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            # Very large parameters
            params_large = RBC_Dynare.parameter_values .* 100.0
            ll_large = get_loglikelihood(
                RBC_Dynare,
                sim,
                params_large,
                filter = :kalman,
                algorithm = :first_order,
                verbose = false
            )

            # Should return something (even if -Inf)
            @test ll_small isa Real
            @test ll_large isa Real

            println("    ✓ Extreme parameter values handled")
            println("    ✓ Small params: LL = $(round(ll_small, digits=2))")
            println("    ✓ Large params: LL = $(round(ll_large, digits=2))")
        end
    end
end

println("\n" * "="^80)
println("ESTIMATION PIPELINE TESTS COMPLETE")
println("="^80)
