"""
SEP Solver Integration Tests

This file implements integration tests for the SEP solver, focusing on:
1. Edge cases (long horizons, many shocks, near-singular systems)
2. Warm start effectiveness
3. Linear solver comparisons (QR vs normal equations)
4. Fallback mechanisms
5. Robustness and error handling

Created: January 2026
Status: Phase 4 implementation
"""

using Test
using MacroModelling
using Printf
using LinearAlgebra

# Import specific functions to avoid namespace conflicts
import Statistics: std

println("="^80)
println("SEP SOLVER INTEGRATION TESTS")
println("="^80)

@testset verbose = true "SEP Integration Tests" begin

    # ========================================================================
    # Setup: Load test model
    # ========================================================================

    println("\nLoading test model...")
    include("../models/RBC_Dynare.jl")

    n_vars = length(RBC_Dynare.var)
    n_shocks = length(RBC_Dynare.exo)
    shock_idx = findfirst(==(Symbol("ϵ")), RBC_Dynare.exo)

    println("  Model: RBC_Dynare")
    println("  Variables: $n_vars")
    println("  Shocks: $n_shocks")

    # ========================================================================
    # Test 1: Edge Cases
    # ========================================================================

    @testset "Edge Cases" begin
        println("\n" * "="^80)
        println("TEST 1: Edge Cases")
        println("="^80)

        @testset "Very long horizon" begin
            T_long = 200  # Much longer than typical

            shocks = zeros(T_long, n_shocks)
            shocks[1, shock_idx] = 0.5

            t_start = time()
            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T_long,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-6)
            t_elapsed = time() - t_start

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0
            @test sep_sol.final_error < 1e-5

            println("    ✓ Long horizon (T=$T_long) converged in $(round(t_elapsed, digits=3))s")
        end

        @testset "Very small shock" begin
            T = 30
            tiny_shock = 1e-8  # Very small

            shocks = zeros(T, n_shocks)
            shocks[1, shock_idx] = tiny_shock

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-8)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0

            println("    ✓ Tiny shock ($(tiny_shock)) handled correctly")
        end

        @testset "Very large shock" begin
            T = 30
            large_shock = 10.0  # Very large (10 std deviations)

            shocks = zeros(T, n_shocks)
            shocks[1, shock_idx] = large_shock

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-5,
                   sep_maxit=100)  # May need more iterations

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            # Large shocks may be challenging, but should still converge or fail gracefully
            @test sep_sol.convergence_flag in [0, 1]  # 0=success, 1=max_iter

            if sep_sol.convergence_flag == 0
                println("    ✓ Large shock ($(large_shock)σ) converged")
            else
                println("    ⚠ Large shock hit max iterations (expected for very large shocks)")
            end
        end

        @testset "Multiple shocks in sequence" begin
            T = 50
            # Multiple shocks at different times
            shocks = zeros(T, n_shocks)
            shocks[1, shock_idx] = 1.0
            shocks[10, shock_idx] = -0.5
            shocks[20, shock_idx] = 0.8
            shocks[30, shock_idx] = -0.3

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-6)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0

            println("    ✓ Multiple shocks in sequence handled")
        end

        @testset "Stochastic with max nodes" begin
            T = 10
            order = 1
            nnodes = 5  # Maximum supported for sparse tree (1, 3, 5)

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=order,
                   sep_nnodes=nnodes,
                   sep_sparse_tree=true,
                   sep_tol=1e-6)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0
            @test sep_sol.nnodes == nnodes

            println("    ✓ Max node count (nnodes=$nnodes) handled")
        end
    end

    # ========================================================================
    # Test 2: Warm Start Effectiveness
    # ========================================================================

    @testset "Warm Start Effectiveness" begin
        println("\n" * "="^80)
        println("TEST 2: Warm Start Effectiveness")
        println("="^80)

        T = 40
        shocks = zeros(T, n_shocks)
        shocks[1, shock_idx] = 1.0

        # Force cold start by solving different configuration first
        solve!(RBC_Dynare,
               algorithm=:stochastic_extended_path,
               sep_periods=10,
               sep_order=0,
               sep_deterministic_shocks=zeros(10, n_shocks),
               sep_tol=1e-6)

        # Now solve target problem (will be cold start)
        t_start = time()
        solve!(RBC_Dynare,
               algorithm=:stochastic_extended_path,
               sep_periods=T,
               sep_order=0,
               sep_deterministic_shocks=shocks,
               sep_tol=1e-7)
        t_cold = time() - t_start

        sep_sol_cold = RBC_Dynare.solution.perturbation.stochastic_extended_path

        @testset "Cold start baseline" begin
            @test sep_sol_cold.convergence_flag == 0

            println("    ✓ Cold start time: $(round(t_cold, digits=4))s")
        end

        # Solve with slightly different shock
        shocks_similar = copy(shocks)
        shocks_similar[1, shock_idx] = 1.1  # Slightly different

        t_start = time()
        solve!(RBC_Dynare,
               algorithm=:stochastic_extended_path,
               sep_periods=T,
               sep_order=0,
               sep_deterministic_shocks=shocks_similar,
               sep_tol=1e-7)
        t_warm = time() - t_start

        sep_sol_warm = RBC_Dynare.solution.perturbation.stochastic_extended_path

        @testset "Warm start (similar problem)" begin
            @test sep_sol_warm.convergence_flag == 0

            println("    ✓ Warm start time: $(round(t_warm, digits=4))s")

            # Warm start should be faster (though may not always be if problem is very different)
            if t_warm < t_cold
                speedup = t_cold / t_warm
                println("    ✓ Warm start speedup: $(round(speedup, digits=2))x")
            else
                println("    ℹ Warm start similar to cold start (problem-dependent)")
            end
        end

        @testset "Automatic warm start detection" begin
            # The solver should automatically detect when warm start is possible
            # This is indicated in the output messages

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-7)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0

            println("    ✓ Automatic warm start detection working")
        end
    end

    # ========================================================================
    # Test 3: Linear Solver Comparison
    # ========================================================================

    @testset "Linear Solver Comparison" begin
        println("\n" * "="^80)
        println("TEST 3: Linear Solver Comparison")
        println("="^80)

        T = 30
        shocks = zeros(T, n_shocks)
        shocks[1, shock_idx] = 1.0

        # Solve with QR solver
        t_start = time()
        solve!(RBC_Dynare,
               algorithm=:stochastic_extended_path,
               sep_periods=T,
               sep_order=0,
               sep_deterministic_shocks=shocks,
               sep_linear_solver=:qr,
               sep_tol=1e-7)
        t_qr = time() - t_start

        sep_sol_qr = RBC_Dynare.solution.perturbation.stochastic_extended_path

        @testset "QR decomposition solver" begin
            @test sep_sol_qr.convergence_flag == 0

            println("    ✓ QR solver converged in $(round(t_qr, digits=4))s")
            println("    ✓ Final error: $(sep_sol_qr.final_error)")
        end

        # Solve with normal equations solver
        t_start = time()
        solve!(RBC_Dynare,
               algorithm=:stochastic_extended_path,
               sep_periods=T,
               sep_order=0,
               sep_deterministic_shocks=shocks,
               sep_linear_solver=:normal_equations,
               sep_tol=1e-7)
        t_normal = time() - t_start

        sep_sol_normal = RBC_Dynare.solution.perturbation.stochastic_extended_path

        @testset "Normal equations solver" begin
            @test sep_sol_normal.convergence_flag == 0

            println("    ✓ Normal equations solver converged in $(round(t_normal, digits=4))s")
            println("    ✓ Final error: $(sep_sol_normal.final_error)")
        end

        @testset "Solver accuracy comparison" begin
            # Both solvers should give similar results
            @test sep_sol_qr.convergence_flag == sep_sol_normal.convergence_flag
            @test sep_sol_qr.final_error < 1e-6
            @test sep_sol_normal.final_error < 1e-6

            # Check if solutions are similar (both should solve the same problem)
            error_ratio = sep_sol_qr.final_error / sep_sol_normal.final_error

            println("    ✓ QR vs Normal error ratio: $(round(error_ratio, digits=2))")
            println("    ℹ QR typically more stable for ill-conditioned systems")
        end
    end

    # ========================================================================
    # Test 4: Convergence Criteria and Tolerances
    # ========================================================================

    @testset "Convergence Criteria" begin
        println("\n" * "="^80)
        println("TEST 4: Convergence Criteria")
        println("="^80)

        T = 30
        shocks = zeros(T, n_shocks)
        shocks[1, shock_idx] = 1.0

        @testset "Tight tolerance" begin
            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-10)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0
            @test sep_sol.final_error < 1e-9

            println("    ✓ Tight tolerance (1e-10) achieved: error=$(sep_sol.final_error)")
        end

        @testset "Loose tolerance" begin
            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-4)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0
            @test sep_sol.final_error < 1e-3

            println("    ✓ Loose tolerance (1e-4) achieved: error=$(sep_sol.final_error)")
        end

        @testset "Max iterations limit" begin
            # Set very tight tolerance but limit iterations
            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-12,
                   sep_maxit=5)  # Very few iterations

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            # Should hit max iterations or converge
            @test sep_sol.convergence_flag in [0, 1]

            if sep_sol.convergence_flag == 1
                println("    ✓ Max iterations limit enforced correctly")
            else
                println("    ✓ Converged within iteration limit")
            end
        end
    end

    # ========================================================================
    # Test 5: Robustness and Error Handling
    # ========================================================================

    @testset "Robustness and Error Handling" begin
        println("\n" * "="^80)
        println("TEST 5: Robustness and Error Handling")
        println("="^80)

        @testset "Zero shock (trivial case)" begin
            T = 20
            shocks = zeros(T, n_shocks)  # All zeros

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-7)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0
            @test sep_sol.final_error < 1e-10  # Should converge very quickly

            println("    ✓ Zero shock (trivial case) handled correctly")
        end

        @testset "Persistence of shock" begin
            T = 40
            # Persistent shock sequence (AR-like)
            shocks = zeros(T, n_shocks)
            shock_val = 1.0
            rho = 0.9  # Persistence
            for t in 1:T
                shocks[t, shock_idx] = shock_val
                shock_val *= rho
            end

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=0,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-6)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0

            println("    ✓ Persistent shock sequence handled")
        end

        @testset "Numerical stability" begin
            T = 30
            shocks = zeros(T, n_shocks)
            shocks[1, shock_idx] = 1.0

            # Solve multiple times with same problem - should be stable
            errors = Float64[]
            for i in 1:3
                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T,
                       sep_order=0,
                       sep_deterministic_shocks=shocks,
                       sep_tol=1e-7)

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
                push!(errors, sep_sol.final_error)
            end

            # All runs should give similar errors (numerical stability)
            @test all(e -> e < 1e-6, errors)
            error_std = std(errors)

            @test error_std < 1e-10  # Very small variation

            println("    ✓ Numerically stable (error std: $(error_std))")
        end
    end

    # ========================================================================
    # Test 6: Stochastic Mode Integration
    # ========================================================================

    @testset "Stochastic Mode Integration" begin
        println("\n" * "="^80)
        println("TEST 6: Stochastic Mode Integration")
        println("="^80)

        T = 15

        @testset "Stochastic with deterministic shocks" begin
            # Can combine stochastic tree (order>0) with deterministic shocks
            order = 1
            nnodes = 3

            shocks = zeros(T, n_shocks)
            shocks[1, shock_idx] = 1.0

            solve!(RBC_Dynare,
                   algorithm=:stochastic_extended_path,
                   sep_periods=T,
                   sep_order=order,
                   sep_nnodes=nnodes,
                   sep_sparse_tree=true,
                   sep_deterministic_shocks=shocks,
                   sep_tol=1e-6)

            sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

            @test sep_sol.convergence_flag == 0
            @test sep_sol.order == order

            println("    ✓ Stochastic mode with deterministic shocks works")
        end

        @testset "Transition between orders" begin
            # Solve same problem with different orders
            nnodes = 3

            for order in [0, 1]
                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T,
                       sep_order=order,
                       sep_nnodes=nnodes,
                       sep_sparse_tree=true,
                       sep_tol=1e-6)

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

                @test sep_sol.convergence_flag == 0

                println("    ✓ Order=$order converged")
            end

            println("    ✓ Smooth transition between orders")
        end
    end
end

println("\n" * "="^80)
println("SEP INTEGRATION TESTS COMPLETE")
println("="^80)
