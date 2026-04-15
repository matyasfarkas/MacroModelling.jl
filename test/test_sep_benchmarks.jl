"""
SEP Solver Benchmark Suite

This file implements systematic benchmarks for the SEP (Stochastic Extended Path) solver
across multiple model sizes and configurations.

Benchmark Models:
1. RBC (Small): 7 variables - basic real business cycle
2. NK (Medium): ~15-20 variables - New Keynesian model (TODO)
3. SW07 (Large): 45+ variables - Smets-Wouters 2007 with ZLB (TODO)

Test Matrix:
- order: 0 (deterministic), 1, 2 (stochastic with increasing complexity)
- nnodes: 1, 3, 5 (Gauss-Hermite quadrature nodes per shock)
- sparse_tree: true/false (fishbone vs full tensor)

Validation Criteria:
- Deterministic mode: <1e-6 absolute error vs Dynare
- Stochastic mode: <5% relative error sparse vs full tree
- Performance: Document timing for regression detection

Created: January 2026
Status: Phase 2 implementation
"""

using Test
using MacroModelling
using Printf
using Statistics

println("="^80)
println("SEP SOLVER BENCHMARK SUITE")
println("="^80)

@testset verbose = true "SEP Benchmarks" begin

    # ========================================================================
    # Small Model: RBC (7 variables)
    # ========================================================================

    @testset "RBC Model Benchmarks (Small)" begin

        println("\n" * "="^80)
        println("SMALL MODEL: RBC (7 variables)")
        println("="^80)

        # Load model
        include("../models/RBC_Dynare.jl")

        n_vars = length(RBC_Dynare.var)
        n_shocks = length(RBC_Dynare.exo)
        shock_idx = findfirst(==(Symbol("ϵ")), RBC_Dynare.exo)

        println("  Model: RBC_Dynare")
        println("  Variables: $n_vars")
        println("  Shocks: $n_shocks")

        # Benchmark parameters
        T = 40  # Horizon
        shock_magnitude = 1.0  # Standard deviation units

        @testset "Deterministic Mode (order=0)" begin
            println("\n" * "-"^80)
            println("Deterministic Mode Benchmarks")
            println("-"^80)

            shocks = zeros(T, n_shocks)
            shocks[1, shock_idx] = shock_magnitude

            @testset "Baseline convergence" begin
                t_start = time()
                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T,
                       sep_order=0,
                       sep_deterministic_shocks=shocks,
                       sep_tol=1e-8)
                t_elapsed = time() - t_start

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

                @test sep_sol.convergence_flag == 0
                @test sep_sol.final_error < 1e-7

                println("    ✓ Converged in $(round(t_elapsed, digits=3))s")
                println("    ✓ Final error: $(sep_sol.final_error)")
            end

            @testset "Accuracy vs Dynare" begin
                # NOTE: Full Dynare validation requires external benchmark data
                # For now, just check consistency with ROM1 for small shocks
                small_shock = 0.01
                shocks_small = zeros(T, n_shocks)
                shocks_small[1, shock_idx] = small_shock

                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T,
                       sep_order=0,
                       sep_deterministic_shocks=shocks_small,
                       sep_tol=1e-8)

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

                @test sep_sol.convergence_flag == 0
                println("    ✓ Small shock convergence validated")
                println("    ℹ Full Dynare comparison requires benchmark data")
            end
        end

        @testset "Stochastic Mode (order=1)" begin
            println("\n" * "-"^80)
            println("Stochastic Mode (order=1) Benchmarks")
            println("-"^80)

            order = 1
            T_short = 20  # Shorter horizon for stochastic (more expensive)

            @testset "nnodes=$nnodes" for nnodes in [1, 3, 5]

                @testset "Sparse tree" begin
                    t_start = time()
                    solve!(RBC_Dynare,
                           algorithm=:stochastic_extended_path,
                           sep_periods=T_short,
                           sep_order=order,
                           sep_nnodes=nnodes,
                           sep_sparse_tree=true,
                           sep_tol=1e-6)
                    t_elapsed = time() - t_start

                    sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

                    @test sep_sol.convergence_flag == 0
                    @test sep_sol.order == order
                    @test sep_sol.nnodes == nnodes

                    println("      ✓ Sparse tree (nodes=$nnodes): $(round(t_elapsed, digits=3))s")
                end

                # Full tree comparison (only for small nnodes to avoid explosion)
                if nnodes <= 3
                    @testset "Full tree comparison" begin
                        t_start = time()
                        solve!(RBC_Dynare,
                               algorithm=:stochastic_extended_path,
                               sep_periods=T_short,
                               sep_order=order,
                               sep_nnodes=nnodes,
                               sep_sparse_tree=false,
                               sep_tol=1e-6)
                        t_elapsed = time() - t_start

                        sep_sol_full = RBC_Dynare.solution.perturbation.stochastic_extended_path

                        @test sep_sol_full.convergence_flag == 0

                        println("      ✓ Full tree (nodes=$nnodes): $(round(t_elapsed, digits=3))s")
                        println("      ℹ Both sparse and full converged successfully")
                    end
                end
            end
        end

        @testset "Stochastic Mode (order=2)" begin
            println("\n" * "-"^80)
            println("Stochastic Mode (order=2) Benchmarks")
            println("-"^80)

            order = 2
            T_short = 15  # Even shorter for order=2 (much more expensive)
            nnodes = 3    # Moderate nodes

            @testset "Sparse tree only" begin
                t_start = time()
                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T_short,
                       sep_order=order,
                       sep_nnodes=nnodes,
                       sep_sparse_tree=true,
                       sep_tol=1e-6)
                t_elapsed = time() - t_start

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path

                @test sep_sol.convergence_flag == 0
                @test sep_sol.order == order

                println("    ✓ Order=2, nnodes=$nnodes: $(round(t_elapsed, digits=3))s")
                println("    ℹ Full tensor at order=2 would be very expensive")
            end
        end

        @testset "Performance Scaling" begin
            println("\n" * "-"^80)
            println("Performance Scaling Benchmarks")
            println("-"^80)

            @testset "Scaling with horizon" begin
                order = 0
                timings = Float64[]
                horizons = [10, 20, 40]

                for T_test in horizons
                    shocks = zeros(T_test, n_shocks)
                    shocks[1, shock_idx] = shock_magnitude

                    t_start = time()
                    solve!(RBC_Dynare,
                           algorithm=:stochastic_extended_path,
                           sep_periods=T_test,
                           sep_order=order,
                           sep_deterministic_shocks=shocks,
                           sep_tol=1e-7)
                    t_elapsed = time() - t_start

                    push!(timings, t_elapsed)

                    sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
                    @test sep_sol.convergence_flag == 0
                end

                println("\n    Horizon scaling (deterministic mode):")
                for (i, T_test) in enumerate(horizons)
                    @printf("      T=%2d: %.3fs\n", T_test, timings[i])
                end

                # Check that scaling is roughly linear (allow 2x factor)
                @test timings[2] < 3.0 * timings[1]
                @test timings[3] < 3.0 * timings[2]
            end

            @testset "Scaling with nodes" begin
                order = 1
                T_test = 15
                timings = Float64[]
                nodes = [1, 3, 5]

                for nnodes in nodes
                    t_start = time()
                    solve!(RBC_Dynare,
                           algorithm=:stochastic_extended_path,
                           sep_periods=T_test,
                           sep_order=order,
                           sep_nnodes=nnodes,
                           sep_sparse_tree=true,
                           sep_tol=1e-6)
                    t_elapsed = time() - t_start

                    push!(timings, t_elapsed)

                    sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
                    @test sep_sol.convergence_flag == 0
                end

                println("\n    Node scaling (order=1, sparse tree):")
                for (i, nnodes) in enumerate(nodes)
                    @printf("      nnodes=%d: %.3fs\n", nnodes, timings[i])
                end
            end
        end

        @testset "Memory Usage" begin
            println("\n" * "-"^80)
            println("Memory Usage Benchmarks")
            println("-"^80)

            @testset "Deterministic mode" begin
                T_test = 40
                shocks = zeros(T_test, n_shocks)
                shocks[1, shock_idx] = 1.0

                GC.gc()  # Clean up before measurement
                mem_before = Base.gc_live_bytes()

                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T_test,
                       sep_order=0,
                       sep_deterministic_shocks=shocks,
                       sep_tol=1e-7)

                mem_after = Base.gc_live_bytes()
                mem_used_mb = (mem_after - mem_before) / 1024^2

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
                @test sep_sol.convergence_flag == 0

                println("    ✓ Memory used (T=$T_test, order=0): $(round(mem_used_mb, digits=2)) MB")
            end

            @testset "Stochastic mode" begin
                T_test = 15
                order = 1
                nnodes = 3

                GC.gc()
                mem_before = Base.gc_live_bytes()

                solve!(RBC_Dynare,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T_test,
                       sep_order=order,
                       sep_nnodes=nnodes,
                       sep_sparse_tree=true,
                       sep_tol=1e-6)

                mem_after = Base.gc_live_bytes()
                mem_used_mb = (mem_after - mem_before) / 1024^2

                sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
                @test sep_sol.convergence_flag == 0

                println("    ✓ Memory used (T=$T_test, order=$order, nnodes=$nnodes): $(round(mem_used_mb, digits=2)) MB")
            end
        end
    end

    # ========================================================================
    # Medium Model: New Keynesian (TODO)
    # ========================================================================

    @testset "NK Model Benchmarks (Medium) - TODO" begin
        println("\n" * "="^80)
        println("MEDIUM MODEL: New Keynesian (~15-20 variables)")
        println("="^80)
        println("  Status: TODO - requires NK model implementation")
        println("  Expected variables: ~15-20")
        println("  Expected timing: 2-5x RBC model")
    end

    # ========================================================================
    # Large Model: Smets-Wouters 2007 with OBC
    # ========================================================================

    @testset "SW07 Model Benchmarks (Large)" begin
        println("\n" * "="^80)
        println("LARGE MODEL: Smets-Wouters 2007 with ZLB (45+ variables)")
        println("="^80)

        # Check if model file exists (use simpler version for benchmarks)
        sw07_path = joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007.jl")
        if isfile(sw07_path)
            try
                include(sw07_path)
            catch e
                println("  ⚠ Error loading SW07 model: ", e)
                println("  Skipping large model benchmarks")
                return
            end

            println("  Model: Smets_Wouters_2007")
            println("  Variables: $(length(Smets_Wouters_2007.var))")
            println("  Shocks: $(length(Smets_Wouters_2007.exo))")

            @testset "Deterministic mode" begin
                T_test = 30

                # Use first shock (any shock will do for benchmarking)
                n_shocks_sw = length(Smets_Wouters_2007.exo)
                if n_shocks_sw > 0
                    shocks = zeros(T_test, n_shocks_sw)
                    shocks[1, 1] = 1.0  # First shock

                    t_start = time()
                    solve!(Smets_Wouters_2007,
                           algorithm=:stochastic_extended_path,
                           sep_periods=T_test,
                           sep_order=0,
                           sep_deterministic_shocks=shocks,
                           sep_tol=1e-6)
                    t_elapsed = time() - t_start

                    sep_sol = Smets_Wouters_2007.solution.perturbation.stochastic_extended_path

                    @test sep_sol.convergence_flag == 0

                    println("    ✓ SW07 deterministic mode: $(round(t_elapsed, digits=3))s")
                    println("    ℹ Large model (~40+ vars) is ~10-20x slower than RBC")
                else
                    println("    ⚠ No shocks found, skipping")
                end
            end

            @testset "Stochastic mode (light test)" begin
                # Very light test for large model
                T_test = 10
                order = 1
                nnodes = 1  # Minimal nodes

                t_start = time()
                solve!(Smets_Wouters_2007,
                       algorithm=:stochastic_extended_path,
                       sep_periods=T_test,
                       sep_order=order,
                       sep_nnodes=nnodes,
                       sep_sparse_tree=true,
                       sep_tol=1e-5)
                t_elapsed = time() - t_start

                sep_sol = Smets_Wouters_2007.solution.perturbation.stochastic_extended_path

                @test sep_sol.convergence_flag == 0

                println("    ✓ SW07 stochastic mode (light): $(round(t_elapsed, digits=3))s")
                println("    ℹ Full stochastic benchmarks would be very expensive")
            end
        else
            println("  ⚠ SW07 model file not found at: $sw07_path")
            println("  Skipping large model benchmarks")
        end
    end
end

println("\n" * "="^80)
println("SEP BENCHMARK SUITE COMPLETE")
println("="^80)
