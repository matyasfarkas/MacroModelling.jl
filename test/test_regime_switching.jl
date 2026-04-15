"""
Regime-Switching Gate Logic Tests

This file validates the regime-switching gate calibration and likelihood computation
for the hybrid ROM-SEP estimation framework.

Test Coverage:
1. Gate calibration (quantile-based thresholds)
2. Regime assignment (linear ROM vs nonlinear SEP)
3. Hard-gate likelihood computation
4. Empirical Bayes calibration
5. Threshold robustness

The regime-switching framework uses hard gates based on forecast errors and model residuals:
- Linear regime: Use fast ROM (perturbation) when forecast errors are small
- Nonlinear regime: Use accurate SEP (stochastic extended path) when errors are large

Created: January 2026
Status: Regime-switching estimation test suite
"""

using Test
using Printf
using LinearAlgebra
using Statistics
using Random

println("="^80)
println("REGIME-SWITCHING GATE LOGIC TESTS")
println("="^80)

# ========================================================================
# Helper Functions (defined at module level for access across testsets)
# ========================================================================

function calibrate_quantile(e::AbstractVector, f::AbstractVector, target_share::Float64;
                            tol::Float64=1e-4, maxiter::Int=50)
            @assert 0 < target_share < 1 "target_share must be in (0,1)"

            lo = 0.0
            hi = 1.0
            for iter in 1:maxiter
                q = (lo + hi) / 2
                tau_e = quantile(e, q)
                tau_f = quantile(f, q)
                share = mean((e .> tau_e) .| (f .> tau_f))

                if abs(share - target_share) < tol
                    return q, tau_e, tau_f, share
                end

                if share > target_share
                    lo = q
                else
                    hi = q
                end
            end

            q = (lo + hi) / 2
            tau_e = quantile(e, q)
            tau_f = quantile(f, q)
            share = mean((e .> tau_e) .| (f .> tau_f))

            return q, tau_e, tau_f, share
        end

function assign_regimes(e::AbstractVector, f::AbstractVector, tau_e::Float64, tau_f::Float64)
    # Returns true for nonlinear regime (SEP), false for linear regime (ROM)
    return (e .> tau_e) .| (f .> tau_f)
end

function compute_regime_likelihood(ll_rom::Vector{Float64},
                                   ll_sep::Vector{Float64},
                                   regimes::Union{BitVector,Vector{Bool}})
    # Hard gate: use ROM likelihood for linear periods, SEP for nonlinear
    ll_total = 0.0
    for t in 1:length(regimes)
        if regimes[t]
            ll_total += ll_sep[t]  # Nonlinear → SEP
        else
            ll_total += ll_rom[t]  # Linear → ROM
        end
    end
    return ll_total
end

function empirical_bayes_gate(e::Vector{Float64},
                               f::Vector{Float64},
                               ll_rom::Vector{Float64},
                               ll_sep::Vector{Float64};
                               target_shares=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    best_share = 0.0
    best_ll = -Inf
    best_regimes = nothing

    for target in target_shares
        q, tau_e, tau_f, achieved = calibrate_quantile(e, f, target)
        regimes = (e .> tau_e) .| (f .> tau_f)

        ll = 0.0
        for t in 1:length(regimes)
            ll += regimes[t] ? ll_sep[t] : ll_rom[t]
        end

        if ll > best_ll
            best_ll = ll
            best_share = achieved
            best_regimes = regimes
        end
    end

    return best_share, best_ll, best_regimes
end

@testset verbose = true "Regime-Switching Tests" begin

    # ========================================================================
    # Test 1: Gate Calibration - Quantile Method
    # ========================================================================

    @testset "Gate Calibration" begin
        println("\n" * "="^80)
        println("TEST 1: Gate Calibration (Quantile Method)")
        println("="^80)

        @testset "Basic calibration" begin
            Random.seed!(42)
            n = 1000

            # Generate synthetic forecast errors and residuals
            e = abs.(randn(n))  # Forecast errors
            f = abs.(randn(n))  # Model residuals

            target_share = 0.2  # Want 20% in nonlinear regime

            q, tau_e, tau_f, achieved_share = calibrate_quantile(e, f, target_share)

            @test 0 < q < 1
            @test tau_e > 0
            @test tau_f > 0
            @test abs(achieved_share - target_share) < 1e-2

            println("    ✓ Calibration converged")
            println("    ✓ Target share: $(target_share)")
            println("    ✓ Achieved share: $(round(achieved_share, digits=4))")
            println("    ✓ Quantile: $(round(q, digits=4))")
            println("    ✓ τ_ε: $(round(tau_e, digits=4)), τ_f: $(round(tau_f, digits=4))")
        end

        @testset "Different target shares" begin
            Random.seed!(123)
            n = 1000
            e = abs.(randn(n))
            f = abs.(randn(n))

            for target in [0.1, 0.2, 0.3, 0.5]
                q, tau_e, tau_f, achieved = calibrate_quantile(e, f, target)

                @test abs(achieved - target) < 5e-2  # Within 5%

                println("    ✓ Target $(target): achieved $(round(achieved, digits=4))")
            end
        end

        @testset "Edge cases" begin
            Random.seed!(456)
            n = 500
            e = abs.(randn(n))
            f = abs.(randn(n))

            # Very small share
            q_small, tau_e_small, tau_f_small, share_small = calibrate_quantile(e, f, 0.05)
            @test share_small < 0.1

            # Large share
            q_large, tau_e_large, tau_f_large, share_large = calibrate_quantile(e, f, 0.8)
            @test share_large > 0.7

            println("    ✓ Small share (5%): $(round(share_small, digits=4))")
            println("    ✓ Large share (80%): $(round(share_large, digits=4))")
        end

        @testset "Convergence properties" begin
            Random.seed!(789)
            n = 1000
            e = abs.(randn(n))
            f = abs.(randn(n))

            target = 0.25

            # Should converge quickly
            q, tau_e, tau_f, share = calibrate_quantile(e, f, target, maxiter=100)

            @test abs(share - target) < 2e-3  # Tight tolerance (allow 0.002 for numerical issues)

            println("    ✓ Converges with tight tolerance")
        end
    end

    # ========================================================================
    # Test 2: Regime Assignment
    # ========================================================================

    @testset "Regime Assignment" begin
        println("\n" * "="^80)
        println("TEST 2: Regime Assignment")
        println("="^80)

        @testset "Basic assignment" begin
            Random.seed!(42)
            n = 500

            e = abs.(randn(n))
            f = abs.(randn(n))

            tau_e = quantile(e, 0.8)
            tau_f = quantile(f, 0.8)

            regimes = assign_regimes(e, f, tau_e, tau_f)

            @test regimes isa BitVector
            @test length(regimes) == n

            n_nonlinear = sum(regimes)
            share_nonlinear = n_nonlinear / n

            @test 0.10 < share_nonlinear < 0.40  # Should be in reasonable range (~20% ± variation)

            println("    ✓ Regimes assigned")
            println("    ✓ Linear regime: $(n - n_nonlinear) periods")
            println("    ✓ Nonlinear regime: $(n_nonlinear) periods")
            println("    ✓ Share nonlinear: $(round(share_nonlinear, digits=4))")
        end

        @testset "Regime persistence" begin
            # In practice, nonlinear regimes (crises) tend to cluster
            Random.seed!(123)
            n = 1000

            # Create clustered errors (crisis periods)
            e = abs.(randn(n))
            crisis_periods = 100:150
            e[crisis_periods] .+= 2.0  # Large errors during crisis

            f = abs.(randn(n))
            f[crisis_periods] .+= 1.5

            tau_e = quantile(e, 0.95)
            tau_f = quantile(f, 0.95)

            regimes = assign_regimes(e, f, tau_e, tau_f)

            # Most crisis periods should be in nonlinear regime
            crisis_nonlinear = sum(regimes[crisis_periods])
            crisis_share = crisis_nonlinear / length(crisis_periods)

            @test crisis_share > 0.5  # Most crisis periods nonlinear

            println("    ✓ Crisis periods detected")
            println("    ✓ Crisis share nonlinear: $(round(crisis_share, digits=4))")
        end

        @testset "Threshold sensitivity" begin
            Random.seed!(456)
            n = 500
            e = abs.(randn(n))
            f = abs.(randn(n))

            # Different thresholds → different regime assignments
            tau_e_loose = quantile(e, 0.9)
            tau_f_loose = quantile(f, 0.9)

            tau_e_tight = quantile(e, 0.5)
            tau_f_tight = quantile(f, 0.5)

            regimes_loose = assign_regimes(e, f, tau_e_loose, tau_f_loose)
            regimes_tight = assign_regimes(e, f, tau_e_tight, tau_f_tight)

            share_loose = mean(regimes_loose)
            share_tight = mean(regimes_tight)

            @test share_tight > share_loose  # Tighter → more nonlinear

            println("    ✓ Loose thresholds: $(round(share_loose, digits=4)) nonlinear")
            println("    ✓ Tight thresholds: $(round(share_tight, digits=4)) nonlinear")
        end
    end

    # ========================================================================
    # Test 3: Hard-Gate Likelihood
    # ========================================================================

    @testset "Hard-Gate Likelihood" begin
        println("\n" * "="^80)
        println("TEST 3: Hard-Gate Likelihood")
        println("="^80)

        @testset "Basic likelihood computation" begin
            n = 100

            # Simulate likelihoods
            Random.seed!(42)
            ll_rom = -10.0 .+ randn(n)  # ROM likelihoods
            ll_sep = -15.0 .+ 2.0 * randn(n)  # SEP likelihoods (more variable, lower on average)

            # Random regime assignment
            regimes = rand(Bool, n)

            ll_total = compute_regime_likelihood(ll_rom, ll_sep, regimes)

            @test isfinite(ll_total)
            @test ll_total < 0  # Log-likelihood should be negative

            println("    ✓ Likelihood computed: $(round(ll_total, digits=2))")
            println("    ✓ ROM periods: $(sum(.!regimes))")
            println("    ✓ SEP periods: $(sum(regimes))")
        end

        @testset "All linear vs all nonlinear" begin
            n = 100
            Random.seed!(123)

            ll_rom = -10.0 .+ randn(n)
            ll_sep = -15.0 .+ 2.0 * randn(n)

            # All linear (ROM)
            regimes_linear = falses(n)
            ll_all_rom = compute_regime_likelihood(ll_rom, ll_sep, regimes_linear)
            @test ll_all_rom ≈ sum(ll_rom)

            # All nonlinear (SEP)
            regimes_nonlinear = trues(n)
            ll_all_sep = compute_regime_likelihood(ll_rom, ll_sep, regimes_nonlinear)
            @test ll_all_sep ≈ sum(ll_sep)

            println("    ✓ All ROM: $(round(ll_all_rom, digits=2))")
            println("    ✓ All SEP: $(round(ll_all_sep, digits=2))")
        end

        @testset "Likelihood sensitivity to regime assignment" begin
            n = 100
            Random.seed!(456)

            # ROM is better on average (faster, good for linear periods)
            ll_rom = -8.0 .+ 0.5 * randn(n)
            # SEP is worse on average but captures nonlinearities
            ll_sep = -12.0 .+ randn(n)

            # Few nonlinear periods (optimal for ROM)
            regimes_few = rand(n) .< 0.1
            ll_few_nl = compute_regime_likelihood(ll_rom, ll_sep, regimes_few)

            # Many nonlinear periods
            regimes_many = rand(n) .< 0.5
            ll_many_nl = compute_regime_likelihood(ll_rom, ll_sep, regimes_many)

            # Fewer nonlinear → better likelihood (since ROM is better on avg)
            @test ll_few_nl > ll_many_nl

            println("    ✓ 10% nonlinear: $(round(ll_few_nl, digits=2))")
            println("    ✓ 50% nonlinear: $(round(ll_many_nl, digits=2))")
        end
    end

    # ========================================================================
    # Test 4: Empirical Bayes Calibration
    # ========================================================================

    @testset "Empirical Bayes Calibration" begin
        println("\n" * "="^80)
        println("TEST 4: Empirical Bayes Calibration")
        println("="^80)

        @testset "Basic EB calibration" begin
            Random.seed!(789)
            n = 200

            e = abs.(randn(n))
            f = abs.(randn(n))

            # ROM better for small errors, SEP better for large errors
            ll_rom = zeros(n)
            ll_sep = zeros(n)

            for t in 1:n
                if e[t] < 1.0 && f[t] < 1.0
                    ll_rom[t] = -5.0 + 0.5 * randn()
                    ll_sep[t] = -8.0 + randn()
                else
                    ll_rom[t] = -15.0 + 2.0 * randn()
                    ll_sep[t] = -10.0 + randn()
                end
            end

            best_share, best_ll, best_regimes = empirical_bayes_gate(e, f, ll_rom, ll_sep)

            @test 0 < best_share < 1
            @test isfinite(best_ll)
            @test best_regimes isa BitVector

            println("    ✓ EB calibration completed")
            println("    ✓ Optimal share: $(round(best_share, digits=4))")
            println("    ✓ Best likelihood: $(round(best_ll, digits=2))")
        end

        @testset "EB vs fixed share" begin
            Random.seed!(101112)
            n = 200

            e = abs.(randn(n))
            f = abs.(randn(n))

            # ROM better overall
            ll_rom = -8.0 .+ randn(n)
            ll_sep = -12.0 .+ 2.0 * randn(n)

            # EB should choose low share (mostly ROM)
            best_share, best_ll, best_regimes = empirical_bayes_gate(e, f, ll_rom, ll_sep)

            # Fixed 50% share
            q_50, tau_e_50, tau_f_50, _ = calibrate_quantile(e, f, 0.5)
            regimes_50 = (e .> tau_e_50) .| (f .> tau_f_50)
            ll_50 = sum(ifelse.(regimes_50, ll_sep, ll_rom))

            @test best_ll > ll_50  # EB should be better

            println("    ✓ EB likelihood: $(round(best_ll, digits=2))")
            println("    ✓ Fixed 50% likelihood: $(round(ll_50, digits=2))")
            println("    ✓ EB improvement: $(round(best_ll - ll_50, digits=2))")
        end
    end

    # ========================================================================
    # Test 5: Threshold Robustness
    # ========================================================================

    @testset "Threshold Robustness" begin
        println("\n" * "="^80)
        println("TEST 5: Threshold Robustness")
        println("="^80)

        @testset "Stability with noise" begin
            Random.seed!(131415)
            n = 500

            # Base errors
            e_base = abs.(randn(n))
            f_base = abs.(randn(n))

            target = 0.2

            # Calibrate on base
            q_base, tau_e_base, tau_f_base, share_base = calibrate_quantile(e_base, f_base, target)

            # Add small noise
            e_noise = e_base .+ 0.1 * randn(n)
            f_noise = f_base .+ 0.1 * randn(n)

            q_noise, tau_e_noise, tau_f_noise, share_noise = calibrate_quantile(e_noise, f_noise, target)

            # Thresholds should be similar
            @test abs(tau_e_base - tau_e_noise) / tau_e_base < 0.2  # Within 20%
            @test abs(tau_f_base - tau_f_noise) / tau_f_base < 0.2

            println("    ✓ Thresholds stable under noise")
            println("    ✓ Base τ_ε: $(round(tau_e_base, digits=4))")
            println("    ✓ Noise τ_ε: $(round(tau_e_noise, digits=4))")
        end

        @testset "Consistency across samples" begin
            Random.seed!(161718)

            target = 0.15

            # Multiple independent samples
            thresholds_e = Float64[]
            thresholds_f = Float64[]

            for i in 1:5
                e = abs.(randn(500))
                f = abs.(randn(500))

                q, tau_e, tau_f, share = calibrate_quantile(e, f, target)

                push!(thresholds_e, tau_e)
                push!(thresholds_f, tau_f)
            end

            # Thresholds should have moderate variation
            std_e = std(thresholds_e)
            std_f = std(thresholds_f)

            @test std_e < 0.5  # Reasonably stable
            @test std_f < 0.5

            println("    ✓ Threshold std across samples:")
            println("      τ_ε: $(round(std_e, digits=4))")
            println("      τ_f: $(round(std_f, digits=4))")
        end
    end
end

println("\n" * "="^80)
println("REGIME-SWITCHING TESTS COMPLETE")
println("="^80)
