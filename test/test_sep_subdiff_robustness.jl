using Test
using MacroModelling
using LinearAlgebra

@testset "Subdifferential Kink Detection" begin
    toy_level = (var = [:r], parameters = [:R_bar], parameter_values = [1.0])
    toy_log = (var = [:r], parameters = [:R_bar], parameter_values = [0.0])

    at_kink_level, _, _ = MacroModelling.detect_zlb_kink([1.0], toy_level; kink_tol = 1e-8)
    @test at_kink_level

    away_level, _, _ = MacroModelling.detect_zlb_kink([1.01], toy_level; kink_tol = 1e-8)
    @test !away_level

    # log(r) = R_bar form used in HLT OBC model: with R_bar = 0, r = 1 is the kink.
    at_kink_log, _, _ = MacroModelling.detect_zlb_kink([1.0], toy_log; kink_tol = 1e-8)
    @test at_kink_log

    away_log, _, _ = MacroModelling.detect_zlb_kink([1.01], toy_log; kink_tol = 1e-8)
    @test !away_log
end

@testset "HMC Covariance Guards" begin
    residual_func(ε) = [sum(abs2, ε)]

    # Zero covariance should not throw; it should trigger deterministic fallback.
    r_zero, diag_zero = MacroModelling.hmc_expectation(
        residual_func,
        zeros(2),
        zeros(2, 2),
        8;
        warmup = 0,
        leapfrog_steps = 2,
        step_size = 0.05,
        verbose = false
    )
    @test length(r_zero) == 1
    @test isfinite(r_zero[1])
    @test get(diag_zero, "mode", "") == "deterministic_fallback"
    @test get(diag_zero, "reason", "") == "zero_covariance"

    # Positive definite covariance should run standard HMC path and return finite moments.
    Σ = Matrix(Diagonal([0.25, 1.0]))
    r_pd, diag_pd = MacroModelling.hmc_expectation(
        residual_func,
        zeros(2),
        Σ,
        8;
        warmup = 2,
        leapfrog_steps = 2,
        step_size = 0.05,
        verbose = false
    )
    @test length(r_pd) == 1
    @test isfinite(r_pd[1])
    @test haskey(diag_pd, "n_samples")
end

@testset "HMC Order-1 Branching Structure" begin
    include(joinpath(@__DIR__, "..", "models", "RBC_Dynare.jl"))
    m = RBC_Dynare

    solve!(m;
        algorithm = :stochastic_extended_path,
        sep_periods = 2,
        sep_order = 1,
        sep_nnodes = 3,
        sep_sparse_tree = true,
        sep_expectation_method = :hmc,
        hmc_samples = 8,
        hmc_warmup = 2,
        hmc_leapfrog_steps = 2,
        hmc_step_size = 0.05,
        sep_maxit = 50,
        sep_tol = 1e-4,
        silent = true
    )

    sep = m.solution.perturbation.stochastic_extended_path
    @test sep !== nothing
    @test sep.layout.K > 1
    @test sep.layout.G[3] > 1
end
