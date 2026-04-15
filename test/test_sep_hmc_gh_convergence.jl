using Test
using MacroModelling
using Random
using Statistics

@testset "SEP HMC vs GH Convergence (RBC small model)" begin
    include(joinpath(@__DIR__, "..", "models", "RBC_Dynare.jl"))
    m = RBC_Dynare

    periods = 6
    shocks = zeros(length(m.exo), periods)
    shocks[1, 1] = 0.05
    shocks[1, 2] = -0.03
    shocks[1, 3] = 0.02

    gh = simulate_sep_extended_path(
        m;
        periods = periods,
        shocks = shocks,
        sep_horizon = 8,
        sep_order = 1,
        sep_nnodes = 3,
        sep_sparse_tree = true,
        sep_tol = 1e-3,
        sep_maxit = 160,
        sep_expectation_method = :gauss_hermite,
        sep_line_search = true,
        sep_linear_solver = :qr,
        silent = true
    )

    @test !gh.errorflag
    sim_gh = Array(gh.simulation)

    metrics = Dict{Int, Tuple{Float64, Float64}}()
    for hs in (64, 256)
        Random.seed!(1234)
        hmc = simulate_sep_extended_path(
            m;
            periods = periods,
            shocks = shocks,
            sep_horizon = 8,
            sep_order = 1,
            sep_nnodes = 3,
            sep_sparse_tree = true,
            sep_tol = 1e-3,
            sep_maxit = 200,
            sep_expectation_method = :hmc,
            hmc_samples = hs,
            hmc_warmup = hs ÷ 2,
            hmc_leapfrog_steps = 6,
            hmc_step_size = 0.03,
            sep_line_search = true,
            sep_linear_solver = :qr,
            silent = true
        )
        @test !hmc.errorflag
        sim_hmc = Array(hmc.simulation)
        dmax = maximum(abs, sim_gh .- sim_hmc)
        rmse = sqrt(mean((sim_gh .- sim_hmc) .^ 2))
        metrics[hs] = (dmax, rmse)
    end

    dmax_64, rmse_64 = metrics[64]
    dmax_256, rmse_256 = metrics[256]

    # Practical closeness gate for small-model validation.
    @test dmax_64 < 0.02
    @test dmax_256 < 0.02
    @test rmse_64 < 0.01
    @test rmse_256 < 0.01
end
