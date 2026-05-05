using Test
using Random
using MacroModelling

println("="^80)
println("SEP INVERSION FILTER LIKELIHOOD TESTS")
println("="^80)

@testset verbose = true "SEP Inversion Filter Likelihood" begin
    @testset "Logdet method handles singular square Jacobians explicitly" begin
        J = [1.0 0.0; 0.0 0.0]
        @test MacroModelling._sep_inv_logabsdet(J; method = :exact) == -Inf
        @test MacroModelling._sep_inv_logabsdet(J; method = :svd_pseudodet) ≈ 0.0

        diag = MacroModelling._sep_inv_logdet_diagnostics(J; method = :svd_pseudodet)
        @test diag["sep_inv_logdet_method"] == "svd_pseudodet"
        @test diag["sep_inv_logdet_rank"] == 1
    end

    include("../models/RBC_Dynare.jl")

    Random.seed!(20260225)
    sep_sim = simulate_sep_extended_path(
        RBC_Dynare;
        periods = 3,
        sep_horizon = 6,
        sep_order = 1,
        sep_nnodes = 3,
        sep_sparse_tree = true,
        sep_maxit = 60,
        sep_tol = 1e-6,
        sep_accept_tol = 1e-3,
        shock_scaling = :none,
        silent = true,
    )

    @test !sep_sim.errorflag
    data = sep_sim.simulation(Variables = [:Output])

    @testset "Runs and returns finite likelihood" begin
        MacroModelling.reset_sep_inversion_last_diagnostics!()
        ll = get_loglikelihood(
            RBC_Dynare,
            data,
            RBC_Dynare.parameter_values;
            algorithm = :stochastic_extended_path,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = -1e12,
        )

        @test isfinite(ll)
        @test ll > -1e11

        diag = MacroModelling.get_sep_inversion_last_diagnostics()
        @test diag !== nothing
        @test diag isa AbstractDict
        @test get(diag, "status", nothing) == "ok"
        @test get(diag, "kind", nothing) == "sep_inversion_filter"
        @test get(diag, "n_periods", 0) == size(data, 2)
        @test get(diag, "sep_order", nothing) !== nothing
    end

    @testset "Deterministic across repeated calls" begin
        ll1 = get_loglikelihood(
            RBC_Dynare,
            data,
            RBC_Dynare.parameter_values;
            algorithm = :stochastic_extended_path,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = -1e12,
        )
        ll2 = get_loglikelihood(
            RBC_Dynare,
            data,
            RBC_Dynare.parameter_values;
            algorithm = :stochastic_extended_path,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = -1e12,
        )

        @test ll1 ≈ ll2 atol = 1e-8 rtol = 1e-8
    end

    @testset "Accepts SEP runtime override keywords" begin
        MacroModelling.reset_sep_inversion_last_diagnostics!()
        ll = get_loglikelihood(
            RBC_Dynare,
            data,
            RBC_Dynare.parameter_values;
            algorithm = :stochastic_extended_path,
            filter = :inversion,
            verbose = false,
            on_failure_loglikelihood = -1e12,
            sep_periods = 20,
            sep_order = 1,
            sep_nnodes = 3,
            sep_maxit = 80,
            sep_tol = 1e-7,
            sep_accept_tol = 0.25,
            sep_inv_maxit = 8,
            sep_inv_resid_tol = 1e-6,
            sep_inv_step_tol = 1e-6,
            sep_inv_lambda = 1e-4,
            sep_inv_predict_tol = 1e-9,
            sep_inv_logdet_method = :svd_pseudodet,
        )

        @test isfinite(ll)
        @test ll > -1e11

        diag = MacroModelling.get_sep_inversion_last_diagnostics()
        @test diag isa AbstractDict
        @test get(diag, "status", nothing) == "ok"
        @test get(diag, "sep_periods", nothing) == 20
        @test get(diag, "sep_order", nothing) == 1
        @test get(diag, "sep_nnodes", nothing) == 3
        @test get(diag, "sep_inv_maxit", nothing) == 8
        @test get(diag, "sep_inv_predict_tol", nothing) == 1e-9
        @test get(diag, "sep_inv_logdet_method", nothing) == "svd_pseudodet"
    end
end
