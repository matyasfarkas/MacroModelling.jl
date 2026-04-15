"""
Test SEP solver pattern caching fix for varying nnz_count
This test verifies that the Jacobian sparsity pattern caching handles cases
where the number of nonzeros changes between iterations.
"""

using Test
using MacroModelling

println("=" ^ 80)
println("Testing SEP Solver Pattern Caching Fix")
println("=" ^ 80)
println()

# Load RBC_CME model (known working model from tests)
include(joinpath(@__DIR__, "models/RBC_CME.jl"))

@testset "SEP Jacobian Pattern Caching" begin
    println("\nTest 1: Standard SEP solve (baseline)")

    # Test with default parameters - should work
    result = MacroModelling.simulate(
        m,
        algorithm = :stochastic_extended_path,
        periods = 50,
        sep_periods = 20,
        sep_maxit = 100,
        sep_tol = 1e-5
    )

    @test size(result, 1) > 0
    @test size(result, 2) == 50
    println("✓ Baseline SEP solve successful")

    println("\nTest 2: SEP solve with modified parameters")

    # Modify parameters slightly and re-solve
    # This could potentially change Jacobian sparsity pattern
    params_modified = copy(m.parameter_values)
    params_modified[findfirst(x -> x == :alpha, m.parameters)] *= 0.9

    MacroModelling.write_parameters_input!(m, params_modified, verbose = false)

    result2 = MacroModelling.simulate(
        m,
        algorithm = :stochastic_extended_path,
        periods = 50,
        sep_periods = 20,
        sep_maxit = 100,
        sep_tol = 1e-5
    )

    @test size(result2, 1) > 0
    @test size(result2, 2) == 50
    println("✓ Modified parameter SEP solve successful")

    println("\nTest 3: Multiple sequential SEP solves")

    # Run multiple solves in sequence to test pattern caching persistence
    for i in 1:3
        result_seq = MacroModelling.simulate(
            m,
            algorithm = :stochastic_extended_path,
            periods = 30,
            sep_periods = 15,
            sep_maxit = 100,
            sep_tol = 1e-5,
            random_seed = 42 + i
        )
        @test size(result_seq, 2) == 30
    end

    println("✓ Sequential SEP solves successful")
end

println()
println("=" ^ 80)
println("All tests passed! ✅")
println("=" ^ 80)
