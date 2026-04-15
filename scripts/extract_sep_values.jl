# Extract SEP steady state values from SW07_HLT for Dynare comparison
# Run this after solve!() has completed

using MacroModelling
using Printf
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

function parse_arg_int(key::String, default::Int)
    for arg in ARGS
        startswith(arg, key * "=") || continue
        return parse(Int, split(arg, "=", limit = 2)[2])
    end
    return default
end

skip_order2 = "--skip-order2" in ARGS
sep_periods = parse_arg_int("--sep-periods", 10)
sep_maxit = parse_arg_int("--sep-maxit", 100)

sep_convergence_status(flag) = flag == 0 ? "SUCCESS" : "FAILED"

println("="^70)
println("MacroModelling.jl SEP Results Extract")
println("="^70)

# Load the model
mm_model = load_hlt_model(normpath(joinpath(@__DIR__, "..")), "Smets_Wouters_2007_HLT"; mod = @__MODULE__)

# Solve SEP(1)
println("\nSolving SEP(1): periods=$(sep_periods), order=1, nnodes=3...")
solve!(mm_model,
       algorithm = :stochastic_extended_path,
       sep_periods = sep_periods,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = sep_maxit,
       silent = true)

sep_sol_1 = mm_model.solution.perturbation.stochastic_extended_path

# Extract steady state values
layout = sep_sol_1.layout
yss_indices = 1:layout.ny_
yss_1 = sep_sol_1.Y[yss_indices]

# Key variables to report
test_vars = [:y, :c, :inve, :pinf, :lab]

println("\n" * "="^70)
println("SEP(1) RESULTS - periods=$(sep_periods), order=1")
println("="^70)
@printf("Convergence: %s\n", sep_convergence_status(sep_sol_1.convergence_flag))
@printf("Convergence flag: %d\n", sep_sol_1.convergence_flag)
@printf("Final error: %.3e\n", sep_sol_1.final_error)
@printf("Runtime: %.2f seconds\n\n", sep_sol_1.runtime_seconds)

println("Key steady state values:")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    var_idx = findfirst(==(var), mm_model.var)
    if !isnothing(var_idx)
        @printf("  %-12s  %12.8f\n", string(var), yss_1[var_idx])
    end
end

if skip_order2
    println("\nSkipping SEP(2) solve (--skip-order2).")
    exit(0)
end

# Solve SEP(2)
println("\n" * "="^70)
println("Solving SEP(2): periods=$(sep_periods), order=2, nnodes=3...")
println("="^70)

solve!(mm_model,
       algorithm = :stochastic_extended_path,
       sep_periods = sep_periods,
       sep_order = 2,
       sep_nnodes = 3,
       sep_maxit = sep_maxit,
       silent = true)

sep_sol_2 = mm_model.solution.perturbation.stochastic_extended_path

# Extract steady state values
layout_2 = sep_sol_2.layout
yss_indices_2 = 1:layout_2.ny_
yss_2 = sep_sol_2.Y[yss_indices_2]

println("\n" * "="^70)
println("SEP(2) RESULTS - periods=$(sep_periods), order=2")
println("="^70)
@printf("Convergence: %s\n", sep_convergence_status(sep_sol_2.convergence_flag))
@printf("Convergence flag: %d\n", sep_sol_2.convergence_flag)
@printf("Final error: %.3e\n", sep_sol_2.final_error)
@printf("Runtime: %.2f seconds\n\n", sep_sol_2.runtime_seconds)

println("Key steady state values:")
println("  Variable      Value")
println("  " * "-"^30)
for var in test_vars
    var_idx = findfirst(==(var), mm_model.var)
    if !isnothing(var_idx)
        @printf("  %-12s  %12.8f\n", string(var), yss_2[var_idx])
    end
end

# Comparison
println("\n" * "="^70)
println("COMPARISON: SEP(1) vs SEP(2)")
println("="^70)
println("  Variable      SEP(1)         SEP(2)         Difference")
println("  " * "-"^65)

for var in test_vars
    var_idx = findfirst(==(var), mm_model.var)
    if !isnothing(var_idx)
        val1 = yss_1[var_idx]
        val2 = yss_2[var_idx]
        diff = val2 - val1
        @printf("  %-12s  %12.8f  %12.8f  %12.8f\n",
                string(var), val1, val2, diff)
    end
end

println("\n" * "="^70)
println("COPY THESE VALUES TO COMPARE WITH DYNARE OUTPUT")
println("="^70)
