# Test if shock covariance is being extracted correctly
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SHOCK COVARIANCE EXTRACTION TEST")
println("="^70)

println("\nSolving with SEP (verbose=true to see shock info)...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 3,
       sep_order = 1,
       sep_nnodes = 3,
       silent = false)  # Keep verbose ON

sep_sol = m.solution.perturbation.stochastic_extended_path

println("\n✓ SEP solved")
println("Convergence flag: ", sep_sol.convergence_flag)
println("Final error: ", sep_sol.final_error)
