# Minimal test of SEP IRF scaling
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("MINIMAL SEP IRF SCALING TEST")
println("="^70)

# Solve with SEP - use minimal settings for speed
println("\nSolving with SEP (periods=5, nnodes=3)...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 5,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 50,  # Fewer iterations for speed
       silent = false)

sep_sol = m.solution.perturbation.stochastic_extended_path

println("\n✓ SEP solved")
println("Convergence flag: ", sep_sol.convergence_flag)

# Extract IRF directly
println("\nExtracting IRF for 1.0σ epinf shock...")
irf = MacroModelling.get_sep_irf(m, :epinf, 1.0, periods=5)

println("\nIRF values for y (output):")
for t in 0:5
    val = irf[:y][t+1]
    println("  t=$t: $(round(val, digits=8))")
end

impact = irf[:y][2]
println("\n" * "="^70)
println("RESULT")
println("="^70)
println("Impact response of y: $(round(impact, digits=8))")
println("\nIf scaling works correctly:")
println("  - Value should NOT be zero (bug fix working)")
println("  - Value should be scaled to 1.0σ shock (not √3σ ≈ 1.73σ)")
println("="^70)
