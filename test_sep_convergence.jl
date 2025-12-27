# Test SEP with better convergence settings
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SEP CONVERGENCE TEST")
println("="^70)

println("\nSolving with SEP (more iterations, tighter tolerance)...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 5,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 200,  # More iterations
       sep_tol = 1e-8,   # Tighter tolerance
       silent = false)

sep_sol = m.solution.perturbation.stochastic_extended_path

println("\n✓ SEP solved")
println("Convergence flag: ", sep_sol.convergence_flag)
println("Final error: ", sep_sol.final_error)

# Check values
layout = sep_sol.layout
Y = sep_sol.Y

# Get steady state
yss_indices = MacroModelling.index_y(layout, 0, 1)
yss = Y[yss_indices]

# Check group 1175 at t=1 (epinf shock)
y_idx = findfirst(==(:y), m.var)
y_ss = yss[y_idx]

y_indices_g1 = MacroModelling.index_y(layout, 1, 1)
y_indices_g1175 = MacroModelling.index_y(layout, 1, 1175)

y_g1 = Y[y_indices_g1][y_idx]
y_g1175 = Y[y_indices_g1175][y_idx]

println("\nVariable: y (output)")
println("  Steady state: ", y_ss)
println("  Group 1 (no shock) at t=1: ", y_g1, " (dev: ", y_g1 - y_ss, ")")
println("  Group 1175 (epinf shock) at t=1: ", y_g1175, " (dev: ", y_g1175 - y_ss, ")")
println("\nShock should cause deviation of ~", abs(y_g1175 - y_g1))
