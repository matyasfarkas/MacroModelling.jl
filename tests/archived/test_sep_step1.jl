# Step 1: Just solve SEP and check what variables are returned
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("Solving with SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 5,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 50,
       silent = true)

println("✓ SEP solved")

# Extract IRF
println("\nExtracting IRF...")
irf = MacroModelling.get_sep_irf(m, :epinf, 1.0, periods=5)

println("\nIRF object type: ", typeof(irf))
println("IRF size: ", size(irf))
println("\nVariable names in IRF:")
println(axiskeys(irf, 1))

println("\nFirst few values:")
println(irf[1:min(5, end), 1:3])
