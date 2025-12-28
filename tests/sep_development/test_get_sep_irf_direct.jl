# Direct test of get_sep_irf function
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("Solving with SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       silent = true)

println("Calling get_sep_irf for 1.0σ epinf shock...")
irf = MacroModelling.get_sep_irf(m, :epinf, 1.0, periods=10)

println("\nIRF values for y:")
for t in 0:10
    println("  t=$t: $(irf[:y][t+1])")
end

println("\nImpact response: $(irf[:y][2])")
