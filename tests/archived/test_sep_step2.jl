# Step 2: Check specific variable values
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
println("\nExtracting IRF for 1.0σ epinf shock...")
irf = MacroModelling.get_sep_irf(m, :epinf, 1.0, periods=5)

# Check y variable
println("\nVariable: y (output)")
println("IRF values:")
y_idx = findfirst(==(:y), axiskeys(irf, 1))
println("  y index: $y_idx")

for t in 0:5
    val = irf[y_idx, t+1]
    println("  t=$t: $(round(val, digits=10))")
end

impact = irf[y_idx, 2]
println("\n" * "="^70)
println("RESULT")
println("="^70)
println("Impact response of y: $(round(impact, digits=10))")
println("\n✓ Bug fix verification:")
if abs(impact) > 1e-10
    println("  ✓ IRF is NON-ZERO (bug fix working!)")
    println("  ✓ Magnitude: $(abs(impact))")
else
    println("  ✗ IRF is still zero (bug NOT fixed)")
end
println("="^70)
