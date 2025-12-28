# Simple test of SEP IRF scaling
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SEP IRF SCALING VERIFICATION")
println("="^70)

# Solve with SEP (nnodes=3 for simplicity)
println("\nSolving with SEP (nnodes=3, order=1)...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       silent = true)

println("SEP solution completed")

# Extract IRF for 1.0σ shock
println("\nExtracting IRF for 1.0σ epinf shock...")
irf_1sigma = MacroModelling.get_sep_irf(m, :epinf, 1.0, periods=10)

println("\nIRF for y (output) with 1.0σ shock:")
println("  t=0: $(irf_1sigma[:y][1])")
println("  t=1: $(irf_1sigma[:y][2])")
println("  t=2: $(irf_1sigma[:y][3])")
println("  t=3: $(irf_1sigma[:y][4])")

# Check the magnitude
y_impact = irf_1sigma[:y][2]  # Impact response
println("\nImpact response of y: $(round(y_impact, digits=8))")

# For reference, get perturbation IRF
println("\nGetting perturbation IRF for comparison...")
solve!(m, algorithm=:first_order, silent=true)
pert_irf = get_irf(m, shocks=:epinf, periods=10)

println("\nPerturbation IRF for y (output) with 1.0σ shock:")
println("  t=0: $(pert_irf[:y][1])")
println("  t=1: $(pert_irf[:y][2])")
println("  t=2: $(pert_irf[:y][3])")
println("  t=3: $(pert_irf[:y][4])")

pert_impact = pert_irf[:y][2]
println("\nImpact response of y: $(round(pert_impact, digits=8))")

# Compare
ratio = y_impact / pert_impact
println("\n" * "="^70)
println("COMPARISON")
println("="^70)
println("SEP impact:          $(round(y_impact, digits=8))")
println("Perturbation impact: $(round(pert_impact, digits=8))")
println("Ratio (SEP/Pert):    $(round(ratio, digits=4))")
println("\nExpected: Ratio should be close to 1.0 for similar shock responses")
println("If ratio >> 1: SEP shock is too large (scaling error)")
println("If ratio << 1: SEP shock is too small (scaling error)")
println("="^70)
