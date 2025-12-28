# Direct comparison: SEP vs Perturbation IRFs
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SEP vs PERTURBATION IRF COMPARISON")
println("="^70)

# First: Perturbation IRF
println("\n1. Computing PERTURBATION IRF (1st order)...")
solve!(m, algorithm=:first_order, silent=true)
pert_irf = get_irf(m, shocks=:epinf, periods=10)

println("\nPerturbation IRF for y:")
y_pert = pert_irf(:y, :, :epinf)
for t in 0:5
    println("  t=$t: $(round(y_pert[t+1], digits=8))")
end
pert_impact = y_pert[2]

# Second: SEP IRF
println("\n2. Computing SEP IRF...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,  # More iterations
       silent = true)

sep_irf = MacroModelling.get_sep_irf(m, :epinf, 1.0, periods=10)

println("\nSEP IRF for y:")
y_idx = findfirst(==(:y), axiskeys(sep_irf, 1))
y_sep = sep_irf[y_idx, :]
for t in 0:5
    println("  t=$t: $(round(y_sep[t+1], digits=8))")
end
sep_impact = y_sep[2]

# Comparison
println("\n" * "="^70)
println("COMPARISON")
println("="^70)
println("Perturbation impact: $(round(pert_impact, digits=8))")
println("SEP impact:          $(round(sep_impact, digits=8))")
println("Ratio (SEP/Pert):    $(round(sep_impact/pert_impact, digits=4))")
println("\nExpected: Ratio should be close to 1.0")
println("If ratio << 1: SEP shock is too small")
println("If ratio >> 1: SEP shock is too large")
println("="^70)
