# Test weight indexing bug
# Compare W[kidx] vs W[cg]

using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("WEIGHT INDEXING DIAGNOSTIC")
println("="^70)

# Solve with SEP
println("\nSolving SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,    # Lbr = 1
       sep_nnodes = 3,
       sep_maxit = 50,
       silent = true)

sep_sol = m.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout
X = sep_sol.X
W = sep_sol.W
K = sep_sol.nnodes^layout.dε

println("\nWeights array W:")
println("  Size: ", size(W))
println("  First 10 elements: ", W[1:min(10, length(W))])
println("  Sum of weights: ", sum(W), " (should be 1.0)")

println("\n" * "="^70)
println("COMPARISON AT t=1")
println("="^70)

t = 1
Gt = layout.G[t+1]
println("\nAt t=$t:")
println("  Number of groups: $Gt")
println("  K (shock combinations): $K")

# Test for first few groups
println("\nWeight indexing for first 5 groups:")
for g in 1:min(5, Gt)
    cgs = MacroModelling.child_groups(layout, t, g)

    # Reference approach
    k_ref = mod(g - 1, K) + 1

    println("  g=$g:")
    println("    Children: $(collect(cgs))")
    println("    Reference: k = $k_ref")

    for (kidx, cg) in enumerate(cgs)
        w_ref = W[kidx]      # Reference: uses enumeration index
        w_mm = W[cg]         # MacroModelling: uses child group

        println("      Child $cg: kidx=$kidx, W[kidx]=$w_ref, W[cg]=$w_mm")

        if w_ref != w_mm
            println("      ⚠ MISMATCH! W[kidx] ≠ W[cg]")
        end
    end
end

println("\n" * "="^70)
println("KEY INSIGHT")
println("="^70)
println("For Lbr=1 at t=1:")
println("  - Each group g has single child: cgs = [g]")
println("  - Enumeration index: kidx = 1 (always)")
println("  - Child group: cg = g (varies)")
println()
println("Reference uses: W[kidx] = W[1] for ALL groups")
println("MacroModelling uses: W[cg] = W[g] (different for each group)")
println()
println("W[1] = $(W[1])")
println("But W should have different weights for different shock combinations!")
println("="^70)

# Now check what W actually represents
println("\n" * "="^70)
println("UNDERSTANDING W")
println("="^70)
println("W has $(length(W)) elements (should equal K=$K)")
println()
println("For nnodes=3, nshocks=$(layout.dε):")
println("  K = 3^$(layout.dε) = $K")
println("  Each W[k] is the probability weight for shock combination k")
println()
println("Sample weights:")
for k in [1, 100, 500, 1000, 1500, 2000, K]
    if k <= length(W)
        println("  W[$k] = $(W[k])")
    end
end
