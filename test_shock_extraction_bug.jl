# Test to diagnose shock extraction bug
# Compare reference vs MacroModelling implementation

using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SHOCK EXTRACTION DIAGNOSTIC")
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

println("\nTree structure:")
println("  K (shock combinations) = $K")
println("  Lbr (branching periods) = $(layout.Lbr)")
println("  T (total periods) = $(sep_sol.periods)")
println("  dε (number of shocks) = $(layout.dε)")

println("\nNumber of groups at each time:")
for t in 0:min(3, sep_sol.periods)
    Gt = layout.G[t+1]
    println("  t=$t: $Gt groups")
end

# Test shock extraction logic
println("\n" * "="^70)
println("SHOCK EXTRACTION COMPARISON")
println("="^70)

# Reference implementation logic:
# k = mod(g - 1, K) + 1
println("\nReference Implementation (node_shock):")
for t in 1:min(3, sep_sol.periods)
    println("  t=$t:")
    Gt = layout.G[t+1]
    for g in 1:min(5, Gt)
        k_ref = mod(g - 1, K) + 1
        cgs = MacroModelling.child_groups(layout, t, g)
        println("    g=$g -> k=$k_ref, children=$(collect(cgs))")
    end
    if Gt > 5
        println("    ... ($(Gt-5) more groups)")
    end
end

println("\nMacroModelling Implementation (current):")
println("  Uses: ε_curr = X[:, cg] where cg is child group")
for t in 1:min(3, sep_sol.periods)
    println("  t=$t:")
    Gt = layout.G[t+1]
    for g in 1:min(5, Gt)
        cgs = MacroModelling.child_groups(layout, t, g)
        println("    g=$g -> children=$(collect(cgs)), uses X[:, cg] for each cg")
    end
    if Gt > 5
        println("    ... ($(Gt-5) more groups)")
    end
end

# Check if they give same results for Lbr=1
println("\n" * "="^70)
println("EQUIVALENCE CHECK (Lbr=1)")
println("="^70)

all_match = true
for t in 1:sep_sol.periods
    Gt = layout.G[t+1]
    for g in 1:Gt
        cgs = MacroModelling.child_groups(layout, t, g)

        # Reference: uses current group g
        k_ref = mod(g - 1, K) + 1

        # MacroModelling: uses child group cg
        # For Lbr=1, t>=1: cgs = g:g, so cg = g
        cg_mm = only(cgs)  # Should be single element

        if k_ref != cg_mm
            all_match = false
            println("  MISMATCH at t=$t, g=$g: k_ref=$k_ref, cg_mm=$cg_mm")
        end
    end
end

if all_match
    println("✓ For Lbr=1, reference and MacroModelling use SAME shock indices")
    println("  So the magnitude issue is NOT due to shock extraction bug!")
else
    println("✗ FOUND DIFFERENCES in shock extraction!")
end

println("\n" * "="^70)
println("CONCLUSION")
println("="^70)
println("If all match: The small IRF magnitude is NOT due to shock extraction.")
println("Need to investigate other parts of the SEP solver.")
println("="^70)
