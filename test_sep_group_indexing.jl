# Test SEP group indexing for IRF extraction
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SEP GROUP INDEXING DIAGNOSTIC")
println("="^70)

# Solve with SEP
println("\n1. Solving SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 5,  # Short for testing
       sep_order = 1,
       sep_nnodes = 3,
       silent = true)

sep_sol = m.solution.perturbation.stochastic_extended_path
println("✓ SEP converged")

# Get basic info
layout = sep_sol.layout
Y = sep_sol.Y
nnodes = sep_sol.nnodes
nshocks = layout.dε

println("\n2. SEP Configuration:")
println("  Periods (T): ", sep_sol.periods)
println("  Branching order (L): ", sep_sol.order)
println("  GH nodes per shock: ", nnodes)
println("  Number of shocks: ", nshocks)
println("  Total GH nodes (K): ", layout.K, " (should be ", nnodes^nshocks, ")")

# Shock to test: epinf (price markup)
shock_name = :epinf
shock_idx = findfirst(==(shock_name), m.exo)
println("\n3. Testing shock: ", shock_name, " (index: ", shock_idx, ")")

# Calculate group index for this shock
zero_node_idx = (nnodes + 1) ÷ 2  # Middle node for zero
target_node_idx = nnodes  # Highest node for positive shock

println("  Zero node index: ", zero_node_idx)
println("  Target node index (for +1σ shock): ", target_node_idx)

# Build group index using base-nnodes arithmetic
shock_group_t1 = let
    idx_val = 0
    for d in 1:nshocks
        if d == shock_idx
            contrib = (target_node_idx - 1) * nnodes^(d - 1)
            idx_val += contrib
            println("  Shock $d (SHOCKED): contributes ", contrib, " to idx")
        else
            contrib = (zero_node_idx - 1) * nnodes^(d - 1)
            idx_val += contrib
            println("  Shock $d (at zero): contributes ", contrib, " to idx")
        end
    end
    idx_val + 1
end

println("\n4. Calculated group index: ", shock_group_t1)
println("  (out of ", layout.K, " total groups)")

# Check if this group is valid
if shock_group_t1 > layout.K
    println("  ⚠️  WARNING: Group index exceeds total groups!")
end

# Get steady state (t=0, group 1)
println("\n5. Extracting values at different groups:")
yss_indices = MacroModelling.index_y(layout, 0, 1)
yss = Y[yss_indices]

# Get variable index for output (y)
var_idx = findfirst(==(:y), m.var)
println("  Variable: y (index: ", var_idx, ")")
println("  Steady state value: ", yss[var_idx])

# Check values at t=1 for different groups
println("\n  Values at t=1:")
for g in [1, shock_group_t1, min(shock_group_t1+1, layout.K)]
    if g <= layout.K
        y_indices = MacroModelling.index_y(layout, 1, g)
        yt = Y[y_indices]
        println("    Group ", g, ": y = ", yt[var_idx], " (deviation: ", yt[var_idx] - yss[var_idx], ")")
    end
end

# Sample a few more groups to see if ANY have non-zero deviations
println("\n6. Sampling other groups at t=1 to find non-zero deviations:")
for g in [1, 10, 100, 500, 1000, 1500, 2000, layout.K]
    if g <= layout.K
        y_indices = MacroModelling.index_y(layout, 1, g)
        yt = Y[y_indices]
        dev = yt[var_idx] - yss[var_idx]
        if abs(dev) > 1e-10
            println("  ✓ Group ", g, ": deviation = ", dev)
        end
    end
end

println("\n" * "="^70)
