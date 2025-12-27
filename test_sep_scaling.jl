# Test SEP IRF scaling
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("SEP IRF SCALING TEST")
println("="^70)

# Solve with SEP
println("\nSolving with SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       silent = true)

# Test different shock sizes for epinf
shock_sizes = [0.5, 1.0, 1.5, 2.0]

println("\nTesting IRF scaling for different shock sizes:")
println("Shock: epinf (inflation markup shock)")
println("Variable: y (output)")

for shock_size in shock_sizes
    irf = get_irf(m,
                  shocks = :epinf,
                  shock_size = shock_size,
                  algorithm = :stochastic_extended_path,
                  periods = 10)

    # Get impact response (t=1)
    y_impact = irf[:y][2]  # Index 2 is t=1 (index 1 is t=0)

    println("\nShock size: $(shock_size)σ")
    println("  Impact response of y: $(round(y_impact, digits=6))")

    if shock_size > 0.5
        ratio = y_impact / prev_impact
        println("  Ratio to previous: $(round(ratio, digits=4)) (should be ≈ $(shock_size/prev_size))")
    end

    global prev_impact = y_impact
    global prev_size = shock_size
end

println("\n" * "="^70)
println("If scaling is correct, the ratios should match the shock size ratios")
println("="^70)
