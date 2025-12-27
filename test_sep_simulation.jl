# Test SEP stochastic simulation
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("TESTING SEP STOCHASTIC SIMULATION")
println("="^70)

# Solve SEP
println("\n1. Solving SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 40,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = true)

println("   ✓ SEP solution complete")

# Test stochastic simulation
println("\n2. Running stochastic simulation...")
sim_result, shocks_used = simulate_sep(m,
                                       periods = 100,
                                       burn_in = 50,
                                       sep_horizon = 40,
                                       sep_order = 1,
                                       sep_nnodes = 3,
                                       random_seed = 123,
                                       silent = true)

println("   ✓ Simulation complete")
println("   Result dimensions: ", size(sim_result))
println("   Shocks dimensions: ", size(shocks_used))

# Check some statistics
println("\n3. Simulation statistics:")
y_idx = findfirst(==(:y), axiskeys(sim_result, 1))
y_path = sim_result[y_idx, :]

println("   Variable: y (output)")
println("   Mean: ", mean(y_path))
println("   Std: ", std(y_path))
println("   Min: ", minimum(y_path))
println("   Max: ", maximum(y_path))

# Get steady state for comparison
SS = get_steady_state(m, derivatives=false)
y_ss = Float64(SS(:y))
println("   Deterministic SS: ", y_ss)
println("   Deviation from SS (mean): ", mean(y_path) - y_ss)

println("\n" * "="^70)
println("TEST COMPLETE")
println("="^70)
