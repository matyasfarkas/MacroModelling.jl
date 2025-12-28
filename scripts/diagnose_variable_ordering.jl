# Diagnostic: Determine exact variable ordering in SEP solution

using MacroModelling
using Printf

println("="^70)
println("DIAGNOSTIC: Variable Ordering in SEP Solution")
println("="^70)

# Load FS2000
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

println("\n1. Model structure")
println("   model.var (length=$(length(FS2000.var))):")
for (i, v) in enumerate(FS2000.var)
    println("     $i: $v")
end

println("\n2. Checking timings structure")
println("   timings.var:")
try
    for (i, v) in enumerate(FS2000.timings.var)
        println("     $i: $v")
    end
catch e
    println("   Error accessing timings.var: $e")
end

# Get deterministic steady state
SS = get_steady_state(FS2000)
ss_vars = axiskeys(SS, 1)
ss_cols = axiskeys(SS, 2)
col_idx = findfirst(==(:Steady_state), ss_cols)

println("\n3. Deterministic steady state values (in SS order):")
for (i, var) in enumerate(ss_vars)
    val = SS[i, col_idx]
    @printf("   %2d. %-12s = %12.8f\n", i, string(var), val)
end

# Run SEP
println("\n4. Running SEP...")
MacroModelling.solve!(FS2000,
                     algorithm = :stochastic_extended_path,
                     sep_periods = 10,
                     sep_order = 1,
                     sep_nnodes = 3,
                     sep_maxit = 100,
                     silent = true)

sep_sol = FS2000.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout
yss_sep = sep_sol.Y[1:layout.ny_]

println("\n5. SEP Y vector (first $(layout.ny_) elements):")
for i in 1:layout.ny_
    @printf("   Y[%2d] = %12.8f\n", i, yss_sep[i])
end

# Try to reverse-engineer the mapping
println("\n6. Attempting to match SEP values to steady state:")
println("   (Finding which SS value matches each SEP Y value)")
tolerance = 1e-6
for i in 1:layout.ny_
    sep_val = yss_sep[i]
    found = false
    for (j, var) in enumerate(ss_vars)
        ss_val = SS[j, col_idx]
        if abs(sep_val - ss_val) < tolerance
            @printf("   Y[%2d] = %12.8f  matches  SS[%-12s] = %12.8f\n",
                    i, sep_val, string(var), ss_val)
            found = true
            break
        end
    end
    if !found
        @printf("   Y[%2d] = %12.8f  NO MATCH in SS\n", i, sep_val)
    end
end

println("\n" * "="^70)
