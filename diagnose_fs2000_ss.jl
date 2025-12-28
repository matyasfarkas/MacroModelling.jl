# Diagnostic: Compare FS2000 deterministic steady state vs SEP solution

using MacroModelling
using Printf

println("="^70)
println("DIAGNOSTIC: FS2000 Steady State vs SEP Solution")
println("="^70)

# Load the FS2000 model from MacroModelling package
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

println("\n1. Model loaded")
println("   Variables (total): ", length(FS2000.var))
println("   Variables (list): ", FS2000.var)
println("   Shocks: ", FS2000.exo)
println("   Parameters: ", length(FS2000.parameters))

# Get deterministic steady state
println("\n2. Computing deterministic steady state...")
SS = get_steady_state(FS2000)

println("\n3. Deterministic Steady State:")
println("  Variable      Value")
println("  " * "-"^30)
test_vars = [:m, :P, :c, :e, :W, :R, :k, :d, :n, :l, :gy_obs, :gp_obs, :y, :dA]
ss_vars = axiskeys(SS, 1)
ss_cols = axiskeys(SS, 2)
col_idx = findfirst(==(:Steady_state), ss_cols)
for var in test_vars
    if var ∈ ss_vars
        row_idx = findfirst(==(var), ss_vars)
        val = SS[row_idx, col_idx]
        @printf("  %-12s  %12.8f\n", string(var), val)
    else
        println("  %-12s  NOT IN SS", string(var))
    end
end

# Check for auxiliary variables
println("\n4. Auxiliary variables:")
aux_vars = filter(v -> contains(string(v), "ᴸ"), FS2000.var)
println("  Auxiliary vars: ", aux_vars)
if length(aux_vars) > 0
    println("  Auxiliary variable steady states:")
    for var in aux_vars
        if var ∈ ss_vars
            row_idx = findfirst(==(var), ss_vars)
            val = SS[row_idx, col_idx]
            @printf("    %-12s  %12.8f\n", string(var), val)
        else
            println("    %-12s  NOT IN STEADY STATE!", string(var))
        end
    end
end

# Now run SEP and compare
println("\n5. Running SEP solver...")
solve!(FS2000,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = true)

sep_sol = FS2000.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout
yss_sep = sep_sol.Y[1:layout.ny_]

println("\n6. SEP Solution (stochastic steady state):")
println("  Variable      SEP Value      Det SS         Difference")
println("  " * "-"^65)
for var in test_vars
    var_idx = findfirst(==(var), FS2000.var)
    if !isnothing(var_idx)
        sep_val = yss_sep[var_idx]
        if var ∈ ss_vars
            ss_row = findfirst(==(var), ss_vars)
            ss_val = SS[ss_row, col_idx]
            diff = sep_val - ss_val
            @printf("  %-12s  %12.8f  %12.8f  %12.8f\n",
                    string(var), sep_val, ss_val, diff)
        else
            @printf("  %-12s  %12.8f  NOT IN DET SS\n",
                    string(var), sep_val)
        end
    end
end

# Check auxiliary variables in SEP solution
println("\n7. Auxiliary variables in SEP solution:")
for var in aux_vars
    var_idx = findfirst(==(var), FS2000.var)
    if !isnothing(var_idx)
        sep_val = yss_sep[var_idx]
        @printf("  %-12s  %12.8f\n", string(var), sep_val)
    end
end

println("\n8. CRITICAL FINDINGS:")
println("   - If SEP values match deterministic SS → SEP solver is working correctly")
println("   - If R ≈ 0 in SEP → SEP found wrong equilibrium")
println("   - If auxiliary variables are 0 in SEP → initialization problem")
println("\n" * "="^70)
