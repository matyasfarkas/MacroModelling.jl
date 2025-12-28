# Find the exact mapping between SEP Y indices and variable names

using MacroModelling
using Printf

println("="^70)
println("SEP VARIABLE MAPPING DIAGNOSTIC")
println("="^70)

# Load FS2000
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

# Get deterministic SS for reference values
SS = get_steady_state(FS2000)
ss_vars = axiskeys(SS, 1)
ss_cols = axiskeys(SS, 2)
col_idx = findfirst(==(:Steady_state), ss_cols)

println("\n1. Variable orderings:")
println("\n  model.var order (length=$(length(FS2000.var))):")
for (i, v) in enumerate(FS2000.var)
    println("    $i: $v")
end

println("\n  timings.var order (length=$(length(FS2000.timings.var))):")
for (i, v) in enumerate(FS2000.timings.var)
    println("    $i: $v")
end

# Run SEP
println("\n2. Running SEP...")
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

println("\n3. SEP Y values (length=$(length(yss_sep))):")
for i in 1:layout.ny_
    @printf("  Y[%2d] = %12.8f\n", i, yss_sep[i])
end

# Create value-to-variable mapping from steady state
println("\n4. Reverse engineering SEP variable order:")
println("   (Finding which SS variable each Y[i] corresponds to)")

tolerance = 1e-6
sep_to_var = Vector{Union{Symbol, Nothing}}(nothing, layout.ny_)

for i in 1:layout.ny_
    sep_val = yss_sep[i]

    # Skip zeros (auxiliary variables)
    if abs(sep_val) < tolerance
        sep_to_var[i] = Symbol("ZERO_$(i)")
        continue
    end

    # Find matching SS value
    for (j, var) in enumerate(ss_vars)
        ss_val = SS[j, col_idx]
        if abs(sep_val - ss_val) < tolerance
            sep_to_var[i] = var
            break
        end
    end
end

println("\n5. SEP Y index → Variable name mapping:")
for i in 1:layout.ny_
    var_name = isnothing(sep_to_var[i]) ? "NO_MATCH" : string(sep_to_var[i])
    @printf("  Y[%2d] → %-15s = %12.8f\n", i, var_name, yss_sep[i])
end

# Check if it matches model.var order
println("\n6. Does SEP Y match model.var order?")
matches_model = true
for i in 1:min(length(FS2000.var), layout.ny_)
    expected = FS2000.var[i]
    actual = sep_to_var[i]
    match_str = (expected == actual) ? "✓" : "✗"
    println("  Y[$i]: expected=$(expected), actual=$(actual) $match_str")
    if expected != actual
        matches_model = false
    end
end
println("\n  Overall: SEP uses model.var order? ", matches_model ? "YES" : "NO")

# Check if it matches timings.var order
println("\n7. Does SEP Y match timings.var order?")
matches_timings = true
for i in 1:min(length(FS2000.timings.var), layout.ny_)
    expected = FS2000.timings.var[i]
    actual = sep_to_var[i]
    match_str = (expected == actual) ? "✓" : "✗"

    # For zeros, check if expected is auxiliary
    if isnothing(actual) || startswith(string(actual), "ZERO")
        # Check if expected is auxiliary
        if contains(string(expected), "ᴸ") || contains(string(expected), "ᴾ")
            match_str = "✓ (auxiliary)"
            actual = expected
        end
    end

    println("  Y[$i]: expected=$(expected), actual=$(actual) $match_str")
    if expected != actual && !(contains(string(expected), "ᴸ") || contains(string(expected), "ᴾ"))
        matches_timings = false
    end
end
println("\n  Overall: SEP uses timings.var order? ", matches_timings ? "YES" : "NO")

println("\n" * "="^70)
println("CONCLUSION:")
if matches_model
    println("  SEP Y vector follows model.var order")
elseif matches_timings
    println("  SEP Y vector follows timings.var order")
else
    println("  SEP Y vector uses CUSTOM order - need to determine mapping!")
    println("\n  Suggested mapping for test file:")
    println("  # Create SEP variable index map")
    println("  sep_var_map = Dict(")
    for i in 1:layout.ny_
        if !isnothing(sep_to_var[i]) && !startswith(string(sep_to_var[i]), "ZERO")
            println("      :$(sep_to_var[i]) => $i,")
        end
    end
    println("  )")
end
println("="^70)
