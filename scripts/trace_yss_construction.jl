# Trace exactly how yss is constructed in SEP solver

using MacroModelling
using Printf

println("="^70)
println("TRACING yss CONSTRUCTION")
println("="^70)

# Load FS2000
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

# Replicate the exact logic from sep_solver.jl lines 160-179
SS_result = get_steady_state(FS2000, return_variables_only=true, derivatives=false)

ss_keys = try
    axiskeys(SS_result, 1)
catch
    Symbol[]
end

println("\n1. SS_result keys (length=$(length(ss_keys))):")
for (i, k) in enumerate(ss_keys)
    println("  $i: $k")
end

println("\n2. FS2000.var (length=$(length(FS2000.var))):")
for (i, v) in enumerate(FS2000.var)
    println("  $i: $v")
end

println("\n3. Building yss vector (simulating sep_solver.jl lines 169-179):")
yss = Float64[]
for (i, var) in enumerate(FS2000.var)
    if var ∈ ss_keys
        val = Float64(SS_result(var))
        push!(yss, val)
        @printf("  yss[%2d] = %12.8f  (from SS_result(%s))\n", i, val, var)
    else
        push!(yss, 0.0)
        @printf("  yss[%2d] = %12.8f  (auxiliary %s - initialized to 0)\n", i, 0.0, var)
    end
end

println("\n4. Final yss vector (length=$(length(yss))):")
for i in 1:length(yss)
    @printf("  yss[%2d] = %12.8f   (var: %s)\n", i, yss[i], FS2000.var[i])
end

println("\n5. Checking if y is in yss:")
y_idx = findfirst(==(Symbol("y")), FS2000.var)
println("  y is at index $y_idx in FS2000.var")
println("  yss[$y_idx] = $(yss[y_idx])")
println("  Expected y value from SS = $(Float64(SS_result(:y)))")

println("\n" * "="^70)
