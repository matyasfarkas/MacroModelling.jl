# Diagnose how to properly index the Y vector from SEP solution

using MacroModelling
using Printf

println("="^70)
println("DIAGNOSING SEP Y VECTOR INDEXING")
println("="^70)

# Load FS2000
include(joinpath(dirname(pathof(MacroModelling)), "..", "models", "FS2000.jl"))

# Run SEP
solve!(FS2000,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = true)

sep_sol = FS2000.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout

println("\n1. Layout structure:")
println("  T (periods): ", layout.T)
println("  Lbr (branching order): ", layout.Lbr)
println("  K (GH nodes): ", layout.K)
println("  ny_ (num variables): ", layout.ny_)
println("  voff (variable offsets): ", layout.voff)
println("  G (groups at each time): ", layout.G)
println("  Total length of Y: ", length(sep_sol.Y))

println("\n2. Expected vs actual:")
total_vars = sum(layout.G .* layout.ny_)
println("  Expected total vars in Y: ", total_vars)
println("  Actual length of Y: ", length(sep_sol.Y))

println("\n3. Extracting variables at t=0 (steady state) using Y[1:ny_]:")
Y_test1 = sep_sol.Y[1:layout.ny_]
for i in 1:min(5, layout.ny_)
    @printf("  Y[%2d] = %12.8f\n", i, Y_test1[i])
end
println("  ...")

println("\n4. Using layout.voff[1] to get t=0 variables:")
idx_t0 = layout.voff[1] .+ (1:layout.ny_)
println("  Indices for t=0: ", idx_t0)
Y_t0 = sep_sol.Y[idx_t0]
for i in 1:min(5, layout.ny_)
    @printf("  Y[%2d] = %12.8f  (var: %s)\n", idx_t0[i], Y_t0[i], FS2000.var[i])
end
println("  ...")
@printf("  Y[%2d] = %12.8f  (var: %s)\n", idx_t0[18], Y_t0[18], FS2000.var[18])

println("\n5. Comparing to deterministic SS:")
SS = get_steady_state(FS2000)
ss_vars = axiskeys(SS, 1)
ss_cols = axiskeys(SS, 2)
col_idx = findfirst(==(:Steady_state), ss_cols)
y_row = findfirst(==(:y), ss_vars)
y_ss = SS[y_row, col_idx]
println("  Deterministic SS for y: ", y_ss)
println("  SEP Y[18] (if using 1:ny_): ", Y_test1[18])
println("  SEP Y at t=0 for position 18: ", Y_t0[18])

println("\n6. CONCLUSION:")
if abs(Y_t0[18] - y_ss) < 1e-6
    println("  ✓ Using voff[1] + (1:ny_) gives CORRECT values matching deterministic SS")
    println("  ✗ Simply using Y[1:ny_] is WRONG!")
elseif abs(Y_test1[18] - y_ss) < 1e-6
    println("  ✓ Using Y[1:ny_] directly gives correct values")
else
    println("  ✗ Neither method matches - deeper investigation needed")
end

println("="^70)
