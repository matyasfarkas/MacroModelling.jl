# Diagnostic: Check SEP Shock Scaling
# Compare shock values in SEP tree vs expected values

using MacroModelling
using LinearAlgebra
using Printf

println("="^70)
println("SEP SHOCK SCALING DIAGNOSTIC")
println("="^70)

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

# Get shock std dev
shock_idx = findfirst(==(:epinf), m.exo)
param_idx = findfirst(==(:z_epinf), m.parameters)
σ_shock = m.parameter_values[param_idx]

println("\n1. Shock parameters:")
println("   Shock: epinf (index $shock_idx)")
println("   σ_shock: $σ_shock")
println("   Expected GH node values: [", -√3*σ_shock, ", 0, ", √3*σ_shock, "]")

# Solve SEP with minimal settings
println("\n2. Solving SEP with T=5, L=1, nnodes=3...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 5,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 50,
       silent = true)

println("   ✓ SEP solved")

# Check the solution structure
sep_sol = m.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout
Y = sep_sol.Y

println("\n3. SEP solution structure:")
println("   Periods (T): ", sep_sol.periods)
println("   Order (Lbr): ", sep_sol.order)
println("   Nodes (nnodes): ", sep_sol.nnodes)
println("   Total groups (K): ", layout.K)
println("   Number of shocks: ", layout.dε)
println("   Solution vector size: ", length(Y))

# Extract steady state
yss_indices = 1:layout.ny_
yss = Y[yss_indices]

# For nnodes=3, compute group index for positive epinf shock
nshocks = length(m.exo)
nnodes = 3
zero_node = 2  # Middle node
pos_node = 3   # Highest node

# Group index where epinf is at pos_node, others at zero_node
idx = 0
for d in 1:nshocks
    global idx  # Need global in script scope
    if d == shock_idx
        idx += (pos_node - 1) * nnodes^(d - 1)
    else
        idx += (zero_node - 1) * nnodes^(d - 1)
    end
end
shocked_group = idx + 1

println("\n4. Group indices:")
println("   Zero shock group: 1")
println("   epinf +√3σ shock group: $shocked_group")

# Extract states at t=1 for both groups
baseline_indices = (layout.voff[2] + 1):(layout.voff[2] + layout.ny_)
shocked_indices = (layout.voff[2] + (shocked_group-1)*layout.ny_ + 1):(layout.voff[2] + shocked_group*layout.ny_)

y_baseline_t1 = Y[baseline_indices]
y_shocked_t1 = Y[shocked_indices]

# Compare a few key variables
test_vars = [:y, :c, :pinf]
println("\n5. Responses at t=1 (for +√3σ shock in SEP tree):")
println("   Variable      Baseline      Shocked       Diff        Diff/σ")
println("   " * "-"^65)

for var in test_vars
    var_idx = findfirst(==(var), m.var)
    baseline_val = y_baseline_t1[var_idx]
    shocked_val = y_shocked_t1[var_idx]
    diff = shocked_val - baseline_val
    diff_per_sigma = diff / σ_shock  # Response per 1σ shock

    @printf("   %-12s  %10.6f  %10.6f  %10.6f  %10.2f\n",
            string(var), baseline_val, shocked_val, diff, diff_per_sigma)
end

# Now get perturbation IRF for comparison
println("\n6. Computing 1st-order perturbation IRF for comparison...")
solve!(m, algorithm=:first_order, silent=true)
pert_irf = get_irf(m, shocks=:epinf, periods=5)

println("\n7. Perturbation IRF at t=1 (for 1σ shock):")
println("   Variable      IRF(t=1)")
println("   " * "-"^30)

for var in test_vars
    var_idx_irf = findfirst(==(var), axiskeys(pert_irf, 1))
    if !isnothing(var_idx_irf)
        irf_val = pert_irf[var_idx_irf, 2, 1]  # t=1 (index 2), shock 1
        @printf("   %-12s  %10.6f\n", string(var), irf_val)
    end
end

println("\n8. Expected scaling:")
println("   SEP tree has shock magnitude: √3 * σ = ", √3 * σ_shock)
println("   Perturbation has shock magnitude: 1 * σ = ", σ_shock)
println("   Ratio: ", √3)
println()
println("   If SEP and perturbation agree, we should have:")
println("   SEP_response / √3 ≈ Pert_response")
println()

# Check the ratio
println("9. SCALING CHECK:")
println("   Variable      SEP/√3        Pert          Ratio")
println("   " * "-"^55)

for var in test_vars
    var_idx = findfirst(==(var), m.var)
    var_idx_irf = findfirst(==(var), axiskeys(pert_irf, 1))

    if !isnothing(var_idx_irf)
        shocked_val = y_shocked_t1[var_idx]
        baseline_val = y_baseline_t1[var_idx]
        sep_resp = (shocked_val - baseline_val) / √3  # Scale down by √3
        pert_resp = pert_irf[var_idx_irf, 2, 1]
        ratio = sep_resp / pert_resp

        @printf("   %-12s  %10.6f  %10.6f  %10.2f\n",
                string(var), sep_resp, pert_resp, ratio)
    end
end

println("\n" * "="^70)
println("INTERPRETATION:")
println("="^70)
println("If ratio ≈ 1.0: SEP scaling is CORRECT")
println("If ratio ≈ 100: SEP has 100x amplification bug")
println("If ratio ≈ 0.01: SEP scaling is inverted")
println("="^70)
