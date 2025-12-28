# Test simulation-based SEP IRF vs Perturbation IRF
using MacroModelling
using Statistics
using AxisKeys: dimnames

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("TESTING SIMULATION-BASED SEP IRF")
println("="^70)

shock_name = :eps_r  # Monetary policy shock
irf_horizon = 20

# Solve first-order perturbation for comparison
println("\n1. Solving first-order perturbation...")
solve!(m, algorithm=:first_order, silent=true)
pert_irf = get_irf(m, shocks=shock_name, periods=irf_horizon)
println("   ✓ Perturbation IRF complete")

# Solve SEP
println("\n2. Solving SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 40,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = true)
println("   ✓ SEP solution complete")

# Compute simulation-based SEP IRF
println("\n3. Computing simulation-based SEP IRF...")
println("   (This involves burn-in + 2 simulations, may take a moment...)")
sep_irf = get_sep_irf(m, shock_name, 1.0,
                      periods=irf_horizon,
                      burn_in=100,
                      random_seed=123,
                      silent=false)  # Show progress
println("   ✓ SEP IRF complete")

# Compare IRFs for key variables
println("\n4. COMPARISON:")
println("="^70)

test_vars = [:y, :c, :labobs, :pinfobs]

for var in test_vars
    println("\n   Variable: $var")
    println("   " * "-"^60)

    # Get perturbation IRF
    y_idx_pert = findfirst(==(:Variables), dimnames(pert_irf))
    var_idx_pert = findfirst(==(var), axiskeys(pert_irf, y_idx_pert))
    y_pert = pert_irf[var_idx_pert, :, shock_name]

    # Get SEP IRF
    var_idx_sep = findfirst(==(var), axiskeys(sep_irf, 1))
    y_sep = sep_irf[var_idx_sep, :]

    # Compare
    impact_pert = y_pert[2]  # t=1 (index 2 because pert includes t=0)
    impact_sep = y_sep[2]     # t=1

    println("   Perturbation impact (t=1): $(round(impact_pert, sigdigits=6))")
    println("   SEP impact (t=1):          $(round(impact_sep, sigdigits=6))")
    println("   Ratio (SEP/Pert):          $(round(impact_sep/impact_pert, digits=3))")

    # Compute correlation over horizon
    # Match dimensions: pert_irf has t=0, sep_irf has t=0
    min_len = min(length(y_pert), length(y_sep))
    corr = cor(y_pert[1:min_len], y_sep[1:min_len])
    println("   Correlation:               $(round(corr, digits=4))")
end

println("\n" * "="^70)
println("INTERPRETATION:")
println("="^70)
println("""
For nonlinear models (SEP), IRFs are path-dependent and computed as:
  IRF(t) = E[y_t | shock at t=0, starting from SSS] - E[y_t | no shock, from SSS]

For linear models (perturbation), IRFs are path-independent:
  IRF(t) = deviation from deterministic steady state

Key expectations:
  - Magnitudes may differ due to nonlinearity and SSS != deterministic SS
  - Sign and shape should be similar for mild nonlinearities
  - Correlation should be high (>0.9) if SEP approximates linearization well
  - Lower correlation indicates significant nonlinear effects
""")

println("="^70)
println("TEST COMPLETE")
println("="^70)
