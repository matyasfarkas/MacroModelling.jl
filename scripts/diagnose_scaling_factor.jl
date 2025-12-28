# Diagnostic: Find the source of the 100x discrepancy
using MacroModelling

include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT

println("="^70)
println("DIAGNOSING THE 100x SCALING ISSUE")
println("="^70)

# Get shock standard deviation from model
shock_name = :epinf
param_name = Symbol("z_", shock_name)
param_idx = findfirst(==(param_name), m.parameters)
σ_epinf = m.parameter_values[param_idx]

println("\n1. SHOCK STANDARD DEVIATION:")
println("   σ_epinf (from model) = $σ_epinf")
println("   σ_epinf² = $(σ_epinf^2)")

# Solve and get SEP solution details
println("\n2. SOLVING SEP...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = 10,
       sep_order = 1,
       sep_nnodes = 3,
       sep_maxit = 100,
       silent = true)

sep_sol = m.solution.perturbation.stochastic_extended_path
X = sep_sol.X
W = sep_sol.W
layout = sep_sol.layout

# Find epinf shock index
shock_idx = findfirst(==(shock_name), m.exo)
println("\n3. SHOCK MATRIX X:")
println("   Shock index for :epinf = $shock_idx")
println("   X dimensions: $(size(X))")
println("   Number of shocks: $(layout.dε)")
println("   K = nnodes^nshocks = 3^$(layout.dε) = $(3^layout.dε)")

# Check shock values in X
println("\n4. SHOCK VALUES IN X (for :epinf, shock index $shock_idx):")
println("   X[$shock_idx, 1] = $(X[shock_idx, 1])")
println("   X[$shock_idx, 2] = $(X[shock_idx, 2])")
println("   X[$shock_idx, 3] = $(X[shock_idx, 3])")
println("   X[$shock_idx, 1094] (middle) = $(X[shock_idx, 1094])")

# Check if X is scaled by σ
println("\n5. SCALING ANALYSIS:")
println("   If X is in units of σ:")
println("     X[$shock_idx, 3] / σ_epinf = $(X[shock_idx, 3] / σ_epinf)")
println("   Expected GH node at +√3 = $(√3)")

# Check perturbation vs SEP IRFs
println("\n6. IRF COMPARISON:")
solve!(m, algorithm=:first_order, silent=true)
pert_irf = get_irf(m, shocks=shock_name, periods=10)
y_pert = pert_irf(:y, :, shock_name)

solve!(m, algorithm=:stochastic_extended_path, sep_periods=10,
       sep_order=1, sep_nnodes=3, sep_maxit=100, silent=true)
sep_irf = MacroModelling.get_sep_irf(m, shock_name, 1.0, periods=10)
y_idx = findfirst(==(:y), axiskeys(sep_irf, 1))
y_sep = sep_irf[y_idx, :]

println("   Perturbation impact (t=1): $(y_pert[2])")
println("   SEP impact (t=1): $(y_sep[2])")
println("   Ratio (SEP/Pert): $(y_sep[2]/y_pert[2])")
println("   Factor needed: $(y_pert[2]/y_sep[2])")

println("\n7. HYPOTHESIS:")
factor = y_pert[2]/y_sep[2]
println("   Current ratio is ~$(round(factor, digits=1))x too small")
println("   Possible causes:")
println("   - Shock covariance not properly applied?")
println("   - Double scaling somewhere?")
println("   - Weight normalization issue?")
println("="^70)
