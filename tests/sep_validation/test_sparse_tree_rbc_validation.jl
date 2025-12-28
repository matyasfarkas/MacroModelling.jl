# RBC Sparse Tree Validation Test
# Compares sparse tree vs full tree on RBC model
# Also prepares output for Dynare comparison

using MacroModelling
using Printf

println("="^80)
println("RBC MODEL: SPARSE TREE VS FULL TREE VALIDATION")
println("="^80)

# Load RBC model
include("models/RBC_Dynare.jl")

# Test parameters
test_periods = 10
test_order = 1
test_nnodes = 3

println("\nTest configuration:")
println("  Periods (T): $test_periods")
println("  Branching order (Lbr): $test_order")
println("  Nodes per dimension (m): $test_nnodes")
println("  Number of shocks (H): 1")

# Expected tree sizes
full_tree_nodes = test_nnodes^1  # m^H for 1 shock
sparse_tree_nodes = 1 + 1*(test_nnodes-1)  # 1 + H*(m-1) for 1 shock

println("\nExpected tree sizes:")
println("  Full tree: $full_tree_nodes nodes per branching period")
println("  Sparse tree: $sparse_tree_nodes nodes per branching period")
println("  Reduction factor: $(full_tree_nodes/sparse_tree_nodes)×")

println("\n" * "-"^80)
println("SOLVING WITH FULL TREE (tensor product)")
println("-"^80)

# Solve with full tree
solve!(RBC_Dynare,
       algorithm=:stochastic_extended_path,
       sep_periods=test_periods,
       sep_order=test_order,
       sep_nnodes=test_nnodes,
       sep_sparse_tree=false,
       silent=false)

# Extract solution
sep_full = RBC_Dynare.solution.perturbation.stochastic_extended_path
layout_full = sep_full.layout

println("\nFull tree solution:")
println("  Total Y vector length: $(length(sep_full.Y))")
println("  Groups per period: $(layout_full.G)")
println("  Tree structure verified: $(layout_full.sparse ? "SPARSE" : "FULL")")

# Extract steady state from full tree
yss_full = sep_full.Y[layout_full.voff[1] .+ (1:layout_full.ny_)]
var_names = RBC_Dynare.var

println("\nFull tree steady state (first 7 variables):")
for i in 1:min(7, length(var_names))
    @printf("  %-15s  %.8f\n", String(var_names[i]), yss_full[i])
end

println("\n" * "-"^80)
println("SOLVING WITH SPARSE TREE (fishbone monomial)")
println("-"^80)

# Solve with sparse tree
solve!(RBC_Dynare,
       algorithm=:stochastic_extended_path,
       sep_periods=test_periods,
       sep_order=test_order,
       sep_nnodes=test_nnodes,
       sep_sparse_tree=true,
       silent=false)

# Extract solution
sep_sparse = RBC_Dynare.solution.perturbation.stochastic_extended_path
layout_sparse = sep_sparse.layout

println("\nSparse tree solution:")
println("  Total Y vector length: $(length(sep_sparse.Y))")
println("  Groups per period: $(layout_sparse.G)")
println("  Tree structure verified: $(layout_sparse.sparse ? "SPARSE" : "FULL")")

# Extract steady state from sparse tree
yss_sparse = sep_sparse.Y[layout_sparse.voff[1] .+ (1:layout_sparse.ny_)]

println("\nSparse tree steady state (first 7 variables):")
for i in 1:min(7, length(var_names))
    @printf("  %-15s  %.8f\n", String(var_names[i]), yss_sparse[i])
end

println("\n" * "="^80)
println("COMPARISON: FULL vs SPARSE TREE")
println("="^80)

# Compute differences
max_abs_diff = maximum(abs.(yss_full - yss_sparse))
rel_diff = abs.((yss_full - yss_sparse) ./ (abs.(yss_full) .+ 1e-10))
max_rel_diff = maximum(rel_diff)

println("\nSteady state comparison:")
println(@sprintf("  Maximum absolute difference: %.2e", max_abs_diff))
println(@sprintf("  Maximum relative difference: %.2e", max_rel_diff))

println("\nDetailed comparison (all variables):")
println(@sprintf("  %-15s  %12s  %12s  %12s  %12s", "Variable", "Full Tree", "Sparse Tree", "Abs Diff", "Rel Diff"))
println("  " * "-"^75)
for i in 1:length(var_names)
    abs_diff = abs(yss_full[i] - yss_sparse[i])
    rel_diff_i = abs_diff / (abs(yss_full[i]) + 1e-10)
    @printf("  %-15s  %12.8f  %12.8f  %12.2e  %12.2e\n",
            String(var_names[i]), yss_full[i], yss_sparse[i], abs_diff, rel_diff_i)
end

# Test assertion
tolerance = 1e-6
if max_abs_diff < tolerance
    println("\n✅ TEST PASSED: Sparse tree matches full tree within tolerance ($tolerance)")
else
    println("\n❌ TEST FAILED: Difference exceeds tolerance")
    println("   Expected: < $tolerance")
    println("   Got: $max_abs_diff")
end

println("\n" * "="^80)
println("DYNARE COMPARISON REQUEST")
println("="^80)

println("\nTo validate against Dynare, please run the following in Matlab:")
println("""
% Load RBC model
dynare rbc.mod

% Run sparse tree SEP (using Adjemian-Juillard fishbone method)
options_.ep.stochastic.order = $test_order;
options_.ep.periods = $test_periods;
options_.ep.replic_nb = 1;
options_.ep.stochastic.algo = 3;  % Fishbone sparse tree

% Run extended_path
dr = extended_path([], $test_periods);

% Extract steady state
yss_dynare = oo_.endo_simul(:,1);

% Display key variables
fprintf('Dynare sparse tree steady state:\\n');
fprintf('  efficiency: %.8f\\n', yss_dynare(M_.endo_names == 'efficiency'));
fprintf('  Efficiency: %.8f\\n', yss_dynare(M_.endo_names == 'Efficiency'));
fprintf('  Output:     %.8f\\n', yss_dynare(M_.endo_names == 'Output'));
fprintf('  Capital:    %.8f\\n', yss_dynare(M_.endo_names == 'Capital'));
fprintf('  Consumption:%.8f\\n', yss_dynare(M_.endo_names == 'Consumption'));
fprintf('  Labour:     %.8f\\n', yss_dynare(M_.endo_names == 'Labour'));
fprintf('  Investment: %.8f\\n', yss_dynare(M_.endo_names == 'Investment'));

% Save for comparison
save('rbc_dynare_sparse_sep.mat', 'yss_dynare', 'oo_', 'M_');
""")

println("\nExpected MacroModelling.jl values for comparison:")
for i in 1:min(7, length(var_names))
    @printf("  %-15s  %.8f\n", String(var_names[i]), yss_sparse[i])
end

println("\n" * "="^80)
println("PERFORMANCE COMPARISON")
println("="^80)

# Note: For RBC with 1 shock, the difference is minimal
# The real benefit appears with multiple shocks
println("\nNote: RBC has only 1 shock, so:")
println("  Full tree: $(full_tree_nodes) nodes")
println("  Sparse tree: $(sparse_tree_nodes) nodes")
println("  Reduction: $(full_tree_nodes/sparse_tree_nodes)× (minimal for 1 shock)")
println("\nFor models with H shocks and m=$test_nnodes nodes:")
println("  Full tree: m^H nodes")
println("  Sparse tree: 1 + H×(m-1) nodes")
println("\nExample with H=7 shocks (like SW07):")
println("  Full: 3^7 = 2,187 nodes")
println("  Sparse: 1 + 7×2 = 15 nodes")
println("  Reduction: 146×")

println("\n" * "="^80)
