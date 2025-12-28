# RBC Sparse Tree SEP IRF Validation
# Validates MacroModelling.jl sparse tree implementation against Dynare benchmark
#
# Methodology:
# 1. Load Dynare IRF benchmark data (already in % deviation format)
# 2. Run MacroModelling.jl SEP with sparse tree for +3σ and -3σ shocks
# 3. Construct stochastic funnel baseline (ts) with decreasing branching order
# 4. Compute IRFs: 100*(path/path[1] - 1) for both tt and ts
# 5. Compare with Dynare benchmark

using MacroModelling
using CSV
using DataFrames
using Printf
using Statistics

println("="^80)
println("RBC SPARSE TREE SEP IRF VALIDATION")
println("="^80)

# Variable names and order from Dynare
var_names = ["Capital", "Output", "Labour", "Consumption", "Efficiency", "efficiency", "Investment"]

println("\n1. Loading Dynare benchmark data...")
println("-"^80)

# Load positive shock benchmark
pos_path = "/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SEP/RBC_irf_pos3.csv"
df_pos = CSV.read(pos_path, DataFrame, header=false)
n_periods_pos = size(df_pos, 1)
println("  Positive shock (+3σ): $n_periods_pos periods")

# Load negative shock benchmark
neg_path = "/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SEP/RBC_irf_neg3.csv"
df_neg = CSV.read(neg_path, DataFrame, header=false)
n_periods_neg = size(df_neg, 1)
println("  Negative shock (-3σ): $n_periods_neg periods")

# Extract data (columns 1-7 = ts, columns 8-14 = tt)
dynare_pos_ts = Matrix(df_pos[:, 1:7])
dynare_pos_tt = Matrix(df_pos[:, 8:14])
dynare_neg_ts = Matrix(df_neg[:, 1:7])
dynare_neg_tt = Matrix(df_neg[:, 8:14])

println("\n2. Loading RBC model...")
println("-"^80)
include("models/RBC_Dynare.jl")

# Get deterministic steady state
dss = RBC_Dynare.solution.non_stochastic_steady_state
println("  Deterministic steady state:")
for (i, var) in enumerate(var_names)
    var_idx = findfirst(==(Symbol(var)), RBC_Dynare.var)
    if !isnothing(var_idx)
        @printf("    %-15s  %.8f\n", var, dss[var_idx])
    end
end

println("\n3. Running MacroModelling.jl SEP with sparse tree...")
println("-"^80)

# Configuration matching Dynare
maxorder = 10
sigma_epsilon = 0.1  # From rbc.mod
shock_magnitude_pos = 3.0   # +3σ
shock_magnitude_neg = -3.0  # -3σ
total_periods = 60  # Match Dynare output

println("  Configuration:")
println("    Max branching order: $maxorder")
println("    Shock magnitude (+): $(shock_magnitude_pos)σ = $(shock_magnitude_pos * sigma_epsilon)")
println("    Shock magnitude (-): $(shock_magnitude_neg)σ = $(shock_magnitude_neg * sigma_epsilon)")
println("    Total periods: $total_periods")
println("    Sparse tree: fishbone algorithm")

# Shock index (epsilon shock)
shock_idx = findfirst(==(Symbol("epsilon")), RBC_Dynare.exo)

println("\n4. Computing POSITIVE shock (+3σ) IRFs...")
println("-"^80)

# Create shock sequence: +3σ at t=1, zero elsewhere
shock_sequence_pos = zeros(total_periods, 1)  # 1 shock dimension (epsilon)
shock_sequence_pos[1, 1] = shock_magnitude_pos * sigma_epsilon  # +3σ in absolute terms

# Solve SEP for shocked path (tt) with +3σ shock at t=1
println("  Solving shocked path (tt) with order=$maxorder...")
solve!(RBC_Dynare,
       algorithm=:stochastic_extended_path,
       sep_periods=total_periods,
       sep_order=maxorder,
       sep_nnodes=3,
       sep_sparse_tree=true,  # Use sparse tree
       sep_deterministic_shocks=shock_sequence_pos)  # NEW parameter

# Extract SEP solution
sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout

# Extract tt path (shocked path with full branching order)
tt_pos = zeros(total_periods, length(var_names))
for t in 1:total_periods
    voff_t = layout.voff[t]
    y_t = sep_sol.Y[voff_t .+ (1:layout.ny_)]
    for (i, var) in enumerate(var_names)
        var_idx = findfirst(==(Symbol(var)), RBC_Dynare.var)
        tt_pos[t, i] = y_t[var_idx]
    end
end

println("  Sample tt path values (period 1):")
@printf("    Output: %.6f, Capital: %.6f\n", tt_pos[1, 2], tt_pos[1, 1])

# TODO: Implement stochastic funnel baseline (ts) construction
# This requires iterative SEP solving with decreasing order
# For now, we'll just compare the tt paths

println("\n5. Computing percentage deviations from initial state...")
println("-"^80)

# Apply pdss transformation: 100*(data/data[1] - 1)
function pdss(data::Matrix{Float64})
    return 100.0 .* (data ./ data[1:1, :] .- 1.0)
end

tt_pos_pct = pdss(tt_pos)

println("  MacroModelling.jl tt path (% deviation, period 1):")
for (i, var) in enumerate(var_names)
    @printf("    %-15s  %9.4f%%\n", var, tt_pos_pct[1, i])
end

println("\n  Dynare tt path (% deviation, period 1):")
for (i, var) in enumerate(var_names)
    @printf("    %-15s  %9.4f%%\n", var, dynare_pos_tt[1, i])
end

println("\n6. Comparing MacroModelling.jl vs Dynare (tt paths only)...")
println("-"^80)

println("\nPeriod | Variable      | MM tt (%) | Dynare tt (%) |  Diff (%) | Rel Error")
println("-"^80)

for t in 1:min(20, total_periods)
    for (i, var) in enumerate(var_names)
        mm_val = tt_pos_pct[t, i]
        dynare_val = dynare_pos_tt[t, i]
        diff = mm_val - dynare_val

        # Relative error (handle division by zero)
        rel_err = abs(dynare_val) > 1e-6 ? abs(diff / dynare_val) : abs(diff)

        if t <= 5 || (t <= 10 && i == 2)  # Show first 5 periods + Output for periods 6-10
            @printf("%6d | %-13s | %9.4f | %13.4f | %9.6f | %9.2e\n",
                    t, var, mm_val, dynare_val, diff, rel_err)
        end
    end
end

println("\n7. Summary statistics (Output variable)...")
println("-"^80)

output_idx = 2  # Output is second variable
mm_output = tt_pos_pct[:, output_idx]
dynare_output = dynare_pos_tt[:, output_idx]

abs_errors = abs.(mm_output - dynare_output)
rel_errors = abs_errors ./ (abs.(dynare_output) .+ 1e-10)

@printf("  Max absolute error: %.6f%%\n", maximum(abs_errors))
@printf("  Mean absolute error: %.6f%%\n", mean(abs_errors))
@printf("  Max relative error: %.6e\n", maximum(rel_errors))
@printf("  Mean relative error: %.6e\n", mean(rel_errors))

println("\n" * "="^80)
println("NEXT STEPS")
println("="^80)
println("""
To complete the validation:

1. Implement stochastic funnel baseline (ts) construction:
   - Iteratively solve SEP with decreasing order (10 → 0)
   - Use end state of each solve as initial state for next
   - Requires API extension: sep_initial_state parameter

2. Compute full IRF: IRF = pdss(tt) - pdss(ts)

3. Validate asymmetry: Compare +3σ vs -3σ shock responses

Current validation shows comparison of tt (shocked) paths only.
""")

println("\n" * "="^80)
