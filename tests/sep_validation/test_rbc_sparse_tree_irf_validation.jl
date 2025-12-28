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
pos_path = joinpath(@__DIR__, "SEP", "RBC_irf_pos3.csv")
df_pos = CSV.read(pos_path, DataFrame, header=false)
n_periods_pos = size(df_pos, 1)
println("  Positive shock (+3σ): $n_periods_pos periods")

# Load negative shock benchmark
neg_path = joinpath(@__DIR__, "SEP", "RBC_irf_neg3.csv")
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
include(joinpath(@__DIR__, "..", "..", "models", "RBC_Dynare.jl"))

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
shock_magnitude_pos = 3.0   # +3σ (innovations(1) = 3)
shock_magnitude_neg = -3.0  # -3σ (innovations(1) = -3)
dynare_horizon = 400  # options_.ep.periods in rbc.mod
total_periods = 60  # Match Dynare output length

println("  Configuration:")
println("    Max branching order: $maxorder")
println("    Shock magnitude (+): $(shock_magnitude_pos) (innovations)")
println("    Shock magnitude (-): $(shock_magnitude_neg) (innovations)")
println("    SEP horizon: $dynare_horizon")
println("    Total periods: $total_periods")
println("    Sparse tree: fishbone algorithm")

# Shock index (epsilon shock)
shock_idx = findfirst(==(Symbol("epsilon")), RBC_Dynare.exo)

function shock_std_param(model, shock_sym)
    param_name = Symbol("z_", shock_sym)
    param_idx = findfirst(==(param_name), model.parameters)
    if param_idx !== nothing
        return model.parameter_values[param_idx], param_name
    end
    return 1.0, nothing  # Dynare shocks block default
end

shock_std, shock_std_name = shock_std_param(RBC_Dynare, Symbol("epsilon"))
if isnothing(shock_std_name)
    println("  Shock std: default 1.0 (no z_epsilon parameter found)")
else
    println("  Shock std: $(shock_std) (from parameter $(shock_std_name))")
end
println("  Applied shock (+): $(shock_magnitude_pos * shock_std)")
println("  Applied shock (-): $(shock_magnitude_neg * shock_std)")

function extract_path_matrix(sep_sol, layout, var_names, total_periods)
    path = zeros(total_periods, length(var_names))
    for t_idx in 1:total_periods
        t = t_idx - 1  # t=0 is steady state
        y_t = sep_sol.Y[layout.voff[t+1] .+ (1:layout.ny_)]
        for (i, var) in enumerate(var_names)
            var_idx = findfirst(==(Symbol(var)), RBC_Dynare.var)
            path[t_idx, i] = y_t[var_idx]
        end
    end
    return path
end

function funnel_baseline(maxorder, total_periods, shock_value, dss, var_names, sep_horizon)
    ny = length(RBC_Dynare.var)
    ds = zeros(1, ny)
    ds[1, :] .= dss
    prev = copy(dss)

    for order in maxorder:-1:0
        if order == maxorder
            shock_sequence = zeros(sep_horizon, 1)
            shock_sequence[1, 1] = shock_value
            periods = 1
        elseif order == 0
            shock_sequence = zeros(sep_horizon, 1)
            periods = total_periods
        else
            shock_sequence = zeros(sep_horizon, 1)
            periods = 1
        end

        if order == 0
            initial_guess = repeat(dss, sep_horizon + 1)
            initial_guess[1:ny] .= prev
        else
            initial_guess = nothing
        end

        solve!(RBC_Dynare,
               algorithm=:stochastic_extended_path,
               sep_periods=sep_horizon,
               sep_order=order,
               sep_nnodes=3,
               sep_sparse_tree=true,
               sep_initial_guess=initial_guess,
               sep_initial_state=prev,
               sep_deterministic_shocks=shock_sequence)

        sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
        layout = sep_sol.layout

        if order == 0
            for t in 1:periods
                y_t = sep_sol.Y[layout.voff[t+1] .+ (1:layout.ny_)]
                ds = vcat(ds, transpose(y_t))
                prev = y_t
            end
        else
            y_t = sep_sol.Y[layout.voff[2] .+ (1:layout.ny_)]
            ds = vcat(ds, transpose(y_t))
            prev = y_t
        end
    end

    ts = zeros(total_periods, length(var_names))
    for (i, var) in enumerate(var_names)
        var_idx = findfirst(==(Symbol(var)), RBC_Dynare.var)
        ts[:, i] = ds[1:total_periods, var_idx]
    end
    return ts
end

function run_shock_case(shock_value, dynare_tt, dynare_ts, label)
    println("\n4. Computing $label shock IRFs...")
    println("-"^80)

    shock_sequence = zeros(dynare_horizon, 1)
    shock_sequence[1, 1] = shock_value * shock_std

    println("  Solving shocked path (tt) with order=$maxorder...")
    solve!(RBC_Dynare,
           algorithm=:stochastic_extended_path,
           sep_periods=dynare_horizon,
           sep_order=maxorder,
           sep_nnodes=3,
           sep_sparse_tree=true,
           sep_initial_state=dss,
           sep_deterministic_shocks=shock_sequence)

    sep_sol = RBC_Dynare.solution.perturbation.stochastic_extended_path
    layout = sep_sol.layout

    tt_path = extract_path_matrix(sep_sol, layout, var_names, total_periods)
    println("  Sample tt path values (period 1):")
    @printf("    Output: %.6f, Capital: %.6f\n", tt_path[1, 2], tt_path[1, 1])

    println("\n  Building funnel baseline (ts)...")
    ts_path = funnel_baseline(maxorder, total_periods, shock_value * shock_std, dss, var_names, dynare_horizon)

    println("\n5. Computing percentage deviations from initial state...")
    println("-"^80)

    tt_pct = pdss(tt_path)
    ts_pct = pdss(ts_path)

    println("  MacroModelling.jl tt path (% deviation, period 1):")
    for (i, var) in enumerate(var_names)
        @printf("    %-15s  %9.4f%%\n", var, tt_pct[1, i])
    end

    println("\n  Dynare tt path (% deviation, period 1):")
    for (i, var) in enumerate(var_names)
        @printf("    %-15s  %9.4f%%\n", var, dynare_tt[1, i])
    end

    println("\n6. Comparing MacroModelling.jl vs Dynare (tt paths only)...")
    println("-"^80)

    println("\nPeriod | Variable      | MM tt (%) | Dynare tt (%) |  Diff (%) | Rel Error")
    println("-"^80)

    for t in 1:min(20, total_periods)
        for (i, var) in enumerate(var_names)
            mm_val = tt_pct[t, i]
            dynare_val = dynare_tt[t, i]
            diff = mm_val - dynare_val

            rel_err = abs(dynare_val) > 1e-6 ? abs(diff / dynare_val) : abs(diff)

            if t <= 5 || (t <= 10 && i == 2)
                @printf("%6d | %-13s | %9.4f | %13.4f | %9.6f | %9.2e\n",
                        t, var, mm_val, dynare_val, diff, rel_err)
            end
        end
    end

    println("\n7. Summary statistics (Output variable, tt only)...")
    println("-"^80)

    output_idx = 2
    mm_output = tt_pct[:, output_idx]
    dynare_output = dynare_tt[:, output_idx]

    abs_errors = abs.(mm_output - dynare_output)
    rel_errors = abs_errors ./ (abs.(dynare_output) .+ 1e-10)

    @printf("  Max absolute error: %.6f%%\n", maximum(abs_errors))
    @printf("  Mean absolute error: %.6f%%\n", mean(abs_errors))
    @printf("  Max relative error: %.6e\n", maximum(rel_errors))
    @printf("  Mean relative error: %.6e\n", mean(rel_errors))

    println("\n8. Comparing MacroModelling.jl vs Dynare (ts paths only)...")
    println("-"^80)

    println("\nPeriod | Variable      | MM ts (%) | Dynare ts (%) |  Diff (%) | Rel Error")
    println("-"^80)

    for t in 1:min(20, total_periods)
        for (i, var) in enumerate(var_names)
            mm_val = ts_pct[t, i]
            dynare_val = dynare_ts[t, i]
            diff = mm_val - dynare_val

            rel_err = abs(dynare_val) > 1e-6 ? abs(diff / dynare_val) : abs(diff)

            if t <= 5 || (t <= 10 && i == 2)
                @printf("%6d | %-13s | %9.4f | %13.4f | %9.6f | %9.2e\n",
                        t, var, mm_val, dynare_val, diff, rel_err)
            end
        end
    end

    println("\n9. Summary statistics (Output variable, ts only)...")
    println("-"^80)

    mm_output_ts = ts_pct[:, output_idx]
    dynare_output_ts = dynare_ts[:, output_idx]

    abs_errors_ts = abs.(mm_output_ts - dynare_output_ts)
    rel_errors_ts = abs_errors_ts ./ (abs.(dynare_output_ts) .+ 1e-10)

    @printf("  Max absolute error: %.6f%%\n", maximum(abs_errors_ts))
    @printf("  Mean absolute error: %.6f%%\n", mean(abs_errors_ts))
    @printf("  Max relative error: %.6e\n", maximum(rel_errors_ts))
    @printf("  Mean relative error: %.6e\n", mean(rel_errors_ts))

    irf_pct = tt_pct - ts_pct
    dynare_irf = dynare_tt - dynare_ts

    println("\n10. Summary statistics (Output variable, IRF = tt - ts)...")
    println("-"^80)

    mm_output_irf = irf_pct[:, output_idx]
    dynare_output_irf = dynare_irf[:, output_idx]

    abs_errors_irf = abs.(mm_output_irf - dynare_output_irf)
    rel_errors_irf = abs_errors_irf ./ (abs.(dynare_output_irf) .+ 1e-10)

    @printf("  Max absolute error: %.6f%%\n", maximum(abs_errors_irf))
    @printf("  Mean absolute error: %.6f%%\n", mean(abs_errors_irf))
    @printf("  Max relative error: %.6e\n", maximum(rel_errors_irf))
    @printf("  Mean relative error: %.6e\n", mean(rel_errors_irf))

    return tt_pct, ts_pct
end

println("\n5. Computing percentage deviations from initial state...")
println("-"^80)

# Apply pdss transformation: 100*(data/data[1] - 1)
function pdss(data::Matrix{Float64})
    return 100.0 .* (data ./ data[1:1, :] .- 1.0)
end

tt_pos_pct, ts_pos_pct = run_shock_case(shock_magnitude_pos, dynare_pos_tt, dynare_pos_ts, "POSITIVE (+3σ)")
tt_neg_pct, ts_neg_pct = run_shock_case(shock_magnitude_neg, dynare_neg_tt, dynare_neg_ts, "NEGATIVE (-3σ)")

println("\n" * "="^80)
println("NEXT STEPS")
println("="^80)
println("""
Next steps:

1. Compare IRF = pdss(tt) - pdss(ts) against Dynare CSVs.
2. Evaluate asymmetry: +3σ vs -3σ.
3. Decide whether to enforce nonlinear residuals in deterministic SEP.
""")

println("\n" * "="^80)
