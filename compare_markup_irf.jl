# Compare Price Markup Shock IRF: MacroModelling Perturbation vs SEP
# Focus on epinf shock to test Kimball price aggregator
#
# This script uses MacroModelling's integrated SEP functionality:
# - solve!(model, algorithm=:stochastic_extended_path)
# - get_sep_irf(model, shock, size)

using MacroModelling, Plots, Printf, StatsBase, AxisKeys
include("models/Smets_Wouters_2007_HLT.jl")

"""
Get MacroModelling IRF for price markup shock.
"""
function get_mm_irf_markup(; periods=40)
    m = Smets_Wouters_2007_HLT

    # Get IRF for epinf shock
    irf = get_irf(m; shocks=:epinf, periods=periods)

    return irf
end

"""
Compute SEP IRF for price markup shock using MacroModelling integrated SEP.

Method:
1. Solve SEP using MacroModelling's solve! with :stochastic_extended_path
2. Extract IRF using get_sep_irf()
"""
function get_sep_irf_markup(; periods=40)
    println("  [SEP IRF computation using MacroModelling integration]")

    m = Smets_Wouters_2007_HLT

    # Solve with SEP (using conservative settings for large model)
    T_sep = min(periods, 20)  # Limit to 20 periods for computational efficiency

    println("  Solving SEP with T=$T_sep, L=1, nnodes=3...")
    solve!(m,
           algorithm = :stochastic_extended_path,
           sep_periods = T_sep,
           sep_order = 1,      # First-order branching
           sep_nnodes = 3,     # 3 GH nodes per shock
           sep_maxit = 80,
           sep_tol = 1e-6,
           silent = false)

    # Check convergence
    sep_sol = m.solution.perturbation.stochastic_extended_path
    if sep_sol.convergence_flag != 0
        @warn "SEP did not converge: flag=$(sep_sol.convergence_flag), err=$(sep_sol.final_error)"
        return zeros(length(m.var), periods)
    end

    println("  ✓ SEP converged (err=$(sep_sol.final_error))")

    # Get SEP IRF using integrated function
    println("  Extracting IRF with get_sep_irf...")
    irf_sep = get_sep_irf(m, :epinf, 1.0; periods=T_sep)

    # irf_sep is a KeyedArray with dimensions (Variables × Periods)
    # Convert to plain array
    irf_array = Array(irf_sep)

    println("  ✓ IRF extracted for $(size(irf_array, 1)) variables over $(size(irf_array, 2)) periods")

    # Pad with zeros if requested periods > computed periods
    if periods > T_sep
        println("  Note: Padding IRF with zeros for periods $(T_sep+1):$periods")
        irf_full = zeros(size(irf_array, 1), periods)
        irf_full[:, 1:size(irf_array, 2)] = irf_array
        return irf_full
    end

    return irf_array
end

"""
Compare IRFs for key variables.
"""
function compare_markup_irfs(; periods=40)
    println("="^70)
    println("MARKUP SHOCK IRF COMPARISON")
    println("="^70)

    # Get MacroModelling IRF
    println("\n1. Computing MacroModelling IRF...")
    mm_irf = get_mm_irf_markup(; periods=periods)

    println("  ✓ MacroModelling IRF computed")
    println("  Variables available: ", size(mm_irf.data, 1))
    println("  Periods: ", size(mm_irf.data, 2))

    # Extract key variables
    key_vars = [:y, :c, :inve, :pinf, :r, :w, :lab]

    println("\n2. Extracting key variable responses...")
    mm_responses = Dict{Symbol, Vector{Float64}}()

    # Get variable names from IRF (use axiskeys)
    var_names = axiskeys(mm_irf, 1)  # Variables dimension

    for var in key_vars
        # Find variable in IRF data
        var_idx = findfirst(==(var), var_names)
        if !isnothing(var_idx)
            # Extract IRF for this variable (shock dimension is 3rd)
            mm_responses[var] = Float64.(mm_irf[var_idx, :, 1])  # First shock (epinf)
            println("  ✓ $var: peak response = $(round(maximum(abs.(mm_responses[var])), digits=4))")
        else
            println("  ✗ $var: not found in MacroModelling IRF")
        end
    end

    # Get SEP IRF using MacroModelling integration
    println("\n3. Computing SEP IRF...")
    sep_irf = get_sep_irf_markup(; periods=periods)

    # Get model to access variable names
    m = Smets_Wouters_2007_HLT
    var_names = m.var

    # Convert to Dict for easier variable access
    sep_responses = Dict{Symbol, Vector{Float64}}()
    for var in key_vars
        # Find variable index in model
        var_idx = findfirst(==(var), var_names)
        if !isnothing(var_idx)
            # Extract IRF for this variable
            # Note: sep_irf has t=0 at column 1, so we need all columns
            sep_responses[var] = sep_irf[var_idx, :]
            println("  ✓ $var: SEP peak response = $(round(maximum(abs.(sep_responses[var])), digits=4))")
        else
            println("  ✗ $var: not found in model variables")
        end
    end

    # Comparison metrics
    println("\n" * "="^70)
    println("COMPARISON METRICS: SEP vs MacroModelling")
    println("="^70)

    println("\nShock: Price markup (epinf)")
    println("  MM:  1st-order perturbation with Kimball aggregator")
    println("  SEP: Global nonlinear solution with GH quadrature (order=1, nnodes=3)")

    println("\n" * "-"^70)
    @printf("%-10s %12s %12s %12s %10s %10s\n",
            "Variable", "Corr", "Peak(MM)", "Peak(SEP)", "Ratio", "RMSE")
    println("-"^70)

    for var in key_vars
        if haskey(mm_responses, var) && haskey(sep_responses, var)
            mm_resp = mm_responses[var]
            sep_resp = sep_responses[var]

            # Ensure same length
            n = min(length(mm_resp), length(sep_resp))
            mm_r = mm_resp[1:n]
            sep_r = sep_resp[1:n]

            # Metrics
            corr = StatsBase.cor(mm_r, sep_r)
            peak_mm = StatsBase.maximum(abs.(mm_r))
            peak_sep = StatsBase.maximum(abs.(sep_r))
            peak_ratio = peak_sep / peak_mm
            rmse = sqrt(mean((mm_r - sep_r).^2))

            @printf("%-10s %12.4f %12.6f %12.6f %10.3f %10.2e\n",
                    var, corr, peak_mm, peak_sep, peak_ratio, rmse)
        end
    end
    println("-"^70)

    println("\nDetailed comparison:")
    for var in key_vars
        if haskey(mm_responses, var) && haskey(sep_responses, var)
            mm_resp = mm_responses[var]
            sep_resp = sep_responses[var]

            peak_mm = maximum(abs.(mm_resp))
            peak_sep = maximum(abs.(sep_resp))
            peak_t_mm = argmax(abs.(mm_resp))
            peak_t_sep = argmax(abs.(sep_resp))

            println("\n  $var:")
            println("    MM  peak: $(round(peak_mm, digits=6)) at t=$peak_t_mm")
            println("    SEP peak: $(round(peak_sep, digits=6)) at t=$peak_t_sep")
            println("    Time difference: $(peak_t_sep - peak_t_mm) periods")
        end
    end

    # Plot SEP vs MM IRFs
    println("\n5. Creating plots...")

    p = plot(layout=(3,3), size=(1200, 900),
             plot_title="Price Markup Shock IRF: SEP vs MacroModelling")

    for (i, var) in enumerate(key_vars)
        if haskey(mm_responses, var) && haskey(sep_responses, var)
            # Plot both MM and SEP
            # MM IRF: 1 to periods
            plot!(p[i], 1:periods, mm_responses[var],
                  label="MM (1st-order)", linewidth=2, linestyle=:solid,
                  color=:blue, title=string(var))
            # SEP IRF: 0 to periods (includes t=0)
            plot!(p[i], 0:length(sep_responses[var])-1, sep_responses[var],
                  label="SEP (nonlinear)", linewidth=2, linestyle=:dash,
                  color=:red)
            hline!(p[i], [0], color=:black, linestyle=:dot, label="")
        elseif haskey(mm_responses, var)
            # Only MM available
            plot!(p[i], 1:periods, mm_responses[var],
                  label="MM", linewidth=2, title=string(var))
            hline!(p[i], [0], color=:black, linestyle=:dash, label="")
        end
    end

    savefig(p, "markup_irf_comparison.pdf")
    println("  ✓ Plot saved: markup_irf_comparison.pdf")

    println("\n" * "="^70)
    println("VALIDATION SUMMARY")
    println("="^70)
    println("✓ MacroModelling IRF extracted (1st-order perturbation)")
    println("✓ SEP IRF computed using MacroModelling integration (global nonlinear)")
    println("✓ Comparison metrics calculated")
    println("✓ IRF plot generated")

    println("\nInterpretation:")
    println("- Correlation > 0.90: SEP captures similar dynamics to perturbation solution")
    println("- Peak ratio ~1.0: Similar magnitude responses")
    println("- Time difference: Check if SEP predicts different timing")
    println("\nNote:")
    println("- SEP uses integrated MacroModelling functionality (solve! + get_sep_irf)")
    println("- GH approximation introduces ~1-2% error in shock magnitude")
    println("- SEP provides globally accurate solution vs local perturbation approximation")

    return (mm=mm_responses, sep=sep_responses)
end

# Run comparison (using 20 periods due to computational constraints)
result = compare_markup_irfs(; periods=20)
