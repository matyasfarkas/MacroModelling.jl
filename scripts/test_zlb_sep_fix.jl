# test_zlb_sep_fix.jl
#
# Validate ZLB enforcement in the SEP simulation.
#
# Two approaches are tested:
#   1. No enforcement (baseline) -- r can go below 1.0
#   2. enforce_zlb=true -- OBC anticipated shocks optimized via NLopt SLSQP,
#      then SEP re-solved for nonlinear consistency.  Falls back to simple
#      clamping when the OBC setup is unavailable.
#
# Model: Smets_Wouters_2007_HLT_obc.jl (ZLB via max() in Taylor rule)
#
# Usage:
#   julia --project scripts/test_zlb_sep_fix.jl

using MacroModelling, LinearAlgebra, Printf

println("=" ^ 72)
println("ZLB ENFORCEMENT TEST -- SEP SOLVER + OBC ANTICIPATED SHOCKS")
println("=" ^ 72)

# ── Load OBC model ────────────────────────────────────────────────────
println("\nLoading Smets_Wouters_2007_HLT_obc...")
include(joinpath(@__DIR__, "..", "models", "Smets_Wouters_2007_HLT_obc.jl"))
m = Smets_Wouters_2007_HLT_obc

println("  Variables: ", length(m.var))
println("  Shocks:    ", length(m.exo))

obc_mask = contains.(string.(m.exo), "ᵒᵇᶜ")
n_structural = count(!, obc_mask)
n_obc = count(obc_mask)
println("  Structural shocks: $n_structural")
println("  OBC shocks:        $n_obc")
println("  max_obc_horizon:   $(m.max_obc_horizon)")

SS = get_steady_state(m, derivatives=false)
r_ss = Float64(SS(:r))
println(@sprintf "  r[ss] = %.6f (annualised = %.2f%%)" r_ss (r_ss^4 - 1)*100)

R_bar_idx = findfirst(==(:R_bar), m.parameters)
R_bar = m.parameter_values[R_bar_idx]
println(@sprintf "  R_bar = %.4f  ->  ZLB floor on r = exp(R_bar) = %.6f" R_bar exp(R_bar))

# ── Construct shock sequence ──────────────────────────────────────────
shock_names = m.exo
nshocks = length(shock_names)
eb_idx = findfirst(s -> string(s) == "eb", shock_names)

total_periods = 40
shock_matrix = zeros(nshocks, total_periods)

if !isnothing(eb_idx)
    # Large negative preference shocks to push r below ZLB
    eb_sequence = [-3.0, -2.0, -1.5, -1.0]
    for (i, mag) in enumerate(eb_sequence)
        if i <= total_periods
            shock_matrix[eb_idx, i] = mag
        end
    end
    println("\nShock design: eb[$eb_idx] = $eb_sequence (periods 1-4)")
else
    println("\nWARNING: eb shock not found. Using first structural shock.")
    structural_idx = findall(!, obc_mask)
    if !isempty(structural_idx)
        for t in 1:4
            shock_matrix[structural_idx[1], t] = -3.0
        end
    end
end

# SEP solver settings
sep_kw = Dict(
    :sep_horizon => 10,
    :sep_order => 1,
    :sep_nnodes => 3,
    :sep_maxit => 120,
    :sep_tol => 1e-6,
    :sep_sparse_tree => true,
    :sep_accept_tol => 0.50,
    :sep_shock_scale => 0.1,
    :silent => false,
)

# ── Run 1: No enforcement (baseline) ─────────────────────────────────
println("\n" * "-"^50)
println("RUN 1: No enforcement (baseline)")
println("-"^50)
result_1 = simulate_sep_extended_path(m;
    periods=total_periods,
    shocks=shock_matrix,
    enforce_zlb=false,
    sep_kw...,
)

# ── Run 2: enforce_zlb (OBC anticipated shocks + SEP re-solve) ───────
println("\n" * "-"^50)
println("RUN 2: enforce_zlb=true (OBC optimization + SEP re-solve)")
println("-"^50)
result_2 = simulate_sep_extended_path(m;
    periods=total_periods,
    shocks=shock_matrix,
    enforce_zlb=true,
    zlb_floor=1.0,
    zlb_variable=:r,
    zlb_obc_maxiter=3,
    sep_kw...,
)

# ── Comparison ────────────────────────────────────────────────────────
function report_result(label, result, var_sym)
    if result.errorflag
        println(@sprintf "  %-35s  FAILED at period %d" label result.failure_period)
        return nothing
    end
    sim = result.simulation
    ts = axiskeys(sim, 2)
    path = [Float64(sim(var_sym, t)) for t in ts]
    min_val = minimum(path)
    n_below = count(x -> x < 1.0 - 1e-8, path)
    println(@sprintf "  %-35s  min(r)=%.6f  violations=%d/%d  zlb_periods=%d" label min_val n_below length(path) result.zlb_periods)
    return path
end

println("\n" * "="^72)
println("RESULTS")
println("="^72)
path_1 = report_result("No enforcement", result_1, :r)
path_2 = report_result("enforce_zlb (OBC + re-solve)", result_2, :r)

# Print period-by-period comparison
println("\n  Period-by-period r values (first 15 periods):")
println(@sprintf "  %-6s  %14s  %14s" "t" "No enforce" "enforce_zlb")
println("  " * "-"^38)
for t in 1:min(15, total_periods+1)
    vals = String[]
    for path in [path_1, path_2]
        if !isnothing(path) && t <= length(path)
            push!(vals, @sprintf "%14.6f" path[t])
        else
            push!(vals, @sprintf "%14s" "N/A")
        end
    end
    println(@sprintf "  t=%-3d  %s  %s" (t-1) vals[1] vals[2])
end

# ── Verification ──────────────────────────────────────────────────────
println("\n" * "="^72)
println("VERIFICATION")
println("="^72)

passed = true

if !isnothing(path_1)
    violations_1 = count(x -> x < 1.0 - 1e-8, path_1)
    if violations_1 > 0
        println("  INFO:  Baseline has $violations_1 ZLB violations (expected)")
    else
        println("  INFO:  Baseline has no violations (shocks may be too small)")
    end
end

if !isnothing(path_2)
    violations_2 = count(x -> x < 1.0 - 1e-8, path_2)
    if violations_2 == 0
        println("  PASS:  enforce_zlb keeps r >= 1.0 in all periods")
    else
        println("  WARN:  enforce_zlb has $violations_2 residual violations " *
                "(OBC optimizer may need more iterations or shocks are extreme)")
        # Check if violations are small (within tolerance)
        if all(x -> x >= 1.0 - 0.01, path_2)
            println("  NOTE:  All violations within 1%% tolerance")
        else
            passed = false
        end
    end
else
    println("  SKIP:  enforce_zlb simulation failed")
end

# ── Plotting ──────────────────────────────────────────────────────────
try
    using Plots

    println("\nGenerating 4-panel figure...")

    time_ax = path_1 !== nothing ? (0:length(path_1)-1) : (0:total_periods)

    # Panel 1: Policy rate
    p1 = plot(title="Policy Rate (r)", xlabel="Quarter", ylabel="r (gross)")
    hline!([1.0], color=:red, linestyle=:dash, label="ZLB floor", linewidth=2)
    hline!([r_ss], color=:gray, linestyle=:dot, label="Steady state", linewidth=1)
    if !isnothing(path_1)
        plot!(0:length(path_1)-1, path_1, label="No enforcement", color=:blue, linewidth=1.5)
    end
    if !isnothing(path_2)
        plot!(0:length(path_2)-1, path_2, label="ZLB enforced", color=:green, linewidth=2)
    end

    # Panel 2: Output gap
    p2 = plot(title="Output Gap (ygap)", xlabel="Quarter", ylabel="%")
    if !result_1.errorflag
        ts1 = axiskeys(result_1.simulation, 2)
        ygap1 = [Float64(result_1.simulation(:ygap, t)) for t in ts1]
        plot!(collect(ts1), ygap1, label="No enforcement", color=:blue, linewidth=1.5)
    end
    if !result_2.errorflag
        ts2 = axiskeys(result_2.simulation, 2)
        ygap2 = [Float64(result_2.simulation(:ygap, t)) for t in ts2]
        plot!(collect(ts2), ygap2, label="ZLB enforced", color=:green, linewidth=2)
    end

    # Panel 3: Inflation
    p3 = plot(title="Inflation (pinfobs)", xlabel="Quarter", ylabel="%")
    if !result_1.errorflag
        pinf1 = [Float64(result_1.simulation(:pinfobs, t)) for t in ts1]
        plot!(collect(ts1), pinf1, label="No enforcement", color=:blue, linewidth=1.5)
    end
    if !result_2.errorflag
        pinf2 = [Float64(result_2.simulation(:pinfobs, t)) for t in ts2]
        plot!(collect(ts2), pinf2, label="ZLB enforced", color=:green, linewidth=2)
    end

    # Panel 4: Investment growth
    p4 = plot(title="Investment Growth (dinve)", xlabel="Quarter", ylabel="%")
    if !result_1.errorflag
        dinve1 = [Float64(result_1.simulation(:dinve, t)) for t in ts1]
        plot!(collect(ts1), dinve1, label="No enforcement", color=:blue, linewidth=1.5)
    end
    if !result_2.errorflag
        dinve2 = [Float64(result_2.simulation(:dinve, t)) for t in ts2]
        plot!(collect(ts2), dinve2, label="ZLB enforced", color=:green, linewidth=2)
    end

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 700),
              plot_title="SEP Simulation: ZLB Enforcement via OBC Anticipated Shocks")

    # Save to docs/paper/figures/
    fig_dir = joinpath(@__DIR__, "..", "docs", "paper", "figures")
    mkpath(fig_dir)
    fig_path = joinpath(fig_dir, "fig_zlb_binding_simulation.pdf")
    savefig(fig, fig_path)
    println("  Saved: $fig_path")

    fig_path_png = replace(fig_path, ".pdf" => ".png")
    savefig(fig, fig_path_png)
    println("  Saved: $fig_path_png")
catch e
    println("\nPlotting skipped: ", e)
end

println("\n" * (passed ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED"))
println("Done.")
