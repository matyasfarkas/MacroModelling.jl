using MacroModelling, StatsPlots          # StatsPlots triggers plotting.jl
include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT                # no parentheses

# First-order perturbation (default)
plot_irf(m; shocks = :epinf)              # or MacroModelling.plot_irf

# Second-order perturbation
plot_irf!(m,
    shocks = :epinf,
    algorithm = :second_order)

# Pruned third-order perturbation
plot_irf!(m,
    shocks = :epinf,
    algorithm = :pruned_third_order)

# Stochastic Extended Path (SEP) - Global nonlinear solution
println("\nSolving with SEP...")
solve!(m,
    algorithm = :stochastic_extended_path,
    sep_periods = 20,
    sep_order = 1,
    sep_nnodes = 3,
    silent = false)

# Get SEP IRF
println("Extracting SEP IRF...")
irf_sep = get_sep_irf(m, :epinf, 1.0; periods = 20)

# Add SEP IRF to the plot
# Get the variable names and plot for each variable
vars = m.var
for (i, v) in enumerate(vars)
    # Find the subplot for this variable and add SEP line
    plot!(irf_sep[i, :],
          label = "SEP",
          linewidth = 2,
          linestyle = :dash,
          subplot = i)
end

println("✓ SEP IRF added to comparison plot")