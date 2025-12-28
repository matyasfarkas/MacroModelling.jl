using MacroModelling, StatsPlots          # StatsPlots triggers plotting.jl
include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT                # no parentheses

# First-order perturbation (default)
p = plot_irf(m; shocks = :epinf)          # Capture the plot

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

# Add SEP IRF to the existing plot
# Note: irf_sep has dimensions (variables × periods)
# Time axis is 0:20, so we need periods+1 points
nvars = size(irf_sep, 1)
time_axis = 0:size(irf_sep, 2)-1

for i in 1:nvars
    plot!(p, time_axis, irf_sep[i, :],
          label = "SEP",
          linewidth = 2,
          linestyle = :dash,
          color = :black,
          subplot = i)
end

println("✓ SEP IRF added to comparison plot")
display(p)