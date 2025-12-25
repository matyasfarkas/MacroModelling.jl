using MacroModelling, StatsPlots          # StatsPlots triggers plotting.jl
include("models/Smets_Wouters_2007_HLT.jl")
m = Smets_Wouters_2007_HLT                # no parentheses
plot_irf(m; shocks = :epinf)              # or MacroModelling.plot_irf

plot_irf!(m,
    shocks = :epinf,
    algorithm = :second_order)

plot_irf!(m,
    shocks = :epinf,
    algorithm = :pruned_third_order)