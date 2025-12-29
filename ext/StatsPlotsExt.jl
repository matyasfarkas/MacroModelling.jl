module StatsPlotsExt

using MacroModelling
using StatsPlots

function __init__()
    # Load plotting definitions into MacroModelling when StatsPlots is available.
    Base.eval(MacroModelling, :(import StatsPlots))
    Base.include(MacroModelling, joinpath(@__DIR__, "..", "src", "plotting.jl"))
end

end
