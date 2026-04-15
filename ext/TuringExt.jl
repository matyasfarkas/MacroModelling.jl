module TuringExt

using MacroModelling
using Turing

function __init__()
    # Load prior helper distributions into MacroModelling when Turing is available.
    Base.eval(MacroModelling, :(import Turing))
    Base.include(MacroModelling, joinpath(@__DIR__, "..", "src", "priors.jl"))
end

end
