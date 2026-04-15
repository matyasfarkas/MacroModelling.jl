# Export SW07_HLT model to Dynare .mod format

using MacroModelling
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

# Load the SW07_HLT model
mm_model = load_hlt_model(normpath(joinpath(@__DIR__, "..")), "Smets_Wouters_2007_HLT"; mod = @__MODULE__)

# Write to Dynare .mod file
println("Exporting Smets_Wouters_2007_HLT to Dynare format...")
write_mod_file(mm_model)

println("✓ Export complete!")
println("Generated file: Smets_Wouters_2007_HLT.mod")
