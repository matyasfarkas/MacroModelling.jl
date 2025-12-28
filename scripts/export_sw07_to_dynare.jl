# Export SW07_HLT model to Dynare .mod format

using MacroModelling

# Load the SW07_HLT model
include("models/Smets_Wouters_2007_HLT.jl")

# Write to Dynare .mod file
println("Exporting Smets_Wouters_2007_HLT to Dynare format...")
write_mod_file(Smets_Wouters_2007_HLT)

println("✓ Export complete!")
println("Generated file: Smets_Wouters_2007_HLT.mod")
