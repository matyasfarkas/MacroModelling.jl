# Export Gali_2015_chapter_3_nonlinear model to Dynare .mod format

using MacroModelling

# Load the Gali model
include("models/Gali_2015_chapter_3_nonlinear.jl")

# Write to Dynare .mod file
println("Exporting Gali_2015_chapter_3_nonlinear to Dynare format...")
write_mod_file(Gali_2015_chapter_3_nonlinear)

println("✓ Export complete!")
println("Generated file: Gali_2015_chapter_3_nonlinear.mod")
