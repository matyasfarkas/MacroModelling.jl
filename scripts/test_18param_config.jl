"""
Test script for 18-parameter configuration

Verifies that the Phase 1 18-parameter configuration loads correctly
and can be used for dataset generation.
"""

using Pkg
Pkg.activate(".")

# Load parameter configuration
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

println("="^80)
println("Testing 18-Parameter Configuration")
println("="^80)

# Test 1: Load parameter specs
println("\n[Test 1] Loading parameter specifications...")
try
    specs_wide = get_parameter_specs(:phase1_18params)
    specs_narrow = get_parameter_specs(:phase1_18params_narrow)

    println("  ✓ Wide priors: $(length(specs_wide)) parameters")
    println("  ✓ Narrow priors: $(length(specs_narrow)) parameters")
    @assert length(specs_wide) == 18 "Expected 18 parameters (wide)"
    @assert length(specs_narrow) == 18 "Expected 18 parameters (narrow)"
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 2: Extract parameter names
println("\n[Test 2] Extracting parameter names...")
try
    names = get_parameter_names(:phase1_18params_narrow)
    println("  ✓ Parameters: ", join(string.(names), ", "))
    @assert :crhoa in names "Missing crhoa"
    @assert :cprobp in names "Missing cprobp"
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 3: Get parameter bounds
println("\n[Test 3] Getting parameter bounds...")
try
    bounds = get_parameter_bounds(:phase1_18params_narrow)
    println("  ✓ Bounds for 18 parameters:")
    for (name, (lb, ub)) in bounds
        println("    $name: [$lb, $ub]")
    end
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 4: Get priors
println("\n[Test 4] Getting parameter priors...")
try
    priors = get_parameter_priors(:phase1_18params_narrow)
    println("  ✓ Priors for 18 parameters:")
    for (name, dist) in priors
        println("    $name: $dist")
    end
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 5: Get baseline values
println("\n[Test 5] Getting baseline calibration...")
try
    baseline = get_phase1_18param_baseline()
    println("  ✓ Baseline values:")
    for (name, val) in baseline
        println("    $name = $(round(val, digits=4))")
    end
    @assert length(baseline) == 18 "Expected 18 baseline values"
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

# Test 6: Print summary
println("\n[Test 6] Printing parameter summary...")
try
    print_parameter_summary(:phase1_18params_narrow)
    println("  ✓ Summary printed successfully")
catch e
    println("  ✗ FAILED: $e")
    rethrow(e)
end

println("\n" * "="^80)
println("All tests passed! ✓")
println("="^80)
println("\nNext steps:")
println("1. Run dataset generation with: --param-set phase1_18params_narrow")
println("2. Use existing hlt_sep_surrogate_dataset_generate.jl script")
println("3. Dataset will have 18-dimensional parameter space")
println("="^80)
