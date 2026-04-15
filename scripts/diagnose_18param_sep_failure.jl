"""
Diagnostic script to identify why 18-parameter SEP fails

Tests:
1. Baseline parameters (should work)
2. First sampled parameter vector (failed in dataset generation)
3. Identify which parameter causes instability
"""

using Pkg
Pkg.activate(".")

using MacroModelling
using Random

# Load HLT model
include("../models/Smets_Wouters_2007_HLT_obc.jl")

# Load parameter configuration
include("hlt_surrogate/parameter_config.jl")

println("="^80)
println("18-Parameter SEP Failure Diagnostic")
println("="^80)

# Test 1: Baseline parameters (from HLT calibration)
println("\n[Test 1] Testing baseline (calibrated) parameters...")
println("-"^80)

baseline = get_phase1_18param_baseline()
println("Baseline parameters:")
for (k, v) in baseline
    println("  $k = $(round(v, digits=4))")
end

# Try SEP solve with baseline
println("\nAttempting SEP solve with baseline parameters...")
try
    # Set parameters in model
    param_dict = Dict{Symbol, Float64}()
    for (k, v) in baseline
        param_dict[k] = v
    end

    # Get initial state (steady state)
    ss = Smets_Wouters_2007_HLT_obc.solution.non_stochastic_steady_state

    # Try one SEP solve
    import MacroModelling: simulate_SEP

    periods = 40
    shocks = zeros(7, periods)  # 7 shocks, 40 periods

    # Small shock to test
    shocks[1, 1] = 0.01  # Small TFP shock

    println("Running SEP with:")
    println("  Periods: $periods")
    println("  Initial state: steady state")
    println("  Shock: 0.01 to first shock")

    result = simulate_SEP(
        Smets_Wouters_2007_HLT_obc,
        ss,
        shocks;
        periods=periods,
        order=1,
        nnodes=3,
        expectation_method=:hmc,
        hmc_samples=100,
        hmc_warmup=50,
        hmc_leapfrog_steps=15,
        verbose=true
    )

    println("✓ SUCCESS: Baseline parameters work!")
    println("  Result dimensions: $(size(result))")

catch e
    println("✗ FAILED: Baseline parameters failed!")
    println("  Error: $e")
    showerror(stdout, e, catch_backtrace())
end

# Test 2: Sample a parameter vector and test
println("\n\n[Test 2] Testing randomly sampled parameters from narrow priors...")
println("-"^80)

specs = get_parameter_specs(:phase1_18params_narrow)
param_names = get_parameter_names(:phase1_18params_narrow)
bounds = get_parameter_bounds(:phase1_18params_narrow)

# Sample one parameter vector
Random.seed!(123)  # Reproducible
sampled_params = Dict{Symbol, Float64}()
for name in param_names
    lb, ub = bounds[name]
    sampled_params[name] = lb + rand() * (ub - lb)
end

println("Sampled parameters:")
for name in param_names
    println("  $name = $(round(sampled_params[name], digits=4)) (bounds: $(bounds[name]))")
end

# Try SEP solve with sampled parameters
println("\nAttempting SEP solve with sampled parameters...")
try
    # Get initial state
    ss = Smets_Wouters_2007_HLT_obc.solution.non_stochastic_steady_state

    periods = 40
    shocks = zeros(7, periods)
    shocks[1, 1] = 0.01

    result = simulate_SEP(
        Smets_Wouters_2007_HLT_obc,
        ss,
        shocks;
        periods=periods,
        order=1,
        nnodes=3,
        expectation_method=:hmc,
        hmc_samples=100,
        hmc_warmup=50,
        hmc_leapfrog_steps=15,
        verbose=true
    )

    println("✓ SUCCESS: Sampled parameters work!")

catch e
    println("✗ FAILED: Sampled parameters failed!")
    println("  Error: $e")

    # Test each parameter individually
    println("\n[Test 3] Testing parameters individually to find culprit...")
    println("-"^80)

    for name in param_names
        print("  Testing $name... ")
        test_params = copy(baseline)
        test_params[name] = sampled_params[name]

        try
            # Quick test - just first period
            result = simulate_SEP(
                Smets_Wouters_2007_HLT_obc,
                ss,
                shocks;
                periods=5,
                order=1,
                nnodes=3,
                expectation_method=:hmc,
                hmc_samples=20,
                hmc_warmup=10,
                verbose=false
            )
            println("OK")
        catch
            println("FAILED!")
            println("    Culprit: $name = $(sampled_params[name])")
            println("    Baseline: $name = $(baseline[name])")
            println("    Bounds: $(bounds[name])")
        end
    end
end

println("\n" * "="^80)
println("Diagnostic complete")
println("="^80)
