# Test: Stochastic Funnel IRF Validation
# Replicates Dynare's rbc.mod IRF methodology and validates against benchmark data
#
# Methodology (from user clarification):
# - Start at deterministic steady state
# - t=1: Solve with order=10, shock=-3σ, for 1 period
# - t=2: Solve with order=9, zero shock, for 1 period
# - t=3-10: Continue decreasing order by 1 each period
# - t=11+: Solve deterministic (order=0) for remaining 80 periods
#
# This creates a "stochastic funnel" that gradually reduces uncertainty
# IRF = tt (shocked path) - ts (baseline funnel path)

using MacroModelling
using CSV
using DataFrames
using Printf

println("="^80)
println("STOCHASTIC FUNNEL IRF VALIDATION")
println("="^80)

# Load RBC model
println("\n1. Loading RBC_Dynare model...")
include("models/RBC_Dynare.jl")

# Load benchmark data
println("\n2. Loading Dynare benchmark data...")
benchmark_path = "/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SEP/RBC_irf.csv"
df = CSV.read(benchmark_path, DataFrame, header=false, silencewarnings=true)

# Parse the benchmark data
# The CSV has: ts (cols 1-7), empty cols, tt (remaining columns)
# But CSV.jl drops empty columns, so we need to count actual columns
ncols = size(df, 2)
println("   CSV has $ncols columns")

ts_vars = ["Capital", "Output", "Labour", "Consumption", "Efficiency", "efficiency", "Investment"]
tt_vars = ts_vars  # Same variables

# Extract ts and tt paths (skip header rows 0-1, take rows 2+)
# Convert to Float64 matrix
ts_data = Float64.(Matrix(df[3:end, 1:7]))  # Row 3 onwards in CSV = period 0 onwards
tt_data = Float64.(Matrix(df[3:end, (ncols-6):ncols]))  # Last 7 columns

n_periods = size(ts_data, 1)
println("   Loaded $n_periods periods of data")
println("   Variables: ", join(ts_vars, ", "))

# Configuration matching Dynare
maxorder = 10
shock_magnitude = -3.0  # -3 standard deviations
sigma_epsilon = 0.1     # From rbc.mod: sigma = 0.100
total_periods = 80

println("\n3. Configuration:")
println("   Max order: $maxorder")
println("   Shock magnitude: $(shock_magnitude)σ = $(shock_magnitude * sigma_epsilon)")
println("   Total periods: $total_periods")

# Extract deterministic steady state from benchmark (period 0)
println("\n4. Deterministic steady state from Dynare benchmark:")
dss_benchmark = ts_data[1, :]  # Period 0 from benchmark

var_indices = [findfirst(==(Symbol(var)), RBC_Dynare.var) for var in ts_vars]

for i in 1:length(ts_vars)
    @printf("   %-15s  %.8f\n", ts_vars[i], dss_benchmark[i])
end

println("\n" * "="^80)
println("IMPLEMENTING STOCHASTIC FUNNEL ALGORITHM")
println("="^80)

# TODO: Implement the iterative forward construction
# This requires:
# 1. Ability to solve SEP with varying order parameter
# 2. Ability to extract state at specific time point
# 3. Ability to use that state as initial condition for next solve

println("\n⚠️  IMPLEMENTATION NOTE:")
println("The stochastic funnel requires solving SEP multiple times with:")
println("  - Different branching orders (10, 9, 8, ..., 1, 0)")
println("  - Different initial conditions (using end state from previous solve)")
println("  - Different shock sequences")
println("")
println("Current MacroModelling.jl SEP solver limitations:")
println("  - solve!() computes full path, not single step")
println("  - No API to extract state at specific time point as initial condition")
println("  - No API to vary branching order dynamically")
println("")
println("Two implementation approaches:")
println("")
println("Option A: Extend SEP solver API")
println("  - Add sep_initial_state parameter to solve!()")
println("  - Add sep_horizon parameter (solve for N periods, not full tree)")
println("  - Current sep_order is already available")
println("  - Iteratively call solve!() with decreasing order")
println("")
println("Option B: Direct access to sep_solver.jl internals")
println("  - Call stochastic_extended_path() directly")
println("  - Build shock sequences manually")
println("  - Extract Y vectors at each step")
println("  - More complex but doesn't require API changes")
println("")
println("Recommendation: Option A (cleaner, more maintainable)")

println("\n" * "="^80)
println("NEXT STEPS")
println("="^80)
println("")
println("1. Extend solve!() to accept:")
println("   - sep_initial_state::Vector{Float64} (optional)")
println("   - sep_horizon::Int (optional, default = sep_periods)")
println("")
println("2. Implement iterative funnel construction:")
println("""
   # Shocked path (tt)
   solve!(model, algorithm=:stochastic_extended_path,
          sep_periods=80, sep_order=10,
          sep_shocks=[shock_at_t1; zeros(79)])
   tt = extract_path(model)

   # Baseline funnel path (ts)
   ts = zeros(81, n_vars)  # Period 0 to 80
   ts[1, :] = dss

   current_state = dss
   for order in 10:-1:1
       solve!(model, algorithm=:stochastic_extended_path,
              sep_periods=1, sep_order=order,
              sep_initial_state=current_state,
              sep_shocks=[shock_at_t1])
       current_state = extract_state_at_period(model, 1)
       ts[12-order, :] = current_state  # t=1 for order=10, ..., t=10 for order=1
   end

   # Deterministic continuation
   solve!(model, algorithm=:stochastic_extended_path,
          sep_periods=80, sep_order=0,
          sep_initial_state=current_state)
   ts[12:end, :] = extract_path(model)
""")
println("")
println("3. Compare tt and ts with benchmark data")
println("")
println("4. Compute IRF = tt - ts and validate against Dynare")

println("\n" * "="^80)
