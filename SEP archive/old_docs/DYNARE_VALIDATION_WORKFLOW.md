# Dynare SEP Validation Workflow

This document describes the workflow for validating MacroModelling.jl's Stochastic Extended Path (SEP) implementation against Dynare.

## Overview

We use a **bidirectional translation approach** to validate SEP:
1. **Export MacroModelling → Dynare**: Take a known-working model and export it to Dynare format
2. **Run both implementations**: Execute SEP in both systems with identical parameters
3. **Compare results**: Validate that steady states and dynamics match

## Files Created

### 1. Model Export Script
**File**: `export_sw07_to_dynare.jl`

```julia
using MacroModelling

# Load the SW07_HLT model
include("models/Smets_Wouters_2007_HLT.jl")

# Write to Dynare .mod file
println("Exporting Smets_Wouters_2007_HLT to Dynare format...")
write_mod_file(Smets_Wouters_2007_HLT)

println("✓ Export complete!")
println("Generated file: Smets_Wouters_2007_HLT.mod")
```

### 2. Generated Dynare File
**File**: `Smets_Wouters_2007_HLT.mod`

- 317 lines
- 66 variables
- 7 shocks (ea, eb, eg, em, epinf, eqs, ew)
- 49 parameters
- Includes two extended_path tests:
  - `extended_path(periods=10, order=1)` - branching length = 1
  - `extended_path(periods=10, order=2)` - branching length = 2
- Diagnostic output for key variables: y, c, inve, pinf, lab

### 3. MacroModelling Test Script
**File**: `test_sw07_HLT_sep_for_dynare_comparison.jl`

Runs identical SEP configuration in MacroModelling.jl:
- SEP(1): periods=10, order=1, nnodes=3
- SEP(2): periods=10, order=2, nnodes=3
- Reports same key variables for direct comparison

## Running the Validation

### Step 1: Run MacroModelling Test

```bash
julia --project=. test_sw07_HLT_sep_for_dynare_comparison.jl
```

Expected output:
```
======================================================================
MacroModelling.jl SEP TESTS: SW07_HLT
Matching Dynare extended_path(periods=10, order={1,2})
======================================================================

1. Model loaded successfully
   Model variables: 66
   Model shocks: [:ea, :eb, :eg, :em, :epinf, :eqs, :ew]
   Model parameters: 49

======================================================================
TEST 1: SEP(1) - extended_path(periods=10, order=1)
======================================================================
Configuration:
  - periods (T) = 10
  - order (branching length) = 1
  - nnodes = 3 (Gauss-Hermite quadrature)

Solution status:
  - Convergence: ✓
  - Final error: <small number>
  - Runtime: <seconds> seconds

Key steady state values (SEP order=1):
  Variable      Value
  ------------------------------
  y                <value>
  c                <value>
  inve             <value>
  pinf             <value>
  lab              <value>

[... SEP(2) output ...]
```

### Step 2: Run Dynare Test

In MATLAB or Octave:

```matlab
>> dynare Smets_Wouters_2007_HLT.mod
```

Expected output:
```
================================================================
DYNARE EXTENDED PATH TEST 1: periods=10, order=1
================================================================

[Dynare extended_path() output]

Key steady state values (SEP order=1):
  y    = <value>
  c    = <value>
  inve = <value>
  pinf = <value>
  lab  = <value>

================================================================
DYNARE EXTENDED PATH TEST 2: periods=10, order=2
================================================================

[... similar output ...]
```

### Step 3: Compare Results

Manually compare the reported steady state values:

| Variable | MacroModelling SEP(1) | Dynare SEP(1) | Difference |
|----------|----------------------|---------------|------------|
| y        | ?                    | ?             | ?          |
| c        | ?                    | ?             | ?          |
| inve     | ?                    | ?             | ?          |
| pinf     | ?                    | ?             | ?          |
| lab      | ?                    | ?             | ?          |

Repeat for SEP(2).

## Expected Outcomes

### If Results Match Closely
- ✓ Validates MacroModelling SEP implementation
- ✓ Confirms shock scaling is correct
- ✓ Builds confidence in both order=1 and order=2

### If Results Differ Significantly

Potential sources of difference:

1. **Quadrature method**
   - MacroModelling: Gauss-Hermite quadrature
   - Dynare: May use Unscented transform (check .mod options)

2. **Shock scaling**
   - Verify that both use identical shock standard deviations
   - Check GH node placement: [-√3σ, 0, +√3σ]

3. **Newton solver settings**
   - Tolerance differences
   - Line search algorithms
   - Maximum iterations

4. **Initial guess**
   - MacroModelling: Uses deterministic SS or previous solution (warm start)
   - Dynare: Check initialization strategy

5. **Parameter interpretation**
   - Confirm "order" means the same thing (branching length)
   - Verify "periods" is interpreted identically

## Key Implementation Details

### MacroModelling.jl SEP

From `src/MacroModelling.jl:6925-6947`:

```julia
# SEP solver with automatic warm start
if haskey(model.solution, :perturbation) &&
   haskey(model.solution.perturbation, :stochastic_extended_path)
    # Reuse previous solution as initial guess
    initial_guess = model.solution.perturbation.stochastic_extended_path.Y
end

solve!(model,
       algorithm = :stochastic_extended_path,
       sep_periods = T,
       sep_order = Lbr,
       sep_nnodes = nnodes,
       sep_initial_guess = initial_guess)
```

### Dynare SEP

From Dynare documentation:
- `extended_path(periods=T, order=Lbr)`:
  - `T`: Simulation horizon
  - `Lbr`: Branching length (how many periods ahead to branch decision tree)

## Critical Fix: Float64 Type Support

**Problem**: Dynare translations failed with:
```
MethodError: Cannot `convert` an object of type Float64 to an object of type Union{Int64, Expr, Symbol}
```

**Solution**: Modified `src/MacroModelling.jl:2936`:

```julia
# BEFORE:
function simplify(ex::Expr)::Union{Expr,Symbol,Int}

# AFTER:
function simplify(ex::Expr)::Union{Expr,Symbol,Int,Float64}
```

This fix allows SymPy-evaluated Float64 literals from Dynare parameter expressions to pass through the `@model` macro.

## Next Steps

After validation:

1. **Document differences**: If any systematic differences emerge, document them
2. **Integration tests**: Add automated tests comparing against Dynare reference solutions
3. **Extended validation**: Test with other models (DSGE_RU, simpler models)
4. **Quadrature options**: Consider implementing Unscented transform as alternative to Gauss-Hermite

## References

- MacroModelling.jl SEP implementation: `src/sep_solver.jl`
- Dynare extended_path: https://www.dynare.org/manual/
- Gauss-Hermite quadrature: `src/sep_solver.jl` (quadrature node generation)
- Plan document: `/Users/matyasfarkas/.claude/plans/cheerful-inventing-orbit.md`
