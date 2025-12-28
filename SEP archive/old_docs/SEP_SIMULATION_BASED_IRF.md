# SEP Simulation-Based IRF Implementation - December 26, 2024

## Overview

Implemented stochastic simulation capability for SEP and proper nonlinear IRF computation following Dynare's extended path approach.

## Key Conceptual Change

### Old Approach (Tree Extraction)
- **Method**: Extract IRF by following a shocked path through the pre-solved SEP tree
- **Formula**: `IRF(t) = y_shocked(t) - deterministic_SS`
- **Problem**: Conceptually incorrect for nonlinear models
  - Starts from deterministic steady state, not stochastic steady state
  - Doesn't properly account for path dependence
  - Magnitude issues (100x too small initially, fixed but still problematic)

### New Approach (Simulation-Based)
- **Method**: Dynare-style stochastic simulation with burn-in and comparison
- **Formula**: `IRF(t) = E[y_t | shock at t=0, from SSS] - E[y_t | no shock, from SSS]`
- **Advantages**:
  - Starts from stochastic steady state (SSS)
  - Properly captures nonlinear dynamics
  - IRF is difference between two stochastic paths
  - Matches economic interpretation of nonlinear IRFs

## Implementation Details

### 1. Stochastic Simulation (`simulate_sep`)

**File**: `src/sep_simulation.jl`

**Purpose**: Sequential stochastic simulation using SEP for expectations

**Algorithm** (following Dynare's extended_path):
```
For each period t:
  1. Observe realized shock ε_t
  2. Solve SEP problem from current state y_{t-1} with T-period horizon
  3. Extract first-period decision from SEP solution
  4. Update state: y_t = f(y_{t-1}, ε_t, SEP_decision)
  5. Move to next period
```

**Key Functions**:
- `simulate_sep(𝓂; periods, initial_state, shocks, burn_in, ...)`: Main simulation driver
- `sep_step(𝓂, y_current, ε_current, ...)`: One-step-ahead decision using SEP
- `shock_to_group(ε, nnodes, nshocks)`: Maps realized shocks to SEP tree groups

**Current Implementation Note**:
- `sep_step` uses approximation: maps realized shock to nearest GH node in pre-solved SEP tree
- This avoids expensive re-solving of SEP at each step
- For models close to linear, approximation is reasonable
- For highly nonlinear models, could be improved by re-solving SEP from current state

### 2. Simulation-Based IRF (`get_sep_irf`)

**File**: `src/sep_irf.jl`

**Purpose**: Compute IRF as difference between stochastic paths

**Algorithm**:
```
Step 1: Burn-in simulation (100+ periods with random shocks)
        → Reach stochastic steady state (SSS)
        → Extract final state y_SSS

Step 2: Baseline simulation from SSS (zero shocks)
        → Path: y_baseline(0), y_baseline(1), ..., y_baseline(T)

Step 3: Shocked simulation from SSS (shock applied at t=1)
        → Shock matrix: ε[shock_idx, 1] = shock_size * σ, zeros elsewhere
        → Path: y_shocked(0), y_shocked(1), ..., y_shocked(T)

Step 4: Compute IRF
        → IRF(t) = y_shocked(t) - y_baseline(t)
```

**Parameters**:
- `periods`: IRF horizon (default: 40)
- `burn_in`: Burn-in periods to reach SSS (default: 100)
- `random_seed`: For reproducibility
- `silent`: Suppress progress messages

**Key Features**:
- Properly implements nonlinear IRF concept
- Both paths start from same SSS state
- Difference isolates effect of the shock
- Matches Dynare's extended path philosophy

### 3. Backward Compatibility

**Old Function**: Renamed to `get_sep_irf_tree()`
- Still available for comparison or special use cases
- Extracts path directly from pre-solved SEP tree
- Faster but conceptually less accurate

**New Function**: `get_sep_irf()` (replaces old)
- Uses simulation-based approach
- Slower but conceptually correct
- Default method going forward

## Files Modified/Created

### Modified:
1. `src/MacroModelling.jl`
   - Line 169: Added `include("sep_simulation.jl")`
   - Line 194: Added `simulate_sep` to exports

2. `src/sep_solver.jl`
   - Lines 56-63: Fixed `child_groups()` to return K copies for t >= Lbr
   - Lines 334-364: Fixed shock extraction and weight indexing

3. `src/sep_irf.jl`
   - Lines 134-205: Renamed old `get_sep_irf` to `get_sep_irf_tree`
   - Lines 208-343: New simulation-based `get_sep_irf`

### Created:
1. `src/sep_simulation.jl` (NEW)
   - `simulate_sep()`: Main simulation function
   - `sep_step()`: One-step SEP decision
   - `shock_to_group()`: Shock-to-node mapping

2. `test_sep_simulation.jl` (NEW)
   - Basic test of simulation infrastructure

3. `test_sep_irf_simulation.jl` (NEW)
   - Comprehensive test comparing SEP vs Perturbation IRFs
   - Includes correlation analysis and interpretation

## Testing

### Test 1: Basic Simulation (`test_sep_simulation.jl`)
Tests that `simulate_sep` runs without errors and produces reasonable output.

### Test 2: IRF Comparison (`test_sep_irf_simulation.jl`)
Compares simulation-based SEP IRF with first-order perturbation IRF:
- Variables tested: y, c, labobs, pinfobs
- Metrics: Impact magnitude, correlation over horizon
- Expected: High correlation (>0.9) for near-linear models

## Economic Interpretation

### First-Order (Linear) IRF:
- **Definition**: Deviation from deterministic steady state
- **Path independence**: Same IRF regardless of history
- **Formula**: `IRF(t) = y(t) - y_SS`

### Higher-Order (Nonlinear) IRF:
- **Definition**: Expected path difference starting from stochastic steady state
- **Path dependence**: Depends on starting state and shock realization
- **Formula**: `IRF(t) = E[y(t) | shocked, SSS] - E[y(t) | baseline, SSS]`

### Key Differences:
1. **Starting point**: SSS ≠ deterministic SS in nonlinear models
2. **Baseline comparison**: Stochastic path with typical shocks vs zero deviation
3. **Magnitude**: Can differ significantly due to nonlinearity
4. **Shape**: May differ if model exhibits strong nonlinear effects

## User's Key Insight

> "The Gauss Hermite approximation's node density is not as important for the solution, as the right timing!"

This insight led to discovering the shock timing bug and understanding that:
- Correct shock attribution to time periods is critical
- Node density (nnodes=3 vs 5) is less important than getting the timing right
- Sequential simulation properly handles shock timing by construction

## Future Improvements

### 1. Full Re-solving in `sep_step`
**Current**: Maps to pre-solved SEP tree (approximation)
**Potential**: Solve fresh SEP problem from current state each step
**Trade-off**: Much slower but potentially more accurate for highly nonlinear models

### 2. Ensemble IRFs
**Idea**: Compute IRFs from multiple SSS starting points
**Benefit**: Captures distribution of possible IRF paths in nonlinear models
**Output**: Mean IRF + confidence bands

### 3. Higher-Order SEP
**Current**: sep_order=1 (one period of branching)
**Potential**: sep_order=2 or higher for better expectation approximation
**Trade-off**: Exponentially more expensive (K^Lbr groups)

### 4. Conditional IRFs
**Idea**: IRF conditional on specific state or shock history
**Use case**: State-dependent dynamics in nonlinear models

## References

### Dynare Implementation:
- File: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/dynare EP/matlab/ep/extended_path.m`
- Approach: Period-by-period perfect foresight with stochastic shocks
- Our implementation follows this philosophy adapted for SEP

### Previous Fixes:
1. **Shock timing bug** (SEP_SHOCK_TIMING_BUG_FIX.md)
   - Fixed shock extraction in solver loop
   - Changed from child group indexing to current group indexing

2. **child_groups() fix** (this session)
   - Returns K copies for proper Gauss-Hermite integration
   - Critical for correct expectation computation

## Conclusion

The simulation-based IRF implementation provides:
- ✓ Conceptually correct nonlinear IRFs
- ✓ Proper stochastic steady state baseline
- ✓ Path-dependent dynamics
- ✓ Compatibility with Dynare's extended path philosophy
- ✓ Foundation for future extensions (ensemble IRFs, higher-order SEP)

This represents a significant improvement over the tree-extraction method and aligns with the user's understanding of nonlinear IRFs.
