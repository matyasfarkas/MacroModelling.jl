# LMMCP Porting Feasibility Assessment

**Date**: January 19, 2026
**Purpose**: Assess feasibility of porting Dynare's LMMCP solver to MacroModelling.jl for OBC enforcement in SEP
**Requested by**: User (research goal: make SEP work with proper OBC enforcement)

---

## Executive Summary

### Feasibility: **MEDIUM** (3-4 weeks implementation)

**Bottom Line**: Porting LMMCP to Julia is **technically feasible** and would enable proper OBC enforcement in SEP. The code is self-contained, well-documented, and has a permissive license. However, it requires:

1. **Translation effort**: ~626 lines of MATLAB → ~800-1000 lines of Julia
2. **Integration work**: Connecting LMMCP to MacroModelling's SEP infrastructure
3. **Testing and validation**: Extensive testing against Dynare results
4. **Numerical tuning**: Solver parameters may need adjustment for Julia/QMIPF

**Alternative recommendation**: Consider using existing Julia MCP solvers (PATHSolver.jl, Complementarity.jl) which may be faster to integrate but lack LMMCP's specific advantages.

---

## Part 1: LMMCP Code Analysis

### 1.1 Code Structure

**File**: `/Dynare/matlab/lmmcp/lmmcp.m`
**Lines of code**: 626 lines (excluding subfunctions Phi and DPhi)
**Total with subfunctions**: ~826 lines

**Main components**:

```matlab
lmmcp.m (626 lines)
├── Initialization (lines 100-243)
│   ├── Default options
│   ├── Index set definitions (I_l, I_u, I_lu)
│   ├── Initial function evaluation
│   └── Watchdog initialization
├── Phase I: Preprocessor (lines 245-338)
│   ├── Projected Levenberg-Marquardt
│   ├── Trust region method
│   └── Early termination if converged
├── Phase II: Main algorithm (lines 340-435)
│   ├── Levenberg-Marquardt direction
│   ├── Nonmonotone line search
│   ├── Watchdog strategy
│   └── Convergence checking
└── Subfunctions (lines 460-626)
    ├── Phi (NCP function) (102 lines)
    └── DPhi (Jacobian of Phi) (166 lines)
```

### 1.2 Algorithm Overview

**Core Approach**: Semismooth least squares formulation

LMMCP solves the mixed complementarity problem:
```
Find X such that:
  LB = X     =>   F(X) > 0
  LB < X < UB =>   F(X) = 0
      X = UB =>   F(X) < 0
```

By minimizing:
```
Ψ(X) = 0.5 * ||Φ(X)||²
```

Where `Φ(X)` is a Fischer-Burmeister-type NCP function that enforces complementarity.

**Key features**:

1. **Two-phase approach**:
   - Phase I (preprocessing): Fast projected LM with trust region
   - Phase II (main): Full LM with nonmonotone line search

2. **NCP function** (Fischer-Burmeister variant):
   - For lower-bounded: `φ_l = sqrt(a² + b²) - a - b ≈ 0`
   - For box-constrained: Combination of lower/upper bounds
   - Semismooth (not differentiable everywhere, but directionally differentiable)

3. **Levenberg-Marquardt regularization**:
   - Adaptive damping: `μ = 1e-6/(k+1)` based on condition number
   - Handles near-singularity better than pure Newton

4. **Watchdog strategy**:
   - Saves best iterate within last `kwatch` iterations
   - Prevents getting stuck in bad regions
   - Resets to best if no improvement

5. **Nonmonotone line search**:
   - Uses maximum of last `m` function values
   - More robust than monotone line search
   - Armijo-type rule with backtracking

### 1.3 Dependencies

**MATLAB-specific functions used**:

| MATLAB Function | Purpose | Julia Equivalent |
|-----------------|---------|------------------|
| `condest()` | Condition number estimate | `cond()` from LinearAlgebra |
| `sparse()` | Sparse matrix construction | `sparse()` from SparseArrays |
| `speye()` | Sparse identity matrix | `sparse(I, n, n)` |
| `\` operator | Linear solve (LU/QR) | `\` operator (same) |
| `norm()` | Vector/matrix norm | `norm()` from LinearAlgebra |
| `max()`, `min()` | Element-wise max/min | `max.()`, `min.()` (broadcast) |
| `sqrt()` | Square root | `sqrt()` (same) |

**External dependencies**: None! LMMCP is completely self-contained.

### 1.4 License

**License**: Unlimited permission (confirmed by Kanzow & Petra)

From lines 89-98:
```matlab
% Christian Kanzow sent a private message to Dynare Team on July 8, 2014,
% confirming the free software status of lmmcp and granting unlimited
% permission to use, copy, modifiy or redistribute the file.

% Copyright © 2005 Christian Kanzow and Stefania Petra
% Copyright © 2013 Christophe Gouel
% Copyright © 2014-2017 Dynare Team
%
% Unlimited permission is granted to everyone to use, copy, modify or
% distribute this software.
```

✅ **No licensing barriers to porting to Julia**

### 1.5 Numerical Complexity

**Per iteration cost**:

1. **Function evaluation**: `F(X)` - model-specific, O(neq)
2. **NCP function**: `Φ(X, F(X))` - O(2*neq) operations
3. **Jacobian of NCP**: `DΦ(X, F(X), DF(X))` - O(4*neq²) operations
4. **Linear solve**: `(DΦ'*DΦ + μI) \ (DΦ'*(-Φ))` - O(neq³) for dense, O(neq) for sparse

**For QMIPF model** (120 variables, 73 state variables, T=20 periods):
- `neq = 120 * 20 = 2400` equations
- **Dense solve**: ~14 billion ops per iteration (expensive!)
- **Sparse solve**: ~120,000 ops per iteration (feasible)

**Sparsity is critical** for large models like QMIPF.

---

## Part 2: MacroModelling SEP Architecture Analysis

### 2.1 Current SEP Solver Structure

**File**: `src/sep_solver.jl`
**Lines of code**: 1656 lines
**Solver**: Standard Newton with Levenberg-Marquardt regularization

**Key functions**:

```julia
sep_solve_mm!(𝓂::ℳ, parameters; opts, initial_guess, initial_state)
├── build_dynamic_residual_jacobian(𝓂)     # Symbolics → function
├── build_parameters_and_ss_values(...)     # Parameter mapping
├── solve_deterministic_path(...)           # order=0 mode
└── Main Newton loop (lines 1278-1654)
    ├── Residual/Jacobian assembly
    ├── Newton step: (J'*J + λI) \ (J'*(-R))
    ├── Line search
    └── Convergence check
```

### 2.2 Deterministic Path Solver (Perfect Foresight)

**Location**: Lines 619-884 in `sep_solver.jl`

This is the **most relevant** part for OBC enforcement:

```julia
function solve_deterministic_path(
    𝓂::ℳ,
    parameters::Vector{Float64},
    opts::SEPSolverOptions,
    ...
)
    # Build stacked system for t=1,...,T
    for t in 1:T
        # Get states at t-1, t, t+1
        # Compute F(y_{t-1}, y_t, y_{t+1}, ε_t)
        # Store residual and Jacobian blocks
    end

    # Build sparse Jacobian (tridiagonal block structure)
    J = sparse(...)

    # Newton iteration (line 723-883)
    for it in 1:opts.maxit
        # Compute residual and Jacobian
        err = maximum(abs, R)

        # Newton step with LM regularization (line 823-836)
        Δ = (J'*J + lm_lambda*I) \ (J'*(-R))  # ← REPLACE THIS WITH LMMCP!

        # Line search (optional)
        # Update Y
        # Check convergence
    end
end
```

**This is where LMMCP integration would go**: Lines 823-836 in `solve_deterministic_path`.

### 2.3 Current Solver Limitations

**Why standard Newton fails for OBC**:

1. **No complementarity structure**: Treats `F(X) = 0` as standard system
2. **Box constraints not enforced**: `THETA ≥ 0`, `BLIM ≥ 0` not respected during solve
3. **max() operators smoothed**: Solver may linearize max() functions

**What LMMCP would add**:

1. ✅ Explicit complementarity: `THETA · BLIM = 0`
2. ✅ Box constraint enforcement during solve
3. ✅ Specialized NCP function designed for complementarity
4. ✅ Better convergence for systems with active constraints

### 2.4 Integration Points

**Three possible integration strategies**:

#### Option A: Replace deterministic path solver (RECOMMENDED)

**Location**: `solve_deterministic_path` function (lines 619-884)

**Changes required**:
1. Add `lb` and `ub` vectors to specify bounds on variables
2. Replace Newton step (lines 823-836) with LMMCP call
3. Map MacroModelling variables to bounded/unbounded sets
4. Handle initial conditions (y₀ fixed, y₁...yT bounded)

**Advantages**:
- Minimal disruption to existing code
- Only affects deterministic mode (order=0)
- Can keep existing stochastic SEP unchanged

**Disadvantages**:
- Only helps with `sep_order=0` (perfect foresight)
- Doesn't address stochastic SEP with OBC

#### Option B: Create new MCP-aware SEP solver

**Approach**: New function `sep_solve_mm_mcp!` parallel to `sep_solve_mm!`

**Changes required**:
1. Copy entire `sep_solve_mm!` function
2. Modify Newton loop to use LMMCP
3. Add bound specification for all state variables
4. Handle branching tree with MCP at each node

**Advantages**:
- Could work with stochastic SEP (order > 0)
- More general solution
- Keeps existing solver as fallback

**Disadvantages**:
- Much larger implementation (~1000+ lines)
- Complex interaction with branching tree
- Unclear how to handle expectations with complementarity

#### Option C: Hybrid approach (BEST FOR RESEARCH)

**Approach**: Use LMMCP for deterministic mode, flag when OBC active

**Implementation**:
1. Detect when model has OBC (check for ϵᵒᵇᶜ shocks)
2. In `solve_deterministic_path`, use LMMCP instead of standard Newton
3. For stochastic SEP, recommend order=0 (deterministic) when OBC present
4. Add clear warnings when user tries stochastic SEP with OBC

**Advantages**:
- Solves user's immediate problem (QMIPF with OBC)
- Minimal code disruption
- Clear scope and limitations
- Can be extended later

---

## Part 3: Implementation Strategy

### 3.1 Phase 1: LMMCP Port (1-2 weeks)

**Step 1.1: Create LMMCP module** (3-4 days)

Create `src/lmmcp.jl`:

```julia
module LMMCP

using LinearAlgebra, SparseArrays

"""
Options for LMMCP solver
"""
struct LMMCPOptions
    MaxIter::Int
    TolFun::Float64
    preprocess::Bool
    presteps::Int
    Display::Symbol
    # ... other options

    function LMMCPOptions(;
        MaxIter=500,
        TolFun=√eps(),
        preprocess=true,
        presteps=20,
        Display=:none,
        # ... defaults
    )
        new(MaxIter, TolFun, preprocess, presteps, Display, ...)
    end
end

"""
    lmmcp(FUN, x0, lb, ub, options, args...)

Solve mixed complementarity problem.

# Arguments
- `FUN`: Function returning (F, DF) - residual and Jacobian
- `x0`: Initial guess
- `lb`: Lower bounds
- `ub`: Upper bounds
- `options`: LMMCPOptions
- `args...`: Additional arguments passed to FUN

# Returns
- `x`: Solution
- `Fval`: Final residual F(x)
- `exitflag`: 0=max iterations, 1=converged, -1=stationary point
- `output`: NamedTuple with (iterations, Psix, normDPsix)
- `Jacob`: Jacobian DF(x)
"""
function lmmcp(
    FUN::Function,
    x::Vector{Float64},
    lb::Vector{Float64},
    ub::Vector{Float64},
    options::LMMCPOptions,
    args...
)
    # Initialization (translate lines 100-243)
    n = length(x)
    x = max.(lb, min.(x, ub))  # Project to feasible region

    # Define index sets
    Big = 1e10
    I_l = (lb .> -Big) .& (ub .> Big)      # Lower bounded only
    I_u = (lb .< -Big) .& (ub .< Big)      # Upper bounded only
    I_lu = (lb .> -Big) .& (ub .< Big)     # Box bounded
    Indexset = zeros(Int, n)
    Indexset[I_l] .= 1
    Indexset[I_u] .= 2
    Indexset[I_lu] .= 3

    # Initial function evaluation
    Fx, DFx = FUN(x, args...)

    # NCP function evaluation
    Phix = Phi(x, Fx, lb, ub, λ1, λ2, n, Indexset)
    Psix = 0.5 * dot(Phix, Phix)
    DPhix = DPhi(x, Fx, DFx, lb, ub, λ1, λ2, n, Indexset)
    DPsix = DPhix' * Phix

    # Phase I: Preprocessor (translate lines 245-338)
    if options.preprocess
        # Projected Levenberg-Marquardt with trust region
        # ... implementation
    end

    # Phase II: Main algorithm (translate lines 340-435)
    for k in 1:options.MaxIter
        # Compute LM direction
        # Nonmonotone line search
        # Watchdog strategy
        # Update
        # Check convergence
    end

    return x, Fx, exitflag, output, DFx
end

# Subfunctions
function Phi(x, Fx, lb, ub, λ1, λ2, n, Indexset)
    # Fischer-Burmeister NCP function (translate lines 460-481)
    # ...
end

function DPhi(x, Fx, DFx, lb, ub, λ1, λ2, n, Indexset)
    # Jacobian of Phi (translate lines 484-625)
    # ...
end

end # module LMMCP
```

**Translation challenges**:

| MATLAB Pattern | Julia Translation | Difficulty |
|----------------|-------------------|------------|
| `sparse(I, J, V, m, n)` | `sparse(I, J, V, m, n)` | Easy |
| `speye(n)` | `sparse(I, n, n)` | Easy |
| Logical indexing: `x(I)` | `x[I]` | Easy |
| `condest(A)` | `cond(A)` or `cond(A, Inf)` | Medium (may need iterative estimator) |
| `[A; B]` (vertical concat) | `vcat(A, B)` or `[A; B]` | Easy |
| `mod(k, m) + 1` | `mod(k-1, m) + 1` (0-based → 1-based) | Medium |
| `LZ = false(n,1)` | `LZ = falses(n)` | Easy |

**Step 1.2: Unit tests** (1-2 days)

Create `test/test_lmmcp.jl`:

```julia
using Test
include("../src/lmmcp.jl")

@testset "LMMCP Unit Tests" begin
    # Test 1: Simple complementarity problem
    @testset "Simple MCP" begin
        # x ≥ 0, F(x) = x - 1 => x = 1
        F(x) = (x .- 1.0, Matrix(I, length(x), length(x)))
        x0 = [0.5]
        lb = [0.0]
        ub = [Inf]
        opts = LMMCPOptions(MaxIter=100, TolFun=1e-10)

        x, Fx, flag, output, _ = lmmcp(F, x0, lb, ub, opts)

        @test flag == 1  # Converged
        @test x[1] ≈ 1.0 atol=1e-8
        @test Fx[1] ≈ 0.0 atol=1e-8
    end

    # Test 2: Box-constrained problem
    @testset "Box MCP" begin
        # 0 ≤ x ≤ 2, F(x) = x² - 1 => x = 1
        F(x) = ([x[1]^2 - 1], [2*x[1]]')
        x0 = [0.5]
        lb = [0.0]
        ub = [2.0]
        opts = LMMCPOptions(MaxIter=100, TolFun=1e-10)

        x, Fx, flag, output, _ = lmmcp(F, x0, lb, ub, opts)

        @test flag == 1
        @test x[1] ≈ 1.0 atol=1e-8
    end

    # Test 3: Debt limit constraint (simplified QMIPF)
    @testset "Debt limit MCP" begin
        # THETA ≥ 0, BLIM ≥ 0, THETA · BLIM = 0
        # Given: BLIM = NFA + m*Y
        # If BLIM > 0: THETA = 0 (slack)
        # If BLIM = 0: THETA > 0 (binding)

        m = 0.1
        NFA = -5.0  # Below limit
        Y = 10.0

        function debt_limit_residual(x)
            THETA, BLIM_slack = x
            # Equations:
            # 1. BLIM = NFA + m*Y (definition)
            # 2. Complementarity enforced by MCP
            BLIM = NFA + m * Y
            F = [
                THETA,  # Will be 0 if BLIM > 0
                BLIM + BLIM_slack - (NFA + m*Y)  # Slack variable
            ]
            J = [
                1.0  0.0;
                0.0  1.0
            ]
            return F, J
        end

        x0 = [0.0, 0.0]
        lb = [0.0, 0.0]  # Both ≥ 0
        ub = [Inf, Inf]
        opts = LMMCPOptions(MaxIter=100, TolFun=1e-10)

        x, Fx, flag, output, _ = lmmcp(debt_limit_residual, x0, lb, ub, opts)

        @test flag == 1
        # Check complementarity: THETA ≥ 0, BLIM ≥ 0, THETA*BLIM ≈ 0
    end
end
```

**Step 1.3: Validation against Dynare** (2-3 days)

Create test problems with known solutions from Dynare:

1. Download Dynare test suite MCP examples
2. Port 3-5 test cases to Julia
3. Run both Dynare LMMCP and Julia LMMCP
4. Compare:
   - Final solution vectors
   - Number of iterations
   - Function evaluations
   - Final residual norms

**Acceptance criteria**: Julia LMMCP matches Dynare LMMCP to within 1e-6 relative error on all test cases.

### 3.2 Phase 2: SEP Integration (1-2 weeks)

**Step 2.1: Modify solve_deterministic_path** (3-4 days)

In `src/sep_solver.jl`, around line 620:

```julia
function solve_deterministic_path(
    𝓂::ℳ,
    parameters::Vector{Float64},
    opts::SEPSolverOptions,
    initial_guess::Union{Nothing,Vector{Float64}},
    yss::Vector{Float64},
    SS_and_pars::Vector{Float64},
    SS_result,
    initial_state::Union{Nothing,Vector{Float64}}=nothing;
    use_mcp::Bool=false,  # NEW: Enable MCP solver
    var_bounds::Union{Nothing,NamedTuple}=nothing  # NEW: Specify bounds
)
    # ... existing setup code ...

    # NEW: If use_mcp=true and bounds specified, use LMMCP
    if use_mcp && !isnothing(var_bounds)
        return solve_deterministic_path_mcp(
            𝓂, parameters, opts, initial_guess, yss, SS_and_pars,
            SS_result, initial_state, var_bounds
        )
    end

    # ... existing Newton solver (fallback) ...
end

function solve_deterministic_path_mcp(
    𝓂::ℳ,
    parameters::Vector{Float64},
    opts::SEPSolverOptions,
    initial_guess::Union{Nothing,Vector{Float64}},
    yss::Vector{Float64},
    SS_and_pars::Vector{Float64},
    SS_result,
    initial_state::Union{Nothing,Vector{Float64}},
    var_bounds::NamedTuple  # (lb, ub) for each variable
)
    # Build residual/Jacobian function for LMMCP

    # Define bounds for stacked problem
    # Y = [y₁, y₂, ..., yT] (each yₜ has ny_ variables)
    neq = ny_ * T
    lb = repeat(var_bounds.lb, T)
    ub = repeat(var_bounds.ub, T)

    # Residual function for LMMCP
    function mcp_residual(Y_flat, args...)
        # Reshape Y_flat to (ny_ × T)
        # Compute F(Y) = stacked dynamic equations
        # Compute Jacobian J
        return F, J
    end

    # Initial guess
    Y0 = isnothing(initial_guess) ? repeat(yss, T) : initial_guess

    # Solve with LMMCP
    lmmcp_opts = LMMCP.LMMCPOptions(
        MaxIter = opts.maxit,
        TolFun = opts.tol,
        Display = opts.verbose ? :iter : :none
    )

    Y_sol, F_sol, exitflag, output, J_sol = LMMCP.lmmcp(
        mcp_residual,
        Y0,
        lb,
        ub,
        lmmcp_opts
    )

    # Convert output to MacroModelling format
    flag = exitflag == 1 ? 0 : (exitflag == 0 ? 1 : 2)

    return (flag=flag, Y=Y_sol, layout=layout, err=output.Psix)
end
```

**Step 2.2: Add bound specification for QMIPF** (1-2 days)

In model file or solver call:

```julia
# Specify which variables have bounds
var_bounds = (
    lb = fill(-Inf, ny_),  # Default: unbounded
    ub = fill(Inf, ny_)
)

# Set bounds for OBC variables
theta_idx = findfirst(==(Symbol("THETA")), m.var)
blim_idx = findfirst(==(Symbol("BLIM")), m.var)

var_bounds.lb[theta_idx] = 0.0  # THETA ≥ 0
var_bounds.lb[blim_idx] = 0.0   # BLIM ≥ 0

# Solve with MCP
solve!(m,
       algorithm=:stochastic_extended_path,
       sep_order=0,  # Deterministic mode
       sep_use_mcp=true,  # Enable MCP solver
       sep_var_bounds=var_bounds)
```

**Step 2.3: Create QMIPF test script** (2-3 days)

Create `scripts/test_qmipf_with_mcp.jl`:

```julia
using MacroModelling
include("../models/QMIPF_final.jl")
m = QMIPF_step9e_Real_UIP

# Test 1: Large negative shock (should bind constraint)
println("Test 1: Large negative shock (BLIM should hit 0)")

# Specify bounds
theta_idx = findfirst(==(Symbol("THETA")), m.var)
blim_idx = findfirst(==(Symbol("BLIM")), m.var)

var_bounds = (
    lb = fill(-Inf, length(m.var)),
    ub = fill(Inf, length(m.var))
)
var_bounds.lb[theta_idx] = 0.0
var_bounds.lb[blim_idx] = 0.0

# Create large negative shock sequence
shock_seq = zeros(20, length(m.exo))
eps_y_idx = findfirst(==(Symbol("EPS_Y_ST")), m.exo)
shock_seq[1, eps_y_idx] = -5.0  # Large negative shock

# Solve with MCP
solve!(m,
       algorithm=:stochastic_extended_path,
       sep_order=0,
       sep_periods=20,
       sep_deterministic_shocks=shock_seq,
       sep_use_mcp=true,
       sep_var_bounds=var_bounds,
       silent=false)

# Extract results
sep_sol = m.solution.perturbation.stochastic_extended_path
layout = sep_sol.layout
Y = sep_sol.Y

# Get THETA and BLIM paths
theta_path = [Y[layout.voff[t+1] + theta_idx] for t in 1:20]
blim_path = [Y[layout.voff[t+1] + blim_idx] for t in 1:20]

println("\nResults:")
println("  BLIM range: ", extrema(blim_path))
println("  THETA range: ", extrema(theta_path))
println("  Periods with BLIM ≈ 0: ", count(x -> x < 1e-4, blim_path))
println("  Periods with THETA > 0: ", count(x -> x > 1e-4, theta_path))
println("  Complementarity check: THETA·BLIM = ",
        [theta_path[t] * blim_path[t] for t in 1:20])

# Test 2: Compare with Dynare
println("\nTest 2: Compare with Dynare QMIPF results")
# Load Dynare results (if available)
# Compare IRFs
```

### 3.3 Phase 3: Validation & Documentation (3-5 days)

**Step 3.1: Validation tests**

1. **Convergence tests**: Does LMMCP converge when standard Newton fails?
2. **Accuracy tests**: Does THETA spike properly when BLIM < 0?
3. **Performance tests**: How does LMMCP compare to standard Newton in speed?
4. **Robustness tests**: Different penalty_kappa values, different shocks

**Step 3.2: Documentation**

Create `Documentation/LMMCP_USER_GUIDE.md`:

```markdown
# Using LMMCP for OBC Enforcement in MacroModelling.jl

## When to Use MCP Solver

Use the MCP solver when:
- Your model has occasionally binding constraints (OBC)
- You need complementarity enforcement: THETA ≥ 0, BLIM ≥ 0, THETA·BLIM = 0
- Standard SEP gives THETA ≈ 0 when constraint should bind

## Basic Usage

...
```

---

## Part 4: Technical Challenges

### 4.1 Translation Challenges (MATLAB → Julia)

#### Challenge 1.1: Condition number estimation

**MATLAB**: `condest(A)` - iterative condition estimator
**Julia**: `cond(A)` - computes SVD (expensive for large matrices)

**Solution**: Use iterative condition estimator from `LinearAlgebra` or `IterativeSolvers.jl`:

```julia
# Option A: Use full cond() for small problems (n < 100)
if n < 100
    κ = cond(DPhix' * DPhix)
end

# Option B: Use norm-based heuristic
κ_est = norm(DPhix' * DPhix) * norm(inv(DPhix' * DPhix))

# Option C: Use IterativeSolvers.jl
using IterativeSolvers
κ_est = 1.0 / minimum(svdvals(DPhix' * DPhix))
```

**Impact**: Medium - affects LM parameter choice, but heuristics work

#### Challenge 1.2: Sparse identity matrix

**MATLAB**: `speye(n)` returns sparse identity
**Julia**: Must use `sparse(I, n, n)` or `SparseArrays.sparse(1:n, 1:n, 1.0)`

**Solution**:
```julia
using SparseArrays, LinearAlgebra
speye(n) = sparse(I, n, n)
```

**Impact**: Easy

#### Challenge 1.3: Logical indexing differences

**MATLAB**: `x(I)` where `I` is logical vector
**Julia**: `x[I]` works the same

**MATLAB**: `x(I) = value` assigns to selected elements
**Julia**: `x[I] .= value` (need broadcast for scalar)

**Solution**: Be careful with broadcasting `.=` vs `=`

**Impact**: Easy, just syntax differences

### 4.2 Numerical Challenges

#### Challenge 2.1: Sparsity structure

**Problem**: QMIPF with T=20 periods has 2400 equations. Dense Jacobian of NCP function `DΦ` would be 2400×2400 = 5.76M elements.

**LMMCP approach**: Uses sparse Jacobian construction in `DPhi` function

**MacroModelling integration**: Need to preserve sparsity through:
1. Dynamic equation Jacobian (already sparse in MacroModelling)
2. NCP function Jacobian (LMMCP constructs sparse)
3. Normal equations `J'*J` (may densify!)

**Solution**: Use sparse-aware normal equations or iterative solvers

#### Challenge 2.2: Ill-conditioning near complementarity

**Problem**: When BLIM → 0, the system becomes poorly conditioned:
- Jacobian has near-zero eigenvalues
- LM regularization is critical
- Condition number >> 1e15

**LMMCP handles this**: Adaptive μ based on condition number

**Solution**: Ensure LM parameter adapts correctly in Julia implementation

#### Challenge 2.3: Multiple equilibria

**Problem**: Complementarity problems can have multiple solutions:
- THETA = 0, BLIM > 0 (slack)
- THETA > 0, BLIM = 0 (binding)
- Both valid locally, but only one is economically correct

**LMMCP approach**: Initial guess and watchdog strategy guide to "best" solution

**Solution**: Provide good initial guess from steady state or previous time period

### 4.3 Integration Challenges

#### Challenge 3.1: MacroModelling's equation structure

**Current**: Equations written as `F(y_{t-1}, y_t, y_{t+1}, ε_t) = 0`

**MCP needs**: Equation-variable pairing for bounds
- Which equations involve THETA?
- Which equations involve BLIM?
- How to map bounds to stacked system?

**Solution**:

Option A (Simple): Apply bounds to ALL periods uniformly
```julia
lb = repeat(var_bounds_single, T)
```

Option B (Advanced): Time-varying bounds (e.g., constraint only binds after shock)

#### Challenge 3.2: OBC shock interaction

**Current**: MacroModelling adds ϵᵒᵇᶜ anticipated shocks for OBC

**MCP approach**: Directly enforces complementarity

**Potential conflict**: Both systems trying to handle OBC differently

**Solution**: When `sep_use_mcp=true`, disable MacroModelling's OBC system:
```julia
if use_mcp
    # Don't parse OBC, treat max() as regular function
    # Let LMMCP handle constraint enforcement
end
```

#### Challenge 3.3: Stochastic SEP (order > 0)

**Problem**: LMMCP is designed for deterministic systems. How to handle:
- Branching tree with multiple scenarios?
- Expectations over child nodes?
- Different groups having different constraint states?

**Analysis**: This is **very hard** and may not be feasible in Phase 1

**Solution**: Phase 1 only supports `sep_order=0` (deterministic mode)

Future work could explore:
- Node-specific MCP at each branch point
- Separate LMMCP solve per scenario
- Hybrid: MCP for trunk, standard Newton for branches

---

## Part 5: Alternative Approaches

### 5.1 Julia MCP Packages

Instead of porting LMMCP, use existing Julia MCP solvers:

#### Option A: PATHSolver.jl

**Package**: https://github.com/chkwon/PATHSolver.jl
**Solver**: PATH (Dirkse & Ferris) - industry standard MCP solver
**License**: Requires PATH license (free for academic use)

**Pros**:
- ✅ Mature, well-tested
- ✅ Fast and robust
- ✅ Already in Julia ecosystem
- ✅ Used by QuantEcon.jl

**Cons**:
- ❌ Non-standard license (PATH license required)
- ❌ Binary dependencies (compiled Fortran)
- ❌ Interface may not match MacroModelling structure
- ❌ Requires C interface wrapping

**Integration effort**: 1-2 weeks (similar to LMMCP)

#### Option B: Complementarity.jl

**Package**: https://github.com/chkwon/Complementarity.jl
**Approach**: Reformulates MCP as nonlinear optimization using NLsolve.jl or JuMP.jl

**Pros**:
- ✅ Pure Julia
- ✅ No external dependencies
- ✅ Easy to install and use
- ✅ MIT license

**Cons**:
- ❌ Reformulation overhead (MCP → optimization)
- ❌ May be slower than specialized MCP solvers
- ❌ Less mature than PATH/LMMCP

**Integration effort**: 1 week

#### Option C: NLsolve.jl with bounds

**Package**: https://github.com/JuliaNLSolvers/NLsolve.jl
**Approach**: Use trust region method with box constraints

**Pros**:
- ✅ Already used in Julia ecosystem
- ✅ Pure Julia, MIT license
- ✅ Good documentation

**Cons**:
- ❌ NOT an MCP solver (doesn't enforce complementarity)
- ❌ May not handle `THETA·BLIM = 0` properly
- ❌ Would need manual reformulation

**Integration effort**: 3-5 days, but **unlikely to work** for true complementarity

### 5.2 Reformulation Approaches

Instead of MCP solver, reformulate the problem:

#### Approach A: Smooth penalty method

**Idea**: Replace complementarity with smooth penalty:

```julia
# Instead of: THETA ≥ 0, BLIM ≥ 0, THETA·BLIM = 0
# Use: ψ = THETA·BLIM + (1/μ)·(min(THETA,0)² + min(BLIM,0)²)
# Add to objective: minimize 0.5||F||² + μ·ψ
```

**Pros**:
- ✅ Can use standard nonlinear solver
- ✅ No new solver needed

**Cons**:
- ❌ Approximation (not exact complementarity)
- ❌ Requires tuning μ
- ❌ **Already tried** - this is what penalty method does!

**Verdict**: Unlikely to work better than current approach

#### Approach B: Fischer-Burmeister reformulation

**Idea**: Replace MCP with nonlinear equations using FB function:

```julia
# Original MCP: THETA ≥ 0, BLIM ≥ 0, THETA·BLIM = 0
# FB reformulation: sqrt(THETA² + BLIM²) - THETA - BLIM = 0
```

**Pros**:
- ✅ Exact reformulation
- ✅ Can use standard Newton solver

**Cons**:
- ❌ FB function is non-smooth (not differentiable at 0)
- ❌ Standard Newton may struggle
- ❌ **This is what LMMCP does internally!**

**Verdict**: Essentially recreating LMMCP - better to port LMMCP directly

#### Approach C: Interior point method

**Idea**: Use barrier functions to enforce bounds:

```julia
# Add log barriers: -μ·(log(THETA) + log(BLIM))
# Solve with decreasing μ → 0
```

**Pros**:
- ✅ Standard technique
- ✅ Works with Newton solver

**Cons**:
- ❌ Requires strictly interior initial point
- ❌ Multiple solves (continuation in μ)
- ❌ Expensive for large systems

**Verdict**: Possible, but more complex than LMMCP

### 5.3 Comparison Matrix

| Approach | Implementation Time | Likely to Work | Maintenance | Performance |
|----------|-------------------|----------------|-------------|-------------|
| **Port LMMCP** | 3-4 weeks | ✅ High | Medium | ⭐⭐⭐⭐ |
| **PATHSolver.jl** | 2-3 weeks | ✅ High | Low (external) | ⭐⭐⭐⭐⭐ |
| **Complementarity.jl** | 1-2 weeks | ⚠️ Medium | Low | ⭐⭐⭐ |
| **NLsolve.jl bounds** | 3-5 days | ❌ Low | Low | ⭐⭐ |
| **Smooth penalty** | 1-2 days | ❌ Low (already tried) | N/A | ⭐⭐ |
| **FB reformulation** | 1-2 weeks | ⚠️ Medium | Medium | ⭐⭐⭐ |
| **Interior point** | 2-3 weeks | ⚠️ Medium | High | ⭐⭐ |

---

## Part 6: Recommendations

### 6.1 Short-term (Next 2-4 weeks)

**Recommendation**: **Port LMMCP to Julia** for deterministic SEP mode

**Rationale**:
1. ✅ Most likely to work (proven algorithm for MCP)
2. ✅ Self-contained (no external dependencies)
3. ✅ Clear license (unlimited permission)
4. ✅ Solves user's immediate problem (QMIPF with OBC)
5. ✅ Educational value (understand MCP solving)

**Scope**:
- Phase 1: Port LMMCP (~800-1000 lines Julia)
- Phase 2: Integrate with `solve_deterministic_path` (order=0 mode)
- Phase 3: Test with QMIPF model, compare to Dynare

**Limitations accepted**:
- Only works with `sep_order=0` (deterministic mode)
- Stochastic SEP (order > 0) still won't enforce OBC
- User must specify variable bounds explicitly

**Success criteria**:
- QMIPF model solves with BLIM hitting 0
- THETA spikes when constraint binds
- Results match (or improve) Dynare LMMCP

### 6.2 Medium-term (1-3 months)

After successful LMMCP integration, consider:

**Option 1: Extend to stochastic SEP** (if needed by research)

**Approach**: MCP at each branch point
- Modify `sep_solve_mm!` to use LMMCP for branching nodes
- Handle expectations over child nodes with MCP
- Complex, but feasible

**Effort**: 3-4 weeks additional

**Option 2: Performance optimization**

- Sparse matrix optimization in LMMCP
- Precompute index sets
- Parallel Jacobian evaluation
- Custom LM parameter heuristics for DSGE models

**Effort**: 1-2 weeks

**Option 3: Alternative Julia MCP solvers**

- Try PATHSolver.jl as alternative backend
- Benchmark LMMCP vs PATH vs Complementarity.jl
- Provide user choice of MCP solver

**Effort**: 1-2 weeks

### 6.3 Long-term (6+ months)

**Contribute back to MacroModelling.jl**

Once LMMCP integration is stable and tested:

1. Open PR to MacroModelling.jl repository
2. Propose MCP solver as optional backend
3. Add documentation for OBC with MCP
4. Provide QMIPF as example model

**Benefits**:
- Other researchers can use MCP for OBC
- Community maintenance and testing
- Potential performance improvements from community

---

## Part 7: Implementation Timeline

### Detailed Schedule (3-4 weeks)

**Week 1: LMMCP Core Port**

| Day | Task | Hours | Deliverable |
|-----|------|-------|-------------|
| Mon | Setup: Create `src/lmmcp.jl`, study algorithm | 4 | Module structure |
| Tue | Translate initialization (lines 100-243) | 6 | Initialization code |
| Wed | Translate Phase I preprocessor (lines 245-338) | 6 | Phase I working |
| Thu | Translate Phase II main loop (lines 340-435) | 6 | Phase II working |
| Fri | Translate Phi function (lines 460-481) | 4 | NCP function working |

**Week 2: LMMCP Subfunctions & Testing**

| Day | Task | Hours | Deliverable |
|-----|------|-------|-------------|
| Mon | Translate DPhi function (lines 484-625) | 8 | Full LMMCP translated |
| Tue | Fix compilation errors, type issues | 6 | Code compiles |
| Wed | Create unit tests, simple MCP problems | 6 | Tests passing |
| Thu | Validate against Dynare test cases | 6 | Match Dynare results |
| Fri | Debug numerical issues, tune parameters | 4 | Robust on test cases |

**Week 3: SEP Integration**

| Day | Task | Hours | Deliverable |
|-----|------|-------|-------------|
| Mon | Create `solve_deterministic_path_mcp` function | 6 | Integration function |
| Tue | Add bound specification interface | 4 | User API for bounds |
| Wed | Connect to MacroModelling solver dispatch | 4 | Full integration |
| Thu | Create QMIPF test script | 4 | Test on real model |
| Fri | Debug integration issues | 6 | QMIPF solving with MCP |

**Week 4: Validation & Documentation**

| Day | Task | Hours | Deliverable |
|-----|------|-------|-------------|
| Mon | Run QMIPF with various shocks, compare to Dynare | 6 | Validation results |
| Tue | Test convergence, accuracy, performance | 6 | Benchmark results |
| Wed | Write user guide and documentation | 6 | LMMCP_USER_GUIDE.md |
| Thu | Create examples and tutorials | 4 | Example scripts |
| Fri | Final testing, bug fixes, cleanup | 6 | Production ready |

**Total effort**: ~120 hours (3-4 weeks full-time)

### Milestones

| Milestone | Date | Success Criteria |
|-----------|------|------------------|
| M1: LMMCP Translated | End Week 1 | Code compiles, runs on simple problem |
| M2: LMMCP Validated | End Week 2 | Matches Dynare on 5 test cases |
| M3: SEP Integrated | End Week 3 | QMIPF solves with `sep_use_mcp=true` |
| M4: Production Ready | End Week 4 | Documentation complete, all tests pass |

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| DPhi function too complex | Medium | High | Start with simplified version, iterate |
| Numerical instability | Medium | High | Extensive testing, parameter tuning |
| Integration breaks existing SEP | Low | High | Keep original solver intact, new code path |
| Performance too slow | Medium | Medium | Profile and optimize, use sparse matrices |
| QMIPF still doesn't converge | Medium | High | Fallback to PATHSolver.jl or reformulation |

---

## Part 8: Cost-Benefit Analysis

### Costs

**Development time**: 120 hours (3-4 weeks)

**Maintenance**:
- Initial: 10-20 hours (bug fixes, documentation updates)
- Ongoing: 2-5 hours/month (user support, edge cases)

**Risk of failure**: ~20-30% (numerical issues, integration problems)

**Alternative cost**: If LMMCP fails, fallback to PATHSolver.jl adds 1-2 weeks

### Benefits

**Research unblocked**:
- ✅ QMIPF model works with OBC enforcement
- ✅ Can calibrate debt limit properly
- ✅ Nonlinear SEP solution with complementarity
- ✅ Results comparable to Dynare

**Technical advancement**:
- ✅ First MCP solver integration in MacroModelling.jl
- ✅ Enables broader class of models (inequality constraints)
- ✅ Opens door for financial frictions, ZLB, capacity constraints

**Scientific contribution**:
- ✅ Potential publication on MCP methods in Julia DSGE tools
- ✅ Could be contributed back to MacroModelling.jl
- ✅ Advances open-source macro tooling

### ROI Calculation

**Immediate value** (assuming research paper publication):
- Paper value: High (PhD dissertation chapter, publication)
- Time saved: ~2-4 weeks vs implementing model in Dynare
- Quality improvement: Native Julia integration, faster iteration

**Long-term value**:
- Other models with OBC become feasible
- Tool can be reused for future research
- Contribution to research community

**Verdict**: **High ROI** if research requires nonlinear OBC enforcement

---

## Part 9: Final Recommendation

### Executive Decision

**PROCEED with LMMCP porting** under these conditions:

1. ✅ **User commits 3-4 weeks** to implementation and testing
2. ✅ **Acceptance criteria**: QMIPF THETA spikes when BLIM < 0
3. ✅ **Fallback plan**: If LMMCP fails, try PATHSolver.jl (2 weeks additional)
4. ✅ **Scope limitation**: Phase 1 only addresses deterministic SEP (order=0)
5. ✅ **Documentation**: Clear user guide and limitations documented

### Phased Approach

**Phase 1** (3-4 weeks): Port LMMCP, integrate with deterministic SEP
**Success metric**: QMIPF works with `sep_order=0, sep_use_mcp=true`

**Phase 2** (optional, 1-2 weeks): Performance optimization
**Success metric**: Solve time < 2x standard Newton

**Phase 3** (optional, 3-4 weeks): Extend to stochastic SEP
**Success metric**: Works with `sep_order=1` (single-period branching)

### Decision Point

**After Week 2**: Evaluate LMMCP validation results
- ✅ If matches Dynare on test cases → Continue to integration
- ❌ If numerical issues persist → Pivot to PATHSolver.jl

**After Week 3**: Evaluate QMIPF integration
- ✅ If THETA spikes correctly → Continue to production
- ❌ If still not enforcing → Debug or pivot to alternative

### Alternative Quick Win (1 week)

If full LMMCP port seems too risky, try **Complementarity.jl first**:

**Week 1 only**:
1. Install Complementarity.jl (1 hour)
2. Reformulate QMIPF MCP problem (1 day)
3. Test with various shocks (2 days)
4. Compare to standard SEP results (1 day)

**Decision**: If Complementarity.jl works, save 2-3 weeks vs LMMCP port

**If not**: Still have time to port LMMCP (original plan)

---

## Conclusion

### Summary

**Porting LMMCP to Julia is feasible** and represents the best path forward for:

1. ✅ Enabling proper OBC enforcement in MacroModelling.jl
2. ✅ Solving the user's QMIPF model with debt limit constraint
3. ✅ Providing a foundation for broader class of complementarity problems

**Key success factors**:
- Self-contained code with no dependencies
- Clear algorithm description and references
- Unlimited-use license from original authors
- Existing validation test cases from Dynare

**Recommended path**:
1. Start with LMMCP port (3-4 weeks)
2. Integrate with deterministic SEP (order=0 mode)
3. Test extensively with QMIPF model
4. Document limitations and usage

**Risk management**:
- Keep original solver as fallback
- Decision points at Week 2 and Week 3
- Alternative (Complementarity.jl or PATHSolver.jl) if needed

### Next Steps

If user approves:

1. **Create project plan** with detailed task breakdown
2. **Set up LMMCP development branch** in MacroModelling
3. **Begin Week 1 tasks** (module structure, initialization)
4. **Weekly progress reports** with blockers identified early
5. **Decision point meetings** at Week 2 and Week 3

---

**Status**: ✅ Assessment complete
**Recommendation**: PROCEED with LMMCP porting (3-4 week implementation)
**Alternative**: Try Complementarity.jl first (1 week quick test)

