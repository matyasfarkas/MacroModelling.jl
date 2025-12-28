# Fishbone Sparse Tree Implementation - Progress Report

**Date**: December 27, 2024
**Status**: IN PROGRESS - Core structures implemented, equation assembly next

## Summary

Implementing the Adjemian-Juillard fishbone sparse tree method in MacroModelling.jl to reduce computational complexity from O(K^(p×H)) to O(p×m×H) where:
- K = nnodes^H (tensor product)
- m = nnodes (sparse monomial)
- p = branching order
- H = number of shocks

**Example**: SW07 model with H=7 shocks, p=1, m=3:
- Full tree: 3^7 = 2,187 nodes
- Sparse tree: 1 + 7×(3-1) = 15 nodes
- **146× reduction!**

## Completed Components ✅

### 1. Data Structures

**SEPLayout struct** (lines 34-51 in `sep_solver.jl`):
```julia
struct SEPLayout
    T::Int                    # Periods
    Lbr::Int                  # Branching order
    K::Int                    # Total GH nodes (full tree: nnodes^dε, sparse: dε*m)
    G::Vector{Int}            # Groups at each time
    voff::Vector{Int}         # Variable offsets
    eoff::Vector{Int}         # Shock offsets
    ny_::Int                  # Number of variables
    dε::Int                   # Number of shocks
    sparse::Bool              # Fishbone sparse tree flag
    m::Int                    # Nodes per dimension (sparse tree)
end
```

- Added `sparse::Bool` and `m::Int` fields
- Added backward-compatible constructor (line 48-51)

**SEPSolverOptions struct** (lines 9-31):
```julia
struct SEPSolverOptions
    periods::Int
    order::Int
    nnodes::Int
    maxit::Int
    tol::Float64
    verbose::Bool
    shock_scale::Float64
    sparse_tree::Bool    # NEW: Enable sparse tree
end
```

### 2. Navigation Functions

**Sparse tree navigation** (lines 73-125):
- `is_trunk_node(layout, t, g)` - Check if node is on trunk
- `parent_group_sparse(layout, t, g)` - Parent in sparse tree
- `child_groups_sparse(layout, t, g)` - Children in sparse tree

**Key insight**: Only trunk (group 1) branches. Side branches are deterministic.

### 3. Quadrature Functions

**1D Gauss-Hermite** (lines 181-194):
```julia
function gh_nodes_1d(nnodes::Int)
    # Returns nodes and weights for single dimension
    # Supports nnodes = 1, 3, 5
end
```

**Sparse shock nodes** (lines 196-239):
```julia
function build_sparse_shock_nodes(m::Int, dε::Int, Σ::Matrix{Float64})
    # Monomial rule: H dimensions × m nodes = H*m total nodes
    # Returns: X_sparse, W_sparse, shock_map
end
```

**Monomial vs Tensor**:
- Full tree: All combinations of shocks → m^H nodes
- Sparse tree: One shock at a time → H*m nodes

### 4. Tree Construction

**Modified tree builder** (lines 383-456):
```julia
if opts.sparse_tree
    # Sparse: monomial rule
    X, W, shock_map = build_sparse_shock_nodes(m, dε, Σ)
    K = dε * m

    # Groups: 1 trunk + H*(m-1) side branches
    G_branch = 1 + dε*(m-1)
    for t in branching periods
        G[t+1] = G_branch  # Same groups at each period
    end
else
    # Full: tensor product
    X, W = gh_tensor_nodes_weights(m, dε)
    K = m^dε

    for t in branching periods
        G[t+1] = K^t  # Exponential growth
    end
end
```

## Remaining Work 🚧

### 5. Equation Assembly (CRITICAL - NEXT STEP)

**Location**: Lines ~490-630 in `sep_solver.jl`

**Current structure**:
```julia
for t in 1:T
    for g in 1:Gt
        if t <= Lbr && dε > 0
            # Branching node: expectation over children
            for (kidx, cg) in enumerate(child_groups)
                r += W[kidx] * residual
            end
        else
            # Non-branching node: deterministic
            r = residual
        end
    end
end
```

**Needed modifications**:

1. **Use sparse navigation functions**:
   - Replace `child_groups()` with `child_groups_sparse()` for sparse tree
   - Replace `parent_group()` with `parent_group_sparse()`

2. **Shock extraction for sparse tree**:
   - Full tree: `k_shock = mod(g - 1, K) + 1`
   - Sparse tree: Need to map group `g` to shock dimension `h` and node index `k`

3. **Weight indexing**:
   - Full tree: `W[kidx]` from tensor product
   - Sparse tree: `W[shock_idx]` from monomial rule

**Pseudocode for sparse tree equation assembly**:
```julia
if layout.sparse
    if is_trunk_node(layout, t, g)
        # Trunk: branches over all shock dimensions
        for h in 1:dε  # Each shock dimension
            for k in 1:m  # Each GH node
                shock_idx = shock_map[(h, k)]
                ε_curr = X[:, shock_idx]
                w = W[shock_idx]

                cg = get_child_group_for_shock(h, k)
                # Compute residual with this shock
                r += w * residual
            end
        end
    else
        # Side branch: deterministic continuation
        ε_curr = extract_shock_for_branch(g)
        r = residual  # No expectation
    end
else
    # Full tree logic (existing code)
end
```

### 6. Helper Functions Needed

**Group-to-shock mapping** (for sparse tree):
```julia
function get_shock_from_group(layout::SEPLayout, g::Int)
    # For side branch group g, return (shock_dim, node_index)
    # Group 1 = trunk
    # Groups 2 to 1+H*(m-1) = side branches
    !layout.sparse && error("Only for sparse tree")

    if g == 1
        return nothing  # Trunk
    else
        # Map g to (h, k)
        branch_idx = g - 1  # 1-indexed to 0-indexed
        h = div(branch_idx - 1, m - 1) + 1  # Shock dimension
        k = mod(branch_idx - 1, m - 1) + 2  # Node index (skip k=1 which is trunk)
        return (h, k)
    end
end

function get_child_group_for_shock(h::Int, k::Int, m::Int)
    # Given shock dimension h and node k, return child group index
    if k == 1
        return 1  # Zero-shock node stays on trunk
    else
        return 1 + (h-1)*(m-1) + (k-1)  # Side branch
    end
end
```

### 7. Validation Tests

**test_sparse_tree_rbc.jl** (to be created):
```julia
# Compare full vs sparse tree on RBC model
solve!(RBC_Dynare, algorithm=:stochastic_extended_path,
       sep_periods=10, sep_order=1, sep_nnodes=3,
       sparse_tree=false)
yss_full = extract_steady_state(RBC_Dynare)

solve!(RBC_Dynare, algorithm=:stochastic_extended_path,
       sep_periods=10, sep_order=1, sep_nnodes=3,
       sparse_tree=true)
yss_sparse = extract_steady_state(RBC_Dynare)

# Check: steady states should match
@test maximum(abs.(yss_full - yss_sparse)) < 1e-6
```

### 8. High-Level API Integration

**MacroModelling.jl solve!()** - Add parameter:
```julia
function solve!(model;
                algorithm=:first_order,
                sep_periods=20,
                sep_order=1,
                sep_nnodes=3,
                sep_sparse_tree=false,  # NEW
                ...)
    if algorithm == :stochastic_extended_path
        opts = SEPSolverOptions(
            periods=sep_periods,
            order=sep_order,
            nnodes=sep_nnodes,
            sparse_tree=sep_sparse_tree  # Pass through
        )
        # ...
    end
end
```

## File Modifications

### Modified Files ✅

1. **src/sep_solver.jl** (~250 lines changed):
   - SEPLayout: +2 fields, +constructor
   - SEPSolverOptions: +1 field
   - Navigation: +3 functions (50 lines)
   - Quadrature: +2 functions (75 lines)
   - Tree construction: ~70 lines modified

### Files to Modify 🚧

2. **src/sep_solver.jl** (equation assembly):
   - Lines 490-630: Add sparse tree logic
   - Estimate: +150 lines

3. **src/MacroModelling.jl**:
   - solve!() function: Add sep_sparse_tree parameter
   - Estimate: +5 lines

### Files to Create 🚧

4. **test_sparse_tree_rbc.jl**:
   - Validation test
   - Estimate: ~100 lines

5. **test_sparse_tree_sw07.jl**:
   - Stress test on large model
   - Estimate: ~80 lines

## Testing Strategy

### Phase 1: Unit Tests
1. ✅ Test SEPLayout construction
2. ✅ Test navigation functions
3. ✅ Test sparse shock node generation
4. 🚧 Test group-to-shock mapping

### Phase 2: Integration Tests
1. 🚧 RBC model: Full vs Sparse (should match)
2. 🚧 FS2000 model: Full vs Sparse
3. 🚧 SW07 model: Sparse tree (full tree may be too large)

### Phase 3: Performance Tests
1. 🚧 Measure solve time: Full vs Sparse
2. 🚧 Measure memory usage
3. 🚧 Validate complexity reduction

## Complexity Analysis

**Full tree** (current implementation):
- Nodes per period t ≤ p: G(t) = K^t = (m^H)^t
- Total nodes: ∑(t=0 to p) K^t = O(K^p) = O(m^(H×p))
- For SW07: H=7, m=3, p=1 → 3^7 = 2,187 nodes

**Sparse tree** (fishbone):
- Nodes per period: G(t) = 1 + H×(m-1)
- Total nodes: (p+1) × [1 + H×(m-1)] = O(p×m×H)
- For SW07: H=7, m=3, p=1 → 1 + 7×2 = 15 nodes
- **Reduction factor**: 2,187 / 15 = **146×**

## Next Immediate Steps

1. **Implement helper functions** (30 min):
   - `get_shock_from_group(layout, g)`
   - `get_child_group_for_shock(h, k, m)`

2. **Modify equation assembly** (2-3 hours):
   - Lines 490-630 in sep_solver.jl
   - Add sparse tree branch in residual computation
   - Update shock extraction logic
   - Update weight indexing

3. **Test compilation** (30 min):
   - Create simple test file
   - Verify code compiles without errors

4. **Create validation test** (1 hour):
   - test_sparse_tree_rbc.jl
   - Compare full vs sparse on RBC model

5. **Debug and iterate** (2-4 hours):
   - Fix indexing bugs
   - Verify steady states match
   - Check Newton convergence

**Total estimated time**: 6-9 hours of focused work

## References

- **Documentation**: `FISHBONE_SPARSE_TREE.md`
- **Theory**: Adjemian-Juillard (2025), SEP working paper
- **Dynare reference**: `/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SW07_development/ep-mj-30-years-master/`

## Notes

- Sparse tree is an **approximation** to full tree
- Accuracy depends on shock correlations
- Best for models with many independent shocks
- SW07 perfect use case: 7 shocks, would be 3^7 = 2,187 → 15 nodes
