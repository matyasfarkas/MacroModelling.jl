# SEP IRF Shock Size Scaling Fix

**Date**: December 25, 2024
**Issue**: SEP IRFs need to be scaled to match requested shock size

## Problem

SEP solutions are computed at specific Gauss-Hermite (GH) quadrature nodes:
- For `nnodes=3`: nodes at [-√3, 0, +√3] standard deviations
- For `nnodes=5`: nodes at more complex locations

When a user requests an IRF for a 1.0σ shock, but the SEP solution only has responses at the √3σ node, the IRF needs to be scaled to match the requested magnitude.

## Solution

Added shock size scaling to the IRF extraction process:

### 1. Calculate Scale Factor (sep_irf.jl lines 85-100)

```julia
# For nnodes=3, GH nodes are at [-√3, 0, √3] standard deviations
# We need to scale the IRF to match the requested shock_size
if nnodes == 3
    gh_nodes_1d = [-√3, 0.0, √3]
    actual_shock_std = abs(gh_nodes_1d[target_node_idx])
elseif nnodes ==5
    gh_nodes_1d = [-√(5+2√(10/7)), -√(5-2√(10/7)), 0.0, √(5-2√(10/7)), √(5+2√(10/7))]
    actual_shock_std = abs(gh_nodes_1d[target_node_idx])
else
    # Default: no scaling for unknown nnodes
    actual_shock_std = abs(shock_size)
end

# Scaling factor to convert from GH node magnitude to requested shock_size
scale_factor = abs(shock_size) / max(actual_shock_std, 1e-10)
```

### 2. Return Scale Factor (sep_irf.jl line 122)

Changed return value of `extract_sep_irf`:

```julia
# Before:
return irf

# After:
return irf, scale_factor
```

### 3. Apply Scaling (sep_irf.jl lines 182-189)

Modified `get_sep_irf` to apply the scaling factor:

```julia
# Extract IRF and scale factor
irf, scale_factor = extract_sep_irf(sep_sol, shock_idx, shock_size, var_indices)

# Convert to deviations from steady state and apply scaling
for i in 1:size(irf, 1)
    ss_val = irf[i, 1]
    irf[i, :] .-= ss_val
    irf[i, :] .*= scale_factor  # Scale to match requested shock_size
end
```

## How It Works

1. **Identify target GH node**: Based on shock sign (positive/negative), select the appropriate GH node
2. **Calculate actual shock magnitude**: Get the standard deviation magnitude at that node (e.g., √3 for nnodes=3)
3. **Compute scale factor**: `scale_factor = requested_shock_size / actual_node_std`
4. **Apply to IRF deviations**: Multiply all deviation values by the scale factor

## Example

For `nnodes=3` and a 1.0σ shock request:
- GH node is at √3σ ≈ 1.732σ
- Scale factor = 1.0 / 1.732 ≈ 0.577
- IRF values are scaled down by factor of 0.577

This ensures the IRF represents a 1.0σ shock, not a 1.732σ shock.

## Files Modified

- `src/sep_irf.jl`: Added scaling logic to `extract_sep_irf` and `get_sep_irf`

## Testing

Created test scripts to verify scaling:
- `test_sep_scaling.jl`: Tests multiple shock sizes
- `test_sep_scaling_simple.jl`: Compares SEP vs perturbation IRFs
- `test_get_sep_irf_direct.jl`: Direct test of get_sep_irf function

## Impact

This fix ensures that SEP IRFs have the correct magnitude when compared to perturbation methods or when users specify a specific shock size. The scaling is automatic and transparent to the user.
