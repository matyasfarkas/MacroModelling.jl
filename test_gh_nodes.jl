# Test what the actual GH node values are after transformation
using MacroModelling, LinearAlgebra

# Test GH nodes for 3 nodes, 7 shocks
nnodes = 3
nshocks = 7

# Create GH nodes (before transformation)
if nnodes == 3
    x1d = [-√3, 0.0, √3]
    w1d = [π/6, 2π/3, π/6]
end

K = nnodes^nshocks
X = zeros(nshocks, K)
W = zeros(K)

for k in 1:K
    w_prod = 1.0
    idx = k - 1
    for d in 1:nshocks
        local_idx = mod(idx, nnodes) + 1
        X[d, k] = x1d[local_idx]
        w_prod *= w1d[local_idx]
        idx = div(idx, nnodes)
    end
    W[k] = w_prod
end

println("GH nodes BEFORE transformation:")
println("For shock 5 (epinf):")
for k in [1, 100, 500, 1000, 1175, 1500, 2000, 2187]
    println("  Group $k: shock values = ", X[:, k])
end

# Now transform by shock covariance (as in SEP solver)
# SW07 shock std devs
σ_vec = [0.4618, 1.8513, 0.609, 0.2397, 0.1455, 0.6017, 0.2089]  # ea, eb, eg, em, epinf, eqs, ew
Σ = diagm(σ_vec .^ 2)
L = cholesky(Σ).L
X_transformed = L * X

println("\n" * "="^70)
println("GH nodes AFTER transformation by Σ:")
println("Shock 5 (epinf) has σ = $(σ_vec[5])")
println("\nFor shock 5 at different groups:")
for k in [1, 100, 500, 1000, 1175, 1500, 2000, 2187]
    println("  Group $k: epinf shock = $(X_transformed[5, k])")
end

# Find group 1175 (which should be: all shocks at zero node except shock 5 at highest node)
println("\n" * "="^70)
println("Group 1175 (target for epinf +1σ shock):")
println("All shock values:")
for (i, name) in enumerate([:ea, :eb, :eg, :em, :epinf, :eqs, :ew])
    println("  $name: $(X_transformed[i, 1175])")
end

println("\nExpected: epinf ≈ √3 * $(σ_vec[5]) = $(√3 * σ_vec[5])")
println("Actual: epinf = $(X_transformed[5, 1175])")
