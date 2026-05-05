#!/usr/bin/env julia
# Pool the four warm-started surrogate HMC chains and report posterior means,
# standard deviations, 5th/95th percentiles, and per-chain posterior-mean LL
# for the extended-sample posterior table in the paper.

using Serialization, Statistics, Printf

const CHAIN_FILES = [
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls",
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed5.jls",
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed6.jls",
    ".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000_seed7.jls",
]

chains = [deserialize(f) for f in CHAIN_FILES]
names  = Symbol.(chains[1]["theta_names"])
mats   = [c["chain"] for c in chains]
pooled = vcat(mats...)

println("=" ^ 78)
println("POOLED POSTERIOR — 4 WARM-STARTED SURROGATE CHAINS (seeds 42, 5, 6, 7)")
println("=" ^ 78)
println("  Total draws: ", size(pooled, 1), "  (", length(chains), " chains × ", size(mats[1], 1), ")")
println()

# Per-chain posterior-mean log-likelihood (if available)
for (i, c) in enumerate(chains)
    if haskey(c, "loglik")
        ll = c["loglik"]
        @printf "  Chain %d  mean LL = %9.2f  (min=%9.2f, max=%9.2f)\n" i mean(ll) minimum(ll) maximum(ll)
    end
end
println()

@printf "%-12s %10s %10s %10s %10s\n" "Parameter" "Mean" "Std" "Q5" "Q95"
println("-" ^ 58)
for (k, n) in enumerate(names)
    col = pooled[:, k]
    @printf "%-12s %10.4f %10.4f %10.4f %10.4f\n" String(n) mean(col) std(col) quantile(col, 0.05) quantile(col, 0.95)
end
println("-" ^ 58)
