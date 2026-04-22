#!/usr/bin/env julia
# ============================================================================
# MULTI-CHAIN R-HAT DIAGNOSTICS — Gelman-Rubin Split R-hat, ESS Bulk/Tail
# ============================================================================
#
# Loads multiple NUTS-HMC chain files (.jls), computes Gelman-Rubin split
# R-hat, bulk ESS, and tail ESS via MCMCChains.jl, then generates a LaTeX
# convergence diagnostics table and a human-readable report.
#
# Each chain .jls file is a Dict with at minimum:
#   "chain"       → Matrix{Float64} of shape (n_samples, n_theta)
#   "theta_names" → Vector{Symbol}
#
# Usage:
#   julia --project=. scripts/multi_chain_rhat.jl \
#       --chains="chain1.jls,chain2.jls,chain3.jls,chain4.jls" \
#       --out=.local_artifacts/rhat_diagnostics/ \
#       [--label=linear] [--verbose]
# ============================================================================

using Serialization, Statistics, Printf, Dates
using MCMCChains

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_kv_string(args, key, default)
    for arg in args
        if startswith(arg, "$key=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end

chains_arg = parse_kv_string(ARGS, "--chains", "")
out_dir    = parse_kv_string(ARGS, "--out", ".local_artifacts/rhat_diagnostics")
label      = parse_kv_string(ARGS, "--label", "linear")
verbose    = any(==("--verbose"), ARGS)

if chains_arg == ""
    error("Usage: julia scripts/multi_chain_rhat.jl --chains=\"c1.jls,c2.jls,...\" [--out=DIR] [--label=linear] [--verbose]")
end

chain_paths = String.(strip.(split(chains_arg, ",")))
n_chains = length(chain_paths)

println("=" ^ 78)
println("MULTI-CHAIN R-HAT DIAGNOSTICS")
println("Started: $(now())")
println("=" ^ 78)
println("  Label:    $label")
println("  Chains:   $n_chains")
for (c, p) in enumerate(chain_paths)
    println("    [$c] $p")
end
println("  Output:   $out_dir")
println("  Verbose:  $verbose")

mkpath(out_dir)

# ============================================================================
# Step 1: Load Chain Files
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 1: Loading chain files")
println("-" ^ 78)

chain_matrices = Vector{Matrix{Float64}}(undef, n_chains)
theta_names = Symbol[]

for (c, path) in enumerate(chain_paths)
    global theta_names
    if !isfile(path)
        error("Chain file not found: $path")
    end
    r = deserialize(path)

    if !haskey(r, "chain")
        error("Chain file $path missing key \"chain\".")
    end
    if !haskey(r, "theta_names")
        error("Chain file $path missing key \"theta_names\".")
    end

    mat = r["chain"]::Matrix{Float64}
    names_c = Symbol.(r["theta_names"])

    if c == 1
        theta_names = names_c
    else
        if names_c != theta_names
            error("Parameter names in chain $c ($path) do not match chain 1.\n" *
                  "  Chain 1: $theta_names\n  Chain $c: $names_c")
        end
    end

    chain_matrices[c] = mat
    println("  Chain $c: $(size(mat, 1)) draws × $(size(mat, 2)) params  ($path)")
end

n_theta = length(theta_names)
sample_counts = [size(m, 1) for m in chain_matrices]
min_samples = minimum(sample_counts)
max_samples = maximum(sample_counts)

println("\n  Parameters:    $n_theta")
println("  Draws/chain:   min=$min_samples  max=$max_samples")

if min_samples != max_samples
    println("  NOTE: chains have unequal lengths; truncating all to $min_samples draws.")
end

# ============================================================================
# Step 2: Build MCMCChains.Chains Object
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 2: Constructing MCMCChains.Chains object")
println("-" ^ 78)

stacked = Array{Float64}(undef, min_samples, n_theta, n_chains)
for (c, mat) in enumerate(chain_matrices)
    stacked[:, :, c] = mat[1:min_samples, :]
end

chn = Chains(stacked, theta_names)
println("  Chains object: $(size(stacked, 1)) draws × $n_theta params × $n_chains chains")

# ============================================================================
# Step 3: Compute R-hat and ESS Diagnostics
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 3: Computing R-hat, ESS bulk, ESS tail")
println("-" ^ 78)

# ess_rhat returns a NamedTuple-table with columns :ess, :ess_per_sec, :rhat
# We use the extended version that also gives ess_bulk, ess_tail
diag_df = MCMCChains.ess_rhat(chn)

# Extract per-parameter diagnostics into simple vectors
rhat_vals    = Vector{Float64}(undef, n_theta)
ess_bulk_vals = Vector{Float64}(undef, n_theta)
ess_tail_vals = Vector{Float64}(undef, n_theta)

# MCMCChains.ess_rhat returns a NamedTuple with columns accessible by name.
# Column names depend on version; try the standard accessors.
diag_nt = diag_df.nt

# Identify column names for robust extraction
col_names = keys(diag_nt)
has_ess_bulk = :ess_bulk in col_names
has_ess_tail = :ess_tail in col_names
has_ess      = :ess in col_names
has_rhat     = :rhat in col_names

if verbose
    println("  Diagnostics columns: $col_names")
end

for i in 1:n_theta
    # R-hat
    if has_rhat
        rhat_vals[i] = diag_nt.rhat[i]
    else
        rhat_vals[i] = NaN
    end

    # ESS bulk
    if has_ess_bulk
        ess_bulk_vals[i] = diag_nt.ess_bulk[i]
    elseif has_ess
        ess_bulk_vals[i] = diag_nt.ess[i]
    else
        ess_bulk_vals[i] = NaN
    end

    # ESS tail
    if has_ess_tail
        ess_tail_vals[i] = diag_nt.ess_tail[i]
    elseif has_ess
        ess_tail_vals[i] = diag_nt.ess[i]
    else
        ess_tail_vals[i] = NaN
    end
end

# ============================================================================
# Step 4: Flag Problematic Parameters
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 4: Convergence assessment")
println("-" ^ 78)

n_warn    = sum(rhat_vals .> 1.01)
n_concern = sum(rhat_vals .> 1.1)

# Console summary table
@printf("  %-12s %8s %10s %10s %s\n", "Parameter", "R-hat", "ESS bulk", "ESS tail", "Flag")
println("  " * "-" ^ 55)

for i in 1:n_theta
    flag = ""
    if rhat_vals[i] > 1.1
        flag = "** CONCERN **"
    elseif rhat_vals[i] > 1.01
        flag = "* warning *"
    end
    @printf("  %-12s %8.4f %10.0f %10.0f %s\n",
            theta_names[i], rhat_vals[i], ess_bulk_vals[i], ess_tail_vals[i], flag)
end

println("  " * "-" ^ 55)
println("  Parameters with R-hat > 1.01: $n_warn")
println("  Parameters with R-hat > 1.1:  $n_concern")
println("  Max R-hat:  $(round(maximum(rhat_vals), digits=4))  ($(theta_names[argmax(rhat_vals)]))")
println("  Min ESS bulk: $(round(minimum(ess_bulk_vals), digits=0))  ($(theta_names[argmin(ess_bulk_vals)]))")
println("  Min ESS tail: $(round(minimum(ess_tail_vals), digits=0))  ($(theta_names[argmin(ess_tail_vals)]))")

# ============================================================================
# Step 5: Generate LaTeX Table
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 5: Generating LaTeX table")
println("-" ^ 78)

param_display = Dict(
    :crhoa   => raw"\rho_a",
    :crhob   => raw"\rho_b",
    :crhog   => raw"\rho_g",
    :crhoqs  => raw"\rho_{qs}",
    :crhopinf => raw"\rho_\pi",
    :crhow   => raw"\rho_w",
    :crhoms  => raw"\rho_{ms}",
    :z_ea    => raw"\sigma_a",
    :z_eb    => raw"\sigma_b",
    :z_eg    => raw"\sigma_g",
    :z_eqs   => raw"\sigma_{qs}",
    :z_epinf => raw"\sigma_\pi",
    :z_ew    => raw"\sigma_w",
    :z_em    => raw"\sigma_{ms}",
    :cprobp  => raw"\xi_p",
    :cindp   => raw"\iota_p",
    :curvp   => raw"\varepsilon_p",
    :cprobw  => raw"\xi_w",
)

param_groups = [
    ("Shock persistence",
     [:crhoa, :crhob, :crhog, :crhoqs, :crhopinf, :crhow, :crhoms]),
    ("Shock volatility",
     [:z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em]),
    ("Structural",
     [:cprobp, :cindp, :curvp, :cprobw]),
]

label_safe = replace(label, "_" => "\\_")

latex_path = joinpath(out_dir, "rhat_diagnostics_$(label).tex")
open(latex_path, "w") do io
    println(io, "% Multi-chain convergence diagnostics: $label")
    println(io, "% Chains: $n_chains, draws/chain: $min_samples, generated: $(now())")
    println(io, raw"\begin{table}[htbp]")
    println(io, raw"\centering")
    println(io, "\\caption{Multi-Chain Convergence Diagnostics: $(label_safe)}")
    println(io, "\\label{tab:rhat_$(label)}")
    println(io, raw"\begin{tabular}{lccc}")
    println(io, raw"\hline\hline")
    println(io, raw"Parameter & $\hat{R}$ & ESS (bulk) & ESS (tail) \\")
    println(io, raw"\hline")

    for (gi, (group_name, params)) in enumerate(param_groups)
        println(io, "\\multicolumn{4}{l}{\\textit{$(group_name)}} \\\\")
        for sym in params
            idx = findfirst(==(sym), theta_names)
            idx === nothing && continue
            dname = get(param_display, sym, string(sym))
            @printf(io, "\$%s\$ & %.3f & %.0f & %.0f \\\\\n",
                    dname, rhat_vals[idx], ess_bulk_vals[idx], ess_tail_vals[idx])
        end
        if gi < length(param_groups)
            println(io, raw"\hline")
        end
    end

    println(io, raw"\hline\hline")
    println(io, raw"\end{tabular}")
    println(io, raw"\begin{minipage}{0.92\textwidth}")
    println(io, "\\footnotesize\\textit{Notes:} Split \\(\\hat{R}\\) and effective sample size diagnostics computed via \\texttt{MCMCChains.jl} from $n_chains independent NUTS-HMC chains ($min_samples post-warmup draws each). ",
            "Values of \\(\\hat{R} > 1.01\\) suggest incomplete convergence; \\(\\hat{R} > 1.1\\) indicates serious concern. ",
            "ESS (bulk) measures mixing in the center of the distribution; ESS (tail) measures mixing in the 5\\% and 95\\% quantiles.")
    println(io, raw"\end{minipage}")
    println(io, raw"\end{table}")
end

println("  LaTeX table saved: $latex_path")

# ============================================================================
# Step 6: Save Results
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 6: Saving results")
println("-" ^ 78)

# Save serialized diagnostics
diag_results = Dict{String,Any}(
    "label"          => label,
    "n_chains"       => n_chains,
    "chain_paths"    => chain_paths,
    "min_samples"    => min_samples,
    "sample_counts"  => sample_counts,
    "theta_names"    => theta_names,
    "rhat"           => rhat_vals,
    "ess_bulk"       => ess_bulk_vals,
    "ess_tail"       => ess_tail_vals,
    "n_rhat_warn"    => n_warn,
    "n_rhat_concern" => n_concern,
    "timestamp"      => string(now()),
)

jls_path = joinpath(out_dir, "rhat_diagnostics_$(label).jls")
serialize(jls_path, diag_results)
println("  Diagnostics .jls:  $jls_path")

# Save human-readable text report
txt_path = joinpath(out_dir, "rhat_diagnostics_$(label).txt")
open(txt_path, "w") do io
    println(io, "MULTI-CHAIN CONVERGENCE DIAGNOSTICS — $(uppercase(label))")
    println(io, "=" ^ 78)
    println(io, "Date:           $(now())")
    println(io, "Chains:         $n_chains")
    for (c, p) in enumerate(chain_paths)
        println(io, "  [$c] $p  ($(sample_counts[c]) draws)")
    end
    println(io, "Draws/chain:    $min_samples (truncated to shortest)")
    println(io, "Parameters:     $n_theta")
    println(io, "")
    @printf(io, "%-12s %8s %10s %10s %s\n",
            "Parameter", "R-hat", "ESS bulk", "ESS tail", "Flag")
    println(io, "-" ^ 60)
    for i in 1:n_theta
        flag = ""
        if rhat_vals[i] > 1.1
            flag = "** CONCERN **"
        elseif rhat_vals[i] > 1.01
            flag = "* warning *"
        end
        @printf(io, "%-12s %8.4f %10.0f %10.0f %s\n",
                theta_names[i], rhat_vals[i], ess_bulk_vals[i], ess_tail_vals[i], flag)
    end
    println(io, "-" ^ 60)
    println(io, "")
    println(io, "Summary:")
    println(io, "  Parameters with R-hat > 1.01: $n_warn")
    println(io, "  Parameters with R-hat > 1.1:  $n_concern")
    @printf(io, "  Max R-hat:     %.4f  (%s)\n", maximum(rhat_vals), theta_names[argmax(rhat_vals)])
    @printf(io, "  Min ESS bulk:  %.0f  (%s)\n", minimum(ess_bulk_vals), theta_names[argmin(ess_bulk_vals)])
    @printf(io, "  Min ESS tail:  %.0f  (%s)\n", minimum(ess_tail_vals), theta_names[argmin(ess_tail_vals)])
end

println("  Text report:       $txt_path")

# ============================================================================
# Done
# ============================================================================

println("\n" * "=" ^ 78)
println("MULTI-CHAIN R-HAT DIAGNOSTICS COMPLETE")
println("Finished: $(now())")
println("  R-hat > 1.01: $n_warn / $n_theta parameters")
println("  R-hat > 1.1:  $n_concern / $n_theta parameters")
println("  Max R-hat:    $(round(maximum(rhat_vals), digits=4))")
println("=" ^ 78)
