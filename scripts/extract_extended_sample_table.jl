#!/usr/bin/env julia
# ============================================================================
# Extract Extended-Sample Posterior Table for Paper
# ============================================================================
# Usage:
#   julia --project=. scripts/extract_extended_sample_table.jl \
#       --linear=.local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls \
#       --surrogate=.local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls
# ============================================================================

using Serialization, Statistics, Printf

function parse_kv_string(args, key, default)
    for arg in args
        if startswith(arg, "$key=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return default
end

linear_path    = parse_kv_string(ARGS, "--linear", "")
surrogate_path = parse_kv_string(ARGS, "--surrogate", "")

if linear_path == "" && surrogate_path == ""
    error("Provide at least one of --linear=<chain.jls> or --surrogate=<chain.jls>")
end

# Parameter display names for LaTeX (chain uses z_ea/cprobp/cprobw naming)
param_latex = Dict(
    :crhoa   => raw"$\rho_a$",
    :crhob   => raw"$\rho_b$",
    :crhog   => raw"$\rho_g$",
    :crhoqs  => raw"$\rho_{qs}$",
    :crhopinf => raw"$\rho_{\pi}$",
    :crhow   => raw"$\rho_w$",
    :crhoms  => raw"$\rho_{ms}$",
    :z_ea    => raw"$\sigma_a$",
    :z_eb    => raw"$\sigma_b$",
    :z_eg    => raw"$\sigma_g$",
    :z_eqs   => raw"$\sigma_{qs}$",
    :z_epinf => raw"$\sigma_{\pi}$",
    :z_ew    => raw"$\sigma_w$",
    :z_em    => raw"$\sigma_{ms}$",
    :cprobp  => raw"$\xi_p$",
    :cindp   => raw"$\iota_p$",
    :curvp   => raw"$\varepsilon_p$",
    :cprobw  => raw"$\xi_w$",
)

param_desc = Dict(
    :crhoa   => "TFP",
    :crhob   => "Risk premium",
    :crhog   => "Government",
    :crhoqs  => "Investment",
    :crhopinf => "Price markup",
    :crhow   => "Wage markup",
    :crhoms  => "Monetary",
    :z_ea    => "TFP",
    :z_eb    => "Risk premium",
    :z_eg    => "Government",
    :z_eqs   => "Investment",
    :z_epinf => "Price markup",
    :z_ew    => "Wage markup",
    :z_em    => "Monetary",
    :cprobp  => "Calvo prices",
    :cindp   => "Price indexation",
    :curvp   => "Kimball curvature",
    :cprobw  => "Calvo wages",
)

param_groups = [
    ("Shock Persistence", [:crhoa, :crhob, :crhog, :crhoqs, :crhopinf, :crhow, :crhoms]),
    ("Shock Volatilities", [:z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em]),
    ("Structural Parameters", [:cprobp, :cindp, :curvp, :cprobw]),
]

function ess_batch_means(chain::Vector{Float64}; batch_size::Int=50)
    n = length(chain)
    n < 2 * batch_size && return Float64(n)
    n_batches = n ÷ batch_size
    batch_means = [mean(chain[(i-1)*batch_size+1 : i*batch_size]) for i in 1:n_batches]
    var_total = var(chain)
    var_batch = var(batch_means)
    var_batch < 1e-20 && return Float64(n)
    return n * var_total / (batch_size * var_batch)
end

function load_chain_results(path::AbstractString, label::String)
    println("Loading $label: $path")
    r = deserialize(path)
    chain = r["chain"]
    theta_names = Symbol.(r["theta_names"])
    n_samples = r["n_samples"]
    n_divergent = r["n_divergent"]
    ll_post_mean = r["ll_post_mean"]
    elapsed = r["elapsed_seconds"]

    n_theta = length(theta_names)
    results = Dict{Symbol, NamedTuple}()

    min_ess = Inf
    for i in 1:n_theta
        name = theta_names[i]
        post_mean = mean(chain[:, i])
        post_std = std(chain[:, i])
        q05 = quantile(chain[:, i], 0.05)
        q95 = quantile(chain[:, i], 0.95)
        ess = ess_batch_means(chain[:, i])
        min_ess = min(min_ess, ess)
        results[name] = (mean=post_mean, std=post_std, q05=q05, q95=q95, ess=ess)
    end

    println("  Draws:       $n_samples")
    println("  Divergences: $n_divergent")
    println("  LL at mean:  $(round(ll_post_mean, digits=0))")
    println("  Min ESS:     $(round(min_ess, digits=0))")
    println("  Elapsed:     $(round(elapsed/3600, digits=1)) hours")

    return (results=results, n_samples=n_samples, n_divergent=n_divergent,
            ll_post_mean=ll_post_mean, min_ess=round(Int, min_ess),
            elapsed_hours=elapsed/3600, theta_names=theta_names)
end

# Load chains
lin = nothing
sur = nothing

if linear_path != ""
    lin = load_chain_results(linear_path, "Linear HMC")
end
if surrogate_path != ""
    sur = load_chain_results(surrogate_path, "Surrogate HMC")
end

# Generate LaTeX table
println("\n" * "=" ^ 72)
println("LaTeX Table: Extended-Sample Posterior (1959Q1–2025Q1)")
println("=" ^ 72)

has_both = lin !== nothing && sur !== nothing
ncols = has_both ? 6 : 4

println(raw"\begin{table}[htbp]")
println(raw"\centering")
println(raw"\caption{Extended-Sample Posterior Comparison (1959Q1--2025Q1)}")
println(raw"\label{tab:extended_sample_posterior}")

if has_both
    println(raw"\begin{tabular}{llrrrr}")
    println(raw"\toprule")
    println(raw"Parameter & Description & Kalman & RS Surrogate & \$\Delta\$ & RS 90\% CI \\")
else
    label = lin !== nothing ? "Kalman" : "RS Surrogate"
    println(raw"\begin{tabular}{llrr}")
    println(raw"\toprule")
    println("Parameter & Description & Post.\\ Mean & 90\\% CI \\\\")
end

println(raw"\midrule")

for (group_name, params) in param_groups
    println("\\multicolumn{$(ncols)}{l}{\\textit{$(group_name)}} \\\\")
    for p in params
        ltx = get(param_latex, p, string(p))
        desc = get(param_desc, p, string(p))

        if has_both
            lm = lin.results[p].mean
            sm = sur.results[p].mean
            delta = sm - lm
            s05 = sur.results[p].q05
            s95 = sur.results[p].q95

            delta_sign = delta >= 0 ? raw"$+$" : raw"$-$"
            delta_abs = abs(delta)

            if p == :curvp
                @printf("%s & %s & %.1f & %.1f & %s%.1f & [%.1f, %.1f] \\\\\n",
                        ltx, desc, lm, sm, delta_sign, delta_abs, s05, s95)
            else
                @printf("%s & %s & %.3f & %.3f & %s%.3f & [%.3f, %.3f] \\\\\n",
                        ltx, desc, lm, sm, delta_sign, delta_abs, s05, s95)
            end
        elseif lin !== nothing
            lm = lin.results[p].mean
            l05 = lin.results[p].q05
            l95 = lin.results[p].q95
            if p == :curvp
                @printf("%s & %s & %.1f & [%.1f, %.1f] \\\\\n", ltx, desc, lm, l05, l95)
            else
                @printf("%s & %s & %.3f & [%.3f, %.3f] \\\\\n", ltx, desc, lm, l05, l95)
            end
        else
            sm = sur.results[p].mean
            s05 = sur.results[p].q05
            s95 = sur.results[p].q95
            if p == :curvp
                @printf("%s & %s & %.1f & [%.1f, %.1f] \\\\\n", ltx, desc, sm, s05, s95)
            else
                @printf("%s & %s & %.3f & [%.3f, %.3f] \\\\\n", ltx, desc, sm, s05, s95)
            end
        end
    end
    println(raw"\midrule")
end

# Remove last \midrule and replace with \bottomrule
println(raw"\end{tabular}")
println(raw"\smallskip")

# Notes
if has_both
    ll_lin = round(Int, lin.ll_post_mean)
    ll_sur = round(Int, sur.ll_post_mean)
    println("{\\footnotesize \\textit{Notes}: Extended sample 1959Q1--2025Q1 (\$T=265\$). ``Kalman'' reports posterior means from linear NUTS-HMC ($(lin.n_samples) draws, $(lin.n_divergent) divergences, ESS \$\\geq $(lin.min_ess)\$). ``RS Surrogate'' reports regime-switching NUTS-HMC ($(sur.n_samples) draws, $(sur.n_divergent) divergences, ESS \$\\geq $(sur.min_ess)\$). Log-likelihood at posterior mean: Kalman \$$(ll_lin)\$; RS \$$(ll_sur)\$. \\$\\Delta\\$ is the shift in posterior means. 90\\% CI is the 5th--95th percentile.}")
elseif lin !== nothing
    ll_lin = round(Int, lin.ll_post_mean)
    println("{\\footnotesize \\textit{Notes}: Extended sample 1959Q1--2025Q1 (\$T=265\$). Linear NUTS-HMC via AdvancedHMC.jl, $(lin.n_samples) draws, $(lin.n_divergent) divergences, ESS \$\\geq $(lin.min_ess)\$. Log-likelihood at posterior mean: \$$(ll_lin)\$.}")
else
    ll_sur = round(Int, sur.ll_post_mean)
    println("{\\footnotesize \\textit{Notes}: Extended sample 1959Q1--2025Q1 (\$T=265\$). RS Surrogate NUTS-HMC, $(sur.n_samples) draws, $(sur.n_divergent) divergences, ESS \$\\geq $(sur.min_ess)\$. Log-likelihood at posterior mean: \$$(ll_sur)\$.}")
end

println(raw"\end{table}")

# Also print a summary comparison with the standard-sample results
println("\n" * "=" ^ 72)
println("Parameter Shifts: Standard (T=184) → Extended (T=265)")
println("=" ^ 72)

# Standard-sample values from the paper (Table 18param_switching_posterior)
std_kalman = Dict(
    :crhoa => 0.987, :crhob => 0.762, :crhog => 0.986, :crhoqs => 0.782,
    :crhopinf => 0.059, :crhow => 0.375, :crhoms => 0.182,
    :z_ea => 0.502, :z_eb => 0.122, :z_eg => 0.508, :z_eqs => 0.361,
    :z_epinf => 0.143, :z_ew => 0.204, :z_em => 0.245,
    :cprobp => 0.740, :cindp => 0.932, :curvp => 39.3, :cprobw => 0.947,
)
std_rs = Dict(
    :crhoa => 0.980, :crhob => 0.590, :crhog => 0.974, :crhoqs => 0.818,
    :crhopinf => 0.042, :crhow => 0.078, :crhoms => 0.069,
    :z_ea => 0.503, :z_eb => 0.275, :z_eg => 0.513, :z_eqs => 0.353,
    :z_epinf => 0.147, :z_ew => 0.330, :z_em => 0.248,
    :cprobp => 0.755, :cindp => 0.886, :curvp => 104.2, :cprobw => 0.929,
)

@printf("%-12s %10s %10s %10s %10s\n", "Parameter", "Std Kalman", "Ext Kalman", "Std RS", "Ext RS")
println("-" ^ 54)
for (_, params) in param_groups
    for p in params
        ext_k = lin !== nothing ? @sprintf("%.3f", lin.results[p].mean) : "—"
        ext_r = sur !== nothing ? @sprintf("%.3f", sur.results[p].mean) : "—"
        if p == :curvp
            ext_k = lin !== nothing ? @sprintf("%.1f", lin.results[p].mean) : "—"
            ext_r = sur !== nothing ? @sprintf("%.1f", sur.results[p].mean) : "—"
            @printf("%-12s %10.1f %10s %10.1f %10s\n", p, std_kalman[p], ext_k, std_rs[p], ext_r)
        else
            @printf("%-12s %10.3f %10s %10.3f %10s\n", p, std_kalman[p], ext_k, std_rs[p], ext_r)
        end
    end
end
