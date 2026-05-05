#!/usr/bin/env julia
# ============================================================================
# NORMALIZED EULER ERRORS — post-processing of euler_errors_results.jls
# ============================================================================
# Reads the per-period × per-equation Euler residual arrays produced by
# scripts/euler_errors_study.jl and reports:
#   1.  log10(|R_{i,t}|) statistics per method:  max, mean, median, p95.
#   2.  Per-equation max |R_i| / |yss_i| where yss_i is the steady-state
#       value of the model variable in mm_model.var that has the same
#       index as equation i (the equation-to-variable pairing follows
#       MacroModelling's standard convention: the i-th dynamic equation
#       solves for the i-th endogenous variable).
#   3.  log10 of the normalized residuals' max/mean per method.
#   4.  Top-5 worst (equation, period) pairs per method.
# Outputs:
#   .local_artifacts/euler_errors/euler_errors_normalized_table.tex
#   .local_artifacts/euler_errors/EULER_ERRORS_NORMALIZED.md
# ============================================================================

using Serialization, Statistics, Printf, Dates, LinearAlgebra
using MacroModelling

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

results_jls = joinpath(REPO_ROOT, ".local_artifacts/euler_errors/euler_errors_results.jls")
out_dir     = joinpath(REPO_ROOT, ".local_artifacts/euler_errors")
out_tex     = joinpath(out_dir, "euler_errors_normalized_table.tex")
out_md      = joinpath(out_dir, "EULER_ERRORS_NORMALIZED.md")

println("=" ^ 78)
println("NORMALIZED EULER ERRORS — post-processing")
println("Started: $(now())")
println("=" ^ 78)

isfile(results_jls) || error("Missing $results_jls.  Run euler_errors_study.jl first.")

result = deserialize(results_jls)

theta_names = result["theta_names"]
post_mean   = result["post_mean"]
shocks_recovered = result["shocks_recovered"]
Y_rom1  = result["Y_rom1"]
Y_nn    = result["Y_nn"]
Y_nng   = result["Y_nn_gated"]
Y_sep   = result["Y_sep"]
gate_mask = result["gate_mask"]
T_obs   = result["T_obs"]

println("  T_obs = $T_obs,  pooled posterior θ name examples: $(first(theta_names, 4))")

# ============================================================================
# Reload model + dynamic-residual function so we can rebuild per (i, t)
# residuals on demand.  This is required because the saved .jls only stores
# per-period sup-norms, not the full (n_eq × T_obs) residual matrix.
# ============================================================================

println("\n--- Loading model ---")
include(joinpath(REPO_ROOT, "scripts/hlt_surrogate/hlt_model_loader_utils.jl"))
mm_model = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)

data_path = result["data_path"]
payload   = MacroModelling.load_hlt_synthetic_scenario(data_path)
shock_sigmas = payload["shock_sigmas"]

# Apply posterior mean to model
theta_param_idx = let idx = indexin(theta_names, mm_model.parameters)
    Int.(idx)
end
base_parameters = copy(mm_model.parameter_values)
post_mean_full_params = copy(base_parameters)
for (j, idx) in enumerate(theta_param_idx)
    post_mean_full_params[idx] = post_mean[j]
end
mm_model.parameter_values .= post_mean_full_params
MacroModelling.solve!(mm_model)

resid_func, _jac_func, vars_raw, parameters_and_SS, resid_buffer, _jac_buffer =
    MacroModelling.build_dynamic_residual_jacobian(mm_model)
n_eq = length(resid_buffer)
println("  n_eq = $n_eq,  n_dyn_vars = $(length(vars_raw))")

var_kind, var_idx = MacroModelling.build_dyn_var_maps(mm_model, vars_raw)

SS_result = MacroModelling.get_steady_state(mm_model, derivatives=false)
yss_full = [Float64(SS_result(var)) for var in mm_model.var]
params_and_ss_vals = MacroModelling.build_parameters_and_ss_values(
    parameters_and_SS, post_mean_full_params, mm_model, yss_full, SS_result)

# Steady-state normalizer per equation: the i-th dynamic equation's typical scale.
# We use max(|yss_i|, 1.0) to avoid division by ~0 for equations whose associated
# variable has near-zero steady state (gap variables, deviations, log-deviations).
yss_norm = max.(abs.(yss_full[1:n_eq]), 1.0)

# ============================================================================
# Re-evaluate the full per-(i, t) residual matrix for each method
# ============================================================================

function fill_dyn!(dyn_buf, Y, t, shocks)
    # vars_raw layout: future, present, past, shock — fill via var_kind/var_idx
    for k in 1:length(vars_raw)
        kind = var_kind[k]
        idx  = var_idx[k]
        if kind === :future
            dyn_buf[k] = Y[idx, t + 2]    # y_{t+1}
        elseif kind === :present
            dyn_buf[k] = Y[idx, t + 1]    # y_t
        elseif kind === :past
            dyn_buf[k] = Y[idx, t]        # y_{t-1}
        else  # :shock
            dyn_buf[k] = shocks[idx, t]
        end
    end
end

function full_residuals(Y::AbstractMatrix; label::AbstractString)
    R = fill(NaN, n_eq, T_obs - 1)  # t = 1..T_obs-1 (need y_{t+1})
    dyn_buf = zeros(Float64, length(vars_raw))
    nfail = 0
    for t in 1:(T_obs - 1)
        fill_dyn!(dyn_buf, Y, t, shocks_recovered)
        # resid_func has signature resid_func!(R, params, vars)
        resid_func(resid_buffer, params_and_ss_vals, dyn_buf)
        if !all(isfinite, resid_buffer)
            nfail += 1
            continue
        end
        R[:, t] .= resid_buffer
    end
    println("  [$label]  computed full residuals;  $nfail / $(T_obs - 1) periods had non-finite residuals")
    return R
end

println("\n--- Re-evaluating full per-(equation, period) residuals ---")
R_sep  = (Y_sep === nothing) ? nothing : full_residuals(Y_sep;  label="SEP")
R_rom1 = full_residuals(Y_rom1; label="ROM1")
R_nng  = full_residuals(Y_nng;  label="ROM1+NN gated")

# ============================================================================
# Compute summary stats per method
# ============================================================================

function summarize(R::AbstractMatrix, name::String)
    finite = filter(isfinite, vec(R))
    if isempty(finite)
        return Dict("label" => name, "n_finite" => 0)
    end
    abs_finite = abs.(finite)
    abs_finite_pos = filter(>(0), abs_finite)
    log10_vals = log10.(abs_finite_pos)
    # Per-equation max
    per_eq = [maximum(filter(isfinite, abs.(R[i, :])); init=NaN) for i in 1:size(R, 1)]
    per_eq_finite = filter(isfinite, per_eq)
    # Normalized: divide each equation's residual by yss_norm[i]
    R_norm = abs.(R) ./ yss_norm
    finite_norm = filter(isfinite, vec(R_norm))
    finite_norm_pos = filter(>(0), finite_norm)
    log10_norm = isempty(finite_norm_pos) ? Float64[] : log10.(finite_norm_pos)
    # Top-5 worst (i, t)
    flat_idx = sortperm(vec(abs.(R)); rev=true)[1:min(5, length(R))]
    worst_pairs = [
        (i = ((k - 1) % size(R, 1)) + 1,
         t = div(k - 1, size(R, 1)) + 1,
         val = R[((k - 1) % size(R, 1)) + 1, div(k - 1, size(R, 1)) + 1])
        for k in flat_idx
    ]
    return Dict(
        "label"     => name,
        "n_finite"  => length(finite),
        "max_abs"   => isempty(abs_finite) ? NaN : maximum(abs_finite),
        "mean_abs"  => isempty(abs_finite) ? NaN : mean(abs_finite),
        "median_abs"=> isempty(abs_finite) ? NaN : median(abs_finite),
        "p95_abs"   => isempty(abs_finite) ? NaN : quantile(abs_finite, 0.95),
        "log10_max" => isempty(log10_vals) ? NaN : maximum(log10_vals),
        "log10_mean"=> isempty(log10_vals) ? NaN : mean(log10_vals),
        "log10_median" => isempty(log10_vals) ? NaN : median(log10_vals),
        "log10_p95" => isempty(log10_vals) ? NaN : quantile(log10_vals, 0.95),
        "max_norm"  => isempty(finite_norm_pos) ? NaN : maximum(finite_norm_pos),
        "mean_norm" => isempty(finite_norm_pos) ? NaN : mean(finite_norm_pos),
        "median_norm" => isempty(finite_norm_pos) ? NaN : median(finite_norm_pos),
        "log10_max_norm"    => isempty(log10_norm) ? NaN : maximum(log10_norm),
        "log10_mean_norm"   => isempty(log10_norm) ? NaN : mean(log10_norm),
        "log10_median_norm" => isempty(log10_norm) ? NaN : median(log10_norm),
        "per_eq_max"      => per_eq,
        "worst_pairs"     => worst_pairs,
    )
end

println("\n--- Summarizing ---")
S_rom1 = summarize(R_rom1, "ROM1")
S_nng  = summarize(R_nng,  "ROM1+NN gated")
S_sep  = R_sep === nothing ? nothing : summarize(R_sep, "SEP")

function fmt_log10(x)
    (isnan(x) || !isfinite(x)) && return "---"
    return @sprintf("%+.3f", x)
end

println()
println("=== Normalized Euler errors (R / max(|y_ss|, 1)) ===")
@printf("  %-22s  %12s  %12s  %12s  %12s\n",
        "Method", "max ratio", "log10(max)", "median ratio", "log10(median)")
for S in [S_sep, S_rom1, S_nng]
    S === nothing && continue
    @printf("  %-22s  %12.4e  %12s  %12.4e  %12s\n",
            S["label"], S["max_norm"], fmt_log10(S["log10_max_norm"]),
            S["median_norm"], fmt_log10(S["log10_median_norm"]))
end

# ============================================================================
# Write LaTeX table
# ============================================================================

tex = IOBuffer()
println(tex, "% Normalized Euler errors (autogenerated by scripts/euler_errors_normalize.jl)")
println(tex, "% Generated: $(now())")
println(tex, "\\begin{table}[ht]")
println(tex, "\\centering")
println(tex, "\\caption{Normalized nonlinear Euler residuals across DSGE solution methods at the pooled")
println(tex, "warm-started surrogate posterior mean.  Each per-equation residual is divided by")
println(tex, "the steady-state value of the equation's associated endogenous variable")
println(tex, "(\$\\max(|y^{\\rm ss}_i|, 1)\$ to avoid division by near-zero log-deviation variables).")
println(tex, "Reported quantities are taken across all (equation, period) pairs with finite residuals.}")
println(tex, "\\label{tab:euler_errors_normalized}")
println(tex, "\\small")
println(tex, "\\begin{tabular}{l cccc}")
println(tex, "\\toprule")
println(tex, "Method & \$\\max(|R| / |y^{\\rm ss}|)\$ & median & \$\\log_{10}(\\max)\$ & \$\\log_{10}(\\text{median})\$ \\\\")
println(tex, "\\midrule")
function tex_row(io, S, name)
    if S === nothing
        @printf(io, "%-32s & --- & --- & --- & --- \\\\\n", name)
        return
    end
    @printf(io, "%-32s & \$%.3e\$ & \$%.3e\$ & \$%s\$ & \$%s\$ \\\\\n",
            name, S["max_norm"], S["median_norm"],
            fmt_log10(S["log10_max_norm"]), fmt_log10(S["log10_median_norm"]))
end
tex_row(tex, S_sep,  "SEP")
tex_row(tex, S_rom1, "ROM1")
tex_row(tex, S_nng,  "ROM1 + NN (gate-on only)")
println(tex, "\\bottomrule")
println(tex, "\\end{tabular}")
println(tex, "\\end{table}")
write(out_tex, String(take!(tex)))
println("\nLaTeX table: $out_tex")

# ============================================================================
# Write markdown summary
# ============================================================================

md = IOBuffer()
println(md, "# Normalized Euler errors — SEP vs ROM1 vs ROM1+NN gated")
println(md)
println(md, "_Autogenerated by `scripts/euler_errors_normalize.jl` on $(now())._")
println(md)
println(md, "## Setup")
println(md)
println(md, "- Per-equation residuals divided by `max(|y_ss_i|, 1.0)` (steady-state value)")
println(md, "- Across (equation, period) pairs with finite residuals")
println(md, "- Reference: `.local_artifacts/euler_errors/euler_errors_results.jls`")
println(md)
println(md, "## Headline (normalized)")
println(md)
println(md, "| Method | max |R/y_ss| | median |R/y_ss| | log10(max) | log10(median) |")
println(md, "|---|---|---|---|---|")
function md_row(io, S, name)
    if S === nothing
        println(io, "| $name | --- | --- | --- | --- |")
        return
    end
    @printf(io, "| %s | %.3e | %.3e | %s | %s |\n",
            name, S["max_norm"], S["median_norm"],
            fmt_log10(S["log10_max_norm"]), fmt_log10(S["log10_median_norm"]))
end
md_row(md, S_sep,  "SEP")
md_row(md, S_rom1, "ROM1")
md_row(md, S_nng,  "ROM1+NN (gate-on only)")

println(md)
println(md, "## Top-5 worst (equation, period) pairs per method")
println(md)
for S in [S_sep, S_rom1, S_nng]
    S === nothing && continue
    println(md, "**$(S["label"])**:")
    for w in S["worst_pairs"]
        @printf(md, "- eq#%d, t=%d, R=%.3e (R/y_ss=%.3e)\n",
                w.i, w.t, w.val, abs(w.val) / yss_norm[w.i])
    end
    println(md)
end
write(out_md, String(take!(md)))
println("Markdown: $out_md")
println("\nDone.")
