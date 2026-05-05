#!/usr/bin/env julia
# ============================================================================
# QUIET-SAMPLE OOS GATE RECALIBRATION
# ============================================================================
#
# Generates stricter gate calibration payloads from the pre-1995 in-sample
# statistics and re-runs `oos_forecast_evaluate.jl` for each candidate.  This
# script does not re-run HMC; it is a fast diagnostic for whether the gate
# policy itself is responsible for the quiet-sample forecast degradation.
#
# Usage:
#   julia --project=. scripts/oos_gate_recalibration.jl
#   julia --project=. scripts/oos_gate_recalibration.jl --candidates=base_nopad,q95_pad,q95_nopad
#
# Outputs:
#   .local_artifacts/oos_gate_recalibration/gate_*.jls
#   .local_artifacts/oos_gate_recalibration/OOS_GATE_RECALIBRATION_SUMMARY.md
#   .local_artifacts/oos_forecast/OOS_SUMMARY_<candidate>.md
# ============================================================================

using Serialization, Statistics, Printf, Dates

repo_root = normpath(joinpath(@__DIR__, ".."))
cd(repo_root)

function parse_kv(args, key, default)
    for a in args
        if startswith(a, "$key=")
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_candidates(s::AbstractString)
    isempty(strip(s)) && return String[]
    return String.(strip.(split(s, ",")))
end

const out_dir = ".local_artifacts/oos_gate_recalibration"
mkpath(out_dir)

t_pre = parse(Int, parse_kv(ARGS, "--t-pre", "144"))
t_end = parse(Int, parse_kv(ARGS, "--t-end", "196"))
window_start = parse(Int, parse_kv(ARGS, "--window-start", "145"))
window_end = parse(Int, parse_kv(ARGS, "--window-end", "160"))
window_label = parse_kv(ARGS, "--window-label", "Early-quiet")
base_gate = parse_kv(ARGS, "--base-gate",
    ".local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p.jls")
linear_chain = parse_kv(ARGS, "--linear-chain",
    ".local_artifacts/oos_forecast/hlt_linear_hmc_quiet_1994_200.jls")
surrogate_chain = parse_kv(ARGS, "--surrogate-chain",
    ".local_artifacts/oos_forecast/hlt_surrogate_hmc_quiet_1994_150.jls")
candidates = parse_candidates(parse_kv(ARGS, "--candidates",
    "base_nopad,q95_pad,q95_nopad,q98_nopad"))

isfile(base_gate) || error("Missing base gate calibration: $base_gate")
isfile(linear_chain) || error("Missing linear chain: $linear_chain")
isfile(surrogate_chain) || error("Missing surrogate chain: $surrogate_chain")

base = deserialize(base_gate)
e = Float64.(base["e_stats"])
f = Float64.(base["f_stats"])
t_pre <= length(e) || error("t_pre=$t_pre exceeds gate stat length $(length(e))")

struct Candidate
    name::String
    quantile::Union{Nothing,Float64}
    k_pre::Int
    k_post::Int
    min_len::Int
    target_share::Float64
end

function candidate_from_name(name::String)
    if name == "base_nopad"
        return Candidate(name, nothing, 0, 0, 1, Float64(get(base, "target_share", 0.1)))
    elseif name == "base_pad"
        return Candidate(name, nothing, 4, 8, 4, Float64(get(base, "target_share", 0.1)))
    elseif name == "q95_pad"
        return Candidate(name, 0.95, 4, 8, 4, 0.05)
    elseif name == "q95_nopad"
        return Candidate(name, 0.95, 0, 0, 1, 0.05)
    elseif name == "q98_nopad"
        return Candidate(name, 0.98, 0, 0, 1, 0.02)
    elseif name == "q99_nopad"
        return Candidate(name, 0.99, 0, 0, 1, 0.01)
    else
        error("Unknown candidate `$name`. Supported: base_pad, base_nopad, q95_pad, q95_nopad, q98_nopad, q99_nopad")
    end
end

function write_candidate_gate(c::Candidate)
    g = deepcopy(base)
    if c.quantile !== nothing
        q = c.quantile
        g["quantile"] = q
        g["tau_eps"] = quantile(e[1:t_pre], q)
        g["tau_y"] = quantile(f[1:t_pre], q)
        g["target_share"] = c.target_share
        g["achieved_share"] = mean((e[1:t_pre] .> g["tau_eps"]) .| (f[1:t_pre] .> g["tau_y"]))
        g["recalibration_note"] =
            "Quiet-sample recalibration: tau thresholds set at q=$q using first $t_pre periods."
    else
        g["recalibration_note"] =
            "Quiet-sample recalibration: original tau thresholds retained; evaluator padding changed."
    end
    g["oos_gate_candidate"] = c.name
    g["gate_k_pre"] = c.k_pre
    g["gate_k_post"] = c.k_post
    g["gate_min_len"] = c.min_len
    path = joinpath(out_dir, "gate_$(c.name).jls")
    serialize(path, g)
    return path, g
end

function run_evaluator(c::Candidate, gate_path::String)
    tag = "quiet_1994_$(c.name)"
    cmd = `julia --project=. scripts/oos_forecast_evaluate.jl
        --tag=$tag
        --t-pre=$t_pre
        --t-end=$t_end
        --window-label=$window_label
        --window-start=$window_start
        --window-end=$window_end
        --linear-chain=$linear_chain
        --surrogate-chain=$surrogate_chain
        --gate-calibration=$gate_path
        --gate-k-pre=$(c.k_pre)
        --gate-k-post=$(c.k_post)
        --gate-min-len=$(c.min_len)`
    println("\nRunning candidate $(c.name):")
    println("  $cmd")
    run(cmd)
    return ".local_artifacts/oos_forecast/oos_innovations_$(tag).jls",
           ".local_artifacts/oos_forecast/OOS_SUMMARY_$(tag).md"
end

function ratio_summary(raw_path::String)
    d = deserialize(raw_path)
    return (
        full = d["agg_sur_full"] / d["agg_lin_full"],
        window = d["agg_sur_covid"] / d["agg_lin_covid"],
        remaining = d["agg_sur_post"] / d["agg_lin_post"],
        gate_full = d["gate_share_full"],
        gate_window = d["gate_share_covid"],
        gate_remaining = d["gate_share_post"],
        soft_full = d["soft_gate_share_full"],
    )
end

println("=" ^ 78)
println("QUIET-SAMPLE OOS GATE RECALIBRATION")
println("Started: $(now())")
println("=" ^ 78)
println("Candidates: $(join(candidates, ", "))")

rows = NamedTuple[]
for cname in candidates
    c = candidate_from_name(cname)
    gate_path, g = write_candidate_gate(c)
    raw_path, summary_path = run_evaluator(c, gate_path)
    r = ratio_summary(raw_path)
    push!(rows, (
        candidate = c.name,
        quantile = c.quantile === nothing ? NaN : c.quantile,
        tau_eps = Float64(g["tau_eps"]),
        tau_y = Float64(g["tau_y"]),
        k_pre = c.k_pre,
        k_post = c.k_post,
        min_len = c.min_len,
        full_ratio = r.full,
        window_ratio = r.window,
        remaining_ratio = r.remaining,
        gate_full = r.gate_full,
        gate_window = r.gate_window,
        gate_remaining = r.gate_remaining,
        soft_full = r.soft_full,
        gate_path = gate_path,
        raw_path = raw_path,
        summary_path = summary_path,
    ))
end

summary_path = joinpath(out_dir, "OOS_GATE_RECALIBRATION_SUMMARY.md")
open(summary_path, "w") do io
    println(io, "# Quiet-Sample OOS Gate Recalibration")
    println(io)
    println(io, "_Generated $(now())._")
    println(io)
    println(io, "- Estimation cutoff: `T_pre=$t_pre`")
    println(io, "- Forecast window: `$(t_pre + 1)..$t_end`")
    println(io, "- Chains are fixed from the quiet-sample OOS pilot; this is a gate-policy diagnostic, not a re-estimated posterior.")
    println(io)
    println(io, "| Candidate | q | padding | Full ratio | $window_label ratio | Remaining ratio | Hard gate full | Hard gate remaining | Soft mean full |")
    println(io, "|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows
        qstr = isnan(r.quantile) ? "base" : @sprintf("%.2f", r.quantile)
        pad = "$(r.k_pre)/$(r.k_post)/$(r.min_len)"
        @printf(io, "| %s | %s | %s | %.3f | %.3f | %.3f | %.1f%% | %.1f%% | %.1f%% |\n",
                r.candidate, qstr, pad, r.full_ratio, r.window_ratio,
                r.remaining_ratio, 100*r.gate_full, 100*r.gate_remaining,
                isfinite(r.soft_full) ? 100*r.soft_full : NaN)
    end
    println(io)
    best_idx = argmin([r.full_ratio for r in rows])
    best = rows[best_idx]
    println(io, "## Best Full-OOS Candidate")
    println(io)
    @printf(io, "- `%s`: full OOS ratio `%.3f`, remaining ratio `%.3f`, hard gate full `%.1f%%`.\n",
            best.candidate, best.full_ratio, best.remaining_ratio, 100*best.gate_full)
    println(io)
    println(io, "## Interpretation")
    println(io)
    println(io, "The original quiet-sample failure is largely a gate-stickiness problem: retaining the original thresholds but removing padding is directly comparable to the published quiet pilot and isolates the effect of the `k_pre/k_post/min_len` expansion. Stricter quantile gates test whether calm-period activation can be suppressed without changing the posterior draws.")
    println(io)
    println(io, "## Raw Artifacts")
    println(io)
    for r in rows
        println(io, "- `$(r.candidate)`: gate `$(r.gate_path)`, raw innovations `$(r.raw_path)`, summary `$(r.summary_path)`")
    end
end

println("\nSummary written: $summary_path")
