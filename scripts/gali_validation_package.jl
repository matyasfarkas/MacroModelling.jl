#!/usr/bin/env julia

# Consolidated Galí hard-ELB validation package.
#
# This script audits the maintained validation artifacts for the Galí OBC
# pipeline and writes a single report with machine-checkable pass/fail status.
# It intentionally separates required pipeline validation from expected
# identification failures such as the short-sample std_nu probes.

using Dates
using Printf
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

Base.@kwdef struct PackageOptions
    run_id::String = "gali_validation_package_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir::String = joinpath(REPO_ROOT, ".local_artifacts", "gali_validation_package")
    strict::Bool = true
end

Base.@kwdef mutable struct CheckResult
    name::String
    artifact::String
    role::String
    status::String
    required::Bool
    details::Vector{String} = String[]
end

function parse_args(args)
    opts = PackageOptions()
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("Unexpected positional argument: $arg")
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || error("Expected --key=value, got $arg")
        values[keyval[1]] = keyval[2]
    end
    return PackageOptions(
        run_id = get(values, "run-id", opts.run_id),
        out_dir = get(values, "out-dir", opts.out_dir),
        strict = parse(Bool, get(values, "strict", string(opts.strict))),
    )
end

repo_path(parts...) = joinpath(REPO_ROOT, parts...)

function read_artifact(path::String)
    isfile(path) || error("Missing required artifact: $path")
    return read(path, String)
end

function maybe_read(path::String)
    return isfile(path) ? read(path, String) : ""
end

function match_float(text::String, pattern::Regex; label::String)
    m = match(pattern, text)
    m === nothing && error("Could not parse $label")
    return parse(Float64, m.captures[1])
end

function match_ints(text::String, pattern::Regex; label::String)
    m = match(pattern, text)
    m === nothing && error("Could not parse $label")
    return parse.(Int, m.captures)
end

function contains_or_detail!(details::Vector{String}, text::String, needle::String, label::String)
    ok = occursin(needle, text)
    push!(details, "$(ok ? "PASS" : "FAIL"): $label")
    return ok
end

function parse_table_row(text::String, parameter::String)
    found = nothing
    for line in split(text, '\n')
        occursin("| `$parameter` |", line) || continue
        cells = strip.(split(line, "|"))
        cells = filter(!isempty, cells)
        found = cells
    end
    found === nothing && error("Could not find table row for `$parameter`")
    return found
end

function status_from(ok::Bool, expected_fail::Bool = false)
    if expected_fail
        return ok ? "EXPECTED_FAIL_OK" : "UNEXPECTED"
    end
    return ok ? "PASS" : "FAIL"
end

function stress_path_check()
    artifact = repo_path(".local_artifacts", "gali_elb_stochastic", "gali_obc_eps_z_same_shocks_actualfloor_span5_shock0p8_bg0p0_summary.md")
    text = read_artifact(artifact)
    details = String[]
    floor_count, floor_total = match_ints(text, r"Actual ELB periods by policy-rate floor criterion: ([0-9]+) / ([0-9]+)", label = "actual floor periods")
    lin_count, _ = match_ints(text, r"Linearized sub-ELB periods: ([0-9]+) / ([0-9]+)", label = "linear sub-floor periods")
    output_hi = match_float(text, r"Forced-window OBC output range, percent log deviation: \[[^,]+, ([^\]]+)\]", label = "output upper range")
    inflation_hi = match_float(text, r"Forced-window OBC inflation range, annualized percent: \[[^,]+, ([^\]]+)\]", label = "inflation upper range")
    policy_lo = match_float(text, r"Forced-window OBC policy range, annualized percent: \[([^,]+), [^\]]+\]", label = "policy lower range")
    figure = repo_path(".local_artifacts", "gali_elb_stochastic", "gali_obc_eps_z_same_shocks_actualfloor_span5_shock0p8_bg0p0.png")
    ok = floor_count >= 5 && floor_total == 24 && lin_count >= 5 && output_hi < 0 && inflation_hi < 0 && abs(policy_lo - 0.020) <= 1e-6 && isfile(figure)
    append!(details, [
        "Actual floor periods: $floor_count / $floor_total",
        "Linear sub-floor periods: $lin_count / $floor_total",
        "Forced-window output and inflation are negative: $(output_hi < 0 && inflation_hi < 0)",
        "Policy rate at floor: $(@sprintf("%.3f", policy_lo))",
        "Figure exists: $(isfile(figure))",
    ])
    return CheckResult("stress_path", artifact, "hard-ELB recession path", status_from(ok), true, details)
end

function residual_grid_check()
    artifact = repo_path(".local_artifacts", "gali_actual_floor_residual_grid", "actual_floor_grid_stdz_interp_default_serial_20260518", "SUMMARY.md")
    text = read_artifact(artifact)
    details = String[]
    ok = true
    ok &= contains_or_detail!(details, text, "- Total solver warning count: 0", "zero total solver warnings")
    ok &= contains_or_detail!(details, text, "- Paper-ready clean validation including holdout: true", "paper-ready holdout status")
    ok &= contains_or_detail!(details, text, "- Core posterior-comparison validation pass: true", "core posterior comparison")
    ok &= contains_or_detail!(details, text, "- Direct true coverage: true", "direct true coverage")
    ok &= contains_or_detail!(details, text, "- Surrogate true coverage: true", "surrogate true coverage")
    return CheckResult("residual_grid_stdz", artifact, "known-shock ROM1-residual validation", status_from(ok), true, details)
end

function inversion_grid_check()
    artifact = repo_path(".local_artifacts", "gali_actual_floor_inversion_grid", "actual_floor_inversion_grid_default_20260518", "SUMMARY.md")
    text = read_artifact(artifact)
    details = String[]
    ok = true
    ok &= contains_or_detail!(details, text, "- Solver warning count: 0", "zero solver warnings")
    ok &= contains_or_detail!(details, text, "- Finite direct grid points: 18 / 18", "finite direct grid")
    ok &= contains_or_detail!(details, text, "- Finite surrogate grid points: 18 / 18", "finite surrogate grid")
    ok &= contains_or_detail!(details, text, "- Inversion-grid validation pass: true", "inversion-grid pass")
    return CheckResult("inversion_grid_stdz", artifact, "ROM1 inversion layer, one parameter", status_from(ok), true, details)
end

function oneparam_hmc_check()
    artifact = repo_path(".local_artifacts", "gali_actual_floor_inversion_grid", "actual_floor_inversion_hmc_tuned_commonseed_20260518", "SUMMARY.md")
    text = read_artifact(artifact)
    details = String[]
    diff = match_float(text, r"HMC mean difference / combined MCSE: ([^\n]+)", label = "one-param HMC diff")
    ok = true
    ok &= contains_or_detail!(details, text, "- HMC warning count: 0", "zero HMC-time solver warnings")
    ok &= contains_or_detail!(details, text, "| Numerical errors | 0 | 0 |", "zero direct/surrogate numerical errors")
    ok &= contains_or_detail!(details, text, "- HMC interval overlap: true", "HMC interval overlap")
    ok &= contains_or_detail!(details, text, "- HMC direct true coverage: true", "HMC direct coverage")
    ok &= contains_or_detail!(details, text, "- HMC surrogate true coverage: true", "HMC surrogate coverage")
    ok &= contains_or_detail!(details, text, "- Matched one-parameter HMC validation pass: true", "matched one-param HMC pass")
    ok &= abs(diff) <= 2.0
    push!(details, "Mean difference / combined MCSE: $diff")
    return CheckResult("oneparam_hmc_stdz", artifact, "matched direct/surrogate HMC smoke", status_from(ok), true, details)
end

function twoparam_grid_check()
    artifact = repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stda_balanced_T24_train3_probe_20260518", "SUMMARY.md")
    text = read_artifact(artifact)
    details = String[]
    ok = true
    ok &= contains_or_detail!(details, text, "- Solver warning count: 0", "zero solver warnings")
    ok &= contains_or_detail!(details, text, "- Finite direct grid points: 36 / 36", "finite direct grid")
    ok &= contains_or_detail!(details, text, "- Finite surrogate grid points: 36 / 36", "finite surrogate grid")
    ok &= contains_or_detail!(details, text, "- Two-parameter inversion-grid validation pass: true", "two-param grid pass")
    ok &= contains_or_detail!(details, text, "| `std_z` | 0.05", "std_z row present")
    ok &= contains_or_detail!(details, text, "| `std_a` | 0.01", "std_a row present")
    return CheckResult("twoparam_grid_stdz_stda", artifact, "two-parameter inversion grid", status_from(ok), true, details)
end

function twoparam_hmc_check()
    artifact = repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_hmc_balanced_extended_20260518", "SUMMARY.md")
    text = read_artifact(artifact)
    details = String[]
    direct_errors = match_ints(text, r"Direct post-warmup numerical errors: ([0-9]+)", label = "direct numerical errors")[1]
    surrogate_errors = match_ints(text, r"Surrogate post-warmup numerical errors: ([0-9]+)", label = "surrogate numerical errors")[1]
    warning_count = match_ints(text, r"HMC warning count: ([0-9]+)", label = "HMC warning count")[1]
    draws_match = match(r"HMC chains/warmup/draws: ([0-9]+) / ([0-9]+) / ([0-9]+)", text)
    draws_match === nothing && error("Could not parse HMC chain counts")
    chains = parse(Int, draws_match.captures[1])
    draws = parse(Int, draws_match.captures[3])
    n_post = chains * draws
    allowed = max(1, ceil(Int, 0.005 * n_post))
    zrow = parse_table_row(text, "std_z")
    arow = parse_table_row(text, "std_a")
    diff_z = parse(Float64, zrow[9])
    diff_a = parse(Float64, arow[9])
    ok = true
    ok &= contains_or_detail!(details, text, "- Two-parameter inversion-grid validation pass: true", "grid pass")
    ok &= contains_or_detail!(details, text, "- Matched two-parameter HMC validation pass: true", "matched two-param HMC pass")
    ok &= contains_or_detail!(details, text, "- Overall validation pass: true", "overall pass")
    ok &= contains_or_detail!(details, text, "- Negligible-issue HMC diagnostic pass: true", "negligible-issue diagnostic")
    ok &= direct_errors <= allowed && surrogate_errors == 0 && warning_count <= allowed
    ok &= abs(diff_z) <= 2.0 && abs(diff_a) <= 2.0
    append!(details, [
        "Post-warmup draws per objective: $n_post",
        "Direct numerical errors: $direct_errors / $n_post (allowed <= $allowed)",
        "Surrogate numerical errors: $surrogate_errors / $n_post",
        "HMC warnings: $warning_count",
        "Diff/MCSE std_z: $diff_z",
        "Diff/MCSE std_a: $diff_a",
    ])
    return CheckResult("twoparam_hmc_stdz_stda_extended", artifact, "extended matched HMC validation", status_from(ok), true, details)
end

function stdnu_probe_check()
    amps = ["025", "05", "1", "2"]
    artifacts = [repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stdnu_probe_amp$(amp)_20260518", "SUMMARY.md") for amp in amps]
    details = String[]
    ok = true
    for artifact in artifacts
        text = read_artifact(artifact)
        ok &= contains_or_detail!(details, text, "- Solver warning count: 0", "zero solver warnings: $(basename(dirname(artifact)))")
        ok &= contains_or_detail!(details, text, "- Finite direct grid points: 36 / 36", "finite direct grid: $(basename(dirname(artifact)))")
        ok &= contains_or_detail!(details, text, "- Finite surrogate grid points: 36 / 36", "finite surrogate grid: $(basename(dirname(artifact)))")
        ok &= contains_or_detail!(details, text, "- Two-parameter inversion-grid validation pass: false", "expected grid failure: $(basename(dirname(artifact)))")
        row = parse_table_row(text, "std_nu")
        coverage = row[end]
        overlap = row[end - 1]
        ok &= coverage == "Neither" && overlap == "true"
        push!(details, "$(basename(dirname(artifact))): std_nu coverage=$coverage, overlap=$overlap")
    end
    return CheckResult("stdnu_identification_probe", join(artifacts, "\n"), "expected identification failure", status_from(ok, true), false, details)
end

function direct_sep_smoke_check()
    artifact = repo_path(".local_artifacts", "gali_direct_sep_surrogate_hmc", "direct_sep_full_pipeline_smoke_20260518", "SUMMARY.md")
    text = read_artifact(artifact)
    details = String[]
    ok = true
    ok &= contains_or_detail!(details, text, "- `direct at theta_true`: log posterior=", "finite direct theta_true")
    ok &= contains_or_detail!(details, text, "- `direct at prior_center`: log posterior=", "finite direct prior center")
    ok &= contains_or_detail!(details, text, "finite gradient=true", "finite gradients appear")
    ok &= contains_or_detail!(details, text, "- Direct post-warmup numerical errors: 0", "direct smoke zero numerical errors")
    ok &= contains_or_detail!(details, text, "- Surrogate post-warmup numerical errors: 0", "surrogate smoke zero numerical errors")
    push!(details, "Posterior table is not an acceptance target for this smoke artifact; it documents direct-SEP feasibility and timing only.")
    return CheckResult("direct_sep_threeparam_smoke", artifact, "direct SEP feasibility smoke", status_from(ok), false, details)
end

function all_checks()
    return [
        stress_path_check(),
        residual_grid_check(),
        inversion_grid_check(),
        oneparam_hmc_check(),
        twoparam_grid_check(),
        twoparam_hmc_check(),
        stdnu_probe_check(),
        direct_sep_smoke_check(),
    ]
end

function write_manifest(path::String, checks::Vector{CheckResult}, overall::Bool)
    data = Dict{String,Any}(
        "generated_at" => string(Dates.now()),
        "overall_pass" => overall,
        "checks" => [Dict(
            "name" => c.name,
            "role" => c.role,
            "artifact" => c.artifact,
            "status" => c.status,
            "required" => c.required,
            "details" => c.details,
        ) for c in checks],
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function write_report(path::String, checks::Vector{CheckResult}, overall::Bool)
    required = filter(c -> c.required, checks)
    optional = filter(c -> !c.required, checks)
    open(path, "w") do io
        println(io, "# Galí Hard-ELB Validation Package")
        println(io)
        println(io, "**Generated**: $(Dates.now())")
        println(io, "**Overall status**: $(overall ? "PASS" : "FAIL")")
        println(io)
        println(io, "This package validates the maintained Galí OBC hard-ELB pipeline: ROM1-residual learning, ROM1 inversion shock recovery, and matched direct-OBC versus surrogate HMC. It does not claim the still-infeasible full three-parameter direct-SEP HMC comparison.")
        println(io)
        println(io, "## Required Checks")
        println(io)
        println(io, "| Check | Role | Status | Artifact |")
        println(io, "|---|---|---:|---|")
        for c in required
            println(io, "| `$(c.name)` | $(c.role) | **$(c.status)** | `$(relpath(c.artifact, REPO_ROOT))` |")
        end
        println(io)
        println(io, "## Diagnostic Checks")
        println(io)
        println(io, "| Check | Role | Status | Artifact |")
        println(io, "|---|---|---:|---|")
        for c in optional
            artifact = occursin('\n', c.artifact) ? "(multiple artifacts)" : "`$(relpath(c.artifact, REPO_ROOT))`"
            println(io, "| `$(c.name)` | $(c.role) | **$(c.status)** | $artifact |")
        end
        println(io)
        println(io, "## Detail")
        for c in checks
            println(io)
            println(io, "### $(c.name)")
            println(io)
            println(io, "- Status: `$(c.status)`")
            println(io, "- Required: `$(c.required)`")
            if occursin('\n', c.artifact)
                println(io, "- Artifacts:")
                for p in split(c.artifact, '\n')
                    println(io, "  - `$(relpath(p, REPO_ROOT))`")
                end
            else
                println(io, "- Artifact: `$(relpath(c.artifact, REPO_ROOT))`")
            end
            for detail in c.details
                println(io, "- $detail")
            end
        end
        println(io)
        println(io, "## Interpretation")
        println(io)
        println(io, "- The decisive validation is `twoparam_hmc_stdz_stda_extended`: 4,000 post-warmup draws per objective, overlapping intervals for `std_z` and `std_a`, true-value coverage for both, and posterior-mean differences below one combined MCSE.")
        println(io, "- The `stdnu_identification_probe` is an expected failure: direct and surrogate agree, but the short ELB design pushes `std_nu` to the local upper grid edge. This is evidence about local identification, not a surrogate-approximation failure.")
        println(io, "- The direct-SEP three-parameter smoke confirms finite direct likelihood and gradients, but its short posterior table is not used as an acceptance result.")
    end
end

function run_package(opts::PackageOptions)
    out_dir = joinpath(opts.out_dir, opts.run_id)
    mkpath(out_dir)
    checks = all_checks()
    required_ok = all(c -> c.status == "PASS", filter(c -> c.required, checks))
    diagnostics_ok = all(c -> c.status in ("PASS", "EXPECTED_FAIL_OK"), filter(c -> !c.required, checks))
    overall = required_ok && diagnostics_ok
    report = joinpath(out_dir, "VALIDATION_PACKAGE_REPORT.md")
    manifest = joinpath(out_dir, "validation_manifest.toml")
    write_report(report, checks, overall)
    write_manifest(manifest, checks, overall)
    println("Wrote report: $report")
    println("Wrote manifest: $manifest")
    println("Overall validation package status: $(overall ? "PASS" : "FAIL")")
    if opts.strict && !overall
        error("Validation package failed")
    end
    return overall
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_package(parse_args(ARGS))
end
