using Test
using Serialization
import TOML

module HLTAcceptanceSmokeScript
include(joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_acceptance_smoke.jl"))
end

const HLT_ACCEPTANCE_KNOWN_RUN = joinpath(
    @__DIR__, "..", ".local_artifacts", "hlt_validation_runs", "hlt3_20260225_210217"
)

@testset "HLT Acceptance Smoke Helpers" begin
    synthetic = Dict{String,Any}(
        "vol_start" => 3,
        "vol_end" => 5,
    )
    chain_summary = Dict{String,Any}(
        "gate_mask" => Bool[false, true, true, false, false],
        "gate_share" => 0.4,
    )
    sw = HLTAcceptanceSmokeScript.switching_metrics(chain_summary, synthetic; min_overlap = 1)
    @test sw["gate_share"] == 0.4
    @test sw["gate_vol_overlap_count"] == 1
    @test sw["gate_vol_overlap_indices"] == [3]

    chain_summary_bad = Dict{String,Any}("gate_mask" => Bool[true, true, true, true, true])
    @test_throws ErrorException HLTAcceptanceSmokeScript.switching_metrics(chain_summary_bad, synthetic)
end

@testset "HLT Acceptance Smoke Model Loader Helpers" begin
    file_obc, sym_obc = HLTAcceptanceSmokeScript.hlt_model_file_and_symbol("Smets_Wouters_2007_HLT_obc")
    @test file_obc == "Smets_Wouters_2007_HLT_obc.jl"
    @test sym_obc == :Smets_Wouters_2007_HLT_obc

    file_lin, sym_lin = HLTAcceptanceSmokeScript.hlt_model_file_and_symbol("Smets_Wouters_2007_HLT")
    @test file_lin == "Smets_Wouters_2007_HLT.jl"
    @test sym_lin == :Smets_Wouters_2007_HLT

    @test_throws ErrorException HLTAcceptanceSmokeScript.hlt_model_file_and_symbol("unsupported_hlt_model")
end

@testset "HLT Acceptance Smoke Recovery Metrics" begin
    chain_summary = Dict{String,Any}(
        "theta_true" => [0.6, 0.47, 10.0],
        "post_mean_theta" => [0.61, 0.50, 12.0],
    )
    rec = HLTAcceptanceSmokeScript.recovery_metrics(chain_summary; thresholds = [0.1, 0.15, 20.0])
    @test rec["theta_recovery_pass"] == true
    @test rec["theta_abs_error"] ≈ [0.01, 0.03, 2.0]

    @test_throws ErrorException HLTAcceptanceSmokeScript.recovery_metrics(chain_summary; thresholds = [0.001, 0.15, 20.0])
end

@testset "HLT Acceptance Smoke Benchmark Result Parsing" begin
    mktempdir() do d
        p = joinpath(d, "bench.jls")
        serialize(p, Dict{String,Any}(
            "algorithm_requested" => "stochastic_extended_path",
            "benchmark_preset" => "direct_sep_gated_smoke_order1_tuned",
            "selected_period_indices" => [3, 4],
            "evaluation_period_indices" => [4],
            "context_period_indices" => [3],
            "benchmark_is_subset" => true,
            "results" => Dict{String,Any}(
                "true" => Dict{String,Any}(
                    "status" => "ok",
                    "fom_loglik" => -12.3,
                    "algorithm_effective" => "stochastic_extended_path",
                    "recovery_ladder_enabled" => true,
                    "recovery_ladder_attempted" => false,
                    "recovery_rung_used" => nothing,
                    "attempts_count" => 1,
                    "sep_floor_failure_class" => nothing,
                ),
            ),
        ))
        parsed = HLTAcceptanceSmokeScript.parse_true_benchmark_result(p)
        @test parsed["status"] == "ok"
        @test parsed["fom_loglik"] == -12.3
        @test parsed["selected_period_indices"] == [3, 4]
    end
end

@testset "HLT Acceptance Smoke CLI Dry Run" begin
    isdir(HLT_ACCEPTANCE_KNOWN_RUN) || begin
        @info "Skipping CLI dry-run test because known run artifact is missing" HLT_ACCEPTANCE_KNOWN_RUN
        return
    end
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_acceptance_smoke.jl")
        out = joinpath(d, "result.toml")
        summary = joinpath(d, "summary.md")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script $HLT_ACCEPTANCE_KNOWN_RUN --dry-run=true --run-inversion-benchmark-panel=false --out=$out --summary=$summary --quiet=true --run-id-tag=testrun`)
        @test isfile(out)
        @test isfile(summary)
        payload = TOML.parsefile(out)
        @test payload["status"] == "ok"
        @test payload["dry_run"] == true
        @test payload["truth_shock_fit"]["status"] == "skipped_dry_run"
        @test !haskey(payload, "fom_vs_rom1")
    end
end

@testset "HLT Acceptance Smoke CLI Integration (optional)" begin
    if get(ENV, "RUN_HLT_ACCEPTANCE_INTEGRATION", "false") in ("1", "true", "TRUE")
        isdir(HLT_ACCEPTANCE_KNOWN_RUN) || error("Known HLT validation run artifact missing: $HLT_ACCEPTANCE_KNOWN_RUN")
        mktempdir() do d
            script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_acceptance_smoke.jl")
            out = joinpath(d, "result.toml")
            summary = joinpath(d, "summary.md")
            run(`julia --project=$(joinpath(@__DIR__, "..")) $script $HLT_ACCEPTANCE_KNOWN_RUN --out=$out --summary=$summary --quiet=true`)
            payload = TOML.parsefile(out)
            @test payload["status"] == "ok"
            @test payload["truth_shock_fit"]["direct_sep_better_than_rom1_fit_region"] == true
        end
    else
        @test true
    end
end
