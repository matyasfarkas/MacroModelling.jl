using Test
import TOML
using Serialization

module HLTValidationHarnessScript
include(joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl"))
end

@testset "HLT Validation Harness (dry-run)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --artifact-root=$d --mode=smoke`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))

        manifest_path = joinpath(run_dir, "manifests", "run_manifest.toml")
        summary_path = joinpath(run_dir, "manifests", "RUN_SUMMARY.md")
        @test isfile(manifest_path)
        @test isfile(summary_path)

        manifest = TOML.parsefile(manifest_path)
        @test manifest["mode"] == "smoke"
        @test manifest["dry_run"] == true
        @test haskey(manifest, "steps")
        @test haskey(manifest["steps"], "dataset_generate")
        @test haskey(manifest["steps"], "switching_estimation")
        @test manifest["steps"]["switching_estimation"]["status"] == "planned"
    end
end

@testset "HLT Validation Harness (quick-smoke dry-run)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --quick-smoke=true --skip-fom=true --artifact-root=$d --mode=smoke`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest_path = joinpath(run_dir, "manifests", "run_manifest.toml")
        @test isfile(manifest_path)

        manifest = TOML.parsefile(manifest_path)
        @test manifest["mode"] == "smoke"
        @test manifest["quick_smoke"] == true
        @test haskey(manifest["steps"], "dataset_generate")
        @test haskey(manifest["steps"], "switching_estimation")
        @test !haskey(manifest["steps"], "fom_benchmark")
    end
end

@testset "HLT Validation Harness (bare boolean flags)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run --quick-smoke --skip-fom --artifact-root=$d --mode=smoke`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))

        @test manifest["dry_run"] == true
        @test manifest["quick_smoke"] == true
        @test !haskey(manifest["steps"], "fom_benchmark")
        @test manifest["steps"]["dataset_generate"]["status"] == "planned"
    end
end

@testset "HLT Validation Harness (quick-smoke FOM config)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --quick-smoke=true --artifact-root=$d --mode=smoke`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))
        @test haskey(manifest["steps"], "fom_benchmark")

        cmd = manifest["steps"]["fom_benchmark"]["command"]
        @test occursin("--benchmark-preset=direct_sep_gated_smoke", cmd)
        @test occursin("--recovery-ladder=true", cmd)
        @test occursin("--labels=true", cmd)
        @test occursin("--fallback-algorithm=first_order", cmd)
        @test !occursin("--period-selection=gated", cmd)
    end
end

@testset "HLT Validation Harness (quick-smoke FOM preset override)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --quick-smoke=true --artifact-root=$d --mode=smoke --quick-smoke-fom-preset=direct_sep_gated_smoke_order1_tuned`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))
        @test manifest["quick_smoke_fom_preset"] == "direct_sep_gated_smoke_order1_tuned"

        cmd = manifest["steps"]["fom_benchmark"]["command"]
        @test occursin("--benchmark-preset=direct_sep_gated_smoke_order1_tuned", cmd)
        @test occursin("--recovery-ladder=true", cmd)
    end
end

@testset "HLT Validation Harness (require direct FOM ok flag)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --quick-smoke=true --artifact-root=$d --mode=smoke --require-direct-fom-ok=true`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))
        @test manifest["require_direct_fom_ok"] == true
    end
end

@testset "HLT Validation Harness (acceptance smoke dry-run config)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --quick-smoke=true --skip-fom=true --artifact-root=$d --mode=smoke --run-acceptance-smoke=true --require-acceptance-smoke-ok=true --acceptance-smoke-run-inversion-benchmark-panel=true --acceptance-smoke-direct-fit-margin=0.01 --acceptance-smoke-theta-tols=0.1,0.2,3.0`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))

        @test manifest["run_acceptance_smoke"] == true
        @test manifest["require_acceptance_smoke_ok"] == true
        @test manifest["acceptance_smoke_run_inversion_benchmark_panel"] == true
        @test manifest["acceptance_smoke_direct_fit_margin"] == 0.01
        @test manifest["acceptance_smoke_theta_tols"] == [0.1, 0.2, 3.0]
        @test haskey(manifest["steps"], "acceptance_smoke")

        cmd = manifest["steps"]["acceptance_smoke"]["command"]
        @test occursin("hlt_sep_surrogate_acceptance_smoke.jl", cmd)
        @test occursin("--run-inversion-benchmark-panel=true", cmd)
        @test occursin("--min-truthshock-direct-improvement=0.01", cmd)
        @test occursin("--theta-tols=0.1,0.2,3.0", cmd)
    end
end

@testset "HLT Validation Harness (acceptance smoke parser)" begin
    mktempdir() do d
        missing_path = joinpath(d, "missing.toml")
        @test_throws ErrorException HLTValidationHarnessScript.validate_acceptance_smoke_output(missing_path)

        p = joinpath(d, "acceptance.toml")
        open(p, "w") do io
            TOML.print(io, Dict{String,Any}(
                "status" => "failed",
                "switching" => Dict("gate_share" => 0.25, "gate_vol_overlap_count" => 1),
                "recovery" => Dict("theta_abs_error" => [0.01, 0.02, 3.0], "theta_recovery_pass" => false),
                "truth_shock_fit" => Dict("status" => "skipped_dry_run"),
            ))
        end
        meta = HLTValidationHarnessScript.validate_acceptance_smoke_output(p)
        @test meta["acceptance_smoke_status"] == "failed"
        @test meta["acceptance_smoke_gate_share"] == 0.25
        @test meta["acceptance_smoke_gate_vol_overlap_count"] == 1
        @test meta["acceptance_smoke_theta_recovery_pass"] == false
        @test meta["acceptance_smoke_truth_shock_fit_status"] == "skipped_dry_run"
    end
end

@testset "HLT Validation Harness (benchmark bounded dry-run config)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --artifact-root=$d --mode=benchmark --benchmark-profile=bounded --samples=200 --chains=2`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))

        @test manifest["mode"] == "benchmark"
        @test manifest["benchmark_profile"] == "bounded"
        @test manifest["samples"] == 200
        @test manifest["chains"] == 2

        dataset_cmd = manifest["steps"]["dataset_generate"]["command"]
        @test occursin("--theta-sampling=grid", dataset_cmd)
        @test occursin("--grid=2", dataset_cmd)
        @test occursin("--rom-orders=1", dataset_cmd)
        @test occursin("--stable-prefix", dataset_cmd)

        switch_cmd = manifest["steps"]["switching_estimation"]["command"]
        @test occursin("--samples=200", switch_cmd)
        @test occursin("--chains=2", switch_cmd)

        fom_cmd = manifest["steps"]["fom_benchmark"]["command"]
        @test occursin("--benchmark-preset=direct_sep_gated_smoke_order1_tuned", fom_cmd)
        @test occursin("--recovery-ladder=true", fom_cmd)
    end
end

@testset "HLT Validation Harness (benchmark reuse skip-estimation dry-run)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run_dir = joinpath(d, "hlt3_reuse")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --mode=benchmark --benchmark-profile=bounded --benchmark-skip-build=true --benchmark-skip-estimation=true --run-dir=$run_dir`)

        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))
        @test manifest["benchmark_skip_build"] == true
        @test manifest["benchmark_skip_estimation"] == true
        @test !haskey(manifest["steps"], "dataset_generate")
        @test !haskey(manifest["steps"], "surrogate_train")
        @test !haskey(manifest["steps"], "synthetic_data_generate")
        @test !haskey(manifest["steps"], "gate_calibration")
        @test !haskey(manifest["steps"], "switching_estimation")
        @test haskey(manifest["steps"], "fom_benchmark")
    end
end

@testset "HLT Validation Harness (benchmark reuse input validation)" begin
    mktempdir() do d
        @test_throws ErrorException HLTValidationHarnessScript.validate_benchmark_reuse_inputs(d)

        mkpath(joinpath(d, "dataset"))
        mkpath(joinpath(d, "synthetic"))
        for p in (
            joinpath(d, "dataset", "hlt_sep_surrogate_trained.jls"),
            joinpath(d, "synthetic", "hlt_sep_synth_data.jls"),
            joinpath(d, "synthetic", "gate_calibration.jls"),
        )
            write(p, "")
        end
        @test HLTValidationHarnessScript.validate_benchmark_reuse_inputs(d) === nothing
        @test_throws ErrorException HLTValidationHarnessScript.validate_benchmark_reuse_inputs(d; require_chain = true)
        write(joinpath(d, "synthetic", "hlt_sep_surrogate_estimation_chain.jls"), "")
        @test HLTValidationHarnessScript.validate_benchmark_reuse_inputs(d; require_chain = true) === nothing
    end
end

@testset "HLT Validation Harness (benchmark full dry-run config)" begin
    mktempdir() do d
        script = joinpath(@__DIR__, "..", "scripts", "hlt_sep_surrogate_validate_hlt3.jl")
        run(`julia --project=$(joinpath(@__DIR__, "..")) $script --dry-run=true --artifact-root=$d --mode=benchmark --benchmark-profile=full`)

        run_dirs = filter(name -> startswith(name, "hlt3_"), readdir(d))
        @test !isempty(run_dirs)
        run_dir = joinpath(d, first(run_dirs))
        manifest = TOML.parsefile(joinpath(run_dir, "manifests", "run_manifest.toml"))

        @test manifest["benchmark_profile"] == "full"
        dataset_cmd = manifest["steps"]["dataset_generate"]["command"]
        @test occursin("--theta-samples=36", dataset_cmd)
        @test occursin("--rom-orders=1,2", dataset_cmd)
    end
end
