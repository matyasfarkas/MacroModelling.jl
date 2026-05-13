using Test
using TOML

include(joinpath(@__DIR__, "..", "scripts", "gali_direct_sep_surrogate_hmc_validation.jl"))

@testset "Gali direct SEP vs surrogate HMC validation runner" begin
    @testset "Dry-run manifest schema" begin
        out_dir = mktempdir()
        opts = parse_args([
            "--stage=smoke",
            "--dry-run=true",
            "--out-dir=$out_dir",
            "--run-id=test_schema",
        ])
        cfg = apply_overrides(stage_config(opts.stage), opts)
        model = load_model()
        manifest = write_manifest(out_dir, opts, cfg, model)
        manifest_disk = TOML.parsefile(joinpath(out_dir, "manifest.toml"))

        @test manifest["stage"] == "smoke"
        @test manifest_disk["stage"] == "smoke"
        @test manifest_disk["dry_run"] == true
        @test manifest_disk["model"] == "Gali_2015_chapter_3_obc"
        @test manifest_disk["observables"] == ["log_y", "pi_ann", "i_ann"]
        @test manifest_disk["theta_names"] == ["std_a", "std_z", "std_nu"]
        @test manifest_disk["theta_true"] == THETA_TRUE
        @test manifest_disk["theta_baseline"] == THETA_BASELINE
        @test manifest_disk["prior_lower"] == PRIOR_LOWER
        @test manifest_disk["prior_upper"] == PRIOR_UPPER
        @test manifest_disk["direct_objective"] == "measurement_error"
        @test manifest_disk["direct_logdet_method"] == "exact"
        @test manifest_disk["hmc_objectives"] == "both"
        @test manifest_disk["surrogate_design"] == "local_path"
        @test manifest_disk["rom_filter_anchor_repeats"] == ROM_FILTER_ANCHOR_REPEATS
        @test manifest_disk["direct_audit_draws"] == 0
        @test manifest_disk["hmc_initial_step_size"] > 0
        @test "synthetic_dgp.jls" in manifest_disk["artifact_schema"]
        @test "direct_audit_payload.jls" in manifest_disk["artifact_schema"]
        @test "comparison_table.tex" in manifest_disk["artifact_schema"]
    end

    @testset "CLI dry-run executes" begin
        out_dir = mktempdir()
        cmd = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "scripts", "gali_direct_sep_surrogate_hmc_validation.jl")) --stage=smoke --dry-run=true --out-dir=$out_dir --run-id=test_cli`
        @test success(pipeline(cmd; stdout = devnull, stderr = stderr))
        manifest_disk = TOML.parsefile(joinpath(out_dir, "manifest.toml"))
        @test manifest_disk["artifact_schema"][1] == "manifest.toml"
    end

    @testset "Optional one-period finite-likelihood smoke" begin
        if get(ENV, "RUN_GALI_LIKELIHOOD_SMOKE", "0") == "1"
            out_dir = mktempdir()
            opts = parse_args([
                "--stage=smoke",
                "--likelihood-smoke-only=true",
                "--periods=1",
                "--train-samples=8",
                "--direct-objective=measurement_error",
                "--out-dir=$out_dir",
                "--run-id=test_likelihood",
            ])
            result = run_validation(opts)
            @test all(check -> check["ok"], result["checks"])
            @test isfile(joinpath(out_dir, "synthetic_dgp.jls"))
            @test isfile(joinpath(out_dir, "surrogate_bundle.jls"))
            @test isfile(joinpath(out_dir, "SUMMARY.md"))
        else
            @test true
        end
    end
end
