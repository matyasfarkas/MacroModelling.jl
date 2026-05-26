using Test
using TOML

include(joinpath(@__DIR__, "..", "scripts", "gali_elb_rom1_residual_validation.jl"))

@testset "Gali ELB ROM1-residual validation runner" begin
    @testset "Dry-run manifest schema" begin
        out_dir = mktempdir()
        opts = parse_elb_args([
            "--stage=smoke",
            "--dry-run=true",
            "--out-dir=$out_dir",
            "--run-id=test_elb_schema",
        ])
        cfg = apply_elb_overrides(elb_stage_config(opts.stage), opts)
        manifest = write_elb_manifest(out_dir, opts, cfg, ELB_MODEL)
        manifest_disk = TOML.parsefile(joinpath(out_dir, "manifest.toml"))

        @test manifest["stage"] == "smoke"
        @test manifest_disk["model"] == "Gali_2015_chapter_3_obc"
        @test manifest_disk["hard_nonlinearity"] == "ELB max operator in R equation"
        @test manifest_disk["first_parameter_block"] == ["std_nu"]
        @test manifest_disk["observables"] == ["log_y", "pi_ann", "i_ann"]
        @test manifest_disk["theta_true"] == ELB_THETA_TRUE
        @test manifest_disk["theta_baseline"] == ELB_THETA_BASELINE
        @test manifest_disk["target_definition"] == "observable residual: direct SEP observable minus ROM1 observable"
        @test occursin("known-state", manifest_disk["posterior_objective"])
        @test "grid_payload.jls" in manifest_disk["artifact_schema"]
        @test "next: replace known shocks with ROM1/surrogate inversion-filter shocks" in manifest_disk["scale_up_sequence"]
    end

    @testset "CLI dry-run executes" begin
        out_dir = mktempdir()
        cmd = `$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) $(joinpath(@__DIR__, "..", "scripts", "gali_elb_rom1_residual_validation.jl")) --stage=smoke --dry-run=true --out-dir=$out_dir --run-id=test_elb_cli`
        @test success(pipeline(cmd; stdout = devnull, stderr = stderr))
        manifest_disk = TOML.parsefile(joinpath(out_dir, "manifest.toml"))
        @test manifest_disk["artifact_schema"][1] == "manifest.toml"
    end

    @testset "Optional executable ELB smoke" begin
        if get(ENV, "RUN_GALI_ELB_VALIDATION_SMOKE", "0") == "1"
            out_dir = mktempdir()
            opts = parse_elb_args([
                "--stage=smoke",
                "--out-dir=$out_dir",
                "--run-id=test_elb_smoke",
                "--periods=2",
                "--train-samples=4",
                "--grid-size=3",
                "--hidden-units=4",
            ])
            result = run_elb_validation(opts)
            @test isfile(joinpath(out_dir, "synthetic_dgp.jls"))
            @test isfile(joinpath(out_dir, "surrogate_bundle.jls"))
            @test isfile(joinpath(out_dir, "grid_payload.jls"))
            @test isfile(joinpath(out_dir, "SUMMARY.md"))
            @test haskey(result["acceptance"], "stage_pass")
        else
            @test true
        end
    end
end
