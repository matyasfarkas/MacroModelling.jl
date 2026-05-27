using Test
using TOML

include(joinpath(@__DIR__, "..", "scripts", "hlt_reduced_bridge_validation.jl"))

@testset "HLT reduced validation bridge scaffold" begin
    @testset "argument parsing and defaults" begin
        opts = parse_bridge_args(["--stage=design", "--dry-run", "--parameter-block=investment_4p"])
        cfg = apply_overrides(stage_defaults(opts.stage), opts)
        manifest = bridge_manifest(opts, cfg)

        @test opts.dry_run
        @test manifest["model"] == "Smets_Wouters_2007_HLT_obc"
        @test manifest["theta_names"] == ["crhob", "crhoqs", "z_eb", "z_eqs"]
        @test manifest["observables"] == BRIDGE_DEFAULT_OBSERVABLES
        @test manifest["grid_points"] == 81
        @test "direct_grid_payload.jls" in manifest["artifact_schema"]
        @test occursin("Reduced SW07-HLT", manifest["purpose"])
    end

    @testset "dry run writes manifest and summary" begin
        out_dir = mktempdir()
        opts = parse_bridge_args([
            "--stage=smoke",
            "--dry-run=true",
            "--out-dir=$out_dir",
            "--run-id=test_bridge",
            "--periods=8",
            "--grid-axis=2",
            "--observables=dyobs,dinveobs,robs",
        ])
        manifest = run_bridge(opts)
        run_dir = joinpath(out_dir, "test_bridge")
        manifest_path = joinpath(run_dir, "manifest.toml")
        summary_path = joinpath(run_dir, "SUMMARY.md")
        disk = TOML.parsefile(manifest_path)

        @test manifest["periods"] == 8
        @test isfile(manifest_path)
        @test isfile(summary_path)
        @test disk["grid_points"] == 16
        @test disk["observables"] == ["dyobs", "dinveobs", "robs"]
        @test occursin("Reduced SW07-HLT Validation Bridge", read(summary_path, String))
    end
end
