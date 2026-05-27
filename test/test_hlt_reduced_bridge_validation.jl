using Test
using TOML

include(joinpath(@__DIR__, "..", "scripts", "hlt_reduced_bridge_validation.jl"))

@testset "HLT reduced validation bridge" begin
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
            "--observables=dy,dinve,robs",
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
        @test disk["observables"] == ["dy", "dinve", "robs"]
        @test occursin("Reduced SW07-HLT Validation Bridge", read(summary_path, String))
    end

    if get(ENV, "RUN_HLT_BRIDGE_EXEC_SMOKE", "0") == "1"
        @testset "executable smoke stage" begin
            out_dir = mktempdir()
            opts = parse_bridge_args([
                "--stage=smoke",
                "--out-dir=$out_dir",
                "--run-id=test_bridge_exec_smoke",
                "--periods=1",
                "--grid-axis=1",
                "--direct-eval-points=1",
                "--sep-horizon=2",
                "--sep-maxit=20",
                "--observables=dy,dinve,robs",
            ])
            manifest = run_bridge(opts)
            run_dir = joinpath(out_dir, "test_bridge_exec_smoke")
            @test manifest["smoke_linear_ok_count"] == 1
            @test haskey(manifest, "smoke_direct_ok_count")
            @test isfile(joinpath(run_dir, "direct_grid_payload.jls"))
            @test isfile(joinpath(run_dir, "comparison_table.tex"))
        end
    end
end
