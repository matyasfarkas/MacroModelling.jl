using Test
using Serialization
using TOML

module RobustnessSweepTest
include(joinpath(@__DIR__, "..", "scripts", "hlt_bridge_robustness_sweep.jl"))
end

module InversionBridgeTest
include(joinpath(@__DIR__, "..", "scripts", "hlt_bridge_inversion_filter_compare.jl"))
end

@testset "HLT bridge scaffold scripts" begin
    @testset "robustness sweep parser" begin
        opts = RobustnessSweepTest.parse_args([
            "--stage=postprocess",
            "--run-id=test_grid5",
            "--dataset-dir=/tmp/test_grid5",
            "--grid=5",
            "--epochs=12",
            "--obs-sigma-scale=1.25",
        ])
        @test opts.stage == "postprocess"
        @test opts.run_id == "test_grid5"
        @test opts.grid == 5
        @test opts.epochs == 12
        @test opts.obs_sigma_scale == 1.25
    end

    @testset "inversion bridge dry-run manifest" begin
        tmp = mktempdir()
        dataset_path = joinpath(tmp, "dataset.jls")
        surrogate_path = joinpath(tmp, "surrogate.jls")
        out_dir = joinpath(tmp, "out")
        dataset = Dict{String,Any}(
            "meta" => Dict{String,Any}(
                "theta_names" => [:crhob, :crhoqs, :z_eb, :z_eqs],
                "theta_grid" => [[0.55, 0.68, 1.20, 0.53], [0.65, 0.77, 1.85, 0.72]],
                "observables" => [:dy, :dc],
            ),
            "X" => zeros(10, 2),
            "Y" => [1.0 1.1; 2.0 2.2],
            "Y_rom1" => [0.9 1.0; 1.9 2.1],
        )
        surrogate = Dict{String,Any}("frozen" => "placeholder")
        serialize(dataset_path, dataset)
        serialize(surrogate_path, surrogate)

        opts = InversionBridgeTest.parse_args([
            "--dataset=$dataset_path",
            "--surrogate=$surrogate_path",
            "--out-dir=$out_dir",
            "--periods=2",
            "--direct-eval-points=1",
            "--dry-run=true",
        ])
        manifest = InversionBridgeTest.run_inversion_bridge(opts)
        disk = TOML.parsefile(joinpath(out_dir, "manifest.toml"))

        @test manifest["dry_run"] == true
        @test manifest["truth_in_validation_split"] == true
        @test disk["param_set"] == "investment_4p_supported"
        @test isfile(joinpath(out_dir, "SUMMARY.md"))
        @test "direct_inversion_grid.jls" in manifest["artifact_schema"]
    end
end
