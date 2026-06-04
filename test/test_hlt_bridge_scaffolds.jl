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
            "--surrogate-objective=dynamic-ridge-residual",
            "--dynamic-calibration-train-points=1",
            "--dynamic-ridge-lambda=0.01",
            "--dry-run=true",
        ])
        manifest = InversionBridgeTest.run_inversion_bridge(opts)
        disk = TOML.parsefile(joinpath(out_dir, "manifest.toml"))

        @test manifest["dry_run"] == true
        @test manifest["truth_in_validation_split"] == true
        @test manifest["surrogate_objective"] == "dynamic-ridge-residual"
        @test manifest["dynamic_calibration_train_points"] == 1
        @test manifest["dynamic_ridge_lambda"] == 0.01
        @test disk["param_set"] == "investment_4p_supported"
        @test isfile(joinpath(out_dir, "SUMMARY.md"))
        @test "direct_inversion_grid.jls" in manifest["artifact_schema"]
        @test "dynamic_ridge_residual.jls" in manifest["artifact_schema"]
    end

    @testset "dynamic ridge helper functions" begin
        predict_tuple = (state, shock, theta) -> begin
            obs = [state[1] + shock[1] + theta[1]]
            state_next = [state[1] + 2.0 * shock[1]]
            return obs, state_next
        end
        shocks = reshape([1.0, 2.0, 3.0], 1, :)
        states, rom_obs = InversionBridgeTest.rollout_path_cache(
            predict_tuple,
            [1.0],
            shocks,
            [0.5],
            1,
        )
        @test states ≈ reshape([1.0, 3.0, 7.0], 1, :)
        @test rom_obs ≈ reshape([2.5, 5.5, 10.5], 1, :)

        X_dyn = InversionBridgeTest.dynamic_feature_matrix(
            states,
            shocks,
            [0.5, 0.75],
            "state-shock-theta-time",
        )
        @test size(X_dyn) == (5, 3)
        @test X_dyn[end, :] ≈ [0.0, 0.5, 1.0]

        X_train = [
            0.0 1.0 2.0 3.0 4.0;
            1.0 -1.0 2.0 -2.0 0.5
        ]
        Y_train = [
            1.0 + 2.0 * X_train[1, i] - 0.5 * X_train[2, i] for i in 1:size(X_train, 2)
        ]'
        ridge = InversionBridgeTest.fit_dynamic_ridge_residual(X_train, Matrix(Y_train), 0.0)
        Y_hat = InversionBridgeTest.predict_dynamic_ridge(ridge, X_train)
        @test Y_hat ≈ Matrix(Y_train) atol = 1.0e-10
    end
end
