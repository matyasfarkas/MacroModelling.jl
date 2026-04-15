using Test
using Serialization
using MCMCChains
using TOML

include(joinpath(@__DIR__, "..", "scripts", "extract_hlt_real_data_tables.jl"))

@testset "HLT real-data table extraction" begin
    mktempdir() do tmp
        chain_path = joinpath(tmp, "chain_payload.jls")
        out_dir = joinpath(tmp, "tables")
        generated_dir = joinpath(tmp, "generated")
        manifest_path = joinpath(tmp, "run_manifest.toml")

        vals = zeros(300, 3, 2)
        vals[:, 1, :] .= 0.6 .+ 0.02 .* randn(300, 2)
        vals[:, 2, :] .= 0.45 .+ 0.03 .* randn(300, 2)
        vals[:, 3, :] .= 10.0 .+ 0.8 .* randn(300, 2)
        chain = Chains(vals, [:cprobp, :cindp, :curvp])

        payload = Dict{String,Any}(
            "chain" => chain,
            "gate_share" => 0.2,
            "gate_mask" => Bool[true, false, true, false, false],
            "gate_info" => Dict(
                "gate_mode" => "hard",
                "k_pre" => 0,
                "k_post" => 0,
                "min_len" => 1,
                "filter" => "kalman",
                "tau_eps" => 1.2,
                "tau_y" => 0.8,
            ),
            "shock_filter" => "inversion",
            "linear_filter" => "inversion",
            "synthetic_path" => "dummy_payload_path.jls",
        )
        serialize(chain_path, payload)

        open(manifest_path, "w") do io
            TOML.print(io, Dict(
                "name" => "test-run",
                "mode" => "benchmark",
                "created_at" => "2026-03-04T12:00:00",
                "git_commit" => "abc123",
                "run_dir" => tmp,
                "samples" => 300,
                "chains" => 2,
                "steps" => Dict(
                    "switching_estimation" => Dict("elapsed_s" => 120.0),
                    "gate_calibration" => Dict("elapsed_s" => 5.0),
                    "surrogate_train" => Dict("elapsed_s" => 8.0),
                ),
            ))
        end

        outputs = extract_hlt_real_data_tables(
            chain_path;
            out_dir = out_dir,
            run_manifest_path = manifest_path,
            generated_dir = generated_dir,
        )

        @test isfile(outputs["posterior_tex"])
        @test isfile(outputs["mcmc_tex"])
        @test isfile(outputs["gate_tex"])
        @test isfile(outputs["runmeta_tex"])
        @test isfile(outputs["summary_toml"])
        @test isfile(outputs["summary_json"])

        summary = TOML.parsefile(outputs["summary_toml"])
        @test haskey(summary, "param_rows")
        @test length(summary["param_rows"]) == 3
        @test haskey(summary, "chain_meta")
        @test haskey(summary, "gate_stats")

        @test isfile(joinpath(generated_dir, basename(outputs["posterior_tex"])))
        @test isfile(joinpath(generated_dir, basename(outputs["summary_toml"])))

        tex_text = read(outputs["posterior_tex"], String)
        @test occursin("real-data application", tex_text)
        @test occursin("cprobp", tex_text)
    end
end
