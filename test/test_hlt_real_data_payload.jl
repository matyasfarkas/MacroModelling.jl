using Test

include(joinpath(@__DIR__, "..", "scripts", "hlt_real_data_payload.jl"))

@testset "HLT real-data payload builder" begin
    csv_path = joinpath(@__DIR__, "data", "usmodel_update.csv")

    payload = build_hlt_real_data_payload(
        csv_path = csv_path,
        model_name = "Smets_Wouters_2007_HLT_obc",
        sample_start = 47,
        sample_end = 290,
        prefix_end = 46,
        obs_sigma_mode = :data_std,
        obs_sigma_scale = 0.1,
        obs_sigma_floor = 1e-4,
    )

    @test payload["model"] == "Smets_Wouters_2007_HLT_obc"
    @test payload["sample_idx"] == collect(47:290)
    @test size(payload["obs_data"]) == (7, 244)
    @test size(payload["shocks"]) == (48, 244)
    @test length(payload["s0"]) > 0
    @test payload["theta_names"] == [:cprobp, :cindp, :curvp]
    @test payload["theta_true"] === nothing

    required = [
        "obs_data", "s0", "shocks", "theta_true", "theta_names",
        "state_names", "observables", "shock_sigmas", "obs_sigma",
        "sample_idx", "model",
    ]
    @test all(k -> haskey(payload, k), required)

    @test all(isfinite, payload["obs_data"])
    @test all(isfinite, payload["s0"])
    @test all(isfinite, payload["shock_sigmas"])
    @test all(isfinite, payload["obs_sigma"])
    @test all(payload["obs_sigma"] .> 0)
end
