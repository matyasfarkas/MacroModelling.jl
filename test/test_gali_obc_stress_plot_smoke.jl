using Test

include(joinpath(@__DIR__, "..", "scripts", "gali_obc_stochastic_comparison_plot.jl"))

@testset "Gali OBC adverse stress plot setup" begin
    opts = StochCompareOptions()
    model = Gali_2015_chapter_3_obc
    shocks = build_stochastic_shocks(model, opts)

    eps_z_idx = find_idx(model.exo, :eps_z)
    eps_nu_idx = find_idx(model.exo, :eps_nu)
    forced_window = opts.elb_period:min(opts.periods, opts.elb_period + opts.elb_span - 1)

    @test opts.shock_scale == 0.0
    @test opts.elb_shock_name == :eps_z
    @test opts.elb_shock > 0.0
    @test all(shocks[eps_z_idx, forced_window] .== opts.elb_shock)
    @test all(shocks[eps_nu_idx, :] .== 0.0)
    @test all(shocks[eps_z_idx, setdiff(1:opts.periods, forced_window)] .== 0.0)
end
