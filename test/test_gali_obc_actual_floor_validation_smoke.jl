using Test

include(joinpath(@__DIR__, "..", "scripts", "gali_obc_actual_floor_residual_grid_validation.jl"))

@testset "Gali actual-floor residual-grid validation setup" begin
    opts = FloorGridOptions()
    model = Gali_2015_chapter_3_obc
    shocks = effective_shock_matrix(model, opts, FLOOR_THETA_TRUE)

    eps_z_idx = find_idx(model.exo, :eps_z)
    eps_nu_idx = find_idx(model.exo, :eps_nu)
    forced_window = opts.elb_period:min(opts.periods, opts.elb_period + opts.elb_span - 1)
    grid = theta_grid(opts.grid_size)

    @test FLOOR_THETA_NAME == :std_z
    @test opts.shock_scale == 0.0
    @test opts.elb_shock_name == :eps_z
    @test opts.elb_span == 5
    @test opts.grid_size == 9
    @test opts.train_thetas == 9
    @test all(shocks[eps_z_idx, forced_window] .== opts.elb_shock)
    @test all(shocks[eps_nu_idx, :] .== 0.0)
    @test issorted(grid)
    @test any(isapprox.(grid, FLOOR_THETA_TRUE; rtol = 1e-12, atol = 1e-14))

    if get(ENV, "RUN_GALI_ACTUAL_FLOOR_GRID_SMOKE", "0") == "1"
        out_dir = mktempdir()
        result = run_floor_grid_validation(FloorGridOptions(out_dir = out_dir, run_id = "test_actual_floor_grid"))
        @test result["core_clean_solver"]
        @test result["fully_clean_solver"]
        @test result["interval_overlap"]
        @test result["direct_cover"]
        @test result["surrogate_cover"]
        @test isfile(joinpath(out_dir, "test_actual_floor_grid", "SUMMARY.md"))
    else
        @test true
    end
end
