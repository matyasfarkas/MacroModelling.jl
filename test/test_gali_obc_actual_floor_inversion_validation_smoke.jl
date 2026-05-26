using Test

include(joinpath(@__DIR__, "..", "scripts", "gali_obc_actual_floor_inversion_grid_validation.jl"))

@testset "Gali actual-floor inversion-grid validation setup" begin
    opts = InversionGridOptions()
    floor_opts = floor_options(opts)
    model = Gali_2015_chapter_3_obc
    shocks = effective_shock_matrix(model, floor_opts, FLOOR_THETA_TRUE)
    grid = theta_grid(opts.grid_size)

    eps_z_idx = find_idx(model.exo, :eps_z)
    eps_nu_idx = find_idx(model.exo, :eps_nu)
    forced_window = opts.elb_period:min(opts.periods, opts.elb_period + opts.elb_span - 1)

    @test FLOOR_THETA_NAME == :std_z
    @test opts.shock_scale == 0.0
    @test opts.elb_shock_name == :eps_z
    @test opts.elb_span == 5
    @test opts.grid_size == 17
    @test opts.train_thetas == 9
    @test opts.inv_maxit >= 10
    @test all(shocks[eps_z_idx, forced_window] .== opts.elb_shock)
    @test all(shocks[eps_nu_idx, :] .== 0.0)
    @test issorted(grid)
    @test isfinite(inversion_log_prior(FLOOR_PRIOR_LOWER))
    @test isfinite(inversion_log_prior(FLOOR_PRIOR_UPPER + 8 * eps(Float64)))

    if get(ENV, "RUN_GALI_ACTUAL_FLOOR_INVERSION_GRID_SMOKE", "0") == "1"
        out_dir = mktempdir()
        result = run_inversion_grid_validation(
            InversionGridOptions(
                periods = 12,
                grid_size = 5,
                train_thetas = 5,
                out_dir = out_dir,
                run_id = "test_actual_floor_inversion_grid",
            ),
        )
        @test result["finite_direct_grid_points"] == length(result["grid"])
        @test result["finite_surrogate_grid_points"] == length(result["grid"])
        @test result["recovery_failures"] == 0
        @test result["solver_warning_count_total"] == 0
        @test result["interval_overlap"]
        @test result["direct_cover"]
        @test result["surrogate_cover"]
        @test isfile(joinpath(out_dir, "test_actual_floor_inversion_grid", "SUMMARY.md"))
    else
        @test true
    end
end
