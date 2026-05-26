using Test

include(joinpath(@__DIR__, "..", "scripts", "gali_obc_actual_floor_twoparam_inversion_grid_validation.jl"))

@testset "Gali actual-floor two-parameter inversion-grid validation setup" begin
    opts = TwoParamOptions(
        periods = 24,
        z_ident_period = 12,
        z_ident_span = 6,
        z_ident_shock = 1.0,
        z_ident_pattern = "alternating",
        policy_shock_period = 18,
        policy_shock_span = 6,
        policy_shock = 1.0,
        policy_shock_pattern = "alternating",
        second_std_name = :std_a,
        second_shock_name = :eps_a,
        second_baseline = 0.01,
        second_true = 0.01,
    )
    model = Gali_2015_chapter_3_obc
    shocks = base_two_param_shocks(model, opts)

    eps_z_idx = find_idx(model.exo, :eps_z)
    eps_a_idx = find_idx(model.exo, :eps_a)
    eps_nu_idx = find_idx(model.exo, :eps_nu)
    forced_window = opts.elb_period:min(opts.periods, opts.elb_period + opts.elb_span - 1)
    z_ident_window = opts.z_ident_period:min(opts.periods, opts.z_ident_period + opts.z_ident_span - 1)
    second_window = opts.policy_shock_period:min(opts.periods, opts.policy_shock_period + opts.policy_shock_span - 1)

    @test theta_names(opts) == [:std_z, :std_a]
    @test theta_baseline(opts) == [0.05, 0.01]
    @test theta_true(opts) == [0.05, 0.01]
    @test isfinite(log_prior(theta_true(opts), opts))
    @test all(shocks[eps_z_idx, forced_window] .== opts.elb_shock)
    @test shocks[eps_z_idx, z_ident_window] == [isodd(j) ? opts.z_ident_shock : -opts.z_ident_shock for j in 1:length(z_ident_window)]
    @test shocks[eps_a_idx, second_window] == [isodd(j) ? opts.policy_shock : -opts.policy_shock for j in 1:length(second_window)]
    @test all(shocks[eps_nu_idx, :] .== 0.0)
    @test issorted(axis_grid(1, opts.axis_size, opts))
    @test issorted(axis_grid(2, opts.axis_size, opts))

    if get(ENV, "RUN_GALI_TWOPARAM_INVERSION_GRID_SMOKE", "0") == "1"
        out_dir = mktempdir()
        result = run_twoparam_validation(
            TwoParamOptions(
                periods = 24,
                axis_size = 3,
                train_axis_size = 3,
                z_ident_period = 12,
                z_ident_span = 6,
                z_ident_shock = 1.0,
                z_ident_pattern = "alternating",
                policy_shock_period = 18,
                policy_shock_span = 6,
                policy_shock = 1.0,
                policy_shock_pattern = "alternating",
                second_std_name = :std_a,
                second_shock_name = :eps_a,
                second_baseline = 0.01,
                second_true = 0.01,
                out_dir = out_dir,
                run_id = "test_twoparam_inversion_grid",
            ),
        )
        @test result["validation_pass"]
        @test result["finite_direct_grid_points"] == size(result["theta_points"], 2)
        @test result["finite_surrogate_grid_points"] == size(result["theta_points"], 2)
        @test result["recovery_failures"] == 0
        @test result["solver_warning_count_total"] == 0
        @test isfile(joinpath(out_dir, "test_twoparam_inversion_grid", "SUMMARY.md"))
    else
        @test true
    end
end
