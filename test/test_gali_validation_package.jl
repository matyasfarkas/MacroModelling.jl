using Test

include(joinpath(@__DIR__, "..", "scripts", "gali_validation_package.jl"))

@testset "Gali validation package artifact audit" begin
    required_artifacts = [
        repo_path(".local_artifacts", "gali_elb_stochastic", "gali_obc_eps_z_same_shocks_actualfloor_span5_shock0p8_bg0p0_summary.md"),
        repo_path(".local_artifacts", "gali_actual_floor_residual_grid", "actual_floor_grid_stdz_interp_default_serial_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_inversion_grid", "actual_floor_inversion_grid_default_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_inversion_grid", "actual_floor_inversion_hmc_tuned_commonseed_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stda_balanced_T24_train3_probe_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_hmc_balanced_extended_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stdnu_probe_amp025_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stdnu_probe_amp05_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stdnu_probe_amp1_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_actual_floor_twoparam_inversion_grid", "twoparam_inversion_grid_stdnu_probe_amp2_20260518", "SUMMARY.md"),
        repo_path(".local_artifacts", "gali_direct_sep_surrogate_hmc", "direct_sep_full_pipeline_smoke_20260518", "SUMMARY.md"),
    ]
    artifacts_available = all(isfile, required_artifacts)
    force_audit = get(ENV, "RUN_GALI_VALIDATION_PACKAGE_AUDIT", "0") == "1"

    if artifacts_available || force_audit
        out_dir = mktempdir()
        opts = PackageOptions(
            run_id = "test_gali_validation_package",
            out_dir = out_dir,
            strict = true,
        )
        @test run_package(opts) == true
        report = joinpath(out_dir, "test_gali_validation_package", "VALIDATION_PACKAGE_REPORT.md")
        manifest = joinpath(out_dir, "test_gali_validation_package", "validation_manifest.toml")
        @test isfile(report)
        @test isfile(manifest)
        report_text = read(report, String)
        @test occursin("**Overall status**: PASS", report_text)
        @test occursin("twoparam_hmc_stdz_stda_extended", report_text)
        @test occursin("stdnu_identification_probe", report_text)
    else
        @info "Skipping Galí package artifact audit; set RUN_GALI_VALIDATION_PACKAGE_AUDIT=1 after regenerating .local_artifacts."
        @test true
    end
end
