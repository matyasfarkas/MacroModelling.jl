#!/usr/bin/env julia
# ============================================================================
# SHOCK-SCALE COVERAGE ANALYSIS AND FIGURE GENERATION
# ============================================================================
#
# Reads results from shock_scale_grid_simulation.jl and produces:
# 1. Convergence heatmap (shock_scale × theta index → convergence rate)
# 2. Convergence rate curve with uncertainty bands
# 3. Shock coverage plot (training envelope vs smoothed real-data shocks)
# 4. Residual quality vs shock_scale (box plot)
#
# Usage:
#   julia --project=. scripts/analyze_shock_scale_coverage.jl
#
# ============================================================================

using Serialization
import Statistics: mean, median, std, quantile, var
using LinearAlgebra
using Printf
using Dates
using DataFrames
using MacroModelling
using AxisKeys
using Plots
using Plots.PlotMeasures

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

# ============================================================================
# CLI Arguments
# ============================================================================

function script_repo_root()
    return normpath(joinpath(@__DIR__, ".."))
end

grid_dir   = parse_arg_string(ARGS, "--grid-dir",
                 joinpath(script_repo_root(), ".local_artifacts", "shock_scale_grid"))
chain_path = parse_arg_string(ARGS, "--chain", "")
data_path  = parse_arg_string(ARGS, "--data", "")
fig_dir    = parse_arg_string(ARGS, "--fig-dir",
                 joinpath(script_repo_root(), "docs", "paper", "Figures"))
fig_prefix = parse_arg_string(ARGS, "--fig-prefix", "shock_scale")

println("=" ^ 78)
println("SHOCK-SCALE COVERAGE ANALYSIS")
println("Started: $(now())")
println("=" ^ 78)

# ============================================================================
# Step 1: Load Grid Results
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 1: Loading grid simulation results")
println("-" ^ 78)

p1_path = joinpath(grid_dir, "convergence_grid.jls")
if !isfile(p1_path)
    error("Phase 1 results not found at: $(p1_path)\nRun shock_scale_grid_simulation.jl first.")
end

p1_data = Serialization.deserialize(p1_path)

convergence_matrix  = p1_data["convergence_matrix"]
residual_matrix     = p1_data["residual_matrix"]
max_residual_matrix = p1_data["max_residual_matrix"]
zlb_matrix          = p1_data["zlb_matrix"]
shock_scales        = p1_data["shock_scales"]
theta_grid          = p1_data["theta_grid"]
theta_names_grid    = p1_data["theta_names"]
detail_records      = p1_data["detail_records"]
settings            = p1_data["settings"]

n_scales = length(shock_scales)
n_thetas = size(theta_grid, 1)
shock_scale_max = get(p1_data, "shock_scale_max", NaN)
all_scales_and_rates = get(p1_data, "all_scales_and_rates", Dict{Float64,Float64}())

# Load Phase 2 refinement if available
p2_path = joinpath(grid_dir, "frontier_refinement.jls")
has_refinement = isfile(p2_path)
if has_refinement
    p2_data = Serialization.deserialize(p2_path)
    shock_scale_max = p2_data["shock_scale_max"]
    all_scales_and_rates = p2_data["all_scales_and_rates"]
    refinement_scales = p2_data["refinement_scales"]
    println("  Phase 2 refinement loaded. shock_scale_max = $(shock_scale_max)")
end

println("  Grid: $(n_scales) scales × $(n_thetas) thetas")
println("  Shock scales: $(shock_scales)")
println("  shock_scale_max: $(shock_scale_max)")

# ============================================================================
# Step 2: Convergence Statistics Summary
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 2: Convergence statistics")
println("-" ^ 78)

println("\n  Per-scale convergence rates:")
println("  " * "-" ^ 60)
@printf("  %8s  %6s  %6s  %6s  %6s  %6s  %5s\n",
        "Scale", "Mean", "P10", "P50", "P90", "Min", "ZLB%")
println("  " * "-" ^ 60)
for si in 1:n_scales
    conv_rates = filter(!isnan, convergence_matrix[si, :])
    if isempty(conv_rates)
        continue
    end
    m   = mean(conv_rates)
    p10 = quantile(conv_rates, 0.10)
    p50 = quantile(conv_rates, 0.50)
    p90 = quantile(conv_rates, 0.90)
    mn  = minimum(conv_rates)
    zf  = 100 * sum(zlb_matrix[si, :]) / n_thetas
    @printf("  %8.2f  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%  %4.1f%%\n",
            shock_scales[si], 100*m, 100*p10, 100*p50, 100*p90, 100*mn, zf)
end
println("  " * "-" ^ 60)

# Residual summary
println("\n  Per-scale SEP residual summary (median across thetas):")
println("  " * "-" ^ 44)
@printf("  %8s  %10s  %10s  %10s\n", "Scale", "P25", "Median", "P75")
println("  " * "-" ^ 44)
for si in 1:n_scales
    resids = filter(!isnan, residual_matrix[si, :])
    if isempty(resids)
        continue
    end
    @printf("  %8.2f  %10.2e  %10.2e  %10.2e\n",
            shock_scales[si], quantile(resids, 0.25), median(resids), quantile(resids, 0.75))
end
println("  " * "-" ^ 44)

# ============================================================================
# Step 3: Recover Smoothed Shocks from Real Data
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 3: Recovering smoothed shocks from real data")
println("-" ^ 78)

# Find data payload
if data_path == ""
    artifacts_dir = joinpath(script_repo_root(), ".local_artifacts", "hlt_18param_realdata")
    for candidate in [
        "hlt_real_data_payload.jls",
        "hlt_real_data_payload_extended_18p.jls",
    ]
        p = joinpath(artifacts_dir, candidate)
        if isfile(p)
            global data_path = p
            break
        end
    end
end

# Find chain for posterior mean theta
if chain_path == ""
    artifacts_dir = joinpath(script_repo_root(), ".local_artifacts", "hlt_18param_realdata")
    for candidate in [
        "hlt_linear_hmc_chain_2000_seed99.jls",
        "hlt_linear_hmc_chain_1000.jls",
        "hlt_kalman_mh_chain_50k.jls",
    ]
        p = joinpath(artifacts_dir, candidate)
        if isfile(p)
            global chain_path = p
            break
        end
    end
end

smoothed_shocks = nothing
shock_labels = Symbol[]

if data_path != "" && chain_path != ""
    println("  Data: $(basename(data_path))")
    println("  Chain: $(basename(chain_path))")

    # Load model (linear, for inversion filter)
    model_linear = load_hlt_model(script_repo_root(), "Smets_Wouters_2007_HLT"; mod = @__MODULE__)

    # Load data
    payload = Serialization.deserialize(data_path)
    payload = payload isa Dict ? payload : Dict(pairs(payload))
    obs_data = payload["obs_data"]
    observables = Symbol.(payload["observables"])
    theta_names_data = Symbol.(payload["theta_names"])

    # Load chain and get posterior mean
    chain_data = Serialization.deserialize(chain_path)
    chain_data = chain_data isa Dict ? chain_data : Dict(pairs(chain_data))
    chain_mat = chain_data["chain"]
    chain_tnames = Symbol.(chain_data["theta_names"])
    theta_post_mean = vec(mean(chain_mat, dims=1))

    # Build parameter vector at posterior mean
    base_params_linear = copy(model_linear.parameter_values)
    theta_param_idx = Int.(indexin(chain_tnames, model_linear.parameters))
    full_params = copy(base_params_linear)
    full_params[theta_param_idx] = theta_post_mean
    MacroModelling.write_parameters_input!(model_linear, full_params, verbose = false)

    # Run first-order inversion filter to recover smoothed shocks
    println("  Running first-order inversion filter...")

    d_obs, T_obs = size(obs_data)
    obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)

    # Get first-order state-space matrices
    sol = MacroModelling.get_solution(model_linear, full_params; algorithm = :first_order)
    # get_solution(model, params) returns (ss_vec, state_space_matrix, converged)
    ss_vec, SS_mat, sol_converged = sol
    SS = Float64.(SS_mat)

    # Manual inversion filter to extract smoothed shocks
    T_timings = model_linear.timings
    n_exo = T_timings.nExo
    n_past = T_timings.nPast_not_future_and_mixed

    cond_var_idx = indexin(observables,
                           sort(union(T_timings.aux, T_timings.var, T_timings.exo_present)))

    # Initial state at steady state (zeros in deviation form)
    state = zeros(size(SS, 1))

    # State-space decomposition
    S_obs = SS[cond_var_idx, 1:n_past]
    S_shock = SS[cond_var_idx, end-n_exo+1:end]
    inv_S_shock = try
        inv(S_shock)
    catch
        pinv(S_shock)
    end

    # Recover shocks period by period
    smoothed_shock_matrix = zeros(n_exo, T_obs)
    data_dev = Float64.(obs_data)

    for t in 1:T_obs
        y_pred = S_obs * state[T_timings.past_not_future_and_mixed_idx]
        innovation = data_dev[:, t] - y_pred
        eps_t = inv_S_shock * innovation
        smoothed_shock_matrix[:, t] = eps_t
        global state = SS * vcat(state[T_timings.past_not_future_and_mixed_idx], eps_t)
    end

    # Label the shocks
    shock_names_all = model_linear.exo
    obc_mask_linear = contains.(string.(shock_names_all), "ᵒᵇᶜ")
    structural_idx_linear = findall(!, obc_mask_linear)

    smoothed_shocks = smoothed_shock_matrix
    shock_labels = shock_names_all

    println("  Recovered $(n_exo) × $(T_obs) shock matrix")
    println("  Max |shock|: $(round(maximum(abs.(smoothed_shock_matrix)), digits=2))")

    # Summary per shock
    println("\n  Smoothed shock summary (|standardized|):")
    println("  " * "-" ^ 50)
    @printf("  %12s  %8s  %8s  %8s  %8s\n", "Shock", "Mean", "P95", "P99", "Max")
    println("  " * "-" ^ 50)

    # Get shock standard deviations from model parameters
    for (i, sname) in enumerate(shock_names_all)
        if obc_mask_linear[i]
            continue
        end
        abs_shocks = abs.(smoothed_shock_matrix[i, :])
        sigma_k = MacroModelling.sep_irf_shock_std(model_linear, sname)
        standardized = abs_shocks ./ max(sigma_k, 1e-12)
        @printf("  %12s  %8.2f  %8.2f  %8.2f  %8.2f  (σ=%.4f)\n",
                sname,
                mean(standardized),
                quantile(standardized, 0.95),
                quantile(standardized, 0.99),
                maximum(standardized),
                sigma_k)
    end
    println("  " * "-" ^ 50)
else
    println("  Skipping shock recovery (data or chain not found)")
    if data_path == ""
        println("  Missing: real-data payload (provide --data=<path>)")
    end
    if chain_path == ""
        println("  Missing: posterior chain (provide --chain=<path>)")
    end
end

# ============================================================================
# Step 4: Coverage Analysis
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 4: Shock coverage analysis")
println("-" ^ 78)

if smoothed_shocks !== nothing
    model_obc = load_hlt_model(script_repo_root(), "Smets_Wouters_2007_HLT_obc"; mod = @__MODULE__)
    shock_names_obc = model_obc.exo
    obc_mask_obc = contains.(string.(shock_names_obc), "ᵒᵇᶜ")
    structural_idx_obc = findall(!, obc_mask_obc)

    println("\n  Coverage at each shock_scale:")
    println("  (fraction of smoothed shocks within training ±scale×σ envelope)")
    println("  " * "-" ^ 60)
    @printf("  %8s", "Scale")
    for idx in structural_idx_obc
        @printf("  %8s", shock_names_obc[idx])
    end
    @printf("  %8s\n", "Joint")
    println("  " * "-" ^ 60)

    # For each shock_scale, the training data contains shocks drawn as N(0, (scale*σ_k)^2).
    # The 99th percentile of |N(0,1)| is about 2.576.
    # At scale s, the max shock magnitude expected in training is roughly s * σ_k * q99_factor.
    # For 5 trajectories × sim_periods periods per theta, the effective max is larger.
    # Use a conservative coverage criterion: |ε̂_t,k| < scale * σ_k * 3.0 (conservative bound)
    coverage_factor = 3.0  # ~99.7% of N(0,1)

    # Map structural indices between linear and OBC model
    model_linear_loaded = isdefined(@__MODULE__, :model_linear) ? model_linear : load_hlt_model(script_repo_root(), "Smets_Wouters_2007_HLT"; mod = @__MODULE__)

    coverage_by_scale = Dict{Float64, Dict{Symbol, Float64}}()

    for scale in shock_scales
        coverage_dict = Dict{Symbol, Float64}()
        all_covered = trues(size(smoothed_shocks, 2))

        for (si, idx) in enumerate(structural_idx_obc)
            sname = shock_names_obc[idx]
            sigma_k = MacroModelling.sep_irf_shock_std(model_obc, sname)

            # Find corresponding shock in linear model
            linear_idx = findfirst(==(sname), shock_labels)
            if linear_idx === nothing
                coverage_dict[sname] = NaN
                continue
            end

            threshold = scale * sigma_k * coverage_factor
            abs_smoothed = abs.(smoothed_shocks[linear_idx, :])
            covered = abs_smoothed .< threshold
            coverage_dict[sname] = mean(covered)
            all_covered .&= covered
        end

        coverage_dict[:joint] = mean(all_covered)
        coverage_by_scale[scale] = coverage_dict

        @printf("  %8.2f", scale)
        for idx in structural_idx_obc
            sname = shock_names_obc[idx]
            @printf("  %7.1f%%", 100 * get(coverage_dict, sname, NaN))
        end
        @printf("  %7.1f%%\n", 100 * coverage_dict[:joint])
    end
    println("  " * "-" ^ 60)

    # Report shock_scale needed for 95% and 99% joint coverage
    for target in [0.95, 0.99]
        scale_needed = NaN
        for scale in sort(shock_scales)
            if haskey(coverage_by_scale, scale) && coverage_by_scale[scale][:joint] >= target
                scale_needed = scale
                break
            end
        end
        @printf("\n  Minimum scale for %.0f%% joint coverage: %.2f\n",
                100*target, scale_needed)
    end
else
    println("  Skipping coverage analysis (no smoothed shocks available)")
end

# ============================================================================
# Step 5: Generate Figures
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 5: Generating publication figures")
println("-" ^ 78)

begin
    mkpath(fig_dir)
    gr()
        # ------------------------------------------------------------------
        # Figure 1: Convergence Heatmap
        # ------------------------------------------------------------------
        println("  Generating convergence heatmap...")

        # Sort thetas by posterior density (approximate: use distance from mean)
        theta_means = vec(mean(theta_grid, dims=1))
        theta_stds = vec(std(theta_grid, dims=1))
        theta_distances = [sqrt(mean(((theta_grid[ti, :] .- theta_means) ./ max.(theta_stds, 1e-12)).^2))
                          for ti in 1:n_thetas]
        sort_order = sortperm(theta_distances)

        sorted_conv = convergence_matrix[:, sort_order]'  # (n_thetas, n_scales) for heatmap

        p1 = heatmap(
            shock_scales, 1:n_thetas, sorted_conv,
            xlabel = "Shock scale",
            ylabel = "Theta index (sorted by posterior distance)",
            title = "SEP Convergence Rate",
            color = :RdYlGn,
            clims = (0.0, 1.0),
            colorbar_title = "Convergence rate",
            size = (700, 500),
            dpi = 300,
            margin = 5mm,
        )

        savefig(p1, joinpath(fig_dir, "$(fig_prefix)_convergence_heatmap.pdf"))
        println("    Saved: $(fig_prefix)_convergence_heatmap.pdf")

        # ------------------------------------------------------------------
        # Figure 2: Convergence Rate Curve
        # ------------------------------------------------------------------
        println("  Generating convergence rate curve...")

        mean_rates = [mean(filter(!isnan, convergence_matrix[si, :])) for si in 1:n_scales]
        p10_rates  = [quantile(filter(!isnan, convergence_matrix[si, :]), 0.10) for si in 1:n_scales]
        p90_rates  = [quantile(filter(!isnan, convergence_matrix[si, :]), 0.90) for si in 1:n_scales]

        p2 = plot(
            shock_scales, 100 .* mean_rates,
            ribbon = (100 .* (mean_rates .- p10_rates), 100 .* (p90_rates .- mean_rates)),
            fillalpha = 0.3,
            xlabel = "Shock scale",
            ylabel = "Convergence rate (%)",
            title = "SEP Convergence vs Shock Scale",
            label = "Mean (10th-90th pct band)",
            linewidth = 2,
            marker = :circle,
            markersize = 4,
            size = (600, 400),
            dpi = 300,
            margin = 5mm,
            ylims = (0, 105),
            legend = :bottomleft,
        )

        # Add threshold line
        conv_threshold_pct = get(settings, "convergence_threshold", 0.80) * 100
        hline!(p2, [conv_threshold_pct],
               linestyle = :dash, color = :red, linewidth = 1.5,
               label = "$(round(Int, conv_threshold_pct))% threshold")

        # Mark shock_scale_max
        if !isnan(shock_scale_max)
            vline!(p2, [shock_scale_max],
                   linestyle = :dot, color = :blue, linewidth = 1.5,
                   label = "scale_max = $(round(shock_scale_max, digits=2))")
        end

        savefig(p2, joinpath(fig_dir, "$(fig_prefix)_convergence_curve.pdf"))
        println("    Saved: $(fig_prefix)_convergence_curve.pdf")

        # ------------------------------------------------------------------
        # Figure 3: Shock Coverage Plot
        # ------------------------------------------------------------------
        if smoothed_shocks !== nothing
            println("  Generating shock coverage plot...")

            n_structural = length(structural_idx_obc)
            p3 = plot(
                layout = (1, 1),
                size = (700, 400),
                dpi = 300,
                margin = 5mm,
            )

            # Joint coverage vs shock_scale
            joint_cov = [get(get(coverage_by_scale, s, Dict()), :joint, NaN)
                        for s in shock_scales]

            plot!(p3, shock_scales, 100 .* joint_cov,
                  xlabel = "Shock scale",
                  ylabel = "Joint coverage (%)",
                  title = "Training Data Coverage of Smoothed Real-Data Shocks",
                  label = "Joint coverage",
                  linewidth = 2,
                  marker = :circle,
                  markersize = 4,
                  legend = :bottomright,
                  ylims = (0, 105),
            )

            # Per-shock coverage lines
            for (si, idx) in enumerate(structural_idx_obc)
                sname = shock_names_obc[idx]
                per_shock_cov = [get(get(coverage_by_scale, s, Dict()), sname, NaN)
                                for s in shock_scales]
                plot!(p3, shock_scales, 100 .* per_shock_cov,
                      label = string(sname),
                      linewidth = 1,
                      linestyle = :dash,
                      alpha = 0.7,
                )
            end

            # Add reference lines
            hline!(p3, [95.0], linestyle=:dot, color=:gray, linewidth=1, label="95% target")
            if !isnan(shock_scale_max)
                vline!(p3, [shock_scale_max], linestyle=:dot, color=:blue, linewidth=1.5, label="scale_max")
            end

            savefig(p3, joinpath(fig_dir, "$(fig_prefix)_coverage.pdf"))
            println("    Saved: $(fig_prefix)_coverage.pdf")
        else
            println("  Skipping coverage plot (no smoothed shocks)")
        end

        # ------------------------------------------------------------------
        # Figure 4: Residual Quality vs Shock Scale
        # ------------------------------------------------------------------
        println("  Generating residual quality plot...")

        # Collect residuals from detail_records
        p4 = plot(
            xlabel = "Shock scale",
            ylabel = "Max SEP residual (log scale)",
            title = "SEP Solution Quality vs Shock Scale",
            size = (600, 400),
            dpi = 300,
            margin = 5mm,
            yscale = :log10,
            legend = :topleft,
        )

        for si in 1:n_scales
            resids = filter(!isnan, max_residual_matrix[si, :])
            if isempty(resids)
                continue
            end
            # Violin-style: show quantiles as error bars
            med = median(resids)
            q25 = quantile(resids, 0.25)
            q75 = quantile(resids, 0.75)

            scatter!(p4, [shock_scales[si]], [med],
                     yerror = ([med - q25], [q75 - med]),
                     color = :steelblue,
                     markersize = 5,
                     label = si == 1 ? "Median (IQR)" : "",
            )
        end

        # Add accept_tol line
        hline!(p4, [get(settings, "sep_accept_tol", 0.35)],
               linestyle = :dash, color = :red, linewidth = 1.5,
               label = "accept_tol = $(get(settings, "sep_accept_tol", 0.35))")

        savefig(p4, joinpath(fig_dir, "$(fig_prefix)_residual_quality.pdf"))
        println("    Saved: $(fig_prefix)_residual_quality.pdf")
end

# ============================================================================
# Step 6: LaTeX Table Output
# ============================================================================

println("\n" * "-" ^ 78)
println("STEP 6: LaTeX table for paper")
println("-" ^ 78)

println("\n% --- Paste into paper ---")
println("\\begin{table}[htbp]")
println("\\centering")
println("\\caption{SEP Convergence Rate by Shock Scale}")
println("\\label{tab:shock_scale_convergence}")
println("\\begin{tabular}{l" * repeat("c", 5) * "}")
println("\\hline\\hline")
println("Shock scale & Mean & P10 & P90 & Min & ZLB\\% \\\\")
println("\\hline")
for si in 1:n_scales
    conv_rates = filter(!isnan, convergence_matrix[si, :])
    if isempty(conv_rates)
        continue
    end
    m   = mean(conv_rates)
    p10 = quantile(conv_rates, 0.10)
    p90 = quantile(conv_rates, 0.90)
    mn  = minimum(conv_rates)
    zf  = sum(zlb_matrix[si, :]) / n_thetas
    @printf("%.2f & %.1f\\%% & %.1f\\%% & %.1f\\%% & %.1f\\%% & %.1f\\%% \\\\\n",
            shock_scales[si], 100*m, 100*p10, 100*p90, 100*mn, 100*zf)
end
println("\\hline\\hline")
println("\\end{tabular}")
if !isnan(shock_scale_max)
    println("\\begin{tablenotes}")
    println("\\small")
    ct = get(settings, "convergence_threshold", 0.80) * 100
    println("\\item Maximum reliable shock scale: $(round(shock_scale_max, digits=2)) (convergence \\geq $(round(Int, ct))\\%)")
    println("\\end{tablenotes}")
end
println("\\end{table}")
println("% --- End table ---")

# ============================================================================
# Done
# ============================================================================

println("\n" * "=" ^ 78)
println("ANALYSIS COMPLETE")
println("=" ^ 78)
if !isnan(shock_scale_max)
    println("  shock_scale_max = $(shock_scale_max)")
end
println("  Figures saved to: $(fig_dir)")
println("  Done: $(now())")
