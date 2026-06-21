#!/usr/bin/env julia
# ============================================================================
# HLT CURVATURE SURFACE
# ============================================================================
#
# Computes SEP-minus-ROM1 one-step forecast-error surfaces around a chosen HLT
# parameter center. The metric is the RMSE of observable forecast errors:
#
#   e_t(theta) = y_SEP,t+1 - y_ROM1,t+1
#
# evaluated at the same SEP state, same structural shocks, and same parameters.
# Large RMSE means the first-order forecast misses more of the local nonlinear
# transition. This is a visual companion to the local ablation tables.
#
# Usage:
#   julia --project=. scripts/hlt_posterior_mean_curvature_surface.jl --dry-run=true
#   julia --project=. scripts/hlt_posterior_mean_curvature_surface.jl --grid-size=5 --n-shock-paths=2
# ============================================================================

using MacroModelling
using Random
using Serialization
using Dates
using LinearAlgebra
using Printf
using Statistics
using Plots

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_rom_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_float_list(raw::String)
    vals = Float64[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        push!(vals, parse(Float64, s))
    end
    return vals
end

function parse_pair_list(raw::String)
    pairs = Tuple{Symbol, Symbol}[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        parts = split(s, ":"; limit=2)
        length(parts) == 2 || error("Malformed surface pair '$s'. Use par1:par2.")
        push!(pairs, (Symbol(strip(parts[1])), Symbol(strip(parts[2]))))
    end
    return pairs
end

function parse_param_overrides(raw::String)
    overrides = Dict{Symbol, Float64}()
    isempty(strip(raw)) && return overrides
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        parts = split(s, "="; limit=2)
        length(parts) == 2 || error("Malformed parameter override '$s'. Use name=value.")
        overrides[Symbol(strip(parts[1]))] = parse(Float64, strip(parts[2]))
    end
    return overrides
end

function parse_param_bounds(raw::String)
    bounds = Dict{Symbol, Tuple{Float64, Float64}}()
    isempty(strip(raw)) && return bounds
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        parts = split(s, "="; limit=2)
        length(parts) == 2 || error("Malformed parameter bounds '$s'. Use name=lower:upper.")
        vals = split(strip(parts[2]), ":"; limit=2)
        length(vals) == 2 || error("Malformed parameter bounds '$s'. Use name=lower:upper.")
        lower = parse(Float64, strip(vals[1]))
        upper = parse(Float64, strip(vals[2]))
        isfinite(lower) && isfinite(upper) && lower < upper ||
            error("Invalid bounds for $(strip(parts[1])): require finite lower < upper.")
        bounds[Symbol(strip(parts[1]))] = (lower, upper)
    end
    return bounds
end

const STD_RANGE_MODE = "std"
const MULTIPLIER_RANGE_MODE = "multiplier"

function format_param_overrides(overrides::Dict{Symbol, Float64})
    isempty(overrides) && return "none"
    pairs = sort(collect(overrides); by = p -> String(p[1]))
    return join(["$(p[1])=$(p[2])" for p in pairs], ", ")
end

function format_param_bounds(bounds::Dict{Symbol, Tuple{Float64, Float64}})
    isempty(bounds) && return "none"
    pairs = sort(collect(bounds); by = p -> String(p[1]))
    return join(["$(p[1])=$(p[2][1]):$(p[2][2])" for p in pairs], ", ")
end

function format_bound(x::Real)
    isfinite(x) || return ""
    return @sprintf("%.8g", x)
end

function find_param_index(model, pname::Symbol)
    idx = findfirst(==(pname), model.parameters)
    idx === nothing && error("Parameter '$pname' not found in $(model.model_name).")
    return idx
end

function find_var_index(model, vname::Symbol)
    idx = findfirst(==(vname), model.var)
    idx === nothing && error("Variable '$vname' not found in $(model.model_name).")
    return idx
end

function finite_simulation(res)
    res === nothing && return false
    try
        sim = Array(res.simulation)
        return !isempty(sim) && all(isfinite, sim)
    catch
        return false
    end
end

function max_sep_error(res)
    res === nothing && return Inf
    if hasproperty(res, :sep_errors)
        vals = Float64[]
        for e in res.sep_errors
            isfinite(e) && push!(vals, abs(Float64(e)))
        end
        !isempty(vals) && return maximum(vals)
    end
    return getproperty(res, :errorflag) ? Inf : 0.0
end

function sep_solution_accepted(res, accept_tol::Float64)
    finite_simulation(res) || return false
    return !res.errorflag || max_sep_error(res) <= accept_tol
end

function draw_shocks(rng::AbstractRNG, model, total_periods::Int, shock_scale::Float64)
    shock_names = model.exo
    shocks = zeros(length(shock_names), total_periods)
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    structural_idx = findall(!, obc_mask)
    isempty(structural_idx) && return shocks
    sigmas = ones(length(structural_idx))
    shocks[structural_idx, :] .= Diagonal(sigmas .* shock_scale) *
                                 randn(rng, length(structural_idx), total_periods)
    return shocks
end

function load_posterior_mean(chain_path::String, theta_names::Vector{Symbol})
    isfile(chain_path) || error("Posterior chain not found: $chain_path")
    payload = Serialization.deserialize(chain_path)
    chain = payload["chain"]
    chain_theta_names = haskey(payload, "theta_names") ? Symbol.(payload["theta_names"]) : theta_names
    if chain_theta_names != theta_names
        reindex = [findfirst(==(n), chain_theta_names) for n in theta_names]
        all(!isnothing, reindex) || error("Could not reindex chain theta names to parameter config.")
        chain = chain[:, reindex]
    end
    return vec(Statistics.mean(chain, dims=1)), size(chain, 1)
end

function load_posterior_stats(chain_path::String)
    stats = Dict{Symbol, NamedTuple}()
    isfile(chain_path) || return stats
    try
        payload = Serialization.deserialize(chain_path)
        haskey(payload, "chain") || return stats
        haskey(payload, "theta_names") || return stats
        chain = payload["chain"]
        names = Symbol.(payload["theta_names"])
        for (j, pname) in enumerate(names)
            col = Float64.(chain[:, j])
            finite_col = filter(isfinite, col)
            isempty(finite_col) && continue
            stats[pname] = (
                mean = Statistics.mean(finite_col),
                sd = length(finite_col) > 1 ? Statistics.std(finite_col) : NaN,
                q05 = Statistics.quantile(finite_col, 0.05),
                q95 = Statistics.quantile(finite_col, 0.95),
                n = length(finite_col),
            )
        end
    catch e
        @warn "Could not load posterior stats for density-scaled grid" chain_path exception=(e, catch_backtrace())
    end
    return stats
end

function prior_sd_from_spec(spec::ParameterSpec)
    params = spec.prior_params
    if spec.prior_type == :Normal && haskey(params, :σ)
        return Float64(params.σ)
    end
    return NaN
end

function unique_surface_parameters(pairs)
    params = Symbol[]
    for pair in pairs
        for pname in pair
            pname in params || push!(params, pname)
        end
    end
    return params
end

function build_density_scales(pairs, specs::Vector{ParameterSpec}, posterior_stats,
                              sd_overrides::Dict{Symbol, Float64},
                              bounds_overrides::Dict{Symbol, Tuple{Float64, Float64}},
                              base_params::Vector{Float64}, model;
                              fallback_rel_sd::Float64,
                              std_radius::Float64)
    spec_map = Dict(s.name => s for s in specs)
    scales = Dict{Symbol, NamedTuple}()
    for pname in unique_surface_parameters(pairs)
        pidx = find_param_index(model, pname)
        center = base_params[pidx]
        source = "fallback relative calibration scale"
        sd = abs(center) * fallback_rel_sd
        if haskey(sd_overrides, pname)
            sd = sd_overrides[pname]
            source = "manual prior/calibration sd override"
        elseif haskey(posterior_stats, pname) && isfinite(posterior_stats[pname].sd) && posterior_stats[pname].sd > 0
            sd = posterior_stats[pname].sd
            source = "posterior chain sd"
        elseif haskey(spec_map, pname)
            prior_sd = prior_sd_from_spec(spec_map[pname])
            if isfinite(prior_sd) && prior_sd > 0
                sd = prior_sd
                source = "parameter-config prior sd"
            end
        end
        isfinite(sd) && sd > 0 || error("Nonpositive density scale for parameter $pname")
        lower = NaN
        upper = NaN
        bounds_source = "none"
        if haskey(bounds_overrides, pname)
            lower, upper = bounds_overrides[pname]
            bounds_source = "manual density bounds override"
        elseif haskey(spec_map, pname)
            lower, upper = spec_map[pname].bounds
            bounds_source = "parameter-config bounds"
        end
        if !isfinite(lower) && center > 0 && center - std_radius * sd <= 0
            sd_cap = 0.95 * center / std_radius
            sd = min(sd, sd_cap)
            source = "$source; capped to keep grid positive"
        end
        if isfinite(lower) && center - std_radius * sd < lower
            source = "$source; truncated at lower density support"
        end
        if isfinite(upper) && center + std_radius * sd > upper
            source = "$source; truncated at upper density support"
        end
        scales[pname] = (center=center, sd=sd, lower=lower, upper=upper,
                         source=source, bounds_source=bounds_source)
    end
    return scales
end

function apply_estimated_theta!(params::Vector{Float64}, model, theta_names::Vector{Symbol}, theta::Vector{Float64})
    for (j, tname) in enumerate(theta_names)
        pidx = findfirst(==(tname), model.parameters)
        pidx !== nothing && (params[pidx] = theta[j])
    end
    return params
end

function run_sep(model, shocks, sim_periods::Int, burn_in::Int, seed::Int;
                 sep_horizon::Int, sep_maxit::Int, sep_tol::Float64,
                 sep_accept_tol::Float64, sep_fallback_solver::Union{Nothing, Symbol})
    local res
    try
        res = MacroModelling.simulate_sep_extended_path(
            model;
            periods          = sim_periods,
            burn_in          = burn_in,
            sep_horizon      = sep_horizon,
            sep_order        = 1,
            sep_nnodes       = 3,
            sep_maxit        = sep_maxit,
            sep_tol          = sep_tol,
            sep_sparse_tree  = true,
            sep_linear_solver = :normal_equations,
            sep_fallback_solver = sep_fallback_solver,
            sep_stall_iters  = 25,
            sep_stall_rel_tol = 1e-4,
            sep_stall_abs_tol = 1e-10,
            sep_line_search  = true,
            sep_line_search_maxit = 6,
            sep_line_search_factor = 0.5,
            sep_line_search_min_alpha = 1e-4,
            sep_lm_lambda    = 1e-8,
            sep_lm_lambda_scale = 10.0,
            sep_lm_lambda_min = 1e-12,
            sep_lm_lambda_max = 1e4,
            sep_shock_scale  = 1.0,
            sep_accept_tol   = sep_accept_tol,
            shock_scaling    = :none,
            shocks           = shocks,
            random_seed      = seed,
            silent           = true,
        )
    catch e
        if e isa LinearAlgebra.SingularException || e isa LinearAlgebra.PosDefException ||
           e isa DomainError || e isa ArgumentError
            return (accepted=false, res=nothing, max_error=Inf, status=string(typeof(e)))
        end
        rethrow()
    end
    accepted = sep_solution_accepted(res, sep_accept_tol)
    return (accepted=accepted, res=res, max_error=max_sep_error(res),
            status=accepted ? "accepted" : "rejected")
end

function cell_forecast_rmse(model, params::Vector{Float64}, obs_idx::Vector{Int};
                            shock_scale::Float64, n_shock_paths::Int,
                            sim_periods::Int, burn_in::Int, seed0::Int,
                            sep_horizon::Int, sep_maxit::Int, sep_tol::Float64,
                            sep_accept_tol::Float64, sep_fallback_solver::Union{Nothing, Symbol})
    local rom_cache
    try
        rom_cache = build_rom_cache(model, 1; params=params, use_obc=true)
    catch e
        return (rmse=NaN, mse=NaN, n_samples=0, accepted_paths=0,
                attempted_paths=n_shock_paths, max_error=Inf, status="rom_failed: $(typeof(e))")
    end

    try
        MacroModelling.write_parameters_input!(model, params, verbose=false)
        MacroModelling.solve!(model; algorithm=:first_order, dynamics=true, obc=true, silent=true)
    catch e
        return (rmse=NaN, mse=NaN, n_samples=0, accepted_paths=0,
                attempted_paths=n_shock_paths, max_error=Inf, status="first_order_failed: $(typeof(e))")
    end

    sse = 0.0
    n = 0
    accepted_paths = 0
    max_errors = Float64[]

    for path_id in 1:n_shock_paths
        seed = seed0 + path_id
        shocks = draw_shocks(MersenneTwister(seed), model, sim_periods + burn_in, shock_scale)

        sep = run_sep(model, shocks, sim_periods, burn_in, seed;
                      sep_horizon=sep_horizon, sep_maxit=sep_maxit,
                      sep_tol=sep_tol, sep_accept_tol=sep_accept_tol,
                      sep_fallback_solver=sep_fallback_solver)
        push!(max_errors, sep.max_error)
        sep.accepted || continue
        accepted_paths += 1

        sim = Array(sep.res.simulation)
        sim_shocks = sep.res.shocks
        T_avail = min(sim_periods, size(sim, 2) - 1)
        for t in 1:T_avail
            local rom_next
            try
                rom_next = rom_step_full(rom_cache, sim[:, t], sim_shocks[:, t])
            catch
                continue
            end
            delta = sim[obs_idx, t + 1] .- rom_next[obs_idx]
            sse += sum(delta .^ 2)
            n += length(obs_idx)
        end
    end

    mse = n > 0 ? sse / n : NaN
    rmse = isfinite(mse) && mse >= 0 ? sqrt(mse) : NaN
    finite_errors = filter(isfinite, max_errors)
    max_error = isempty(finite_errors) ? Inf : maximum(finite_errors)
    status = accepted_paths == n_shock_paths ? "ok" :
             accepted_paths > 0 ? "partial" : "failed"
    return (rmse=rmse, mse=mse, n_samples=n, accepted_paths=accepted_paths,
            attempted_paths=n_shock_paths, max_error=max_error, status=status)
end

function matrix_from_records(records, pair_index::Int, nx::Int, ny::Int, key::String)
    z = fill(NaN, ny, nx)
    for rec in records
        rec["pair_index"] == pair_index || continue
        z[rec["iy"], rec["ix"]] = rec[key]
    end
    return z
end

function fill_isolated_missing_for_plot(z::AbstractMatrix{<:Real})
    out = Array{Float64}(z)
    ny, nx = size(out)
    for iy in 1:ny, ix in 1:nx
        isfinite(out[iy, ix]) && continue
        vals = Float64[]
        for jy in max(1, iy - 1):min(ny, iy + 1), jx in max(1, ix - 1):min(nx, ix + 1)
            jy == iy && jx == ix && continue
            v = out[jy, jx]
            isfinite(v) && push!(vals, v)
        end
        length(vals) >= 3 && (out[iy, ix] = Statistics.mean(vals))
    end
    return out
end

function pretty_parameter_name(p::Symbol)
    labels = Dict(
        :csadjcost => "investment adjustment cost",
        :czcap => "capital utilization",
        :curvp => "price Kimball",
        :curvw => "wage Kimball",
    )
    return get(labels, p, String(p))
end

function short_parameter_name(p::Symbol)
    labels = Dict(
        :csadjcost => "Adj. cost",
        :czcap => "Utilization",
        :curvp => "Price Kimball",
        :curvw => "Wage Kimball",
    )
    return get(labels, p, String(p))
end

function center_axis_value(range_mode::String)
    range_mode == STD_RANGE_MODE ? 0.0 : 1.0
end

function axis_unit_label(range_mode::String)
    range_mode == STD_RANGE_MODE ? "std. dev. from center" : "multiplier"
end

function parameter_axis_label(p::Symbol, range_mode::String)
    "$(pretty_parameter_name(p)) $(axis_unit_label(range_mode))"
end

function pretty_tick_label(x::Real)
    abs(x) < 1e-10 && return "0"
    return @sprintf("%.3g", x)
end

function axis_ticks(range_mode::String, grid_values::Vector{Float64})
    if range_mode == STD_RANGE_MODE
        lo = minimum(grid_values)
        hi = maximum(grid_values)
        radius = max(abs(lo), abs(hi))
        ticks = radius <= 1.01 ? [-1.0, -0.5, 0.0, 0.5, 1.0] :
                radius <= 2.01 ? [-2.0, -1.0, 0.0, 1.0, 2.0] :
                collect(range(lo, hi; length=5))
        ticks = [t for t in ticks if t >= lo - 1e-8 && t <= hi + 1e-8]
        return (ticks, pretty_tick_label.(ticks))
    end
    ticks = length(grid_values) <= 9 ? grid_values : grid_values[1:2:end]
    return (ticks, pretty_tick_label.(ticks))
end

function display_text(raw::String)
    text = replace(raw, "_" => " ")
    text = replace(text, " pm1sd" => " (+/- 1 s.d.)")
    text = replace(text, " pm2sd" => " (+/- 2 s.d.)")
    return text
end

function grid_value_to_parameter(center::Float64, grid_value::Float64,
                                 pname::Symbol, range_mode::String, density_scales)
    if range_mode == STD_RANGE_MODE
        scale = density_scales[pname]
        value = center + grid_value * scale.sd
        haskey(scale, :lower) && isfinite(scale.lower) && (value = max(value, scale.lower))
        haskey(scale, :upper) && isfinite(scale.upper) && (value = min(value, scale.upper))
        return value
    end
    return center * grid_value
end

function grid_range_description(range_mode::String, grid_values::Vector{Float64})
    if range_mode == STD_RANGE_MODE
        return "standard-deviation offsets `[$(minimum(grid_values)), $(maximum(grid_values))]`"
    end
    return "multipliers `[$(minimum(grid_values)), $(maximum(grid_values))]`"
end

function save_paper_curvature_plot(records, pairs, grid_values, out_dir, paper_figure_dir::String;
                                   figure_title::String,
                                   center_label::String,
                                   paper_figure_stem::String,
                                   range_mode::String,
                                   density_scales)
    display_title = display_text(figure_title)
    display_center_label = display_text(center_label)
    fig_dir = joinpath(out_dir, "figures")
    mkpath(fig_dir)
    !isempty(strip(paper_figure_dir)) && mkpath(paper_figure_dir)

    caxis = center_axis_value(range_mode)
    center_idx = argmin(abs.(grid_values .- caxis))
    contour_values = Float64[]
    for (pi, pair) in enumerate(pairs)
        z = fill_isolated_missing_for_plot(
            matrix_from_records(records, pi, length(grid_values), length(grid_values), "rmse")
        )
        append!(contour_values, filter(isfinite, vec(1000.0 .* z)))
    end
    contour_min = isempty(contour_values) ? 0.0 : floor(minimum(contour_values))
    contour_max = isempty(contour_values) ? 1.0 : ceil(maximum(contour_values))
    contour_levels = collect(range(contour_min, contour_max; length=12))
    display_ticks = axis_ticks(range_mode, grid_values)

    contour_panels = Any[]
    cross_values = Float64[]
    ranges = Float64[]
    range_labels = String[]

    for (pi, pair) in enumerate(pairs)
        z_raw = matrix_from_records(records, pi, length(grid_values), length(grid_values), "rmse")
        z = fill_isolated_missing_for_plot(z_raw)
        z1000 = 1000.0 .* z
        title = pi == 1 ? "A. Real-side curvature" : "B. Kimball curvature"
        p = contourf(
            grid_values, grid_values, z1000;
            xlabel = parameter_axis_label(pair[1], range_mode),
            ylabel = parameter_axis_label(pair[2], range_mode),
            title = title,
            colorbar_title = "forecast RMSE x 1000",
            aspect_ratio = :equal,
            c = :turbo,
            levels = contour_levels,
            clims = (contour_min, contour_max),
            xlims = (minimum(grid_values), maximum(grid_values)),
            ylims = (minimum(grid_values), maximum(grid_values)),
            xticks = display_ticks,
            yticks = display_ticks,
            framestyle = :box,
            tickfontsize = 8,
            guidefontsize = 10,
            titlefontsize = 12,
        )
        contour!(p, grid_values, grid_values, z1000;
                 levels=contour_levels, color=:black, linewidth=0.85,
                 alpha=0.65, label="")
        scatter!(p, [caxis], [caxis]; marker=:circle, color=:red, markerstrokecolor=:black,
                 markerstrokewidth=0.5, markersize=4.5, label="")
        push!(contour_panels, p)

        append!(cross_values, filter(isfinite, vec(z1000)))
        finite_z = filter(isfinite, vec(z_raw))
        push!(ranges, isempty(finite_z) ? NaN : 1000.0 * (maximum(finite_z) - minimum(finite_z)))
        push!(range_labels, pi == 1 ? "real side" : "Kimball")
    end

    cross_pad = isempty(cross_values) ? 1.0 : max(0.5, 0.08 * (maximum(cross_values) - minimum(cross_values)))
    ymin = isempty(cross_values) ? 0.0 : minimum(cross_values) - cross_pad
    ymax = isempty(cross_values) ? 1.0 : maximum(cross_values) + cross_pad

    p_cross = plot(;
        xlabel = range_mode == STD_RANGE_MODE ?
                 "Standard deviations around $(display_center_label)" :
                 "Multiplier around $(display_center_label)",
        ylabel = "Forecast RMSE x 1000",
        title = "C. Local cross-sections",
        xlims = (minimum(grid_values), maximum(grid_values)),
        ylims = (ymin, ymax),
        xticks = display_ticks,
        framestyle = :box,
        legend = :bottomleft,
        legendfontsize = 8,
        tickfontsize = 8,
        guidefontsize = 10,
        titlefontsize = 12,
    )
    colors = [:steelblue4, :darkorange3, :forestgreen, :firebrick3]
    styles = [:solid, :solid, :dash, :dash]
    line_i = 0
    for (pi, pair) in enumerate(pairs)
        z = fill_isolated_missing_for_plot(
            matrix_from_records(records, pi, length(grid_values), length(grid_values), "rmse")
        )
        for (param_pos, pname) in enumerate(pair)
            line_i += 1
            ys = param_pos == 1 ? 1000.0 .* z[center_idx, :] :
                 1000.0 .* z[:, center_idx]
            plot!(p_cross, grid_values, ys;
                  label = short_parameter_name(pname),
                  color = colors[line_i],
                  linestyle = styles[line_i],
                  linewidth = 2.2,
                  marker = :circle,
                  markersize = 3)
        end
    end
    vline!(p_cross, [caxis]; color=:black, linestyle=:dot, linewidth=1.1, label="")

    p_bar = bar(
        range_labels, ranges;
        ylabel = "RMSE range over grid x 1000",
        title = "D. Surface amplitude",
        color = [:steelblue4, :forestgreen],
        label = "",
        framestyle = :box,
        xlims = (0.4, length(ranges) + 0.6),
        ylims = (0.0, maximum(filter(isfinite, ranges)) * 1.18),
        tickfontsize = 8,
        guidefontsize = 10,
        titlefontsize = 12,
    )
    for (i, val) in enumerate(ranges)
        isfinite(val) && annotate!(p_bar, i, val * 1.035, text(@sprintf("%.1f", val), 9, :center))
    end

    panels = Any[]
    append!(panels, contour_panels)
    push!(panels, p_cross)
    push!(panels, p_bar)
    fig = plot(
        panels...;
        layout = (2, 2),
        size = (1180, 900),
        left_margin = 5 * Plots.mm,
        bottom_margin = 5 * Plots.mm,
        plot_title = display_title,
        plot_titlefontsize = 16,
    )

    artifact_pdf = joinpath(fig_dir, "posterior_mean_curvature_surface_paper.pdf")
    artifact_png = joinpath(fig_dir, "posterior_mean_curvature_surface_paper.png")
    savefig(fig, artifact_pdf)
    savefig(fig, artifact_png)

    if !isempty(strip(paper_figure_dir)) && !isempty(strip(paper_figure_stem))
        savefig(fig, joinpath(paper_figure_dir, "$(paper_figure_stem).pdf"))
        savefig(fig, joinpath(paper_figure_dir, "$(paper_figure_stem).png"))
    end
end

function save_surface_plots(records, pairs, grid_values, out_dir;
                            paper_figure_dir::String="",
                            figure_title::String="ROM1 Forecast Errors",
                            center_label::String="local center",
                            paper_figure_stem::String="fig_hlt_posterior_mean_curvature_surface",
                            range_mode::String=MULTIPLIER_RANGE_MODE,
                            density_scales=Dict{Symbol, NamedTuple}())
    display_center_label = replace(center_label, "_" => " ")
    fig_dir = joinpath(out_dir, "figures")
    mkpath(fig_dir)
    gr()
    finite_logs = Float64[]
    for r in records
        rmse = Float64(r["rmse"])
        isfinite(rmse) && rmse > 0 && push!(finite_logs, log10(rmse))
    end
    common_clims = isempty(finite_logs) ? nothing : (minimum(finite_logs), maximum(finite_logs))

    heatmaps = Any[]
    for (pi, pair) in enumerate(pairs)
        z = matrix_from_records(records, pi, length(grid_values), length(grid_values), "rmse")
        logz = log10.(z)
        p = heatmap(
            grid_values, grid_values, logz;
            xlabel = parameter_axis_label(pair[1], range_mode),
            ylabel = parameter_axis_label(pair[2], range_mode),
            title = "$(pair[1]) x $(pair[2])",
            colorbar_title = "log10 RMSE",
            aspect_ratio = :equal,
            c = :viridis,
            clims = common_clims,
        )
        contour!(p, grid_values, grid_values, logz; levels=8, color=:white, linewidth=1, alpha=0.65)
        caxis = center_axis_value(range_mode)
        scatter!(p, [caxis], [caxis]; marker=:circle, color=:red, label=display_center_label)
        savefig(p, joinpath(fig_dir, "surface_heatmap_$(pair[1])_$(pair[2]).pdf"))
        savefig(p, joinpath(fig_dir, "surface_heatmap_$(pair[1])_$(pair[2]).png"))
        push!(heatmaps, p)

        p3 = surface(
            grid_values, grid_values, z;
            xlabel = parameter_axis_label(pair[1], range_mode),
            ylabel = parameter_axis_label(pair[2], range_mode),
            zlabel = "RMSE",
            title = "ROM1 forecast-error surface: $(pair[1]) x $(pair[2])",
            c = :viridis,
            camera = (35, 30),
        )
        savefig(p3, joinpath(fig_dir, "surface_3d_$(pair[1])_$(pair[2]).pdf"))
        savefig(p3, joinpath(fig_dir, "surface_3d_$(pair[1])_$(pair[2]).png"))

        center_idx = argmin(abs.(grid_values .- center_axis_value(range_mode)))
        px = plot(
            grid_values, z[center_idx, :];
            xlabel = "$(pair[1]) $(axis_unit_label(range_mode))",
            ylabel = "RMSE",
            title = "Posterior-mean cross-section",
            marker = :circle,
            label = String(pair[1]),
            linewidth = 2,
        )
        plot!(px, grid_values, z[:, center_idx];
              marker=:square, label=String(pair[2]), linewidth=2)
        vline!(px, [center_axis_value(range_mode)]; color=:black, linestyle=:dash, label="")
        savefig(px, joinpath(fig_dir, "surface_cross_section_$(pair[1])_$(pair[2]).pdf"))
        savefig(px, joinpath(fig_dir, "surface_cross_section_$(pair[1])_$(pair[2]).png"))
    end

    if length(heatmaps) > 1
        combo = plot(heatmaps...; layout=(1, length(heatmaps)), size=(520 * length(heatmaps), 460))
        savefig(combo, joinpath(fig_dir, "surface_heatmap_comparison.pdf"))
        savefig(combo, joinpath(fig_dir, "surface_heatmap_comparison.png"))
    end

    save_paper_curvature_plot(records, pairs, grid_values, out_dir, paper_figure_dir;
                              figure_title=figure_title,
                              center_label=center_label,
                              paper_figure_stem=paper_figure_stem,
                              range_mode=range_mode,
                              density_scales=density_scales)
end

function finite_mean(xs)
    vals = filter(isfinite, xs)
    isempty(vals) ? NaN : Statistics.mean(vals)
end

function surface_payload(records, pairs, grid_values, base_params, observables, chain_path,
                         theta_source, center_label, figure_title, run_id,
                         range_mode, density_scales)
    return Dict(
        "records" => records,
        "pairs" => pairs,
        "grid_values" => grid_values,
        "multipliers" => grid_values,
        "base_params" => base_params,
        "observables" => observables,
        "chain_path" => chain_path,
        "theta_source" => theta_source,
        "center_label" => center_label,
        "figure_title" => figure_title,
        "run_id" => run_id,
        "range_mode" => range_mode,
        "density_scales" => density_scales,
        "timestamp" => now(),
    )
end

function record_key(r::Dict{String, Any})
    return (Int(r["pair_index"]), Int(r["ix"]), Int(r["iy"]))
end

function ordered_grid_indices(grid_values::Vector{Float64}, center_first::Bool, range_mode::String)
    idx = collect(eachindex(grid_values))
    center_idx = argmin(abs.(grid_values .- center_axis_value(range_mode)))
    center_first || return idx
    return sort(idx; by = i -> (abs(i - center_idx), i))
end

run_id = parse_arg_string(ARGS, "--run-id", "posterior_mean_curvature_surface_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
out_root = parse_arg_string(ARGS, "--output-root",
    joinpath(REPO_ROOT, ".local_artifacts", "hlt_posterior_mean_curvature_surface"))
out_dir = parse_arg_string(ARGS, "--output-dir", joinpath(out_root, run_id))
chain_path = parse_arg_string(ARGS, "--chain",
    joinpath(REPO_ROOT, ".local_artifacts", "hlt_18param_realdata", "hlt_surrogate_hmc_extended_18p_pooled_8000_20260609.jls"))
if !isfile(chain_path)
    chain_path = joinpath(REPO_ROOT, ".local_artifacts", "hlt_18param_realdata", "hlt_surrogate_hmc_extended_18p_2000.jls")
end

grid_size = parse_arg_int(ARGS, "--grid-size", 5)
mult_min = parse_arg_float(ARGS, "--mult-min", 0.8)
mult_max = parse_arg_float(ARGS, "--mult-max", 1.2)
range_mode = lowercase(strip(parse_arg_string(ARGS, "--range-mode", MULTIPLIER_RANGE_MODE)))
range_mode in (MULTIPLIER_RANGE_MODE, STD_RANGE_MODE) ||
    error("--range-mode must be `$(MULTIPLIER_RANGE_MODE)` or `$(STD_RANGE_MODE)`.")
std_radius = parse_arg_float(ARGS, "--std-radius", 1.0)
std_radius > 0 || error("--std-radius must be positive.")
fallback_rel_sd = parse_arg_float(ARGS, "--fallback-rel-sd", 0.10)
fallback_rel_sd > 0 || error("--fallback-rel-sd must be positive.")
density_sd_overrides = parse_param_overrides(parse_arg_string(ARGS, "--density-sd-overrides", ""))
density_bounds_overrides = parse_param_bounds(parse_arg_string(ARGS, "--density-bounds-overrides", ""))
shock_scale = parse_arg_float(ARGS, "--shock-scale", 0.25)
n_shock_paths = parse_arg_int(ARGS, "--n-shock-paths", 2)
sim_periods = parse_arg_int(ARGS, "--sim-periods", 4)
burn_in = parse_arg_int(ARGS, "--burn-in", 2)
seed0 = parse_arg_int(ARGS, "--seed", 42)
sep_horizon = parse_arg_int(ARGS, "--sep-horizon", 4)
sep_maxit = parse_arg_int(ARGS, "--sep-maxit", 120)
sep_tol = parse_arg_float(ARGS, "--sep-tol", 1e-5)
sep_accept_tol = parse_arg_float(ARGS, "--sep-accept-tol", 0.5)
fallback_raw = lowercase(strip(parse_arg_string(ARGS, "--sep-fallback-solver", "qr")))
sep_fallback_solver = fallback_raw in ("", "none", "nothing", "null") ? nothing : Symbol(fallback_raw)
dry_run = parse_arg_bool(ARGS, "--dry-run", false)
plot_only = parse_arg_bool(ARGS, "--plot-only", false)
resume = parse_arg_bool(ARGS, "--resume", false)
center_first = parse_arg_bool(ARGS, "--center-first", false)
paper_figure_dir = parse_arg_string(ARGS, "--paper-figure-dir",
    joinpath(REPO_ROOT, "docs", "SurrogateNN_paper", "figures"))
paper_figure_stem = parse_arg_string(ARGS, "--paper-figure-stem", "fig_hlt_posterior_mean_curvature_surface")
theta_source = lowercase(strip(parse_arg_string(ARGS, "--theta-source", "chain")))
theta_source in ("chain", "model") || error("--theta-source must be `chain` or `model`.")
center_label = parse_arg_string(ARGS, "--center-label",
    theta_source == "chain" ? "posterior mean" : "fixed baseline")
figure_title = parse_arg_string(ARGS, "--figure-title",
    theta_source == "chain" ? "Posterior-Mean ROM1 Forecast Errors" : "Fixed-Calibration ROM1 Forecast Errors")
pairs = parse_pair_list(parse_arg_string(ARGS, "--surface-pairs", "csadjcost:czcap,curvp:curvw"))
param_overrides = parse_param_overrides(parse_arg_string(ARGS, "--param-overrides",
    "calfa=0.2,cfc=1.2,cg=0.18,chabb=0.67,clandaw=1.1,constebeta=0.3,constepinf=0.7,crdy=0.0,crpi=1.5,crr=0.73,cry=0.125,csadjcost=4.89,csigl=2.0,ctou=0.025,ctrend=0.4,curvw=8.31,czcap=0.431818"))

grid_size >= 2 || error("--grid-size must be at least 2")
mult_min > 0 && mult_max > mult_min || error("Require 0 < --mult-min < --mult-max")
mkpath(out_dir)

grid_values = range_mode == STD_RANGE_MODE ?
    collect(range(-std_radius, std_radius; length=grid_size)) :
    collect(range(mult_min, mult_max; length=grid_size))
grid_desc = grid_range_description(range_mode, grid_values)

println("=" ^ 78)
println("HLT CURVATURE SURFACE")
println("Started: $(now())")
println("=" ^ 78)
println("  Run id:       $run_id")
println("  Output:       $out_dir")
println("  Chain:        $chain_path")
println("  Grid:         $(grid_size)x$(grid_size), $grid_desc")
println("  Range mode:   $range_mode")
range_mode == STD_RANGE_MODE && println("  Fallback sd:  $(fallback_rel_sd * 100)% of center for fixed parameters")
range_mode == STD_RANGE_MODE && println("  SD overrides: $(format_param_overrides(density_sd_overrides))")
range_mode == STD_RANGE_MODE && println("  Bounds:       $(format_param_bounds(density_bounds_overrides))")
println("  Pairs:        $(join(["$(p[1]):$(p[2])" for p in pairs], ", "))")
println("  Shock scale:  $shock_scale")
println("  Shock paths:  $n_shock_paths")
println("  Periods:      $sim_periods + $burn_in burn-in")
println("  SEP:          horizon=$sep_horizon maxit=$sep_maxit tol=$sep_tol accept_tol=$sep_accept_tol")
println("  Paper figures:$paper_figure_dir")
println("  Paper stem:   $paper_figure_stem")
println("  Theta source: $theta_source")
println("  Center label: $center_label")
println("  Overrides:    $(format_param_overrides(param_overrides))")
println("  Dry run:      $dry_run")
println("  Plot only:    $plot_only")
println("  Resume:       $resume")
println("  Center first: $center_first")
flush(stdout)

if plot_only
    results_path = joinpath(out_dir, "curvature_surface_results.jls")
    isfile(results_path) || error("Saved results not found: $results_path")
    payload = Serialization.deserialize(results_path)
    saved_records = payload["records"]
    saved_pairs = payload["pairs"]
    saved_grid_values = get(payload, "grid_values", payload["multipliers"])
    saved_center_label = get(payload, "center_label", center_label)
    saved_figure_title = get(payload, "figure_title", figure_title)
    saved_range_mode = get(payload, "range_mode", MULTIPLIER_RANGE_MODE)
    saved_density_scales = get(payload, "density_scales", Dict{Symbol, NamedTuple}())
    save_surface_plots(saved_records, saved_pairs, saved_grid_values, out_dir;
                       paper_figure_dir=paper_figure_dir,
                       figure_title=saved_figure_title,
                       center_label=saved_center_label,
                       paper_figure_stem=paper_figure_stem,
                       range_mode=saved_range_mode,
                       density_scales=saved_density_scales)
    println("Plot-only render complete.")
    exit(0)
end

println("\nLoading HLT model...")
HLT = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT_obc"; mod=@__MODULE__)
MacroModelling.solve!(HLT, silent=true)
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
obs_idx = [find_var_index(HLT, v) for v in observables]

specs = get_phase1_18param_specs()
theta_names = [s.name for s in specs]
theta_mean = Float64[]
n_chain_draws = 0
if theta_source == "chain"
    theta_mean, n_chain_draws = load_posterior_mean(chain_path, theta_names)
end

base_params = Float64.(HLT.parameter_values)
theta_source == "chain" && apply_estimated_theta!(base_params, HLT, theta_names, theta_mean)
for (pname, pval) in param_overrides
    pidx = find_param_index(HLT, pname)
    base_params[pidx] = pval
end

pair_indices = [(find_param_index(HLT, p[1]), find_param_index(HLT, p[2])) for p in pairs]
posterior_stats = load_posterior_stats(chain_path)
density_scales = range_mode == STD_RANGE_MODE ?
    build_density_scales(pairs, specs, posterior_stats, density_sd_overrides,
                         density_bounds_overrides,
                         base_params, HLT; fallback_rel_sd=fallback_rel_sd,
                         std_radius=std_radius) :
    Dict{Symbol, NamedTuple}()
grid_order = ordered_grid_indices(grid_values, center_first, range_mode)

manifest_path = joinpath(out_dir, "RUN_MANIFEST.md")
open(manifest_path, "w") do io
    println(io, "# HLT Curvature Surface")
    println(io)
    println(io, "**Started:** $(now())")
    println(io)
    println(io, "- Theta source: `$theta_source`")
    theta_source == "chain" && println(io, "- Chain: `$chain_path`")
    theta_source == "chain" && println(io, "- Chain draws used for posterior mean: `$n_chain_draws`")
    println(io, "- Center label: `$center_label`")
    println(io, "- Grid: `$(grid_size)x$(grid_size)`, $(grid_range_description(range_mode, grid_values))")
    println(io, "- Range mode: `$range_mode`")
    if range_mode == STD_RANGE_MODE
        println(io, "- Fallback relative sd for fixed parameters: `$fallback_rel_sd`")
        println(io)
        println(io, "| Parameter | Center | SD | Lower | Upper | Source | Bounds |")
        println(io, "|---|---:|---:|---:|---:|---|---|")
        for pname in sort(collect(keys(density_scales)); by=String)
            s = density_scales[pname]
            lower = format_bound(s.lower)
            upper = format_bound(s.upper)
            @printf(io, "| `%s` | %.8g | %.8g | %s | %s | %s | %s |\n",
                    pname, s.center, s.sd, lower, upper, s.source, s.bounds_source)
        end
        println(io)
    end
    println(io, "- Surface pairs: `$(join(["$(p[1]):$(p[2])" for p in pairs], ", "))`")
    println(io, "- Shock scale: `$shock_scale`")
    println(io, "- Shock paths per cell: `$n_shock_paths`")
    println(io, "- Shock seeds: common across all grid cells")
    println(io, "- Periods: `$sim_periods` after `$burn_in` burn-in")
    println(io, "- SEP: horizon `$sep_horizon`, maxit `$sep_maxit`, tol `$sep_tol`, accept tol `$sep_accept_tol`")
    println(io, "- Maintained overrides: `$(format_param_overrides(param_overrides))`")
    println(io, "- Center-first traversal: `$center_first`")
end
println("Manifest: $manifest_path")

base_csv = joinpath(out_dir, "posterior_mean_parameters.csv")
open(base_csv, "w") do io
    println(io, "parameter,value")
    for pname in sort(Symbol.(HLT.parameters); by=String)
        pidx = find_param_index(HLT, pname)
        @printf(io, "%s,%.12g\n", pname, base_params[pidx])
    end
end

if range_mode == STD_RANGE_MODE
    density_csv = joinpath(out_dir, "density_scales.csv")
    open(density_csv, "w") do io
        println(io, "parameter,center,sd,lower,upper,source,bounds_source")
        for pname in sort(collect(keys(density_scales)); by=String)
            s = density_scales[pname]
            @printf(io, "%s,%.12g,%.12g,%.12g,%.12g,%s,%s\n",
                    pname, s.center, s.sd, s.lower, s.upper, s.source, s.bounds_source)
        end
    end
end

if dry_run
    println("Dry run complete.")
    exit(0)
end

records = Vector{Dict{String, Any}}()
checkpoint_path = joinpath(out_dir, "curvature_surface_checkpoint.jls")
if resume && isfile(checkpoint_path)
    checkpoint = Serialization.deserialize(checkpoint_path)
    loaded_records = get(checkpoint, "records", Dict{String, Any}[])
    records = Vector{Dict{String, Any}}(loaded_records)
    println("Loaded checkpoint with $(length(records)) completed cells: $checkpoint_path")
    flush(stdout)
end
completed = Set(record_key(r) for r in records)
total_cells = length(pairs) * grid_size * grid_size
cell = 0

for (pi, pair) in enumerate(pairs)
    p1idx, p2idx = pair_indices[pi]
    p1base = base_params[p1idx]
    p2base = base_params[p2idx]
    println("\n--- Surface $(pi)/$(length(pairs)): $(pair[1]) x $(pair[2]) ---")
    flush(stdout)

    for iy in grid_order, ix in grid_order
        gy = grid_values[iy]
        gx = grid_values[ix]
        global cell += 1
        key = (pi, ix, iy)
        if key in completed
            @printf("  cell %3d/%3d %-10s=% .3f %-8s=% .3f status=checkpoint\n",
                    cell, total_cells, String(pair[1]), gx, String(pair[2]), gy)
            flush(stdout)
            continue
        end
        params = copy(base_params)
        params[p1idx] = grid_value_to_parameter(p1base, gx, pair[1], range_mode, density_scales)
        params[p2idx] = grid_value_to_parameter(p2base, gy, pair[2], range_mode, density_scales)
        seed = seed0 * 1_000_000
        t0 = time()
        stats = cell_forecast_rmse(HLT, params, obs_idx;
            shock_scale=shock_scale,
            n_shock_paths=n_shock_paths,
            sim_periods=sim_periods,
            burn_in=burn_in,
            seed0=seed,
            sep_horizon=sep_horizon,
            sep_maxit=sep_maxit,
            sep_tol=sep_tol,
            sep_accept_tol=sep_accept_tol,
            sep_fallback_solver=sep_fallback_solver)
        elapsed = time() - t0
        @printf("  cell %3d/%3d %-10s=% .3f %-8s=% .3f rmse=%10.5g paths=%d/%d status=%s time=%.1fs\n",
                cell, total_cells, String(pair[1]), gx, String(pair[2]), gy,
                stats.rmse, stats.accepted_paths, stats.attempted_paths,
                stats.status, elapsed)
        push!(records, Dict{String, Any}(
            "pair_index" => pi,
            "pair" => "$(pair[1]):$(pair[2])",
            "x_parameter" => String(pair[1]),
            "y_parameter" => String(pair[2]),
            "ix" => ix,
            "iy" => iy,
            "x_grid_value" => gx,
            "y_grid_value" => gy,
            "x_multiplier" => range_mode == MULTIPLIER_RANGE_MODE ? gx : params[p1idx] / p1base,
            "y_multiplier" => range_mode == MULTIPLIER_RANGE_MODE ? gy : params[p2idx] / p2base,
            "x_std_offset" => range_mode == STD_RANGE_MODE ? gx : NaN,
            "y_std_offset" => range_mode == STD_RANGE_MODE ? gy : NaN,
            "x_value" => params[p1idx],
            "y_value" => params[p2idx],
            "rmse" => stats.rmse,
            "mse" => stats.mse,
            "n_samples" => stats.n_samples,
            "accepted_paths" => stats.accepted_paths,
            "attempted_paths" => stats.attempted_paths,
            "max_sep_error" => stats.max_error,
            "status" => stats.status,
            "elapsed_seconds" => elapsed,
        ))
        push!(completed, key)
        Serialization.serialize(checkpoint_path, surface_payload(
            records, pairs, grid_values, base_params, observables, chain_path,
            theta_source, center_label, figure_title, run_id,
            range_mode, density_scales))
        flush(stdout)
    end
end

results_path = joinpath(out_dir, "curvature_surface_results.jls")
Serialization.serialize(results_path, surface_payload(
    records, pairs, grid_values, base_params, observables, chain_path,
    theta_source, center_label, figure_title, run_id,
    range_mode, density_scales))
println("\nResults saved: $results_path")

csv_path = joinpath(out_dir, "curvature_surface.csv")
open(csv_path, "w") do io
    println(io, "pair_index,pair,x_parameter,y_parameter,ix,iy,x_grid_value,y_grid_value,x_multiplier,y_multiplier,x_std_offset,y_std_offset,x_value,y_value,rmse,mse,n_samples,accepted_paths,attempted_paths,max_sep_error,status,elapsed_seconds")
    for r in records
        @printf(io, "%d,%s,%s,%s,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%d,%.12g,%s,%.6f\n",
                r["pair_index"], r["pair"], r["x_parameter"], r["y_parameter"],
                r["ix"], r["iy"], r["x_grid_value"], r["y_grid_value"],
                r["x_multiplier"], r["y_multiplier"], r["x_std_offset"], r["y_std_offset"],
                r["x_value"], r["y_value"], r["rmse"], r["mse"], r["n_samples"],
                r["accepted_paths"], r["attempted_paths"], r["max_sep_error"],
                r["status"], r["elapsed_seconds"])
    end
end
println("CSV saved: $csv_path")

save_surface_plots(records, pairs, grid_values, out_dir;
                   paper_figure_dir=paper_figure_dir,
                   figure_title=figure_title,
                   center_label=center_label,
                   paper_figure_stem=paper_figure_stem,
                   range_mode=range_mode,
                   density_scales=density_scales)

summary_path = joinpath(out_dir, "CURVATURE_SURFACE_SUMMARY.md")
open(summary_path, "w") do io
    println(io, "# HLT Curvature Surface")
    println(io)
    println(io, "**Generated:** $(now())")
    println(io)
    println(io, "Metric: RMSE of one-step observable forecast errors `SEP - ROM1`, evaluated at the same SEP state, same shocks, and same parameters.")
    println(io)
    println(io, "- Theta source: `$theta_source`")
    theta_source == "chain" && println(io, "- Chain: `$chain_path`")
    theta_source == "chain" && println(io, "- Chain draws used for posterior mean: `$n_chain_draws`")
    println(io, "- Center label: `$center_label`")
    println(io, "- Grid: `$(grid_size)x$(grid_size)`, $(grid_range_description(range_mode, grid_values))")
    println(io, "- Range mode: `$range_mode`")
    if range_mode == STD_RANGE_MODE
        println(io, "- Density-scale provenance:")
        println(io)
        println(io, "| Parameter | Center | SD | Lower | Upper | Source | Bounds |")
        println(io, "|---|---:|---:|---:|---:|---|---|")
        for pname in sort(collect(keys(density_scales)); by=String)
            s = density_scales[pname]
            lower = format_bound(s.lower)
            upper = format_bound(s.upper)
            @printf(io, "| `%s` | %.8g | %.8g | %s | %s | %s | %s |\n",
                    pname, s.center, s.sd, lower, upper, s.source, s.bounds_source)
        end
        println(io)
    end
    println(io, "- Shock scale: `$shock_scale`")
    println(io, "- Shock paths per cell: `$n_shock_paths`")
    println(io, "- Shock seeds: common across all grid cells, so surface changes isolate parameter changes.")
    println(io, "- Periods: `$sim_periods` after `$burn_in` burn-in")
    println(io)
    println(io, "| Surface | Accepted cells | Mean RMSE | Center RMSE | Min RMSE | Max RMSE |")
    println(io, "|---|---:|---:|---:|---:|---:|")
    center_idx = argmin(abs.(grid_values .- center_axis_value(range_mode)))
    for (pi, pair) in enumerate(pairs)
        rs = [r for r in records if r["pair_index"] == pi]
        rmses = [Float64(r["rmse"]) for r in rs if isfinite(Float64(r["rmse"]))]
        accepted_cells = count(r -> r["accepted_paths"] == r["attempted_paths"], rs)
        center = filter(r -> r["ix"] == center_idx && r["iy"] == center_idx, rs)
        center_rmse = isempty(center) ? NaN : center[1]["rmse"]
        @printf(io, "| `%s:%s` | %d/%d | %.6g | %.6g | %.6g | %.6g |\n",
                pair[1], pair[2], accepted_cells, length(rs), finite_mean(rmses),
                center_rmse, isempty(rmses) ? NaN : minimum(rmses),
                isempty(rmses) ? NaN : maximum(rmses))
    end
    println(io)
    println(io, "Figures are in `figures/`. The paper contour figure compares the real-side surface (`csadjcost`, `czcap`) to the Kimball surface (`curvp`, `curvw`) using the RMSE of one-step SEP-minus-ROM1 observable forecast errors.")
end
println("Summary saved: $summary_path")

println("\n" * "=" ^ 78)
println("DONE: $(now())")
println("=" ^ 78)
