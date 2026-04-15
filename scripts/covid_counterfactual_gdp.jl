#!/usr/bin/env julia
# ============================================================================
# COVID COUNTERFACTUAL GDP PATHS: LINEAR VS SURROGATE POSTERIORS
# ============================================================================
#
# Computes counterfactual GDP paths during COVID (2019Q1-2022Q4) under the
# linear versus surrogate posterior means.
#
# Method:
#   1. At each posterior mean, run the Kalman smoother to extract smoothed
#      states and shocks over the full sample.
#   2. Build a "no-COVID counterfactual" by replacing shocks in 2020Q1-2021Q2
#      with zeros and simulating forward from the smoothed state at 2019Q4.
#   3. Compute cumulative GDP loss = sum of (actual - counterfactual) dy.
#
# Key insight: the linear model attributes more of the COVID GDP drop to
# exogenous shocks (because it has larger sigma values), while the surrogate
# attributes more to endogenous propagation of smaller shocks through the
# investment/capital nonlinearities.
#
# Usage:
#   julia --project=. scripts/covid_counterfactual_gdp.jl
#
# Inputs (hardcoded paths):
#   - Linear chain:    .local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls
#   - Surrogate chain: .local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_2000.jls
#   - Data payload:    .local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p.jls
#
# Output:
#   - docs/paper/figures/covid_counterfactual_gdp.pdf
# ============================================================================

using MacroModelling
using Serialization
using LinearAlgebra
using AxisKeys
using Printf
using Dates
import Statistics: mean, std, quantile

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "parameter_config.jl"))

# ============================================================================
# Paths
# ============================================================================

const REPO_ROOT  = normpath(joinpath(@__DIR__, ".."))
const ARTIFACTS  = joinpath(REPO_ROOT, ".local_artifacts", "hlt_18param_realdata")
const FIG_DIR    = joinpath(REPO_ROOT, "docs", "paper", "figures")

const LINEAR_CHAIN_PATH    = joinpath(ARTIFACTS, "hlt_linear_hmc_extended_18p_2000.jls")
const SURROGATE_CHAIN_PATH = joinpath(ARTIFACTS, "hlt_surrogate_hmc_extended_18p_2000.jls")
const PAYLOAD_PATH         = joinpath(ARTIFACTS, "hlt_real_data_payload_extended_18p.jls")

mkpath(FIG_DIR)

println("=" ^ 72)
println("COVID COUNTERFACTUAL GDP ANALYSIS")
println("Started: $(now())")
println("=" ^ 72)

# ============================================================================
# Step 1: Load Data Payload
# ============================================================================

println("\n--- Step 1: Loading data payload ---")
isfile(PAYLOAD_PATH) || error("Data payload not found: $PAYLOAD_PATH")
payload = Serialization.deserialize(PAYLOAD_PATH)

obs_data    = Float64.(payload["obs_data"])         # (7, T)
observables = Symbol.(payload["observables"])       # [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
theta_names = Symbol.(payload["theta_names"])

d_obs, T_obs = size(obs_data)
println("  Observables: $observables ($d_obs x $T_obs)")

# Time mapping: Period 1 = 1959Q1, so 2019Q1 = period 241
# The estimation sample: rows 47 to 311 of the CSV (1959Q1-2025Q1), T=265
const PERIOD_2019Q1 = 241    # 1959Q1 + 240 quarters
const PERIOD_2019Q4 = 244
const PERIOD_2020Q1 = 245
const PERIOD_2020Q2 = 246
const PERIOD_2021Q2 = 250
const PERIOD_2022Q4 = 256

# Verify T_obs is consistent
@assert T_obs >= PERIOD_2022Q4 "Data sample too short: T=$T_obs, need at least $PERIOD_2022Q4"

# Observable index for dy (GDP growth)
dy_idx = findfirst(==(:dy), observables)
dy_idx === nothing && error("Observable :dy not found in $observables")

# Plot window
const PLOT_START = PERIOD_2019Q1
const PLOT_END   = min(PERIOD_2022Q4, T_obs)

# Quarter labels for x-axis
function quarter_label(period::Int; base_year=1959, base_quarter=1)
    total_q = (base_quarter - 1) + (period - 1)
    year = base_year + total_q ÷ 4
    q = (total_q % 4) + 1
    return "$(year)Q$(q)"
end

plot_periods = PLOT_START:PLOT_END
quarter_labels = [quarter_label(p) for p in plot_periods]

# COVID zeroing window: 2020Q1 through 2021Q2
const ZERO_START = PERIOD_2020Q1
const ZERO_END   = PERIOD_2021Q2

println("  Plot window:   $(quarter_label(PLOT_START)) - $(quarter_label(PLOT_END))")
println("  COVID zeroing: $(quarter_label(ZERO_START)) - $(quarter_label(ZERO_END))")

# Build KeyedArray for Kalman smoother
obs_data_ka = KeyedArray(obs_data; Variable=observables, Time=1:T_obs)

# ============================================================================
# Step 2: Load Model
# ============================================================================

println("\n--- Step 2: Loading HLT model (non-OBC, first-order) ---")
mm_model = load_hlt_model(REPO_ROOT, "Smets_Wouters_2007_HLT"; mod=@__MODULE__)
println("  Model: $(mm_model.model_name)")
println("  Variables: $(length(mm_model.var))")
println("  Shocks:    $(length(mm_model.exo))")

# Parameter index mapping
theta_param_idx = let idx_any = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx_any)
        missing_names = theta_names[isnothing.(idx_any)]
        error("Theta names not found in model parameters: $missing_names")
    end
    Int.(idx_any)
end

# ============================================================================
# Step 3: Load Posterior Chains and Compute Posterior Means
# ============================================================================

println("\n--- Step 3: Loading posterior chains ---")

function load_chain_posterior_mean(path::String, label::String)
    if !isfile(path)
        println("  WARNING: $label chain not found: $path")
        return nothing, nothing
    end
    chain_data = Serialization.deserialize(path)
    chain_matrix = chain_data["chain"]           # (n_draws, n_params)
    chain_theta  = Symbol.(chain_data["theta_names"])
    n_draws = size(chain_matrix, 1)

    # Reindex to match theta_names ordering if needed
    if chain_theta != theta_names
        reindex = [findfirst(==(n), chain_theta) for n in theta_names]
        if any(isnothing, reindex)
            error("$label chain missing parameters: $(theta_names[isnothing.(reindex)])")
        end
        chain_matrix = chain_matrix[:, reindex]
    end

    post_mean = vec(mean(chain_matrix, dims=1))
    ll_post_mean = get(chain_data, "ll_post_mean", NaN)
    n_divergent  = get(chain_data, "n_divergent", -1)

    println("  $label: $n_draws draws, LL at mean = $(round(ll_post_mean, digits=1)), div = $n_divergent")
    return post_mean, chain_data
end

theta_linear, chain_linear       = load_chain_posterior_mean(LINEAR_CHAIN_PATH, "Linear HMC")
theta_surrogate, chain_surrogate = load_chain_posterior_mean(SURROGATE_CHAIN_PATH, "Surrogate HMC")

# We need at least one chain to proceed
if theta_linear === nothing && theta_surrogate === nothing
    error("No chains found. Need at least one of:\n  $LINEAR_CHAIN_PATH\n  $SURROGATE_CHAIN_PATH")
end

# ============================================================================
# Step 4: Kalman Smoother + Counterfactual Simulation
# ============================================================================

println("\n--- Step 4: Computing counterfactual GDP paths ---")

"""
Build parameter vector with theta overrides applied to the model's base parameters.
"""
function build_param_pairs(theta_vec::Vector{Float64})
    return [theta_names[i] => theta_vec[i] for i in eachindex(theta_names)]
end

"""
Run the Kalman smoother at given posterior parameters and compute counterfactual GDP.

Returns a NamedTuple with:
  - actual_dy:        actual GDP growth over the plot window
  - counterfactual_dy: counterfactual (no-COVID shocks) GDP growth
  - smoothed_shocks:   full smoothed shock matrix (for diagnostics)
  - cumulative_loss:   sum of (actual - counterfactual) dy over COVID quarters
"""
function compute_counterfactual(theta_vec::Vector{Float64}, label::String)
    println("\n  Processing: $label")
    param_pairs = build_param_pairs(theta_vec)

    # --- Run Kalman smoother to get smoothed variables and shocks ---
    t0 = time()
    smoothed_vars = get_estimated_variables(
        mm_model, obs_data_ka;
        parameters = param_pairs,
        algorithm  = :first_order,
        filter     = :kalman,
        smooth     = true,
        data_in_levels = false,
        levels     = false,         # return deviations from SS, not levels
        verbose    = false,
    )
    dt_vars = time() - t0

    smoothed_shocks = get_estimated_shocks(
        mm_model, obs_data_ka;
        parameters = param_pairs,
        algorithm  = :first_order,
        filter     = :kalman,
        smooth     = true,
        data_in_levels = false,
        verbose    = false,
    )
    dt_total = time() - t0
    @printf("    Kalman smoother: %.2f s\n", dt_total)

    # Extract actual dy from smoothed variables
    # The smoother returns all model variables; find :dy
    var_names = collect(axiskeys(smoothed_vars, 1))
    dy_var_idx = findfirst(x -> x == :dy || x == "dy", var_names)
    dy_var_idx === nothing && error("Variable :dy not found in smoother output")

    actual_dy_full = Matrix(smoothed_vars)[dy_var_idx, :]  # (T_obs,)

    # Extract smoothed shocks as a plain matrix
    shock_names  = collect(axiskeys(smoothed_shocks, 1))
    shocks_matrix = Matrix(smoothed_shocks)  # (n_shocks, T_obs)
    n_shocks = size(shocks_matrix, 1)

    @printf("    Smoothed shocks: %d shocks x %d periods\n", n_shocks, T_obs)

    # --- Build first-order solution matrices for forward simulation ---
    # The first-order state-space:  x_t = T_mat * x_{t-1,state} + c + R_mat * eps_t
    # where x is the full variable vector in deviations from steady state.
    #
    # From get_solution, the solution_matrix has structure:
    #   columns = [past_not_future_and_mixed_{t-1}, Volatility(=1), shocks_t]
    #   rows    = all model variables

    MacroModelling.solve!(mm_model; parameters=param_pairs, algorithm=:first_order, dynamics=true, silent=true)

    sol = mm_model.solution.perturbation.first_order.solution_matrix
    n_vars = size(sol, 1)
    n_state = length(mm_model.timings.past_not_future_and_mixed)
    n_exo   = length(mm_model.timings.exo)

    # Transition matrix (T_mat): maps state_{t-1} to vars_t
    T_mat = sol[:, 1:n_state]
    # Shock impact matrix (R_mat) — first-order solution has no constant column
    # Layout: [state cols | shock cols]  (n_state + n_exo columns total)
    n_cols = size(sol, 2)
    if n_cols == n_state + 1 + n_exo
        # Has volatility/constant column
        c_vec = sol[:, n_state + 1]
        R_mat = sol[:, n_state + 2 : n_state + 1 + n_exo]
    elseif n_cols == n_state + n_exo
        # No constant column (standard first-order perturbation)
        c_vec = zeros(n_vars)
        R_mat = sol[:, n_state + 1 : n_state + n_exo]
    else
        error("Unexpected solution matrix size: $n_vars × $n_cols (expected $n_state + $n_exo = $(n_state + n_exo) or $(n_state + 1 + n_exo) cols)")
    end

    # State indices: which entries of the full variable vector are the states
    state_idx = mm_model.timings.past_not_future_and_mixed_idx

    # The smoother shock names have ₍ₓ₎ suffix; model exo names don't.
    # Build mapping from smoother shock ordering to model exo ordering.
    model_exo_names = mm_model.timings.exo
    shock_reindex = Int[]
    for me in model_exo_names
        me_str = string(me) * "₍ₓ₎"
        idx = findfirst(x -> string(x) == me_str || x == me_str, shock_names)
        if idx === nothing
            # Try without suffix
            idx = findfirst(x -> string(x) == string(me), shock_names)
        end
        if idx === nothing
            push!(shock_reindex, 0)  # shock not found, will be zero
        else
            push!(shock_reindex, idx)
        end
    end

    # Reindex shocks to match model ordering
    shocks_reindexed = zeros(n_exo, T_obs)
    for (j, si) in enumerate(shock_reindex)
        si > 0 && (shocks_reindexed[j, :] .= shocks_matrix[si, :])
    end

    # --- Forward simulation: counterfactual with zeroed COVID shocks ---
    #
    # We use the smoothed state at 2019Q4 as our starting point and simulate
    # forward through 2022Q4. In the counterfactual, shocks during 2020Q1-2021Q2
    # are set to zero; shocks outside that window use their smoothed values.

    # Get smoothed state at 2019Q4 from the smoothed variables
    # The state is the subset of variables at past_not_future_and_mixed_idx
    all_vars_smoothed = Matrix(smoothed_vars)   # (n_model_vars, T_obs)
    state_2019q4 = all_vars_smoothed[state_idx, PERIOD_2019Q4]

    # Simulation horizon: from 2019Q4+1 forward
    sim_start = PERIOD_2019Q4 + 1   # = 2020Q1
    sim_end   = PLOT_END
    n_sim = sim_end - sim_start + 1

    # -- Actual path: propagate with actual smoothed shocks (verify it matches) --
    actual_sim = zeros(n_vars, n_sim)
    state_prev = state_2019q4
    for t in 1:n_sim
        period = sim_start + t - 1
        eps_t = shocks_reindexed[:, period]
        x_t = T_mat * state_prev + c_vec + R_mat * eps_t
        actual_sim[:, t] = x_t
        state_prev = x_t[state_idx]
    end

    # -- Counterfactual path: zero shocks in 2020Q1-2021Q2 --
    cf_sim = zeros(n_vars, n_sim)
    state_prev = state_2019q4
    for t in 1:n_sim
        period = sim_start + t - 1
        eps_t = copy(shocks_reindexed[:, period])
        # Zero out shocks during the COVID window
        if ZERO_START <= period <= ZERO_END
            eps_t .= 0.0
        end
        x_t = T_mat * state_prev + c_vec + R_mat * eps_t
        cf_sim[:, t] = x_t
        state_prev = x_t[state_idx]
    end

    # Extract dy from forward simulations
    # Need the index of :dy in model variables (timings.var)
    model_var_names = mm_model.timings.var
    dy_model_idx = findfirst(==(:dy), model_var_names)
    dy_model_idx === nothing && error(":dy not in model timings.var")

    actual_dy_sim = actual_sim[dy_model_idx, :]
    cf_dy_sim     = cf_sim[dy_model_idx, :]

    # For the full plot window (2019Q1-2022Q4), use smoother output for
    # 2019Q1-2019Q4, and the forward simulation for 2020Q1-2022Q4
    n_plot = length(plot_periods)
    actual_dy_plot = zeros(n_plot)
    cf_dy_plot     = zeros(n_plot)

    for (i, p) in enumerate(plot_periods)
        actual_dy_plot[i] = actual_dy_full[p]
        if p <= PERIOD_2019Q4
            # Before the counterfactual divergence: actual = counterfactual
            cf_dy_plot[i] = actual_dy_full[p]
        else
            t_sim = p - sim_start + 1
            cf_dy_plot[i] = cf_dy_sim[t_sim]
        end
    end

    # Cumulative GDP loss: sum of (actual - counterfactual) over COVID quarters
    # A negative value of (actual - counterfactual) dy means GDP fell more than
    # counterfactual, i.e., a COVID-induced loss. The sign convention: if actual
    # dy is lower than counterfactual dy, the loss is positive.
    covid_quarters = ZERO_START:ZERO_END
    loss_per_quarter = zeros(length(covid_quarters))
    for (i, p) in enumerate(covid_quarters)
        t_sim = p - sim_start + 1
        # loss = counterfactual_dy - actual_dy  (positive = output lost)
        loss_per_quarter[i] = cf_dy_sim[t_sim] - actual_dy_sim[t_sim]
    end
    cumulative_loss = sum(loss_per_quarter)

    # Also compute cumulative loss through 2022Q4 for the full recovery picture
    full_window = ZERO_START:PLOT_END
    full_loss = 0.0
    for p in full_window
        t_sim = p - sim_start + 1
        full_loss += cf_dy_sim[t_sim] - actual_dy_sim[t_sim]
    end

    # Report shock sizes during COVID
    shock_norms = [norm(shocks_reindexed[:, p]) for p in covid_quarters]
    @printf("    Shock norm (2020Q1): %.3f\n", shock_norms[1])
    @printf("    Shock norm (2020Q2): %.3f\n", shock_norms[2])
    @printf("    Mean shock norm (COVID window): %.3f\n", mean(shock_norms))
    @printf("    Cumulative GDP loss (2020Q1-2021Q2): %.2f pp\n", cumulative_loss)
    @printf("    Cumulative GDP loss (2020Q1-2022Q4): %.2f pp\n", full_loss)

    return (
        actual_dy        = actual_dy_plot,
        counterfactual_dy = cf_dy_plot,
        smoothed_shocks  = shocks_reindexed,
        cumulative_loss  = cumulative_loss,
        full_loss        = full_loss,
        loss_per_quarter = loss_per_quarter,
        shock_norms      = shock_norms,
    )
end

# Run counterfactual for each available posterior
results = Dict{String, Any}()

if theta_linear !== nothing
    results["Linear"] = compute_counterfactual(theta_linear, "Linear (Kalman) posterior mean")
end

if theta_surrogate !== nothing
    results["Surrogate"] = compute_counterfactual(theta_surrogate, "Surrogate (RS) posterior mean")
end

# ============================================================================
# Step 5: Plotting
# ============================================================================

println("\n--- Step 5: Generating figures ---")

using Plots
gr()
default(
    fontfamily     = "Computer Modern",
    titlefontsize  = 11,
    guidefontsize  = 10,
    tickfontsize   = 8,
    legendfontsize = 9,
    linewidth      = 2.0,
    dpi            = 300,
)

# Determine which results we have for consistent coloring
has_linear    = haskey(results, "Linear")
has_surrogate = haskey(results, "Surrogate")

# --- Main Figure: Actual vs Counterfactual GDP growth ---

n_plot = length(plot_periods)
tick_positions = 1:2:n_plot
tick_labels = quarter_labels[tick_positions]

fig = plot(
    size          = (780, 480),
    margin        = 5Plots.mm,
    bottom_margin = 14Plots.mm,
    left_margin   = 8Plots.mm,
)

# Shade the COVID zeroing window
zero_start_idx = findfirst(==(ZERO_START), collect(plot_periods))
zero_end_idx   = findfirst(==(ZERO_END), collect(plot_periods))
if zero_start_idx !== nothing && zero_end_idx !== nothing
    vspan!(fig, [zero_start_idx - 0.5, zero_end_idx + 0.5];
           color=:gray90, label="", alpha=0.5)
end

# Plot actual GDP growth (same for both posteriors since it comes from data,
# but the smoother output may differ slightly; use the first available)
ref_key = has_linear ? "Linear" : "Surrogate"
plot!(fig, 1:n_plot, results[ref_key].actual_dy;
      color=:black, linewidth=2.5, label="Actual GDP growth (data)",
      linestyle=:solid)

# Counterfactual paths
if has_linear
    plot!(fig, 1:n_plot, results["Linear"].counterfactual_dy;
          color=RGB(0.2, 0.4, 0.8), linewidth=2.0,
          linestyle=:dash,
          label="Counterfactual: Linear posterior")
end

if has_surrogate
    plot!(fig, 1:n_plot, results["Surrogate"].counterfactual_dy;
          color=RGB(0.8, 0.2, 0.2), linewidth=2.0,
          linestyle=:dashdot,
          label="Counterfactual: Surrogate posterior")
end

plot!(fig,
    xlabel  = "",
    ylabel  = "Quarter-on-quarter GDP growth (dev. from SS)",
    title   = "COVID Counterfactual: Actual vs No-Shock GDP Paths",
    legend  = :bottomright,
    xticks  = (collect(tick_positions), tick_labels),
    xrotation = 45,
)

# Add annotation for the zeroing window
annotate!(fig, [(mean([zero_start_idx, zero_end_idx]), -0.02,
                 Plots.text("Shocks zeroed", 8, :gray40, :center))])

figpath = joinpath(FIG_DIR, "covid_counterfactual_gdp.pdf")
savefig(fig, figpath)
println("  Saved: $figpath")

# --- Supplementary Figure: Cumulative GDP gap ---

fig2 = plot(
    size          = (780, 420),
    margin        = 5Plots.mm,
    bottom_margin = 14Plots.mm,
    left_margin   = 8Plots.mm,
)

# Shade
if zero_start_idx !== nothing && zero_end_idx !== nothing
    vspan!(fig2, [zero_start_idx - 0.5, zero_end_idx + 0.5];
           color=:gray90, label="", alpha=0.5)
end

# Cumulative gap: counterfactual_dy - actual_dy  (positive = loss)
if has_linear
    gap_linear = cumsum(results["Linear"].counterfactual_dy .- results["Linear"].actual_dy)
    plot!(fig2, 1:n_plot, gap_linear;
          color=RGB(0.2, 0.4, 0.8), linewidth=2.0, fillalpha=0.15,
          fill=0, label="Cumulative gap: Linear")
end

if has_surrogate
    gap_surrogate = cumsum(results["Surrogate"].counterfactual_dy .- results["Surrogate"].actual_dy)
    plot!(fig2, 1:n_plot, gap_surrogate;
          color=RGB(0.8, 0.2, 0.2), linewidth=2.0, fillalpha=0.15,
          fill=0, label="Cumulative gap: Surrogate")
end

hline!(fig2, [0.0]; color=:gray50, linestyle=:dot, linewidth=0.8, label="")

plot!(fig2,
    xlabel  = "",
    ylabel  = "Cumulative GDP gap (pp, counterfactual - actual)",
    title   = "Cumulative COVID GDP Loss by Posterior",
    legend  = :topleft,
    xticks  = (collect(tick_positions), tick_labels),
    xrotation = 45,
)

figpath2 = joinpath(FIG_DIR, "covid_counterfactual_gdp_cumulative.pdf")
savefig(fig2, figpath2)
println("  Saved: $figpath2")

# ============================================================================
# Step 6: Comparison Table
# ============================================================================

println("\n" * "=" ^ 72)
println("COVID COUNTERFACTUAL GDP COMPARISON TABLE")
println("=" ^ 72)

println("\n  Methodology:")
println("    - Smoothed states/shocks via Kalman smoother at posterior mean")
println("    - Counterfactual: shocks zeroed in $(quarter_label(ZERO_START))-$(quarter_label(ZERO_END))")
println("    - Forward simulation from smoothed state at $(quarter_label(PERIOD_2019Q4))")
println()

@printf("  %-28s", "Metric")
has_linear && @printf("  %14s", "Linear")
has_surrogate && @printf("  %14s", "Surrogate")
println()
println("  " * "-" ^ (28 + (has_linear ? 16 : 0) + (has_surrogate ? 16 : 0)))

@printf("  %-28s", "Cumul. loss (2020Q1-2021Q2)")
has_linear && @printf("  %12.2f pp", results["Linear"].cumulative_loss)
has_surrogate && @printf("  %12.2f pp", results["Surrogate"].cumulative_loss)
println()

@printf("  %-28s", "Cumul. loss (2020Q1-2022Q4)")
has_linear && @printf("  %12.2f pp", results["Linear"].full_loss)
has_surrogate && @printf("  %12.2f pp", results["Surrogate"].full_loss)
println()

# Mean shock norm during COVID
@printf("  %-28s", "Mean shock norm (COVID)")
has_linear && @printf("  %14.3f", mean(results["Linear"].shock_norms))
has_surrogate && @printf("  %14.3f", mean(results["Surrogate"].shock_norms))
println()

# Peak shock (2020Q2)
@printf("  %-28s", "Shock norm at 2020Q2")
has_linear && @printf("  %14.3f", results["Linear"].shock_norms[2])
has_surrogate && @printf("  %14.3f", results["Surrogate"].shock_norms[2])
println()

# Per-quarter losses
println()
println("  Per-quarter GDP loss (counterfactual - actual, pp):")
@printf("  %-12s", "Quarter")
has_linear && @printf("  %10s", "Linear")
has_surrogate && @printf("  %10s", "Surrogate")
println()
println("  " * "-" ^ (12 + (has_linear ? 12 : 0) + (has_surrogate ? 12 : 0)))

covid_qs = collect(ZERO_START:ZERO_END)
for (i, p) in enumerate(covid_qs)
    @printf("  %-12s", quarter_label(p))
    has_linear && @printf("  %10.3f", results["Linear"].loss_per_quarter[i])
    has_surrogate && @printf("  %10.3f", results["Surrogate"].loss_per_quarter[i])
    println()
end

# Key insight summary
if has_linear && has_surrogate
    println("\n" * "-" ^ 72)
    println("  KEY FINDING:")
    println("  The linear posterior attributes a cumulative GDP loss of",
            @sprintf(" %.2f pp", results["Linear"].cumulative_loss),
            " during 2020Q1-2021Q2,")
    println("  while the surrogate posterior attributes",
            @sprintf(" %.2f pp", results["Surrogate"].cumulative_loss), ".")

    diff = results["Linear"].cumulative_loss - results["Surrogate"].cumulative_loss
    if diff > 0
        println("  The linear model attributes $(round(diff, digits=2)) pp MORE loss to",
                " exogenous shocks,")
        println("  consistent with larger estimated shock volatilities compensating for")
        println("  the absence of nonlinear endogenous propagation mechanisms.")
    else
        println("  The surrogate model attributes $(round(-diff, digits=2)) pp MORE loss to",
                " exogenous shocks.")
    end

    println("\n  Shock sizes at 2020Q2:")
    @printf("    Linear:    %.3f\n", results["Linear"].shock_norms[2])
    @printf("    Surrogate: %.3f\n", results["Surrogate"].shock_norms[2])
    ratio = results["Linear"].shock_norms[2] / max(results["Surrogate"].shock_norms[2], 1e-10)
    @printf("    Ratio (linear/surrogate): %.2f\n", ratio)
end

# ============================================================================
# Step 7: Print Parameter Comparison at Posterior Means (shock sigmas)
# ============================================================================

if has_linear && has_surrogate
    println("\n" * "=" ^ 72)
    println("SHOCK VOLATILITY COMPARISON AT POSTERIOR MEANS")
    println("=" ^ 72)

    sigma_params = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_epinf, :z_ew, :z_em]
    sigma_labels = ["TFP", "Risk premium", "Government", "Investment",
                    "Price markup", "Wage markup", "Monetary"]

    @printf("  %-16s %-16s %10s %10s %10s\n",
            "Parameter", "Description", "Linear", "Surrogate", "Ratio")
    println("  " * "-" ^ 64)

    for (j, p) in enumerate(sigma_params)
        pi = findfirst(==(p), theta_names)
        if pi !== nothing
            v_lin = theta_linear[pi]
            v_sur = theta_surrogate[pi]
            rat = v_lin / max(v_sur, 1e-10)
            @printf("  %-16s %-16s %10.4f %10.4f %10.2f\n",
                    p, sigma_labels[j], v_lin, v_sur, rat)
        end
    end
end

println("\n" * "=" ^ 72)
println("COVID COUNTERFACTUAL GDP ANALYSIS COMPLETE")
println("Finished: $(now())")
println("=" ^ 72)
