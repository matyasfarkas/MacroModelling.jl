using MacroModelling, AxisKeys, Printf

include("../models/Smets_Wouters_2007_HLT.jl")

m = Smets_Wouters_2007_HLT

shock = :epinf
shock_size = 1.0
sep_periods = 40
sep_order = 1
sep_nnodes = 3
sep_sparse_tree = true
sep_tol = 1e-8
sep_maxit = 300

function build_shock_sequence(m::MacroModelling.ℳ, shock::Symbol, shock_size::Float64, periods::Int)
    shock_idx = findfirst(==(shock), m.exo)
    shock_idx === nothing && error("Shock $shock not found in model")
    nshocks = length(m.exo)
    shock_sequence = zeros(periods, nshocks)
    shock_sequence[1, shock_idx] = shock_size
    return shock_sequence
end

function build_yss(m::MacroModelling.ℳ, SS_result)
    ss_keys = try
        axiskeys(SS_result, 1)
    catch
        Symbol[]
    end

    SS = m.solution.non_stochastic_steady_state
    yss = Float64[]
    for (i, var) in enumerate(m.var)
        if var in ss_keys
            push!(yss, Float64(SS_result(var)))
        else
            push!(yss, Float64(SS[i]))
        end
    end

    return yss
end

function sep_residual_summary(m::MacroModelling.ℳ, sep_sol::MacroModelling.sep_solution, deterministic_shocks::Union{Nothing,Matrix{Float64}}; top_n::Int=10)
    layout = sep_sol.layout
    T = sep_sol.periods
    Lbr = sep_sol.order
    ny = layout.ny_
    Y = sep_sol.Y

    dyn_resid_func, _, vars_raw, parameters_and_SS, resid_buffer, _ =
        MacroModelling.build_dynamic_residual_jacobian(m)

    params = m.parameter_values
    SS_result = MacroModelling.get_steady_state(m; parameters=params, return_variables_only=false, derivatives=false)
    yss = build_yss(m, SS_result)
    params_and_ss = MacroModelling.build_parameters_and_ss_values(parameters_and_SS, params, m, yss, SS_result)
    var_kind, var_idx = MacroModelling.build_dyn_var_maps(m, vars_raw)
    dyn_values = zeros(length(vars_raw))

    shock_names = m.exo
    obc_mask = contains.(string.(shock_names), "ᵒᵇᶜ")
    stochastic_idx = findall(x -> !x, obc_mask)
    dε = length(stochastic_idx)
    n_exo = length(shock_names)

    Σ = zeros(dε, dε)
    for (i, shock_pos) in enumerate(stochastic_idx)
        shock_name = shock_names[shock_pos]
        param_name = Symbol("z_", shock_name)
        param_idx = findfirst(==(param_name), m.parameters)
        σ = param_idx === nothing ? 1.0 : params[param_idx]
        Σ[i, i] = σ^2
    end

    nnodes = sep_sol.nnodes
    if layout.sparse
        X, W, _ = MacroModelling.build_sparse_shock_nodes(nnodes, dε, Σ)
        K = size(X, 2)
    else
        X, W = MacroModelling.gh_tensor_nodes_weights(nnodes, dε)
        MacroModelling.transform_nodes!(X, Σ)
        MacroModelling.reorder_nodes_zero_first!(X, W)
        K = size(X, 2)
    end

    eq_max = fill(0.0, ny)
    eq_t = fill(0, ny)
    eq_g = fill(0, ny)

    eps_det = zeros(n_exo)
    eps_full = zeros(n_exo)
    r_sum = zeros(ny)

    get_y = (t, g) -> (t >= T + 1 ? yss : view(Y, MacroModelling.index_y(layout, t, g)))

    for t in 1:T
        Gt = layout.G[t + 1]
        for g in 1:Gt
            pg = layout.sparse ? MacroModelling.parent_group_sparse(layout, t, g) : MacroModelling.parent_group(layout, t, g)
            yl = view(Y, MacroModelling.index_y(layout, t - 1, pg))
            yc = view(Y, MacroModelling.index_y(layout, t, g))
            cgs = layout.sparse ? MacroModelling.child_groups_sparse(layout, t, g) : MacroModelling.child_groups(layout, t, g)
            node_branches = layout.sparse ? ((t <= Lbr) && (g == 1) && (dε > 0)) : ((t <= Lbr) && (dε > 0))

            if node_branches
                fill!(r_sum, 0.0)
                for (kidx, cg) in enumerate(cgs)
                    yl1 = get_y(t + 1, cg)

                    if layout.sparse
                        shock_idx = kidx
                        ε_to_child = view(X, :, shock_idx)
                        wk = W[shock_idx]
                    else
                        k_shock = mod(g - 1, K) + 1
                        ε_to_child = view(X, :, k_shock)
                        wk = W[kidx]
                    end

                    if !isnothing(deterministic_shocks)
                        eps_det .= view(deterministic_shocks, t, :)
                        eps_full .= eps_det
                    else
                        fill!(eps_full, 0.0)
                    end

                    if t > 1 && dε > 0
                        eps_full[stochastic_idx] .= ε_to_child
                    end

                    MacroModelling.fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, eps_full)
                    dyn_resid_func(resid_buffer, params_and_ss, dyn_values)
                    @inbounds for i in 1:ny
                        r_sum[i] += wk * resid_buffer[i]
                    end
                end

                for i in 1:ny
                    val = abs(r_sum[i])
                    if val > eq_max[i]
                        eq_max[i] = val
                        eq_t[i] = t
                        eq_g[i] = g
                    end
                end
            else
                cg = first(cgs)
                yl1 = get_y(t + 1, cg)

                if layout.sparse && g > 1
                    info = MacroModelling.branch_info_sparse(layout, g)
                    if !isnothing(info) && t == info[1]
                        if !isnothing(deterministic_shocks)
                            eps_det .= view(deterministic_shocks, t, :)
                            eps_full .= eps_det
                        else
                            fill!(eps_full, 0.0)
                        end
                        if dε > 0
                            eps_full[stochastic_idx] .= view(X, :, info[2])
                        end
                    else
                        if !isnothing(deterministic_shocks)
                            eps_det .= view(deterministic_shocks, t, :)
                            eps_full .= eps_det
                        else
                            fill!(eps_full, 0.0)
                        end
                    end
                else
                    if !isnothing(deterministic_shocks)
                        eps_det .= view(deterministic_shocks, t, :)
                        eps_full .= eps_det
                    else
                        fill!(eps_full, 0.0)
                    end
                end

                MacroModelling.fill_dyn_values!(dyn_values, var_kind, var_idx, yl, yc, yl1, eps_full)
                dyn_resid_func(resid_buffer, params_and_ss, dyn_values)

                for i in 1:ny
                    val = abs(resid_buffer[i])
                    if val > eq_max[i]
                        eq_max[i] = val
                        eq_t[i] = t
                        eq_g[i] = g
                    end
                end
            end
        end
    end

    order = sortperm(eq_max, rev=true)
    println("Top residuals:")
    for idx in order[1:min(top_n, length(order))]
        eq = m.dyn_equations[idx]
        @printf("  eq %3d  max|res|=%0.6g  t=%d g=%d  %s\n", idx, eq_max[idx], eq_t[idx], eq_g[idx], eq)
    end
    @printf("Global max residual: %0.6g\n", maximum(eq_max))

    return (eq_max = eq_max, eq_t = eq_t, eq_g = eq_g)
end

shock_sequence = build_shock_sequence(m, shock, shock_size, sep_periods)
sss_state = MacroModelling.sep_irf_stochastic_state(m, :second_order; silent = true)

println("Running SEP solve with tightened tolerance...")
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = sep_periods,
       sep_order = sep_order,
       sep_nnodes = sep_nnodes,
       sep_maxit = sep_maxit,
       sep_tol = sep_tol,
       sep_sparse_tree = sep_sparse_tree,
       sep_initial_state = sss_state,
       sep_deterministic_shocks = shock_sequence,
       silent = true)

sep_sol = m.solution.perturbation.stochastic_extended_path
println("SEP convergence flag: ", sep_sol.convergence_flag)
println("SEP final error: ", sep_sol.final_error)

println("\nResidual summary (shocked path):")
sep_residual_summary(m, sep_sol, shock_sequence; top_n = 12)

println("\nResidual summary (zero-shock path):")
zero_sequence = zeros(sep_periods, length(m.exo))
solve!(m,
       algorithm = :stochastic_extended_path,
       sep_periods = sep_periods,
       sep_order = sep_order,
       sep_nnodes = sep_nnodes,
       sep_maxit = sep_maxit,
       sep_tol = sep_tol,
       sep_sparse_tree = sep_sparse_tree,
       sep_initial_state = sss_state,
       sep_deterministic_shocks = zero_sequence,
       silent = true)

sep_sol_zero = m.solution.perturbation.stochastic_extended_path
println("SEP convergence flag (zero shock): ", sep_sol_zero.convergence_flag)
println("SEP final error (zero shock): ", sep_sol_zero.final_error)
sep_residual_summary(m, sep_sol_zero, zero_sequence; top_n = 12)
