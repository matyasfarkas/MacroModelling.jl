using Test
using MacroModelling
using Random
using Statistics
using Serialization
using AxisKeys
using MCMCChains

@testset "Regime Switching API" begin
    @testset "Gate Calibration and Assignment" begin
        Random.seed!(1234)
        e = abs.(randn(2000))
        f = abs.(randn(2000))
        res = calibrate_gate(e, f; config = GateCalibrationConfig(target_share = 0.2))
        @test 0 < res.quantile < 1
        @test res.tau_eps > 0
        @test res.tau_y > 0
        @test abs(res.achieved_share - 0.2) < 0.05

        mask = assign_regimes(e, f, res.tau_eps, res.tau_y; k_pre = 2, k_post = 2, min_len = 3)
        @test mask isa BitVector
        @test length(mask) == length(e)
        @test sum(mask) >= round(Int, res.achieved_share * length(e))

        fixed_tau_eps = quantile(e, 0.8)
        tau_y, share_y = MacroModelling.calibrate_tau_y(e, f, fixed_tau_eps, 0.2)
        @test isfinite(tau_y)
        @test abs(share_y - 0.2) < 0.05
        @test MacroModelling.gate_share(e, f, fixed_tau_eps, tau_y) ≈ share_y atol = 1e-8

        fixed_tau_y = quantile(f, 0.75)
        tau_eps, share_e = MacroModelling.calibrate_tau_eps(e, f, fixed_tau_y, 0.2)
        @test isfinite(tau_eps)
        @test abs(share_e - 0.2) < 0.05
        @test MacroModelling.gate_share(e, f, tau_eps, fixed_tau_y) ≈ share_e atol = 1e-8

        obs_data = [2.0 3.0; 4.0 7.0]
        lin_obs = [1.0 1.0; 2.0 3.0]
        shocks = [1.0 2.0; 5.0 6.0; 2.0 4.0]
        obs_sigma = [1.0, 2.0]
        shock_sigmas = [1.0, 0.0, 2.0]
        e_stat, f_stat = MacroModelling.compute_gate_stat_series(
            obs_data,
            lin_obs,
            shocks,
            obs_sigma,
            shock_sigmas;
            shock_norm = :linf,
            error_norm = :l2,
        )
        @test e_stat ≈ [1.0, 2.0]
        @test f_stat ≈ [sqrt(2.0), 2sqrt(2.0)]

        # Simulate get_irf ordering (sorted variable order) and extra lead period.
        lin2d = [
            10.0 11.0 12.0 13.0;
            20.0 21.0 22.0 23.0;
            30.0 31.0 32.0 33.0;
        ]
        lin_irf_like = reshape(lin2d, 3, 4, 1)
        aligned = MacroModelling.align_linear_observable_path(
            lin_irf_like,
            [:a, :c],
            [:b, :c, :a];
            T = 3,
            periods = 1,
            model_name = "Toy",
            label = "Toy linear sim",
        )
        @test size(aligned) == (2, 3)
        @test aligned[1, :] == [20.0, 21.0, 22.0]  # :a maps to second sorted row then trimmed
        @test aligned[2, :] == [10.0, 11.0, 12.0]  # :c maps to first sorted row then trimmed

        struct ToyGateTimings
            var::Vector{Symbol}
        end
        struct ToyGateModel
            model_name::String
            timings::ToyGateTimings
        end
        function MacroModelling.get_irf(model::ToyGateModel; kwargs...)
            @test haskey(kwargs, :shocks)
            @test haskey(kwargs, :variables)
            @test haskey(kwargs, :periods)
            @test haskey(kwargs, :parameters)
            @test haskey(kwargs, :initial_state)
            T = size(kwargs[:shocks], 2)
            periods = kwargs[:periods]
            # Rows correspond to sorted requested variable indices (:c, :a), helper reorders to (:a, :c).
            out = [
                1.0 2.0 3.0;
                3.0 4.0 5.0;
            ]
            @test size(out, 2) == T + periods
            return reshape(out, size(out, 1), size(out, 2), 1)
        end
        toy_model = ToyGateModel("ToyGate", ToyGateTimings([:b, :c, :a]))
        toy_obs = [4.0 8.0; 1.0 0.0]
        toy_shocks = [2.0 4.0; 7.0 9.0; 3.0 6.0]
        toy_obs_sigma = [1.0, 2.0]
        toy_shock_sigmas = [2.0, 0.0, 3.0]
        function MacroModelling.get_estimated_shocks(model::ToyGateModel, data; kwargs...)
            @test kwargs[:algorithm] == :first_order
            @test kwargs[:filter] == :kalman
            @test kwargs[:data_in_levels] == true
            @test kwargs[:smooth] == false
            @test kwargs[:verbose] == false
            @test kwargs[:parameters] == [1.0]
            return [2.0 4.0; 7.0 9.0; 3.0 6.0]
        end
        shocks_est2 = MacroModelling.estimate_observed_shocks_matrix(
            toy_model,
            toy_obs,
            [:a, :c];
            parameters = [1.0],
            filter = :kalman,
            expected_rows = 3,
            expected_cols = 2,
            label = "Toy shocks",
        )
        @test shocks_est2 == toy_shocks
        @test_throws ErrorException MacroModelling.estimate_observed_shocks_matrix(
            toy_model, toy_obs, [:a, :c]; parameters = [1.0], expected_rows = 2)

        function MacroModelling.get_estimated_variables(model::ToyGateModel, data; kwargs...)
            @test kwargs[:algorithm] == :first_order
            @test kwargs[:filter] == :inversion
            @test kwargs[:data_in_levels] == true
            @test kwargs[:levels] == true
            @test kwargs[:smooth] == false
            @test kwargs[:verbose] == false
            @test kwargs[:parameters] == [2.0]
            arr = [10.0 11.0; 20.0 21.0; 30.0 31.0]
            return KeyedArray(arr; Variable = [:c, :b, :a], Time = [1, 2])
        end
        vars_est2, var_names2 = MacroModelling.estimate_observed_variables_matrix(
            toy_model,
            toy_obs,
            [:a, :c];
            parameters = [2.0],
            filter = :inversion,
            expected_rows = 3,
            expected_cols = 2,
            label = "Toy vars",
        )
        @test vars_est2 == [10.0 11.0; 20.0 21.0; 30.0 31.0]
        @test var_names2 == [:c, :b, :a]
        @test_throws ErrorException MacroModelling.estimate_observed_variables_matrix(
            toy_model, toy_obs, [:a, :c]; parameters = [2.0], filter = :inversion, expected_cols = 3)

        base_params = [10.0, 20.0, 30.0]
        model_params = [:p1, :p2, :p3]
        theta_names = [:p3, :p1]
        theta_vals = [3.5, 1.5]
        @test MacroModelling.extract_named_parameters(base_params, model_params, theta_names) == [30.0, 10.0]
        @test MacroModelling.override_named_parameters(base_params, model_params, theta_names, theta_vals) == [1.5, 20.0, 3.5]
        @test MacroModelling.parameters_with_theta_mode(base_params, model_params, theta_names, theta_vals; theta_mode = :synthetic) == [1.5, 20.0, 3.5]
        @test MacroModelling.parameters_with_theta_mode(base_params, model_params, theta_names, theta_vals; theta_mode = :baseline) == base_params
        @test_throws ErrorException MacroModelling.override_named_parameters(base_params, model_params, theta_names, [1.0])
        @test_throws ErrorException MacroModelling.parameters_with_theta_mode(base_params, model_params, theta_names, nothing; theta_mode = :synthetic)
        @test_throws ErrorException MacroModelling.extract_named_parameters(base_params, model_params, [:missing])

        function MacroModelling.get_loglikelihood_per_period(model::ToyGateModel, data, params; kwargs...)
            @test params == [1.5, 20.0, 3.5]
            @test kwargs[:algorithm] == :first_order
            @test kwargs[:filter] == :kalman
            @test kwargs[:on_failure_loglikelihood] == -99.0
            @test kwargs[:presample_periods] == 0
            @test kwargs[:initial_covariance] == :theoretical
            @test kwargs[:verbose] == false
            return [-1.0, -2.0]
        end
        ll_lin = MacroModelling.linear_model_loglik_per_period(
            toy_model,
            KeyedArray(randn(2, 2); Variable = [:a, :c], Time = [1, 2]),
            theta_vals,
            theta_names;
            model_parameter_names = model_params,
            base_parameters = base_params,
            on_failure_loglikelihood = -99.0,
        )
        @test ll_lin == [-1.0, -2.0]
        ll_lin_cached_idx = MacroModelling.linear_model_loglik_per_period(
            toy_model,
            KeyedArray(randn(2, 2); Variable = [:a, :c], Time = [1, 2]),
            theta_vals,
            theta_names;
            model_parameter_names = model_params,
            base_parameters = base_params,
            theta_idx = [3, 1],
            on_failure_loglikelihood = -99.0,
        )
        @test ll_lin_cached_idx == [-1.0, -2.0]
        @test_throws ErrorException MacroModelling.linear_model_loglik_per_period(
            toy_model,
            KeyedArray(randn(2, 2); Variable = [:a, :c], Time = [1, 2]),
            theta_vals,
            theta_names;
            model_parameter_names = model_params,
            base_parameters = base_params,
            theta_idx = [4, 1],
            on_failure_loglikelihood = -99.0,
        )

        function predict_toy(state, shocks, θ)
            obs_pred = [state[1] + shocks[1] + θ[1]]
            state_next = [0.8 * state[1] + 0.2 * shocks[1]]
            return obs_pred, state_next
        end
        ll_cond = MacroModelling.conditional_loglik_per_period(
            (state, shock_t, θ_local) -> ([state[1] + shock_t[1] + θ_local[1]], [state[1] + shock_t[1]]),
            [0.0],
            [1.0 0.0],
            [0.0],
            [1.0 1.0],
            [1.0],
        )
        @test size(ll_cond) == (2,)
        @test ll_cond[1] ≈ -0.5 * log(2 * pi) atol = 1e-8
        @test ll_cond[2] ≈ -0.5 * log(2 * pi) atol = 1e-8

        @test MacroModelling.split_observation_state([1.0, 2.0, 3.0], 2) == ([1.0, 2.0], [3.0])
        @test_throws ErrorException MacroModelling.split_observation_state([1.0], 2)

        full_predict = (state, shock_t, θ_local) -> [state[1] + shock_t[1] + θ_local[1], state[1] + shock_t[1]]
        obs_split, state_split = MacroModelling.predict_from_full(full_predict, [0.0], [1.0], [0.0], 1)
        @test obs_split == [1.0]
        @test state_split == [1.0]

        residual_obs = (state, shock_t, θ_local) -> [0.5]
        obs_resid, state_resid = MacroModelling.predict_additive_residual(
            full_predict,
            residual_obs,
            [0.0],
            [1.0],
            [0.0],
            1;
            allow_full_residual = true,
        )
        @test obs_resid == [1.5]
        @test state_resid == [1.0]

        residual_full = (state, shock_t, θ_local) -> [0.25, -0.5]
        obs_resid_full, state_resid_full = MacroModelling.predict_additive_residual(
            full_predict,
            residual_full,
            [0.0],
            [1.0],
            [0.0],
            1;
            allow_full_residual = true,
        )
        @test obs_resid_full == [1.25]
        @test state_resid_full == [0.5]
        @test_throws ErrorException MacroModelling.predict_additive_residual(
            full_predict,
            residual_full,
            [0.0],
            [1.0],
            [0.0],
            1;
            allow_full_residual = false,
        )
        @test_throws ErrorException MacroModelling.predict_additive_residual(
            full_predict,
            (state, shock_t, θ_local) -> [0.1, 0.2, 0.3],
            [0.0],
            [1.0],
            [0.0],
            1;
            allow_full_residual = true,
        )

        ll_additive = MacroModelling.additive_residual_loglik_per_period(
            full_predict,
            residual_obs,
            [0.0],
            [1.0 0.0],
            [0.0],
            [1.5 1.5],
            [1.0];
            d_obs = 1,
            allow_full_residual = true,
        )
        @test size(ll_additive) == (2,)
        @test ll_additive[1] ≈ -0.5 * log(2 * pi) atol = 1e-8
        @test ll_additive[2] ≈ -0.5 * log(2 * pi) atol = 1e-8

        obs_roll = MacroModelling.rollout_observations(
            (state, shock_t, θ_local) -> ([state[1] + shock_t[1] + θ_local[1]], [state[1] + shock_t[1]]),
            [0.0],
            [1.0 0.0],
            [0.0],
        )
        @test size(obs_roll) == (1, 2)
        @test vec(obs_roll) ≈ [1.0, 1.0]
        @test_throws ErrorException MacroModelling.rollout_observations(
            (state, shock_t, θ_local) -> begin
                if shock_t[1] > 0.5
                    return [state[1]], [state[1]]
                end
                return [state[1], 0.0], [state[1]]
            end,
            [0.0],
            [1.0 0.0],
            [0.0],
        )
        @test_throws ErrorException MacroModelling.rollout_observations(
            (state, shock_t, θ_local) -> ([NaN], [state[1]]),
            [0.0],
            [1.0 0.0],
            [0.0];
            check_finite = true,
        )
        state_adv = MacroModelling.advance_state(
            (state, shock_t, θ_local) -> ([state[1] + shock_t[1] + θ_local[1]], [state[1] + shock_t[1]]),
            [0.0],
            [1.0 0.0],
            [0.0],
            2,
        )
        @test state_adv ≈ [1.0]
        @test_throws ErrorException MacroModelling.advance_state(
            (state, shock_t, θ_local) -> ([state[1]], [state[1]]),
            [0.0],
            [1.0 0.0],
            [0.0],
            3,
        )

        eps_step, state_next_step, ll_step = MacroModelling.inversion_step(
            predict_toy,
            [0.0],
            [1.0],
            [0.0],
            [0.1],
            [0.5, 0.0],
            [1];
            maxit = 12,
            tol = 1e-8,
            lambda = 1e-6,
        )
        @test size(eps_step) == (2,)
        @test eps_step[2] == 0.0
        @test isfinite(ll_step)
        @test isfinite(state_next_step[1])

        ll_inv, shocks_inv = MacroModelling.inversion_loglik_per_period(
            predict_toy,
            [0.0],
            [0.0],
            [1.0 0.5],
            [0.1],
            [0.5, 0.0];
            maxit = 12,
            tol = 1e-8,
            lambda = 1e-6,
        )
        @test size(ll_inv) == (2,)
        @test size(shocks_inv) == (2, 2)
        @test all(isfinite, ll_inv)
        @test all(isfinite, shocks_inv)

        batch_called = Ref(false)
        ll_batch, _ = MacroModelling.inversion_loglik_per_period(
            predict_toy,
            [0.0],
            [0.0],
            [1.0 0.5],
            [0.1],
            [0.5, 0.0];
            batch_eval_residual_fn = X_nn -> begin
                batch_called[] = true
                fill(0.5, 1, size(X_nn, 2))
            end,
            maxit = 12,
            tol = 1e-8,
            lambda = 1e-6,
        )
        @test batch_called[]
        @test size(ll_batch) == size(ll_inv)
        @test sum(ll_batch) < sum(ll_inv)

        ll_lin_sampling = MacroModelling.linear_reference_loglik_per_period(
            [0.0],
            [0.0],
            [1.0 0.0],
            [1.0 1.0],
            [1.0],
            [0.5, 0.0];
            shock_filter = :sampling,
            linear_filter = :kalman,
            predict_linear = (state, shock_t, θ_local) -> ([state[1] + shock_t[1] + θ_local[1]], [state[1] + shock_t[1]]),
            kalman_linear_loglik = θ_local -> fill(-99.0, 2),
        )
        @test ll_lin_sampling ≈ ll_cond

        kalman_called = Ref(false)
        ll_lin_kalman = MacroModelling.linear_reference_loglik_per_period(
            [0.0],
            [0.0],
            zeros(2, 2),
            [1.0 0.5],
            [0.1],
            [0.5, 0.0];
            shock_filter = :inversion,
            linear_filter = :kalman,
            predict_linear = predict_toy,
            kalman_linear_loglik = θ_local -> begin
                kalman_called[] = true
                return [-3.0, -4.0]
            end,
        )
        @test kalman_called[]
        @test ll_lin_kalman == [-3.0, -4.0]

        ll_lin_inversion = MacroModelling.linear_reference_loglik_per_period(
            [0.0],
            [0.0],
            zeros(2, 2),
            [1.0 0.5],
            [0.1],
            [0.5, 0.0];
            shock_filter = :inversion,
            linear_filter = :inversion,
            predict_linear = predict_toy,
            kalman_linear_loglik = θ_local -> fill(-99.0, 2),
            inversion_maxit = 12,
            inversion_tol = 1e-8,
            inversion_lambda = 1e-6,
        )
        @test ll_lin_inversion ≈ ll_inv atol = 1e-8
        @test_throws ErrorException MacroModelling.linear_reference_loglik_per_period(
            [0.0], [0.0], zeros(2, 2), [1.0 0.5], [0.1], [0.5, 0.0];
            shock_filter = :inversion,
            linear_filter = :unknown,
            predict_linear = predict_toy,
        )

        eps_nostruct, _, ll_nostruct = MacroModelling.inversion_step(
            predict_toy,
            [0.0],
            [1.0],
            [0.0],
            [0.1],
            [0.0, 0.0],
            Int[];
            maxit = 2,
        )
        @test eps_nostruct == [0.0, 0.0]
        @test isfinite(ll_nostruct)

        eps_mean = [1.0 -1.0; 0.25 0.5]
        shock_sigmas_eps = [0.5, 0.0, 2.0]
        shocks_from_eps = MacroModelling.build_shocks_from_eps(eps_mean, shock_sigmas_eps, nothing)
        @test size(shocks_from_eps) == (3, 2)
        @test shocks_from_eps[1, :] ≈ [0.5, -0.5]
        @test shocks_from_eps[2, :] == [0.0, 0.0]
        @test shocks_from_eps[3, :] ≈ [0.5, 1.0]

        guided_base = fill(1.0, 3, 5)
        shocks_sub = MacroModelling.build_shocks_from_eps(
            eps_mean,
            shock_sigmas_eps,
            guided_base;
            sample_idx = [2, 4],
            T_full = 5,
        )
        @test size(shocks_sub) == (3, 5)
        @test shocks_sub[:, 1] == [1.0, 1.0, 1.0]
        @test shocks_sub[1, 2] ≈ 1.5
        @test shocks_sub[3, 2] ≈ 1.5
        @test shocks_sub[1, 4] ≈ 0.5
        @test shocks_sub[3, 4] ≈ 2.0
        @test shocks_sub[:, 5] == [1.0, 1.0, 1.0]

        @test_throws ErrorException MacroModelling.build_shocks_from_eps(randn(3, 2), shock_sigmas_eps, nothing)
        @test_throws ErrorException MacroModelling.build_shocks_from_eps(eps_mean, shock_sigmas_eps, guided_base; sample_idx = [2, 6], T_full = 5)

        chain_names = [:cprobp, Symbol("ε[1,1]"), Symbol("ε[2,1]"), Symbol("ε[1,2]")]
        chain_arr = zeros(2, length(chain_names), 1)
        chain_arr[:, 1, 1] .= [0.6, 0.7]
        chain_arr[:, 2, 1] .= [1.0, 3.0]
        chain_arr[:, 3, 1] .= [2.0, 4.0]
        chain_arr[:, 4, 1] .= [5.0, 7.0]
        toy_chain = Chains(chain_arr, chain_names)
        θdraws = MacroModelling.theta_draws(toy_chain, [:cprobp])
        @test size(θdraws) == (2, 1)
        @test vec(θdraws[:, 1]) == [0.6, 0.7]
        theta_vec_chain = Chains(chain_arr[:, 1:1, :], [Symbol("theta_vec[1]")])
        θdraws_vec = MacroModelling.theta_draws(theta_vec_chain, [:cprobp])
        @test size(θdraws_vec) == (2, 1)
        @test vec(θdraws_vec[:, 1]) == [0.6, 0.7]
        eps_means = MacroModelling.epsilon_means_from_chain(toy_chain)
        @test size(eps_means) == (2, 2)
        @test eps_means[1, 1] ≈ 2.0
        @test eps_means[2, 1] ≈ 3.0
        @test eps_means[1, 2] ≈ 6.0
        @test isnan(eps_means[2, 2])
        eps_means_sub = MacroModelling.epsilon_means_from_chain(toy_chain; sample_idx = [4])
        @test size(eps_means_sub) == (2, 1)
        @test eps_means_sub[1, 1] ≈ 2.0

        lin_obs2, e_stat2, f_stat2 = MacroModelling.compute_linear_gate_stats_from_shocks(
            toy_model,
            toy_obs,
            [:a, :c],
            toy_shocks,
            toy_obs_sigma,
            toy_shock_sigmas;
            periods = 1,
            parameters = [1.0],
            initial_state = [0.0],
            shock_norm = :linf,
            error_norm = :l2,
        )
        @test lin_obs2 == [3.0 4.0; 1.0 2.0]
        @test e_stat2 == [1.0, 2.0]
        @test f_stat2 ≈ [1.0, sqrt(17.0)]

        state_last = MacroModelling.linear_filter_initial_state(
            toy_model,
            toy_obs,
            [:a, :c],
            [:b, :a];
            parameters = [2.0],
            filter = :inversion,
        )
        @test state_last == [21.0, 31.0]

        state_full0 = MacroModelling.linear_filter_full_state_initial(
            toy_model,
            toy_obs,
            [:a, :c];
            parameters = [2.0],
            filter = :inversion,
        )
        @test state_full0 == [10.0, 20.0, 30.0]

        struct ToyGateModelFilter
            model_name::String
            timings::ToyGateTimings
        end
        function MacroModelling.get_estimated_shocks(model::ToyGateModelFilter, data; kwargs...)
            @test kwargs[:algorithm] == :first_order
            @test kwargs[:filter] == :kalman
            @test kwargs[:data_in_levels] == true
            @test kwargs[:smooth] == false
            @test kwargs[:verbose] == false
            @test kwargs[:parameters] == [1.0]
            return toy_shocks
        end
        function MacroModelling.get_estimated_variables(model::ToyGateModelFilter, data; kwargs...)
            @test kwargs[:algorithm] == :first_order
            @test kwargs[:filter] == :kalman
            @test kwargs[:data_in_levels] == true
            @test kwargs[:levels] == true
            @test kwargs[:smooth] == false
            @test kwargs[:verbose] == false
            @test kwargs[:parameters] == [1.0]
            arr = [10.0 11.0; 20.0 21.0; 30.0 31.0]
            return KeyedArray(arr; Variable = [:c, :b, :a], Time = [1, 2])
        end
        function MacroModelling.get_irf(model::ToyGateModelFilter; kwargs...)
            T = size(kwargs[:shocks], 2)
            periods = kwargs[:periods]
            out = [
                1.0 2.0 3.0;
                3.0 4.0 5.0;
            ]
            @test size(out, 2) == T + periods
            return reshape(out, size(out, 1), size(out, 2), 1)
        end
        toy_model_filter = ToyGateModelFilter("ToyGateFilter", ToyGateTimings([:b, :c, :a]))
        lin_obs3, shocks3, e_stat3, f_stat3 = MacroModelling.compute_linear_gate_stats_from_filter(
            toy_model_filter,
            toy_obs,
            [:a, :c],
            toy_obs_sigma,
            toy_shock_sigmas,
            [:b, :a];
            periods = 1,
            parameters = [1.0],
            filter = :kalman,
            shock_norm = :linf,
            error_norm = :l2,
        )
        @test lin_obs3 == lin_obs2
        @test shocks3 == toy_shocks
        @test e_stat3 == e_stat2
        @test f_stat3 == f_stat2
    end

    @testset "Gate Bias and Probabilities" begin
        Random.seed!(99)
        scores = randn(300)
        target = 0.15
        bias = calibrate_gate_bias(scores, target)
        probs = 1 ./(1 .+ exp.(-(bias .+ scores)))
        @test abs(mean(probs) - target) < 0.03

        cfg = RegimeSwitchConfig(gate_mode = :soft, tau_eps = 1.0, tau_y = 1.0, beta_eps = 2.0, beta_y = 2.0)
        e = abs.(randn(100))
        f = abs.(randn(100))
        gp = gate_probabilities(e, f, cfg)
        @test length(gp) == 100
        @test all(0 .< gp .< 1)
    end

    @testset "Switching Likelihood" begin
        ll_rom = collect(range(-3.0, -1.0; length = 10))
        ll_fom = ll_rom .+ 0.3
        hard_mask = BitVector([i % 2 == 0 for i in 1:10])

        hard_res = compute_switching_loglikelihood(ll_rom, ll_fom; hard_mask = hard_mask)
        @test hard_res.total ≈ sum([hard_mask[i] ? ll_fom[i] : ll_rom[i] for i in eachindex(ll_rom)])
        @test length(hard_res.per_period) == 10

        probs = fill(0.25, 10)
        soft_res = compute_switching_loglikelihood(
            ll_rom,
            ll_fom;
            gate_probs = probs,
            config = SwitchingLikelihoodConfig(gate_mode = :soft, soft_mixture = :logsumexp),
        )
        @test isfinite(soft_res.total)
        @test length(soft_res.per_period) == 10
        soft_total = mix_loglikelihood(
            ll_fom,
            ll_rom,
            probs;
            config = SwitchingLikelihoodConfig(gate_mode = :soft, soft_mixture = :logsumexp),
        )
        @test soft_total ≈ soft_res.total

        cmp = evaluate_switching_vs_fom(hard_res.total, sum(ll_fom))
        @test haskey(cmp, :total_diff)
        @test haskey(cmp, :rmse)
    end

    @testset "Diagnostics" begin
        mask = BitVector([false, true, true, false, true, false, false, true, true, true])
        stats = compute_gate_stats(mask)
        @test stats["periods_total"] == 10
        @test stats["periods_nonlinear"] == 6
        @test stats["episodes"] == 3

        runs = MacroModelling.contiguous_true_runs(mask)
        @test runs == [2:3, 5:5, 8:10]
        @test MacroModelling.choose_gated_run(runs, :first) == 2:3
        @test MacroModelling.choose_gated_run(runs, :last) == 8:10
        @test MacroModelling.choose_gated_run(runs, :longest) == 8:10

        selected_idx, eval_idx, ctx_idx, note =
            MacroModelling.select_gated_block_periods(mask, :first, 1, 2)
        @test selected_idx == [1, 2, 3]
        @test eval_idx == [2, 3]
        @test ctx_idx == [1]
        @test occursin("Selected first block 2:3", note)
        @test occursin("context 1:1", note)

        selected_empty, eval_empty, ctx_empty, note_empty =
            MacroModelling.select_gated_block_periods(falses(5), :first, 1, 1)
        @test isempty(selected_empty)
        @test isempty(eval_empty)
        @test isempty(ctx_empty)
        @test note_empty == "No gated periods found."

        overlap = episode_overlap(mask, 2, 6)
        @test overlap["window_periods"] == 5
        @test 0 <= overlap["share_window_nonlinear"] <= 1

        ll_rom = -ones(10)
        ll_fom = -0.5 .* ones(10)
        decomp = summarize_loglik_decomposition(ll_rom, ll_fom, mask)
        @test decomp["periods_nonlinear"] == 6
        @test decomp["ll_mixed_total"] ≈ sum([mask[i] ? ll_fom[i] : ll_rom[i] for i in 1:10])

        struct ToyChunkChain
            info::Dict{Symbol,Any}
        end
        toy_chunk = ToyChunkChain(Dict(
            :internals => Dict(
                :avg_acceptance_rate => 0.87,
                :count_divergences => 2,
                :step_size => 0.015,
            ),
        ))
        @test MacroModelling.chunk_stats(toy_chunk) == (0.87, 2, 0.015)
        @test MacroModelling.chunk_stats(ToyChunkChain(Dict{Symbol,Any}())) == (nothing, nothing, nothing)

        callback_rows = Tuple{Int,Int,Int,Int,Int}[]
        chunked = MacroModelling.run_chunked_sampling(
            5,
            2;
            sample_chunk = (n_i, i, _n_chunks) -> fill(i, n_i),
            concat_chunks = (a, b) -> vcat(a, b),
            on_chunk = (i, n_chunks, n_i, chunk, samps, _elapsed) ->
                push!(callback_rows, (i, n_chunks, n_i, length(chunk), length(samps))),
        )
        @test chunked == [1, 1, 2, 2, 3]
        @test callback_rows == [(1, 3, 2, 2, 2), (2, 3, 2, 2, 4), (3, 3, 1, 1, 5)]
        @test_throws ErrorException MacroModelling.run_chunked_sampling(0, 1; sample_chunk = (n, i, t) -> [n], concat_chunks = vcat)
        @test_throws ErrorException MacroModelling.run_chunked_sampling(1, 0; sample_chunk = (n, i, t) -> [n], concat_chunks = vcat)
    end

    @testset "IO Loaders" begin
        mktempdir() do d
            surrogate_path = joinpath(d, "surrogate.jls")
            serialize(surrogate_path, Dict("frozen" => Dict("dummy" => true), "meta" => Dict("rom_residual" => true), "validation_rmse" => 0.12))
            bundle = load_hlt_surrogate_bundle(surrogate_path)
            @test bundle.path == surrogate_path
            @test bundle.meta["rom_residual"] == true

            synthetic_path = joinpath(d, "synthetic.jls")
            serialize(synthetic_path, Dict(
                "obs_data" => randn(2, 5),
                "s0" => randn(3),
                "shocks" => randn(2, 5),
                "theta_true" => randn(3),
                "observables" => [:a, :b],
                "theta_names" => [:p1, :p2, :p3],
            ))
            syn = load_hlt_synthetic_scenario(synthetic_path)
            @test haskey(syn, "obs_data")
            @test size(syn["shocks"], 2) == 5

            gate_path = joinpath(d, "gate_calibration.jls")
            serialize(gate_path, Dict(
                "tau_eps" => 1.2,
                "tau_y" => 3.4,
                "achieved_share" => 0.25,
            ))
            gate = load_hlt_gate_calibration(gate_path)
            @test gate["tau_eps"] == 1.2
            @test gate["tau_y"] == 3.4

            chain_dict_path = joinpath(d, "chain_payload.jls")
            serialize(chain_dict_path, Dict("chain" => [1.0, 2.0], "post_mean_theta" => [0.1, 0.2, 0.3]))
            chain_payload = load_hlt_chain_payload(chain_dict_path)
            @test haskey(chain_payload, "chain")
            @test chain_payload["post_mean_theta"] == [0.1, 0.2, 0.3]

            chain_raw_path = joinpath(d, "chain_raw.jls")
            serialize(chain_raw_path, [1, 2, 3])
            chain_payload_wrapped = load_hlt_chain_payload(chain_raw_path)
            @test chain_payload_wrapped["chain"] == [1, 2, 3]

            chain_summary_path = joinpath(d, "chain_summary.jls")
            serialize(chain_summary_path, Dict("summary_source_chain_path" => "foo.jls", "post_mean_theta" => [0.4, 0.5, 0.6]))
            chain_summary = load_hlt_chain_summary(chain_summary_path)
            @test chain_summary["summary_source_chain_path"] == "foo.jls"

            bench_chain_path = joinpath(d, "bench_chain.jls")
            bench_chain_summary_path = MacroModelling.hlt_benchmark_chain_summary_path(bench_chain_path)
            @test bench_chain_summary_path == joinpath(d, "bench_chain_summary.jls")
            @test "gate_mask" in MacroModelling.hlt_benchmark_chain_summary_keys()

            heavy_bench_payload = Dict(
                "chain" => [1, 2, 3],
                "theta_true" => [0.1, 0.2, 0.3],
                "post_mean_theta" => [0.11, 0.19, 0.31],
                "gate_mask" => [true, false, true],
                "gate_share" => 2 / 3,
                "extra_large_blob" => randn(20, 20),
            )
            serialize(bench_chain_path, heavy_bench_payload)
            extracted = MacroModelling.extract_hlt_benchmark_chain_summary(heavy_bench_payload)
            @test haskey(extracted, "theta_true")
            @test haskey(extracted, "post_mean_theta")
            @test haskey(extracted, "gate_mask")
            @test !haskey(extracted, "extra_large_blob")

            ensure_calls = Ref(0)
            bench_payload, bench_summary_cached_path, bench_used_summary = MacroModelling.load_hlt_chain_payload_for_benchmark(
                bench_chain_path;
                prefer_summary = false,
                build_summary_cache = true,
                ensure_deserialize_modules! = () -> (ensure_calls[] += 1),
            )
            @test ensure_calls[] == 1
            @test bench_used_summary == false
            @test bench_summary_cached_path == bench_chain_summary_path
            @test isfile(bench_chain_summary_path)
            @test haskey(bench_payload, "theta_true")
            @test !haskey(bench_payload, "extra_large_blob")

            ensure_calls[] = 0
            bench_payload_cached, _, bench_used_summary_cached = MacroModelling.load_hlt_chain_payload_for_benchmark(
                bench_chain_path;
                prefer_summary = true,
                build_summary_cache = true,
                ensure_deserialize_modules! = () -> (ensure_calls[] += 1),
            )
            @test bench_used_summary_cached == true
            @test ensure_calls[] == 0
            @test bench_payload_cached["theta_true"] == bench_payload["theta_true"]

            validated = validate_hlt_chain_payload(Dict(
                "chain" => [1, 2, 3],
                "post_mean_theta" => [0.1, 0.2, 0.3],
                "gate_mask" => [true, false, true],
            ); require_chain = true, gate_mask_key = "gate_mask", require_gate_mask = true, sample_length = 3)
            @test validated["chain"] == [1, 2, 3]
            @test_throws ErrorException validate_hlt_chain_payload(Dict("chain" => [1], "post_mean_theta" => 0.5))
            @test_throws ErrorException validate_hlt_chain_payload(Dict("chain" => [1], "gate_mask" => [1, 0]); gate_mask_key = "gate_mask", sample_length = 3)

            dataset_path = joinpath(d, "dataset.jls")
            serialize(dataset_path, Dict("X" => randn(3, 2), "Y" => randn(2, 2)))
            ds = load_hlt_dataset_payload(dataset_path)
            @test size(ds["X"]) == (3, 2)

            fom_path = joinpath(d, "fom.jls")
            serialize(fom_path, Dict("results" => Dict("true" => Dict("status" => "ok"))))
            fom = load_hlt_fom_benchmark_payload(fom_path)
            @test haskey(fom["results"], "true")

            checkpoint = MacroModelling.build_chain_checkpoint_payload([1, 2, 3];
                                                                       chunks_done = 2,
                                                                       samples_done = 5,
                                                                       timestamp = 123.5)
            @test checkpoint["chain"] == [1, 2, 3]
            @test checkpoint["chunks_done"] == 2
            @test checkpoint["samples_done"] == 5
            @test checkpoint["timestamp"] == 123.5
            @test_throws ErrorException MacroModelling.build_chain_checkpoint_payload([1]; chunks_done = -1, samples_done = 1)
            @test_throws ErrorException MacroModelling.build_chain_checkpoint_payload([1]; chunks_done = 1, samples_done = -1)
        end
    end
end
