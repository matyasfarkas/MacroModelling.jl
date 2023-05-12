using MacroModelling
import Turing
import Turing: NUTS, sample, logpdf
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL: logjoint
import LinearAlgebra as ℒ
import ChainRulesCore: @ignore_derivatives, ignore_derivatives

cd("D:/Customtools/.julia/packages/MacroModelling/em8im/test")
include("models/FS2000.jl")

FS2000 = m

# load data
dat = CSV.read("data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)


function calculate_filterfree_loglikelihood(m, data, observables ; parameters = nothing, filter = :filter_free, shock_distribution = Normal(), algorithm = :first_order, verbose::Bool = false,tol::AbstractFloat = eps())
    # Checks
    @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= m.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    @ignore_derivatives sort!(observables)

    @ignore_derivatives solve!(m, verbose = verbose)

    if isnothing(parameters)
        parameters = m.parameter_values
    else
        ub = @ignore_derivatives fill(1e12+rand(),length(m.parameters) + length(m.➕_vars))
        lb = @ignore_derivatives -ub

        for (i,v) in enumerate(m.bounded_vars)
            if v ∈ m.parameters
                @ignore_derivatives lb[i] = m.lower_bounds[i]
                @ignore_derivatives ub[i] = m.upper_bounds[i]
            end
        end

        if min(max(parameters,lb),ub) != parameters 
            return -Inf
        end
    end

    SS_and_pars, solution_error = m.SS_solve_func(parameters, m, verbose)

    if solution_error > tol || isnan(solution_error)
        return -Inf
    end

    NSSS_labels = @ignore_derivatives [sort(union(m.exo_present,m.var))...,m.calibration_equations_parameters...]

    obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)

    data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]

    observables_and_states = @ignore_derivatives sort(union(m.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(m.aux,m.var,m.exo_present)))))

    solution = get_solution(m, parameters, algorithm = algorithm)

    if algorithm == :first_order
        𝐒₁ = solution[2]
    else
        𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
    end

    # Gaussian Prior
    # we can probably skip this because it is computationally expensive and should drop out in sampling - MF, we cannot as the initial condition needs to be scaled 
    calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = m.timings.past_not_future_and_mixed_idx)    
    long_run_covariance = calculate_covariance_(solution[2])
    # println(long_run_covariance)
    x0 = zeros(m.timings.nPast_not_future_and_mixed)
    # x0 ~ Turing.filldist(shock_distribution,m.timings.nPast_not_future_and_mixed) # Initial conditions  check dimensions!!!
    #x0 = rand(shock_distribution,m.timings.nPast_not_future_and_mixed)
    initial_conditions = long_run_covariance  * x0

    # draw errors
    # ϵ_draw = zeros(m.timings.nExo * size(data, 2))
    ϵ_draw ~ Turing.filldist(shock_distribution, m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
    # ϵ_draw = rand(shock_distribution, m.timings.nExo * size(data, 2))
    ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))

    state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2) )

    if algorithm == :first_order

        aug_state = [initial_conditions
        ϵ[:,1]]
        state[:,1] .=  𝐒₁ * aug_state
    
        for t in 2:size(data, 2)
            aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                        ϵ[:,t]]
            state[:,t] .=  𝐒₁ * aug_state 
        end
    elseif algorithm == :second_order
        aug_state = [initial_conditions
        1 
        ϵ[:,1]]
        state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

        for t in 2:size(data, 2)
            aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                        1 
                        ϵ[:,t]]
            state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
        end
    elseif algorithm == :pruned_second_order
        aug_state = [initial_conditions
        1 
        ϵ[:,1]]
        state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

        for t in 2:size(data, 2)
            aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                        1 
                        ϵ[:,t]]
            state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
        end
    end


    state_deviations = data - state[obs_indices,:]

    # could use sum of squared instead - Yes, now I implemented with smallish ME
    make_sure_state_equals_observable = sum([Turing.logpdf(Turing.MvNormal(zeros(size(data)[1]),Matrix(10^(-20)*ℒ.I, size(data)[1], size(data)[1])), state_deviations[:,t]) for t in 1:size(data, 2)])

    return make_sure_state_equals_observable
end
#=
alp     = 0.356
bet     = 0.993
gam     = 0.0085
mst     = 1.0002
rho     = 0.129
psi     = 0.65
del     = 0.01
z_e_a   = 0.035449
z_e_m   = 0.008862



parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]
shock_distribution = Turing.Normal()
algorithm = :first_order 
verbose::Bool = false
tol::AbstractFloat = eps()
filter = :filter_free
calculate_filterfree_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
=#

filter = :filter_free
shock_distribution = MacroModelling.Normal
algorithm = :first_order
verbose = false
tol = eps()
get_solution(m, m.parameter_values, algorithm = :first_order)

Turing.@model function FS2000_loglikelihood_function(data, model, observables)
    alp     ~ Beta(0.356, 0.02, μσ = true)
    bet     ~ Beta(0.993, 0.002, μσ = true)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    psi     ~ Beta(0.65, 0.05, μσ = true)
    del     ~ Beta(0.01, 0.005, μσ = true)
    z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)

        # draw errors
      # ϵ_draw = zeros(m.timings.nExo * size(data, 2))
      ϵ_draw ~ Turing.filldist(Turing.Normal(), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
      # ϵ_draw = rand(shock_distribution, m.timings.nExo * size(data, 2))
      ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))

    parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]
    shock_distribution = Turing.Normal()
    algorithm = :first_order 
    verbose::Bool = false
    tol::AbstractFloat = eps()
    filter = :filter_free
      # Checks
      @assert length(observables) == size(data)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
      @assert length(observables) <= m.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."
  
      @ignore_derivatives sort!(observables)
  
      @ignore_derivatives solve!(m, verbose = verbose)
  
      if isnothing(parameters)
          parameters = m.parameter_values
      else
          ub = @ignore_derivatives fill(1e12+rand(),length(m.parameters) + length(m.➕_vars))
          lb = @ignore_derivatives -ub
  
          for (i,v) in enumerate(m.bounded_vars)
              if v ∈ m.parameters
                  @ignore_derivatives lb[i] = m.lower_bounds[i]
                  @ignore_derivatives ub[i] = m.upper_bounds[i]
              end
          end
  
          if min(max(parameters,lb),ub) != parameters 
              return -Inf
          end
      end
  
      SS_and_pars, solution_error = m.SS_solve_func(parameters, m, verbose)
  
      if solution_error > tol || isnan(solution_error)
          return -Inf
      end
  
      NSSS_labels = @ignore_derivatives [sort(union(m.exo_present,m.var))...,m.calibration_equations_parameters...]
  
      obs_indices = @ignore_derivatives indexin(observables,NSSS_labels)
  
      data_in_deviations = collect(data(observables)) .- SS_and_pars[obs_indices]
  
      observables_and_states = @ignore_derivatives sort(union(m.timings.past_not_future_and_mixed_idx,indexin(observables,sort(union(m.aux,m.var,m.exo_present)))))
  
      solution = get_solution(m, parameters, algorithm = algorithm)
  
      if algorithm == :first_order
          𝐒₁ = solution[2]
      else
          𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
      end
  
      # Gaussian Prior
      # we can probably skip this because it is computationally expensive and should drop out in sampling - MF, we cannot as the initial condition needs to be scaled 
      calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = m.timings.past_not_future_and_mixed_idx)    
      long_run_covariance = calculate_covariance_(solution[2])
      # println(long_run_covariance)
      x0 = zeros(m.timings.nPast_not_future_and_mixed)
      # x0 ~ Turing.filldist(shock_distribution,m.timings.nPast_not_future_and_mixed) # Initial conditions  check dimensions!!!
      #x0 = rand(shock_distribution,m.timings.nPast_not_future_and_mixed)
      initial_conditions = long_run_covariance  * x0
  
      state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2) )
  
      if algorithm == :first_order
  
          aug_state = [initial_conditions
          ϵ[:,1]]
          state[:,1] .=  𝐒₁ * aug_state
      
          for t in 2:size(data, 2)
              aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                          ϵ[:,t]]
              state[:,t] .=  𝐒₁ * aug_state 
          end
      elseif algorithm == :second_order
          aug_state = [initial_conditions
          1 
          ϵ[:,1]]
          state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
  
          for t in 2:size(data, 2)
              aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                          1 
                          ϵ[:,t]]
              state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
          end
      elseif algorithm == :pruned_second_order
          aug_state = [initial_conditions
          1 
          ϵ[:,1]]
          state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
  
          for t in 2:size(data, 2)
              aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                          1 
                          ϵ[:,t]]
              state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
          end
      end
  
  
      state_deviations = data - state[obs_indices,:]
  
      # could use sum of squared instead - Yes, now I implemented with smallish ME
      make_sure_state_equals_observable = sum([Turing.logpdf(Turing.MvNormal(zeros(size(data)[1]),Matrix(10^(-20)*ℒ.I, size(data)[1], size(data)[1])), state_deviations[:,t]) for t in 1:size(data, 2)])
  
    Turing.@addlogprob! make_sure_state_equals_observable#calculate_filterfree_loglikelihood(model, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, observables)



n_samples = 100

# using Zygote
# Turing.setadbackend(:zygote)
samps = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)#, init_params = sol)

function calculate_posterior_loglikelihood(parameters)
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters
    log_lik = 0
    log_lik -= calculate_filterfree_loglikelihood(FS2000, data(observables), observables; parameters = parameters)
    log_lik -= logpdf(Beta(0.356, 0.02, μσ = true),alp)
    log_lik -= logpdf(Beta(0.993, 0.002, μσ = true),bet)
    log_lik -= logpdf(Normal(0.0085, 0.003),gam)
    log_lik -= logpdf(Normal(1.0002, 0.007),mst)
    log_lik -= logpdf(Beta(0.129, 0.223, μσ = true),rho)
    log_lik -= logpdf(Beta(0.65, 0.05, μσ = true),psi)
    log_lik -= logpdf(Beta(0.01, 0.005, μσ = true),del)
    log_lik -= logpdf(InverseGamma(0.035449, Inf, μσ = true),z_e_a)
    log_lik -= logpdf(InverseGamma(0.008862, Inf, μσ = true),z_e_m)
    return log_lik
end


sol = Optim.optimize(calculate_posterior_loglikelihood, 
[0,0,-10,-10,0,0,0,0,0], [1,1,10,10,1,1,1,100,100] ,FS2000.parameter_values, 
Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)

@testset "Estimation results" begin
    @test isapprox(sol.minimum, -1343.7491257498598, rtol = eps(Float32))
    @test isapprox(mean(samps).nt.mean, [0.40248024934137033, 0.9905235783816697, 0.004618184988033483, 1.014268215459915, 0.8459140293740781, 0.6851143053372912, 0.0025570276255960107, 0.01373547787288702, 0.003343985776134218], rtol = 1e-2)
end



MacroModelling.plot_model_estimates(FS2000, data, parameters = sol.minimizer)
plot_shock_decomposition(FS2000, data)
