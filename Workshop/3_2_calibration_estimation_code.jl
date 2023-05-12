#= Calibration and estimation of models
We will use the [Schorfheide (2000)](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.582) model which is often used as a benchmark for estimation. The model features a monetary policy rule without feedback, a Cobb-Douglas production function of capital and labor, cash-in-advance, and a financial intermediary. We will first calibrate the non stochastic steady state, then the theoretical moments, and last but not least set up the codes to estimate the model.=#

using MacroModelling


using Optim, LineSearches


using StatsPlots, Statistics, CSV, DataFrames



@model FS2000 begin
    P[0] / (c[1] * P[1] * m[0]) = β * P[1] * (α * exp( - α * a[1]) * k[0] ^ (α - 1) * n[1] ^ (1 - α) + (1 - δ) * exp( - a[1])) / (c[2] * P[2] * m[1])

    W[0] = l[0] / n[0]

    l[0] / n[0] = (ψ / (1 - ψ)) * (c[0] * P[0] / (1 - n[0])) 

    R[0] = P[0] * (1 - α) * exp( - α * a[0]) * k[-1] ^ α * n[0] ^ ( - α) / W[0]

    1 / (c[0] * P[0]) = β * P[0] * (1 - α) * exp( - α * a[0]) * k[-1] ^ α * n[0] ^ (1 - α) / (m[0] * l[0] * c[1] * P[1])

    c[0] + k[0] = exp( - α * a[0]) * k[-1] ^ α * n[0] ^ (1 - α) + (1 - δ) * exp( - a[0]) * k[-1]

    P[0] * c[0] = m[0]

    m[0] - 1 + d[0] = l[0]

    y[0] = k[-1] ^ α * n[0] ^ (1 - α) * exp( - α * a[0])

    a[0] = γ + σᵃ * ϵᵃ[x]

    log(m[0]) = (1 - ρ) * log(m̄) + ρ * log(m[-1]) + σᵐ * ϵᵐ[x]

    dy_obs[0] = exp(a[0]) * y[0] / y[-1]

    π_obs[0] = (P[0] / P[-1]) * m[-1] / exp(a[0])

	R_ann[0] = 400 * log(R[0])
	
	y_gap[0] = log(y[0] / y[ss]) * 100

	pi_ann[0] = 400 * log(π_obs[0])
end


@parameters FS2000 begin
    α   = 0.4036
    β   = 0.9904
    γ   = 0.0046
    m̄   = 1.0142
    ρ   = 0.8468
    ψ   = 0.6811
    δ   = 0.0024
    σᵃ  = 0.0138
    σᵐ  = 0.0033
end


get_SS(FS2000)


#= Let's calibrate the SS such that the capital to GDP ratio is 10.4 and the short term interest rate is 7% in annualised terms by using the parameters δ and β. =#


@model FS2000_calib begin
    P[0] / (c[1] * P[1] * m[0]) = β * P[1] * (α * exp( - α * a[1]) * k[0] ^ (α - 1) * n[1] ^ (1 - α) + (1 - δ) * exp( - a[1])) / (c[2] * P[2] * m[1])

    W[0] = l[0] / n[0]

    l[0] / n[0] = (ψ / (1 - ψ)) * (c[0] * P[0] / (1 - n[0])) 

    R[0] = P[0] * (1 - α) * exp( - α * a[0]) * k[-1] ^ α * n[0] ^ ( - α) / W[0]

    1 / (c[0] * P[0]) = β * P[0] * (1 - α) * exp( - α * a[0]) * k[-1] ^ α * n[0] ^ (1 - α) / (m[0] * l[0] * c[1] * P[1])

    c[0] + k[0] = exp( - α * a[0]) * k[-1] ^ α * n[0] ^ (1 - α) + (1 - δ) * exp( - a[0]) * k[-1]

    P[0] * c[0] = m[0]

    m[0] - 1 + d[0] = l[0]

    y[0] = k[-1] ^ α * n[0] ^ (1 - α) * exp( - α * a[0])

    a[0] = γ + σᵃ * ϵᵃ[x]

    log(m[0]) = (1 - ρ) * log(m̄) + ρ * log(m[-1]) + σᵐ * ϵᵐ[x]

    dy_obs[0] = exp(a[0]) * y[0] / y[-1]

    π_obs[0] = (P[0] / P[-1]) * m[-1] / exp(a[0])
	
	R_ann[0] = 400 * log(R[0])
	
	y_gap[0] = log(y[0] / y[ss]) * 100

	π_ann[0] = 400 * log(π_obs[0])
end

# Make the necessary adjustments in the @parameters block. Essentailly we want to R_ann to be 7 and k/y to be 10.4

@parameters FS2000_calib begin
    α   = 0.4036
	
    β 
	
    γ   = 0.0046
    m̄   = 1.0142
    ρ   = 0.8468
    ψ   = 0.6811
	
    δ 
	
    σᵃ  = 0.0138
    σᵐ  = 0.0033
	
	0 < β < 1
	0 < δ < 1
end


#= Having done the necessary changes let's look at the standard output again: =#



#= Next, let's calibrate the standard deviation of the annualised interest rate to be 3.4 and the standard deviation of the output gap be 2. We are going to make this an optimisation problem with 2 targets and 3 unknowns (parameters of the model). We will use the parameters α, γ, and ρ to reach our targets. First we need a function mapping the parameters to the standard deviations. Use the get_statistics function and check the help (?) for details on the arguments =#




#= Now we can use this to optimise over: =#


sum(abs2, get_statistics("<arguments>")[1] - [3.4, 2])


results = optimize(x -> sum(abs2, get_statistics("<arguments>")[1] - [3.4,2]),
	[0, 0, 0], # lower bounds
	[0.5, 2, 1], # upper bounds
	[0.4036, 0.6811, 0.8468], # initial value
	Fminbox(LBFGS(linesearch = BackTracking(order = 3))), # optimisation algorithm
	autodiff = :forward)


#= Let us check the result of the optimisation. First the optimal parameter values and then the implied standard deviations. =#


results.minimizer


get_statistics("<arguments>")[1]



#= ## Estimation =#


#= For the estimation we will first load the data accompanying the Schorfheide (2000) paper. You can find it [here](http://www.thorekockerols.eu/data/FS2000_data.csv)=#

dat = CSV.read("FS2000_data.csv", DataFrame)

data = KeyedArray(Array(dat)',Variable = [:dy_obs,:π_obs],Time = 1:size(dat)[1])


#= Let's plot the two series: =#


plot(collect(data(:dy_obs)),labels = "dy_obs")

plot(collect(data(:π_obs)),labels = "π_obs")


#= Next we are going to combine the prior and kalman filter loglikelihood in one function returning the posterior: =#


import Turing


#= We need to define the observabes and then let's check the kalman ilter loglikelihood with the current parameters. =#


observables = [:dy_obs, :π_obs]

calculate_kalman_filter_loglikelihood(FS2000_calib, data, observables; parameters = FS2000_calib.parameter_values)


#= Next we define model to estimate including the priors for the parameters. =#


Turing.@model function loglikelihood_function(data, m, observables)
    σᵐ   ~ InverseGamma(0.01, Inf, μσ = true)
    σᵃ   ~ InverseGamma(0.01, Inf, μσ = true)
	R̄    ~ Normal(7, 1, 4, 10)
	m̄    ~ Normal(1.0002, 0.007)
	ρ    ~ Beta(0.129, 0.223, μσ = true)
	ψ    ~ Beta(0.65, 0.05, μσ = true)
	α    ~ Beta(0.356, 0.02, μσ = true)
    γ    ~ Normal(0.0085, 0.003)
    k_y  ~ Normal(10.4, 0.5, 7.0, 13.0)
	
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data, observables; parameters = [α, R̄, γ, m̄, ρ, ψ, k_y, σᵃ, σᵐ])
end

loglikelihood = loglikelihood_function(data, FS2000_calib, observables)

import Turing: NUTS, sample, logpdf

samps = sample(loglikelihood, NUTS(), 100, progress = true)

samps 


#= What is also possible is to include system priors on the standard deviation of some variables: =#


Turing.@model function loglikelihood_function_sytem_prior(data, m, observables)
    σᵐ   ~ InverseGamma(0.01, Inf, μσ = true)
    σᵃ   ~ InverseGamma(0.01, Inf, μσ = true)
	R̄    ~ Normal(7, 1, 4, 10)
	m̄    ~ Normal(1.0002, 0.007)
	ρ    ~ Beta(0.129, 0.223, μσ = true)
	ψ    ~ Beta(0.65, 0.05, μσ = true)
	α    ~ Beta(0.356, 0.02, μσ = true)
    γ    ~ Normal(0.0085, 0.003)
    k_y  ~ Normal(10.4, 0.5, 7.0, 13.0)
	
	standard_deviations = get_statistics(m, [α, R̄, γ, m̄, ρ, ψ, k_y, σᵃ, σᵐ], parameters = m.parameters, standard_deviation = [:R_ann,:y_gap])[1]
	
	Turing.@addlogprob! logpdf(Normal(3.4,.01),standard_deviations[1])
	Turing.@addlogprob! logpdf(Normal(2.0,.01),standard_deviations[2])
	
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data, observables; parameters = [α, R̄, γ, m̄, ρ, ψ, k_y, σᵃ, σᵐ])
end

loglikelihood_sytem_prior = loglikelihood_function_sytem_prior(data, FS2000_calib, observables)
