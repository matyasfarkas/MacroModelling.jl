using MacroModelling, JuMP, Ipopt
import Turing, StatsPlots, Random, Statistics
using CSV, DataFrames, Dates
import LinearAlgebra as ℒ
using HypothesisTests, Distributions
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import ForwardDiff as ℱ

@model AS07 begin
	c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA+RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA+RN[1]) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) 

	log(g[0]) = SIGG / 100 * epsg[x] + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	DYA[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	PIC[0] = PA/4 + 100 * log(p[0] / p[ss])

	R[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    RN[0] = RHORN * RN[-1] +  SIGRN / 100 * epsrn[x]
end


@parameters AS07 begin
	RA = 1

	PA = 3.2

	GAMQ =  0 #0.55

	TAU = 2

	NU = 0.1

	KAPPA   = 0.33

	PSIP = 1.5

	PSIY = 0.125

	RHOR = 0.75

	RHOG = 0.95

	RHOZ = 0.9

	SIGR = 0.2

	SIGG = 0.6

	SIGZ = 0.3

	C_o_Y = 0.85

	OMEGA = 0

	XI = 1

    RHORN = 0.9

    SIGRN = 0.01

    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2


end


# load data
dat = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/NG project/NAWM_dataset.csv", DataFrame)
data = KeyedArray(Array(dat[:,2:end])' ,Variable = (Symbol.(names(dat[:,2:end]))) , Time = 1:size(dat)[1] ) #Dates.DateTime.(dat[:,1], Dates.DateFormat("d-u-yyyy"))
observables = [:DYA, :R, :PIC]

# subset observables in data
data = data(observables,:)

Ω = 10^(-4)# eps()
n_samples = 1000

# KF linear state space estimation
Turing.@model function loglikelihood_scaling_function(m, data, observables)
    
    # RA,              1,             1e-5,        10,          gamma_pdf,     0.8,        0.5;
    # PA,              3.2,           1e-5,        20,          gamma_pdf,     4,          2;
    # GAMQ,            0.55,          -5,          5,           normal_pdf,    0.4,        0.2;
    # TAU,             2,             1e-5,        10,          gamma_pdf,     2,          0.5;
    # NU,              0.1,           1e-5,        0.99999,     beta_pdf,      0.1,        0.05;
    # PSIP,            1.5,           1e-5,        10,          gamma_pdf,     1.5,        0.25;
    # PSIY,            0.125,         1e-5,        10,          gamma_pdf,     0.5,        0.25;
    # RHOR,            0.75,          1e-5,        0.99999,     beta_pdf,      0.5,        0.2;
    # RHOG,            0.95,          1e-5,        0.99999,     beta_pdf,      0.8,        0.1;
    # RHOZ,            0.9,           1e-5,        0.99999,     beta_pdf,      0.66,       0.15;
    # SIGR,            0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
    # SIGG,            0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
    # SIGZ,            0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
    # C_o_Y,           0.85,          1e-5,        0.99999,     beta_pdf,      0.85,       0.1;
    # PHI,             50,            1e-5,        100,         gamma_pdf,     50,         20;
    # stderr epsz,     0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
    # stderr epsr,     0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
    # stderr epsg,     0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
    # OMEGA,           0,             -10,         10,          normal_pdf,      0,        1;
    # XI,              1,             0,           2,           uniform_pdf,      ,         ,                    0,                   2;

   
     RA ~  MacroModelling.Normal(1,0.5)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
	 PA ~  MacroModelling.Normal(2.,0.1)    #  MacroModelling.Gamma(4.,2.,μσ = true)

    #  RA ~  MacroModelling.Gamma(1.,1,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #  PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    #  GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    #  TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
    #  NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    #  PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    #  PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    #  RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    #  RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    #  RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    #  SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    #  SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    #  SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    #  OMEGA ~  MacroModelling.Normal(0.,0.2)
     

    #RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
	#PA ~  MacroModelling.Gamma(2.,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
	TAU	~  MacroModelling.Gamma(2.,0.2,μσ = true)
	NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
	RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
	RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
	RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    OMEGA ~  MacroModelling.Normal(0.,0.2)
    #XI ~  Turing.Uniform(0,1)
    C_o_Y	~ MacroModelling.Beta(0.85,0.1,μσ = true)

    
    # SIGFG ~  MacroModelling.InverseGamma( 0.1,4., 10^(-8), 5., μσ = true)
    # SIGRN ~  MacroModelling.InverseGamma( 0.001,1., 10^(-8), 1., μσ = true)
    
    # RHORN = 0.9
    RHORN ~ MacroModelling.Beta( 0.9,0.1,μσ = true)
     #RA = 1
    # PA = 3.2
	# GAMQ = 0
    # TAU = 2
	# NU = 0.1
	KAPPA   = 0.33
    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

	# PSIP = 1.5
	# PSIY = 0.125
	# RHOR = 0.75
	# RHOG = 0.95
	# RHOZ = 0.9
	# SIGR = 0.2
	# SIGG = 0.6
	# SIGZ = 0.3
	# C_o_Y = 0.85
    SIGRN =0.001
	# OMEGA = 0
	XI = 1
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, RHORN, SIGRN]
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

end

loglikelihood_scaling = loglikelihood_scaling_function(AS07, data, observables) # Kalman
samps = Turing.sample(loglikelihood_scaling,Turing.NUTS(),n_samples, progress = true)


StatsPlots.plot(samps)

using StatsBase
summ = summarystats(samps)
parameter_mean  = summ.nt.mean
parameter_std  = summ.nt.std

filtered_states = get_estimated_variables(AS07, data,parameters = parameter_mean)
KF_shocks = get_estimated_shocks(AS07, data, parameters = parameter_mean)

plot_model_estimates(AS07,data(observables))
plot_shock_decomposition(AS07,data(observables))

using Dates
dates = Date(1980, 6, 30):Month(3):Date(2023, 6, 30)
tm_ticks = round.(dates, Quarter(16)) |> unique;

StatsPlots.plot(dates, parameter_mean[1].+  parameter_mean[3]*4 .+ filtered_states(:RN),ribbon= (parameter_std[1]+parameter_std[3])*2,label= "Filtered natural rate estimate",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))

## FF 

Turing.@model function loglikelihood_scaling_function_ff(m, data, observables, Ω) 
     
    # RA,              1,             1e-5,        10,          gamma_pdf,     0.8,        0.5;
     # PA,              3.2,           1e-5,        20,          gamma_pdf,     4,          2;
     # GAMQ,            0.55,          -5,          5,           normal_pdf,    0.4,        0.2;
     # TAU,             2,             1e-5,        10,          gamma_pdf,     2,          0.5;
     # NU,              0.1,           1e-5,        0.99999,     beta_pdf,      0.1,        0.05;
     # PSIP,            1.5,           1e-5,        10,          gamma_pdf,     1.5,        0.25;
     # PSIY,            0.125,         1e-5,        10,          gamma_pdf,     0.5,        0.25;
     # RHOR,            0.75,          1e-5,        0.99999,     beta_pdf,      0.5,        0.2;
     # RHOG,            0.95,          1e-5,        0.99999,     beta_pdf,      0.8,        0.1;
     # RHOZ,            0.9,           1e-5,        0.99999,     beta_pdf,      0.66,       0.15;
     # SIGR,            0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
     # SIGG,            0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # SIGZ,            0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # C_o_Y,           0.85,          1e-5,        0.99999,     beta_pdf,      0.85,       0.1;
     # PHI,             50,            1e-5,        100,         gamma_pdf,     50,         20;
     # stderr epsz,     0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # stderr epsr,     0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
     # stderr epsg,     0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # OMEGA,           0,             -10,         10,          normal_pdf,      0,        1;
     # XI,              1,             0,           2,           uniform_pdf,      ,         ,                    0,                   2;
 
    
    # RA ~  MacroModelling.Normal(1,0.5)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
     PA ~  MacroModelling.Normal(2.,0.1)    #  MacroModelling.Gamma(4.,2.,μσ = true)
     RA =2.0
    #  RA ~  MacroModelling.Gamma(1.,1,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #  PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    #  GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    #  TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
    #  NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    #  PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    #  PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    #  RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    #  RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    #  RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    #  SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    #  SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    #  SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    #  OMEGA ~  MacroModelling.Normal(0.,0.2)
     
 
    #RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #PA ~  MacroModelling.Gamma(2.,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    TAU	~  MacroModelling.Gamma(2.,0.2,μσ = true)
    NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    OMEGA ~  MacroModelling.Normal(0.,0.2)
    #XI ~  Turing.Uniform(0,1)
    C_o_Y	~ MacroModelling.Beta(0.85,0.1,μσ = true)
 
    # scale ~ MacroModelling.Gamma(1.,0.2,μσ = true)
    scale = 0.1
    # SIGFG ~  MacroModelling.InverseGamma( 0.1,4., 10^(-8), 5., μσ = true)
    # SIGRN ~  MacroModelling.InverseGamma( 0.001,1., 10^(-8), 1., μσ = true)
    
    RHORN = 0.
    # RHORN ~ MacroModelling.Beta( 0.9,0.1,μσ = true)
    # RA = 1
    # PA = 3.2
    # GAMQ = 0
    # TAU = 2
    # NU = 0.1
    KAPPA   = 0.33
    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
 
    # PSIP = 1.5
    # PSIY = 0.125
    # RHOR = 0.75
    # RHOG = 0.95
    # RHOZ = 0.9
    # SIGR = 0.2
    # SIGG = 0.6
    # SIGZ = 0.3
    # C_o_Y = 0.85
    SIGRN =0.001
    # OMEGA = 0
    XI = 1
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, RHORN, SIGRN]
 
    algorithm = :first_order
 
     # Shock distribution 
     shock_distribution = Turing.Normal() 
     solution = get_solution(m, parameters, algorithm = algorithm)
 
     if solution[end] != true
          return Turing.@addlogprob! Inf
     end
 
     # draw_shocks(m)
     
     x0 ~ Turing.filldist(shock_distribution, m.timings.nVars) # Initial conditions - Normal!
             #  x0 = rand(shock_distribution,10,1)
     calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
 
     long_run_covariance = calculate_covariance_(solution[2])
    
     initial_conditions = long_run_covariance * x0
 
     𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
     ϵ_draw ~ Turing.filldist(shock_distribution, (m.timings.nExo-1) * size(data, 2))
     
     # ϵ_draw = rand(shock_distribution, (m.timings.nExo-1) * size(data, 2))
     ϵ = zeros(eltype(ϵ_draw), m.timings.nExo, size(data, 2))
     ϵ_skewdraw = zeros(eltype(ϵ_draw),1,size(data, 2))
 
     ϵ[2:end,:] = reshape(ϵ_draw,(m.timings.nExo-1), size(data, 2))
 
     state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
     DF = 0.
     skew_distribution =  Turing.SkewNormal(0,1,DF)
 
     DF_out =zeros(eltype(skew_distribution), size(data, 2))
     DF_out[1] ~  MacroModelling.Normal(DF,eps()) 
 
     ϵ_skewdraw[1] ~ skew_distribution
         # ϵ_skewdraw[1] = rand(skew_distribution,1,1)
     ϵ[:,1] = [ϵ_skewdraw[1]; ϵ[2:end,1]]
 
     #  aug_state = [initial_conditions
     #             1 
     #             ϵ[:,1]]
 
     state[:,1] .=  𝐒₁ * initial_conditions# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
 
     for t in 2:size(data, 2)
         DF  = DF_out[t-1]
         DF_out[t] ~  MacroModelling.Normal(DF,scale*1) 
         skew_distribution =  Turing.SkewNormal(0,1,DF_out[t])
         ϵ_skewdraw[t] ~ skew_distribution
            #  ϵ_skewdraw[1] = rand(skew_distribution,1,1)
         ϵ[:,t] = [ϵ_skewdraw[t]; ϵ[2:end,t]]
         aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                     1 
                     ϵ[:,t]]
         state[:,t] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
     end
 
     observables_index = sort(indexin(observables, m.timings.var))
    
     state_deviations = data - state[observables_index,:] .- solution[1][observables_index]
 
     Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
 end
loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(AS07, data, observables,Ω) # Kalman
n_samples = 100
samps_ff = Turing.sample(loglikelihood_scaling_ff,Turing.NUTS(),n_samples, progress = true)

StatsPlots.plot(samps_ff)

loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(AS07, data, observables,Ω) # Kalman
n_samples = 10
samps_ff = Turing.sample(loglikelihood_scaling_ff,Turing.NUTS(),n_samples, progress = true)

ff_estimated_means = Turing.describe(samps_ff)[1].nt.mean
ff_estimated_std = Turing.describe(samps_ff)[1].nt.std
PSIP = ff_estimated_means[1]
parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, RHORN, SIGRN]

filtered_states = get_estimated_variables(AS07, data,parameters = parameters)

using Dates
dates = Date(1980, 6, 30):Month(3):Date(2023, 6, 30)
tm_ticks = round.(dates, Quarter(16)) |> unique;

StatsPlots.plot(dates, filtered_states(:RN),label= "Filtered natural rate estimate",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))


ff_estimated_parameters = Turing.describe(samps_ff)[1].nt.parameters
ff_estimated_means = Turing.describe(samps_ff)[1].nt.mean
ff_estimated_std = Turing.describe(samps_ff)[1].nt.std
StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
StatsPlots.density!(ff_estimated_means[ff_estimated_parameters_indices])

ff_estimated_parameters_indices = indexin([Symbol("ϵ_skewdraw[$a]") for a in 1:size(data,2)], ff_estimated_parameters )
StatsPlots.plot(ff_estimated_means[ff_estimated_parameters_indices],
                ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices], 
                label = "Posterior mean", 
                title = "Joint: Estimated Latents")


                
Turing.@model function loglikelihood_scaling_function_ff(m, data, observables, Ω, KF_shocks) 

    # RA,              1,             1e-5,        10,          gamma_pdf,     0.8,        0.5;
        # PA,              3.2,           1e-5,        20,          gamma_pdf,     4,          2;
        # GAMQ,            0.55,          -5,          5,           normal_pdf,    0.4,        0.2;
        # TAU,             2,             1e-5,        10,          gamma_pdf,     2,          0.5;
        # NU,              0.1,           1e-5,        0.99999,     beta_pdf,      0.1,        0.05;
        # PSIP,            1.5,           1e-5,        10,          gamma_pdf,     1.5,        0.25;
        # PSIY,            0.125,         1e-5,        10,          gamma_pdf,     0.5,        0.25;
        # RHOR,            0.75,          1e-5,        0.99999,     beta_pdf,      0.5,        0.2;
        # RHOG,            0.95,          1e-5,        0.99999,     beta_pdf,      0.8,        0.1;
        # RHOZ,            0.9,           1e-5,        0.99999,     beta_pdf,      0.66,       0.15;
        # SIGR,            0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
        # SIGG,            0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # SIGZ,            0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # C_o_Y,           0.85,          1e-5,        0.99999,     beta_pdf,      0.85,       0.1;
        # PHI,             50,            1e-5,        100,         gamma_pdf,     50,         20;
        # stderr epsz,     0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # stderr epsr,     0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
        # stderr epsg,     0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # OMEGA,           0,             -10,         10,          normal_pdf,      0,        1;
        # XI,              1,             0,           2,           uniform_pdf,      ,         ,                    0,                   2;
    
    
    # RA ~  MacroModelling.Normal(1,0.5)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
        # PA ~  MacroModelling.Normal(2.,0.01)    #  MacroModelling.Gamma(4.,2.,μσ = true)
        RA =2.0
    #  RA ~  MacroModelling.Gamma(1.,1,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #  PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    #  GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    #  TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
    #  NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    #  PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    #  PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    #  RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    #  RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    #  RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    #  SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    #  SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    #  SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    #  OMEGA ~  MacroModelling.Normal(0.,0.2)
        
    
    #RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #PA ~  MacroModelling.Gamma(2.,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    # GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    # TAU	~  MacroModelling.Gamma(2.,0.2,μσ = true)
    # NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    # PSIP ~  MacroModelling.Gamma(1.5,0.15,μσ = true)
    # PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    # PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    # RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    # RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    # RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    # SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    # SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    # SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    # OMEGA ~  MacroModelling.Normal(0.,0.2)
    #XI ~  Turing.Uniform(0,1)
    # C_o_Y	~ MacroModelling.Beta(0.85,0.1,μσ = true)
    
    # scale ~ MacroModelling.Gamma(1.,0.2,μσ = true)
    # RHODF ~ MacroModelling.Normal(0.999,0.001)
    scale = 10^(-6)
    # SIGFG ~  MacroModelling.InverseGamma( 0.1,4., 10^(-8), 5., μσ = true)
    # SIGRN ~  MacroModelling.InverseGamma( 0.001,1., 10^(-8), 1., μσ = true)
    

    RHORN = 0.
    # RHORN ~ MacroModelling.Beta( 0.9,0.1,μσ = true)
    # RA = 1
    # PA = 3.2
    # GAMQ = 0
    TAU = 2
    NU = 0.1
    KAPPA   = 0.33
    


    # RA=0.0187861472477752
    PA=2.00490052071224
    GAMQ=0.31536520785289
    TAU=1.97980281940216
    NU=0.100689748140405
    PSIP=1.03561156181642
    PSIY=0.0511500893572928
    RHOR=0.911087643865179
    RHOG=0.798290989570892
    RHOZ=0.995222993337873
    SIGR=0.117112521992729
    SIGG=2.40686320492466
    SIGZ=0.0803266132224477
    OMEGA=-0.00266786746974098
    C_o_Y=0.847780402732689
    # RHORN=0.903345973321453
    
    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

    # PSIP = 1.5
    # PSIY = 0.125
    # RHOR = 0.75
    # RHOG = 0.95
    # RHOZ = 0.9
    # SIGR = 0.2
    # SIGG = 0.6
    # SIGZ = 0.3
    # # C_o_Y = 0.85
    SIGRN =0.001
    # # OMEGA = 0
    XI = 1
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, RHORN, SIGRN]
    
    algorithm = :first_order
    
        # Shock distribution 
        shock_distribution = Turing.Normal() 
        solution = get_solution(m, parameters, algorithm = algorithm)
    
        if solution[end] != true
            return Turing.@addlogprob! Inf
        end
    
        # draw_shocks(m)
        
        x0 ~ Turing.filldist(shock_distribution, m.timings.nVars) # Initial conditions - Normal!
                #  x0 = rand(shock_distribution,10,1)
        calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
    
        long_run_covariance = calculate_covariance_(solution[2])
    
        initial_conditions = long_run_covariance * x0
    
        𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
        
        DF = 0.
        skew_distribution =  Turing.SkewNormal(0,1,DF)
        DF_out =zeros(eltype(skew_distribution), size(data, 2))

        DF_out[1] =0
                 
        ϵ_skewdraw = zeros(eltype(skew_distribution),1,size(data, 2))
        ϵ_skewdraw[1] ~ skew_distribution
           

        state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
    
        ϵ_skewdraw[1] ~ skew_distribution
            # ϵ_skewdraw[1] = rand(skew_distribution,1,1)
        ϵ= [ϵ_skewdraw[1]; KF_shocks[2:end,1]]
    
        #  aug_state = [initial_conditions
        #             1 
        #             ϵ[:,1]]
    
        state[:,1] .=  𝐒₁ * initial_conditions# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    
        for t in 2:size(data, 2)
            DF_out[t] ~   MacroModelling.Normal(0.9999*DF_out[t-1],scale*1)    #Turing.Uniform(-1,1)  MacroModelling.Normal(0.999*DF_out[t-1],scale*1)    
            skew_distribution =  Turing.SkewNormal(0,1,DF_out[t])
            ϵ_skewdraw[t] ~ skew_distribution
            #  ϵ_skewdraw[1] = rand(skew_distribution,1,1)
            ϵ = [ϵ_skewdraw[t]; KF_shocks[2:end,t]]
            aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                        1 
                        ϵ]
            state[:,t] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
        end
    
        observables_index = sort(indexin(observables, m.timings.var))
    
        state_deviations = data - state[observables_index,:] .- solution[1][observables_index]
    
        Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
    end

loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(AS07, data, observables,Ω,collect(KF_shocks)) 
n_samples = 1000
samps_ff = Turing.sample(loglikelihood_scaling_ff,Turing.NUTS(),n_samples, progress = true)
                 


ff_estimated_parameters = Turing.describe(samps_ff)[1].nt.parameters
ff_estimated_means = Turing.describe(samps_ff)[1].nt.mean
ff_estimated_std = Turing.describe(samps_ff)[1].nt.std

ff_estimated_parameters_indices = indexin([Symbol("DF_out[$a]") for a in 2:size(data,2)], ff_estimated_parameters )
StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
StatsPlots.density!(ff_estimated_means[ff_estimated_parameters_indices])

StatsPlots.plot(ff_estimated_means[ff_estimated_parameters_indices],
                label = "Posterior mean", 
                title = "Joint: Estimated Latents")

StatsPlots.plot(dates[2:end], ff_estimated_means[ff_estimated_parameters_indices],ribbon= (ff_estimated_std[ff_estimated_parameters_indices])*1.96,label= "Filtered conditional skewness estimate",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))

kk =4 
RWDF = zeros(173)
z = collect( ff_estimated_means[ff_estimated_parameters_indices])
for e in ((@view z[i:i+4]) for i in 1:1:length(z)-4)
    RWDF[kk]= (mean(e))
    kk =kk+1
end
StatsPlots.plot(RWDF)




@model AS07_FG begin
	c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA+RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA+RN[1]) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) * exp(SIGFG / 100 * epsfg[0]) 

	log(g[0]) = SIGG / 100 * epsg[x] + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	DYA[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	PIC[0] = PA/4 + 100 * log(p[0] / p[ss])

	R[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    epsfg[0] = epsf1[-1]

    epsf1[0] = epsf2[-1] + epsf1x[x] 
    
    epsf2[0] = epsf3[-1] + epsf2x[x] 

    epsf3[0] = epsf4[-1] + epsf3x[x] 

    epsf4[0] = epsf5[-1] + epsf4x[x] 

    epsf5[0] = epsf6[-1] + epsf5x[x] 

    epsf6[0] = epsf7[-1] + epsf6x[x] 

    epsf7[0] = epsf8[-1] + epsf7x[x] 

    epsf8[0] = epsf9[-1]  + epsf8x[x] 
    
    epsf9[0] = epsf10[-1]  + epsf9x[x] 

    epsf10[0] = epsf11[-1]  + epsf10x[x] 

    epsf11[0] = epsf12[-1]  + epsf11x[x] 

    epsf12[0] = epsf13[-1]  + epsf12x[x] 

    epsf13[0] = epsf14[-1]  + epsf13x[x] 

    epsf14[0] = epsf15[-1]  + epsf14x[x] 

    epsf15[0] = epsf16[-1]  + epsf15x[x] 

    epsf16[0] = epsf16x[x] 

    RN[0] = RHORN * RN[-1] +  SIGRN / 100 * epsrn[x]
end


@parameters AS07_FG begin
	RA = 1

	PA = 3.2

	GAMQ =  0 #0.55

	TAU = 2

	NU = 0.1

	KAPPA   = 0.33

	PSIP = 1.5

	PSIY = 0.125

	RHOR = 0.75

	RHOG = 0.95

	RHOZ = 0.9

	SIGR = 0.2

	SIGG = 0.6

	SIGZ = 0.3

	C_o_Y = 0.85

	OMEGA = 0

	XI = 1

    SIGFG = 0.1

    RHORN = 0.9

    SIGRN = 0.01

    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2


end

## FF 

Turing.@model function loglikelihood_scaling_function_ff(m, data, observables, Ω , zlbvar, zlblevel,fgshlist) #, kfmean, kfstd
     
    # RA,              1,             1e-5,        10,          gamma_pdf,     0.8,        0.5;
     # PA,              3.2,           1e-5,        20,          gamma_pdf,     4,          2;
     # GAMQ,            0.55,          -5,          5,           normal_pdf,    0.4,        0.2;
     # TAU,             2,             1e-5,        10,          gamma_pdf,     2,          0.5;
     # NU,              0.1,           1e-5,        0.99999,     beta_pdf,      0.1,        0.05;
     # PSIP,            1.5,           1e-5,        10,          gamma_pdf,     1.5,        0.25;
     # PSIY,            0.125,         1e-5,        10,          gamma_pdf,     0.5,        0.25;
     # RHOR,            0.75,          1e-5,        0.99999,     beta_pdf,      0.5,        0.2;
     # RHOG,            0.95,          1e-5,        0.99999,     beta_pdf,      0.8,        0.1;
     # RHOZ,            0.9,           1e-5,        0.99999,     beta_pdf,      0.66,       0.15;
     # SIGR,            0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
     # SIGG,            0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # SIGZ,            0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # C_o_Y,           0.85,          1e-5,        0.99999,     beta_pdf,      0.85,       0.1;
     # PHI,             50,            1e-5,        100,         gamma_pdf,     50,         20;
     # stderr epsz,     0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # stderr epsr,     0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
     # stderr epsg,     0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
     # OMEGA,           0,             -10,         10,          normal_pdf,      0,        1;
     # XI,              1,             0,           2,           uniform_pdf,      ,         ,                    0,                   2;
 
    
     RA ~  MacroModelling.Gamma(1.,0.2,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
     PA ~  MacroModelling.Gamma(3.2,1.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
 
     # RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
     # PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
     # GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
     # TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
     # NU 	~ MacroModelling.Beta( 0.1,0.05,μσ = true)
     # PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
     # PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
     # RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
     # RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
     # RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
     # SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
     # SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
     # SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
     # OMEGA ~  MacroModelling.Normal(0,1)
     # XI ~  Turing.Uniform(0,1)
 
     # RA = 1
     # PA = 3.2
     GAMQ = 0 #0.55
     TAU = 2
     NU = 0.1
     KAPPA   = 0.33
     PSIP = 1.5
     PSIY = 0.125
     RHOR = 0.75
     RHOG = 0.95
     RHOZ = 0.9
     SIGR = 0.2
     SIGG = 0.6
     SIGZ = 0.3
     C_o_Y = 0.85
     OMEGA = 0
     XI = 1
     SIGFG = 0.1
     PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
 
     # data= simulated_data
     # zlbvar = [:INT]
     # zlblevel = 0
     # mpsh = [:epsr]
     # fgshlist = [:epsf1x, :epsf2x, :epsf3x, :epsf4x, :epsf5x, :epsf6x, :epsf7x, :epsf8x]
     # observables = [:INT, :YGR , :INFL ]
     
     zlbindex = sort(indexin(zlbvar, m.timings.var))
     #zlbshindex = sort(indexin(fgshlist, m.timings.exo))
     #mpshindex =  only(sort(indexin(mpsh, m.timings.exo)))
 
     parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, SIGFG]
 
     algorithm = :first_order
 
     # Shock distribution 
     shock_distribution = Turing.Normal()
 
     solution = get_solution(m, parameters, algorithm = algorithm)
 
     if solution[end] != true
          return Turing.@addlogprob! Inf
     end
     # draw_shocks(m)
      x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
      
     calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(m.timings.past_not_future_and_mixed_idx) ) # subset_indices = collect(1:m.timings.nVars))
 
     long_run_covariance = calculate_covariance_(solution[2])
     
     initial_conditions = long_run_covariance * x0
 
     𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
     ϵ_draw ~ Turing.filldist(shock_distribution, (m.timings.nExo-size(indexin(fgshlist, m.timings.exo),1)) * size(data, 2))
 
     ϵ = [zeros( size(indexin(fgshlist, m.timings.exo),1),size(data, 2) ) ; reshape(ϵ_draw, (m.timings.nExo-size(indexin(fgshlist, m.timings.exo),1)) , size(data, 2))]
 
     state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
     
     aug_state = [initial_conditions
                  1 
                  ϵ[:,1]]
     
     state[:,1] .=  𝐒₁ * aug_state#+ solution[3] * ℒ.kron(aug_state_unc, aug_state_unc) / 2 
     ϵ_wzlb = ℱ.value.(ϵ)
 
     for t in 2:size(data,2)
         # Get unconditional FC
         PLM= ℱ.value.(get_irf(m,shocks = [ℱ.value.(ϵ_wzlb[:,1:t]) zeros(size(ϵ_wzlb,1), size(ϵ_wzlb,2)-t)], periods = 0, initial_state = ℱ.value.(state[:,1]+solution[1]),levels = true))
         #MacroModelling.plot_irf(m,shocks = [ϵ[:,1:t]  zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],variables = zlbvar)
         hit = vec(collect(PLM(zlbvar,2:size(fgshlist, 1)+1,:Shock_matrix))).<zlblevel
         spellt = findall(!iszero,hit)
         #shocks[end-3:end,:] .= 0
         #shocks[:,2:end] .= 0
         conditions = Matrix{Union{Nothing,Float64}}(undef,m.timings.nVars,m.timings.nExo)
         conditions[zlbindex, spellt] = collect(ℱ.value.(-PLM(zlbvar,findall(!iszero,hit).+1,:Shock_matrix).+zlblevel) )
         
         # timingtarget = findall(vec(target .!= nothing))
         A = @views solution[2][:,1:m.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(m.timings.nVars))[m.timings.past_not_future_and_mixed_idx,:]
         # A = [:,1:m.timings.nPast_not_future_and_mixed]
         Comp =ℱ.value.(@views m.solution.perturbation.first_order.solution_matrix[:,m.timings.nPast_not_future_and_mixed+1:end])
         for jj =1:size(conditions,2)-1
             Comp = [Comp; A*Comp[end-m.timings.nVars+1:end,:] ]
         end
         ## IPOPT to solve for FG shocks
         model = Model(Ipopt.Optimizer)
         set_attribute(model, "max_cpu_time", 60.0)
         set_attribute(model, "print_level", 0)
         @variable(model, x[1:length(fgshlist)] .>= 0)  
         @objective(model, Min, sum(abs2,x))
 
         # println( Comp[ only(zlbindex) : m.timings.nVars : end, :] |> typeof)
         # println( x |> typeof)
 
         # println( ℱ.value.(ϵ[indexin(setdiff( m.timings.exo,fgshlist), m.timings.exo),2]) |> typeof)
         # println( ℱ.value.(solution[1][only(zlbindex)]) |> typeof)
         # println( ℱ.value.(Comp[ only(zlbindex) : m.timings.nVars : end, :]) * [ ℱ.value.(x) ; ℱ.value.(ϵ[indexin(setdiff( m.timings.exo,fgshlist), m.timings.exo),2])].+ ℱ.value.(solution[1][only(zlbindex)]))
         
         @constraint(model, ℱ.value.(Comp[ only(zlbindex) : m.timings.nVars : end, :]) * [x ; ℱ.value.(ϵ[indexin(setdiff( m.timings.exo,fgshlist), m.timings.exo),2])].+ ℱ.value.(solution[1][only(zlbindex)]).>= ℱ.value.(zlblevel))
         optimize!(model)
             
         ϵ[:,t] = [ℱ.value.(JuMP.value.(x)) ; ϵ[indexin(setdiff( m.timings.exo,fgshlist), m.timings.exo),t]]
         if t == 1
             state = zeros(typeof(initial_conditions[1]), m.timings.nVars, periods)
             aug_state = [initial_conditions
                 1
                 ϵ[:, t]]
         
             state[:, 1] .= 𝐒₁ * aug_state #+ solution[3] * ℒ.kron(aug_state, aug_state) / 2 
         else
             aug_state = [state[m.timings.past_not_future_and_mixed_idx, t-1]
                 1
                 ϵ[:, t]]
             state[:, t] .= 𝐒₁ * aug_state #+ solution[3] * ℒ.kron(aug_state, aug_state) / 2 
         end
     end
      observables_index = sort(indexin(observables, m.timings.var))
 
      state_deviations = data - state[vec(observables_index),:] .- solution[1][observables_index]
 
      Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
 end
 
 observables = [:INT, :YGR , :INFL ]
 
 observables_index = sort(indexin(observables,AS07.timings.var))
 
 data= collect(simulated_data[observables_index,:,1])
 
 #zlbvar = [:INT]
 #mpsh = [:epsr]
 #m = AS07
 #fgshlist = [:epsf1x, :epsf2x, :epsf3x, :epsf4x, :epsf5x, :epsf6x, :epsf7x,:epsf8x ,:epsf9x, :epsf10x,:epsf11x,:epsf12x, :epsf13x, :epsf14x, :epsf15x, :epsf16x ]
 #observables_index = sort(indexin(observables, m.timings.var))
 
 
 loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(AS07, data, observables, Ω, zlbvar, zlblevel,fgshlist) # m, data, observables, Ω , zlbvar, zlblevel,fgshlist  # Filter free
 
 n_samples = 100
 samps_ff = Turing.sample(loglikelihood_scaling_ff, Turing.NUTS(), n_samples, progress = true)#, init_params = sol
 
 StatsPlots.plot(samps_ff)
 
 
                
Turing.@model function loglikelihood_scaling_function_ff(m, data, observables, Ω, KF_shocks) 

    # RA,              1,             1e-5,        10,          gamma_pdf,     0.8,        0.5;
        # PA,              3.2,           1e-5,        20,          gamma_pdf,     4,          2;
        # GAMQ,            0.55,          -5,          5,           normal_pdf,    0.4,        0.2;
        # TAU,             2,             1e-5,        10,          gamma_pdf,     2,          0.5;
        # NU,              0.1,           1e-5,        0.99999,     beta_pdf,      0.1,        0.05;
        # PSIP,            1.5,           1e-5,        10,          gamma_pdf,     1.5,        0.25;
        # PSIY,            0.125,         1e-5,        10,          gamma_pdf,     0.5,        0.25;
        # RHOR,            0.75,          1e-5,        0.99999,     beta_pdf,      0.5,        0.2;
        # RHOG,            0.95,          1e-5,        0.99999,     beta_pdf,      0.8,        0.1;
        # RHOZ,            0.9,           1e-5,        0.99999,     beta_pdf,      0.66,       0.15;
        # SIGR,            0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
        # SIGG,            0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # SIGZ,            0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # C_o_Y,           0.85,          1e-5,        0.99999,     beta_pdf,      0.85,       0.1;
        # PHI,             50,            1e-5,        100,         gamma_pdf,     50,         20;
        # stderr epsz,     0.3,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # stderr epsr,     0.2,           1e-8,        5,           inv_gamma_pdf, 0.3,        4;
        # stderr epsg,     0.6,           1e-8,        5,           inv_gamma_pdf, 0.4,        4;
        # OMEGA,           0,             -10,         10,          normal_pdf,      0,        1;
        # XI,              1,             0,           2,           uniform_pdf,      ,         ,                    0,                   2;
    
    
    # RA ~  MacroModelling.Normal(1,0.5)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
        PA ~  MacroModelling.Normal(2.,0.1)    #  MacroModelling.Gamma(4.,2.,μσ = true)
        RA =2.0
    #  RA ~  MacroModelling.Gamma(1.,1,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #  PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    #  GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    #  TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
    #  NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    #  PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    #  PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    #  RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    #  RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    #  RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    #  SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    #  SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    #  SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    #  OMEGA ~  MacroModelling.Normal(0.,0.2)
        
    
    #RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    #PA ~  MacroModelling.Gamma(2.,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
    TAU	~  MacroModelling.Gamma(2.,0.2,μσ = true)
    NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
    RHOR 	~ MacroModelling.Beta( 0.5,0.2,μσ = true)
    RHOG 	~ MacroModelling.Beta( 0.8,0.1,μσ = true)
    RHOZ 	~ MacroModelling.Beta( 0.66,0.15,μσ = true)
    SIGR ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5., μσ = true)
    SIGG ~ MacroModelling.InverseGamma( 0.3,4., 10^(-8), 5.,μσ = true)
    SIGZ ~ MacroModelling.InverseGamma( 0.4,4., 10^(-8), 5.,μσ = true)
    # OMEGA ~  MacroModelling.Normal(0.,0.2)
    #XI ~  Turing.Uniform(0,1)
    C_o_Y	~ MacroModelling.Beta(0.85,0.1,μσ = true)
    
    # scale ~ MacroModelling.Gamma(1.,0.2,μσ = true)
    scale = 0.1
    # SIGFG ~  MacroModelling.InverseGamma( 0.1,4., 10^(-8), 5., μσ = true)
    # SIGRN ~  MacroModelling.InverseGamma( 0.001,1., 10^(-8), 1., μσ = true)
    
    RHORN = 0.
    # RHORN ~ MacroModelling.Beta( 0.9,0.1,μσ = true)
    # RA = 1
    # PA = 3.2
    # GAMQ = 0
    # TAU = 2
    # NU = 0.1
    KAPPA   = 0.33
    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
    
    # PSIP = 1.5
    # PSIY = 0.125
    # RHOR = 0.75
    # RHOG = 0.95
    # RHOZ = 0.9
    # SIGR = 0.2
    # SIGG = 0.6
    # SIGZ = 0.3
    # C_o_Y = 0.85
    SIGRN =0.001
    OMEGA = 0
    XI = 1
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, RHORN, SIGRN]
    
    algorithm = :first_order
    
        # Shock distribution 
        shock_distribution = Turing.Normal() 
        solution = get_solution(m, parameters, algorithm = algorithm)
    
        if solution[end] != true
            return Turing.@addlogprob! Inf
        end
    
        # draw_shocks(m)
        
        x0 ~ Turing.filldist(shock_distribution, m.timings.nVars) # Initial conditions - Normal!
                #  x0 = rand(shock_distribution,10,1)
        calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
    
        long_run_covariance = calculate_covariance_(solution[2])
    
        initial_conditions = long_run_covariance * x0
    
        𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
        # ϵ_draw ~ Turing.filldist(shock_distribution, (m.timings.nExo-1) * size(data, 2))
         
        # ϵ_draw ~ Turing.filldist(shock_distribution, (m.timings.nExo) * size(data, 2))
        # ϵ_draw = rand(shock_distribution, (m.timings.nExo-1) * size(data, 2))
        DF = 0.
        skew_distribution =  Turing.SkewNormal(0,1,DF)
        DF_out =  zeros(eltype(skew_distribution),size(data, 2))
        DF_out[1] ~  MacroModelling.Normal(DF,0.00000001) 
        
        ϵ =  @ignore_derivatives zeros(eltype(skew_distribution), m.timings.nExo, size(data, 2))
         
    
       # kk = size(data, 2)
        for t in 1:size(data, 2)
            for shki in 2:4
             # kk = kk+1
            # ϵ_draw[kk]  ~ Turing.Normal(only(KF_shocks[shki,t]), eps())
            ϵ[shki, t]  = @ignore_derivatives only(KF_shocks[shki,t])
            end
        end
        
        ϵ_skewdraw = zeros(eltype(skew_distribution),1,size(data, 2))
        ϵ_skewdraw[1] ~ skew_distribution
    
        state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
    
        ϵ_skewdraw[1] ~ skew_distribution
            # ϵ_skewdraw[1] = rand(skew_distribution,1,1)
        ϵ[:,1] =  @ignore_derivatives [ϵ_skewdraw[1]; ϵ[2:end,1]]
    
        #  aug_state = [initial_conditions
        #             1 
        #             ϵ[:,1]]
    
        state[:,1] .=  𝐒₁ * initial_conditions# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    
        for t in 2:size(data, 2)
            DF = DF + only(scale*rand(shock_distribution,1,1) )
            DF_out[t] ~  MacroModelling.Normal(DF,0.00000001) 
            skew_distribution =  Turing.SkewNormal(0,1,DF)
            ϵ_skewdraw[t] ~ skew_distribution
            #  ϵ_skewdraw[1] = rand(skew_distribution,1,1)
            ϵ[:,t] =  @ignore_derivatives [ϵ_skewdraw[t];  ϵ[2:end,t]]
            aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                        1 
                        ϵ[:,t]]
            state[:,t] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
        end
    
        observables_index = sort(indexin(observables, m.timings.var))
    
        state_deviations = data - state[observables_index,:] .- solution[1][observables_index]
    
        Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
    end

loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(AS07, data, observables,Ω,collect(KF_shocks)) 
n_samples = 100
samps_ff = Turing.sample(loglikelihood_scaling_ff,Turing.NUTS(),n_samples, progress = true)
                 


ff_estimated_parameters = Turing.describe(samps_ff)[1].nt.parameters
ff_estimated_means = Turing.describe(samps_ff)[1].nt.mean
ff_estimated_std = Turing.describe(samps_ff)[1].nt.std

ff_estimated_parameters_indices = indexin([Symbol("DF_out")], ff_estimated_parameters )
StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
StatsPlots.density!(ff_estimated_means[ff_estimated_parameters_indices])

StatsPlots.plot(ff_estimated_means[ff_estimated_parameters_indices],
                label = "Posterior mean", 
                title = "Joint: Estimated Latents")

