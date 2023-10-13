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

	gdp_rgd_obs[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	cpiinf_obs[0] = PA/4 + 100 * log(p[0] / p[ss])

	ffr_obs[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

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
dat = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/NG project/GSW12_dataset.csv", DataFrame)
data = KeyedArray(Array(dat[:,2:end])' ,Variable = (Symbol.(names(dat[:,2:end]))) , Time = 1:size(dat)[1] ) #Dates.DateTime.(dat[:,1], Dates.DateFormat("d-u-yyyy"))
observables = [:gdp_rgd_obs, :ffr_obs, :cpiinf_obs]

# subset observables in data
data = data(observables,:)

Ω = 10^(-4)# eps()
n_samples = 100

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
dates = Date(1965, 3, 30):Month(3):Date(2022, 12, 30)
tm_ticks = round.(dates, Quarter(16)) |> unique;

StatsPlots.plot(dates, parameter_mean[1].+  parameter_mean[3]*4 .+ filtered_states(:RN),ribbon= (parameter_std[1]+parameter_std[3])*2,label= "Filtered natural rate estimate",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))
using JLD2


save("KF_HMC_US_FF.jld") 


Ω = 10^(-3)# eps()

Turing.@model function loglikelihood_scaling_function_ff_TWSK(m, data, observables, Ω) 
     
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
    # PA ~  MacroModelling.Normal(3.2,1)    #  MacroModelling.Gamma(4.,2.,μσ = true)
     
    # RA  =  2.0
    # RA ~  MacroModelling.Gamma(1.,1.,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
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
     RA =0.  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
     PA =2.    #  MacroModelling.Gamma(4.,2.,μσ = true)
     GAMQ =0.378
    # GAMQ ~  MacroModelling.Normal(0.33,0.2)         #  MacroModelling.Normal(0.55,0.2) MacroModelling.Normal(0.4,0.2)     # GAMQ ~  MacroModelling.Normal(0.55,0.2)         #  MacroModelling.Normal(0.4,0.2)
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
    # C_o_Y	~ MacroModelling.Beta(0.85,0.1,μσ = true)
 
    # scale ~ MacroModelling.Gamma(1.,0.2,μσ = true)
    scale = 0.001
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
    #  RHOR = 0.75
    #  RHOG = 0.95
    #  RHOZ = 0.9
    #  SIGR = 0.2
    #  SIGG = 0.6
    #  SIGZ = 0.3
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
     ϵ_draw ~ Turing.filldist(shock_distribution, (m.timings.nExo-1) * size(data, 2))
     
     # ϵ_draw = rand(shock_distribution, (m.timings.nExo-1) * size(data, 2))
     ϵ = zeros(eltype(ϵ_draw), m.timings.nExo, size(data, 2))
     ϵ_skewdraw = zeros(eltype(ϵ_draw),1,size(data, 2))
 
     ϵ[2:end,:] = reshape(ϵ_draw,(m.timings.nExo-1), size(data, 2))
 
     state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
     DF = 0.
     skew_distribution =  Turing.SkewNormal(0,1,DF)
 
    #  DF_out =zeros(eltype(skew_distribution), 1)
    #  DF_out ~  Turing.Uniform(-2,2)
     DF_out =zeros(eltype(skew_distribution), size(data, 2))
    #  DF_out[1] ~  MacroModelling.Normal(DF,eps()) 
    DF_out[1] ~  Turing.Uniform(-2,2)
    DF_out = zeros(eltype(skew_distribution), size(data, 2))
    
     ϵ_skewdraw[1] ~ skew_distribution
         # ϵ_skewdraw[1] = rand(skew_distribution,1,1)
     ϵ[:,1] = [ϵ_skewdraw[1]; ϵ[2:end,1]]
     
     skew_distributiont =  Turing.SkewNormal(0,1,DF_out[1])
     #  aug_state = [initial_conditions
     #             1 
     #             ϵ[:,1]]
 
     state[:,1] .=  𝐒₁ * initial_conditions# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
 
     for t in 2:size(data, 2)
         DF  = DF_out[t-1]
         DF_out[t] ~  MacroModelling.Normal(0.999*DF,scale*1) 
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
loglikelihood_scaling_fftwsk = loglikelihood_scaling_function_ff_TWSK(AS07, data, observables,Ω) # Kalman
n_samples =50
samps_fftwsk = Turing.sample(loglikelihood_scaling_fftwsk,Turing.NUTS(),n_samples, progress = true)

ff_estimated_parameters_twsk = Turing.describe(samps_fftwsk)[1].nt.parameters
ff_estimated_means_twsk = Turing.describe(samps_fftwsk)[1].nt.mean
ff_estimated_std_twsk = Turing.describe(samps_fftwsk)[1].nt.std

ff_estimated_parameters_indices_twsk = indexin([Symbol("DF_out[$a]") for a in 1:size(data,2)], ff_estimated_parameters_twsk )
StatsPlots.plot(ff_estimated_means_twsk[ff_estimated_parameters_indices_twsk],
                ribbon = 1.96 * ff_estimated_std_twsk[ff_estimated_parameters_indices_twsk], 
                label = "Posterior mean", 
                title = "Joint: Estimated Latents")
            
              
StatsPlots.plot(dates, ff_estimated_means_twsk[ff_estimated_parameters_indices_twsk], ribbon = 1.96 * ff_estimated_std_twsk[ff_estimated_parameters_indices_twsk],label= "Estimated time-varying AD skewness",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))
