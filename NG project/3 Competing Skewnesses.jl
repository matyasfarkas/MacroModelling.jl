using MacroModelling, JuMP, Ipopt
import Turing, StatsPlots, Random, Statistics
using CSV, DataFrames, Dates
import LinearAlgebra as ℒ
using HypothesisTests, Distributions
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import ForwardDiff as ℱ

@model AS07 begin
	c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])
#	c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA+RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]
#	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA+RN[1]) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) 

	log(g[0]) = SIGG / 100 * epsg[x] + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	DYA[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	PIC[0] = PA/4 + 100 * log(p[0] / p[ss])

	R[0] = RA + PA + GAMQ * 4 + 400 * log(r[0] / r[ss]) # + RN[0]

    # RN[0] = RHORN * RN[-1] +  SIGRN / 100 * epsrn[x]
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

    #RHORN = 0.9

    #SIGRN = 0.01

    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2


end


# load data
dat = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/NG project/NAWM_dataset.csv", DataFrame)
data = KeyedArray(Array(dat[:,2:end])' ,Variable = (Symbol.(names(dat[:,2:end]))) , Time = 1:size(dat)[1] ) #Dates.DateTime.(dat[:,1], Dates.DateFormat("d-u-yyyy"))
observables = [:DYA, :R, :PIC]

# subset observables in data
data = data(observables,:)

Ω = 10^(-3)# eps()
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
 
    
    #RA ~  MacroModelling.Normal(1,0.5)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
    PA ~  MacroModelling.Normal(3.2,1)    #  MacroModelling.Gamma(4.,2.,μσ = true)
     
    # RA  =  2.0
    # RA ~  MacroModelling.Gamma(1.,1,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
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
    RA  =  1.5
    PA ~  MacroModelling.Gamma(2.,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
    GAMQ =0.33
    #GAMQ ~  MacroModelling.Normal(0.33,0.2)         #  MacroModelling.Normal(0.55,0.2) MacroModelling.Normal(0.4,0.2)
    TAU	~  MacroModelling.Gamma(2.,0.2,μσ = true)
    NU 	~ MacroModelling.Beta(0.1,0.05,μσ = true)
    PSIP ~  MacroModelling.Gamma(1.5,0.25,μσ = true)
    PSIY ~  MacroModelling.Gamma(0.5,0.25,μσ = true)
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
    scale = 0.001
    # SIGFG ~  MacroModelling.InverseGamma( 0.1,4., 10^(-8), 5., μσ = true)
    # SIGRN ~  MacroModelling.InverseGamma( 0.001,1., 10^(-8), 1., μσ = true)
    
    #RHORN = 0.
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
     RHOR = 0.75
     RHOG = 0.95
     RHOZ = 0.9
     SIGR = 0.2
     SIGG = 0.6
     SIGZ = 0.3
    C_o_Y = 0.85
    #SIGRN =0.001
    OMEGA = 0
    XI = 1
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI] #, RHORN, SIGRN] 
 
    algorithm = :first_order
      #LR Shock distribution 
      shock_distribution = Turing.Normal() 
      solution = get_solution(m, parameters, algorithm = algorithm)
  
      if solution[end] != true
           return Turing.@addlogprob! Inf
      end
  
      # In sample shock distribution  Horseshoe prior, source: https://discourse.julialang.org/t/regularized-horseshoe-prior/71599/2
      τ ~ truncated(Cauchy(0, 1); lower=0)
      η ~ truncated(Cauchy(0, 1); lower=0)
      λ ~ Turing.filldist(Cauchy(0, 1), m.timings.nExo)
      DF ~ MvNormal(ℒ.Diagonal(((η * τ) .* λ).^2)) # Coefficients
    
     x0 ~ Turing.filldist(shock_distribution, m.timings.nVars) # Initial conditions - Normal!
     ϵ_skewdraw ~ Turing.filldist(Turing.SkewNormal(0,1,0), m.timings.nExo,size(data,2))
     calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
 
     long_run_covariance = calculate_covariance_(solution[2])
    
     initial_conditions = long_run_covariance * x0
     state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))

     𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
     skew_distribution_shk_t =  Turing.SkewNormal(0,1,0)
     DF_out = zeros(eltype(DF),m.timings.nExo, size(data, 2))
     for shk in 1:m.timings.nExo
        DF_out[shk,1] ~  Turing.Uniform(-2,2)
        skew_distribution_shk_t =  Turing.SkewNormal(0,1,DF_out[shk,1])
        ϵ_skewdraw[shk,1] ~ skew_distribution_shk_t
    end
    ϵ = zeros(eltype(ϵ_skewdraw), m.timings.nExo, size(data, 2))
  
    ϵ[:,1] = ϵ_skewdraw[:,1]
     
     state[:,1] .=  𝐒₁ *[initial_conditions[m.timings.past_not_future_and_mixed_idx] 
     1 
     ϵ[:,1]]
      # + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

     for t in 2:size(data, 2)
        # For every t draw anothe tightnes sof the horseshoe prior. 
         τ ~ truncated(Cauchy(0, 1); lower=0)
         η ~ truncated(Cauchy(0, 1); lower=0)
         λ ~ Turing.filldist(Cauchy(0, 1), m.timings.nExo)
         DF_out[:,t] ~   MvNormal(0.999.*DF_out[:, t-1],scale.*(((η * τ) .* λ).^2))
        for shk in 1:m.timings.nExo
        # skew_distribution[shk,t] =  Turing.SkewNormal(0,1,DF_out[shk,t])
        
        ϵ_skewdraw[shk,t] ~ Turing.SkewNormal(0,1,DF_out[shk,t])
            #  ϵ_skewdraw[1] = rand(skew_distribution,1,1)
        end
        ϵ[:,t] =ϵ_skewdraw[:,t]
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
n_samples =20
samps_ff = Turing.sample(loglikelihood_scaling_ff,Turing.NUTS(),n_samples, progress = true)

ff_estimated_parameters = Turing.describe(samps_ff)[1].nt.parameters
ff_estimated_means = Turing.describe(samps_ff)[1].nt.mean
ff_estimated_std = Turing.describe(samps_ff)[1].nt.std

 
using Dates
dates = Date(1980, 9, 30):Month(3):Date(2023, 6, 30)
tm_ticks = round.(dates, Quarter(16)) |> unique;         

# AS07.exo has the shocks, let's collect them all in a vector.

ff_estimated_parameters_indices_shk1 = indexin([Symbol("DF_out[:,$a][1]") for a in 2:size(data,2)], ff_estimated_parameters )
ff_estimated_parameters_indices_shk2 = indexin([Symbol("DF_out[:,$a][2]") for a in 2:size(data,2)], ff_estimated_parameters )
ff_estimated_parameters_indices_shk3 = indexin([Symbol("DF_out[:,$a][3]") for a in 2:size(data,2)], ff_estimated_parameters )
ff_estimated_parameters_indices_shk4 = indexin([Symbol("DF_out[:,$a][4]") for a in 2:size(data,2)], ff_estimated_parameters )

                
# StatsPlots.plot(samps_ff)
      
StatsPlots.plot(dates, ff_estimated_means[ff_estimated_parameters_indices_shk1], ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices_shk1],label= "Estimated time-varying AD skewness",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))

p1=StatsPlots.plot(dates, ff_estimated_means[ff_estimated_parameters_indices_shk1], ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices_shk1],label= "Estimated time-varying AD skewness",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))
p2=StatsPlots.plot(dates, ff_estimated_means[ff_estimated_parameters_indices_shk2], ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices_shk2],label= "Estimated time-varying MP skewness",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))
p3=StatsPlots.plot(dates, ff_estimated_means[ff_estimated_parameters_indices_shk3], ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices_shk3],label= "Estimated time-varying r* skewness",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))
p4=StatsPlots.plot(dates, ff_estimated_means[ff_estimated_parameters_indices_shk4], ribbon = 1.96 * ff_estimated_std[ff_estimated_parameters_indices_shk4],label= "Estimated time-varying AS skewness",xticks=(tm_ticks, Dates.format.(tm_ticks, "yyyy")))


StatsPlots.plot(p1, p2, p3,p4, layout=(2,2), legend=false)

using JLD2


save("TEMMP.jld") 