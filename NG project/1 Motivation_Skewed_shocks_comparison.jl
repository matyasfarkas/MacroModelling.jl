#using Pkg; Pkg.add(["Turing", "ChainRulesCore","CSV","DataFrames","Dates","Distributions","ForwardDiff","HypothesisTests","LinearAlgebra","Random","Statistics","StatsPlots"])
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
    #    RHORN ~ MacroModelling.Beta( 0.9,0.1,μσ = true)
    RHORN ~ Turing.Uniform(-1,1)

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


# load US data
us_dat = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/NG project/US/sw07_data.csv", DataFrame)
us_data = KeyedArray(Array(us_dat[:,1:end])' ,Variable = (Symbol.(names(us_dat[:,1:end]))) , Time = 1:size(us_dat)[1] ) #Dates.DateTime.(dat[:,1], Dates.DateFormat("d-u-yyyy"))
us_data[2,:] = us_data(:R)*4 # Need to multipy Robs from SW07 with 100
observables = [:DYA, :R, :PIC]

# Estimate US model 
us_loglikelihood_scaling = loglikelihood_scaling_function(AS07, us_data, observables) # Kalman
us_samps = Turing.sample(us_loglikelihood_scaling,Turing.NUTS(),n_samples, progress = true)


StatsPlots.plot(us_samps)

using StatsBase
us_summ = summarystats(us_samps)
us_parameter_mean  = us_summ.nt.mean
us_parameter_std  = us_summ.nt.std

us_filtered_states = get_estimated_variables(AS07, us_data,parameters = us_parameter_mean)
plot_model_estimates(AS07,us_data(observables))
plot_shock_decomposition(AS07,us_data(observables))

us_dates = Date(1947, 6, 30):Month(3):Date(2023, 9, 30)
us_tm_ticks = round.(us_dates, Quarter(16)) |> unique;

StatsPlots.plot(us_dates, us_parameter_mean[1].+  us_parameter_mean[3]*4 .+ us_filtered_states(:RN),ribbon= (us_parameter_std[1]+us_parameter_std[3])*2,label= "Filtered natural rate estimate in the US",xticks=(us_tm_ticks, Dates.format.(us_tm_ticks, "yyyy")))






StatsBase.describe(KF_shocks[1,:])

skewness_KF_Rstar = zeros(size(KF_shocks,1))
for i in 1:size(KF_shocks,1)
skewness_KF_Rstar[i] = StatsBase.skewness(KF_shocks[i,:])
end

using JLD2
@save("AS07_TVRstar_KF.jld") 

using Plots

p1=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
p1=StatsPlots.density!((KF_shocks[1,:].-mean(KF_shocks[1,:]))./StatsBase.std(KF_shocks[1,:]), title = "Empirical demand shock distribuiton")

p2=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
p2=StatsPlots.density!((KF_shocks[2,:].-mean(KF_shocks[2,:]))./StatsBase.std(KF_shocks[2,:]), title = "Empirical MP shock distribuiton")

p3=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
p3=StatsPlots.density!((KF_shocks[3,:].-mean(KF_shocks[3,:]))./StatsBase.std(KF_shocks[3,:]), title = "Empirical r* shock distribuiton")

p4=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
p4=StatsPlots.density!((KF_shocks[4,:].-mean(KF_shocks[4,:]))./StatsBase.std(KF_shocks[4,:]), title = "Empirical supply shock distribuiton")

plot(p1, p2, p3, p4, layout=(2,2), legend=false)


# KF linear state space estimation with constant R
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
    
    RHORN = 0
    #RHORN ~ MacroModelling.Beta( 0.9,0.1,μσ = true)
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
samps_RC = Turing.sample(loglikelihood_scaling,Turing.NUTS(),n_samples, progress = true)

sumrc = summarystats(samps_RC)
parameter_mean_rc  = sumrc.nt.mean
parameter_std_rc  = sumrc.nt.std

KF_shocks_rc = get_estimated_shocks(AS07, data, parameters = parameter_mean_rc)
skewness_KF_RC = zeros(size(KF_shocks_rc,1))
for i in 1:size(KF_shocks_rc,1)
skewness_KF_RC[i] = StatsBase.skewness(KF_shocks_rc[i,:].- mean(KF_shocks_rc[2,:]))
end



p1=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
#p1=StatsPlots.density!((KF_shocks_rc[1,:].-mean(KF_shocks_rc[1,:]))./StatsBase.std(KF_shocks[1,:]), title = "Empirical demand shock distribuiton")
p1=StatsPlots.density!((KF_shocks_rc[1,:].-mean(KF_shocks_rc[1,:]))./StatsBase.std(KF_shocks[1,:]), title = "Empirical demand shock distribuiton")

p2=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
p2=StatsPlots.density!((KF_shocks_rc[2,:].-mean(KF_shocks_rc[2,:]))./StatsBase.std(KF_shocks[2,:]), title = "Empirical MP shock distribuiton")

p3=StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue), label ="Standard Normal distibution")
p3=StatsPlots.density!((KF_shocks_rc[4,:].-mean(KF_shocks_rc[4,:]))./StatsBase.std(KF_shocks[4,:]), title = "Empirical supply shock distribuiton")

plot(p1, p2, p3, layout=(2,2), legend=false)
