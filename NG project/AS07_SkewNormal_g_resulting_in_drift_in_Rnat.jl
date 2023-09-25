using MacroModelling
import Turing, StatsPlots , Plots, Random
import LinearAlgebra as ℒ
using HypothesisTests, Distributions, Statistics


# An Schorfheide model
@model AS07 begin
	c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA + RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA + RN[1] ) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) 

	log(g[0]) = SIGG / 100 * (epsg[x]) + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	YGR[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	INFL[0] = PA + 400 * log(p[0] / p[ss])

	INT[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    RN[0] = 0 * RHORN * RN[-1] +  0*SIGRN / 100 * epsrn[x]
end


@parameters AS07 begin
	RA = 1
	PA = 3.2
	GAMQ =  0 #0.55
	TAU = 2
	NU = 0.1
	KAPPA   = 0.33
	PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
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
    SIGRN =1
    RHORN = 0.99999999999
end


# draw from t scaled by approximate invariant variance) for the initial condition
m = AS07

solution = get_solution(m, m.parameter_values, algorithm = :second_order)

calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
long_run_covariance = calculate_covariance_(solution[2])

T =80
skew_param = 4

Random.seed!(12345) #Fix seed to reproduce data
shockdist = Turing.SkewNormal(0,1,skew_param) 


shocks = rand(Turing.Normal(0,1),4,T)
shocks[1,:] = rand(shockdist,1,T) #  shocks = randn(1,periods)

HypothesisTests.ExactOneSampleKSTest(vec(shocks[1,:]),Turing.Normal(0,1))
StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
StatsPlots.density!(shocks[1,:],dims=1) # antithetic shocks


simulated_data = get_irf(m,shocks = shocks, periods = 0, levels = true)#(:k,:,:) |>collect

StatsPlots.plot((simulated_data([:INT,:INFL, :YGR],:,:Shock_matrix)'),label = ["Short-term policy rate" "Inflation" "Real GDP growth"])

StatsPlots.plot((simulated_data([:RN],:,:Shock_matrix)'),label = ["Natural rate trend"])


# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function_normal(m, data, observables, Ω)

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

   
    RA ~  MacroModelling.Normal(1.,0.1)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)

	#PA ~  MacroModelling.Normal(3.2,0.01)    #  MacroModelling.Gamma(4.,2.,μσ = true)
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
	PA = 3.2
	GAMQ = 0
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
    
    SIGRN =1
    RHORN = 0.99999999999

    # PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

        algorithm = :second_order
        parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI,SIGRN,RHORN]
    
        Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    
       
end

Ω =  eps()
n_samples = 1000

loglikelihood_scaling = loglikelihood_scaling_function_normal(m, simulated_data(:,:,:Shock_matrix), [:YGR, :INFL, :INT], Ω) # Kalman
sampsn = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

StatsPlots.plot(sampsn)

using StatsBase
summ = summarystats(sampsn)
parameter_mean  = summ.nt.mean

data = simulated_data([:YGR, :INFL, :INT],:,:Shock_matrix)
get_estimated_shocks(AS07, data, parameters = parameter_mean)
filtered_states = get_estimated_variables(AS07, data)
filtered_shocks = get_estimated_shocks(AS07, data, parameters = parameter_mean)

StatsPlots.density(collect(filtered_shocks[1,:]), fill=(0, .5,:red), label = ["Filtered output growth shock"])
StatsPlots.density!(shocks[1,:],fill=(0, .5,:blue), label = ["True output growth shock"]) # antithetic shocks

	StatsPlots.plot((simulated_data([:r],:,:Shock_matrix)'),label = "True short-term policy rate")
	StatsPlots.plot!((filtered_states([:r],:)'),label = "Filtered short-term policy rate")

## Model with natural rate cycles

@model AS07_drift begin
	c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA + RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA + RN[1] ) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) 

	log(g[0]) = SIGG / 100 * (epsg[x]) + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	YGR[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	INFL[0] = PA + 400 * log(p[0] / p[ss])

	INT[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    RN[0] =  RHORN * RN[-1] +   SIGRN / 100 * epsrn[x]
end


@parameters AS07_drift begin
	RA = 1
	PA = 3.2
	GAMQ =  0 #0.55
	TAU = 2
	NU = 0.1
	KAPPA   = 0.33
	PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
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
    SIGRN =1
    RHORN = 0.9
end

m0 = AS07_drift

# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function_drift(m, data, observables, Ω)

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

   
    RA ~  MacroModelling.Normal(1.,0.1)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)

	#PA ~  MacroModelling.Normal(3.2,0.01)    #  MacroModelling.Gamma(4.,2.,μσ = true)
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
	PA = 3.2
	GAMQ = 0
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
    
	#SIGRN 	~ MacroModelling.InverseGamma( 1.,1. , 10^(-2), 5.,μσ = true)
	RHORN 	~ MacroModelling.Normal(0.9,0.1) 

    SIGRN =1
    #RHORN = 0.99999999999

    # PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

        algorithm = :second_order
        parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI,SIGRN,RHORN]
    
        Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
    
       
end

Ω =  eps()
n_samples = 1000

loglikelihood_scaling = loglikelihood_scaling_function_drift(m0, simulated_data(:,:,:Shock_matrix), [:YGR, :INFL, :INT], Ω) # Kalman
sampsdrift = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

StatsPlots.plot(sampsdrift)

using StatsBase, Plots
summ = summarystats(sampsdrift)
parameter_mean  = summ.nt.mean

data = simulated_data([:YGR, :INFL, :INT],:,:Shock_matrix)
get_estimated_shocks(AS07_drift, data, parameters = parameter_mean)
filtered_states = get_estimated_variables(AS07_drift, data)

StatsPlots.plot((simulated_data([:RN],:,:Shock_matrix)'.+1),label = ["Natural rate trend"],legend=:bottomleft)
StatsPlots.plot!( twinx(),(filtered_states([:RN],:)'.+parameter_mean[1]),color=:red, label = ["Natural rate trend filtered"])


StatsPlots.plot((simulated_data([:r],:,:Shock_matrix)'),label = ["True: short-term policy rate"])
StatsPlots.plot!((filtered_states([:r],:)'),label = ["Short-term policy rate filtered with rnat drift"])




# Model with mean drift adjustment
@model AS07_skewE begin
    c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA + RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA + RN[1] ) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) 

	log(g[0]) = SIGG / 100 * (epsg[x]+skew_param/sqrt(skew_param^2+1)*sqrt(2/3.1415926535897)) + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	YGR[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	INFL[0] = PA + 400 * log(p[0] / p[ss])

	INT[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    RN[0] = 0*RHORN * RN[-1] +  0* SIGRN / 100 * epsrn[x]


    
end

@parameters AS07_skewE begin
	RA = 1
	PA = 3.2
	GAMQ =  0 #0.55
	TAU = 2
	NU = 0.1
	KAPPA   = 0.33
	PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
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
    SIGRN =1
    RHORN = 0.9
    skew_param= 4
end

m1 = AS07_skewE

# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)
    
	skew_param ~ Turing.Uniform(1,8) # skew_param= 4

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

   
    # RA ~  MacroModelling.Normal(1.,0.01)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
	
	# PA ~  MacroModelling.Normal(3.2,0.0001)    #  MacroModelling.Gamma(4.,2.,μσ = true)
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

    RA = 1
	PA = 3.2
	GAMQ = 0
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
    
    SIGRN =1
    RHORN = 0.9

    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

        algorithm = :second_order
        parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI,SIGRN,RHORN, skew_param]
    
        Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
end

loglikelihood_scaling_skewcorr = loglikelihood_scaling_function(m1, simulated_data(:,:,:Shock_matrix),[:YGR, :INFL, :INT], Ω) # Kalman with skew adjustment
samps = Turing.sample(loglikelihood_scaling_skewcorr, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

StatsPlots.plot(samps)

using StatsBase
 summ = summarystats(samps)
 parameter_mean  = summ.nt.mean
data = simulated_data([:YGR, :INFL, :INT],:,:Shock_matrix)
filtered_shocks = get_estimated_shocks(AS07_skewE, data, parameters = parameter_mean)

StatsPlots.density(collect(filtered_shocks[1,:]), fill=(0, .5,:red), label = ["Filtered output growth shock"])
StatsPlots.density!(shocks[1,:],fill=(0, .5,:blue), label = ["True output growth shock"]) # true shocks



StatsPlots.density(collect(filtered_shocks[2,:]), fill=(0, .5,:red),label = "Filtered MP shocks")
StatsPlots.density!(shocks[2,:],fill=(0, .5,:blue),label = "True MP shocks" ) # true shocks


# Model with mean and drift
@model AS07_driftskewE begin
    c[0] ^ (-TAU) = r[0] * 1 / (1 + (RA + RN[1]) / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + (RA + RN[1] ) / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) 

	log(g[0]) = SIGG / 100 * (epsg[x]+skew_param/sqrt(skew_param^2+1)*sqrt(2/3.1415926535897)) + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	YGR[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	INFL[0] = PA + 400 * log(p[0] / p[ss])

	INT[0] = RA + RN[0] + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    RN[0] = RHORN * RN[-1] +  SIGRN / 100 * epsrn[x]


    
end

@parameters AS07_driftskewE begin
	RA = 1
	PA = 3.2
	GAMQ =  0 #0.55
	TAU = 2
	NU = 0.1
	KAPPA   = 0.33
	PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2
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
    SIGRN =1
    RHORN = 0.9
    skew_param= 4
end

m2 = AS07_driftskewE

# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)
    
	skew_param ~ Turing.Uniform(1,8) # skew_param= 4

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

   
    # RA ~  MacroModelling.Normal(1.,0.01)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
	
	# PA ~  MacroModelling.Normal(3.2,0.0001)    #  MacroModelling.Gamma(4.,2.,μσ = true)
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

    RA = 1
	PA = 3.2
	GAMQ = 0
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
    
    SIGRN =1
    RHORN = 0.9

    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

        algorithm = :second_order
        parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI,SIGRN,RHORN, skew_param]
    
        Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)
end

loglikelihood_scaling_skewcorr = loglikelihood_scaling_function(m2, simulated_data(:,:,:Shock_matrix),[:YGR, :INFL, :INT], Ω) # Kalman with skew adjustment
samps_driftskew = Turing.sample(loglikelihood_scaling_skewcorr, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

StatsPlots.plot(samps_driftskew)

using StatsBase
 summ = summarystats(samps_driftskew)
 parameter_mean  = summ.nt.mean
data = simulated_data([:YGR, :INFL, :INT],:,:Shock_matrix)
filtered_shocks = get_estimated_shocks(AS07_driftskewE, data, parameters = parameter_mean)
filtered_states = get_estimated_variables(AS07_driftskewE, data)

StatsPlots.plot((simulated_data([:RN],:,:Shock_matrix)'.+1),label = ["Natural rate"],legend=:bottomleft)
StatsPlots.plot!( (filtered_states([:RN],:)'.+1),color=:red, label = ["Natural rate filtered with trend and mean adjustment"])


StatsPlots.density(collect(filtered_shocks[1,:]), fill=(0, .5,:red), label = ["Filtered output growth shock"])
StatsPlots.density!(shocks[1,:],fill=(0, .5,:blue), label = ["True output growth shock"]) # antithetic shocks

plot_shock_decomposition(AS07_driftskewE, data)
