using MacroModelling
import Turing, StatsPlots, Random, Statistics
import LinearAlgebra as ℒ
using HypothesisTests, Distributions

@model AS07 begin
	c[0] ^ (-TAU) = r[0] * 1 / (1 + RA / 400) * c[1] ^ (-TAU) / p[1] / ((1 + GAMQ / 100) * z[1])

	1 = p[0] * PHI * (p[0] - p[ss]) + 1 / NU * (1 - (c[0] ^ (-TAU)) ^ (-1)) - PHI / (2 * NU) * (p[0] - p[ss]) ^ 2 + 1 / (1 + RA / 400) * PHI * p[1] * (p[1] - p[ss]) * y[1] * c[1] ^ (-TAU) / c[0] ^ (-TAU) / y[0]

	y[0] = c[0] + y[0] * (1 - 1 / g[0]) + y[0] * PHI / 2 * (p[0] - p[ss]) ^ 2

	r[0] = exp(SIGR / 100 * epsr[x]) * r[-1] ^ RHOR * (r[ss] * (p[0] / (1 + PA / 400)) ^ PSIP * (y[0] / ((1 - NU) ^ (1 / TAU) * g[0])) ^ PSIY) ^ (1 - RHOR) * exp(SIGFG / 100 * epsfg[0]) 

	log(g[0]) = SIGG / 100 * epsg[x] + RHOG * log(g[-1]) + (1 - RHOG) * ( - log(C_o_Y))

	log(z[0]) = XI * RHOZ * log(z[-1]) + SIGZ / 100 * epsz[x]

	YGR[0] = GAMQ + 100 * (log(y[0] / y[ss]) - log(y[-1] / y[ss]) + log(z[0] / z[ss]))

	INFL[0] = PA + 400 * log(p[0] / p[ss])

	INT[0] = RA + PA + GAMQ * 4 + 400 * log(r[0] / r[ss])

    epsfg[0] = epsf1[-1]

    epsf1[0] = epsf2[-1] + epsf1x[x] 
    
    epsf2[0] = epsf3[-1] + epsf2x[x] 

    epsf3[0] = epsf4[-1] + epsf3x[x] 

    epsf4[0] = epsf5[-1] + epsf4x[x] 

    epsf5[0] = epsf6[-1] + epsf5x[x] 

    epsf6[0] = epsf7[-1] + epsf6x[x] 

    epsf7[0] = epsf8[-1] + epsf7x[x] 

    epsf8[0] = epsf8x[x] 

end


@parameters AS07 begin
	RA = 1

	PA = 3.2

	GAMQ = 0.55

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

    SIGFG = 0.1

end


# draw shocks
Random.seed!(1)
periods = 40
shockdistR = Distributions.SkewNormal(0,1,6) #  Turing.Beta(10,1) #
shockdistother = Distributions.Normal(0,1)

shocksSK = rand(shockdistR,1,periods) #  shocks = randn(1,periods)
shocks = rand(shockdistother,2,periods) #  shocks = randn(1,periods)
shockstrue = [ zeros(8, periods); shocks[1,:]' ; -shocksSK; shocks[2,:]' ]

#shockstrue[9:11,:] =  shockstrue[9:11,:] ./ Statistics.std(shockstrue[9:11,:],dims = 2)  # antithetic shocks
#shockstrue =shockstrue .- Statistics.mean(shockstrue,dims=2) # antithetic shocks

#shocks /= Statistics.std(shocks)  # antithetic shocks
#shocks .-= Statistics.mean(shocks) # antithetic shocks
# Test for non-normality
        HypothesisTests.ExactOneSampleKSTest(vec(shocksSK),Turing.Normal(0,1))
        StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
        StatsPlots.density!(shocksSK')
        StatsPlots.density!(shockstrue[10,:])
# get simulation
simulated_data = get_irf(AS07,shocks = shockstrue, periods = 0, levels = true) #[1:3,:,:] |>collect #([:YGR ],:,:) |>collect

# plot simulation
MacroModelling.plot_irf(AS07,shocks = shockstrue, periods = 0)
#StatsPlots.plot(shocks')
Ω = 10^(-5)# eps()
n_samples = 1000


Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)
    
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

    RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)
	PA ~  MacroModelling.Gamma(4.,2.,μσ = true)
	GAMQ ~  MacroModelling.Normal(0.4,0.2)
	# TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
	# NU 	~ MacroModelling.Beta( 0.1,0.05,μσ = true)
   
    # RA = 1
	# PA = 3.2
	# GAMQ = 0.55
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
    SIGFG = 0.1

    observables = [:INT, :YGR , :INFL ]
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, SIGFG]

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

end

loglikelihood_scaling = loglikelihood_scaling_function(AS07, simulated_data(:,:,:Shock_matrix), [:INT, :YGR , :INFL ], Ω) # Kalman
samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)
#, init_params = sol


StatsPlots.plot(samps)
mean(samps[["RA"]])


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

    # RA ~  MacroModelling.Gamma(0.8,0.5,μσ = true)
	# PA ~  MacroModelling.Gamma(4,2,μσ = true)
	# GAMQ ~  MacroModelling.Normal(0.4,0.2)
	# TAU	~  MacroModelling.Gamma(2.,0.5,μσ = true)
	# NU 	~ MacroModelling.Beta( 0.1,0.05,μσ = true)
   
    RA = 1
	PA = 3.2
	GAMQ = 0.55
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
    SIGFG = 0.1
    
    data= simulated_data
    zlbvar = [:INT]
    zlblevel = 0
    mpsh = [:epsr]
    fgshlist = [:epsf1x, :epsf2x, :epsf3x, :epsf4x, :epsf5x, :epsf6x, :epsf7x, :epsf8x]

    zlbindex = sort(indexin(zlbvar, m.timings.var))
    zlbshindex = sort(indexin(fgshlist, m.timings.exo))
    mpshindex =  only(sort(indexin(mpsh, m.timings.exo)))

    observables = [:INT, :YGR , :INFL ]
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
     
     x0 = zeros(12,1)

     calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(m.timings.past_not_future_and_mixed_idx) ) # subset_indices = collect(1:m.timings.nVars))

     long_run_covariance = calculate_covariance_(solution[2])
    
     initial_conditions = long_run_covariance * x0

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
    ϵ_draw ~ Turing.filldist(shock_distribution, (m.timings.nExo-size(indexin(fgshlist, m.timings.exo),1)) * size(data, 2))

    ϵ_draw= rand( size(data, 2) *( m.timings.nExo-size(indexin(fgshlist, m.timings.exo),1)))

    # This needs to fix for arbitrary location of FG SHOCKS!
    ϵ = [zeros( size(indexin(fgshlist, m.timings.exo),1),size(data, 2) ) ; reshape(ϵ_draw, (m.timings.nExo-size(indexin(fgshlist, m.timings.exo),1)) , size(data, 2))]

   
    state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
   
    aug_state = [initial_conditions
                 1 
                 ϵ[:,1]]

    state[:,1] .=  𝐒₁ * aug_state #+ solution[3] * ℒ.kron(aug_state, aug_state) / 2 

   
    for t in 2:size(data, 2)
        aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                    1 
                    ϵ[:,t]]
        state[:,t] .=  𝐒₁ * aug_state #+ solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    
    end

    hit = zeros(size(data, 2),1)
    for t = 1:size(data, 2)
        if only(state[zlbindex,t])  - zlblevel <-eps() # .- solution[1][zlbindex...] 
            hit[t,1] = 1;
        #println("ZLB HIT!!")
        end
    end
    consthorizon = zeros(size(data, 2),1)
    for  t = 1:size(data, 2)-size(fgshlist,1)
        consthorizon[t,1] = sum(hit[t:t+size(fgshlist,1),1])
        # Check if horizon is longer than fg shocks - throw an error
        if maximum(vec(consthorizon))>size(fgshlist,1)
            return Turing.@addlogprob! Inf
            println("ZLB too long for model to solve!")
        end
    end
        
    println(["Model spent maximum " maximum(vec(consthorizon)) " horizns at the ZLB!!!"])

## Finding anticipated news shocks that implement the ZLB
    ϵ_wzlb = ϵ
        for hmax = size(fgshlist,1)+1:-1:1
            for t = 1:size(data, 2)
                if consthorizon[t,1] == hmax
                    zlb_ϵ = zeros(m.timings.nExo,hmax+1)
                    conditions = KeyedArray(-(state[zlbindex,t:t+hmax-1] .- (zlblevel)),Variables = zlbvar,Periods = collect(1:hmax))
                    shocks  = KeyedArray(zeros(m.timings.nExo-hmax-1,size(conditions,2)),Variables = setdiff(m.exo,[fgshlist[1:hmax]; mpsh]),Periods = collect(1:hmax)) 
                    #MacroModelling.plot_conditional_forecast(m,conditions,shocks = shocks)
                    zlb_ϵ = get_conditional_forecast(m, conditions, shocks =shocks)[m.timings.nVars+1:end,1:hmax+1] |> collect
                    ϵ_wzlb[:,t:t+hmax] = ϵ[:,t:t+hmax] +zlb_ϵ
                end

            
                if t == 1
                    state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))
                    aug_state = [initial_conditions
                            1 
                            ϵ_wzlb[:,t]]

                    state[:,1] .=  𝐒₁ * aug_state #+ solution[3] * ℒ.kron(aug_state, aug_state) / 2 
                else
                    aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                        1 
                        ϵ_wzlb[:,t]]
                    state[:,t] .=  𝐒₁ * aug_state #+ solution[3] * ℒ.kron(aug_state, aug_state) / 2 
                end
            
            end

            hit = zeros(size(data, 2),1)
            for t = 1:size(data, 2)
                if only(state[zlbindex,t])  - zlblevel <-eps() # .- solution[1][zlbindex...] 
                hit[t,1] = 1;
                #println("ZLB HIT!!")
                end
            end
            consthorizon = zeros(size(data, 2),1)
            for  t = 1:size(data, 2)-size(fgshlist,1)
                consthorizon[t,1] = sum(hit[t:t+size(fgshlist,1),1])
                # Check if horizon is longer than fg shocks - throw an error
                if maximum(vec(consthorizon))>size(fgshlist,1)
                    return Turing.@addlogprob! Inf
                    println("ZLB too long for model to solve!")
                end
            end
            # show(consthorizon')
        end

    
        

     observables_index = sort(indexin(observables, m.timings.var))
    
     state_deviations = data - state[observables_index,:] .- solution[1][observables_index...]

     Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

loglikelihood_scaling_ff = loglikelihood_scaling_function_ff(RBC, collect(simulated_data(:k,:,:Shock_matrix))', [:k], Ω) # ,kf_estimated_means, kf_estimated_std  # Filter free

n_samples = 5000
samps_ff = Turing.sample(loglikelihood_scaling_ff, Turing.NUTS(), n_samples, progress = true)#, init_params = sol
StatsPlots.plot(samps_ff)
