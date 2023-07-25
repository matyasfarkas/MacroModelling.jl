#using Pkg
#Pkg.add(["JuMP", "Ipopt", "Turing", "StatsPlots", "Random","Statistics","LinearAlgebra","HypothesisTests","Distributions","ChainRulesCore","ForwardDiff"])

using MacroModelling, JuMP, Ipopt
import Turing, StatsPlots, Random, Statistics
import LinearAlgebra as ℒ
using HypothesisTests, Distributions
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import ForwardDiff as ℱ

# Brand Mazelis (2019) 
@model BM19 begin
    r[0] = i[0] - π[+1]
    i[0] = RHO * i[-1] + (1- RHO)* ( rstar[0] + pistar + RHOPI * ( π[0] - pistar )   + RHOY * ỹ[0] ) + SIGI/100 * ϵI[0] 
    ϵI[0] = epsI[x]
    rstar[0] = 4 * g[-1] + z[0]
    z[0] = 0.999999999999999*z[-1] + SIGZ/100 * ϵZ[0]
    ϵZ[0] = epsZ[x]
    g[0] = 0.999999999999999* g[-1] + SIGG/100* ϵG[0]
    ϵG[0] = epsG[x]
    π[0] = (1 - BPI) * pistar + BPI/2 * (π[-1] + π[-2])+ BY * ỹ[-1]  + SIGPI/100 *ϵPI[0]  
    ϵPI[0]= epsPI[x]
    ỹ[0] = AY1 * ỹ[-1] + AY2 * ỹ[-2] + AR/2 * (r[-1] + r[-2] - rstar[-1]  -rstar[-2] ) + SIGY/100* ϵY[0]
    ϵY[0] = epsY[x]
    ũ[0] = UY0 * ỹ[0]  + UY1 * ỹ[-1]  + UY2 * ỹ[-2]  + SIGU/100 * ϵu[0]
    ϵu[0]= epsu[x]
    #u[0] = u[-1] + sigUSTAR/100 * ϵustar[0]
    #ϵustar[0] = epsustar[x]
    #UOBS[0] = ũ[0] + u[0]
end



@parameters BM19 begin
    RHO =  0.69
    RHOPI = 0.1
    RHOY = 0.85
    
    AY1 = 1.15
    AY2 = -0.18
    AR  = -0.25

    BPI = 0.72
    BY = 0.13

    UY0 = -0.25
    UY1 = 0.
    UY2 = 0.

    SIGU = 10.
    SIGY = 53.
    SIGPI = 79.
    SIGI  = 82.
    SIGG = 7.
    SIGZ = 52.
    
    SIGYSTAR = 56.
    pistar = 2.
    sigUSTAR = 5.
end
plot_irf(BM19)


m = BM19
# draw shocks
Random.seed!(1)
periods = 20
shockdistR = Distributions.SkewNormal(0,1,3) #  Turing.Beta(10,1) #
shockdistother = Distributions.Normal(0,1)

shocksSK = rand(shockdistR,1,periods) #  shocks = randn(1,periods)
shocks = rand(shockdistother,5,periods) #  shocks = randn(1,periods)
shockstrue = [  shocks[1,:]' ; -shocksSK; shocks[2:end,:] ]

#shockstrue[9:11,:] =  shockstrue[9:11,:] ./ Statistics.std(shockstrue[9:11,:],dims = 2)  # antithetic shocks
shockstrue /= Statistics.std(shockstrue) # antithetic shocks
shockstrue =shockstrue .- Statistics.mean(shockstrue,dims=2)
#shocks /= Statistics.std(shocks)  # antithetic shocks
#shocks .-= Statistics.mean(shocks) # antithetic shocks
# Test for non-normality
        HypothesisTests.ExactOneSampleKSTest(vec(shocksSK),Turing.Normal(0,1))
        StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
        StatsPlots.density!(shocksSK')
        StatsPlots.density!(shockstrue[2,:])
# get simulation
simulated_data = get_irf(BM19,shocks = shockstrue, periods = 0, levels = true)#(:k,:,:) |>collect

MacroModelling.plot_irf(BM19,shocks = shockstrue, periods = 0)


Ω = 10^(-5)# eps()
n_samples = 1000

observables = [:i, :π , :ũ ,:ỹ]

observables_index = sort(indexin(observables,m.timings.var))

# define loglikelihood model - KF
Turing.@model function loglikelihood_scaling_function(m, data, observables, Ω)
    
    RHO ~ Turing.Uniform(0.0, 1.) #0.69
    #RHOPI ~ Turing.Uniform(0.0, 4.) #0.1
    RHOPI =0.1 #0.1
    RHOY ~ Turing.Uniform(0.0, 2.) #0.85
    
    AY1 ~ Turing.Uniform(0.0, 2.) # 1.15
    AY2 ~ Turing.Uniform(-1., 0.) # -0.18
    AR  ~ Turing.Uniform(-0.5, 0.) # -0.25

    #BPI ~ Turing.Uniform(-0.,1) # 0.72
    #BY  ~ Turing.Uniform(-0., 1)

    BPI = 0.72
    BY = 0.13
    UY0 = -0.25
    UY1 = 0.
    UY2 = 0.

    SIGU = 10.
    SIGY = 53.
    SIGPI = 79.
    SIGI  = 82.
    SIGG = 7.
    SIGZ = 52.
    
    SIGYSTAR = 56.
    pistar = 2.
    sigUSTAR = 5.
    
    algorithm = :first_order
    parameters = [RHO, RHOPI, RHOY, AY1, AY2, AR, BPI, BY, UY0, UY1, UY2, SIGU, SIGY, SIGPI, SIGI, SIGG, SIGZ, SIGYSTAR, pistar, sigUSTAR]
    shock_distribution = Turing.Normal()

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = parameters)

    # solution = get_solution(m, parameters, algorithm = algorithm)

    # if solution[end] != true
    #     return Turing.@addlogprob! Inf
    # end
    # # draw_shocks(m)
    # x0 ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
    
    # calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))

    # long_run_covariance = calculate_covariance_(solution[2])
    
    # initial_conditions = long_run_covariance * x0
    # # initial_conditions = x0

    # 𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    # ϵ_draw ~ Turing.filldist(shock_distribution, m.timings.nExo * size(data, 2))

    # ϵ = reshape(ϵ_draw, m.timings.nExo, size(data, 2))

    # state = zeros(typeof(initial_conditions[1]), m.timings.nVars, size(data, 2))

    # aug_state = [initial_conditions
    #             1 
    #             ϵ[:,1]]

    # state[:,1] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    # for t in 2:size(data, 2)
    #     aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
    #                 1 
    #                 ϵ[:,t]]

    #     state[:,t] .=  𝐒₁ * aug_state# + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    # end

    # observables_index = sort(indexin(observables, m.timings.var))
    
    # state_deviations = data - state[observables_index,:] .- solution[1][observables_index...]

    # Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(Ω * ℒ.I(size(data,1))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

loglikelihood_scaling = loglikelihood_scaling_function(m, simulated_data(observables,:,:Shock_matrix), observables, Ω) # Kalman
samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol

