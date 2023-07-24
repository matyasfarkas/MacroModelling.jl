using MacroModelling, JuMP, Ipopt
import Turing, StatsPlots, Random, Statistics
import LinearAlgebra as ℒ
using HypothesisTests, Distributions
import ChainRulesCore: @ignore_derivatives, ignore_derivatives
import ForwardDiff as ℱ

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

    epsf8[0] = epsf9[-1]  + epsf8x[x] 
    
    epsf9[0] = epsf10[-1]  + epsf9x[x] 

    epsf10[0] = epsf11[-1]  + epsf10x[x] 

    epsf11[0] = epsf12[-1]  + epsf11x[x] 

    epsf12[0] = epsf13[-1]  + epsf12x[x] 

    epsf13[0] = epsf14[-1]  + epsf13x[x] 

    epsf14[0] = epsf15[-1]  + epsf14x[x] 

    epsf15[0] = epsf16[-1]  + epsf15x[x] 

    epsf16[0] = epsf16x[x] 


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

    SIGFG = 0.1

end


# draw shocks
Random.seed!(1)
periods = 20
shockdistR = Distributions.SkewNormal(0,1,2) #  Turing.Beta(10,1) #
shockdistother = Distributions.Normal(0,1)

shocksSK = rand(shockdistR,1,periods) #  shocks = randn(1,periods)
shocks = rand(shockdistother,2,periods) #  shocks = randn(1,periods)
shockstrue = [ zeros(16, periods); shocks[1,:]' ; -shocksSK; shocks[2,:]' ]

#shockstrue[9:11,:] =  shockstrue[9:11,:] ./ Statistics.std(shockstrue[9:11,:],dims = 2)  # antithetic shocks
shockstrue =shockstrue .- Statistics.mean(shockstrue,dims=2) # antithetic shocks

#shocks /= Statistics.std(shocks)  # antithetic shocks
#shocks .-= Statistics.mean(shocks) # antithetic shocks
# Test for non-normality
        HypothesisTests.ExactOneSampleKSTest(vec(shocksSK),Turing.Normal(0,1))
        StatsPlots.plot(Distributions.Normal(0,1), fill=(0, .5,:blue))
        StatsPlots.density!(shocksSK')
        StatsPlots.density!(shockstrue[10,:])
# get simulation
# simulated_data = get_irf(AS07,shocks = shockstrue, periods = 0, levels = true) #[1:3,:,:] |>collect #([:YGR ],:,:) |>collect

RA = 1

PA = 3.2

GAMQ = 0 #0.55

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
parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, SIGFG]
m = AS07
solution = get_solution(m, parameters, algorithm = :first_order)

# x0 = randn(m.timings.nPast_not_future_and_mixed) # Initial conditions # ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
x0 = zeros(m.timings.nPast_not_future_and_mixed,1) # Initial conditions # ~ Turing.filldist(Turing.Normal(), m.timings.nPast_not_future_and_mixed) # Initial conditions 
calculate_covariance_ = MacroModelling.calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(m.timings.past_not_future_and_mixed_idx) ) # subset_indices = collect(1:m.timings.nVars))
long_run_covariance = calculate_covariance_(solution[2])
initial_conditions = long_run_covariance * x0

𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
shockstrue[end-2:end, 1] = zeros(1,3)
ϵ = shockstrue

state = zeros(typeof(initial_conditions[1]), m.timings.nVars, periods)
aug_state = [initial_conditions
             1 
             ϵ[:,1]]
state[:,1] .=  𝐒₁ * aug_state#+ solution[3] * ℒ.kron(aug_state_unc, aug_state_unc) / 2 

zlbvar = [:INT]
zlbindex = sort(indexin(zlbvar, m.timings.var))
zlblevel = 0#-(RA + PA + GAMQ * 4)
mpsh = [:epsr]

fgshlist = [:epsf1x, :epsf2x, :epsf3x, :epsf4x, :epsf5x, :epsf6x, :epsf7x,:epsf8x ,:epsf9x, :epsf10x,:epsf11x,:epsf12x, :epsf13x, :epsf14x, :epsf15x, :epsf16x ]
fgstatelist = [:epsf1, :epsf2, :epsf3, :epsf4, :epsf5, :epsf6, :epsf7,:epsf8 ,:epsf9, :epsf10,:epsf11,:epsf12, :epsf13, :epsf14, :epsf15, :epsf16 ]
fgstateidx = sort(indexin(fgstatelist, m.timings.var))

state[fgstateidx,1] = zeros(size(fgstatelist,1),1)

for t in 2:periods
    #aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
    #            1 
    #           ϵ[:,t]]
    #state[:,t] .=  𝐒₁ * aug_state         #+ solution[3] * ℒ.kron(aug_state_unc, aug_state_unc) / 2 
    # Get unconditional FC
    PLM= get_irf(m,shocks = [ϵ[:,1:t] zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],levels = true)
    #MacroModelling.plot_irf(m,shocks = [ϵ[:,1:t]  zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],variables = zlbvar)
    hit = vec(collect(PLM(zlbvar,2:size(fgshlist, 1)+1,:Shock_matrix))).<zlblevel
    spellt = findall(!iszero,hit)
    zlb_ϵ = ℱ.value.(zeros(m.timings.nExo,1))
    shocks = Matrix{Union{Nothing,Float64}}(nothing,m.timings.nExo,10)
    #shocks[end-3:end,:] .= 0
    #shocks[:,2:end] .= 0
    conditions = Matrix{Union{Nothing,Float64}}(undef,m.timings.nVars,m.timings.nExo)
    conditions[zlbindex, spellt] = collect(ℱ.value.(-PLM(zlbvar,findall(!iszero,hit).+1,:Shock_matrix).+zlblevel) )
    
    target = conditions[zlbindex,:]
    # timingtarget = findall(vec(target .!= nothing))
    A = @views solution[2][:,1:m.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(m.timings.nVars))[m.timings.past_not_future_and_mixed_idx,:]
    # A = [:,1:m.timings.nPast_not_future_and_mixed]
    Comp = @views m.solution.perturbation.first_order.solution_matrix[:,m.timings.nPast_not_future_and_mixed+1:end]
    for t =1:size(conditions,2)-1
        Comp = [Comp; A*Comp[end-m.timings.nVars+1:end,:] ]
    end
    # Select conditining variables
    cond_var_idx = findall(vec(conditions) .!= nothing) # .-m.timings.nVars
    ## IPOPT to solve for FG shocks
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "max_cpu_time", 60.0)
    set_attribute(model, "print_level", 0)
    @variable(model, x[1:length(fgshlist)] .>= 0)  
    @objective(model, Min, sum(abs2,x))
    @constraint(model, Comp[ only(zlbindex) : m.timings.nVars : end, :] * [x ; ϵ[indexin(setdiff( m.timings.exo,fgshlist), m.timings.exo),2]] .+ solution[1][only(zlbindex)].>= zlblevel)
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
    # MacroModelling.plot_irf(m,shocks = [ϵ_fg_NLP[:,1:t]  zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1], variables = zlbvar)
    #PLM_cond= get_irf(m,shocks = [ϵ_fg_NLP[:,1:t] zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],levels = true)
    #StatsPlots.plot(PLM[2,:])
    #StatsPlots.plot!(PLM_cond[2,:])



#=    MacroModelling.plot_irf(m,shocks = [ϵ[:,1:t]  zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],variables = zlbvar)
    MacroModelling.plot_irf(m,shocks = [ϵ_fg[:,1:t]  zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],variables = zlbvar)

    PLM_cond_LP= get_irf(m,shocks = [ϵ_fg[:,1:t] zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],levels = true)
    StatsPlots.plot(PLM[2,:])
    StatsPlots.plot!(PLM_cond[2,:])
    PLM_cond[2,5:10] += solerr

    #shocks = ℱ.value.(KeyedArray(shocks, Variables=setdiff(m.exo), Periods=[1:size(shocks,2)]))
    #MacroModelling.plot_conditional_forecast(m,conditions,shocks = shocks)
    zlb_ϵ = get_conditional_forecast(m, conditions, shocks=shocks)[m.timings.nVars+1:end, 1] |> collect
    ## TO IMPLEMENT: get_functions 518 - stack CC into a "Canonical form of future shock IRF/MA, where [I*CC;A*CC; ... ; A^T*CC] is mapping the errors to the conditions" 
    ϵ_wzlb = ℱ.value.(ϵ)
=#

observables = [:INT, :YGR , :INFL ]

observables_index = sort(indexin(observables,AS07.timings.var))

simulated_data =  state[vec(observables_index),:] .+ solution[1][observables_index]
# plot simulation
StatsPlots.plot(simulated_data', label = ["INFL" "INT" "YGR"])

#MacroModelling.plot_irf(AS07,shocks = shockstrue, periods = 0)
#StatsPlots.plot(shocks')
Ω = 10^(-4)# eps()
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

   
    RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
	PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
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
    SIGFG = 0.1
    PHI = TAU*(1-NU)/NU/KAPPA/exp(PA/400)^2

    observables = [:INT, :YGR , :INFL ]
    parameters = [RA, PA, GAMQ, TAU, NU, KAPPA, PSIP, PSIY, RHOR, RHOG, RHOZ, SIGR, SIGG, SIGZ, C_o_Y, OMEGA, XI, SIGFG]

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data, observables; parameters = parameters)

end
data = KeyedArray(convert(Array{Float64,2},simulated_data); Variables=m.var[observables_index], Periods=range(1, step=1, length=periods)) #data= collect(simulated_data[observables_index,:,1])

loglikelihood_scaling = loglikelihood_scaling_function(AS07, data, observables, Ω) # Kalman
samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol


StatsPlots.plot(samps)




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

    RA ~  MacroModelling.Gamma(1.,0.5,μσ = true)  #  MacroModelling.Gamma(0.8,0.5,μσ = true)
	PA ~  MacroModelling.Gamma(3.2,2.,μσ = true)    #  MacroModelling.Gamma(4.,2.,μσ = true)
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
    
        # Get unconditional FC
        PLM= get_irf(m,shocks = [ϵ[:,1:t] zeros(size(shockstrue,1), size(shockstrue,2)-t)], periods = 0, initial_state = state[:,1]+solution[1],levels = true)
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
        Comp = @views m.solution.perturbation.first_order.solution_matrix[:,m.timings.nPast_not_future_and_mixed+1:end]
        for jj =1:size(conditions,2)-1
            Comp = [Comp; A*Comp[end-m.timings.nVars+1:end,:] ]
        end
        ## IPOPT to solve for FG shocks
        model = Model(Ipopt.Optimizer)
        set_attribute(model, "max_cpu_time", 60.0)
        set_attribute(model, "print_level", 0)
        @variable(model, x[1:length(fgshlist)] .>= 0)  
        @objective(model, Min, sum(abs2,x))
        @constraint(model, Comp[ only(zlbindex) : m.timings.nVars : end, :] * [x ; ϵ[indexin(setdiff( m.timings.exo,fgshlist), m.timings.exo),2]] .+ solution[1][only(zlbindex)].>= zlblevel)
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
