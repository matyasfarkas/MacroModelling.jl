using MacroModelling
import Turing
import Turing: NUTS, sample, logpdf
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL: logjoint
import ForwardDiff as ℱ
using StatsPlots 


include("../models/FS2000.jl")

# load data
dat = CSV.read("C:/Users/fm007/Documents/GitHub/MacroModelling.jl/test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)
m = FS2000
n_samples = 1000

## Code for baseline 

Turing.@model function FS2000_loglikelihood_function(data, m, observables)
    alp     ~ Beta(0.356, 0.02, μσ = true)
    bet     ~ Beta(0.993, 0.002, μσ = true)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    psi     ~ Beta(0.65, 0.05, μσ = true)
    del     ~ Beta(0.01, 0.005, μσ = true)
    z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)
    # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])   
    # UNCOMMENT BELOW TO APPLY NARRATIVE RESTRICTIONS
    # parameters_HVD= ℱ.value.([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]); # Need to store them as FWD Difference values to pass to the HVD to avoid error in Turing  
        #  try
        
        #     FVED_contribution_error = ℱ.value.(collect(get_conditional_variance_decomposition(m;  parameters = parameters_HVD)(vari,:,:)[shock,periods])).-narrative_target;
        #     Turing.@addlogprob! Turing.loglikelihood(Turing.truncated(Turing.Normal(0, 100); lower=0),FVED_contribution_error)
    
        # catch
        #     println("Narrative restriction cannot be satisfied. Adding -Inf to the LL function.")
        #     Turing.@addlogprob! -Inf
        # end

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, m, observables)

# using Zygote
# Turing.setadbackend(:zygote)
samps = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)#, init_params = sol)
periods = 10
shock =2
vari = :log_gp_obs;

hvdi = zeros(n_samples,1)
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps.value[i,1:9])
    hvdi[i] = get_conditional_variance_decomposition(FS2000, verbose = true, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,periods]
end
min(hvdi...)

density(hvdi,label= "Posterior distribution of FEVD - no constraint")
alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = mean(samps).nt.mean
hvdmean = get_conditional_variance_decomposition(FS2000, verbose = true, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,periods]
vline!([(hvdmean)], label= "Posterior mean of FEVD - no constraint")

hvdhoriz  = 1:10

# HVD distribuiton of the prior
hvdi_base = zeros(n_samples,1)
for i = 1:n_samples
    parameters_HVD = collect(samps.value[i,1:9])
    hvdi_base[i] = sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))
end

min(hvdi_base...)
density(hvdi_base,trim =true,label= "No importance restriction",legend=:bottomleft)
parameters_HVD= mean(samps).nt.mean
hvdmean_base = sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))
vline!([(hvdmean_base)], label= "Posterior mean")


## OVERWHELMING IMPORTANCE restriction
hvdi_overw_base = zeros(n_samples,1)
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps.value[i,1:9])
    hvdi_overw_base[i] = sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters =[alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz])))
end
using StatsPlots 
density(hvdi_overw_base,label= ["Posterior distribution of the ratio of sum of abs(HVDs) - no restriction"] )
alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = mean(samps).nt.mean
hvdmean_overw_base = sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz])))
vline!([(hvdmean_overw_base)], label= "Mean of posterior distribution of the ratio of sum of abs(HVDs)- no restriction")


println("*********************************************************************")
println("**************  ESTIMATION FINISHED FOR BASELINE  *******************")
println("*********************************************************************")

## CODE FOR NARRATIVE RESTRICTIONS
narrative_target = 0.1;
hvdhoriz  = 1:10
shock =2
vari = :log_gp_obs;
ndraws_w = 1000;
println("*********************************************************************")
println("*************  SETTINGS FOR NARRATIVE RESTRICTIONS  *****************")
println("*********************************************************************")

println("Variance target was set to: ", float(narrative_target*100),  " percent.")
println("HVD cummulated over periods: " , min(hvdhoriz...), " to ", max(hvdhoriz...), ".")
println("Structural shock selected: ", FS2000.timings.exo[shock])
println("Endogenous variable selected: ", vari)

println("*********************************************************************")
println("********************** INITIALIZING ESTIMATION **********************")
println("*********************************************************************")

Turing.@model function FS2000_loglikelihood_function(data, m, observables,narrative_target,hvdhoriz,shock,vari,ndraws_w)
    alp     ~ Beta(0.356, 0.02, μσ = true)
    bet     ~ Beta(0.993, 0.002, μσ = true)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    psi     ~ Beta(0.65, 0.05, μσ = true)
    del     ~ Beta(0.01, 0.005, μσ = true)
    z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)
    # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
    
    parameters_HVD= ℱ.value.([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]); # Need to store them as FWD Difference values to pass to the HVD to avoid error in Turing
    
    try
        cummulated_HVD_contribution_error = ℱ.value.(sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))).-narrative_target;
        Turing.@addlogprob! Turing.loglikelihood(Turing.truncated(Turing.Normal(0, 100); lower=0),cummulated_HVD_contribution_error)
 
        HVD_importance_error = ℱ.value.(sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz]))))-1.0;
        Turing.@addlogprob! Turing.loglikelihood(Turing.truncated(Turing.Normal(0, 100); lower=0),HVD_importance_error)
      
        # IMPORTANCE SAMPLING
        m.parameter_values = parameters_HVD;
        omega_hvd_cont = zeros(1,ndraws_w);omega_hvd_imp = zeros(1,ndraws_w);
        for j = 1:ndraws_w
        simulation = simulate(m, periods= size(data,2))
        omega_hvd_cont[j] = (ℱ.value.(collect(get_shock_decomposition(m,simulation(observables,:,:simulate))(vari,:,:)[shock,periods])).-narrative_target)>0
        omega_hvd_imp[j] =  ℱ.value.(sum(abs.(collect(get_shock_decomposition(m,simulation(observables,:,:simulate), parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,simulation(observables,:,:simulate), parameters = parameters_HVD)(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz]))))>1.0
        end
        wi  = sum(omega_hvd_cont.*omega_hvd_imp)/ndraws_w;
        if wi ==0.
            println("Likelihood weight simulation failed. Adding -Inf to the objective function.")
            Turing.@addlogprob! -Inf
        else
            Turing.@addlogprob!  +log(wi)
        end
    catch
        println("Narrative restriction cannot be satisfied. Adding -Inf to the objective function.")
        Turing.@addlogprob! -Inf
    end
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, observables,narrative_target,hvdhoriz,shock,vari,ndraws_w)

n_samples = 100

# using Zygote
# Turing.setadbackend(:zygote)
samps = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)#, init_params = sol)



## NARRATIVE RESTRICTION
hvdi = zeros(n_samples,1)
for i = 1:n_samples
    parameters_HVD = collect(samps.value[i,1:9])
    hvdi[i] = sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))
end

min(hvdi...)


density(hvdi,trim =true,label= "Importance restriction",legend=:bottomleft)
parameters_HVD= mean(samps).nt.mean
hvdmean = sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))
vline!([(hvdmean)], label= "Posterior mean of Importance restriction")



## OVERWHELMING IMPORTANCE restriction
hvdi = zeros(n_samples,1)
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps.value[i,1:9])
    hvdi[i] = sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters =[alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz])))
end
using StatsPlots 
density(hvdi,label= ["Posterior distribution of the ratio of sum of abs(HVDs)"] )
alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = mean(samps).nt.mean
hvdmean = sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz])))
vline!([(hvdmean)], label= "Mean of posterior distribution of the ratio of sum of abs(HVDs)")



hvdi_violin= zeros(n_samples,size(data,2))
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps.value[i,1:9])
    hvdi_violin[i,:] = get_shock_decomposition(FS2000,  data,parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,:]
end

violin(hvdi_violin, side=:left, linewidth=0, label="")

### 
narrative_target = 0.4;
hvdhoriz  = 1:10
shock =2
vari = :log_gp_obs;

println("*********************************************************************")
println("*************  SETTINGS FOR NARRATIVE RESTRICTIONS  *****************")
println("*********************************************************************")

println("Variance target was set to: ", float(narrative_target*100),  " percent.")
println("HVD cummulated over periods: " , min(hvdhoriz...), " to ", max(hvdhoriz...), ".")
println("Structural shock selected: ", FS2000.timings.exo[shock])
println("Endogenous variable selected: ", vari)

println("*********************************************************************")
println("********************** INITIALIZING ESTIMATION **********************")
println("*********************************************************************")

Turing.@model function FS2000_loglikelihood_function(data, m, observables,narrative_target,hvdhoriz,shock,vari)
    alp     ~ Beta(0.356, 0.02, μσ = true)
    bet     ~ Beta(0.993, 0.002, μσ = true)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    psi     ~ Beta(0.65, 0.05, μσ = true)
    del     ~ Beta(0.01, 0.005, μσ = true)
    z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)
    # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
    
    parameters_HVD= ℱ.value.([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]); # Need to store them as FWD Difference values to pass to the HVD to avoid error in Turing
    
    try
    
        cummulated_HVD_contribution_error = ℱ.value.(sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))).-narrative_target;
        Turing.@addlogprob! Turing.loglikelihood(Turing.truncated(Turing.Normal(0, 100); lower=0),cummulated_HVD_contribution_error)
 
        HVD_importance_error = ℱ.value.(sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz]))))-0.5;
        Turing.@addlogprob! Turing.loglikelihood(Turing.truncated(Turing.Normal(0, 100); lower=0),HVD_importance_error)
       # IMPORTANCE SAMPLING
       m.parameter_values = parameters_HVD
       simulation = simulate(m, periods= size(data,2))
      omega = ℱ.value.(collect(get_shock_decomposition(m,simulation(observables,:,:simulate))(vari,:,:)[shock,periods])).-narrative_target
     wi  ~ Normal(omega,eps)
    catch
        println("Narrative restriction cannot be satisfied. Adding -Inf to the objective function.")
        Turing.@addlogprob! -Inf
    end
    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_loglikelihood = FS2000_loglikelihood_function(data, FS2000, observables,narrative_target,hvdhoriz,shock,vari)

n_samples = 100

# using Zygote
# Turing.setadbackend(:zygote)
samps = sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)#, init_params = sol)



## NARRATIVE RESTRICTION
hvdi = zeros(n_samples,1)
for i = 1:n_samples
    parameters_HVD = collect(samps.value[i,1:9])
    hvdi[i] = sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))
end

min(hvdi...)


density(hvdi,trim =true,label= "Importance restriction",legend=:bottomleft)
parameters_HVD= mean(samps).nt.mean
hvdmean = sum(abs.(collect(get_shock_decomposition(m,data, parameters = parameters_HVD)(vari,:,:)[shock,hvdhoriz])))
vline!([(hvdmean)], label= "Posterior mean of Importance restriction")



## OVERWHELMING IMPORTANCE restriction
hvdi = zeros(n_samples,1)
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps.value[i,1:9])
    hvdi[i] = sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters =[alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz])))
end
using StatsPlots 
density(hvdi,label= ["Posterior distribution of the ratio of sum of abs(HVDs)"] )
alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = mean(samps).nt.mean
hvdmean = sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,hvdhoriz])))./sum(abs.(collect(get_shock_decomposition(m,data, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[setdiff(1:m.timings.nExo+1,shock),hvdhoriz])))
vline!([(hvdmean)], label= "Mean of posterior distribution of the ratio of sum of abs(HVDs)")



hvdi_violin= zeros(n_samples,size(data,2))
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps.value[i,1:9])
    hvdi_violin[i,:] = get_shock_decomposition(FS2000,  data,parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,:]
end

violin(hvdi_violin, side=:left, linewidth=0, label="")




## Code for FEVD 
periods = 10
narrative_target = 0.75
println("*********************************************************************")
println("*************  SETTINGS FOR NARRATIVE RESTRICTIONS  *****************")
println("*********************************************************************")

println("Variance target was set to: ", float(narrative_target*100),  " percent.")
println("FEVD measured at period: " , periods ,".")
println("Structural shock selected: ", FS2000.timings.exo[shock])
println("Endogenous variable selected: ", vari)

println("*********************************************************************")
println("********************** INITIALIZING ESTIMATION **********************")
println("*********************************************************************")

Turing.@model function FS2000_loglikelihood_function_FEVD(data, m, observables,narrative_target,periods,shock,vari)
    alp     ~ Beta(0.356, 0.02, μσ = true)
    bet     ~ Beta(0.993, 0.002, μσ = true)
    gam     ~ Normal(0.0085, 0.003)
    mst     ~ Normal(1.0002, 0.007)
    rho     ~ Beta(0.129, 0.223, μσ = true)
    psi     ~ Beta(0.65, 0.05, μσ = true)
    del     ~ Beta(0.01, 0.005, μσ = true)
    z_e_a   ~ InverseGamma(0.035449, Inf, μσ = true)
    z_e_m   ~ InverseGamma(0.008862, Inf, μσ = true)
    # println([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])   
    # UNCOMMENT BELOW TO APPLY NARRATIVE RESTRICTIONS
    parameters_HVD= ℱ.value.([alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m]); # Need to store them as FWD Difference values to pass to the HVD to avoid error in Turing  
         try
        
            FVED_contribution_error = ℱ.value.(collect(get_conditional_variance_decomposition(m;  parameters = parameters_HVD)(vari,:,:)[shock,periods])).-narrative_target;
            Turing.@addlogprob! Turing.loglikelihood(Turing.truncated(Turing.Normal(0, 100); lower=0),FVED_contribution_error)
    
        catch
            println("Narrative restriction cannot be satisfied. Adding -Inf to the LL function.")
            Turing.@addlogprob! -Inf
        end

    Turing.@addlogprob! calculate_kalman_filter_loglikelihood(m, data(observables), observables; parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])
end

FS2000_loglikelihood_FEVD = FS2000_loglikelihood_function_FEVD(data, m, observables,narrative_target,periods,shock,vari)

n_samples = 100

# using Zygote
# Turing.setadbackend(:zygote)
samps_FEVD = sample(FS2000_loglikelihood_FEVD, NUTS(), n_samples, progress = true)#, init_params = sol)


hvdi = zeros(n_samples,1)
for i = 1:n_samples
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = collect(samps_FEVD.value[i,1:9])
    hvdi[i] = get_conditional_variance_decomposition(FS2000, verbose = true, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,periods]
end
min(hvdi...)

density(hvdi,trim =true,label= "Posterior distribution of FEVD - with constraint")
alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = mean(samps_FEVD).nt.mean
hvdmean = get_conditional_variance_decomposition(FS2000, verbose = true, parameters = [alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m])(vari,:,:)[shock,periods]
vline!([(hvdmean)], label= "Posterior mean of FEVD - with constraint")






# println(mean(samps).nt.mean)

Random.seed!(30)

function calculate_posterior_loglikelihood(parameters)
    alp, bet, gam, mst, rho, psi, del, z_e_a, z_e_m = parameters
    log_lik = 0
    log_lik -= calculate_kalman_filter_loglikelihood(FS2000, data(observables), observables; parameters = parameters)
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



plot_model_estimates(FS2000, data, parameters = sol.minimizer)
MacroModelling.plot_shock_decomposition(FS2000, data)

FS2000 = nothing
m = nothing
# @profview sample(FS2000_loglikelihood, NUTS(), n_samples, progress = true)


# chain_NUTS  = sample(FS2000_loglikelihood, NUTS(), n_samples, init_params = FS2000.parameter_values, progress = true)#, init_params = FS2000.parameter_values)#init_theta = FS2000.parameter_values)

# StatsPlots.plot(chain_NUTS)

# parameter_mean = mean(chain_NUTS)

# pars = ComponentArray(parameter_mean.nt[2],Axis(parameter_mean.nt[1]))

# logjoint(FS2000_loglikelihood, pars)

# function calculate_log_probability(par1, par2, pars_syms, orig_pars, model)
#     orig_pars[pars_syms] = [par1, par2]
#     logjoint(model, orig_pars)
# end

# granularity = 32;

# par1 = :del;
# par2 = :gam;
# par_range1 = collect(range(minimum(chain_NUTS[par1]), stop = maximum(chain_NUTS[par1]), length = granularity));
# par_range2 = collect(range(minimum(chain_NUTS[par2]), stop = maximum(chain_NUTS[par2]), length = granularity));

# p = surface(par_range1, par_range2, 
#             (x,y) -> calculate_log_probability(x, y, [par1, par2], pars, FS2000_loglikelihood),
#             camera=(30, 65),
#             colorbar=false,
#             color=:inferno);


# joint_loglikelihood = [logjoint(FS2000_loglikelihood, ComponentArray(reduce(hcat, get(chain_NUTS, FS2000.parameters)[FS2000.parameters])[s,:], Axis(FS2000.parameters))) for s in 1:length(chain_NUTS)]

# scatter3d!(vec(collect(chain_NUTS[par1])),
#            vec(collect(chain_NUTS[par2])),
#            joint_loglikelihood,
#             mc = :viridis, 
#             marker_z = collect(1:length(chain_NUTS)), 
#             msw = 0,
#             legend = false, 
#             colorbar = false, 
#             xlabel = string(par1),
#             ylabel = string(par2),
#             zlabel = "Log probability",
#             alpha = 0.5);

# p
