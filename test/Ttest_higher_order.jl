

using MacroModelling
import Turing, StatsPlots , Plots, Random
import LinearAlgebra as ℒ

@model RBC begin
    1 / (- k[0]  + (1 - δ ) * k[-1] + (exp(z[-1]) * k[-1]^α)) = (β   / (- k[+1]  + (1 - δ) * k[0] +(exp(z[0]) * k[0]^α))) * (α* exp(z[0]) * k[0] ^(α - 1) + (1 - δ));
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] =  ρ * z[-1] - σ* EPSz[x]
end

@parameters RBC verbose = true begin 
    σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.
end
solution = get_solution(RBC, RBC.parameter_values, algorithm = :second_order)

# draw from t scaled by approximate invariant variance) for the initial condition
m =RBC
calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
long_run_covariance = calculate_covariance_(solution[2])

T =20
initial_conditions_dist = Turing.MvNormal(zeros(m.timings.nPast_not_future_and_mixed),long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) #Turing.MvNormal(SS.data.data[m.timings.past_not_future_and_mixed_idx,1],long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions 
initial_conditions = ℒ.diag(rand(initial_conditions_dist, m.timings.nPast_not_future_and_mixed))
# long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] * randn(m.timings.nPast_not_future_and_mixed)
state = zeros(typeof(initial_conditions[1]),m.timings.nVars, T+1)
state_predictions = zeros(typeof(initial_conditions[1]),m.timings.nVars, T+1)

aug_state = [initial_conditions
1 
0]
ϵ = [0 0.2567178329209457 -1.1127581634083954 1.779713752762057 -1.3694068387087652 0.4598600006094857 0.1319461357213755 0.21210992474923543 0.37965007742056217 -0.36234330914698276 0.04507575971259013 0.2562242956767027 -1.4425668844506196 -0.2559534237970267 -0.40742710317783837 1.5578503125015226 0.05971261026086091 -0.5590041386255554 -0.1841854411460526 2.130480921373996 -0.4417061483171887]

𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])
state[:,1] =  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
state_predictions[:,1] =  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2

for t in 2:T+1
    aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                1 
                ϵ[:,t]]
    state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
end

observables_index = sort(indexin([:k, :z], m.timings.var))
data1 = state[observables_index,2:end]

aug_state = [initial_conditions
1 
0]
for t in 2:T+1
    aug_state = [state_predictions[m.timings.past_not_future_and_mixed_idx,t-1]
                1 
                0]
    state_predictions[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
end

state_deviations = data1[:,1:end] - state_predictions[observables_index,2:end]
sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data1, 2)])

data= [-0.02581238618841974 -0.024755946984579915 -0.03963947262410749 0.027976419590461075 -0.07055923608788306 -0.0019476175150486795 -0.01730477450896961 -0.027839086711853298 -0.04206436289102794 -0.0669079320041197 -0.0468524749184209 -0.04419553873722875 -0.05807085079907702 0.030853497507340906 0.06269835430368484 0.09192597827601115 -0.002387900684187729 -0.02406178637465169 0.007332086142065906 0.024523404933250753 -0.10532304736402888; -9.300233770305984e-6 -0.002569038375963518 0.010613773958891251 -0.015674382735842318 0.010559191839919187 -0.002486761638111019 -0.0018168136848359592 -0.0024844619844595463 -0.0042933931710975315 0.002764754457250321 0.0001021932943241629 -0.002541804297902194 0.013917307984925758 0.005342995834955419 0.005142870198769468 -0.014549929085261332 -0.0035071119196608764 0.004888619002323379 0.002819578211925202 -0.02074089357135492 0.0002688827689009011]

Turing.@model function loglikelihood_scaling_function(m, data, observables)
    #σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    #α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    #β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    #ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    #δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    #γ     ~ Turing.Normal(1, 0.05)
    σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1.

    solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :second_order)
    if solution[end] != true
        return Turing.@addlogprob! Inf
    end
        
    
    initial_conditions ~ Turing.filldist(Turing.TDist(4),m.timings.nPast_not_future_and_mixed) # Initial conditions 

    #xnought ~ Turing.filldist(Turing.Normal(0.,1.),m.timings.nPast_not_future_and_mixed) #Initial shocks
    #calculate_covariance_ = calculate_covariance_AD(solution[2], T = m.timings, subset_indices = collect(1:m.timings.nVars))
    # long_run_covariance = calculate_covariance_(solution[2])
    # initial_conditions = long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] * xnought
    #SS = get_steady_state(m,   parameters = (:σ => σ, :α => α, :β => β, :ρ => ρ, :δ => δ, :γ  => γ ), algorithm = :second_order)
    # initial_conditions ~  Turing.MvNormal(zeros(m.timings.nPast_not_future_and_mixed),long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions  # Turing.MvNormal(SS.data.data[m.timings.past_not_future_and_mixed_idx,1],long_run_covariance[m.timings.past_not_future_and_mixed_idx,m.timings.past_not_future_and_mixed_idx] ) # Initial conditions 

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
    #ϵ_draw ~ Turing.filldist(Turing.Normal(0,1), m.timings.nExo * size(data, 2)) #Shocks are Normally - distributed!

    ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))

    state = zeros(typeof(initial_conditions[1]),m.timings.nVars, size(data, 2)+1)

    # state[m.timings.past_not_future_and_mixed_idx,1] .= initial_conditions

    aug_state = [initial_conditions
    1 
    zeros( m.timings.nExo)]
    state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    for t in 2:size(data, 2)+1
        aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                    1 
                    ϵ[:,t-1]]
        state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    end

    observables_index = sort(indexin(observables, m.timings.var))

    state_deviations = data[:,1:end] - state[observables_index,2:end]
    #println(sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)] ))

    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

loglikelihood_scaling = loglikelihood_scaling_function(RBC, data,[:k,:z])

n_samples = 300
n_adapts = 50
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)

samps = Turing.sample(loglikelihood_scaling, alg, n_samples, progress = true)#, init_params = sol)



#Plot true and estimated latents to see how well we backed them out
noise = [0.2567178329209457 -1.1127581634083954 1.779713752762057 -1.3694068387087652 0.4598600006094857 0.1319461357213755 0.21210992474923543 0.37965007742056217 -0.36234330914698276 0.04507575971259013 0.2562242956767027 -1.4425668844506196 -0.2559534237970267 -0.40742710317783837 1.5578503125015226 0.05971261026086091 -0.5590041386255554 -0.1841854411460526 2.130480921373996 -0.4417061483171887]

symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(samps[:, [Symbol("ϵ_draw[$a]") for a in 1:20], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = Turing.describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
Plots.plot(ϵ_mean[1:end], ribbon=1.96 * ϵ_std[1:end], label="Posterior mean", title = "First-Order Joint: Estimated Latents")
Plots.plot!(noise', label="True values")


Plots.plot(data1')
Plots.plot!(data[:,2:end]')