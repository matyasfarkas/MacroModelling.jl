

using MacroModelling
import Turing, StatsPlots
import LinearAlgebra as ℒ





@model RBC begin
	K[0] = (1 - δ) * K[-1] + I[0]
	Y[0] = Z[0] * K[-1]^α
	Y[0] = C[0] + I[0]
	1 / C[0]^γ = β / C[1]^γ * (α * Y[1] / K[0] + (1 - δ))
	Z[0] = (1 - ρ) + ρ * Z[-1] + σ * ϵ[x]
end


@parameters RBC verbose = true begin 
    σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
    γ = 1
end

zsim = simulate(RBC)
zsim1 = hcat(zsim([:K,:Z],:,:)...)
zdata = ℒ.reshape(zsim1,2,40)

# z_rbc1 = hcat(zsim...)
# z_rbc1 = ℒ.reshape(z_rbc1,size(RBC.var,1),40)



solution = get_solution(RBC, RBC.parameter_values, algorithm = :second_order)

Turing.@model function loglikelihood_scaling_function(m, data, observables)
    σ     ~ MacroModelling.Beta(0.01, 0.02, μσ = true)
    α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)
    ρ     ~ MacroModelling.Beta(0.2, 0.1, μσ = true)
    δ     ~ MacroModelling.Beta(0.02, 0.05, μσ = true)
    γ     ~ Turing.Normal(1, 0.05)
    
    #    initial_conditions ~ Turing.filldist(Turing.TDist(4),m.timings.nPast_not_future_and_mixed) # Initial conditions 

    initial_conditions ~ Turing.filldist(Turing.Normal(0,1),m.timings.nPast_not_future_and_mixed) # Initial conditions 

    solution = get_solution(m, [σ, α, β, ρ, δ, γ], algorithm = :second_order)

    if solution[end] != true
        return Turing.@addlogprob! Inf
    end

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    # ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
    ϵ_draw ~ Turing.filldist(Turing.Normal(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

    ϵ = reshape(ϵ_draw, m.timings.nExo,  size(data, 2))

    state = zeros(typeof(initial_conditions[1]),m.timings.nVars, size(data, 2)+1)

    # state[m.timings.past_not_future_and_mixed_idx,1] .= initial_conditions

    aug_state = [initial_conditions
    1 
    ϵ[:,1]]
    state[:,1] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 

    for t in 2:size(data, 2)+1
        aug_state = [state[m.timings.past_not_future_and_mixed_idx,t-1]
                    1 
                    ϵ[:,t-1]]
        state[:,t] .=  𝐒₁ * aug_state + solution[3] * ℒ.kron(aug_state, aug_state) / 2 
    end

    observables_index = sort(indexin(observables, m.timings.var))

    state_deviations = data[:,1:end] - state[observables_index,2:end]

    Turing.@addlogprob! sum([Turing.logpdf(Turing.MvNormal(ℒ.Diagonal(ones(size(state_deviations,1)))), state_deviations[:,t]) for t in 1:size(data, 2)])
end

n_samples = 1000

loglikelihood_scaling = loglikelihood_scaling_function(RBC, zdata,[:K,:Z])

samps = Turing.sample(loglikelihood_scaling, Turing.NUTS(), n_samples, progress = true)#, init_params = sol)

