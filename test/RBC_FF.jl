

using MacroModelling
import Turing, StatsPlots
import LinearAlgebra as ℒ




@model RBC begin
    1 / (- k[0] + (1 - δ) * k[-1] +  exp(z[-1]) * k[-1]^α ) - (β / (- k[1] + (1 - δ) * k[0] +  exp(z[0]) * k[0]^α )) * (α * exp(z[0]) * k[-1]^(α - 1) + (1 - δ)) =0
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] =  ρ * z[-1] - σ* ϵ[0]
    ϵ[0]= EPSz[x]
end

@parameters RBC verbose = true begin 
    σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
end

zsim = simulate(RBC)
zsim1 = hcat(zsim([:k,:z],:,:)...)
zdata = ℒ.reshape(zsim1,2,40)

Turing.@model function loglikelihood_scaling_function(m, data, observables)
    ρ = 0.2
    δ = 0.02
    σ = 0.01
    α     ~ MacroModelling.Beta(0.5, 0.1, μσ = true)
    β     ~ MacroModelling.Beta(0.95, 0.01, μσ = true)

    #    initial_conditions ~ Turing.filldist(Turing.TDist(4),m.timings.nPast_not_future_and_mixed) # Initial conditions 

    initial_conditions ~ Turing.filldist(Turing.Normal(0,1),m.timings.nPast_not_future_and_mixed) # Initial conditions 
    solution = get_solution(m, [σ, α, β, ρ, δ], algorithm = :second_order)

    if solution[end] != true
        return Turing.@addlogprob! Inf
    end

    𝐒₁ = hcat(solution[2][:,1:m.timings.nPast_not_future_and_mixed], zeros(m.timings.nVars), solution[2][:,m.timings.nPast_not_future_and_mixed+1:end])

    # ϵ_draw ~ Turing.filldist(Turing.TDist(4), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!
    ϵ_draw ~ Turing.filldist(Turing.Normal(0,1.0), m.timings.nExo * size(data, 2)) #Shocks are t-distributed!

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

import   Random
Random.seed!(54321)
n_samples = 1000

loglikelihood_scaling = loglikelihood_scaling_function(RBC, zdata,[:k,:z])
n_samples = 300
n_adapts = 50
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)

samps = Turing.sample(loglikelihood_scaling, alg, n_samples, progress = true)#, init_params = sol)


#Plot true and estimated latents to see how well we backed them out
noise = hcat(zsim([:ϵ],:,:)...)
symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(samps[:, [Symbol("ϵ_draw[$a]") for a in 1:40], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = Turing.describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
plot(ϵ_mean[2:end], ribbon=2 * ϵ_std[2:end], label="Posterior mean", title = "First-Order Joint: Estimated Latents")
plot!(noise', label="True values")
