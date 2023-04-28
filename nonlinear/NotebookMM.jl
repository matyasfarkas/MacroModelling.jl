## SV volatility 
import Pkg; Pkg.instantiate();
# using MacroModelling, Distributions, Random
using MacroModelling, Distributions
import Random
import LinearAlgebra as ℒ
import Plots

# Model declaration
@model RBC begin
    1 / (- k[0] + (1 - δ) * k[-1] +  exp(z[-1]) * k[-1]^α ) - (β / (- k[1] + (1 - δ) * k[0] +  exp(z[0]) * k[0]^α )) * (α * exp(z[0]) * k[-1]^(α - 1) + (1 - δ)) =0
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
end

# Solving the model
sol = get_solution(RBC,RBC.parameter_values, algorithm = :second_order)
m = RBC
# Simulate T observations from a random initial condition
T = 50
Random.seed!(1234) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom
shockdist = TDist(dof) #Shocks are student-t

# draw from t scaled by approximate invariant variance) for the initial condition
x_iv = get_moments(m,m.parameter_values)[2].^2 .* rand(shockdist,size(RBC.var,1))
# LRvar[1:(size(m.var,1)-1),:] .* rand(shockdist,size(m.var,1)-1)

# Generate noise sequence
Random.seed!(1234) #Fix seed to reproduce data
noiseshocks = rand(shockdist,T,size(m.exo,1))
noise = Matrix(noiseshocks') # the ϵ shocks are "noise" in DifferenceEquations for SciML compatibility 

Random.seed!(1234) #Fix seed to reproduce data
volshocks = Matrix(rand(MvNormal(T,1.0))') # the volatility shocks are log-normal
Ω_1 = 0.1
Random.seed!(1234) #Fix seed to reproduce data
obsshocks = ℒ.reshape(rand(MvNormal(T*size(m.var,1),Ω_1)), size(m.var,1), T) #Gaussian observation noise

# Initialize states
x =([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
# x0 is the LR variance * a random
x[1] = x_iv
# Extract second order solution as a state space
𝐒₁ = sol[2]
𝐒₁ = [𝐒₁[:,1:size(x_iv,1)] zeros(size(𝐒₁,1)) 𝐒₁[:,size(x_iv,1)+1:end]]
𝐒₂ = sol[3]


ρ_σ = 0.5 #Persistence of log volatility
μ_σ = 1. #Mean of (prescaling) volatility
σ_σ = 0.1 #Volatility of volatility
vol = [zeros(1) for _ in 1:T]
vol[1] = [μ_σ] #Start at mean


#If needed 2rd order solution can be seen as  sol[3]'  |>Matrix
# 𝐒₃ = sol[4]

# Iterate forward state space to simulate the data
for t in 2:T
    ℒ.mul!(vol[t], ρ_σ, vol[t-1]) # Propate volatility process
    vol[t] .+= (1 - ρ_σ) * μ_σ  # Adds a drift to the volatility
    ℒ.mul!(vol[t], σ_σ, view(volshocks, :, t - 1),1,1) # Adds σ_σ * volshocks[t-1] to vol[t]

    aug_state = [x[t-1]
                    1 
                    exp.(vol[t]) .* noise[:,t-1]]
    x[t] =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 
end
z_rbc = hcat(x...)

for t in 1:T #Add observation noise
    z_rbc[:,t] .+= view(obsshocks,:,t)
end
Plots.plot(z_rbc' )

import Turing
using Zygote
import Turing: @addlogprob!
Turing.setadbackend(:zygote);  # Especially when we sample the latent noise, we will require high-dimensional gradients with reverse-mode AD



#Likelihood evaluation function using `Zygote.Buffer()` to create internal arrays that don't interfere with gradients.
function svlikelihood2(𝐒₁, 𝐒₂, x_iv,Ω_1,observables,noise,volshocks,μ_σ,ρ_σ,σ_σ) #Accumulate likelihood
    # Initialize
    T = size(observables,2)

    u = Zygote.Buffer([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
    vol = Zygote.Buffer([zeros(1) for _ in 1:T]) #Fix type: Array of vector of vectors?
    #aug_state = Zygote.Buffer([[zeros(size(𝐒₁,2)+1),1] for _ in 1:T])
    u[1] = x_iv 
    vol[1] = [μ_σ] #Start at mean: could make random but won't for now
    
    A = [𝐒₁[:,1:size(x_iv,1)] zeros(size(𝐒₁,1)) 𝐒₁[:,size(x_iv,1)+1:end]]#
    
    for t in 2:T
            vol[t] = ρ_σ * vol[t-1] .+ (1 - ρ_σ) * μ_σ .+ σ_σ * volshocks[t - 1]
            aug_state = [u[t-1]
                        1 
                        exp.(vol[t]) .* noise[:,t-1]]
        # sol[3]'  |>Matrix
        u[t] =  A * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 
        # u[t] = A * u[t - 1] .+ exp.(vol[t]) .* (B * noise[t - 1])[:]
    end
    loglik = sum([logpdf(MvNormal(ℒ.Diagonal(Ω_1 * ones(size(observables,1)))), observables[t] .- ℒ.I * u[t][1:size(x_iv,1)]) for t in 1:T])
    return loglik
end
 
# Turing model definition
Turing.@model function rbc_1_svt_jointseq(z, m, dof,Ω_1 )
    α ~ Turing.Uniform(0.2, 0.8)
    β ~ Turing.Uniform(0.5, 0.99)
    ρ = 0.2
    σ = 0.01
    δ = 0.02
    ρ_σ ~ Turing.Beta(2.625, 2.625) #Persistence of log volatility
    μ_σ ~ Normal(1., 0.5) #Mean of (prescaling) volatility
    σ_σ ~ Turing.Uniform(0.03, 0.3) #Volatility of volatility
    T = size(z, 2)
    xnought ~ Turing.filldist(TDist(dof),1) #Initial shocks 
    #ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    #ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    sol = get_solution(m,[σ,α, β, ρ, δ], algorithm = :second_order)#
    x_iv= get_moments(m,[σ,α, β, ρ, δ])[2].^2 .* xnought #scale initial condition to ergodic variance
    
    ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    vsdraw ~ MvNormal(T, 1.0)
    volshocks = reshape(vsdraw,1,T)   

    Turing.@addlogprob! svlikelihood2(sol[2], sol[3],x_iv,Ω_1,z,ϵ,volshocks,μ_σ,ρ_σ,σ_σ) 
end

turing_model = rbc_1_svt_jointseq(z_rbc, RBC, 4, Ω_1 ) # passing observables from before 

n_samples = 100
n_adapts = 20
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)
chain_2_joint = Turing.sample(turing_model, alg, n_samples; progress = true)

