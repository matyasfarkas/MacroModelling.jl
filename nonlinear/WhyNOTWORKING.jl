import Pkg; Pkg.instantiate();
using MacroModelling, LinearAlgebra, Zygote, Distributions, Symbolics, Plots, Random, StatsPlots
# Model declaration
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
end
sol = get_solution(RBC,RBC.parameter_values, algorithm = :second_order)
MOM1= get_moments(RBC,RBC.parameter_values)
LRvar= MOM1[2].^2 
x_iv= LRvar 

m = RBC
z=[ 0.062638   0.053282    0.00118333  0.442814   0.300381  0.150443  0.228132   0.382626   -0.0122483   0.0848671  0.0196158   0.197779    0.782655  0.751345   0.911694   0.754197   0.493297    0.0265917   0.209705    0.0876804;
-0.0979824  0.0126432  -0.12628     0.161212  -0.109357  0.120232  0.0316766  0.0678017  -0.0371438  -0.162375  0.0574594  -0.0564989  -0.18021   0.0749526  0.132553  -0.135002  -0.0143846  -0.0770139  -0.0295755  -0.0943254]

Ω_1 = 0.01

import LinearAlgebra as ℒ
import Turing
import Distributions
import Zygote
 
Turing.setadbackend(:zygote);  # Especially when we sample the latent noise, we will require high-dimensional gradients with reverse-mode AD

#Likelihood evaluation function using `Zygote.Buffer()` to create internal arrays that don't interfere with gradients.
function svlikelihood2(A, 𝐒₂, x_iv,Ω_1,observables,noise,volshocks,μ_σ,ρ_σ,σ_σ) #Accumulate likelihood
    # Initialize
    T = size(observables,2)

    u = Zygote.Buffer([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
    vol = Zygote.Buffer([zeros(1) for _ in 1:T]) #Fix type: Array of vector of vectors?
    #aug_state = Zygote.Buffer([[zeros(size(𝐒₁,2)+1),1] for _ in 1:T])
    u[1] = x_iv 
    vol[1] = [μ_σ] #Start at mean: could make random but won't for now
    
    for t in 2:T
            vol[t] = ρ_σ * vol[t-1] .+ (1 - ρ_σ) * μ_σ .+ σ_σ * volshocks[t - 1]
            aug_state = [u[t-1]
                        1 
                        exp.(vol[t]) .* noise[:,t-1]]
        # sol[3]'  |>Matrix
        u[t] =  A * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 
        # u[t] = A * u[t - 1] .+ exp.(vol[t]) .* (B * noise[t - 1])[:]
    end
    loglik = sum([Distributions.logpdf(Distributions.MvNormal(ℒ.Diagonal(Ω_1 * ones(size(observables,1)))), observables[t] .- ℒ.I * u[t][1:size(x_iv,1)]) for t in 1:T])
    return loglik
end

#= Checking gradient 
T = 20
Random.seed!(1234) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom
shockdist = TDist(dof) #Shocks are student-t


# Generate noise sequence
Random.seed!(1234) #Fix seed to reproduce data
noiseshocks = rand(shockdist,T,size(m.exo,1))
noise = Matrix(noiseshocks') # the ϵ shocks are "noise" in DifferenceEquations for SciML compatibility 

Random.seed!(1234) #Fix seed to reproduce data
volshocks = Matrix(rand(MvNormal(T,1.0))') # the volatility shocks are log-normal
Ω_1 = 0.1
Random.seed!(1234) #Fix seed to reproduce data
obsshocks = ℒ.reshape(rand(MvNormal(T*size(m.var,1),Ω_1)), size(m.var,1), T) #Gaussian observation noise

A = [sol[2][:,1:size(x_iv,1)] zeros(size(sol[2],1)) sol[2][:,size(x_iv,1)+1:end]]#


gradient(μ_σ->svlikelihood2(A, sol[3],x_iv,Ω_1,z,noise,volshocks,μ_σ,ρ_σ,σ_σ),1.)

=#

## Turing model definition
Turing.@model function rbc_1_svt_jointseq2(z, m, dof,Ω_1 )
    α ~ Turing.Uniform(0.2, 0.8)
    β ~ Turing.Uniform(0.5, 0.99)
    ρ = 0.2
    δ = 0.02
    σ = 0.01
    ρ_σ ~ Turing.Beta(2.625, 2.625) #Persistence of log volatility
    μ_σ ~ Turing.Normal(1., 0.5) #Mean of (prescaling) volatility
    σ_σ ~ Turing.Uniform(0.03, 0.3) #Volatility of volatility
    T = size(z, 2)
    xnought ~ Turing.filldist(Distributions.TDist(dof),1) #Initial shocks 
    #ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    #ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    x_iv= get_moments(m,[ σ,α, β, ρ, δ])[2].^2 .* xnought #scale initial condition to ergodic variance

    sol = get_solution(m,[σ, α, β, ρ, δ], algorithm = :second_order)
    
    ϵ_draw ~ Turing.filldist(Distributions.TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    vsdraw ~ Turing.MvNormal(T, 1.0)
    volshocks = ℒ.reshape(vsdraw,1,T)   
    A = [sol[2][:,1:size(x_iv,1)] zeros(size(sol[2],1)) sol[2][:,size(x_iv,1)+1:end]]#
    #sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
     Turing.@addlogprob! svlikelihood2(A, sol[3],x_iv,Ω_1,z,ϵ,volshocks,μ_σ,ρ_σ,σ_σ) 

end


turing_model3 = rbc_1_svt_jointseq2(z, RBC, 4, Ω_1 ) # passing observables from before 

n_samples = 100
n_adapts = 10
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)
chain_2_joint = Turing.sample(turing_model3, alg, n_samples; progress = true)


#Likelihood evaluation function using `Zygote.Buffer()` to create internal arrays that don't interfere with gradients.
function svlikelihoodCV(𝐒₁, 𝐒₂, x_iv,Ω_1,observables,noise) #Accumulate likelihood
    # Initialize
    T = size(observables,2)
    u = Zygote.Buffer([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
    # vol = Zygote.Buffer([zeros(1) for _ in 1:T]) #Fix type: Array of vector of vectors?
    u[1] = x_iv 
    𝐒₁ = [𝐒₁[:,1:size(x_iv,1)] zeros(size(𝐒₁,1)) 𝐒₁[:,size(x_iv,1)+1:end]]
    #vol[1] = [μ_σ] #Start at mean: could make random but won't for now
    for t in 2:T
        #vol[t] = ρ_σ * vol[t-1] .+ (1 - ρ_σ) * μ_σ .+ σ_σ * volshocks[t - 1]
        aug_state = [u[t-1]
                        1 
                        noise[:,t-1]]
        # sol[3]'  |>Matrix
        u[t] =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 
        # u[t] = A * u[t - 1] .+ exp.(vol[t]) .* (B * noise[t - 1])[:]
    end
    loglik = sum([Distributions.logpdf(Distributions.MvNormal(ℒ.Diagonal(Ω_1 * ones(size(observables,1)))), observables[t] .- ℒ.I * u[t][1:size(x_iv,1)-1]) for t in 1:T])
    return loglik
end
 

 m = RBC


 @model RBCSV begin
    1 / (- k[0] + (1 - δ) * k[-1] +  exp(z[-1]) * k[-1]^α ) - (β / (- k[1] + (1 - δ) * k[0] +  exp(z[0]) * k[0]^α )) * (α * exp(z[0]) * k[-1]^(α - 1) + (1 - δ)) =0
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] = ρ * z[-1] + σ[0] * EPSz[x]
    σ[0] =  (1-ρ_σ) * σ̄  + ρ_σ * σ[-1] + σ_σ* EPSzs[x]
end

@parameters RBCSV verbose = true begin 
    ρ_σ = 0.5
    σ̄  = 1
    σ_σ = 0.1
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
end
states = [:k, :z]
sol = get_solution(RBC,RBC.parameter_values, algorithm = :second_order)


# Turing model definition
Turing.@model function rbc_1_svt_jointseqSV(z, m, dof,Ω_1 )
    α ~ Turing.Uniform(0.2, 0.8)
    β ~ Turing.Uniform(0.5, 0.99)
    ρ = 0.2
    δ = 0.02
    ρ_σ ~ Turing.Beta(2.625, 2.625) #Persistence of log volatility
    μ_σ ~ Turing.Normal(1., 0.5) #Mean of (prescaling) volatility
    σ_σ ~ Turing.Uniform(0.03, 0.3) #Volatility of volatility
    σ̄ ~ Turing.Uniform(0,2)
    T = size(z, 2)
    xnought ~ Turing.filldist(Distributions.TDist(dof),1) #Initial shocks 
    #ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    #ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    sol = get_solution(m,[ ρ_σ,σ̄ ,σ_σ,α ,β,ρ,δ], algorithm = :second_order)#
    x_iv= get_moments(m,[ ρ_σ,σ̄ ,σ_σ,α ,β,ρ,δ])[2].^2 .* xnought #scale initial condition to ergodic variance
    
    ϵ_draw ~ Turing.filldist(Distributions.TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    #sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
    Turing.@addlogprob! svlikelihoodCV(sol[2], sol[3],x_iv,Ω_1,z,ϵ) 
end

turing_model3 = rbc_1_svt_jointseqSV(z, RBCSV, 4, Ω_1 ) # passing observables from before 

n_samples = 100
n_adapts = 10
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)
chain_2_joint = Turing.sample(turing_model3, alg, n_samples; progress = true)
