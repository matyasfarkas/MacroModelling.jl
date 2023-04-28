## SV volatility 
import Pkg; Pkg.instantiate();

using MacroModelling, Distributions,Random
import LinearAlgebra as ℒ
import Plots
Random.seed!(1234) #Fix seed to reproduce data


#=
# Model declaration
@model RBC_CV begin
    1 / (- k[0] + (1 - δ) * k[-1] +  exp(z[-1]) * k[-1]^α ) - (β / (- k[1] + (1 - δ) * k[0] +  exp(z[0]) * k[0]^α )) * (α * exp(z[0]) * k[-1]^(α - 1) + (1 - δ)) =0
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] =  ρ * z[-1] - σ* EPSz[x]
end

@parameters RBC_CV verbose = true begin   
    σ = 0.1
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
end

# Solving the model
sol_CV = get_solution(RBC_CV,RBC_CV.parameter_values, algorithm = :first_order) #algorithm = :second_order
=#

## SV MODEL
# Model declaration
@model RBC begin
    1 / (- k[0] + (1 - δ) * k[-1] +  exp(z[-1]) * k[-1]^α ) - (β / (- k[1] + (1 - δ) * k[0] +  exp(z[0]) * k[0]^α )) * (α * exp(z[0]) * k[-1]^(α - 1) + (1 - δ)) =0
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] =  ρ * z[-1] + (σ[0]) * EPSz[x]
    σ[0]^2 =  (1-ρ_σ) * μ_σ + ρ_σ * σ[-1]^2 + σ_σ* EPSzs[x]
end

@parameters RBC verbose = true begin 
    ρ_σ = 0.5
    μ_σ = 0.001
    σ_σ = 0.01
    α = 0.5
    β = 0.95
    ρ = 0.2
    δ = 0.02
end

# Solving the model
# sol = get_solution(RBC,RBC.parameter_values, algorithm = :second_order) #algorithm = :second_order
# Simulating to compuet LR variance
# LRvar=get_moments(RBC,RBC.parameter_values)[2].^2 
m = RBC

sol = get_solution(m, algorithm = :second_order)#

zsim = simulate(RBC)
z_rbc1 = hcat(zsim...)
z_rbc1 = ℒ.reshape(z_rbc1,size(RBC.var,1),40)

Plots.plot(z_rbc1' )

# Re-Estimate the model 
import Turing
using Distributions
using Zygote

#Likelihood evaluation function using `Zygote.Buffer()` to create internal arrays that don't interfere with gradients.
function svlikelihood2(𝐒₁, 𝐒₂, x_iv,Ω_1,observables,noise) #Accumulate likelihood
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
    loglik = sum([logpdf(MvNormal(ℒ.Diagonal(Ω_1 * ones(size(observables,1)))), observables[:,t] .- ℒ.I * u[t][1:size(x_iv,1)]) for t in 1:T])
    return loglik
end
 

 m = RBC
# Turing model definition
Turing.@model function rbc_1_svt_jointseq2(z, m, dof,Ω_1 )
    α ~ Turing.Uniform(0.2, 0.8)
    β ~ Turing.Uniform(0.5, 0.99)
    ρ = 0.2
    δ = 0.02
    ρ_σ ~ Turing.Uniform(0.45, 0.55) #Persistence of log volatility
    μ_σ ~ Turing.Uniform(0, 0.002) #Mean of (prescaling) volatility
    σ_σ ~ Turing.Uniform(0, 0.1) #Volatility of volatility
    T = size(z, 2)
    xnought ~ Turing.filldist(TDist(dof),1) #Initial shocks 
    #ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    #ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    sol = get_solution(m,[ ρ_σ,μ_σ,σ_σ,α ,β,ρ,δ], algorithm = :second_order)#
    x_iv= get_moments(m,[ ρ_σ,μ_σ,σ_σ,α ,β,ρ,δ])[2].^2 .* xnought #scale initial condition to ergodic variance
    
    ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    #sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
    Turing.@addlogprob! svlikelihood2(sol[2], sol[3],x_iv,Ω_1,z,ϵ) 
end


Ω_1 = 0.0000001
turing_model3 = rbc_1_svt_jointseq2(z_rbc1[:,1:20], RBC, 4, Ω_1 ) # passing observables from before 

n_samples = 100
n_adapts = 10
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)
chain_2_joint = Turing.sample(turing_model3, alg, n_samples; progress = true)

#=


























#Likelihood evaluation function using `Zygote.Buffer()` to create internal arrays that don't interfere with gradients.
function svlikelihood(𝐒₁, 𝐒₂, x_iv,Ω_1,observables,noise) #Accumulate likelihood
    # Initialize
    T = size(observables,2)
    u = Zygote.Buffer([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
    u[1] = x_iv 
    𝐒₁ = [𝐒₁[:,1:size(x_iv,1)] zeros(size(𝐒₁,1)) 𝐒₁[:,size(x_iv,1)+1:end]]
    for t in 2:T
        # u[t] = A * u[t - 1] .+ exp.(vol[t]) .* (B * noise[t - 1])[:]
        #vol[t] = ρ_σ * vol[t-1] .+ (1 - ρ_σ) * μ_σ .+ σ_σ * volshocks[t - 1]
        aug_state = [u[t-1]
                        1 
                        noise[:,t-1]]
        # sol[3]'  |>Matrix
        u[t] =  𝐒₁ * aug_state + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 
    end
    loglik = sum([logpdf(MvNormal(ℒ.Diagonal(Ω_1 * ones(size(observables,1)))), observables[t] .- ℒ.I * u[t][1:size(x_iv,1)]) for t in 1:T])
    return loglik
end
 

# Turing model definition
Turing.@model function rbc_1_svt_jointseq(z, m, dof,Ω_1 )
    # Parameters
    α ~ Uniform(0.2, 0.8)
    β ~ Uniform(0.5, 0.99)
    ρ = 0.2
    δ = 0.02
    ρ_σ ~ Turing.Uniform(0.3, 0.7) #Persistence of log volatility
    σ_σ ~ Turing.Uniform(0.05, 0.15) #Volatility of volatility
    σ̄ = 1
    # Sample size
    T = size(z, 2)
    # Sample for initial condition   
    xnought ~ Turing.filldist(Normal(0,1),size(m.var,1)) #Initial shock
    #xnought ~ Turing.filldist(TDist(dof),size(m.var,1)) #Initial shocks
    x_iv= get_moments(m,[ρ_σ,σ̄ ,σ_σ,α,β,ρ,δ])[2].^2 .* xnought #scale initial condition to ergodic variance
    # Sample for the exogenous shocks
    ϵ_draw ~ Turing.filldist(Normal(0,1),size(m.exo,1) .* T) #Shocks are t-distributed!
    #ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    # Solve DSGE with perturbation given parameter draws 
    sol = get_solution(m,[ ρ_σ,σ̄ ,σ_σ,α ,β,ρ,δ], algorithm = :second_order)
    # Compute the log posterior using svlikelihood
    Turing.@addlogprob! svlikelihood(sol[2], sol[3],x_iv,Ω_1,z,ϵ) 
end
# Define zero ME
Ω_1 = 0
# Create the Turing model
turing_model3 = rbc_1_svt_jointseq(z_rbc1[:,10:25], m, 4, Ω_1 ) 

    # Estimate model
    n_samples = 100
    n_adapts = 10
    δ = 0.65
    alg = Turing.NUTS(n_adapts,δ)
    chain_2_joint = Turing.sample(turing_model3, alg, n_samples; progress = true)

   =#
   
   
## WORK IN PROGRESS - Q:Constant drift is missing from state update function.

# Simulate T observations from a random initial condition
T = 20
Random.seed!(1234) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom
shockdist = TDist(dof) #Shocks are student-t
# draw from t scaled by approximate invariant variance) for the initial condition
x_iv = zeros(3,1) # LRvar[1:(size(m.var,1)-1),:] .* rand(shockdist,size(m.var,1)-1)
x_iv[3] = 1 # Setting LR mean of the stoch volatility process
# Generate noise sequence
Random.seed!(1234) #Fix seed to reproduce data
noiseshocks = rand(shockdist,T,size(m.exo,1))
noise = Matrix(noiseshocks') # the ϵ shocks are "noise" in DifferenceEquations for SciML compatibility 

Random.seed!(1234) #Fix seed to reproduce data
volshocks = Matrix(rand(MvNormal(T,1.0))') # the volatility shocks are log-normal


noise[2,:] = volshocks
# Initialize states
x =([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
# x0 is the LR variance * a random
x[1] = x_iv
# Extract second order solution as a state space
𝐒₁ = sol[2]
𝐒₁ = [𝐒₁[:,1:size(x_iv,1)] zeros(size(𝐒₁,1)) 𝐒₁[:,size(x_iv,1)+1:end]]

# 𝐒₂ = sol[3]
#If needed 2rd order solution can be seen as  sol[3]'  |>Matrix
# 𝐒₃ = sol[4]

# Iterate forward state space to simulate the data
for t in 2:T
    aug_state = [x[t-1]
                    1 
                    noise[:,t-1]]
    x[t] =  𝐒₁ * aug_state # + 𝐒₂ * ℒ.kron(aug_state, aug_state) / 2 
end

z_rbc = hcat(x...)
Plots.plot(z_rbc' )



