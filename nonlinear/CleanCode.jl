import Pkg; Pkg.instantiate();
import Pkg; Pkg.add("DifferentiableStateSpaceModels")
using DifferentiableStateSpaceModels, DifferenceEquations, LinearAlgebra, Zygote, Distributions, DiffEqBase, Symbolics, Plots, Random, StatsPlots
using Plots

#Build the RBC model
∞ = Inf
@variables α, β, ρ, δ, σ, Ω_1
@variables t::Integer, k(..), z(..), c(..), q(..)

x = [k, z] # states
y = [c, q] # controls
p = [α, β, ρ, δ, σ, Ω_1] # parameters

H = [1 / c(t) - (β / c(t + 1)) * (α * exp(z(t + 1)) * k(t + 1)^(α - 1) + (1 - δ)),
     c(t) + k(t + 1) - (1 - δ) * k(t) - q(t),
     q(t) - exp(z(t)) * k(t)^α,
     z(t + 1) - ρ * z(t)]  # system of model equations

# analytic solutions for the steady state.  Could pass initial values and run solver and use initial values with steady_states_iv
steady_states = [k(∞) ~ (((1 / β) - 1 + δ) / α)^(1 / (α - 1)),
                 z(∞) ~ 0,
                 c(∞) ~ (((1 / β) - 1 + δ) / α)^(α / (α - 1)) -
                        δ * (((1 / β) - 1 + δ) / α)^(1 / (α - 1)),
                 q(∞) ~ (((1 / β) - 1 + δ) / α)^(α / (α - 1))]


Γ = [σ;;] # matrix for the 1 shock.  The [;;] notation just makes it a matrix rather than vector in julia
η = [0; -1;;] # η is n_x * n_ϵ matrix.  The [;;] notation just makes it a matrix rather than vector in julia

# observation matrix.  order is "y" then "x" variables, so [c,q,k,z] in this example
# Q = [1.0 0  0   0; # select c as first "z" observable
#   0   0  1.0 0] # select k as second "z" observable

# OBSEVABLES ARE CHANGED TO BE THE STATES!
#observation matrix.  order is "y" then "x" variables, so [c,q,k,z] in this example
 Q = [0 0  1.0 0; # select k as first "z" observable
      0 0  0 1.0] # select z as second "z" observable


# diagonal cholesky of covariance matrix for observation noise (so these are standard deviations).  Non-diagonal observation noise not currently supported
Ω = [Ω_1, Ω_1]

# Generates the files and includes if required.  If the model is already created, then just loads
overwrite_model_cache  = true
model_rbc = @make_and_include_perturbation_model("rbc_notebook_example", H, (; t, y, x, p, steady_states, Γ, Ω, η, Q, overwrite_model_cache)) # Convenience macro.  Saves as ".function_cache/rbc_notebook_example.jl"

#Solve model at some fixed parameters
p_f = (ρ = 0.2, δ = 0.02, σ = 0.01, Ω_1 = 0) # Fixed parameters
p_d = (α = 0.5, β = 0.95) # Pseudo-true values
m = model_rbc  # ensure notebook executed above
sol = generate_perturbation(m, p_d, p_f) # Solution to the first-order RBC
# Simulate T observations from a random initial condition
T = 20
Random.seed!(12345) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom
shockdist = TDist(dof) #Shocks are student-t

# draw from t scaled by approximate invariant variance) for the initial condition
x_iv = sol.x_ergodic_var * rand(shockdist,sol.n_x)

# Generate noise sequence
noiseshocks = rand(shockdist,T)
noise = Matrix(noiseshocks') # the ϵ shocks are "noise" in DifferenceEquations for SciML compatibility 

#Solve problem forward with Student-t noise
problem = LinearStateSpaceProblem(sol, x_iv, (0, T); noise)
sim=solve(problem)
# Collapse to simulated observables as a matrix  - as required by current DifferenceEquations.jl likelihood
# see https://github.com/SciML/DifferenceEquations.jl/issues/55 for direct support of this datastructure
z_rbc = hcat(sim.z...)
Plots.plot(sim)

using Turing
using Turing: @addlogprob!
Turing.setadbackend(:zygote);  # Especially when we sample the latent noise, we will require high-dimensional gradients with reverse-mode AD

# Turing model definition
@model function rbc_1_t_joint(z, m, p_f, dof, cache, settings)
    α ~ Uniform(0.2, 0.8)
    β ~ Uniform(0.5, 0.99)
    p_d = (; α, β)
    T = size(z, 2)
    xnought ~ filldist(TDist(dof),m.n_x) #Initial shocks 
    ϵ_draw ~ filldist(TDist(dof),m.n_ϵ * T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, m.n_ϵ, T)
    sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
    if !(sol.retcode == :Success)
        @addlogprob! -Inf
        return
    end
    x_iv = sol.x_ergodic_var * xnought #scale initial condition to ergodic variance
    problem = LinearStateSpaceProblem(sol, x_iv, (0, T), observables = z, noise=ϵ)
    @addlogprob! solve(problem, DirectIteration()).logpdf # should choose DirectIteration() by default if not provided
end
cache = SolverCache(model_rbc, Val(1),  [:α, :β])
settings = PerturbationSolverSettings(; print_level = 0)
p_f = (ρ = 0.2, δ = 0.02, σ = 0.01, Ω_1 = 0.01) # Fixed parameters
z = z_rbc # simulated in previous steps
turing_model = rbc_1_t_joint(z, model_rbc, p_f, dof, cache, settings) # passing observables from before 

n_samples = 300
n_adapts = 50
δ = 0.65
alg = NUTS(n_adapts,δ)
chain_1_joint = sample(turing_model, alg, n_samples; progress = true)

#Plot the chains and posteriors
Plots.plot(chain_1_joint[["α"]]; colordim=:parameter, legend=true)
Plots.plot(chain_1_joint[["β"]]; colordim=:parameter, legend=true)

#Plot true and estimated latents to see how well we backed them out
symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(chain_1_joint[:, [Symbol("ϵ_draw[$a]") for a in 1:21], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
plot(ϵ_mean[2:end], ribbon=2 * ϵ_std[2:end], label="Posterior mean", title = "First-Order Joint: Estimated Latents")
plot!(noise', label="True values")

# Stochastic volatility

# Simulate T observations from a random initial condition
T = 50
Random.seed!(1234) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom
shockdist = TDist(dof) #Shocks are student-t
ρ_σ = 0.5 #Persistence of log volatility
μ_σ = 1. #Mean of (prescaling) volatility
σ_σ = 0.1 #Volatility of volatility

# draw from t scaled by approximate invariant variance) for the initial condition
x_iv = sol.x_ergodic_var * rand(shockdist,sol.n_x)

# Generate noise sequence
Random.seed!(1234) #Fix seed to reproduce data
noise = Matrix(rand(shockdist,T)') # the ϵ shocks are "noise"
Random.seed!(1234) #Fix seed to reproduce data
volshocks = Matrix(rand(MvNormal(T,1.0))') # the volatility shocks are log-normal
Random.seed!(1234) #Fix seed to reproduce data
obsshocks = reshape(rand(MvNormal(T*sol.n_z,p_f[:Ω_1])), sol.n_z, T) #Gaussian observation noise

#Extract solution matrices
A = sol.A
B = sol.B
C = Matrix(I, 2, 2) #sol.C
D = sol.D

# Initialize
u = [zero(x_iv) for _ in 1:T]
u[1] .= x_iv
vol = [zeros(1) for _ in 1:T]
vol[1] = [μ_σ] #Start at mean: could make random but won't for now
#Allocate sequence
z = [zeros(size(C, 1)) for _ in 1:T] 
mul!(z[1], C, u[1])  # update the first of z
for t in 2:T
        mul!(u[t], A, u[t - 1]) # sets u[t] = A * u[t - 1]
        mul!(vol[t], ρ_σ, vol[t-1])
        vol[t] .+= (1 - ρ_σ) * μ_σ
        mul!(vol[t], σ_σ, view(volshocks, :, t - 1),1,1) # adds σ_σ * volshocks[t-1] to vol[t]
        mul!(u[t], exp(vol[t][]) .* B, view(noise, :, t - 1),1,1)
        mul!(z[t], C, u[t]) 
end
for t in 1:T #Add observation noise
        z[t] .+= view(obsshocks,:,t)
end

z_data = hcat(z...)
plot(z_data') #Plot k and z from simulation
plot(hcat(vol...)') #Plot the latent volatility state


#Likelihood evaluation function using `Zygote.Buffer()` to create internal arrays that don't interfere with gradients.
function svlikelihood2(A,B,C,D,x_iv,Ω_1,μ_σ,ρ_σ,σ_σ,observables,noise,volshocks) #Accumulate likelihood
    # Initialize
    T = size(observables,2)
    u = Zygote.Buffer([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
    vol = Zygote.Buffer([zeros(1) for _ in 1:T]) #Fix type: Array of vector of vectors?
    u[1] = x_iv 
    vol[1] = [μ_σ] #Start at mean: could make random but won't for now
    for t in 2:T
        vol[t] = ρ_σ * vol[t-1] .+ (1 - ρ_σ) * μ_σ .+ σ_σ * volshocks[t - 1]
        u[t] = A * u[t - 1] .+ exp.(vol[t]) .* (B * noise[t - 1])[:]
    end
    loglik = sum([logpdf(MvNormal(Diagonal(Ω_1 * ones(size(C, 1)))), observables[t] .- C * u[t]) for t in 1:T])
    return loglik
end

ll = svlikelihood2(sol.A,sol.B,sol.C,sol.D,x_iv,p_f[:Ω_1],μ_σ,ρ_σ,σ_σ,z_data,noise,volshocks)

gradient(x_iv->svlikelihood2(sol.A,sol.B,sol.C,sol.D,x_iv,p_f[:Ω_1],μ_σ,ρ_σ,σ_σ,z_data,noise,volshocks),[0., 0.])

# Turing model definition
@model function rbc_1_svt_jointseq(z, m, p_f, dof, cache, settings)
    α ~ Uniform(0.2, 0.8)
    β ~ Uniform(0.5, 0.99)
    ρ_σ ~ Beta(2.625, 2.625) #Persistence of log volatility
    μ_σ ~ Normal(1., 0.5) #Mean of (prescaling) volatility
    σ_σ ~ Uniform(0.03, 0.3) #Volatility of volatility
    p_d = (; α, β)
    T = size(z, 2)
    xnought ~ filldist(TDist(dof),m.n_x) #Initial shocks 
    ϵ_draw ~ filldist(TDist(dof),m.n_ϵ * T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, m.n_ϵ, T)
    vsdraw ~ MvNormal(T, 1.0)
    volshocks = reshape(vsdraw,1,T)   
    sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
    if !(sol.retcode == :Success)
        @addlogprob! -Inf
        return
    end
    x_iv = sol.x_ergodic_var * xnought #scale initial condition to ergodic variance
    @addlogprob! svlikelihood2(sol.A,sol.B,sol.C,sol.D,x_iv,p_f[:Ω_1],μ_σ,ρ_σ,σ_σ,z,ϵ,volshocks)
end

cache = SolverCache(model_rbc, Val(1),  [:α, :β])
settings = PerturbationSolverSettings(; print_level = 0)
p_f = (ρ = 0.2, δ = 0.02, σ = 0.01, Ω_1 = 0.01) # Fixed parameters
z = z_data # simulated in previous steps
turing_model2 = rbc_1_svt_jointseq(z, model_rbc, p_f, dof, cache, settings) # passing observables from before 

n_samples = 1000
n_adapts = 100
δ = 0.65
alg = NUTS(n_adapts,δ)
chain_2_joint = sample(turing_model2, alg, n_samples; progress = true)


plot(chain_2_joint[["μ_σ"]]; colordim=:parameter, legend=true)

plot(chain_2_joint[["β"]]; colordim=:parameter, legend=true)

#Plot true and estimated latents to see how well we backed them out
symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ_chain = sort(chain_2_joint[:, [Symbol("ϵ_draw[$a]") for a in 1:50], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = describe(ϵ_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
plot(ϵ_mean[1:end], ribbon=2 * ϵ_std[1:end], label="Posterior mean", title = "First-Order Joint: Estimated Shocks")
plot!(noise', label="True values")