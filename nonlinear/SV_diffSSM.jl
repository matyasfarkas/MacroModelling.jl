import Pkg; Pkg.instantiate();
#import Pkg; Pkg.add("DifferentiableStateSpaceModels")
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

# Stochastic volatility

# Simulate T observations from a random initial condition
T = 20
Random.seed!(1234) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom

shockdist = TDist(dof) #Shocks are student-t
ρ_σ = 0.5 #Persistence of log volatility
μ_σ = 1. #Mean of (prescaling) volatility
σ_σ = 0.1 #Volatility of volatility

# draw from t scaled by approximate invariant variance) for the initial condition
x_iv = sol.x_ergodic_var * rand(shockdist,sol.n_x)
x_iv = zeros(2,1) 
# Generate noise sequence
Random.seed!(1234) #Fix seed to reproduce data
noise = Matrix(rand(shockdist,T)') # the ϵ shocks are "noise"

Random.seed!(1234) #Fix seed to reproduce data
volshocks = Matrix(rand(MvNormal(T,1.0))') # the volatility shocks are log-normal
obsshocks = reshape(rand(MvNormal(T*sol.n_z,p_f[:Ω_1])), sol.n_z, T) #Gaussian observation noise

#Extract solution matrices
A = sol.A
B = sol.B
C = sol.C
D = sol.D

# Initialize
u = [zero(x_iv) for _ in 1:T]
u[1] .= x_iv
vol = [zeros(1) for _ in 1:T]
vol[1] = [μ_σ] #Start at mean: could make random but won't for now
#Allocate sequence
z = [Matrix(zeros(2,1)) for _ in 1:T] 
z[1]=C*u[1]   # update the first of z
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
