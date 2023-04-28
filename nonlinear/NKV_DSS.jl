# import Pkg; Pkg.instantiate();
# import Pkg; Pkg.add("DifferentiableStateSpaceModels")
using DifferentiableStateSpaceModels, DifferenceEquations, LinearAlgebra, Zygote, Distributions, DiffEqBase, Symbolics, Plots, Random, StatsPlots
using Plots

#Build the RBC model
∞ = Inf
@variables σ, β, κ, ϕ, ξ, sig_is, sig_pc, sig_mp, Ω_1
@variables t::Integer, c(..), infl(..), i(..),ϵ_is(..),ϵ_pc(..), ϵ_mp(..)

x = [ϵ_is, ϵ_pc, ϵ_mp] # states
y = [c, infl, i] # controls
p = [ σ, β, κ, ϕ, ξ,sig_is, sig_pc, sig_mp, Ω_1] # parameters

H = [ ϵ_is(t+1),
      ϵ_pc(t+1),
      ϵ_mp(t+1),
      c(t) - c(t + 1) + 1/σ * (i(t) - infl(t+1)) - ϵ_is(t),
      infl(t) - β * infl(t + 1) - κ * c(t)- ϵ_pc(t),
      i(t) - ϕ * infl(t) - ξ * c(t)- ϵ_mp(t)]  # system of model equations

# analytic solutions for the steady state.  Could pass initial values and run solver and use initial values with steady_states_iv
steady_states = [  ϵ_is(∞) ~ 0,
                   ϵ_pc(∞) ~ 0,
                   ϵ_mp(∞) ~ 0, 
                   c(∞) ~ 0,
                 infl(∞) ~ 0,
                 i(∞) ~ 0]


Γ = [sig_is; 0 ; 0 ;; 0; sig_pc; 0;; 0; 0; sig_mp;;] # matrix for the 1 shock.  The [;;] notation just makes it a matrix rather than vector in julia
η = [-1; 0; 0;;  0 ;-1 ;0 ;; 0; 0; -1;;] # η is n_x * n_ϵ matrix.  The [;;] notation just makes it a matrix rather than vector in julia

# observation matrix.  order is "y" then "x" variables, so [c, infl, i,ϵ_is,ϵ_pc, ϵ_mp ] in this example
Q = [ 1.0 0 0 0 0 0; # select c as first "z" observable
      0 1.0 0 0 0 0; # select inf as second "z" observable
      0 0 1.0 0 0 0] # select k as third "z" observable

# diagonal cholesky of covariance matrix for observation noise (so these are standard deviations).  Non-diagonal observation noise not currently supported
Ω = [Ω_1, Ω_1, Ω_1]

# Generates the files and includes if required.  If the model is already created, then just loads
overwrite_model_cache  = true
model_nkv = @make_and_include_perturbation_model("nkv_sv", H, (; t, y, x, p, steady_states, Γ, Ω, η, Q, overwrite_model_cache)) # Convenience macro.  Saves as ".function_cache/rbc_notebook_example.jl"


p_f = ( κ= 0.1275 , ϕ  = 1.5, ξ = 0.125, Ω_1 = 0.01,sig_is=1, sig_pc=1, sig_mp=1) # Fixed parameters
p_d = (σ = 1, β = 0.99) # Pseudo-true values
m = model_nkv  # ensure notebook executed above
sol = generate_perturbation(m, p_d, p_f) # Solution to the first-order RBC

# Stochastic volatility

# Simulate T observations from a random initial condition
T = 50
Random.seed!(12435) #Fix seed to reproduce data
dof = 4 #Student t degrees of freedom
shockdist = TDist(dof) #Shocks are student-t
ρ_σ = 0.5* Matrix(1.0I,sol.n_x, sol.n_x)#Persistence of log volatility
μ_σ = [1;1;1] #Mean of (prescaling) volatility
σ_σ = 0.1*Matrix(1.0I,sol.n_x, sol.n_x)#Volatility of volatility

# draw from t scaled by approximate invariant variance) for the initial condition
x_iv = sol.x_ergodic_var * rand(shockdist,sol.n_x)

# Generate noise sequence
noise =  reshape(rand(MvNormal(T*sol.n_x,1.0)), sol.n_x, T) # the ϵ shocks are "noise"
volshocks = reshape(rand(MvNormal(T*sol.n_x,1.0)), sol.n_x, T) # the volatility shocks are log-normal
obsshocks = reshape(rand(MvNormal(T*sol.n_z,p_f[:Ω_1])), sol.n_z, T) #Gaussian observation noise

#Extract solution matrices
A = sol.A
B = sol.B
C = sol.C
D = sol.D

# Initialize
u = [zero(x_iv) for _ in 1:T]
u[1] .= x_iv
vol = [zeros(sol.n_x) for _ in 1:T]
vol[1] = μ_σ #Start at mean: could make random but won't for now
#Allocate sequence
z = [zeros(size(C, 1)) for _ in 1:T] 
mul!(z[1], C, u[1])  # update the first of z
for t in 2:T
        mul!(u[t], A, u[t - 1]) # sets u[t] = A * u[t - 1]
        mul!(vol[t], ρ_σ, vol[t-1])
        vol[t] .+= (Matrix(1.0I,sol.n_x, sol.n_x) - ρ_σ )* μ_σ
        vol[t] .+=  σ_σ * view(volshocks, :, t - 1) #mul!(vol[t], σ_σ, view(volshocks, :, t - 1),1,1) # adds σ_σ * volshocks[t-1] to vol[t]
        mul!(u[t], exp.(vol[t]) .* B, view(noise, :, t - 1))
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
      vol = Zygote.Buffer([zero(x_iv) for _ in 1:T]) #Fix type: Array of vector of vectors?
      u[1] = x_iv 
      vol[1] = μ_σ #Start at mean: could make random but won't for now
      for t in 2:T
          vol[t] = ρ_σ * vol[t-1] .+ (Matrix(1.0I,size(x_iv,1), size(x_iv,1)) - ρ_σ) * μ_σ .+ σ_σ * volshocks[:,t - 1]
          u[t] = A * u[t - 1] .+ exp.(vol[t]) .* (B * noise[:,t - 1])
      end
      loglik = sum([logpdf(MvNormal(Diagonal(Ω_1 * ones(size(C, 1)))), observables[t] .- C * u[t]) for t in 1:T])
      return loglik
  end
  
 # ll = svlikelihood2(sol.A,sol.B,sol.C,sol.D,x_iv,p_f[:Ω_1]*Matrix(1.0I,sol.n_z, sol.n_z),μ_σ,ρ_σ,σ_σ,z_data,noise,volshocks)
  
  #gradient(x_iv->svlikelihood2(sol.A,sol.B,sol.C,sol.D,x_iv,p_f[:Ω_1]*Matrix(1.0I,sol.n_z, sol.n_z),μ_σ,ρ_σ,σ_σ,z_data,noise,volshocks),[0., 0.])
  
using Turing
using Turing: @addlogprob!
Turing.setadbackend(:zygote);  # Especially when we sample the latent noise, we will require high-dimensional gradients with reverse-mode AD


  # Turing model definition
  @model function rbc_1_svt_jointseq(z, m, p_f, dof, cache, settings)
      σ ~ Uniform(0, 2)
      β ~ Uniform(0.5, 0.99)   
      #r = zeros(m.n_x)
      #s  = zeros(m.n_x)
      #for i = 1:m.n_x
      #r[i]~ Beta(2.625, 2.625) #Persistence of log volatility
      #s[i] ~ Uniform(0.03, 0.3)           
      #end
      r ~ filldist(Beta(2.625, 2.625) ,m.n_x)
      s ~ filldist(Uniform(0.03, 0.3) ,m.n_x)
      ρ_σ = Diagonal(r)  #
      ρ_σ = Diagonal(s)  

      #      ρ_σ ~ Beta(2.625, 2.625) #Persistence of log volatility
      #      σ_σ ~ Uniform(0.03, 0.3) #Volatility of volatility
      μ_σ ~ MvNormal(m.n_x, 0.5)
      p_d = (; σ, β)
      T = size(z, 2)
      xnought ~ filldist(TDist(dof),m.n_x) #Initial shocks 
      ϵ_draw ~ filldist(TDist(dof),m.n_ϵ * T) #Shocks are t-distributed!
      ϵ = reshape(ϵ_draw, m.n_ϵ, T)
      vsdraw ~ MvNormal(T*m.n_x, 1.0)
      volshocks = reshape(vsdraw,m.n_x,T)   
      sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
      if !(sol.retcode == :Success)
          @addlogprob! -Inf
          return
      end
      x_iv = sol.x_ergodic_var * xnought #scale initial condition to ergodic variance
      @addlogprob! svlikelihood2(sol.A,sol.B,sol.C,sol.D,x_iv,p_f[:Ω_1]*Matrix(1.0I,m.n_z,m.n_z),μ_σ,ρ_σ,σ_σ,z,ϵ,volshocks)
  end
  
  cache = SolverCache(model_nkv, Val(1),  [:σ, :β])
  settings = PerturbationSolverSettings(; print_level = 0)
  p_f = ( κ= 0.1275 , ϕ  = 1.5, ξ = 0.125, Ω_1 = 0.01,sig_is=1, sig_pc=1, sig_mp=1) # Fixed parameters
  z = z_data # simulated in previous steps
  turing_model2 = rbc_1_svt_jointseq(z, model_nkv, p_f, dof, cache, settings) # passing observables from before 
  
  n_samples = 1000
  n_adapts = 100
  δ = 0.65
  alg = NUTS(n_adapts,δ)
  chain_2_joint = sample(turing_model2, alg, n_samples; progress = true)
#

plot(chain_2_joint[["μ_σ[1]"]]; colordim=:parameter, legend=true)

  symbol_to_int(s) = parse(Int, string(s)[9:end-1])
ϵ1_chain = sort(chain_2_joint[:, [Symbol("ϵ_draw[$a]") for a in 51:100], 1], lt = (x,y) -> symbol_to_int(x) < symbol_to_int(y))
tmp = describe(ϵ1_chain)
ϵ_mean = tmp[1][:, 2]
ϵ_std = tmp[1][:, 3]
plot(ϵ_mean[1:end], ribbon=2 * ϵ_std[1:end], label="Posterior mean", title = "First-Order Joint: Estimated Shocks")
plot!(noise[1,:], label="True values")

