import Pkg; Pkg.instantiate();
#import Pkg; Pkg.add("DifferentiableStateSpaceModels")
#using DifferentiableStateSpaceModels, DifferenceEquations, LinearAlgebra, Zygote, Distributions, DiffEqBase, Symbolics, Plots, Random, StatsPlots


using MacroModelling
@model RBC begin
    1 / (- k[0] + (1 - δ) * k[-1] +  exp(z[-1]) * k[-1]^α ) - (β / (- k[1] + (1 - δ) * k[0] +  exp(z[0]) * k[0]^α )) * (α * exp(z[0]) * k[-1]^(α - 1) + (1 - δ)) =0
    #    1 / c[0] - (β / c[1]) * (α * exp(z[1]) * k[1]^(α - 1) + (1 - δ)) =0
    #    q[0] = exp(z[0]) * k[0]^α 
    z[0] = ρ * z[-1] + σ[0] * EPSz[x]
    σ[0] =  (1-ρ_σ) * σ̄  + ρ_σ * σ[-1] + σ_σ* EPSzs[x]
end

@parameters RBC verbose = true begin 
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

MOM1= get_moments(RBC,RBC.parameter_values)
LRvar= MOM1[2].^2 
x_iv= LRvar 

import LinearAlgebra as ℒ
import Turing
using Distributions
using Zygote
# state = zeros(RBC.timings.nVars)
# shock = zeros(RBC.timings.nExo)
# aug_state = [state
  #  1
  #  shock]

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
    loglik = sum([logpdf(MvNormal(ℒ.Diagonal(Ω_1 * ones(size(observables,1)))), observables[t] .- ℒ.I * u[t][1:size(x_iv,1)-1]) for t in 1:T])
    return loglik
end
 

 m = RBC
# Turing model definition
Turing.@model function rbc_1_svt_jointseq2(z, m, dof,Ω_1 )
    α ~ Turing.Uniform(0.2, 0.8)
    β ~ Turing.Uniform(0.5, 0.99)
    ρ = 0.2
    δ = 0.02
    ρ_σ ~ Turing.Beta(2.625, 2.625) #Persistence of log volatility
    μ_σ ~ Turing.Normal(1., 0.5) #Mean of (prescaling) volatility
    σ_σ ~ Turing.Uniform(0.03, 0.3) #Volatility of volatility
    σ̄ ~ Turing.Uniform(0,2)
    T = size(z, 2)
    xnought ~ Turing.filldist(TDist(dof),1) #Initial shocks 
    #ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    #ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    sol = get_solution(m,[ ρ_σ,σ̄ ,σ_σ,α ,β,ρ,δ], algorithm = :second_order)#
    x_iv= get_moments(m,[ ρ_σ,σ̄ ,σ_σ,α ,β,ρ,δ])[2].^2 .* xnought #scale initial condition to ergodic variance
    
    ϵ_draw ~ Turing.filldist(TDist(dof),size(m.exo,1) .* T) #Shocks are t-distributed!
    ϵ = reshape(ϵ_draw, size(m.exo,1), T)
    #sol = generate_perturbation(m, p_d, p_f, Val(1); cache, settings) 
    Turing.@addlogprob! svlikelihood2(sol[2], sol[3],x_iv,Ω_1,z,ϵ) 
end

#z=[-0.00560077   0.0106522   0.0126372  0.00322475   0.00499652  -0.0231639   0.0111031  -0.00969759    -0.00167925  0.0126746   0.00699829   0.000660944   0.000381123  0.000263827  -0.0189499;  -0.0121921   -0.0243286  -0.0278035  0.0512917   -0.0588481   -0.0073273  -0.0166542  -0.0248834       0.0782577   0.103044   -0.00905971  -0.0269399    -0.000341541  0.0270812    -0.097109]
z1= [-0.050711970748351494 1.4044774222772238 0.684907206029626 -0.19787740491926387 0.1317397877981443 -0.9536828006193436 -3.3867903966447503 -5.70724648352132 -6.412555644249861 -9.884662813230246 -9.185451931992498 -9.908610245618556 -8.418820636261552 -9.138606207272447 -9.544496821221017 -10.643526939416594 -12.253133400971382 -14.499737727541753 -13.927658607749716 -13.773290499852875 -12.545464136034571 -13.120460491444296 -9.42472721928242 -7.63090494018952 -7.5415944181384855 -9.229128993711717 -14.235222891859532 -19.37941602918933 -18.714359994188015 -15.85842470067257 -12.60856257561306 -13.415656385418018 -17.796694842410613 -17.54114806397011 -20.97292530824145 -20.740708521719707 -23.258394229825065 -25.985308822259494 -21.329144405174354 -24.11778471542618; 0.22767019861163798 -0.10948584327748564 -0.1342544368112329 0.04450582229016912 -0.18656136415834668 -0.408732002188916 -0.4000121400074196 -0.17409121513082254 -0.5975818304289582 0.03513979065759125 -0.1746786107242836 0.1608009189295066 -0.17933142776545566 -0.13854986657065754 -0.2551834335742658 -0.3478139777087617 -0.44543779151775315 -0.014163292434340571 -0.07088730526340638 0.0922358282882993 -0.16272079026568823 0.5036669068623346 0.21719056156553004 -0.0524533893535264 -0.354231268305226 -0.8959881370432301 -0.9214765908177791 -0.019485034350445712 0.3365330037636989 0.3967372958900503 -0.24633876436993563 -0.797233781706826 -0.10946988585163642 -0.6759993458646283 -0.13191043807717273 -0.5689310643593226 -0.5804765344270236 0.5365436808195868 -0.5986657898452385 -0.08410907812679; -0.10439100713077967 -0.12385369810242514 -0.06419775055468943 -0.014997817022010355 -0.041850980690822236 -0.029180638267381938 0.07271169448315173 -0.03012036092396792 0.003085091513251039 -0.014675333764106704 0.06003717499605629 -0.006959270181710813 0.04048815889800454 0.10178636635100881 0.024056105357426878 -0.013456303631328223 0.0757855195891028 0.04004579673427046 0.018737013643165584 0.045969965072379396 -0.014180350336304033 0.026448246895713998 -0.011603999243423448 -0.05792592250765555 -0.10844511022845907 -0.08231298247858182 -0.058253135411514466 -0.03652031199577252 -0.04804559279347671 -0.0075676097067182676 0.031153131969132452 0.017677664184882074 -0.0028785273355223097 0.041255916210401034 0.06715498677851128 0.09461681770114314 0.06838236006358879 -0.01805527724816962 -0.04561790305772382 -0.020524944349523323]
Ω_1 = 0.0000001
turing_model3 = rbc_1_svt_jointseq2(z, RBC, 4, Ω_1 ) # passing observables from before 

n_samples = 100
n_adapts = 10
δ = 0.65
alg = Turing.NUTS(n_adapts,δ)
chain_2_joint = Turing.sample(turing_model3, alg, n_samples; progress = true)
