
##### BEGIN EIGHT METHODS

import Base.rand, StatsBase.params
import Random, Distributions, Statistics, StatsBase
using Random, QuadGK,   Roots
using Distributions

struct SkewT{T<:Real} <: Distribution{Univariate,Continuous}
    μ::T  ## first parameter of Skew t Distribution - mean
    σ::T  ## second parameter of Skew t Distribution - std
    α::T  ## third parameter of Skew t Distribution - skewness 
    ν::T  ## fourth parameter of Skew t Distribution - kurtosis (nu of a t-distribution)

    #     inner constructor function to instantiate new SkewT objects
    function SkewT{T}(μ::T, σ::T, α::T,  ν::T ; check_args = true) where {T<:Real}
        #check_args && Distributions.@check_args(Kuma, a>0 && b>0)
        return new{T}(μ,σ,α,ν)
    end
end


function SkewT(μ::Float64, σ::Float64, α::Float64,  ν::Float64; check_args = true)
    return SkewT{Float64}(μ,σ,α,ν,check_args = check_args)
end

# constructor for real params - use promote to make aprams the same type
SkewT(μ::Real, σ::Real, α::Real,  ν::Real ) =SkewT(promote(μ,σ,α,ν)...)
SkewT(μ::Integer, σ::Integer, α::Integer,  ν::Integer ) = SkewT(float(μ),float(σ),float(α),float(ν))


# 0 - helper function
StatsBase.params(d::SkewT) = (d.μ, d.σ, d.α, d.ν) 
# Where μ is ε; σ is Ω; α is α and ν is ν   in (27) Azzalini Capitano (2002)

#1 rand(::AbstractRNG, d::UnivariateDistribution)
function Base.rand(rng::AbstractRNG, d::SkewT)
    (μ, σ, α, ν) = params(d)
    # Use Rejection method to sample from a Uniform and a t-distribution   
    t1 = Distributions.TDist(ν)
    t2 = Distributions.TDist(ν+1)
    # Analogously to rejection sampling for skew-Normal in R:
    # T=1e4 #number of simulations
    # x=NULL
    # alpha = 0
    # while (length(x)<T){
    #     y=rnorm(2*T)
    #     x=c(x,y[runif(2*T)<pnorm(alpha*y)])}
    # x=x[1:T]
    # plot(density(x))    
    # Source: https://stats.stackexchange.com/questions/316314/sampling-from-skew-normal-distribution

    while true
        r = 2/σ* rand(t1)
        xstd = (r - μ) / σ
        wstd = α*sqrt((ν+1)/(ν+xstd^2))*xstd
        rand(rng) < cdf(t2,wstd) &&  return r
    end
    
end


#2 sampler(d::Distribution) - works for sampler(rng::AbstractSampler, d::Distribution)
Distributions.sampler(rng::AbstractRNG,d::SkewT) = Base.rand(rng::AbstractRNG, d::SkewT)


#3 logpdf(d::UnivariateDistribution, x::Real)
function Distributions.pdf(d::SkewT{T}, x::Real) where {T<:Real}
    (μ, σ, α, ν) = params(d)
    # Source: https://www.gamlss.com/wp-content/uploads/2018/01/DistributionsForModellingLocationScaleandShape.pdf
    t1 = Distributions.TDist(ν)
    t2 = Distributions.TDist(ν+1)
    xstd = (x - μ) / σ
    wstd = α*sqrt((ν+1)/(ν+xstd^2))*xstd

    return(2/σ * Turing.pdf(t1,xstd) * Distributions.cdf(t2,wstd))
end

Distributions.logpdf(d::SkewT, x::Real) = log(pdf(d,x))

#4 cdf(d::UnivariateDistribution, x::Real)
function Distributions.cdf(d::SkewT{T}, x::Real) where T<:Real
    integral, err = quadgk(xeval->pdf(d,xeval),-Inf,x)
    if integral >1
        integral = 1
    end
    return integral
end

|
#5 quantile(d::UnivariateDistribution, q::Real)
function Statistics.quantile(d::SkewT{T}, x::Real) where T<:Real
#    x0 = Distributions.quantile.(Distributions.Normal(), [x])
 #   println(x0)
    xc = Roots.find_zero(xeval -> cdf(d, xeval) - x, 0)
    return(xc)
end


#6 minimum(d::UnivariateDistribution)
function Base.minimum(d::SkewT)
    return(-Inf)
end


#7 maximum(d::UnivariateDistribution)
function Base.maximum(d::SkewT)
    return(Inf)
end


#8 insupport(d::UnivariateDistribution, x::Real)
function Distributions.insupport(d::SkewT)
    insupport(d::SkewT, x::Real) = (-Inf) <= x <= (Inf)
end

######################END Eight Methods

#### Test to see if it works

skewt = SkewT(0., 1., 0, 4.0)
quantile(skewt,0.5)

skewtdraws = rand(skewt,1024)
tdraws = rand(Distributions.TDist(4.0),1024)
StatsPlots.density(skewtdraws)
StatsPlots.density!(tdraws)

using Turing

# Define a simple coin flip model with unknown prob θ
@model function sampleSKT(x)
  θ ~ SkewT(0., 1., 3,4)
  # θ ~ SkewNormal(0,1,3)  
  
end

#  Flip 1 coin (heads) - Run sampler, collect results
chn = sample(sampleSKT(1), HMC(0.1, 5), 1000)
using StatsPlots
StatsPlots.plot(chn)