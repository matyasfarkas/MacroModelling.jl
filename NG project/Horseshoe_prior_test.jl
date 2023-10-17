# Horseshoe prior

using Turing

using FillArrays
using LinearAlgebra

struct HorseShoePrior{T}
    X::T
end

@model function (m::HorseShoePrior)(::Val{:original})
    # Independent variable
    X = m.X
    J = size(X, 2)
  
    # Priors
    halfcauchy  = truncated(Cauchy(0, 1); lower=0)
    τ ~ halfcauchy
    λ ~ filldist(halfcauchy, J)
    α ~ TDist(3) # Intercept
    β ~ MvNormal(Diagonal((λ .* τ).^2)) # Coefficients
    σ ~ Exponential(1) # Errors
  
    # Dependent variable
    y ~ MvNormal(α .+ X * β, σ^2 * I)
  
    return (; τ, λ, α, β, σ, y)
  end

  @model function (m::HorseShoePrior)(::Val{:+})
    # Independent variable
    X = m.X
    J = size(X, 2)
  
    # Priors
    τ ~ truncated(Cauchy(0, 1/J); lower=0)
    η ~ truncated(Cauchy(0, 1); lower=0)
    λ ~ filldist(Cauchy(0, 1), J)
    β ~ MvNormal(Diagonal(((η * τ) .* λ).^2)) # Coefficients
    α ~ TDist(3) # Intercept
    σ ~ Exponential(1) # Errors
    
    # Dependent variable
    y ~ MvNormal(α .+ X * β, σ^2 * I)
  
    return (; τ, η, λ, β, y)
  end

  @model function (m::HorseShoePrior)(::Val{:finnish}; τ₀=3, ν_local=1, ν_global=1, slab_df=4, slab_scale=2)
    # Independent variable
    X = m.X
    J = size(X, 2)
  
    # Priors
    z ~ MvNormal(Zeros(J), I) # Standard Normal for Coefs
    α ~ TDist(3) # Intercept
    σ ~ Exponential(1) # Errors
    λ ~ filldist(truncated(TDist(ν_local); lower=0), J)  # local shrinkage
    τ ~ (τ₀ * σ) * truncated(TDist(ν_global); lower=0)  # global shrinkage
    c_aux ~ InverseGamma(0.5 * slab_df, 0.5 * slab_df)
    
    # Transformations
    c = slab_scale * sqrt(c_aux)
    λtilde = λ ./ hypot.(1, (τ / c) .* λ)
    β = τ .* z .* λtilde # Regression coefficients
  
    # Dependent variable
    y ~ MvNormal(α .+ X * β,  σ^2 * I)
    return (; τ, σ, λ, λtilde, z, c, c_aux, α, β, y)
  end


  @model function (m::HorseShoePrior)(::Val{:R2D2}; mean_R2=0.5, prec_R2=2, cons_D2=1)
    # Independent variable
    X = m.X
    J = size(X, 2)
  
    # Priors
    z ~ filldist(Normal(), J)
    α ~ TDist(3) # Intercept
    σ ~ Exponential(1) # Errors
    R2 ~ Beta(mean_R2 * prec_R2, (1 - mean_R2) * prec_R2) # R2 parameter
    ϕ ~ Dirichlet(J, cons_D2)
    τ2 = σ^2 * R2 / (1 - R2)
    
    # Coefficients
    β = z .* sqrt.(ϕ * τ2)
  
    # Dependent variable
    y ~ MvNormal(α .+ X * β,  σ^2 * I)
    return (; σ, z, ϕ, τ2 , R2, α, β, y)
  end
  

  # data
X = randn(100, 2)
X = hcat(X, randn(100) * 2) # let's bias this third variable
y = X[:, 3] .+ (randn(100) * 0.1) # y is X[3] plus a 10% Gaussian noise

# models
model_original = HorseShoePrior(X)(Val(:original));
model_plus = HorseShoePrior(X)(Val(:+));
model_finnish = HorseShoePrior(X)(Val(:finnish));
model_R2D2 = HorseShoePrior(X)(Val(:R2D2));

# Condition on data y
model_original_y = model_original | (; y);
model_plus_y = model_plus | (; y);
model_finnish_y = model_finnish | (; y);
model_R2D2_y = model_R2D2 | (; y);

# sample
fit_original = sample(model_original_y, NUTS(), MCMCThreads(), 2_000, 4)
fit_plus = sample(model_plus_y, NUTS(), MCMCThreads(), 2_000, 4)
fit_finnish = sample(model_finnish_y, NUTS(), MCMCThreads(), 2_000, 4)
fit_R2D2 = sample(model_R2D2_y, NUTS(), MCMCThreads(), 2_000, 4)

summarystats(fit_original)

summarystats(fit_plus)
