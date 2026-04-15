"""
Hamiltonian Monte Carlo for Stochastic Extended Path Expectation Approximation

This module implements HMC sampling to replace Gauss-Hermite quadrature for computing
expectations E_ε[f(y,ε)] in the SEP solver. HMC is particularly effective for models
with occasionally binding constraints (OBC) where value functions have kinks.

Physics motivation: Use Hamiltonian dynamics to explore shock space adaptively,
concentrating samples where residuals are large. More robust than fixed quadrature
nodes near ZLB constraints.

Author: Based on user's HMC expertise
Date: 2026-02-06
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using ForwardDiff

"""
    hmc_step(ε, U, ∇U, Σ; leapfrog_steps=10, step_size=0.1, Σ_inv=nothing)

Single HMC step using leapfrog integrator and Metropolis accept/reject.

# Arguments
- `ε`: Current shock realization (d_ε × 1 vector)
- `U`: Energy function U(ε) = (1/2)||R(y,ε)||²
- `∇U`: Gradient function ∇U(ε) = ∂R/∂ε' * R
- `Σ`: Shock covariance matrix (d_ε × d_ε)
- `leapfrog_steps`: Number of leapfrog integration steps
- `step_size`: Step size for leapfrog integrator

# Returns
- `ε_new`: New shock realization (accepted or rejected)
- `accepted`: Boolean indicating if proposal was accepted
"""
function hmc_step(ε, U, ∇U, Σ; leapfrog_steps::Int=10, step_size::Float64=0.1, Σ_inv=nothing)
    d_ε = length(ε)

    # Sample momentum from N(0, Σ^{-1})
    # Precompute and reuse Σ_inv outside this function when possible.
    Σ_inv_local = isnothing(Σ_inv) ? inv(Σ) : Σ_inv
    p = rand(MvNormal(zeros(d_ε), Σ_inv_local))

    # Compute initial Hamiltonian: H = U(ε) + (1/2) p' Σ p
    H_old = U(ε) + 0.5 * dot(p, Σ * p)

    # Leapfrog integration
    ε_new = copy(ε)
    p_new = copy(p)

    # Half step for momentum
    grad_U = ∇U(ε_new)
    if !all(isfinite, grad_U)
        # Gradient computation failed - reject
        return ε, false
    end
    p_new .-= 0.5 * step_size .* grad_U

    # Full steps for position and momentum
    for i in 1:(leapfrog_steps-1)
        ε_new .+= step_size .* (Σ * p_new)
        grad_U = ∇U(ε_new)
        if !all(isfinite, grad_U)
            return ε, false
        end
        p_new .-= step_size .* grad_U
    end

    # Final full step for position
    ε_new .+= step_size .* (Σ * p_new)

    # Final half step for momentum
    grad_U = ∇U(ε_new)
    if !all(isfinite, grad_U)
        return ε, false
    end
    p_new .-= 0.5 * step_size .* grad_U

    # Compute new Hamiltonian
    H_new = U(ε_new) + 0.5 * dot(p_new, Σ * p_new)

    # Metropolis accept/reject
    # Accept with probability min(1, exp(H_old - H_new))
    if isfinite(H_new) && (rand() < exp(H_old - H_new))
        return ε_new, true
    else
        return ε, false
    end
end

function _deterministic_hmc_fallback(residual_func, d_ε::Int, reason::String)
    ε0 = zeros(d_ε)
    R0 = residual_func(ε0)
    diagnostics = Dict(
        "acceptance_rate" => 1.0,
        "n_samples" => 0,
        "warmup" => 0,
        "mode" => "deterministic_fallback",
        "reason" => reason
    )
    return R0, diagnostics
end

function _prepare_hmc_covariance(Σ::AbstractMatrix{<:Real}, d_ε::Int)
    Σ_mat = Matrix{Float64}(Σ)
    all(isfinite, Σ_mat) || return nothing, nothing, "nonfinite_covariance"

    if maximum(abs.(Σ_mat)) <= eps(Float64)
        return nothing, nothing, "zero_covariance"
    end

    # Enforce symmetry and regularize the diagonal for numerical stability.
    Σ_sym = 0.5 .* (Σ_mat .+ Σ_mat')
    diag_scale = max(maximum(abs.(diag(Σ_sym))), 1.0)
    jitter = max(1e-12, 1e-10 * diag_scale)

    Σ_work = copy(Σ_sym)
    for _ in 1:8
        for i in 1:d_ε
            Σ_work[i, i] = Σ_sym[i, i] + jitter
        end
        try
            Σ_inv = inv(Σ_work)
            if all(isfinite, Σ_inv)
                return Σ_work, Σ_inv, "ok"
            end
        catch
        end
        jitter *= 10.0
    end

    return nothing, nothing, "singular_covariance"
end


"""
    hmc_expectation(residual_func, y_current, Σ, N_samples; kwargs...)

Compute expectation E_ε[R(y,ε)] using HMC sampling.

# Arguments
- `residual_func`: Function R(ε) that evaluates residual at given shock
- `y_current`: Current solution guess (fixed during sampling)
- `Σ`: Shock covariance matrix
- `N_samples`: Number of HMC samples to generate

# Keyword Arguments
- `warmup`: Number of warmup samples to discard (default 50)
- `leapfrog_steps`: Leapfrog steps per HMC iteration (default 10)
- `step_size`: Initial step size (default 0.1)
- `use_tempering`: Enable parallel tempering (default false)
- `temperatures`: Temperature ladder for tempering (default [1.0, 0.5, 0.25])
- `verbose`: Print diagnostic info (default false)

# Returns
- `R_mean`: Monte Carlo estimate of E_ε[R(y,ε)]
- `diagnostics`: Dict with acceptance rate, ESS, etc.
"""
function hmc_expectation(
    residual_func,
    y_current,
    Σ,
    N_samples::Int;
    warmup::Int = 50,
    leapfrog_steps::Int = 10,
    step_size::Float64 = 0.1,
    use_tempering::Bool = false,
    temperatures::Vector{Float64} = [1.0, 0.5, 0.25],
    verbose::Bool = false
)
    d_ε = size(Σ, 1)

    if d_ε == 0
        return _deterministic_hmc_fallback(residual_func, d_ε, "empty_covariance")
    end

    Σ_effective, Σ_inv_effective, cov_status = _prepare_hmc_covariance(Σ, d_ε)
    if Σ_effective === nothing
        return _deterministic_hmc_fallback(residual_func, d_ε, cov_status)
    end

    # Energy function: U(ε) = (1/2)||R(y,ε)||²
    function U(ε)
        R = residual_func(ε)
        if !all(isfinite, R)
            return Inf
        end
        return 0.5 * dot(R, R)
    end

    # Gradient: ∇U(ε) = ∂R/∂ε' * R
    function ∇U(ε)
        # Use ForwardDiff for automatic differentiation
        R = residual_func(ε)
        J_ε = ForwardDiff.jacobian(residual_func, ε)
        return J_ε' * R
    end

    if use_tempering
        # Use parallel tempering (more robust for multimodal landscapes)
        samples, diagnostics = parallel_tempering_hmc(
            U, ∇U, Σ_effective, N_samples;
            temperatures = temperatures,
            warmup = warmup,
            leapfrog_steps = leapfrog_steps,
            step_size = step_size,
            Σ_inv = Σ_inv_effective,
            verbose = verbose
        )
    else
        # Standard HMC
        samples = zeros(d_ε, N_samples + warmup)
        ε = randn(d_ε)  # Initialize from prior N(0,I)

        n_accepted = 0

        for i in 1:(N_samples + warmup)
            ε_new, accepted = hmc_step(ε, U, ∇U, Σ_effective;
                                      leapfrog_steps = leapfrog_steps,
                                      step_size = step_size,
                                      Σ_inv = Σ_inv_effective)

            ε = ε_new
            samples[:, i] = ε

            if accepted
                n_accepted += 1
            end

            if verbose && (i % 20 == 0 || i <= 5)
                acc_rate = n_accepted / i
                println("  HMC iter $i: acceptance=$(round(acc_rate, digits=3))")
            end
        end

        # Discard warmup samples
        samples = samples[:, (warmup+1):end]

        acceptance_rate = n_accepted / (N_samples + warmup)

        diagnostics = Dict(
            "acceptance_rate" => acceptance_rate,
            "n_samples" => N_samples,
            "warmup" => warmup
        )

        if verbose
            println("  HMC complete: acceptance=$(round(acceptance_rate, digits=3))")
        end
    end

    # Compute Monte Carlo expectation
    # E[R(y,ε)] ≈ (1/N) Σᵢ R(y, εᵢ)
    R_sum = zeros(length(residual_func(samples[:, 1])))

    for i in 1:size(samples, 2)
        R_i = residual_func(samples[:, i])
        R_sum .+= R_i
    end

    R_mean = R_sum ./ size(samples, 2)

    return R_mean, diagnostics
end


"""
    parallel_tempering_hmc(U, ∇U, Σ, N_samples; kwargs...)

HMC with parallel tempering for multimodal energy landscapes.

Runs multiple chains at different temperatures and periodically swaps between them
to improve mixing when energy landscape has multiple modes (e.g., at ZLB).

# Arguments
- `U`: Energy function
- `∇U`: Gradient of energy
- `Σ`: Shock covariance
- `N_samples`: Number of samples to generate

# Keyword Arguments
- `temperatures`: Temperature ladder (default [1.0, 0.5, 0.25])
- `warmup`: Warmup samples (default 50)
- `swap_interval`: Swap every N iterations (default 10)
- `leapfrog_steps`: Leapfrog steps (default 10)
- `step_size`: Step size (default 0.1)
- `verbose`: Diagnostic output (default false)

# Returns
- `samples`: Samples from coldest chain (T=temperatures[end])
- `diagnostics`: Dict with acceptance rates, swap rates
"""
function parallel_tempering_hmc(
    U, ∇U, Σ, N_samples::Int;
    temperatures::Vector{Float64} = [1.0, 0.5, 0.25],
    warmup::Int = 50,
    swap_interval::Int = 10,
    leapfrog_steps::Int = 10,
    step_size::Float64 = 0.1,
    Σ_inv = nothing,
    verbose::Bool = false
)
    d_ε = size(Σ, 1)
    n_chains = length(temperatures)

    # Initialize chains
    chains = [randn(d_ε) for _ in 1:n_chains]
    samples_per_chain = [zeros(d_ε, N_samples + warmup) for _ in 1:n_chains]
    n_accepted = zeros(Int, n_chains)
    n_swaps = 0
    n_swap_attempts = 0

    if verbose
        println("  PT-HMC: $n_chains chains, T=$(temperatures)")
    end

    for iter in 1:(N_samples + warmup)
        # Run HMC on each temperature
        for (i, T) in enumerate(temperatures)
            # Tempered energy and gradient
            U_tempered(ε) = U(ε) / T
            ∇U_tempered(ε) = ∇U(ε) / T

            ε_new, accepted = hmc_step(chains[i], U_tempered, ∇U_tempered, Σ;
                                      leapfrog_steps = leapfrog_steps,
                                      step_size = step_size,
                                      Σ_inv = Σ_inv)

            chains[i] = ε_new
            samples_per_chain[i][:, iter] = ε_new

            if accepted
                n_accepted[i] += 1
            end
        end

        # Attempt swaps between adjacent temperatures
        if iter % swap_interval == 0
            for i in 1:(n_chains-1)
                # Metropolis criterion for swapping chains i and i+1
                U_i = U(chains[i])
                U_j = U(chains[i+1])
                T_i = temperatures[i]
                T_j = temperatures[i+1]

                # Swap probability: exp[(1/T_j - 1/T_i)(U_j - U_i)]
                Δ = (1.0/T_j - 1.0/T_i) * (U_j - U_i)

                n_swap_attempts += 1

                if isfinite(Δ) && (rand() < exp(Δ))
                    # Swap
                    chains[i], chains[i+1] = chains[i+1], chains[i]
                    n_swaps += 1
                end
            end
        end

        if verbose && (iter % 50 == 0 || iter <= 5)
            acc_rates = n_accepted ./ iter
            println("  PT-HMC iter $iter: acc=$(round.(acc_rates, digits=3))")
        end
    end

    # Return samples from coldest chain (highest index)
    samples = samples_per_chain[end][:, (warmup+1):end]

    acceptance_rates = n_accepted ./ (N_samples + warmup)
    swap_rate = n_swap_attempts > 0 ? n_swaps / n_swap_attempts : 0.0

    diagnostics = Dict(
        "acceptance_rates" => acceptance_rates,
        "swap_rate" => swap_rate,
        "n_samples" => N_samples,
        "warmup" => warmup,
        "n_chains" => n_chains
    )

    if verbose
        println("  PT-HMC complete: acc=$(round.(acceptance_rates, digits=3)), swap=$(round(swap_rate, digits=3))")
    end

    return samples, diagnostics
end


"""
    adapt_step_size(acceptance_rate, step_size, target_rate; learning_rate=0.01)

Dual averaging step size adaptation to achieve target acceptance rate.

# Arguments
- `acceptance_rate`: Current acceptance rate
- `step_size`: Current step size
- `target_rate`: Target acceptance rate (typically 0.65)
- `learning_rate`: Adaptation rate (default 0.01)

# Returns
- `new_step_size`: Adapted step size
"""
function adapt_step_size(
    acceptance_rate::Float64,
    step_size::Float64,
    target_rate::Float64;
    learning_rate::Float64 = 0.01
)
    # Dual averaging: increase step size if accepting too much, decrease if rejecting too much
    log_step = log(step_size)
    log_step_new = log_step + learning_rate * (target_rate - acceptance_rate)

    new_step_size = exp(log_step_new)

    # Clamp to reasonable range
    return clamp(new_step_size, 0.001, 1.0)
end
