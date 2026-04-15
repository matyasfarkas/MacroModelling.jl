#!/usr/bin/env julia
using Serialization
using Random
using LinearAlgebra
using Statistics
using Dates
import Distributions
using AdvancedHMC
using ForwardDiff
using LogDensityProblems
using MacroModelling
using AxisKeys

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_model_loader_utils.jl"))

script_repo_root() = normpath(joinpath(@__DIR__, ".."))

@inline logistic(x) = inv(1 + exp(-x))

function bounded_transform(z::Real, low::Real, high::Real)
    σ = logistic(z)
    θ = low + (high - low) * σ
    logJ = log(high - low) + log(σ) + log1p(-σ)
    return θ, logJ
end

function inv_bounded_transform(θ::Real, low::Real, high::Real)
    p = (θ - low) / (high - low)
    p = clamp(p, 1e-9, 1 - 1e-9)
    return log(p / (1 - p))
end

function load_hlt_model(model_name::AbstractString; force_obc::Bool, force_no_obc::Bool)
    if force_obc && force_no_obc
        error("Specify only one of --use-obc or --no-obc.")
    end
    use_obc = force_obc || !force_no_obc
    if !force_obc && !force_no_obc && model_name != "Smets_Wouters_2007_HLT"
        use_obc = true
    end
    selected_name = use_obc ? "Smets_Wouters_2007_HLT_obc" : "Smets_Wouters_2007_HLT"
    return load_hlt_model(script_repo_root(), selected_name; mod = @__MODULE__), use_obc
end

function hlt_paper_baseline_params(mm_model)
    base_values = copy(mm_model.parameter_values)
    return base_values
end

function params_from_theta(theta::AbstractVector,
                           theta_names::Vector{Symbol},
                           mm_model,
                           base_params::AbstractVector)
    params = eltype(theta).(base_params)
    if isempty(theta_names)
        return params
    end
    idx = indexin(theta_names, mm_model.parameters)
    if any(isnothing, idx)
        error("Theta names not found in $(mm_model.model_name) parameters.")
    end
    for (i, j) in enumerate(Int.(idx))
        params[j] = theta[i]
    end
    return params
end

function logprior_theta(theta::AbstractVector{<:Real}, prior_mu::Vector{Float64}, prior_sd::Vector{Float64})
    function beta_ab_from_mu_sigma(mu::Float64, sigma::Float64)
        alpha = ((1 - mu) / sigma^2 - 1 / mu) * mu^2
        beta = alpha * (1 / mu - 1)
        return alpha, beta
    end
    a1, b1 = beta_ab_from_mu_sigma(prior_mu[1], prior_sd[1])
    a2, b2 = beta_ab_from_mu_sigma(prior_mu[2], prior_sd[2])
    prior_cprobp = Distributions.truncated(Distributions.Beta(a1, b1), 0.5, 0.95)
    prior_cindp = Distributions.truncated(Distributions.Beta(a2, b2), 0.01, 0.99)
    prior_curvp = Distributions.Normal(prior_mu[3], prior_sd[3])
    return Distributions.logpdf(prior_cprobp, theta[1]) +
           Distributions.logpdf(prior_cindp, theta[2]) +
           Distributions.logpdf(prior_curvp, theta[3])
end

function linear_loglikelihood(mm_model,
                              obs_data_ka,
                              theta::AbstractVector,
                              theta_names::Vector{Symbol},
                              base_params::AbstractVector)
    params = params_from_theta(theta, theta_names, mm_model, base_params)
    ll_vec = MacroModelling.get_loglikelihood_per_period(mm_model, obs_data_ka, params;
                                                         algorithm = :first_order,
                                                         filter = :kalman,
                                                         on_failure_loglikelihood = -1e12,
                                                         presample_periods = 0,
                                                         initial_covariance = :theoretical,
                                                         verbose = false)
    ll = sum(ll_vec)
    if !isfinite(ll) || ll < -1e11
        return -Inf
    end
    return ll
end

function theta_laplace_log_marginal(theta_mean::Vector{Float64}, theta_cov::Matrix{Float64},
                                    loglik_val::Float64, logprior_val::Float64)
    d = length(theta_mean)
    cov_sym = Symmetric(theta_cov)
    logdet_cov = logabsdet(cov_sym)[1]
    return loglik_val + logprior_val + 0.5 * (d * log(2 * pi) + logdet_cov)
end

struct LinearLogDensity
    logposterior::Function
    gradlogposterior::Function
    dim::Int
end

LogDensityProblems.logdensity(p::LinearLogDensity, x::AbstractVector) = p.logposterior(x)
LogDensityProblems.dimension(p::LinearLogDensity) = p.dim
LogDensityProblems.capabilities(::Type{LinearLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(p::LinearLogDensity, x::AbstractVector)
    lp = p.logposterior(x)
    grad = p.gradlogposterior(x)
    return lp, grad
end

function theta_from_unconstrained(z::AbstractVector{<:Real})
    θ1, logJ1 = bounded_transform(z[1], 0.5, 0.95)
    θ2, logJ2 = bounded_transform(z[2], 0.01, 0.99)
    θ3 = z[3]
    return [θ1, θ2, θ3], (logJ1 + logJ2)
end

function main()
    synthetic_path = first_positional_arg(ARGS)
    synthetic_path === nothing && error("Usage: julia hlt_hmc_linear_diagnose.jl <synthetic_data.jls> [--out-dir=...]")

    out_dir = parse_arg_string(ARGS, "--out-dir", "")
    if out_dir == ""
        out_dir = joinpath(@__DIR__, "..", "data", "hlt_hmc_linear_diagnose_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    end
    mkpath(out_dir)

    n_samples = parse_arg_int(ARGS, "--samples", 1000)
    n_adapt = parse_arg_int(ARGS, "--adapt", 500)
    target_accept = parse_arg_float(ARGS, "--target-accept", 0.85)
    seed = parse_arg_int(ARGS, "--seed", 123)
    obs_window = parse_arg_int(ARGS, "--obs-window", 0)
    obs_count = parse_arg_int(ARGS, "--obs-count", 0)
    obs_names_arg = parse_arg_string(ARGS, "--obs-names", "")
    prior_mu = [
        parse_arg_float(ARGS, "--prior-cprobp-mean", 0.5),
        parse_arg_float(ARGS, "--prior-cindp-mean", 0.5),
        parse_arg_float(ARGS, "--prior-curvp-mean", 75.0),
    ]
    prior_sd = [
        parse_arg_float(ARGS, "--prior-cprobp-sd", 0.10),
        parse_arg_float(ARGS, "--prior-cindp-sd", 0.15),
        parse_arg_float(ARGS, "--prior-curvp-sd", 25.0),
    ]
    init_theta = [
        parse_arg_float(ARGS, "--init-cprobp", prior_mu[1]),
        parse_arg_float(ARGS, "--init-cindp", prior_mu[2]),
        parse_arg_float(ARGS, "--init-curvp", prior_mu[3]),
    ]

    println("Loading synthetic data: $synthetic_path")
    synthetic = deserialize(synthetic_path)
    model_name = get(synthetic, "model", "Smets_Wouters_2007_HLT")
    mm_model, use_obc = load_hlt_model(model_name; force_obc = "--use-obc" in ARGS, force_no_obc = "--no-obc" in ARGS)
    obs_data = synthetic["obs_data"]
    observables = get(synthetic, "observables", Symbol[])
    theta_names = get(synthetic, "theta_names", Symbol[])
    theta_true = get(synthetic, "theta_true", nothing)

    if isempty(observables)
        error("Synthetic data missing observables.")
    end
    if obs_names_arg != ""
        requested = Symbol.(split(obs_names_arg, ","))
        obs_idx = indexin(requested, observables)
        if any(isnothing, obs_idx)
            missing = requested[isnothing.(obs_idx)]
            error("Unknown observables in --obs-names: $(missing).")
        end
        obs_data = obs_data[Int.(obs_idx), :]
        observables = requested
    elseif obs_count > 0 && obs_count < length(observables)
        obs_data = obs_data[1:obs_count, :]
        observables = observables[1:obs_count]
    end
    if obs_window > 0 && obs_window < size(obs_data, 2)
        obs_data = obs_data[:, end-obs_window+1:end]
        println("Using last $obs_window periods for diagnostics.")
    end

    obs_data_ka = KeyedArray(obs_data; Variable = observables, Time = 1:size(obs_data, 2))

    Random.seed!(seed)
    base_params = hlt_paper_baseline_params(mm_model)
    init_z = [
        inv_bounded_transform(init_theta[1], 0.5, 0.95),
        inv_bounded_transform(init_theta[2], 0.01, 0.99),
        init_theta[3],
    ]

    function logposterior(z::AbstractVector)
        θ, logJ = theta_from_unconstrained(z)
        lp = logprior_theta(θ, prior_mu, prior_sd)
        if !isfinite(lp)
            return -Inf
        end
        ll = linear_loglikelihood(mm_model, obs_data_ka, θ, theta_names, base_params)
        return lp + ll + logJ
    end

    function grad_logposterior(z::AbstractVector{Float64})
        return ForwardDiff.gradient(logposterior, z)
    end

    println("Model: $(mm_model.model_name) (use_obc=$use_obc)")
    if obs_names_arg != ""
        println("Observables subset: $(observables)")
    elseif obs_count > 0 && obs_count < length(get(synthetic, "observables", Symbol[]))
        println("Observables subset: first $obs_count")
    end
    println("Prior means: $(prior_mu) sds: $(prior_sd)")
    println("Init theta: $(init_theta)")
    println("Init z: $(init_z)")
    if theta_true !== nothing
        println("True theta: $(theta_true)")
    end

    println("Testing log posterior at init...")
    lp_init = logposterior(init_z)
    println("  log posterior: $lp_init")
    if !isfinite(lp_init)
        error("Log posterior not finite at init.")
    end
    println("Testing gradient...")
    grad_init = grad_logposterior(init_z)
    println("  grad norm: $(norm(grad_init)) (all finite: $(all(isfinite, grad_init)))")
    if !all(isfinite, grad_init)
        error("Gradient is not finite.")
    end

    d = length(init_z)
    metric = DiagEuclideanMetric([1.0, 1.0, prior_sd[3]^2])
    log_density = LinearLogDensity(logposterior, grad_logposterior, d)
    hamiltonian = Hamiltonian(metric, log_density)

    println("Finding initial step size...")
    initial_eps = find_good_stepsize(hamiltonian, init_z)
    println("  initial eps: $initial_eps")

    integrator = Leapfrog(initial_eps)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_accept, integrator))

    println("Running NUTS: samples=$n_samples adapt=$n_adapt target_accept=$target_accept")
    samples, stats = sample(hamiltonian, kernel, init_z, n_samples + n_adapt, adaptor, n_adapt;
                            progress = true, verbose = false)

    z_draws = reduce(vcat, [samples[i][1:d]' for i in (n_adapt+1):(n_samples+n_adapt)])
    theta_draws = similar(z_draws)
    for i in 1:size(z_draws, 1)
        θ, _ = theta_from_unconstrained(view(z_draws, i, :))
        theta_draws[i, :] .= θ
    end
    theta_mean = vec(mean(theta_draws, dims = 1))
    theta_cov = Statistics.cov(theta_draws)

    loglik_mean = linear_loglikelihood(mm_model, obs_data_ka, theta_mean, theta_names, base_params)
    logprior_mean = logprior_theta(theta_mean, prior_mu, prior_sd)
    log_marginal = theta_laplace_log_marginal(theta_mean, theta_cov, loglik_mean, logprior_mean)

    accept_rate = mean([s.acceptance_rate for s in stats[(n_adapt+1):end]])
    step_size = mean([s.step_size for s in stats[(n_adapt+1):end]])
    n_div = count(s -> (hasproperty(s, :is_divergent) ? getproperty(s, :is_divergent) :
                        hasproperty(s, :numerical_error) ? getproperty(s, :numerical_error) : false),
                  stats[(n_adapt+1):end])

    println("\nPosterior mean: $theta_mean")
    println("Loglik at posterior mean: $(round(loglik_mean, digits = 2))")
    println("Laplace log marginal (theta-only): $(round(log_marginal, digits = 2))")
    println("Acceptance rate: $(round(accept_rate, digits = 3))")
    println("Step size: $(round(step_size, sigdigits = 3))")
    println("Divergences: $n_div")

    results_path = joinpath(out_dir, "hlt_hmc_linear_diagnose.jls")
    serialize(results_path, Dict(
        "samples" => samples,
        "stats" => stats,
        "z_draws" => z_draws,
        "theta_draws" => theta_draws,
        "theta_mean" => theta_mean,
        "theta_cov" => theta_cov,
        "loglik_post_mean" => loglik_mean,
        "logprior_post_mean" => logprior_mean,
        "log_marginal_theta_laplace" => log_marginal,
        "acceptance_rate" => accept_rate,
        "step_size" => step_size,
        "divergences" => n_div,
        "prior_mean" => prior_mu,
        "prior_sd" => prior_sd,
        "init_theta" => init_theta,
        "init_z" => init_z,
        "theta_true" => theta_true,
        "model" => mm_model.model_name,
        "use_obc" => use_obc,
        "obs_window" => obs_window,
        "obs_count" => obs_count,
        "obs_names" => observables,
        "synthetic_path" => synthetic_path,
    ))

    summary_path = joinpath(out_dir, "summary.txt")
    open(summary_path, "w") do io
        println(io, "HLT manual HMC linear diagnose")
        println(io, "model: $(mm_model.model_name) use_obc=$(use_obc)")
        println(io, "synthetic: $synthetic_path")
        println(io, "samples: $n_samples adapt: $n_adapt")
        println(io, "prior mean: $(prior_mu) sd: $(prior_sd)")
        println(io, "init theta: $(init_theta)")
        println(io, "init z: $(init_z)")
        println(io, "theta mean: $(theta_mean)")
        println(io, "obs window: $(obs_window)")
        if obs_names_arg != ""
            println(io, "obs names: $(observables)")
        elseif obs_count > 0
            println(io, "obs count: $(obs_count)")
        end
        println(io, "loglik post mean: $(loglik_mean)")
        println(io, "laplace log marginal: $(log_marginal)")
        println(io, "acceptance rate: $(accept_rate)")
        println(io, "step size: $(step_size)")
        println(io, "divergences: $(n_div)")
    end

    println("Saved results: $results_path")
    println("Summary: $summary_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
