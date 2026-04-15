@stable default_mode = "disable" begin

# FIX M-01: Use symbolic constant for log(2π) instead of hardcoded literal
const LOG_2PI = log(2π)  # ≈ 1.8378770664093453

# Specialization for :kalman filter
function calculate_loglikelihood(::Val{:kalman}, 
                                algorithm, 
                                observables, 
                                𝐒, 
                                data_in_deviations, 
                                TT, 
                                presample_periods, 
                                initial_covariance, 
                                state, 
                                warmup_iterations, 
                                filter_algorithm, 
                                opts,
                                on_failure_loglikelihood) #; 
                                # timer::TimerOutput = TimerOutput())
    return calculate_kalman_filter_loglikelihood(observables, 
                                                𝐒, 
                                                data_in_deviations, 
                                                TT, 
                                                presample_periods = presample_periods, 
                                                initial_covariance = initial_covariance, 
                                                # timer = timer, 
                                                opts = opts,
                                                on_failure_loglikelihood = on_failure_loglikelihood)
end

function calculate_kalman_filter_loglikelihood(observables::Vector{Symbol}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                on_failure_loglikelihood::U = -Inf,
                                                presample_periods::Int = 0, 
                                                initial_covariance::Symbol = :theoretical,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

    calculate_kalman_filter_loglikelihood(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, opts = opts, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

function calculate_kalman_filter_loglikelihood(observables::Vector{String}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                presample_periods::Int = 0, 
                                                on_failure_loglikelihood::U = -Inf,
                                                initial_covariance::Symbol = :theoretical,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    obs_idx = @ignore_derivatives convert(Vector{Int},indexin(observables,sort(union(T.aux,T.var,T.exo_present))))

    calculate_kalman_filter_loglikelihood(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, opts = opts, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

function calculate_kalman_filter_loglikelihood(observables_index::Vector{Int}, 
                                                𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}}, 
                                                data_in_deviations::Matrix{S},
                                                T::timings; 
                                                # timer::TimerOutput = TimerOutput(), 
                                                presample_periods::Int = 0,
                                                initial_covariance::Symbol = :theoretical,
                                                lyapunov_algorithm::Symbol = :doubling,
                                                on_failure_loglikelihood::U = -Inf,
                                                opts::CalculationOptions = merge_calculation_options())::S where {S <: Real, U <: AbstractFloat}
    observables_and_states = @ignore_derivatives sort(union(T.past_not_future_and_mixed_idx,observables_index))

    A = 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(S, length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx,observables_and_states)),:]
    B = 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)),:]

    𝐁 = B * B'

    # Gaussian Prior
    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, opts = opts)
    # timer = timer, 

    return run_kalman_iterations(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods, verbose = opts.verbose, on_failure_loglikelihood = on_failure_loglikelihood)
    # timer = timer, 
end

function calculate_kalman_filter_loglikelihood_per_period(observables::Vector{Symbol},
                                                            𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}},
                                                            data_in_deviations::Matrix{S},
                                                            T::timings;
                                                            on_failure_loglikelihood::U = -Inf,
                                                            presample_periods::Int = 0,
                                                            initial_covariance::Symbol = :theoretical,
                                                            opts::CalculationOptions = merge_calculation_options())::Vector{S} where {S <: Real, U <: AbstractFloat}
    obs_idx = @ignore_derivatives convert(Vector{Int}, indexin(observables, sort(union(T.aux, T.var, T.exo_present))))

    calculate_kalman_filter_loglikelihood_per_period(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, opts = opts, on_failure_loglikelihood = on_failure_loglikelihood)
end

function calculate_kalman_filter_loglikelihood_per_period(observables::Vector{String},
                                                            𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}},
                                                            data_in_deviations::Matrix{S},
                                                            T::timings;
                                                            presample_periods::Int = 0,
                                                            on_failure_loglikelihood::U = -Inf,
                                                            initial_covariance::Symbol = :theoretical,
                                                            opts::CalculationOptions = merge_calculation_options())::Vector{S} where {S <: Real, U <: AbstractFloat}
    obs_idx = @ignore_derivatives convert(Vector{Int}, indexin(observables, sort(union(T.aux, T.var, T.exo_present))))

    calculate_kalman_filter_loglikelihood_per_period(obs_idx, 𝐒, data_in_deviations, T, presample_periods = presample_periods, initial_covariance = initial_covariance, opts = opts, on_failure_loglikelihood = on_failure_loglikelihood)
end

function calculate_kalman_filter_loglikelihood_per_period(observables_index::Vector{Int},
                                                            𝐒::Union{Matrix{S},Vector{AbstractMatrix{S}}},
                                                            data_in_deviations::Matrix{S},
                                                            T::timings;
                                                            presample_periods::Int = 0,
                                                            initial_covariance::Symbol = :theoretical,
                                                            lyapunov_algorithm::Symbol = :doubling,
                                                            on_failure_loglikelihood::U = -Inf,
                                                            opts::CalculationOptions = merge_calculation_options())::Vector{S} where {S <: Real, U <: AbstractFloat}
    observables_and_states = @ignore_derivatives sort(union(T.past_not_future_and_mixed_idx, observables_index))

    A = 𝐒[observables_and_states, 1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(S, length(observables_and_states)))[@ignore_derivatives(indexin(T.past_not_future_and_mixed_idx, observables_and_states)), :]
    B = 𝐒[observables_and_states, T.nPast_not_future_and_mixed + 1:end]

    C = ℒ.diagm(ones(length(observables_and_states)))[@ignore_derivatives(indexin(sort(observables_index), observables_and_states)), :]

    𝐁 = B * B'

    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, opts = opts)

    return run_kalman_iterations_per_period(A, 𝐁, C, P, data_in_deviations, presample_periods = presample_periods, verbose = opts.verbose, on_failure_loglikelihood = on_failure_loglikelihood)
end

# Specialization for :theoretical
function get_initial_covariance(::Val{:theoretical}, 
                                A::AbstractMatrix{S}, 
                                B::AbstractMatrix{S}; 
                                opts::CalculationOptions = merge_calculation_options())::Matrix{S} where S <: Real
                                # timer::TimerOutput = TimerOutput(), 
    P, _ = solve_lyapunov_equation(A, B, 
                                    lyapunov_algorithm = opts.lyapunov_algorithm, 
                                    tol = opts.tol.lyapunov_tol,
                                    acceptance_tol = opts.tol.lyapunov_acceptance_tol,
                                    verbose = opts.verbose) # timer = timer, 

    return P
end


# Specialization for :diagonal
function get_initial_covariance(::Val{:diagonal}, 
                                A::AbstractMatrix{S}, 
                                B::AbstractMatrix{S}; 
                                opts::CalculationOptions = merge_calculation_options())::Matrix{S} where S <: Real
                                # timer::TimerOutput = TimerOutput(), 
    P = @ignore_derivatives collect(ℒ.I(size(A, 1)) * 10.0)
    return P
end


function run_kalman_iterations(A::Matrix{S}, 
                                𝐁::Matrix{S},
                                C::Matrix{Float64}, 
                                P::Matrix{S}, 
                                data_in_deviations::Matrix{S}; 
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                # timer::TimerOutput = TimerOutput(),
                                verbose::Bool = false)::S where {S <: Float64, U <: AbstractFloat}
    # @timeit_debug timer "Calculate Kalman filter" begin

    u = zeros(S, size(C,2))

    z = C * u

    ztmp = similar(z)

    loglik = S(0.0)

    utmp = similar(u)

    Ctmp = similar(C)

    F = similar(C * C')

    K = similar(C')
    # Ktmp = similar(C')

    tmp = similar(P)
    Ptmp = similar(P)

    # @timeit_debug timer "Loop" begin
    for t in 1:size(data_in_deviations, 2)
        if !all(isfinite.(z)) 
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood 
        end

        ℒ.axpby!(1, data_in_deviations[:, t], -1, z)
        # v = data_in_deviations[:, t] - z

        ℒ.mul!(Ctmp, C, P) # use Octavian.jl
        ℒ.mul!(F, Ctmp, C')
        # F = C * P * C'

        # @timeit_debug timer "LU factorisation" begin
        luF = RF.lu!(F, check = false) ### has to be LU since F will always be symmetric and positive semi-definite but not positive definite (due to linear dependencies)
        # end # timeit_debug

        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        # invF = inv(luF) ###

        # @timeit_debug timer "LU div" begin
        if t > presample_periods
            ℒ.ldiv!(ztmp, luF, z)
            loglik += log(Fdet) + ℒ.dot(z', ztmp) ###
            # loglik += log(Fdet) + z' * invF * z###
            # loglik += log(Fdet) + v' * invF * v###
        end

        # ℒ.mul!(Ktmp, P, C')
        # ℒ.mul!(K, Ktmp, invF)
        ℒ.mul!(K, P, C')
        ℒ.rdiv!(K, luF)
        # K = P * Ct / luF
        # K = P * C' * invF

        # end # timeit_debug
        # @timeit_debug timer "Matmul" begin

        ℒ.mul!(tmp, K, C)
        ℒ.mul!(Ptmp, tmp, P)
        ℒ.axpy!(-1, Ptmp, P)

        ℒ.mul!(Ptmp, A, P)
        ℒ.mul!(P, Ptmp, A')
        ℒ.axpy!(1, 𝐁, P)
        # P = A * (P - K * C * P) * A' + 𝐁

        ℒ.mul!(u, K, z, 1, 1)
        ℒ.mul!(utmp, A, u)
        u .= utmp
        # u = A * (u + K * v)

        ℒ.mul!(z, C, u)
        # z = C * u

        # end # timeit_debug
    end

    # end # timeit_debug
    # end # timeit_debug

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * LOG_2PI) / 2 
end

function run_kalman_iterations_per_period(A::Matrix{S},
                                            𝐁::Matrix{S},
                                            C::Matrix{Float64},
                                            P::Matrix{S},
                                            data_in_deviations::Matrix{S};
                                            presample_periods::Int = 0,
                                            on_failure_loglikelihood::U = -Inf,
                                            verbose::Bool = false)::Vector{S} where {S <: Float64, U <: AbstractFloat}
    u = zeros(S, size(C, 2))

    z = C * u

    ztmp = similar(z)

    utmp = similar(u)

    Ctmp = similar(C)

    F = similar(C * C')

    K = similar(C')

    tmp = similar(P)
    Ptmp = similar(P)

    T = size(data_in_deviations, 2)
    n_obs = size(data_in_deviations, 1)
    loglik = zeros(S, T)
    const_term = -0.5 * n_obs * LOG_2PI

    for t in 1:T
        if !all(isfinite.(z))
            if verbose println("KF not finite at step $t") end
            return fill(zero(S) + on_failure_loglikelihood, T)
        end

        ℒ.axpby!(1, data_in_deviations[:, t], -1, z)

        ℒ.mul!(Ctmp, C, P)
        ℒ.mul!(F, Ctmp, C')

        luF = RF.lu!(F, check = false)

        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return fill(zero(S) + on_failure_loglikelihood, T)
        end

        Fdet = ℒ.det(luF)

        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return fill(zero(S) + on_failure_loglikelihood, T)
        end

        if t > presample_periods
            ℒ.ldiv!(ztmp, luF, z)
            loglik[t] = -0.5 * (log(Fdet) + ℒ.dot(z', ztmp)) + const_term
        end

        ℒ.mul!(K, P, C')
        ℒ.rdiv!(K, luF)

        ℒ.mul!(tmp, K, C)
        ℒ.mul!(Ptmp, tmp, P)
        ℒ.axpy!(-1, Ptmp, P)

        ℒ.mul!(Ptmp, A, P)
        ℒ.mul!(P, Ptmp, A')
        ℒ.axpy!(1, 𝐁, P)

        ℒ.mul!(u, K, z, 1, 1)
        ℒ.mul!(utmp, A, u)
        u .= utmp

        ℒ.mul!(z, C, u)
    end

    return loglik
end



function run_kalman_iterations(A::Matrix{S}, 
                                𝐁::Matrix{S}, 
                                C::Matrix{Float64}, 
                                P::Matrix{S}, 
                                data_in_deviations::Matrix{S}; 
                                presample_periods::Int = 0,
                                on_failure_loglikelihood::U = -Inf,
                                # timer::TimerOutput = TimerOutput(),
                                verbose::Bool = false)::S where {S <: ℱ.Dual, U <: AbstractFloat}
    # @timeit_debug timer "Calculate Kalman filter - forward mode AD" begin
    u = zeros(S, size(C,2))

    z = C * u

    loglik = S(0.0)

    F = similar(C * C')

    K = similar(C')

    for t in 1:size(data_in_deviations, 2)
        if !all(isfinite.(z)) 
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood 
        end

        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false) ###

        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        invF = inv(luF) ###

        if t > presample_periods
            loglik += log(Fdet) + ℒ.dot(v, invF, v)###
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    # end # timeit_debug

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * LOG_2PI) / 2 
end

function run_kalman_iterations_per_period(A::Matrix{S},
                                            𝐁::Matrix{S},
                                            C::Matrix{Float64},
                                            P::Matrix{S},
                                            data_in_deviations::Matrix{S};
                                            presample_periods::Int = 0,
                                            on_failure_loglikelihood::U = -Inf,
                                            verbose::Bool = false)::Vector{S} where {S <: ℱ.Dual, U <: AbstractFloat}
    u = zeros(S, size(C, 2))

    z = C * u

    nT = size(data_in_deviations, 2)
    n_obs = size(data_in_deviations, 1)
    loglik = zeros(S, nT)
    const_term = -0.5 * n_obs * LOG_2PI

    F = similar(C * C')

    K = similar(C')

    for t in 1:nT
        if !all(isfinite.(z))
            if verbose println("KF not finite at step $t") end
            return fill(zero(S) + on_failure_loglikelihood, nT)
        end

        v = data_in_deviations[:, t] - z

        F = C * P * C'

        luF = ℒ.lu(F, check = false)

        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return fill(zero(S) + on_failure_loglikelihood, nT)
        end

        Fdet = ℒ.det(luF)

        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return fill(zero(S) + on_failure_loglikelihood, nT)
        end

        invF = inv(luF)

        if t > presample_periods
            loglik[t] = -0.5 * (log(Fdet) + ℒ.dot(v, invF, v)) + const_term
        end

        K = P * C' * invF

        P = A * (P - K * C * P) * A' + 𝐁

        u = A * (u + K * v)

        z = C * u
    end

    return loglik
end

end # dispatch_doctor

function rrule(::typeof(run_kalman_iterations), 
                    A, 
                    𝐁, 
                    C, 
                    P, 
                    data_in_deviations; 
                    presample_periods = 0,
                    on_failure_loglikelihood = -Inf,
                    # timer::TimerOutput = TimerOutput(),
                    verbose::Bool = false)
    # @timeit_debug timer "Calculate Kalman filter - forward" begin
    T = size(data_in_deviations, 2) + 1

    z = zeros(size(data_in_deviations, 1))

    ū = zeros(size(C,2))

    P̄ = deepcopy(P) 

    temp_N_N = similar(P)

    PCtmp = similar(C')

    F = similar(C * C')

    u = [similar(ū) for _ in 1:T] # used in backward pass

    P = [copy(P̄) for _ in 1:T] # used in backward pass

    CP = [zero(C) for _ in 1:T] # used in backward pass

    K = [similar(C') for _ in 1:T] # used in backward pass

    invF = [similar(F) for _ in 1:T] # used in backward pass

    v = [zeros(size(data_in_deviations, 1)) for _ in 1:T] # used in backward pass

    loglik = 0.0

    # @timeit_debug timer "Loop" begin
        
    for t in 2:T
        if !all(isfinite.(z)) 
            if verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent() 
        end

        v[t] .= data_in_deviations[:, t-1] .- z#[t-1]

        # CP[t] .= C * P̄[t-1]
        ℒ.mul!(CP[t], C, P̄)#[t-1])
    
        # F[t] .= CP[t] * C'
        ℒ.mul!(F, CP[t], C')
    
        luF = RF.lu(F, check = false)
    
        if !ℒ.issuccess(luF)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        Fdet = ℒ.det(luF)

        # Early return if determinant is too small, indicating numerical instability.
        if Fdet < eps(Float64)
            if verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood, x -> NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        
        # invF[t] .= inv(luF)
        copy!(invF[t], inv(luF))
        
        if t - 1 > presample_periods
            loglik += log(Fdet) + ℒ.dot(v[t], invF[t], v[t])
        end

        # K[t] .= P̄[t-1] * C' * invF[t]
        ℒ.mul!(PCtmp, P̄, C')
        ℒ.mul!(K[t], PCtmp, invF[t])

        # P[t] .= P̄[t-1] - K[t] * CP[t]
        ℒ.mul!(P[t], K[t], CP[t], -1, 0)
        P[t] .+= P̄
    
        # P̄[t] .= A * P[t] * A' + 𝐁
        ℒ.mul!(temp_N_N, P[t], A')
        ℒ.mul!(P̄, A, temp_N_N)
        P̄ .+= 𝐁

        # u[t] .= K[t] * v[t] + ū[t-1]
        ℒ.mul!(u[t], K[t], v[t])
        u[t] .+= ū
        
        # ū[t] .= A * u[t]
        ℒ.mul!(ū, A, u[t])

        # z[t] .= C * ū[t]
        ℒ.mul!(z, C, ū)
    end

    llh = -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * LOG_2PI) / 2 

    # initialise derivative variables
    ∂A = zero(A)
    ∂F = zero(F)
    ∂Faccum = zero(F)
    ∂P = zero(P̄)
    ∂ū = zero(ū)
    ∂v = zero(v[1])
    ∂𝐁 = zero(𝐁)
    ∂data_in_deviations = zero(data_in_deviations)
    vtmp = zero(v[1])
    Ptmp = zero(P[1])

    # end # timeit_debug
    # end # timeit_debug

    # pullback
    function kalman_pullback(∂llh)
        # @timeit_debug timer "Calculate Kalman filter - reverse" begin
        ℒ.rmul!(∂A, 0)
        ℒ.rmul!(∂Faccum, 0)
        ℒ.rmul!(∂P, 0)
        ℒ.rmul!(∂ū, 0)
        ℒ.rmul!(∂𝐁, 0)

        # @timeit_debug timer "Loop" begin
        for t in T:-1:2
            if t > presample_periods + 1
                # ∂llh∂F
                # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
                # ∂F = invF[t]' - invF[t]' * v[t] * v[t]' * invF[t]'
                ℒ.mul!(∂F, v[t], v[t]')
                ℒ.mul!(invF[1], invF[t]', ∂F) # using invF[1] as temporary storage
                ℒ.mul!(∂F, invF[1], invF[t]')
                ℒ.axpby!(1, invF[t]', -1, ∂F)
        
                # ∂llh∂ū
                # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
                # z[t] .= C * ū[t]
                # ∂v = (invF[t]' + invF[t]) * v[t]
                copy!(invF[1], invF[t]' .+ invF[t])
                # copy!(invF[1], invF[t]) # using invF[1] as temporary storage
                # ℒ.axpy!(1, invF[t]', invF[1]) # using invF[1] as temporary storage
                ℒ.mul!(∂v, invF[1], v[t])
                # ℒ.mul!(∂ū∂v, C', v[1])
            else
                ℒ.rmul!(∂F, 0)
                ℒ.rmul!(∂v, 0)
            end
        
            # ∂F∂P
            # F[t] .= C * P̄[t-1] * C'
            # ∂P += C' * (∂F + ∂Faccum) * C
            ℒ.axpy!(1, ∂Faccum, ∂F)
            ℒ.mul!(PCtmp, C', ∂F) 
            ℒ.mul!(∂P, PCtmp, C, 1, 1) 
        
            # ∂ū∂P
            # K[t] .= P̄[t-1] * C' * invF[t]
            # u[t] .= K[t] * v[t] + ū[t-1]
            # ū[t] .= A * u[t]
            # ∂P += A' * ∂ū * v[t]' * invF[t]' * C
            ℒ.mul!(CP[1], invF[t]', C) # using CP[1] as temporary storage
            ℒ.mul!(PCtmp, ∂ū , v[t]')
            ℒ.mul!(P[1], PCtmp , CP[1]) # using P[1] as temporary storage
            ℒ.mul!(∂P, A', P[1], 1, 1) 
        
            # ∂ū∂data
            # v[t] .= data_in_deviations[:, t-1] .- z
            # z[t] .= C * ū[t]
            # ∂data_in_deviations[:,t-1] = -C * ∂ū
            ℒ.mul!(u[1], A', ∂ū)
            ℒ.mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
            ℒ.axpy!(1, ∂v, v[1])
            ∂data_in_deviations[:,t-1] .= v[1]
            # ℒ.mul!(∂data_in_deviations[:,t-1], C, ∂ū, -1, 0) # cannot assign to columns in matrix, must be whole matrix 

            # ∂ū∂ū
            # z[t] .= C * ū[t]
            # v[t] .= data_in_deviations[:, t-1] .- z
            # K[t] .= P̄[t-1] * C' * invF[t]
            # u[t] .= K[t] * v[t] + ū[t-1]
            # ū[t] .= A * u[t]
            # step to next iteration
            # ∂ū = A' * ∂ū - C' * K[t]' * A' * ∂ū
            ℒ.mul!(u[1], A', ∂ū) # using u[1] as temporary storage
            ℒ.mul!(v[1], K[t]', u[1]) # using v[1] as temporary storage
            ℒ.mul!(∂ū, C', v[1])
            ℒ.mul!(u[1], C', v[1], -1, 1)
            copy!(∂ū, u[1])
        
            # ∂llh∂ū
            # loglik += logdet(F[t]) + v[t]' * invF[t] * v[t]
            # v[t] .= data_in_deviations[:, t-1] .- z
            # z[t] .= C * ū[t]
            # ∂ū -= ∂ū∂v
            ℒ.mul!(u[1], C', ∂v) # using u[1] as temporary storage
            ℒ.axpy!(-1, u[1], ∂ū)
        
            if t > 2
                # ∂ū∂A
                # ū[t] .= A * u[t]
                # ∂A += ∂ū * u[t-1]'
                ℒ.mul!(∂A, ∂ū, u[t-1]', 1, 1)
        
                # ∂P̄∂A and ∂P̄∂𝐁
                # P̄[t] .= A * P[t] * A' + 𝐁
                # ∂A += ∂P * A * P[t-1]' + ∂P' * A * P[t-1]
                ℒ.mul!(P[1], A, P[t-1]')
                ℒ.mul!(Ptmp ,∂P, P[1])
                ℒ.mul!(P[1], A, P[t-1])
                ℒ.mul!(Ptmp ,∂P', P[1], 1, 1)
                ℒ.axpy!(1, Ptmp, ∂A)
        
                # ∂𝐁 += ∂P
                ℒ.axpy!(1, ∂P, ∂𝐁)
        
                # ∂P∂P
                # P[t] .= P̄[t-1] - K[t] * C * P̄[t-1]
                # P̄[t] .= A * P[t] * A' + 𝐁
                # step to next iteration
                # ∂P = A' * ∂P * A
                ℒ.mul!(P[1], ∂P, A) # using P[1] as temporary storage
                ℒ.mul!(∂P, A', P[1])
        
                # ∂P̄∂P
                # K[t] .= P̄[t-1] * C' * invF[t]
                # P[t] .= P̄[t-1] - K[t] * CP[t]
                # ∂P -= C' * K[t-1]' * ∂P + ∂P * K[t-1] * C 
                ℒ.mul!(PCtmp, ∂P, K[t-1])
                ℒ.mul!(CP[1], K[t-1]', ∂P) # using CP[1] as temporary storage
                ℒ.mul!(∂P, PCtmp, C, -1, 1)
                ℒ.mul!(∂P, C', CP[1], -1, 1)
        
                # ∂ū∂F
                # K[t] .= P̄[t-1] * C' * invF[t]
                # u[t] .= K[t] * v[t] + ū[t-1]
                # ū[t] .= A * u[t]
                # ∂Faccum = -invF[t-1]' * CP[t-1] * A' * ∂ū * v[t-1]' * invF[t-1]'
                ℒ.mul!(u[1], A', ∂ū) # using u[1] as temporary storage
                ℒ.mul!(v[1], CP[t-1], u[1]) # using v[1] as temporary storage
                ℒ.mul!(vtmp, invF[t-1]', v[1], -1, 0)
                ℒ.mul!(invF[1], vtmp, v[t-1]') # using invF[1] as temporary storage
                ℒ.mul!(∂Faccum, invF[1], invF[t-1]')
        
                # ∂P∂F
                # K[t] .= P̄[t-1] * C' * invF[t]
                # P[t] .= P̄[t-1] - K[t] * CP[t]
                # ∂Faccum -= invF[t-1]' * CP[t-1] * ∂P * CP[t-1]' * invF[t-1]'
                ℒ.mul!(CP[1], invF[t-1]', CP[t-1]) # using CP[1] as temporary storage
                ℒ.mul!(PCtmp, CP[t-1]', invF[t-1]')
                ℒ.mul!(K[1], ∂P, PCtmp) # using K[1] as temporary storage
                ℒ.mul!(∂Faccum, CP[1], K[1], -1, 1)
        
            end
        end
        
        ℒ.rmul!(∂P, -∂llh/2)
        ℒ.rmul!(∂A, -∂llh/2)
        ℒ.rmul!(∂𝐁, -∂llh/2)
        ℒ.rmul!(∂data_in_deviations, -∂llh/2)

        # end # timeit_debug
        # end # timeit_debug

        return NoTangent(), ∂A, ∂𝐁, NoTangent(), ∂P, ∂data_in_deviations, NoTangent()
    end
    
    return llh, kalman_pullback
end

@stable default_mode = "disable" begin

function filter_data_with_model(𝓂::ℳ,
    data_in_deviations::KeyedArray{Float64},
    ::Val{:first_order}, # algo
    ::Val{:kalman}; # filter,
    warmup_iterations::Int = 0,
    opts::CalculationOptions = merge_calculation_options(),
    smooth::Bool = true)

    obs_axis = collect(axiskeys(data_in_deviations,1))

    obs_symbols = obs_axis isa String_input ? obs_axis .|> Meta.parse .|> replace_indices : obs_axis

    filtered_and_smoothed = filter_and_smooth(𝓂, data_in_deviations, obs_symbols; opts = opts)

    variables           = filtered_and_smoothed[smooth ? 1 : 5]
    standard_deviations = filtered_and_smoothed[smooth ? 2 : 6]
    shocks              = filtered_and_smoothed[smooth ? 3 : 7]
    decomposition       = filtered_and_smoothed[smooth ? 4 : 8]

    return variables, shocks, standard_deviations, decomposition
end



function filter_and_smooth(𝓂::ℳ, 
                            data_in_deviations::AbstractArray{Float64}, 
                            observables::Vector{Symbol};
                            opts::CalculationOptions = merge_calculation_options())
    # Based on Durbin and Koopman (2012)
    # https://jrnold.github.io/ssmodels-in-stan/filtering-and-smoothing.html#smoothing

    @assert length(observables) == size(data_in_deviations)[1] "Data columns and number of observables are not identical. Make sure the data contains only the selected observables."
    @assert length(observables) <= 𝓂.timings.nExo "Cannot estimate model with more observables than exogenous shocks. Have at least as many shocks as observable variables."

    sort!(observables)

    solve!(𝓂, opts = opts)

    parameters = 𝓂.parameter_values

    SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, parameters, opts = opts)
    
    @assert solution_error < opts.tol.NSSS_acceptance_tol "Could not solve non-stochastic steady state." 

	∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)# |> Matrix

    sol, qme_sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings, opts = opts)

    if solved 𝓂.solution.perturbation.qme_solution = qme_sol end

    A = @views sol[:,1:𝓂.timings.nPast_not_future_and_mixed] * ℒ.diagm(ones(𝓂.timings.nVars))[𝓂.timings.past_not_future_and_mixed_idx,:]

    B = @views sol[:,𝓂.timings.nPast_not_future_and_mixed+1:end]

    C = @views ℒ.diagm(ones(𝓂.timings.nVars))[sort(indexin(observables,sort(union(𝓂.aux,𝓂.var,𝓂.exo_present)))),:]

    𝐁 = B * B'

    P̄ = calculate_covariance(𝓂.parameter_values, 𝓂, opts = opts)[1]

    n_obs = size(data_in_deviations,2)

    v = zeros(size(C,1), n_obs)
    μ = zeros(size(A,1), n_obs+1) # filtered_states
    P = zeros(size(A,1), size(A,1), n_obs+1) # filtered_covariances
    σ = zeros(size(A,1), n_obs) # filtered_standard_deviations
    iF= zeros(size(C,1), size(C,1), n_obs)
    L = zeros(size(A,1), size(A,1), n_obs)
    ϵ = zeros(size(B,2), n_obs) # filtered_shocks

    P[:, :, 1] = P̄

    # Kalman Filter
    for t in axes(data_in_deviations,2)
        v[:, t]     .= data_in_deviations[:, t] - C * μ[:, t]

        F̄ = ℒ.lu(C * P[:, :, t] * C', check = false)

        if !ℒ.issuccess(F̄) 
            @warn "Kalman filter stopped in period $t due to numerical stabiltiy issues."
            break
        end

        iF[:, :, t] .= inv(F̄)
        PCiF         = P[:, :, t] * C' * iF[:, :, t]
        L[:, :, t]  .= A - A * PCiF * C
        P[:, :, t+1].= A * P[:, :, t] * L[:, :, t]' + 𝐁
        σ[:, t]     .= sqrt.(abs.(ℒ.diag(P[:, :, t+1]))) # small numerical errors in this computation
        μ[:, t+1]   .= A * (μ[:, t] + PCiF * v[:, t])
        ϵ[:, t]     .= B' * C' * iF[:, :, t] * v[:, t]
    end


    # Historical shock decompositionm (filter)
    filter_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    filter_decomposition[:,end,:] .= μ[:, 2:end]
    filter_decomposition[:,1:end-2,1] .= B .* repeat(ϵ[:, 1]', size(A,1))
    filter_decomposition[:,end-1,1] .= filter_decomposition[:,end,1] - sum(filter_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        filter_decomposition[:,1:end-2,i] .= A * filter_decomposition[:,1:end-2,i-1]
        filter_decomposition[:,1:end-2,i] .+= B .* repeat(ϵ[:, i]', size(A,1))
        filter_decomposition[:,end-1,i] .= filter_decomposition[:,end,i] - sum(filter_decomposition[:,1:end-2,i],dims=2)
    end
    
    μ̄ = zeros(size(A,1), n_obs) # smoothed_states
    σ̄ = zeros(size(A,1), n_obs) # smoothed_standard_deviations
    ϵ̄ = zeros(size(B,2), n_obs) # smoothed_shocks

    r = zeros(size(A,1))
    N = zeros(size(A,1), size(A,1))

    # Kalman Smoother
    for t in n_obs:-1:1
        r       .= C' * iF[:, :, t] * v[:, t] + L[:, :, t]' * r
        μ̄[:, t] .= μ[:, t] + P[:, :, t] * r
        N       .= C' * iF[:, :, t] * C + L[:, :, t]' * N * L[:, :, t]
        σ̄[:, t] .= sqrt.(abs.(ℒ.diag(P[:, :, t] - P[:, :, t] * N * P[:, :, t]'))) # can go negative
        ϵ̄[:, t] .= B' * r
    end

    # Historical shock decompositionm (smoother)
    smooth_decomposition = zeros(size(A,1), size(B,2)+2, n_obs)

    smooth_decomposition[:,end,:] .= μ̄
    smooth_decomposition[:,1:end-2,1] .= B .* repeat(ϵ̄[:, 1]', size(A,1))
    smooth_decomposition[:,end-1,1] .= smooth_decomposition[:,end,1] - sum(smooth_decomposition[:,1:end-2,1],dims=2)

    for i in 2:size(data_in_deviations,2)
        smooth_decomposition[:,1:end-2,i] .= A * smooth_decomposition[:,1:end-2,i-1]
        smooth_decomposition[:,1:end-2,i] .+= B .* repeat(ϵ̄[:, i]', size(A,1))
        smooth_decomposition[:,end-1,i] .= smooth_decomposition[:,end,i] - sum(smooth_decomposition[:,1:end-2,i],dims=2)
    end

    return μ̄, σ̄, ϵ̄, smooth_decomposition, μ[:, 2:end], σ, ϵ, filter_decomposition
end

end # dispatch_doctor
