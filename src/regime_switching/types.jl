Base.@kwdef struct RegimeSwitchConfig
    gate_mode::Symbol = :hard
    tau_eps::Float64 = 1.95
    tau_y::Float64 = 1.95
    beta_eps::Float64 = 1.0
    beta_y::Float64 = 1.0
    bias::Float64 = 0.0
    k_pre::Int = 0
    k_post::Int = 0
    min_len::Int = 1
    use_eps::Bool = true
    use_y::Bool = true
    hard_threshold::Float64 = 0.5
    prob_floor::Float64 = 1e-4
    prob_ceiling::Float64 = 1 - 1e-4
    soft_mixture::Symbol = :logsumexp
end

Base.@kwdef struct GateCalibrationConfig
    target_share::Float64 = 0.1
    tol::Float64 = 1e-4
    maxiter::Int = 50
    use_eps::Bool = true
    use_y::Bool = true
end

struct GateCalibrationResult
    quantile::Float64
    tau_eps::Float64
    tau_y::Float64
    achieved_share::Float64
end

Base.@kwdef struct SwitchingLikelihoodConfig
    gate_mode::Symbol = :hard
    hard_threshold::Float64 = 0.5
    prob_floor::Float64 = 1e-4
    prob_ceiling::Float64 = 1 - 1e-4
    soft_mixture::Symbol = :logsumexp
end

struct SwitchingLikelihoodResult{T}
    total::T
    per_period::Vector{T}
    hard_mask::BitVector
    gate_probs::Vector{T}
    ll_rom::Vector{T}
    ll_fom::Vector{T}
end

struct SurrogateBundle
    path::String
    frozen::Any
    meta::Dict{Any,Any}
    validation_rmse::Any
    payload::Dict{Any,Any}
end
