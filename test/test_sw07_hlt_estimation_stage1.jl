using MacroModelling
using Zygote
import Turing, Pigeons
import Turing: NUTS, sample, logpdf
import ADTypes: AutoZygote
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

include("../models/Smets_Wouters_2007_HLT.jl")
include("models/SW07_nonlinear.jl")

# load data
dat = CSV.read(joinpath(@__DIR__, "data", "usmodel.csv"), DataFrame)
data = KeyedArray(Array(dat)', Variable = Symbol.(strip.(names(dat))), Time = 1:size(dat)[1])

# declare observables as written in csv file
observables_old = [:dy, :dc, :dinve, :labobs, :pinfobs, :dw, :robs]

# Subsample (1960Q1-2004Q4)
sample_idx = 47:230
data = data(observables_old, sample_idx)

# declare observables as written in model
observables = [:dy, :dc, :dinve, :labobs, :pinfobs, :dwobs, :robs]
data = rekey(data, :Variable => observables)

# Build paper-calibrated baseline parameters from SW07_nonlinear where available
hlt_param_names = string.(Smets_Wouters_2007_HLT.parameters)
paper_param_map = Dict(get_parameters(SW07_nonlinear, values = true))
base_values = copy(Smets_Wouters_2007_HLT.parameter_values)
for (i, name) in enumerate(hlt_param_names)
    if haskey(paper_param_map, name)
        base_values[i] = paper_param_map[name]
    end
end

# Only estimate price Phillips curve parameters (paper focus)
est_names = ["cprobp", "cindp", "curvp"]
est_idx = map(name -> findfirst(==(name), hlt_param_names), est_names)
@assert all(!isnothing, est_idx) "Missing pricing parameter(s) in Smets_Wouters_2007_HLT."
est_idx = Int.(est_idx)
init_params = base_values[est_idx]

dists = [
    Beta(0.5, 0.10, 0.5, 0.95, μσ = true),  # cprobp
    Beta(0.5, 0.15, 0.01, 0.99, μσ = true),  # cindp
    Normal(75.0, 25.0)                      # curvp (Kimball curvature, HLT paper)
]

Turing.@model function SW07_HLT_PC_slope_loglikelihood(data, m, observables, base_values, est_idx, algorithm, filter)
    all_params ~ Turing.arraydist(dists)
    cprobp, cindp, curvp = all_params

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        params = map(eachindex(base_values)) do i
            if i == est_idx[1]
                cprobp
            elseif i == est_idx[2]
                cindp
            elseif i == est_idx[3]
                curvp
            else
                base_values[i]
            end
        end

        llh = get_loglikelihood(
            m,
            data(observables),
            params,
            presample_periods = 4,
            initial_covariance = :diagonal,
            algorithm = algorithm,
            filter = filter,
        )

        Turing.@addlogprob! llh
    end
end

function run_stage(; algorithm, filter, label, n_samples = 500)
    Random.seed!(30)
    loglik = SW07_HLT_PC_slope_loglikelihood(
        data,
        Smets_Wouters_2007_HLT,
        observables,
        base_values,
        est_idx,
        algorithm,
        filter,
    )

    mode = Turing.maximum_a_posteriori(loglik, Optim.NelderMead(), initial_params = init_params)
    println("Mode variable values ($(label)): $(mode.values); Mode loglikelihood: $(mode.lp)")

    samps = @time Turing.sample(loglik, NUTS(adtype = AutoZygote()), n_samples, progress = true, initial_params = mode.values)
    println(samps)
    println("Mean variable values ($(label)): $(mean(samps).nt.mean)")
end

# Stage 1a: first-order perturbation (Kalman filter)
run_stage(algorithm = :first_order, filter = :kalman, label = "first_order")

# Stage 1b: second-order perturbation (pruned, inversion filter)
run_stage(algorithm = :pruned_second_order, filter = :inversion, label = "pruned_second_order")
