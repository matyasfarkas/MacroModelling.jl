#!/usr/bin/env julia
using Serialization
using Random
using Statistics

include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_nn_utils.jl"))
include(joinpath(@__DIR__, "hlt_surrogate", "hlt_sep_surrogate_cli_utils.jl"))

dataset_path = first_positional_arg(ARGS)

if dataset_path === nothing
    error("Usage: julia hlt_sep_surrogate_train.jl <dataset_path> [--epochs=400 --hidden=256 --hidden2=128 --seed=1 --rom-residual=1 --arch=resnet]")
end

arch = Symbol(parse_arg_string(ARGS, "--arch", "mlp"))
nepoch = parse_arg_int(ARGS, "--epochs", arch == :resnet ? 600 : 400)
d_hidden = parse_arg_int(ARGS, "--hidden", arch == :resnet ? 128 : 256)
d_hidden2 = parse_arg_int(ARGS, "--hidden2", 128)
n_blocks = parse_arg_int(ARGS, "--n-blocks", 3)
seed = parse_arg_int(ARGS, "--seed", 1)
batch_size = parse_arg_int(ARGS, "--batch", 0)
η_init = parse_arg_float(ARGS, "--lr", 1e-3)
weight_decay = parse_arg_float(ARGS, "--weight-decay", 1e-5)
clip_norm = parse_arg_float(ARGS, "--clip-norm", 5.0)
out_path = parse_arg_string(ARGS, "--out", "")
rom_residual_arg = parse_arg_string(ARGS, "--rom-residual", "")
obs_only = "--obs-only" in ARGS
only_full_success = "--only-full-success" in ARGS
use_residual_weights = "--residual-weights" in ARGS
activation = Symbol(parse_arg_string(ARGS, "--activation", "silu"))

data = deserialize(dataset_path)
haskey(data, "X") || error("Dataset is missing key \"X\": $dataset_path")
haskey(data, "Y") || error("Dataset is missing key \"Y\": $dataset_path")
haskey(data, "meta") || error("Dataset is missing key \"meta\": $dataset_path")
X = data["X"]
Y = data["Y"]
meta = data["meta"]
X isa AbstractMatrix || error("Dataset X must be a matrix, got $(typeof(X)).")
Y isa AbstractMatrix || error("Dataset Y must be a matrix, got $(typeof(Y)).")
size(X, 2) == size(Y, 2) || error("Dataset sample count mismatch: size(X,2)=$(size(X,2)) vs size(Y,2)=$(size(Y,2)).")
size(X, 1) > 0 || error("Dataset X has zero rows.")
size(Y, 1) > 0 || error("Dataset Y has zero rows.")
size(X, 2) > 0 || error("Dataset has zero samples. Regenerate with more robust SEP settings.")
rom_mode = get(meta, "rom_mode", :baseline)

if only_full_success
    sample_full_success = get(data, "sample_full_success", nothing)
    sample_full_success === nothing && error("Dataset missing sample_full_success; regenerate with stable-prefix sampling.")
    keep = findall(sample_full_success)
    isempty(keep) && error("No fully successful samples found; check SEP settings or disable --only-full-success.")
    X = X[:, keep]
    Y = Y[:, keep]
    println("Filtering to fully successful samples: kept $(length(keep)) / $(length(sample_full_success))")
end

rom_order = 0
if rom_residual_arg != ""
    if lowercase(rom_residual_arg) in ("none", "false", "0")
        rom_order = 0
    else
        rom_order = parse(Int, rom_residual_arg)
    end
end

if rom_order != 0 && !(rom_order in (1, 2))
    error("Unsupported --rom-residual=$rom_residual_arg. Use 1 or 2, or omit for no residual.")
end

Y_rom = nothing
if rom_order == 1
    Y_rom = get(data, "Y_rom1", nothing)
elseif rom_order == 2
    Y_rom = get(data, "Y_rom2", nothing)
end
if rom_order != 0 && Y_rom === nothing
    error("Dataset missing Y_rom$(rom_order). Regenerate dataset with --rom-orders=$(rom_order).")
end
if Y_rom !== nothing
    Y_rom isa AbstractMatrix || error("Dataset Y_rom$(rom_order) must be a matrix, got $(typeof(Y_rom)).")
    size(Y_rom) == size(Y) || error("Y_rom$(rom_order) size mismatch: $(size(Y_rom)) vs Y $(size(Y)).")
end

if obs_only
    obs_idx = get(meta, "obs_idx", Int[])
    if isempty(obs_idx)
        obs_idx = collect(1:length(get(meta, "observables", Symbol[])))
    end
    isempty(obs_idx) && error("obs-only training requested but observables not found in dataset metadata.")
    d_obs = length(obs_idx)
    Y = Y[1:d_obs, :]
    if Y_rom !== nothing
        Y_rom = Y_rom[1:d_obs, :]
    end
end

finite_mask = [all(isfinite, view(X, :, j)) && all(isfinite, view(Y, :, j)) &&
               (Y_rom === nothing || all(isfinite, view(Y_rom, :, j))) for j in 1:size(X, 2)]
if !all(finite_mask)
    kept = count(identity, finite_mask)
    println("Dropping non-finite samples: kept $kept / $(length(finite_mask))")
    kept > 0 || error("All samples are non-finite after filtering.")
    keep = findall(finite_mask)
    X = X[:, keep]
    Y = Y[:, keep]
    if Y_rom !== nothing
        Y_rom = Y_rom[:, keep]
    end
end

Y_target = rom_order == 0 ? Y : (Y .- Y_rom)
all(isfinite, Y_target) || error("Training targets contain non-finite values after preprocessing.")

# RISK-1c: Compute inverse-residual sample weights for quality-aware training
sample_weights = nothing
if use_residual_weights
    sep_residuals = get(data, "sep_residuals", nothing)
    if sep_residuals === nothing
        println("Warning: --residual-weights requested but dataset has no sep_residuals. Training unweighted.")
    else
        # Filter/slice residuals to match current sample set
        if @isdefined(keep)
            sep_residuals = sep_residuals[keep]
        end
        valid_mask = isfinite.(sep_residuals) .& (sep_residuals .> 0)
        if count(valid_mask) > 0
            # Inverse-residual weighting: lower SEP error → higher weight
            # w_j = 1 / (residual_j + ε) then normalize so mean(w) = 1
            ε_floor = 1e-8
            raw_weights = [valid_mask[j] ? 1.0 / (sep_residuals[j] + ε_floor) : 1.0 for j in 1:length(sep_residuals)]
            raw_weights ./= Statistics.mean(raw_weights)
            # Clip extreme weights to avoid single-sample dominance
            clamp!(raw_weights, 0.1, 10.0)
            raw_weights ./= Statistics.mean(raw_weights)
            sample_weights = raw_weights
            n_valid = count(valid_mask)
            println("Residual weighting: $n_valid / $(length(sep_residuals)) samples with valid SEP residuals")
            println("  Weight range: $(round(minimum(sample_weights), sigdigits=3)) – $(round(maximum(sample_weights), sigdigits=3))")
        else
            println("Warning: no finite positive SEP residuals found. Training unweighted.")
        end
    end
end

if out_path == ""
    out_path = joinpath(dirname(dataset_path), "hlt_sep_surrogate_trained.jls")
end

println("Training surrogate")
println("Dataset: $dataset_path")
println("Architecture: $arch")
println("Activation: $activation")
println("X: $(size(X)), Y: $(size(Y))")
if rom_order != 0
    println("Training ROM residual surrogate (order=$rom_order, mode=$rom_mode)")
end
if obs_only
    println("Output target: observables only")
end
d_theta = length(get(meta, "theta_names", Symbol[]))
if arch == :resnet && d_theta == 0
    error("ResNet architecture requires d_theta > 0. Dataset metadata must include theta_names.")
end

Random.seed!(seed)
n_total = size(X, 2)
n_train = n_total == 1 ? 1 : clamp(Int(floor(0.9 * n_total)), 1, n_total - 1)
n_train < 2 && println("Warning: training with only $n_train sample(s); this is for smoke/debug only and may not generalize.")
perm = randperm(n_total)
train_idx = perm[1:n_train]
val_idx = perm[n_train + 1:end]

X_train = X[:, train_idx]
Y_train = Y_target[:, train_idx]
X_val = X[:, val_idx]
Y_val = Y_target[:, val_idx]
Y_val_full = Y[:, val_idx]
Y_rom_val = rom_order == 0 ? nothing : Y_rom[:, val_idx]
# RISK-1c: Slice sample weights to training set
train_weights = sample_weights !== nothing ? sample_weights[train_idx] : nothing

batch = batch_size <= 0 ? nothing : min(batch_size, n_train)

if arch == :resnet
    println("ResNet: d_hidden=$d_hidden, n_blocks=$n_blocks, d_theta=$d_theta")
    frozen = train_resnet!(
        X_train,
        Y_train;
        d_hidden = d_hidden,
        n_blocks = n_blocks,
        d_theta = d_theta,
        nepoch = nepoch,
        η_init = η_init,
        batch_size = batch,
        seed = seed,
        verbose = true,
        weight_decay = weight_decay,
        clip_norm = clip_norm,
    )
else
    println("MLP: d_hidden=$d_hidden, d_hidden2=$d_hidden2")
    frozen = train_mlp!(
        X_train,
        Y_train;
        d_hidden = d_hidden,
        d_hidden2 = d_hidden2,
        nepoch = nepoch,
        η_init = η_init,
        batch_size = batch,
        seed = seed,
        verbose = true,
        weight_decay = weight_decay,
        clip_norm = clip_norm,
        activation = activation,
        sample_weights = train_weights,
    )
end

val_rmse = nothing
val_rmse_resid = nothing
val_rmse_rom = nothing
val_improve = nothing
if isempty(val_idx)
    println("Validation skipped: dataset has a single sample after filtering.")
else
    Y_pred = zeros(size(Y_val))
    for i in 1:size(X_val, 2)
        Y_pred[:, i] = predict_frozen(frozen, X_val[:, i])
    end

    val_rmse_resid = vec(sqrt.(mean((Y_pred .- Y_val) .^ 2, dims = 2)))
    all(isfinite, val_rmse_resid) || error("Validation residual RMSE contains non-finite values.")
    if rom_order == 0
        val_rmse = val_rmse_resid
    else
        Y_pred_full = Y_pred .+ Y_rom_val
        val_rmse = vec(sqrt.(mean((Y_pred_full .- Y_val_full) .^ 2, dims = 2)))
    end
    all(isfinite, val_rmse) || error("Validation RMSE contains non-finite values.")

    println("Validation RMSE (per output dim):")
    println(val_rmse)
    if rom_order != 0
        val_rmse_rom = vec(sqrt.(mean((Y_rom_val .- Y_val_full) .^ 2, dims = 2)))
        val_improve = fill(NaN, length(val_rmse))
        nz = val_rmse_rom .> sqrt(eps(Float64))
        val_improve[nz] .= 1 .- (val_rmse[nz] ./ val_rmse_rom[nz])
        println("ROM baseline RMSE (per output dim):")
        println(val_rmse_rom)
        println("Residual improvement vs ROM (per output dim):")
        println(val_improve)
    end
end

meta_out = copy(meta)
meta_out["rom_residual"] = rom_order != 0
meta_out["rom_residual_order"] = rom_order
meta_out["rom_mode"] = rom_mode
meta_out["output_target"] = obs_only ? "observables" : "full"
meta_out["arch"] = String(arch)
if arch == :resnet
    meta_out["d_theta"] = d_theta
    meta_out["n_blocks"] = n_blocks
end
serialize(out_path, Dict(
    "frozen" => frozen,
    "meta" => meta_out,
    "validation_rmse" => val_rmse,
    "validation_rmse_residual" => val_rmse_resid,
    "validation_rmse_rom" => val_rmse_rom,
    "validation_improvement" => val_improve,
    "train_size" => n_train,
    "val_size" => length(val_idx),
))

println("Saved surrogate: $out_path")
