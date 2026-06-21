#!/usr/bin/env julia
#
# Combine existing combined_v2 dataset with ZLB-binding dataset.
# The result is a single training file with ~64K samples covering both
# normal (shock_scale=0.1) and ZLB-binding (shock_scale=0.4) episodes.
#
# Usage:
#   julia --project=. scripts/combine_zlb_dataset.jl \
#     --base=<baseline_dataset.jls> \
#     --zlb=<zlb_dataset_or_checkpoint.jls> \
#     --out=<combined_dataset.jls>

using Serialization, Statistics, Dates

const REPO_ROOT = dirname(@__DIR__)

function parse_arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix) + 1:end]
    end
    return default
end

# ── Paths ──
combined_v2_path = parse_arg(ARGS, "--base", joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_dataset_combined_v2.jls"))
zlb_path = parse_arg(ARGS, "--zlb", joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls"))
out_path = parse_arg(ARGS, "--out", joinpath(REPO_ROOT,
    ".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_dataset_combined_with_zlb.jls"))

# ── Load combined_v2 ──
println("Loading combined_v2 dataset...")
d2 = deserialize(combined_v2_path)
X2 = d2["X"]
Y2 = d2["Y"]
Y2_rom1 = d2["Y_rom1"]
meta2 = d2["meta"]
N2 = size(X2, 2)
println("  combined_v2: X=$(size(X2)), Y=$(size(Y2)), Y_rom1=$(size(Y2_rom1)), N=$N2")

# ── Load ZLB checkpoint ──
println("Loading ZLB dataset...")
dz = deserialize(zlb_path)
cursor = haskey(dz, "cursor") ? dz["cursor"] : size(dz["X"], 2)
Xz = dz["X"][:, 1:cursor]
Yz = dz["Y"][:, 1:cursor]
Yz_rom1 = dz["Y_rom1"][:, 1:cursor]
Nz = cursor
println("  ZLB: X=$(size(Xz)), Y=$(size(Yz)), Y_rom1=$(size(Yz_rom1)), cursor=$cursor")

# ── Dimension check ──
@assert size(X2, 1) == size(Xz, 1) "X dimension mismatch: $(size(X2,1)) vs $(size(Xz,1))"
@assert size(Y2, 1) == size(Yz, 1) "Y dimension mismatch: $(size(Y2,1)) vs $(size(Yz,1))"
@assert size(Y2_rom1, 1) == size(Yz_rom1, 1) "Y_rom1 dimension mismatch"

# ── Filter non-finite samples from ZLB data ──
finite_mask = [all(isfinite, view(Xz, :, j)) && all(isfinite, view(Yz, :, j)) &&
               all(isfinite, view(Yz_rom1, :, j)) for j in 1:Nz]
n_finite = count(finite_mask)
if n_finite < Nz
    println("  Dropping $(Nz - n_finite) non-finite ZLB samples")
    keep = findall(finite_mask)
    Xz = Xz[:, keep]
    Yz = Yz[:, keep]
    Yz_rom1 = Yz_rom1[:, keep]
    Nz = n_finite
end

# ── Concatenate ──
println("\nCombining datasets...")
X_combined = hcat(X2, Xz)
Y_combined = hcat(Y2, Yz)
Y_rom1_combined = hcat(Y2_rom1, Yz_rom1)
N_total = size(X_combined, 2)
println("  Combined: X=$(size(X_combined)), Y=$(size(Y_combined)), N=$N_total")
println("  Breakdown: $N2 normal + $Nz ZLB-binding = $N_total total")

# ── Build combined sep_residuals if available ──
sep_residuals_combined = nothing
if haskey(dz, "sep_residuals")
    zlb_resids = dz["sep_residuals"][1:cursor]
    if n_finite < cursor
        zlb_resids = zlb_resids[findall(finite_mask)]
    end
    # combined_v2 may or may not have residuals; fill with zeros if missing
    if haskey(d2, "sep_residuals")
        base_resids = d2["sep_residuals"]
    else
        base_resids = zeros(N2)
    end
    sep_residuals_combined = vcat(base_resids, zlb_resids)
    println("  SEP residuals: combined_v2 median=$(round(median(base_resids), sigdigits=3)), " *
            "ZLB median=$(round(median(zlb_resids), sigdigits=3))")
end

# ── Build metadata ──
meta_combined = copy(meta2)
meta_combined["n_samples_base"] = N2
meta_combined["n_samples_zlb"] = Nz
meta_combined["n_samples_total"] = N_total
meta_combined["zlb_shock_scale"] = 0.4
meta_combined["base_shock_scale"] = get(meta2, "shock_scale", 0.1)
meta_combined["base_shock_scaling"] = get(meta2, "shock_scaling", "unknown")
if haskey(dz, "meta")
    meta_combined["zlb_shock_scale"] = get(dz["meta"], "shock_scale", meta_combined["zlb_shock_scale"])
    meta_combined["zlb_shock_scaling"] = get(dz["meta"], "shock_scaling", "unknown")
else
    meta_combined["zlb_shock_scaling"] = "unknown"
end
meta_combined["base_path"] = combined_v2_path
meta_combined["zlb_path"] = zlb_path
meta_combined["combined_date"] = string(Dates.now())

# ── Save ──
println("\nSaving to: $out_path")
result = Dict{String, Any}(
    "X" => X_combined,
    "Y" => Y_combined,
    "Y_rom1" => Y_rom1_combined,
    "meta" => meta_combined,
)
if sep_residuals_combined !== nothing
    result["sep_residuals"] = sep_residuals_combined
end
serialize(out_path, result)

filesize_mb = round(filesize(out_path) / 1e6, digits=1)
println("Done! File size: $(filesize_mb) MB")
println("\nTo train surrogate on combined data:")
println("  julia --project=. scripts/hlt_sep_surrogate_train.jl \\")
println("    $out_path \\")
println("    --rom-residual=1 --hidden=256 --hidden2=128 --epochs=400 --seed=1 \\")
println("    --out=.local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_with_zlb.jls")
