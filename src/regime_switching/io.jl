import Serialization

const HLT_BENCHMARK_CHAIN_SUMMARY_KEYS = String[
    "theta_true",
    "synthetic_path",
    "regime_loglik_true",
    "regime_loglik_true_hard",
    "regime_loglik_true_hard_filtered",
    "loglik_post_mean",
    "loglik_post_mean_filtered",
    "post_mean_theta",
    "shock_filter",
    "linear_filter",
    "gate_share",
    "gate_mask",
]

function _as_dict(payload)
    payload isa AbstractDict || error("Expected serialized payload to be a Dict-like object, got $(typeof(payload)).")
    return Dict{Any,Any}(payload)
end

function _payload_haskey(payload::AbstractDict, key::AbstractString)
    return haskey(payload, key) || haskey(payload, Symbol(key))
end

function _payload_get(payload::AbstractDict, key::AbstractString, default = nothing)
    if haskey(payload, key)
        return payload[key]
    elseif haskey(payload, Symbol(key))
        return payload[Symbol(key)]
    else
        return default
    end
end

function _require_keys(payload::AbstractDict, keys_required::Vector{String}; label::String)
    missing = [k for k in keys_required if !haskey(payload, k)]
    isempty(missing) || error("$label is missing required keys: $(join(missing, ", ")).")
    return nothing
end

function validate_hlt_chain_payload(payload::AbstractDict;
                                    require_chain::Bool = false,
                                    required_keys::Vector{String} = String[],
                                    gate_mask_key::Union{Nothing,AbstractString} = nothing,
                                    require_gate_mask::Bool = false,
                                    sample_length::Union{Nothing,Integer} = nothing,
                                    label::AbstractString = "HLT chain payload")
    missing = String[]
    require_chain && !_payload_haskey(payload, "chain") && push!(missing, "chain")
    for k in required_keys
        _payload_haskey(payload, k) || push!(missing, k)
    end
    if require_gate_mask
        key = gate_mask_key === nothing ? "gate_mask" : String(gate_mask_key)
        _payload_haskey(payload, key) || push!(missing, key)
    end
    isempty(missing) || error("$label is missing required keys: $(join(unique(missing), ", ")).")

    if _payload_haskey(payload, "post_mean_theta")
        post_mean = _payload_get(payload, "post_mean_theta")
        post_mean isa AbstractVector || error("$label key post_mean_theta must be an AbstractVector, got $(typeof(post_mean)).")
        if eltype(post_mean) <: Number || all(x -> x isa Number, post_mean)
            all(isfinite, Float64.(post_mean)) || error("$label key post_mean_theta contains non-finite values.")
        end
    end

    if gate_mask_key !== nothing || require_gate_mask
        key = gate_mask_key === nothing ? "gate_mask" : String(gate_mask_key)
        if _payload_haskey(payload, key)
            mask_val = vec(_payload_get(payload, key))
            try
                mask = Bool.(mask_val)
                if sample_length !== nothing && length(mask) != sample_length
                    error("$label key $(key) length ($(length(mask))) does not match expected sample length $(sample_length).")
                end
            catch err
                msg = sprint(showerror, err)
                error("$label key $(key) could not be coerced to Bool vector: $msg")
            end
        end
    end

    return payload
end

function load_hlt_surrogate_bundle(path::AbstractString)
    isfile(path) || error("Surrogate bundle not found: $path")
    payload = _as_dict(Serialization.deserialize(path))
    _require_keys(payload, ["frozen"], label = "Surrogate bundle")
    meta = haskey(payload, "meta") ? Dict{Any,Any}(payload["meta"]) : Dict{Any,Any}()
    val_rmse = get(payload, "validation_rmse", nothing)
    return SurrogateBundle(String(path), payload["frozen"], meta, val_rmse, payload)
end

function load_hlt_synthetic_scenario(path::AbstractString)
    isfile(path) || error("Synthetic scenario not found: $path")
    payload = _as_dict(Serialization.deserialize(path))
    _require_keys(payload, ["obs_data", "s0", "shocks", "theta_true"], label = "HLT synthetic scenario")
    return payload
end

function load_hlt_gate_calibration(path::AbstractString)
    isfile(path) || error("HLT gate calibration payload not found: $path")
    payload = _as_dict(Serialization.deserialize(path))
    _require_keys(payload, ["tau_eps", "tau_y"], label = "HLT gate calibration payload")
    return payload
end

function load_hlt_chain_payload(path::AbstractString; allow_raw_chain::Bool = true, require_chain::Bool = true)
    isfile(path) || error("HLT chain payload not found: $path")
    raw = Serialization.deserialize(path)
    payload = if raw isa AbstractDict
        Dict{Any,Any}(raw)
    elseif allow_raw_chain
        Dict{Any,Any}("chain" => raw)
    else
        error("Expected HLT chain payload to be a Dict-like object, got $(typeof(raw)).")
    end
    if require_chain
        validate_hlt_chain_payload(payload; require_chain = true)
    end
    return payload
end

function load_hlt_chain_summary(path::AbstractString)
    isfile(path) || error("HLT chain summary payload not found: $path")
    return _as_dict(Serialization.deserialize(path))
end

function hlt_benchmark_chain_summary_path(chain_path::AbstractString)
    return replace(String(chain_path), r"\.jls$" => "_summary.jls")
end

function hlt_benchmark_chain_summary_keys()
    return copy(HLT_BENCHMARK_CHAIN_SUMMARY_KEYS)
end

function extract_hlt_benchmark_chain_summary(payload::AbstractDict;
                                             summary_keys::Vector{String} = hlt_benchmark_chain_summary_keys())
    out = Dict{Any,Any}()
    for k in summary_keys
        if haskey(payload, k)
            out[k] = payload[k]
        elseif haskey(payload, Symbol(k))
            out[k] = payload[Symbol(k)]
        end
    end
    return out
end

function load_hlt_chain_payload_for_benchmark(chain_path::AbstractString;
                                              prefer_summary::Bool = true,
                                              build_summary_cache::Bool = true,
                                              summary_path::Union{Nothing,AbstractString} = nothing,
                                              validate_summary::Bool = true,
                                              ensure_deserialize_modules!::Union{Nothing,Function} = nothing)
    summary = summary_path === nothing ? hlt_benchmark_chain_summary_path(chain_path) : String(summary_path)

    if prefer_summary && isfile(summary)
        payload = load_hlt_chain_summary(summary)
        if validate_summary
            validate_hlt_chain_payload(payload; label = "HLT benchmark chain summary payload")
        end
        return payload, summary, true
    end

    ensure_deserialize_modules! === nothing || ensure_deserialize_modules!()
    heavy_payload = load_hlt_chain_payload(chain_path)
    payload = extract_hlt_benchmark_chain_summary(heavy_payload)
    validate_hlt_chain_payload(payload; label = "HLT benchmark chain payload")

    if build_summary_cache
        try
            Serialization.serialize(summary, payload)
        catch err
            @warn "Failed to write benchmark chain summary cache" summary_path = summary error = sprint(showerror, err)
        end
    end

    return payload, summary, false
end

function load_hlt_dataset_payload(path::AbstractString)
    isfile(path) || error("HLT dataset payload not found: $path")
    payload = _as_dict(Serialization.deserialize(path))
    _require_keys(payload, ["X"], label = "HLT dataset payload")
    return payload
end

function load_hlt_fom_benchmark_payload(path::AbstractString)
    isfile(path) || error("HLT FOM benchmark payload not found: $path")
    payload = _as_dict(Serialization.deserialize(path))
    _require_keys(payload, ["results"], label = "HLT FOM benchmark payload")
    return payload
end

function build_chain_checkpoint_payload(chain;
                                        chunks_done::Integer,
                                        samples_done::Integer,
                                        timestamp::Real = time())
    chunks_done >= 0 || error("chunks_done must be nonnegative, got $chunks_done.")
    samples_done >= 0 || error("samples_done must be nonnegative, got $samples_done.")
    return Dict(
        "chain" => chain,
        "chunks_done" => Int(chunks_done),
        "samples_done" => Int(samples_done),
        "timestamp" => float(timestamp),
    )
end
