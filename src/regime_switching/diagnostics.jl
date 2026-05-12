using Statistics

function _gate_segments(mask::AbstractVector{Bool})
    segs = Tuple{Int,Int}[]
    t = 1
    T = length(mask)
    while t <= T
        if mask[t]
            start = t
            while t <= T && mask[t]
                t += 1
            end
            push!(segs, (start, t - 1))
        else
            t += 1
        end
    end
    return segs
end

function contiguous_true_runs(mask::AbstractVector{Bool})
    segs = _gate_segments(mask)
    runs = UnitRange{Int}[]
    for (a, b) in segs
        push!(runs, a:b)
    end
    return runs
end

function choose_gated_run(runs::Vector{UnitRange{Int}}, strategy::Symbol)
    isempty(runs) && return nothing
    if strategy == :first
        return runs[1]
    elseif strategy == :last
        return runs[end]
    elseif strategy == :longest
        best_idx = 1
        best_len = length(runs[1])
        for i in 2:length(runs)
            len_i = length(runs[i])
            if len_i > best_len
                best_idx = i
                best_len = len_i
            end
        end
        return runs[best_idx]
    else
        error("Unsupported gated block strategy: $strategy")
    end
end

function select_gated_block_periods(gate_mask::AbstractVector{Bool},
                                    strategy::Symbol,
                                    context_periods::Int,
                                    max_eval_periods::Int)
    runs = contiguous_true_runs(gate_mask)
    isempty(runs) && return Int[], Int[], Int[], "No gated periods found."
    run = choose_gated_run(runs, strategy)
    eval_idx = collect(run)
    if max_eval_periods > 0 && length(eval_idx) > max_eval_periods
        eval_idx = eval_idx[1:max_eval_periods]
    end
    ctx_idx = Int[]
    if context_periods > 0 && !isempty(eval_idx)
        first_eval = first(eval_idx)
        if first_eval > 1
            ctx_start = max(1, first_eval - context_periods)
            ctx_idx = collect(ctx_start:(first_eval - 1))
        end
    end
    selected = vcat(ctx_idx, eval_idx)
    note = "Selected $(strategy) block $(first(run)):$(last(run))"
    !isempty(ctx_idx) && (note *= " with context $(first(ctx_idx)):$(last(ctx_idx))")
    return selected, eval_idx, ctx_idx, note
end

function compute_gate_stats(mask::AbstractVector{Bool})
    T = length(mask)
    segs = _gate_segments(mask)
    lens = [b - a + 1 for (a, b) in segs]
    return Dict(
        "periods_total" => T,
        "periods_nonlinear" => sum(mask),
        "periods_linear" => T - sum(mask),
        "share_nonlinear" => T == 0 ? 0.0 : sum(mask) / T,
        "episodes" => length(segs),
        "max_episode_len" => isempty(lens) ? 0 : maximum(lens),
        "min_episode_len" => isempty(lens) ? 0 : minimum(lens),
        "mean_episode_len" => isempty(lens) ? 0.0 : mean(lens),
    )
end

function episode_overlap(mask::AbstractVector{Bool}, window_start::Int, window_end::Int)
    window_end >= window_start || error("window_end must be >= window_start.")
    T = length(mask)
    lo = clamp(window_start, 1, max(T, 1))
    hi = clamp(window_end, 1, max(T, 1))
    if T == 0 || hi < lo
        return Dict(
            "window_start" => window_start,
            "window_end" => window_end,
            "window_periods" => 0,
            "nonlinear_in_window" => 0,
            "share_window_nonlinear" => 0.0,
            "share_nonlinear_inside_window" => 0.0,
        )
    end
    in_window = mask[lo:hi]
    nonlinear_in_window = sum(in_window)
    nonlinear_total = sum(mask)
    return Dict(
        "window_start" => lo,
        "window_end" => hi,
        "window_periods" => hi - lo + 1,
        "nonlinear_in_window" => nonlinear_in_window,
        "share_window_nonlinear" => nonlinear_in_window / (hi - lo + 1),
        "share_nonlinear_inside_window" => nonlinear_total == 0 ? 0.0 : nonlinear_in_window / nonlinear_total,
    )
end

function summarize_loglik_decomposition(ll_rom::AbstractVector,
                                        ll_fom::AbstractVector,
                                        mask::AbstractVector{Bool})
    length(ll_rom) == length(ll_fom) == length(mask) || error("Length mismatch in loglik decomposition inputs.")
    rom = Float64.(ll_rom)
    fom = Float64.(ll_fom)
    m = BitVector(mask)
    mixed = [m[t] ? fom[t] : rom[t] for t in eachindex(m)]
    return Dict(
        "ll_rom_total" => sum(rom),
        "ll_fom_total" => sum(fom),
        "ll_mixed_total" => sum(mixed),
        "ll_rom_linear_periods" => sum(rom[.!m]),
        "ll_rom_nonlinear_periods" => sum(rom[m]),
        "ll_fom_linear_periods" => sum(fom[.!m]),
        "ll_fom_nonlinear_periods" => sum(fom[m]),
        "periods_nonlinear" => sum(m),
        "periods_total" => length(m),
    )
end

function summarize_runtime(; runtime_switching_s::Union{Nothing,Real} = nothing,
                             runtime_fom_s::Union{Nothing,Real} = nothing)
    speedup = (runtime_switching_s === nothing || runtime_fom_s === nothing || runtime_switching_s <= 0) ?
        nothing : float(runtime_fom_s) / float(runtime_switching_s)
    return Dict(
        "runtime_switching_s" => runtime_switching_s === nothing ? nothing : float(runtime_switching_s),
        "runtime_fom_s" => runtime_fom_s === nothing ? nothing : float(runtime_fom_s),
        "speedup" => speedup,
    )
end

function chunk_stats(chain)
    info = get(chain.info, :internals, Dict{Symbol,Any}())
    acc = get(info, :avg_acceptance_rate, nothing)
    div = get(info, :count_divergences, nothing)
    step = get(info, :step_size, nothing)
    return acc, div, step
end

function run_chunked_sampling(total_samples::Integer,
                              chunk_size::Integer;
                              sample_chunk::Function,
                              concat_chunks::Function,
                              on_chunk::Union{Nothing,Function} = nothing)
    total_samples > 0 || error("total_samples must be positive, got $total_samples.")
    chunk_size > 0 || error("chunk_size must be positive, got $chunk_size.")
    n_chunks = cld(total_samples, chunk_size)
    samps = nothing
    start_time = time()
    for i in 1:n_chunks
        n_i = min(chunk_size, total_samples - (i - 1) * chunk_size)
        chunk = sample_chunk(n_i, i, n_chunks)
        samps = samps === nothing ? chunk : concat_chunks(samps, chunk)
        if on_chunk !== nothing
            on_chunk(i, n_chunks, n_i, chunk, samps, time() - start_time)
        end
    end
    return samps
end

function _chain_parameter_symbols(chain)
    try
        return Symbol.(names(chain, :parameters))
    catch
        return Symbol[]
    end
end

function _resolve_theta_symbols(chain, theta_syms::AbstractVector{Symbol})
    available = _chain_parameter_symbols(chain)
    isempty(available) && return collect(theta_syms)

    by_normalized_name = Dict(replace(string(sym), " " => "") => sym for sym in available)
    resolved = Symbol[]
    for (i, sym) in enumerate(theta_syms)
        candidates = (
            string(sym),
            "theta_vec[$i]",
            "theta_vec[$i,1]",
        )
        match_sym = nothing
        for candidate in candidates
            key = replace(candidate, " " => "")
            if haskey(by_normalized_name, key)
                match_sym = by_normalized_name[key]
                break
            end
        end
        if match_sym === nothing
            error("Could not resolve parameter symbol $sym in chain. Available parameters: $(available)")
        end
        push!(resolved, match_sym)
    end
    return resolved
end

function theta_draws(chain, theta_syms::AbstractVector{Symbol})
    resolved_syms = _resolve_theta_symbols(chain, theta_syms)
    arr = Array(chain[:, resolved_syms, :])
    return reshape(arr, :, length(theta_syms))
end

function epsilon_means_from_chain(chain; sample_idx::Union{Nothing,AbstractVector{Int}} = nothing)
    param_syms = names(chain, :parameters)
    eps_meta = Tuple{Symbol,Int,Int}[]
    for sym in param_syms
        name = replace(string(sym), " " => "")
        if occursin("ε[", name) || occursin("ϵ[", name)
            m = match(r"[εϵ]\[(\d+),(\d+)\]", name)
            if m !== nothing
                i = parse(Int, m.captures[1])
                t = parse(Int, m.captures[2])
                push!(eps_meta, (sym, i, t))
            end
        end
    end
    isempty(eps_meta) && return nothing

    max_i = maximum(x -> x[2], eps_meta)
    max_t = maximum(x -> x[3], eps_meta)
    sample_len = sample_idx === nothing ? max_t : length(sample_idx)
    if sample_idx !== nothing && max_t != sample_len
        println("Warning: epsilon index max_t=$max_t does not match sample_idx length=$sample_len.")
    end

    eps_mean = fill(NaN, max_i, sample_len)
    for (sym, i, t) in eps_meta
        vals = Array(chain[:, sym, :])
        if t <= size(eps_mean, 2)
            eps_mean[i, t] = mean(vec(vals))
        end
    end
    return eps_mean
end
