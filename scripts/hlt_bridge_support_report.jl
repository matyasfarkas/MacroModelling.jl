#!/usr/bin/env julia

using Dates
using Printf
using Serialization

function parse_arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) && return arg[length(prefix) + 1:end]
    end
    return default
end

function first_positional(args::Vector{String})
    for arg in args
        startswith(arg, "--") || return arg
    end
    return nothing
end

function as_matrix(rows)
    rows === nothing && return Matrix{Float64}(undef, 0, 0)
    isempty(rows) && return Matrix{Float64}(undef, 0, 0)
    n = length(rows)
    p = length(rows[1])
    out = Matrix{Float64}(undef, n, p)
    for i in 1:n
        length(rows[i]) == p || error("theta_grid has ragged rows.")
        out[i, :] .= Float64.(rows[i])
    end
    return out
end

function bool_vector(x, n::Int)
    x === nothing && return fill(false, n)
    length(x) == n || error("Success vector length $(length(x)) does not match theta grid length $n.")
    return Bool.(x)
end

function residual_summary(residuals)
    residuals === nothing && return nothing
    finite = Float64[r for r in residuals if r isa Real && isfinite(Float64(r))]
    isempty(finite) && return nothing
    sort!(finite)
    mid = cld(length(finite), 2)
    median = isodd(length(finite)) ? finite[mid] : 0.5 * (finite[mid] + finite[mid + 1])
    return Dict(
        "count" => length(finite),
        "median" => median,
        "max" => maximum(finite),
        "min" => minimum(finite),
    )
end

function fmt(x)
    return @sprintf("%.8g", Float64(x))
end

function value_set(values)
    vals = sort(unique(Float64.(values)))
    return isempty(vals) ? "--" : join(fmt.(vals), ", ")
end

function write_report(path::String, payload_path::String, data)
    meta = get(data, "meta", Dict{String,Any}())
    theta_names = Symbol.(get(meta, "theta_names", Symbol[]))
    theta_grid = as_matrix(get(meta, "theta_grid", nothing))
    n = size(theta_grid, 1)
    p = size(theta_grid, 2)
    isempty(theta_names) && (theta_names = [Symbol("theta_$i") for i in 1:p])
    length(theta_names) == p || error("theta_names length $(length(theta_names)) does not match theta grid width $p.")

    success = bool_vector(get(meta, "theta_success", get(meta, "theta_full_success", nothing)), n)
    failure_periods = get(meta, "theta_failure_periods", fill(0, n))
    length(failure_periods) == n || (failure_periods = fill(-1, n))
    ok_idx = findall(success)
    fail_idx = findall(!, success)
    res = residual_summary(get(data, "sep_residuals", nothing))

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# HLT Bridge Support Report")
        println(io)
        println(io, "- Created: `$(Dates.now())`")
        println(io, "- Payload: `$(payload_path)`")
        println(io, "- Model: `$(get(meta, "model", "unknown"))`")
        println(io, "- Theta sampling: `$(get(meta, "theta_sampling", "unknown"))`")
        println(io, "- Grid points per axis: `$(get(meta, "grid_points", "unknown"))`")
        println(io, "- Parameter block: `$(join(String.(theta_names), ", "))`")
        println(io, "- Successful cells: `$(length(ok_idx)) / $n`")
        println(io, "- Failed cells: `$(length(fail_idx)) / $n`")
        println(io, "- SEP horizon/maxit/accept_tol: `$(get(meta, "sep_horizon", "unknown")) / $(get(meta, "sep_maxit", "unknown")) / $(get(meta, "sep_accept_tol", "unknown"))`")
        println(io, "- Shock scale: `$(get(meta, "shock_scale", "unknown"))`")
        if res !== nothing
            println(io, "- Finite residual count: `$(res["count"])`")
            println(io, "- Residual min/median/max: `$(fmt(res["min"])) / $(fmt(res["median"])) / $(fmt(res["max"]))`")
        end
        println(io)
        println(io, "## Feasible Support By Parameter")
        println(io)
        println(io, "| Parameter | Tested values | Successful values | Failed values | Successful range |")
        println(io, "|---|---:|---:|---:|---:|")
        for j in 1:p
            tested = theta_grid[:, j]
            ok_vals = isempty(ok_idx) ? Float64[] : theta_grid[ok_idx, j]
            fail_vals = isempty(fail_idx) ? Float64[] : theta_grid[fail_idx, j]
            range_txt = isempty(ok_vals) ? "--" : "$(fmt(minimum(ok_vals)))--$(fmt(maximum(ok_vals)))"
            println(io, "| `$(theta_names[j])` | $(value_set(tested)) | $(value_set(ok_vals)) | $(value_set(fail_vals)) | $(range_txt) |")
        end
        println(io)
        println(io, "## Failed Cells")
        println(io)
        if isempty(fail_idx)
            println(io, "No failed cells.")
        else
            println(io, "| Cell | Failure period | Theta |")
            println(io, "|---:|---:|---|")
            for i in fail_idx
                theta_txt = join(["$(theta_names[j])=$(fmt(theta_grid[i, j]))" for j in 1:p], ", ")
                println(io, "| $i | $(failure_periods[i]) | `$theta_txt` |")
            end
        end
        println(io)
        println(io, "## Interpretation")
        println(io)
        println(io, "This report maps numerical support for the direct SEP bridge objective. Successful cells are finite-support candidates for a reduced direct/surrogate posterior comparison. Failed cells should be excluded, reached by continuation, or treated as out-of-support for the bridge design.")
    end
end

payload_path = first_positional(ARGS)
payload_path === nothing && error("Usage: julia hlt_bridge_support_report.jl <dataset.jls> [--out=SUMMARY.md]")
data = deserialize(payload_path)
haskey(data, "meta") || error("Payload missing meta dictionary: $payload_path")
out_path = parse_arg(ARGS, "--out", joinpath(dirname(payload_path), "SUPPORT_REPORT.md"))
write_report(out_path, payload_path, data)
println("Wrote support report: $out_path")
