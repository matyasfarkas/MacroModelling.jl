function parse_arg_int(args::Vector{String}, key::String, default::Int)
    for arg in args
        if startswith(arg, key * "=")
            return parse(Int, split(arg, "=", limit = 2)[2])
        end
    end
    return default
end

function parse_arg_float(args::Vector{String}, key::String, default::Float64)
    for arg in args
        if startswith(arg, key * "=")
            return parse(Float64, split(arg, "=", limit = 2)[2])
        end
    end
    return default
end

function parse_arg_symbol(args::Vector{String}, key::String, default::Symbol)
    for arg in args
        if startswith(arg, key * "=")
            return Symbol(split(arg, "=", limit = 2)[2])
        end
    end
    return default
end

function parse_arg_string(args::Vector{String}, key::String, default::String)
    for arg in args
        if startswith(arg, key * "=")
            return String(split(arg, "=", limit = 2)[2])
        end
    end
    return default
end

function parse_arg_bool(args::Vector{String}, key::String, default::Bool)
    for arg in args
        if startswith(arg, key * "=")
            value = lowercase(split(arg, "=", limit = 2)[2])
            return value in ("1", "true", "yes", "y")
        end
    end
    return default
end

function first_positional_arg(args::Vector{String})
    for arg in args
        if !startswith(arg, "--")
            return arg
        end
    end
    return nothing
end
