function hlt_model_file_and_symbol(model_name::AbstractString)
    name = String(model_name)
    if name == "Smets_Wouters_2007_HLT_obc"
        return "Smets_Wouters_2007_HLT_obc.jl", :Smets_Wouters_2007_HLT_obc
    elseif name == "Smets_Wouters_2007_HLT_zlb"
        return "Smets_Wouters_2007_HLT_zlb.jl", :Smets_Wouters_2007_HLT_zlb
    elseif name == "Smets_Wouters_2007_HLT"
        return "Smets_Wouters_2007_HLT.jl", :Smets_Wouters_2007_HLT
    else
        error("Unsupported HLT model name: $name")
    end
end

function ensure_hlt_model_loaded!(root::String, model_name::AbstractString; mod::Module = @__MODULE__)
    file_name, sym = hlt_model_file_and_symbol(model_name)
    if !isdefined(mod, sym)
        include(joinpath(root, "models", file_name))
    end
    return sym
end

function load_hlt_model(root::String, model_name::AbstractString; mod::Module = @__MODULE__)
    sym = ensure_hlt_model_loaded!(root, model_name; mod = mod)
    # invokelatest avoids world-age failures after dynamic model includes in long-running scripts/tests.
    return Base.invokelatest(getfield, mod, sym)
end

function load_hlt_linear_model(root::String; mod::Module = @__MODULE__)
    return load_hlt_model(root, "Smets_Wouters_2007_HLT"; mod = mod)
end
