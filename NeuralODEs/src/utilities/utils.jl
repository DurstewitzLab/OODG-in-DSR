using JSON
using JLD2: @save, @load
using Lux: Chain, Dense, glorot_uniform, cpu, gpu

const RES = "Results"

"""
    create_folder_structure(exp::String, run::Int)

Creates basic saving structure for a single run/experiment.
"""
function create_folder_structure(exp::String, name::String, run::Int)::String
    # create folder
    path_to_run = joinpath(RES, exp, name, format_run_ID(run))
    mkpath(joinpath(path_to_run, "checkpoints"))
    mkpath(joinpath(path_to_run, "plots"))
    return path_to_run
end

function format_run_ID(run::Int)::String
    # only allow three digit numbers
    @assert run < 1000
    return string(run, pad = 3)
end

store_hypers(dict::Dict, path::String) =
    open(joinpath(path, "args.json"), "w") do f
        JSON.print(f, dict, 4)
    end

function convert_to_Float32(dict::Dict)
    for (key, val) in dict
        dict[key] = val isa AbstractFloat ? Float32(val) : val
    end
    return dict
end

load_defaults() = load_json_f32(joinpath(pwd(), "settings", "defaults.json"))

load_json_f32(path) = convert_to_Float32(JSON.parsefile(path))

save_model(p, st, path::String) = @save path p st

function load_model(path::String; epoch::Int = -1)
    latest = find_latest_model(path)
    # instantiate new model instance
    if epoch == -1
        @load latest p st
    else
        # load parameter and state
        @load joinpath(path, "checkpoints", "epoch_$(epoch).jld2") p st
    end

    settings = load_json_f32(joinpath(path, "args.json"))
    N = settings["N"]
    hidden_dims = settings["hidden_dims"]
    Φ = @eval $(Symbol(settings["nonlinearity"]))

    return initialize_chain_model(N, hidden_dims, Φ), p, st
end

function check_for_NaNs(θ)
    nan = false
    for p in θ
        nan = nan || !isfinite(sum(p))
    end
    return nan
end

function initialize_chain_model(N, hidden_dims, Φ; init = glorot_uniform)
    node = Chain(
        Dense(N, hidden_dims[1], Φ; init_weight = init),
        [
            Dense(hidden_dims[i], hidden_dims[i+1], Φ; init_weight = init) for
            i ∈ 1:length(hidden_dims)-1
        ]...,
        Dense(hidden_dims[end], N, init_weight = init),
    )
    return node
end

"""
    find_latest_model(run_path)

Search the folder given by `run_path` for the latest `model_[EPOCH].bson` and
return its path.
"""
function find_latest_model(run_path::String)::String
    files = filter(x -> endswith(x, ".jld2"), readdir(joinpath(run_path, "checkpoints")))
    n = length(files)
    ep_vec = Vector{Int}(undef, n)
    for i = 1:n
        ep_model = split(files[i], "_")[end]
        ep = parse(Int, split(ep_model, ".")[1])
        ep_vec[i] = ep
    end
    return joinpath(run_path, "checkpoints", files[argmax(ep_vec)])
end

replace_win_path(s::String) = replace(s, "\\" => "/")