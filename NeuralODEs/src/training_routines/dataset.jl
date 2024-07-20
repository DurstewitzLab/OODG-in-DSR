using NPZ
using Lux: cpu

abstract type AbstractDataset end

struct Dataset{A <: AbstractArray} <: AbstractDataset
    X::A
end

function load_dataset(path::String; device = cpu)
    X = Float32.(npzread(path)) |> device
    return Dataset(X)
end

function sample_sequence(D::Dataset, T̃::Int, j::Int)
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :, j]
end

function sample_batch(D::Dataset, T̃::Int, S::Int)
    T, N, n = size(D.X)
    Xs = similar(D.X, N, S, T̃ + 1)
    nⱼ = rand(1:n, S)
    Threads.@threads for i = 1:S
        @views Xs[:, i, :] .= sample_sequence(D, T̃, nⱼ[i])'
    end
    return Xs
end