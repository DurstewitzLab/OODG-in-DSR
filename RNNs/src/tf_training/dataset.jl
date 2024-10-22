using NPZ

abstract type AbstractDataset end

struct Dataset{T, N, A <: AbstractArray{T, N}} <: AbstractDataset
    X::A
    name::String
end

function Dataset(path::String, name::String; device = cpu, dtype = Float32)
    X = npzread(path) .|> dtype |> device
    @assert ndims(X) ∈ (2, 3) "Data must be 2 or 3-dimensional but is $(ndims(X))-dimensional."
    if ndims(X) == 2
        X = reshape(X, size(X)..., 1)
    end
    return Dataset(X, name)
end

Dataset(path::String; device = cpu, dtype = Float32) =
    Dataset(path, ""; device = device, dtype = dtype)

struct ExternalInputsDataset{T, N, A <: AbstractArray{T, N}} <: AbstractDataset
    X::A
    S::A
    name::String
end

function ExternalInputsDataset(
    data_path::String,
    inputs_path::String,
    name::String;
    device = cpu,
    dtype = Float32,
)
    X = npzread(data_path) .|> dtype |> device
    S = npzread(inputs_path) .|> dtype |> device

    if ndims(X) == 2
        X = reshape(X, size(X)..., 1)
    end
    if ndims(S) == 2
        S = reshape(S, size(S)..., 1)
    end
    return ExternalInputsDataset(X, S, name)
end

ExternalInputsDataset(
    data_path::String,
    inputs_path::String;
    device = cpu,
    dtype = Float32,
) = ExternalInputsDataset(data_path, inputs_path, ""; device = device, dtype = dtype)

@inbounds """
    sample_sequence(dataset, sequence_length)

Sample a sequence of length `T̃` from a time series X.
"""
function sample_sequence(D::Dataset{T_, 3, A}, T̃::Int, j::Int) where {T_, A}
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return @views D.X[i:i+T̃, :, j]
end

function sample_sequence(D::ExternalInputsDataset{T_, 3, A}, T̃::Int, j::Int) where {T_, A}
    T = size(D.X, 1)
    i = rand(1:T-T̃-1)
    return D.X[i:i+T̃, :, j], D.S[i:i+T̃, :, j]
end

"""
    sample_batch(dataset, seq_len, batch_size)

Sample a batch of sequences of batch size `S` from time series X
(with replacement!).
"""

function sample_batch(D::Dataset{T_, 3, A}, T̃::Int, S::Int) where {T_, A}
    _, N, n = size(D.X)
    Xs = zeros(T_, N, S, T̃ + 1)
    for i = 1:S
        Xs[:, i, :] .= sample_sequence(D, T̃, rand(1:n))'
    end
    return Xs
end

function sample_batch(D::ExternalInputsDataset{T_, 3, A}, T̃::Int, S::Int) where {T_, A}
    N, K = size(D.X, 2), size(D.S, 2)
    n = size(D.X, 3)
    Xs = similar(D.X, N, S, T̃ + 1)
    Ss = similar(D.X, K, S, T̃ + 1)
    for i = 1:S
        X̃, S̃ = sample_sequence(D, T̃, rand(1:n))
        Xs[:, i, :] .= X̃'
        Ss[:, i, :] .= S̃'
    end
    return Xs, Ss
end
