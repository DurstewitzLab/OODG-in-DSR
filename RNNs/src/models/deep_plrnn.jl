using Flux: @functor

# abstract type
abstract type AbstractDeepPLRNN <: AbstractPLRNN end

function initLayers(M::Int, layers::AbstractVector)
    MLPlayers = []
    nₗ = length(layers)
    push!(MLPlayers, Dense(M, layers[1], relu))
    for i = 2:nₗ
        push!(MLPlayers, Dense(layers[i-1], layers[i], relu))
    end
    push!(MLPlayers, Dense(layers[nₗ], M))
    return Chain(MLPlayers...)
end

mutable struct deepPLRNN{V <: AbstractVector, CH, MY} <: AbstractDeepPLRNN
    A::V
    MLP::CH
    C::MY
end
@functor deepPLRNN

# initialization/constructor
function deepPLRNN(M::Int64, layers::AbstractVector)
    MLP = initLayers(M, layers)
    A, _, _ = initialize_A_W_h(M)
    return deepPLRNN(A, MLP, nothing)
end

function deepPLRNN(M::Int64, layers::AbstractVector, K::Int64)
    MLP = initLayers(M, layers)
    A, _, _ = initialize_A_W_h(M)
    C = Flux.glorot_uniform(M, K)
    return deepPLRNN(A, MLP, C)
end

step(m::deepPLRNN, z::AbstractVecOrMat) = m.A .* z .+ m.MLP(z)
