using Flux

abstract type NeuralODEModel end
abstract type CustomNeuralODEModel <: NeuralODEModel end

#clipped shallow
mutable struct NODE_clippedReLU{M <: AbstractMatrix, V <: AbstractVector} <:
               CustomNeuralODEModel
    W₁::M
    W₂::M
    h₁::V
    h₂::V
end
Flux.@functor NODE_clippedReLU
Flux.trainable(ℳ::NODE_clippedReLU) = (W₁ = ℳ.W₁, W₂ = ℳ.W₂, h₁ = ℳ.h₁, h₂ = ℳ.h₂)

function NODE_clippedReLU(N, hidden_dim)
    W₁ = uniform_init((N, hidden_dim))
    W₂ = uniform_init((hidden_dim, N))
    h₁ = zeros(Float32, N)
    h₂ = uniform_init((hidden_dim,))
    return NODE_clippedReLU(W₁, W₂, h₁, h₂)
end

function (ℳ::NODE_clippedReLU)(z::AbstractVecOrMat{T}) where {T}
    W₁, W₂, h₁, h₂ = ℳ.W₁, ℳ.W₂, ℳ.h₁, ℳ.h₂
    W₂z = W₂ * z
    return W₁ * (relu.(W₂z .+ h₂) .- relu.(W₂z)) .+ h₁
end

#tanh
mutable struct NODE_tanh{M <: AbstractMatrix, V <: AbstractVector} <: CustomNeuralODEModel
    W₁::M
    W₂::M
    h₁::V
    h₂::V
end
Flux.@functor NODE_tanh
Flux.trainable(ℳ::NODE_tanh) = (W₁ = ℳ.W₁, W₂ = ℳ.W₂, h₁ = ℳ.h₁, h₂ = ℳ.h₂)

NODE_tanh(N, hidden_dim) = NODE_tanh(general_OHL_init(N, hidden_dim)...)

function (ℳ::NODE_tanh)(z::AbstractVecOrMat{T}) where {T}
    W₁, W₂, h₁, h₂ = ℳ.W₁, ℳ.W₂, ℳ.h₁, ℳ.h₂
    W₂z = W₂ * z
    return W₁ * tanh_fast.(W₂z .+ h₂) .+ h₁
end

#ReLU
mutable struct NODE_ReLU{M <: AbstractMatrix, V <: AbstractVector} <: CustomNeuralODEModel
    W₁::M
    W₂::M
    h₁::V
    h₂::V
end
Flux.@functor NODE_ReLU

NODE_ReLU(N, hidden_dim) = NODE_ReLU(general_OHL_init(N, hidden_dim)...)

function (ℳ::NODE_ReLU)(z::AbstractVecOrMat{T}) where {T}
    W₁, W₂, h₁, h₂ = ℳ.W₁, ℳ.W₂, ℳ.h₁, ℳ.h₂
    return W₁ * relu.(W₂ * z .+ h₂) .+ h₁
end

mutable struct NODE_ELU{M <: AbstractMatrix, V <: AbstractVector} <: CustomNeuralODEModel
    W₁::M
    W₂::M
    h₁::V
    h₂::V
end
Flux.@functor NODE_ELU

NODE_ELU(N, hidden_dim) = NODE_ELU(general_OHL_init(N, hidden_dim)...)

function (ℳ::NODE_ELU)(z::AbstractVecOrMat{T}) where {T}
    W₁, W₂, h₁, h₂ = ℳ.W₁, ℳ.W₂, ℳ.h₁, ℳ.h₂
    return W₁ * elu.(W₂ * z .+ h₂) .+ h₁
end