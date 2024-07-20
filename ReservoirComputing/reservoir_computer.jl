using MKL, LinearAlgebra, NPZ, Plots, Random
using Flux, CUDA, cuDNN, ForwardDiff

add_gaussian_noise!(data, σ) = data .+= σ .* randn!(similar(data))

mutable struct ReservoirComputer{M<:AbstractMatrix,V<:AbstractVector,T<:Real}
    W::M
    Win::M
    Wout::M
    α::T
    σ::T
    β::T
    b::V
    r::M
    Wtot::M
end
Flux.@functor ReservoirComputer

function ReservoirComputer(n, m, α, σ, β, ρ)
    W = randn(Float32, m, m)
    W = ρ * W / maximum(abs.(eigvals(W)))
    Win = σ * randn(Float32, m, n)
    Wout = zeros(Float32, n, m)
    b = β * randn(Float32, m)
    r = zeros(Float32, m, 1)
    return ReservoirComputer(W, Win, Wout, α, σ, β, b, r, W + Win * Wout)
end

function (RC::ReservoirComputer)(x::AbstractVecOrMat)
    r = rc_step(x, RC.r, RC.W, RC.Win, RC.b, RC.α)
    RC.r = r
    return r
end

function (RC::ReservoirComputer)()
    r = rc_step_auto(RC.r, RC.Wtot, RC.b, RC.α)
    x̂ = RC.Wout * r
    RC.r = r
    return x̂
end

update_Wtot!(RC::ReservoirComputer) = RC.Wtot .= RC.W + RC.Win * RC.Wout
reset!(RC::ReservoirComputer, r::AbstractMatrix) = RC.r .= r
reset!(RC::ReservoirComputer) = fill!(RC.r, 0)
reset!(RC::ReservoirComputer, n::Int) = RC.r = fill!(similar(RC.r, size(RC.r, 1), n), 0)

rc_step(
    x::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
    W::AbstractMatrix{T},
    Win::AbstractMatrix{T},
    b::AbstractVector{T},
    α::T,
) where {T} = α .* r + (1 - α) .* tanh_fast.(W * r .+ Win * x .+ b)

rc_step_auto(
    r::AbstractVecOrMat{T},
    Wtot::AbstractMatrix{T},
    b::AbstractVector{T},
    α::T,
) where {T} = α .* r + (1 - α) .* tanh_fast.(Wtot * r .+ b)

function generate_latent_ts(
    RC::ReservoirComputer,
    X::AbstractVector{<:AbstractVecOrMat},
    r₀::AbstractMatrix;
    return_as_3d_array::Bool=false,
)
    reset!(RC, r₀)
    R = [RC(x) for x ∈ X]
    if return_as_3d_array
        M, n = size(RC.r)
        return reshape(reduce(hcat, R), M, n, :)
    else
        return R
    end
end

function generate_latent_ts(RC::ReservoirComputer, X; kwargs...)
    n = size(X[1], 2)
    reset!(RC, n)
    return generate_latent_ts(RC, X, RC.r; kwargs...)
end

function generate(
    RC::ReservoirComputer,
    r₀::AbstractMatrix,
    T::Int;
    return_as_3d_array::Bool=false,
)
    reset!(RC, r₀)
    X = [RC() for t ∈ 1:T]
    if return_as_3d_array
        N, n = size(X[1])
        return reshape(reduce(hcat, X), N, n, :)
    else
        return X
    end
end

generate(RC::ReservoirComputer, T::Int; kwargs...) =
    generate(RC, fill!(similar(RC.r), 0), T; kwargs...)

function fit_rc!(
    RC::ReservoirComputer,
    train_data::AbstractVector{<:AbstractVecOrMat},
    target_data::AbstractVector{<:AbstractVecOrMat},
    λ::Real=0.0f0;
    subsample_spacing::Int=1,
    T_warmup::Int=0,
)
    n = size(train_data[1], 2)
    # init rc
    reset!(RC, n)

    # forward pass teacher forced
    @views rs = generate_latent_ts(RC, train_data)[T_warmup+1:end]
    @views target_data = target_data[T_warmup+1:end]

    if subsample_spacing > 1
        @views rs = rs[1:subsample_spacing:end]
        @views target_data = target_data[1:subsample_spacing:end]
    end

    # as matrix
    R = reduce(hcat, rs)
    X = reduce(hcat, target_data)
    mul!(RC.Wout, X, pinv(R))
    update_Wtot!(RC)
    return nothing
end

function jacobian_rc(
    r::AbstractVector{T},
    α::T,
    Wₜₒₜ::AbstractMatrix{T},
    b::AbstractVector{T},
) where {T<:Real}
    Diagonal(fill!(similar(r), α)) +
    (1 .- α) .* Diagonal(1 .- tanh_fast.(Wₜₒₜ * r + b) .^ 2) * Wₜₒₜ
end

function lyapunov_spectrum(RC::ReservoirComputer, x₁, T; T_tr=1000, ons=1)
    # evolve for transient time Tₜᵣ
    reset!(RC, 1)

    # update latent state
    RC(x₁)

    # transient
    [RC() for t ∈ 1:T_tr]

    # initialize
    γ = fill!(similar(RC.r), zero(eltype(x₁)))

    # initialize as Identity matrix
    ones_diag = Diagonal(fill!(similar(RC.r, length(RC.r)), one(eltype(RC.r))))
    zeros_full = fill!(similar(RC.r, length(RC.r), length(RC.r)), zero(eltype(RC.r)))
    Q = ones_diag + zeros_full

    for t = 1:T
        # evolve state
        RC()

        # compute jacobian
        Q = jacobian_rc(vec(RC.r), RC.α, RC.Wtot, RC.b) * Q

        if t % ons == 0
            # reorthogonalize
            Q, R = qr(Q)

            # accumulate lyapunov exponents
            γ += log.(abs.(diag(R)))
        end
    end
    return γ / T
end

######################################################################################