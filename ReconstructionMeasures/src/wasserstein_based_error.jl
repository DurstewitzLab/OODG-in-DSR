using Statistics
using StatsBase
using LinearAlgebra
using QuadGK
using ThreadsX

function wasserstein_distance_1d(
    x1::AbstractVector{T},
    x2::AbstractVector{T},
    p::Real = 1,
    δ::T = zero(T);
    quadgk_kwargs...,
) where {T <: Real}
    @assert 0 <= δ < 0.5

    !isfinite(sum(x1)) && return T(NaN)
    !isfinite(sum(x2)) && return T(NaN)

    # sort the samples
    x1_s = sort(x1)
    x2_s = sort(x2)

    # quantile functions
    F₁⁻¹(q) = quantile(x1_s, q, sorted = true)
    F₂⁻¹(q) = quantile(x2_s, q, sorted = true)

    # wasserstein distance integral
    𝒲ₚᵖ = quadgk(q -> abs(F₁⁻¹(q) - F₂⁻¹(q))^p, δ, 1 - δ; quadgk_kwargs...)[1] / (1 - 2δ)
    return 𝒲ₚᵖ^(one(T) / p)
end

function wasserstein_distance_1d_mc(
    x1::AbstractVector{T},
    x2::AbstractVector{T},
    p::Real = 1,
    δ::T = zero(T),
    samples::Int = 3000,
) where {T <: Real}
    @assert 0 <= δ < 0.5

    !isfinite(sum(x1)) && return T(NaN)
    !isfinite(sum(x2)) && return T(NaN)

    # quantile functions
    F₁⁻¹(q) = quantile(x1, q)
    F₂⁻¹(q) = quantile(x2, q)

    # probability samples
    qs = range(δ, one(T) - δ, length = samples)

    # wasserstein distance integral
    𝒲ₚᵖ = mean(abs.(F₁⁻¹(qs) .- F₂⁻¹(qs)) .^ p) / (1 - 2δ)
    return 𝒲ₚᵖ^(one(T) / p)
end

function dimensionwise_WD(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T},
    p::Real = 1,
    δ::T = zero(T);
    use_mc::Bool = true,
    mc_samples::Int = 3000,
    normalize_integral::Bool = true,
    quadgk_kwargs...,
) where {T <: Real}
    @assert size(X, 2) == size(X̃, 2)
    @assert 0 <= δ < 0.5

    # compute WD for each dimension
    if use_mc
        wds = wasserstein_distance_1d_mc.(eachcol(X), eachcol(X̃), p, δ, mc_samples)
    else
        wds = wasserstein_distance_1d.(eachcol(X), eachcol(X̃), p, δ; quadgk_kwargs...)
    end

    if normalize_integral
        # normalize each dimension by 2 / |x_max - x_min|
        normalization_factor =
            2 ./ abs.(vec(maximum(X, dims = 1)) .- vec(minimum(X, dims = 1)))
        wds .*= normalization_factor
    end

    return mean(wds)
end

function sliced_WD(
    X::AbstractMatrix{T},
    X̃::AbstractMatrix{T},
    p::Real = 1,
    δ::T = zero(T);
    n_projections::Int = 1000,
    mc_samples::Int = 3000,
    multithreaded = false,
) where {T <: Real}
    N = size(X, 2)
    @assert N == size(X̃, 2)
    @assert 0 <= δ < 0.5

    # projection vectors
    θ = randn(T, N, n_projections)
    θ ./= reduce(hcat, norm.(eachcol(θ)))

    # projected samples
    proj_X = X * θ
    proj_X̃ = X̃ * θ

    # multithreaded?
    _foreach = multithreaded ? ThreadsX.foreach : Base.foreach

    results = zeros(T, n_projections)
    _foreach(1:n_projections) do i
        @views results[i] =
            wasserstein_distance_1d_mc(proj_X[:, i], proj_X̃[:, i], p, δ, mc_samples)
    end

    return mean(results .^ p)^(one(T) / p)
end

"""
    sliced_WD_measure(
        X_true::AbstractArray{T, 3}, 
        X_gen::AbstractArray{T, 3}, 
        p::Real = 1, 
        δ::T = zero(T); 
        reduce_fn = mean, 
        multithreaded::Bool = false, 
        n_projections::Int = 1000, 
        mc_samples::Int = 1000
    ) where {T <: Real}

Calculate the Sliced Wasserstein Distance (SWD) between the true state space `X_true` and 
the generated state space `X_gen`. 

The SWD is computed for the empirical distributions given by each initial condition (3rd axis) and 
then reduced to a single number using the `reduce_fn` function.

# Arguments
- `X_true`: The true state space, a 3D array of size n x N x T, where n is the 
  number of samples/initial conditions, N is the state space dimensionality, and 
  T is the time dimension.
- `X_gen`: The generated state space, a 3D array of size n x N x T, where n is the 
  number of samples/initial conditions, N is the state space dimensionality, and 
  T is the time dimension.
- `p`: The order of the Wasserstein Distance. Default is 1.
- `δ`: Trimming factor to remedy WD's sensitivity to heavily tailed distributions. 
  Defaults to zero.
- `reduce_fn`: A function to reduce the array of WDs to a single number. Default is `mean`.
- `multithreaded`: A flag indicating whether to use multithreading. Default is false.
- `n_projections`: The number of projections to use in the sliced Wasserstein Distance. Default is 1000.
- `mc_samples`: The number of Monte Carlo samples if `use_mc` is true. Default is 1000.

# Returns
- The reduced Sliced Wasserstein Distance between `X_true` and `X_gen`.
"""
function sliced_WD_measure(
    X_true::AbstractArray{T, 3},
    X_gen::AbstractArray{T, 3},
    p::Real = 1,
    δ::T = zero(T);
    reduce_fn = mean,
    multithreaded::Bool = false,
    n_projections::Int = 1000,
    mc_samples::Int = 1000,
) where {T <: Real}
    @assert size(X_true, 2) == size(X_gen, 2)
    @assert size(X_true, 3) == size(X_gen, 3)
    wds =
        sliced_WD.(
            eachslice(X_true, dims = 3),
            eachslice(X_gen, dims = 3),
            p,
            δ;
            n_projections = n_projections,
            mc_samples = mc_samples,
            multithreaded = multithreaded,
        )
    return reduce_fn(wds)
end