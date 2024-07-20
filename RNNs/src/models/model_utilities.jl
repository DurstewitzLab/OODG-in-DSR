using ..PLRNNs
using ..ObservationModels

"""
    generate(model, z₁, T, [S,])

Generate a trajectory of length `T` using model `ℳ` given initial condition `z₁`.

Returns a `T × M` matrix of generated orbits in latent (model) space. If `S` is provided,
then `S` must be a `T' × N` matrix of external inputs with `T' ≥ T`.
"""
function generate(ℳ, z₁::AbstractVector, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))

    # initial condition for model
    Z[1, :] .= z₁

    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= ℳ(Z[t-1, :])
    end
    return Z
end

function generate(ℳ, z₁::AbstractVector, T::Int, S::AbstractMatrix)
    # trajectory placeholder
    Z = similar(z₁, T, length(z₁))

    # initial condition for model
    Z[1, :] .= z₁

    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :] .= ℳ(Z[t-1, :], S[t, :])
    end
    return Z
end

function generate(ℳ, z₁::AbstractMatrix, T::Int)
    # trajectory placeholder
    Z = similar(z₁, T, size(z₁)...)

    # initial condition for model
    Z[1, :, :] .= z₁

    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :, :] .= ℳ(Z[t-1, :, :])
    end
    return Z
end

function generate(ℳ, z₁::AbstractMatrix{T_}, T::Int, S::AbstractArray{T_,3}) where {T_}
    # trajectory placeholder
    Z = similar(z₁, T, size(z₁)...)

    # initial condition for model
    Z[1, :, :] .= z₁

    # evolve initial condition in time
    @views for t = 2:T
        Z[t, :, :] .= ℳ(Z[t-1, :, :], S[t, :, :])
    end
    return Z
end

"""
    generate(ℳ, 𝒪, x₁, T, [S,])

Generate a trajectory of length `T` using latent model `ℳ` and observation model
`𝒪` given initial condition `x₁` in observation space. Estimates latent state by inversion
of `𝒪` and evolves latent state using `ℳ`.

Returns a `T × M` matrix of generated orbits in observation space. If `S` is provided,
then `S` must be a `T' × N` matrix of external inputs with `T' ≥ T`.
"""
function generate(ℳ, 𝒪::ObservationModel, x₁::AbstractMatrix, T::Int)
    z₁ = init_state(𝒪, x₁)
    Z = generate(ℳ, z₁, T)
    X = 𝒪(permutedims(Z, (2, 3, 1)))
    return permutedims(X, (3, 1, 2))
end

function generate(ℳ, 𝒪::ObservationModel, x₁::AbstractVector, T::Int)
    z₁ = init_state(𝒪, x₁)
    Z = generate(ℳ, z₁, T)
    X = 𝒪(Z')
    return permutedims(X, (2, 1))
end

function generate(ℳ, 𝒪::ObservationModel, x₁::AbstractVector, T::Int, S::AbstractMatrix)
    z₁ = init_state(𝒪, x₁)
    Z = generate(ℳ, z₁, T, S)
    return permutedims(𝒪(Z'), (2, 1))
end

function generate(
    ℳ,
    𝒪::ObservationModel,
    x₁::AbstractMatrix{T_},
    T::Int,
    S::AbstractArray{T_,3},
) where {T_}
    z₁ = init_state(𝒪, x₁)
    Z = generate(ℳ, z₁, T, S)
    X = 𝒪(permutedims(Z, (2, 3, 1)))
    return permutedims(X, (3, 1, 2))
end

keep_connectivity_offdiagonal!(m, g) = nothing
keep_connectivity_offdiagonal!(m::Union{AbstractVanillaPLRNN,AbstractDendriticPLRNN}, g) =
    offdiagonal!(g[m.W])

"""
    lyapunov_spectrum(ℳ, z₁, T, Tₜᵣ, ons)

Compute the Lyapunov spectrum of the PLRNN model `ℳ` given initial condition `z₁`.
The system is first evolved for `Tₜᵣ` steps to reach the attractor,
and then the spectrum is computed across `T` steps. Reorthogonalize every `ons` steps.
"""
function lyapunov_spectrum(ℳ, z₁, T; T_tr=1000, ons=1)
    # evolve for transient time Tₜᵣ
    z = copy(z₁)
    for t = 1:T_tr
        z .= ℳ(z)
    end

    # initialize
    γ = zeros(Float32, length(z))

    # initialize as Identity matrix
    Q = Matrix{Float32}(I, length(z), length(z))

    for t = 1:T
        # evolve state
        z = ℳ(z)

        # compute jacobian
        Q = jacobian(ℳ, z) * Q

        if t % ons == 0
            # reorthogonalize
            Q, R = qr(Q)

            # accumulate lyapunov exponents
            γ += log.(abs.(diag(R)))
        end
    end
    return γ / T
end
