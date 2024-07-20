"""
        generalization_error(
                X_true::AbstractArray{T, 3}, 
                X_gen::AbstractArray{T, 3}, 
                λs_true::AbstractVector{T}, 
                λs_gen::AbstractVector{T}, 
                εₛₜₐₜ::T, 
                εₜₒₚ::T; 
                sliced_WD_kwargs...
        ) where {T <: Real}

Calculate the generalization error as defined in the OODG paper. The generalization error 
is calculated given the true and generated trajectories `X_true` and `X_gen`, respectively, 
and their corresponding Lyapunov spectra `λs_true` and `λs_gen`, respectively. 

The tolerances `εₛₜₐₜ` and `εₜₒₚ` are the statistical and topological tolerances, respectively. 

The keyword arguments `sliced_WD_kwargs` are passed to the `sliced_WD_measure` function.

# Arguments
- `X_true`: The true state space, a 3D array of size T x N x n, where n is the 
    number of samples/initial conditions, N is the state space dimensionality, and 
    T is the time dimension.
- `X_gen`: The generated state space, a 3D array of size T x N x n, where n is the 
    number of samples/initial conditions, N is the state space dimensionality, and 
    T is the time dimension.
- `λs_true`: The Lyapunov spectrum of the true state space of shape (N, n).
- `λs_gen`: The Lyapunov spectrum of the generated state space of shape (N, n).
- `εₛₜₐₜ`: The statistical tolerance.
- `εₜₒₚ`: The topological tolerance.
- `sliced_WD_kwargs`: Additional keyword arguments for the `sliced_WD_measure` function.

# Returns
- The generalization error between `X_true` and `X_gen`.
"""
function generalization_error(
    X_true::AbstractArray{T, 3},
    X_gen::AbstractArray{T, 3},
    λs_true::AbstractArray{T, 2},
    λs_gen::AbstractArray{T, 2},
    εₛₜₐₜ::T,
    εₜₒₚ::T;
    return_errors::Bool = false,
    sliced_WD_kwargs...,
) where {T <: Real}
    # number of initial conditions
    n_ini = size(X_true, 3)

    # calculate topological error
    dₜₒₚ = topological_error(λs_true, λs_gen)

    # calcluate statistical error
    dₛₜₐₜ = sliced_WD_measure(X_true, X_gen; reduce_fn = identity, sliced_WD_kwargs...)

    # assess tolerances
    top_indicators = dₜₒₚ .< εₜₒₚ
    stat_indicators = dₛₜₐₜ .< εₛₜₐₜ

    gen_err = 1 - sum(top_indicators .&& stat_indicators) / n_ini
    return return_errors ? (gen_err, dₜₒₚ, dₛₜₐₜ) : gen_err
end