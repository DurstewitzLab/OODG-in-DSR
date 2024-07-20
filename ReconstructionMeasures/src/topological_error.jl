using ThreadsX, Distances

function topological_error(
    X_GT::AbstractArray{T, 3},
    X_re::AbstractArray{T, 3},
    λs_true::AbstractMatrix{T},
    λs_re::AbstractMatrix{T},
    T_trans::Int,
    ε::T;
    rtol_λ_max::T = T(5e-1),
    use_gpu = false,
    batched_distance_matrix_kwargs...,
) where {T <: Real}
    # check that initial conditions are the same
    #@assert X_GT[1, :, :] == X_re[1, :, :]
    n = size(X_GT, 3)

    X_GT_slice = X_GT[T_trans+1:end, :, :]
    X_re_slice = X_re[T_trans+1:end, :, :]

    if use_gpu
        X_GT_slice = CuArray(X_GT_slice)
        X_re_slice = CuArray(X_re_slice)
    end

    # hausdorff distance
    X_dH = hausdorff_distance(X_GT_slice, X_re_slice; batched_distance_matrix_kwargs...)

    # limit set agreement
    X_dH_cond = X_dH .< ε

    # lyapunov spectra agreement
    λ_cond = criterion_lyaps(λs_true, λs_re; rtol_λ_max = rtol_λ_max)

    # error is 1 - ratio of correctly reconstructed init conds
    return 1 - sum(X_dH_cond .&& λ_cond) / n
end

function criterion_lyaps(
    λs_true::AbstractVector{T},
    λs_re::AbstractVector{T};
    rtol_λ_max::T = T(5e-1),
) where {T <: Real}
    # sort descending order
    λs_true_sorted = sort(λs_true, rev = true)
    λs_re_sorted = sort(λs_re, rev = true)

    # signs
    sign_criterion = all(sign.(λs_true_sorted) .== sign.(λs_re_sorted))

    # λ_max tolerance
    abs_rel_error =
        abs(λs_true_sorted[1] - λs_re_sorted[1]) / (abs(λs_true_sorted[1]) + eps(T))
    rtol_λ_max_criterion = abs_rel_error < rtol_λ_max

    # all signs must agree
    return sign_criterion && rtol_λ_max_criterion
end

criterion_lyaps(
    λs_true::AbstractMatrix{T},
    λs_re::AbstractMatrix{T};
    rtol_λ_max::T = T(5e-1),
) where {T <: Real} =
    criterion_lyaps.(eachcol(λs_true), eachcol(λs_re); rtol_λ_max = rtol_λ_max)

function hausdorff_distance(
    A::AbstractArray{T, 3},
    B::AbstractArray{T, 3};
    kwargs...,
) where {T <: Real}
    # compute pairwise distances
    D = batched_distance_matrix(A, B; kwargs...)
    daB = vec(maximum(minimum(D, dims = 2), dims = 1))
    dbA = vec(maximum(minimum(D, dims = 1), dims = 2))
    return max.(daB, dbA)
end