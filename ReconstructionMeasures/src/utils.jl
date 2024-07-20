laplace_smoothing(flat_hist::AbstractVector{T}, α::T = zero(T)) where {T} =
    (flat_hist .+ α) ./ (sum(flat_hist) .+ α .* length(flat_hist))

laplace_smoothing(flat_hist_batch::AbstractMatrix{T}, α::T = zero(T)) where {T} =
    (flat_hist_batch .+ α) ./
    (sum(flat_hist_batch, dims = 1) .+ α .* size(flat_hist_batch, 1))