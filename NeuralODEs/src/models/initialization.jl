function uniform_init(shape::Tuple; eltype::Type{T} = Float32) where {T <: AbstractFloat}
    @assert length(shape) < 3
    din = eltype(shape[end])
    r = 1 / √din
    return uniform(shape, -r, r)
end

function general_OHL_init(N, hidden_dim; init = uniform_init)
    W₁ = Flux.glorot_uniform(N, hidden_dim)
    W₂ = Flux.glorot_uniform(hidden_dim, N)
    h₁ = zeros(Float32, N)
    h₂ = zeros(Float32, hidden_dim)
    return W₁, W₂, h₁, h₂
end