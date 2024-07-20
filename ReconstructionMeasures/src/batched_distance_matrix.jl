using CUDA

function pairwise_distance_batch_kernel(A, B, D)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if i <= size(D, 1) && j <= size(D, 2) && k <= size(D, 3)
        sum = 0.0f0
        for l ∈ axes(A, 2)
            diff = A[i, l, k] - B[j, l, k]
            sum += diff * diff
        end
        D[i, j, k] = sqrt(sum)
    end
    return nothing
end

function estimate_memory_consumption(
    A::AbstractArray{T, 3},
    B::AbstractArray{T, 3},
) where {T <: Real}
    element_size = sizeof(T)  # Size of one element in bytes

    # Size of the output array
    output_size = element_size * size(A, 1) * size(B, 1) * size(A, 3)

    meminfo = CUDA.MemoryInfo()

    # better be safe than sorry: understate max mem by 10%
    batch_size = floor(Int, meminfo.total_bytes / output_size * size(A, 3) * 0.9)

    if batch_size == 0
        error(
            "Not enough GPU memory available! Minimum memory needed: $(round(output_size / 1024^3, digits=1)) GB, \
            Device memory: $(round(meminfo.total_bytes / 1024^3, digits=1)) GB",
        )
    end

    # return size of output in MB, free memory in MB, batch size
    return output_size / 1024^2, meminfo.total_bytes / 1024^2, batch_size
end

function batched_distance_matrix(
    A::Array{T, 3},
    B::Array{T, 3};
    multithreaded = true,
) where {T <: Real}
    # multithread across initial conditions
    _foreach = multithreaded ? ThreadsX.foreach : Base.foreach

    n = size(A, 3)
    C = similar(A, size(A, 1), size(B, 1), n)
    _foreach(1:n) do i
        @views C[:, :, i] = pairwise(Euclidean(), A[:, :, i], B[:, :, i], dims = 1)
    end
    return C
end

function batched_distance_matrix(
    A::CuArray{T, 3},
    B::CuArray{T, 3};
    threads = (8, 8, 8),
) where {T <: Real}
    # estimate memory
    n = size(A, 3)
    out_sz_mb, free_sz_mb, max_batch_sz = estimate_memory_consumption(A, B)
    batch_sz = min(max_batch_sz, n)
    #@show max_batch_sz, batch_sz

    # chunk A and B into batches of size batch_sz, deal with remainder
    C = zeros(T, size(A, 1), size(B, 1), n)
    C_dev = similar(A, (size(A, 1), size(B, 1), batch_sz))

    # no need to chunk if batch_sz is equal to n (whole array fits in memory)
    if n == batch_sz
        @cuda threads = threads blocks = (
            ceil(Int, size(A, 1) / threads[1]),
            ceil(Int, size(B, 1) / threads[2]),
            ceil(Int, batch_sz / threads[3]),
        ) pairwise_distance_batch_kernel(A, B, C_dev)
        C .= Array(C_dev)
        return C
    end

    # batching
    n_batches = ceil(Int, n / batch_sz)
    remainder = n % batch_sz
    for i = 1:n_batches-1
        from = (i - 1) * batch_sz + 1
        to = min(i * batch_sz, n)
        #@show from, to
        @cuda threads = threads blocks = (
            ceil(Int, size(A, 1) / threads[1]),
            ceil(Int, size(B, 1) / threads[2]),
            ceil(Int, to - from + 1 / threads[3]),
        ) pairwise_distance_batch_kernel(A[:, :, from:to], B[:, :, from:to], C_dev)
        C[:, :, from:to] .= Array(C_dev)
    end
    #C_dev = nothing
    CUDA.unsafe_free!(C_dev)

    # handle the case where n is not divisible by batch_size
    if remainder != 0
        C_dev = similar(A, (size(A, 1), size(B, 1), remainder))
        from = n - remainder + 1
        to = n
        #@show from, to
        @cuda threads = threads blocks = (
            ceil(Int, size(A, 1) / threads[1]),
            ceil(Int, size(B, 1) / threads[2]),
            ceil(Int, to - from + 1 / threads[3]),
        ) pairwise_distance_batch_kernel(A[:, :, from:to], B[:, :, from:to], C_dev)
        C[:, :, from:to] .= Array(C_dev)
    end
    return C
end