include("reservoir_computer.jl")
using ReconstructionMeasures

data_ = permutedims(npzread("../data/DUFFING_TRAIN.npy"), (2, 3, 1))#[:, [1], :]
#data_ = permutedims(npzread("lorenz63_test.npy"), (2, 1))#[:, 1:10000]
data_train = data_ .+ 0.02f0 .* randn!(similar(data_))
N, S, T = size(data_)
#N, n, T = size(data_train)
#N, T = size(data_)
data = collect.(eachslice(data_train, dims=3))
train_data = data[1:end-1]
target_data = data[2:end]


# warmup time
N = 2
M = 500
α = 0.7f0
σ = 0.1f0
β = 0.1f0
ρ = 0.9f0
T_warmup = 100

# init
RC = ReservoirComputer(N, M, α, σ, β, ρ) |> gpu

#fit_rc!(RC, train_data[1:2] .|> gpu, target_data[1:2] .|> gpu; subsample_spacing = 1)
CUDA.@time fit_rc!(RC, train_data |> gpu, target_data |> gpu; subsample_spacing=1)

# warm up
generate_latent_ts(RC, train_data[1:T_warmup] |> gpu)

# freely predict
X̂ = generate(RC, RC.r, T - T_warmup, return_as_3d_array=true) |> cpu

# reshape
#X̂ = reduce(hcat, X̂)
X = data_[:, :, T_warmup+1:end]

plot(X[1, :, :]', X[2, :, :]', label=false, color="black")
plot!(X̂[1, :, :]', X̂[2, :, :]', label=false, color="red")

X_perm, X̂_perm = permutedims(X, (3, 1, 2)), permutedims(X̂, (3, 1, 2))
dstsp = state_space_divergence(X_perm, X̂_perm, 30)
pse = power_spectrum_error(X_perm, X̂_perm, 1.0f0)
