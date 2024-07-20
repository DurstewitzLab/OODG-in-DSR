using JLD2, Measurements, Statistics, ProgressMeter, BSON, NPZ
using ReconstructionMeasures
using Flux, CUDA, cuDNN
include("reservoir_computer.jl")
BLAS.set_num_threads(1)

PWD = pwd()

# read in RC results
JLD2.@load joinpath(PWD, "ReservoirComputing", "gs_results_duffing.jld2") gs_results
values(gs_results)

# sort by dstsp
sorted_results = sortperm(collect(values(gs_results)), by=x -> x.pse)
ks = collect(keys(gs_results))[sorted_results]
const BEST = ks[1]

# training data
raw_data = npzread(joinpath(PWD, "data", "DUFFING_TRAIN.npy"))
T, N, _ = size(raw_data)
data = collect.(eachslice(permutedims(raw_data, (2, 3, 1)), dims=3))
train_data = data[1:end-1] |> gpu
target_data = data[2:end] |> gpu
const T_WARMUP = 1
const N_MODELS = 1

# TEST DATA
data_test = npzread(joinpath(PWD, "data", "DUFFING_TEST_GRID.npy"))
data_test_rc = collect.(eachslice(permutedims(data_test, (2, 3, 1)), dims=3)) |> gpu
S = size(data_test, 3)

const PATH = joinpath(PWD, "ReservoirComputing", "results_duffing_rc")
mkpath(PATH)
mkpath(joinpath(PATH, "models"))
mkpath(joinpath(PATH, "trajectories"))
mkpath(joinpath(PATH, "lyap_spectra"))

# MEASURE SETTINGS
const ϵ_sup = 0.12f0
const ϵ_λ = 0.25f0

# true LYAP
const λ_true_left = npzread(joinpath(PWD, "data", "lyap_spectrum_duffing_left.npy")) .|> Float32
#const λ_true_right = npzread("lyap_spectrum_taylor_duffing_right.npy")
const λ_true = repeat(λ_true_left, 1, S)
const T_trans = 3000
const T_lyap = 3000
const ons = 1000

# error arrays
stat_errors = zeros(Float32, N_MODELS)
top_errors = zeros(Float32, N_MODELS)

####### MAIN LOOP
@showprogress "MODEL LOOP PROGRESS" for n = 1:N_MODELS
    # train
    RC = ReservoirComputer(N, BEST.M, BEST.α, BEST.σ, BEST.β, BEST.ρ) |> gpu
    fit_rc!(RC, train_data, target_data; subsample_spacing=1, T_warmup=T_WARMUP)

    # warm up
    R = generate_latent_ts(RC, data_test_rc[1:T_WARMUP])
    X̂_wup = [RC.Wout] .* R |> cpu
    # freely predict
    X̂ = generate(RC, RC.r, T - T_WARMUP) |> cpu
    X̂_cat = reshape(reduce(hcat, [X̂_wup; X̂]), N, S, T)
    X̂_cat = permutedims(X̂_cat, (3, 1, 2))

    # compute statistical error
    e_stat = sliced_WD_measure(data_test, X̂_cat, multithreaded=true)
    stat_errors[n] = e_stat

    # compute lyapunov spectrum
    λ_re = similar(RC.r) |> cpu
    X₁ = data_test_rc[1]
    @showprogress "LYAP PROGRESS model $n / $N_MODELS" for (i, x₁) ∈ enumerate(eachcol(X₁))
        λs = lyapunov_spectrum(RC, x₁, T_lyap; T_tr=T_trans, ons=ons) |> cpu
        λ_re[:, i] .= λs * 100
    end

    # top error
    e_top = topological_error(
        data_test,
        X̂_cat,
        λ_true,
        λ_re[1:N, :],
        T_trans,
        ϵ_sup;
        use_gpu=true,
        rtol_λ_max=ϵ_λ,
    )
    top_errors[n] = e_top

    # saving
    RC_cpu = RC |> cpu
    BSON.@save joinpath(PATH, "models", "RC_$n.bson") RC_cpu
    npzwrite(joinpath(PATH, "trajectories", "generated_$n.npy"), X̂_cat)
    npzwrite(joinpath(PATH, "lyap_spectra", "spectra_$n.npy"), λ_re)
end

# save error files
npzwrite(joinpath(PATH, "stat_errors.npy"), stat_errors)
npzwrite(joinpath(PATH, "top_errors.npy"), top_errors)