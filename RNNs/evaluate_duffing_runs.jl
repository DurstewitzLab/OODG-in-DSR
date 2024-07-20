using GTF, NPZ, ThreadsX, ReconstructionMeasures, ProgressMeter, LinearAlgebra
BLAS.set_num_threads(1)

PWD = pwd()

# TEST DATA
data_test = npzread(joinpath(PWD, "data", "DUFFING_TEST_GRID.npy"))
x₁ = data_test[1, :, :]
T, N, S = size(data_test)

# true LYAP
const λ_true_left =
    npzread(joinpath(PWD, "data", "lyap_spectrum_duffing_left.npy")) .|> Float32
#const λ_true_right = npzread("lyap_spectrum_taylor_duffing_right.npy")
const λ_true = repeat(λ_true_left, 1, S)
const T_trans = 3000
const T_lyap = 3000
const ons = 1000

# MEASURE SETTINGS
const ϵ_sup = 0.12f0
const ϵ_λ = 0.25f0

# trained models by running "duffing_runs_fig3.jl"
MODEL_PATH = joinpath(
    PWD,
    "RNNs/Results/DUFFING_FIG3/one_basin_training-ℳ_clippedShallowPLRNN-𝒪_Identity-M_5-τ_15-H_100-T̃_100",
)

PATH = "RNNs/results_duffing_rnn"
mkpath(PATH)
mkpath(joinpath(PATH, "trajectories"))
mkpath(joinpath(PATH, "lyap_spectra"))
runs = readdir(MODEL_PATH)

# error arrays
stat_errors = zeros(Float32, length(runs))
top_errors = zeros(Float32, length(runs))

for run in runs
    @info "Processing run $run"
    r_int = parse(Int, run)
    model_bson = joinpath(MODEL_PATH, run, "checkpoints/model_5000.bson")
    model, obs_model = load_model(model_bson)

    # trajectory
    X_gen = generate(model, obs_model, x₁, T)

    # compute statistical error
    e_stat = sliced_WD_measure(data_test, X_gen; multithreaded = true)
    stat_errors[r_int] = e_stat

    # lyapunov spectrum
    z₁ = init_state(obs_model, x₁)
    λs = ThreadsX.map(eachcol(z₁)) do z
        lyapunov_spectrum(model, z, T_lyap, T_tr = T_trans, ons = ons)
    end
    λs = reduce(hcat, λs) * 100

    # compute topological error
    e_top = topological_error(
        data_test,
        X_gen,
        λ_true,
        λs[1:N, :],
        T_trans,
        ϵ_sup;
        use_gpu = false,
        rtol_λ_max = ϵ_λ,
    )
    top_errors[r_int] = e_top

    npzwrite(joinpath(PATH, "trajectories", "generated_$r_int.npy"), X_gen)
    npzwrite(joinpath(PATH, "lyap_spectra", "spectra_$r_int.npy"), λs)
end

# save error files
npzwrite(joinpath(PATH, "stat_errors.npy"), stat_errors)
npzwrite(joinpath(PATH, "top_errors.npy"), top_errors)