using Distributed
@everywhere using NeuralODETraining,
    DifferentialEquations, DiffEqFlux, NPZ, ThreadsX, ReconstructionMeasures, ProgressMeter
@everywhere using LinearAlgebra, SharedArrays
@everywhere using TaylorIntegration
@everywhere BLAS.set_num_threads(1)

@everywhere function node_ip!(du, u, p, t)
    node, θ, st = p
    du .= node(u, θ, st)[1]
    return nothing
end

@everywhere function lyapunov_spectrum(vf, ps, x1, T_int; kwargs...)
    tv, xv, λv = lyap_taylorinteg(
        vf,
        x1,
        0.0,
        T_int,
        10,
        Float64.(eps(Float32)),
        ps;
        maxsteps = 10000,
    )
    return λv[end, :]
end

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
const T_int_lyap = 60.0
#const ons = 1000

# MEASURE SETTINGS
const ϵ_sup = 0.12f0
const ϵ_λ = 0.25f0

const PATH = joinpath(PWD, "NeuralODEs", "results_duffing_node")
const MODEL_PATH = joinpath(
    PWD,
    "NeuralODEs",
    "Results/DUFFING_FIG3/one_basin_training-solver_Tsit5-H_[40, 40]",
)
mkpath(PATH)
mkpath(joinpath(PATH, "trajectories"))
mkpath(joinpath(PATH, "lyap_spectra"))
runs = readdir(MODEL_PATH)

# error arrays
stat_errors = SharedVector{Float32}((length(runs),))
top_errors = SharedVector{Float32}((length(runs),))

dummy, _, _ = load_model(joinpath(MODEL_PATH, "001"))
# neural ode from saved model
prob_neuralode = NeuralODE(
    dummy,
    (0.0f0, 40.0f0),
    Tsit5(),
    saveat = 0.01f0,
    reltol = 1.0f-4,
    abstol = 1.0f-4,
)

runs = readdir(MODEL_PATH)
@showprogress pmap(runs) do run
    r_int = parse(Int, run)
    _, p, st = load_model(joinpath(MODEL_PATH, run))

    # trajectory
    @views X_gen = neural_ode_forward_pass(prob_neuralode, x₁, p, st)[:, :, 1:end-1]
    X_gen = permutedims(X_gen, (3, 1, 2))

    # compute statistical error
    e_stat = sliced_WD_measure(data_test, X_gen, multithreaded = true)
    stat_errors[r_int] = e_stat

    # lyapunov spectrum
    ps_lyap = (dummy, Float64.(p), st)
    λs = zeros(Float32, N, S)
    ThreadsX.foreach(1:S) do i
        λs[:, i] =
            lyapunov_spectrum(node_ip!, ps_lyap, Float64.(x₁[:, i]), T_int_lyap) .|> Float32
    end

    # compute topological error
    e_top = topological_error(
        data_test,
        X_gen,
        λ_true,
        λs,
        T_trans,
        ϵ_sup;
        use_gpu = false,
        rtol_λ_max = ϵ_λ,
        multithreaded = true,
    )
    top_errors[r_int] = e_top

    npzwrite(joinpath(PATH, "trajectories", "generated_$r_int.npy"), X_gen)
    npzwrite(joinpath(PATH, "lyap_spectra", "spectra_$r_int.npy"), λs)
    return nothing #(run = run, e_stat = e_stat, e_top = e_top)
end

# save error files
npzwrite(joinpath(PATH, "stat_errors.npy"), stat_errors)
npzwrite(joinpath(PATH, "top_errors.npy"), top_errors)