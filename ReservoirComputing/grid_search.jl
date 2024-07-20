using JLD2, Measurements, Statistics, ProgressMeter, ArgParse
using ReconstructionMeasures
using Flux, CUDA, cuDNN
include("reservoir_computer.jl")
BLAS.set_num_threads(1)
#CUDA.device!(5)

function grid_search(
    data,
    Ms,
    σs,
    βs,
    ρs,
    αs,
    λs;
    device=cpu,
    σ_n=0.0f0,
    T_warmup=100,
    runs_per_setting=1,
    fit_kwargs...,
)
    # data prep
    data_ = permutedims(data, (2, 3, 1))
    N, n, T = size(data_)
    add_gaussian_noise!(data_, σ_n)
    data_ = collect.(eachslice(data_, dims=3))
    train_data = data_[1:end-1] |> device
    target_data = data_[2:end] |> device

    # for measures
    data_measures = data[T_warmup+1:end, :, :]

    # grid search
    iterator = Iterators.product(Ms, σs, βs, ρs, αs, λs)
    results = Dict{NamedTuple,NamedTuple}()

    @showprogress "Grid searching RC hyperparameters..." for (M, σ, β, ρ, α, λ) ∈ iterator
        dstsps, pses = zeros(Float32, runs_per_setting), zeros(Float32, runs_per_setting)
        for run = 1:runs_per_setting
            RC = ReservoirComputer(N, M, α, σ, β, ρ) |> device
            #@show RC.Wout
            fit_rc!(RC, train_data, target_data, λ; fit_kwargs...)
            #@show RC.Wout
            # warm up
            @views generate_latent_ts(RC, train_data[1:T_warmup])
            # freely predict
            X̂ = generate(RC, RC.r, T - T_warmup, return_as_3d_array=true) |> cpu
            X̂_perm = permutedims(X̂, (3, 1, 2))

            # compute measures
            dstsp = state_space_divergence(data_measures, X̂_perm, 30)
            pse = power_spectrum_error(data_measures, X̂_perm, 1.0f0)

            # add
            dstsps[run] = dstsp
            pses[run] = pse
        end
        mean_dstsp, sem_dstsp = mean(dstsps), Float32(std(dstsps) / sqrt(runs_per_setting))
        mean_pse, sem_pse = mean(pses), Float32(std(pses) / sqrt(runs_per_setting))

        # add to results
        results[(M=M, σ=σ, β=β, ρ=ρ, α=α, λ=λ)] =
            (dstsp=mean_dstsp ± sem_dstsp, pse=mean_pse ± sem_pse)
        GC.gc()
    end
    return results
end

# data
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--data"
        help = "path to the data file"
    end

    return parse_args(s)
end

######################################################################################

args = parse_commandline()
dpath = args["data"]
name = dpath[1:findfirst(".npy", dpath)[1]-1]
data = npzread(dpath)
println(size(data))

# settings
M = [1000]
σ = collect(0.0f0:0.1f0:1.0f0)[2:end]
β = collect(0.0f0:0.1f0:1.0f0)
ρ = collect(0.1f0:0.1f0:1.0f0) #collect(0.1f0:0.1f0:1.1f0)
α = collect(0.0f0:0.1f0:1.0f0)[1:end-1]
λ = [0.0f0]

gs_results = grid_search(
    data,
    M,
    σ,
    β,
    ρ,
    α,
    λ,
    subsample_spacing=1,
    T_warmup=100,
    device=gpu,
    runs_per_setting=3,
    σ_n=0.01f0,
)

# save
JLD2.@save "gs_results_$name.jld2" gs_results