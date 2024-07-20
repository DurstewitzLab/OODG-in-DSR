using ArgParse
using Lux
using Lux: Chain, Dense, glorot_uniform, cpu, gpu
using Optimisers

using ..Utilities

function initialize_model(args::AbstractDict, D::AbstractDataset; mod = @__MODULE__)
    # gather args
    N = size(D.X, 2)

    model_name = args["model"]
    hidden_dims = args["hidden_dims"]
    M = args["hidden_dim"]
    nonlinearity = @eval $(Symbol(args["nonlinearity"]))
    init = @eval $(Symbol(args["init_weight"]))

    # model type in correct module scope
    model_t = @eval mod $(Symbol(model_name))

    # specify model args based on model type
    if model_t <: CustomNeuralODEModel
        model_args = (N, M)
        model = Lux.transform(model_t(model_args...))
    else
        model = initialize_chain_model(N, hidden_dims, nonlinearity; init = init)
    end

    #println("Model / # Parameters: $(typeof(model)) / $(num_params(model))")
    return model
end

function initialize_solver(args; mod = @__MODULE__)
    # gather args
    solver_name = args["solver"]

    # solver type in correct module scope
    solver_t = @eval mod $(Symbol(solver_name))

    return solver_t()
end

function initialize_optimizer(args::Dict{String, Any})
    # optimizer chain
    ηₛ = args["start_lr"]::Float32
    #ηₑ = args["end_lr"]::Float32
    #E = args["epochs"]::Int
    #bpe = args["batches_per_epoch"]::Int

    # set gradient clipping
    #if κ > zero(κ)
    #    push!(opt_vec, Optimisers.ClipNorm(κ))
    #end

    # set SGD optimzier (ADAM, RADAM, etc)
    opt_sym = Symbol(args["optimizer"])
    opt = @eval $opt_sym($ηₛ)
    #push!(opt_vec, opt)

    # set exponential decay learning rate scheduler
    #γ = exp(log(ηₑ / ηₛ) / E)
    #decay = ExpDecay(1, γ, bpe, ηₑ, 1)
    #push!(opt_vec, decay)

    return opt#Optimisers.OptimiserChain(opt_vec...)
end

get_device(args::AbstractDict) = @eval $(Symbol(args["device"]))

"""
    argtable()

Prepare the argument table holding the information of all possible arguments
and correct datatypes.
"""
function argtable()
    settings = ArgParseSettings()
    defaults = load_defaults()

    @add_arg_table! settings begin
        # meta
        "--experiment"
        help = "The overall experiment name."
        arg_type = String
        default = defaults["experiment"] |> String

        "--name"
        help = "Name of a single experiment instance."
        arg_type = String
        default = defaults["name"] |> String

        "--run", "-r"
        help = "The run ID."
        arg_type = Int
        default = defaults["run"] |> Int

        "--saving_interval"
        help = "The interval at which scalar quantities are stored measured in epochs."
        arg_type = Int
        default = defaults["saving_interval"] |> Int

        # data
        "--path_to_data", "-d"
        help = "Path to dataset used for training."
        arg_type = String
        default = defaults["path_to_data"] |> String

        "--gaussian_noise_level"
        help = "Noise level of gaussian noise added to teacher signals."
        arg_type = Float32
        default = defaults["gaussian_noise_level"] |> Float32

        "--sequence_length", "-T"
        help = "Length of sequences sampled from the dataset during training."
        arg_type = Int
        default = defaults["sequence_length"] |> Int

        "--sequence_time_delta"
        help = "Integration time delta for training sequences."
        arg_type = Float32
        default = defaults["sequence_time_delta"] |> Float32

        "--batch_size", "-S"
        help = "The number of sequences to pack into one batch."
        arg_type = Int
        default = defaults["batch_size"] |> Int

        "--epochs", "-e"
        help = "The number of epochs to train for."
        arg_type = Int
        default = defaults["epochs"] |> Int

        "--gradient_clipping_norm"
        help = "The norm at which to clip gradients during training."
        arg_type = Float32
        default = defaults["gradient_clipping_norm"] |> Float32

        "--optimizer"
        help = "The optimizer to use for SGD optimization. Must be one provided by Flux.jl."
        arg_type = String
        default = defaults["optimizer"] |> String

        "--start_lr"
        help = "Learning rate passed to the optimizer at the beginning of training."
        arg_type = Float32
        default = defaults["start_lr"] |> Float32

        "--end_lr"
        help = "Target learning rate at the end of training due to exponential decay."
        arg_type = Float32
        default = defaults["end_lr"] |> Float32

        "--device"
        help = "Training device to use."
        arg_type = String
        default = defaults["device"] |> String

        "--BLAS_threads"
        help = "Number of threads to use for BLAS."
        arg_type = Int
        default = defaults["BLAS_threads"] |> Int

        # model
        "--model", "-m"
        help = "RNN to use."
        arg_type = String
        default = defaults["model"] |> String

        "--pretrained_path"
        help = "Path to pretrained model. Leave empty string to train from scratch."
        arg_type = String
        default = defaults["pretrained_path"] |> String

        "--hidden_dim"
        help = "hidden dimension for shallow PLRNN"
        arg_type = Int
        default = defaults["hidden_dim"] |> Int

        "--hidden_dims"
        help = "hidden dimension for shallow PLRNN"
        arg_type = Vector{Int}
        default = defaults["hidden_dims"] |> Vector{Int}

        "--nonlinearity"
        help = "Nonlinearity for Flux.Chain based N-ODE"
        arg_type = String
        default = defaults["nonlinearity"] |> String

        "--init_weight"
        help = "Initialization for N-ODE."
        arg_type = String
        default = defaults["init_weight"] |> String

        "--lat_model_regularization"
        help = "Regularization λ for latent model parameters."
        arg_type = Float32
        default = defaults["lat_model_regularization"] |> Float32

        "--solver"
        help = "ODE solver to use."
        arg_type = String
        default = defaults["solver"] |> String

        "--errtol"
        help = "Error tolerance for ODE solver."
        arg_type = Float32
        default = defaults["errtol"] |> Float32

        "--plot_and_store_predictions"
        help = "Whether to plot and store predictions during training."
        arg_type = Bool
        default = defaults["plot_and_store_predictions"] |> Bool

        # Metrics
        "--D_stsp_scaling"
        help = "GMM scaling parameter."
        arg_type = Float32
        default = defaults["D_stsp_scaling"] |> Float32

        "--D_stsp_bins"
        help = "Number of bins for D_stsp binning method."
        arg_type = Int
        default = defaults["D_stsp_bins"] |> Int

        "--PSE_smoothing"
        help = "Gaussian kernel smoothing σ for power spectrum smoothing."
        arg_type = Float32
        default = defaults["PSE_smoothing"] |> Float32

        "--PE_n"
        help = "n-step ahead prediction error."
        arg_type = Int
        default = defaults["PE_n"] |> Int
    end
    return settings
end

"""
    parse_commandline()

Parses all commandline arguments for execution of `main.jl`.
"""
parse_commandline() = parse_args(argtable())