using Distributed
using ArgParse

function parse_ubermain()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--procs", "-p"
        help = "Number of parallel processes/workers to spawn."
        arg_type = Int
        default = 1

        "--runs", "-r"
        help = "Number of runs per experiment setting."
        arg_type = Int
        default = 5
    end
    return parse_args(s)
end

# parse number of procs, number of runs
ub_args = parse_ubermain()

addprocs(
    ub_args["procs"];
    exeflags = `--threads=$(Threads.nthreads()) --project=$(Base.active_project())`,
)

# make pkgs available in all processes
@everywhere using NeuralODETraining
@everywhere ENV["GKSwstype"] = "nul"

# list arguments here
args = NeuralODETraining.ArgVec([
    Argument("experiment", "LORENZ_FIGA6"),
    Argument("name", "one_basin_training"),
    Argument("path_to_data", "../data/LORENZ_TRAIN.npy"),
    Argument("model", "Chain"),
    Argument("hidden_dims", [[100, 100, 100]], "H"),
    Argument("nonlinearity", "relu"),
    Argument("epochs", 100_000),
    Argument("batch_size", 32),
    Argument("sequence_length", 30),
    Argument("sequence_time_delta", 0.01),
    Argument("start_lr", 1e-3),
    Argument("end_lr", 1e-5),
    Argument("solver", "Tsit5", "solver"),
    Argument("saving_interval", 100),
    Argument("plot_and_store_predictions", true),
])

# run experiments
ubermain(ub_args["runs"], args)