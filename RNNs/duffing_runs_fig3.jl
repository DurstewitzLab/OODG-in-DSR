using Distributed
using ArgParse

@everywhere using LinearAlgebra;
BLAS.set_num_threads(1);

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

# start workers in GTF env
addprocs(
    ub_args["procs"];
    exeflags = `--threads=$(Threads.nthreads()) --project=$(Base.active_project())`,
)

# make pkgs available in all processes
@everywhere using GTF
@everywhere ENV["GKSwstype"] = "nul"

"""
    ubermain(n_runs)

Start multiple parallel trainings, with optional grid search and
multiple runs per experiment.
"""
function ubermain(n_runs::Int, args::GTF.ArgVec)
    # load defaults with correct data types
    defaults = parse_args([], argtable())

    # prepare tasks
    tasks = prepare_tasks(defaults, args, n_runs)
    println(length(tasks))

    # run tasks
    pmap(main_routine, tasks)
end

# list arguments here
args = GTF.ArgVec([
    Argument("experiment", "DUFFING_FIG3"),
    Argument("name", "one_basin_training"),
    Argument("path_to_data", "../data/DUFFING_TRAIN.npy"),
    Argument("model", "clippedShallowPLRNN", "ℳ"),
    Argument("use_gtf", false),
    Argument("partial_forcing", true),
    Argument("observation_model", "Identity", "𝒪"),
    Argument("latent_dim", [5], "M"),
    Argument("teacher_forcing_interval", [15], "τ"),
    Argument("hidden_dim", [100], "H"),
    Argument("sequence_length", [100], "T̃"),
    Argument("batch_size", 32),
    Argument("scalar_saving_interval", 50),
    Argument("image_saving_interval", 50),
    Argument("epochs", 5000),
    Argument("start_lr", 1e-3),
    Argument("end_lr", 1e-6),
])

# run experiments
ubermain(ub_args["runs"], args)