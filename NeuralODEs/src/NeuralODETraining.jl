module NeuralODETraining
using Reexport

@reexport using DifferentialEquations, DiffEqFlux, Lux, ReconstructionMeasures

include("utilities/Utilities.jl")
@reexport using .Utilities

include("models/Models.jl")
@reexport using .Models

include("training_routines/TrainingRoutines.jl")
@reexport using .TrainingRoutines

# meta stuff
include("parsing.jl")
export parse_commandline,
    parse_ubermain,
    initialize_model,
    initialize_optimizer,
    initialize_solver,
    get_device,
    argtable,
    initialize_observation_model

include("multitasking.jl")
export Argument, prepare_tasks, main_routine, ubermain

#include("evaluation.jl")
#export evaluate_experiment

end
