module Models

using Lux, LinearAlgebra

using ..Utilities

export NeuralODEModel,
    CustomNeuralODEModel,
    NODE_clippedReLU,
    NODE_tanh,
    NODE_ReLU,
    NODE_ELU,
    uniform_init,
    general_OHL_init

include("initialization.jl")
include("neural_ode_models.jl")
#include("model_utilities.jl")

end