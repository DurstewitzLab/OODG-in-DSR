module PLRNNs

using Flux, LinearAlgebra

using ..Utilities

export AbstractPLRNN,
    AbstractVanillaPLRNN,
    AbstractDendriticPLRNN,
    AbstractShallowPLRNN,
    AbstractDeepPLRNN,
    PLRNN,
    mcPLRNN,
    dendPLRNN,
    clippedDendPLRNN,
    FCDendPLRNN,
    shallowPLRNN,
    clippedShallowPLRNN,
    deepPLRNN,
    generate,
    lyapunov_spectrum,
    jacobian,
    uniform_init,
    norm_upper_bound,
    keep_connectivity_offdiagonal!,
    prediction_error

include("initialization.jl")
include("vanilla_plrnn.jl")
include("dendritic_plrnn.jl")
include("shallow_plrnn.jl")
include("deep_plrnn.jl")
include("model_utilities.jl")
include("prediction_error.jl")

end