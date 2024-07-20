module ReconstructionMeasures
using PrecompileTools

include("stsp_divergence.jl")
export state_space_divergence

include("pse.jl")
export normalized_and_smoothed_power_spectrum,
    power_spectrum_error, power_spectrum_correlation

include("wasserstein_based_error.jl")
export dimensionwise_WD,
    state_space_measure_WD,
    wasserstein_distance_1d,
    wasserstein_distance_1d_mc,
    sliced_WD,
    sliced_WD_measure

include("topological_error.jl")
export topological_error, criterion_lyaps, hausdorff_distance

include("utils.jl")
export laplace_smoothing

include("batched_distance_matrix.jl")
export batched_distance_matrix

# precompile f32 workloads
@setup_workload begin
    f32 = Float32
    X = randn(f32, 100, 2, 2)
    X_gen = randn(f32, 100, 2, 2)
    bins = 30
    scal = 1.0f0
    λs_true, λs_gen = randn(f32, 2, 2), randn(f32, 2, 2)
    T_limit = 80

    @compile_workload begin
        dstsp = state_space_divergence(X, X_gen, bins)
        dstsp_gmm = state_space_divergence(X, X_gen, scal)
        pse = power_spectrum_error(X, X_gen)
        d_stat = sliced_WD_measure(X, X_gen)
        d_top = topological_error(X, X_gen, λs_true, λs_gen, T_limit, 0.1f0)
    end
end

end
