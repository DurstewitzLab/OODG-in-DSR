using Lux,
    DiffEqFlux,
    DifferentialEquations,
    ComponentArrays,
    Optimisers,
    Random,
    Plots,
    LinearAlgebra,
    MKL,
    Zygote,
    Optimisers,
    JLD2

using Flux: mse

using ..Utilities

neural_ode_forward_pass(node_prob, z₁, ps, st) = Array(node_prob(z₁, ps, st)[1])

function compute_loss(X, node_prob, ps, st)
    z₁ = @view(X[:, :, 1])
    X̃ = neural_ode_forward_pass(node_prob, z₁, ps, st)
    loss = mse(X̃, @view(X[:, :, 2:end])) #+ 1.0f-5 * sum(abs2, p)
    return loss, X̃, ps, st
end

function training_callback(p, loss, X, Z, node_prob, st, settings)
    # print loss
    epoch = settings["epoch"]

    # add loss to history
    settings["losses"][epoch] = loss

    # save model
    if epoch % settings["saving_interval"] == 0
        # print loss
        println("Epoch $epoch | MSE $loss")

        # save model parameters
        model_filename =
            joinpath(settings["save_path"], "checkpoints", "epoch_$(epoch).jld2")

        save_model(p, st, model_filename)

        # plot current prediction against data
        if settings["plot_and_store_predictions"]
            fig = plot_predictions(X, Z; epoch = epoch)
            #display(fig)
            savefig(
                fig,
                joinpath(settings["save_path"], "plots", "prediction_$(epoch).png"),
            )
        end
    end

    # increment epoch counter
    settings["epoch"] += 1
    return false
end

function training_loop(prob_neuralode, D, opt, ps, st, settings)
    ps = ComponentArray(ps)
    st_opt = Optimisers.setup(opt, ps)

    # prep
    T, S = settings["sequence_length"], settings["batch_size"]
    σ_n = settings["gaussian_noise_level"]

    # learning rate scheduling
    ηₛ = settings["start_lr"]
    ηₑ = settings["end_lr"]
    E = settings["epochs"]

    γ = exp(log(ηₑ / ηₛ) / E)

    # save model at initialization
    model_filename = joinpath(settings["save_path"], "checkpoints", "epoch_0.jld2")
    save_model(ps, st, model_filename)

    for epoch = 1:E
        X = sample_batch(D, T, S)

        σ_n > zero(σ_n) ? add_gaussian_noise!(X, σ_n) : nothing

        gs, (loss, X̃, ps, st) = loss_gradient(X, prob_neuralode, ps, st)
        training_callback(ps, loss, X, X̃, prob_neuralode, st, settings)
        Optimisers.update!(st_opt, ps, gs)
        Optimisers.adjust!(st_opt, eta = ηₛ * γ^epoch)
    end

    return ps, st
end

function loss_gradient(X, node_prob, ps, st)
    (loss, ret...), back = pullback(p -> compute_loss(X, node_prob, p, st), ps)
    gs = back((one(loss), repeat([nothing], length(ret))...))[1]
    return gs, (loss, ret...)
end

function initialize_opt_prob(model, solver, settings::AbstractDict)
    errtol = settings["errtol"]
    T = settings["sequence_length"]
    Δt = settings["sequence_time_delta"]

    # sequence time is delta * sequence length
    tₛ = T * Δt

    #node = model
    rng = Random.default_rng()
    p, st = Lux.setup(rng, model)
    t_readout = collect(range(0.0f0, tₛ, length = T))
    prob_neuralode = NeuralODE(
        model,
        (0.0f0, tₛ),
        solver,
        saveat = t_readout,
        reltol = errtol,
        abstol = errtol,
    )
    return prob_neuralode, p, st
end