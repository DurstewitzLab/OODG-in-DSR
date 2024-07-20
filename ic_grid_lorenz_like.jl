using DifferentialEquations, NPZ, Statistics, ColorSchemes, Plots

function generate_trajectories(
    traj_length::Int64,
    init_cond::Vector{Vector{Float64}},
    tspan::Tuple,
    Vectorfield,
    params::Vector{Float64},
)
    t = Float32.(range(tspan[1], tspan[2], length=traj_length))
    X = Array{Float32}(undef, traj_length, length(init_cond), length(init_cond[1]))
    for i = 1:length(init_cond)
        true_prob = ODEProblem(Vectorfield, init_cond[i], tspan, params)
        X[:, i, :] = Float32.(Array(solve(true_prob, Vern9(), saveat=t)))'  #Tsit5()
    end
    return permutedims(X, (1, 3, 2))
end

function Lor(du, u, p, t)
    du[1] = -(p[1] * p[2]) / (p[1] + p[2]) * u[1] - u[2] * u[3] + p[3]
    du[2] = p[1] * u[2] + u[1] * u[3]
    du[3] = p[2] * u[3] + u[1] * u[2]
    return nothing
end

datasize = 16000
tspan = (0.0f0, 80.0f0) # Δt = 0.005
t = Float32.(range(tspan[1], tspan[2], length=datasize))
p = [-10.0, -4, 18.1]

ini1 = [
    [1, 1, 1],
    [3, 5, -1.0],
    [0.2, -0.1, 5],
    [0.2, -0.1, 5],
    [5, 5, 5],
    [2, 2, -0.1],
    [0.1, -0.1, 0.2],
    [1.1, -0.1, 0.1],
    [4, 2, 1],
    [-2, 2, 3],
    [2.1, 3.1, -0.01],
    [0.1, 0.02, 0.3],
    [0.1, 1.54, -0.2],
    [0, 5.0, 0],
    [-5, -5, 0],
]
ini2 = [
    [2, -0.5, 0],
    [0.2, -0.1, -5],
    [0.1, 2, -3],
    [5, -1, 0],
    [1.5, -1, -1.5],
    [0.2, 0.2, -0.2],
    [0.7, -0.7, -0.7],
    [0.1, 0.02, -0.3],
    [-5, 5, 0],
    [0, 0, -5],
    [5, 0.002, -0.3],
    [-1.1, -2.1, -2],
    [-3.2, 2, -1],
    [1, -1, -2],
    [-5, 4, -0.1],
]

D = generate_trajectories(datasize, [ini1; ini2], tspan, Lor, p)

plot(D[1:1:end, 1, :], D[1:1:end, 2, :], D[:, 3, :], label="", alpha=0.9)

m = mean(D)
s = std(D)

minimum(D[:, 1, :])
maximum(D[:, 1, :])

minimum(D[:, 2, :])
maximum(D[:, 2, :])

minimum(D[:, 3, :])
maximum(D[:, 3, :])

x = range(minimum(D[:, 1, :]), maximum(D[:, 1, :]), 5) .|> Float64
y = range(minimum(D[:, 2, :]), maximum(D[:, 2, :]), 5) .|> Float64
z = range(minimum(D[:, 3, :]), maximum(D[:, 3, :]), 5) .|> Float64

inis = vec(collect.(Iterators.product(x, y, z)))

D_full = generate_trajectories(datasize, inis, tspan, Lor, p)

plot(
    D_full[1:1:end, 1, :],
    D_full[1:1:end, 2, :],
    D_full[:, 3, :],
    label="",
    alpha=0.9,
    color=:black,
)
scatter!(
    D_full[1, 1, :],
    D_full[1, 2, :],
    D_full[1, 3, :],
    label="",
    alpha=0.9,
    color=:red,
)

D_std = (D_full .- m) ./ s
standardized_x = (x .- m) ./ s
standardized_y = (y .- m) ./ s
standardized_z = (z .- m) ./ s
l_x = sum(abs.(extrema(standardized_x)))
l_y = sum(abs.(extrema(standardized_y)))
l_z = sum(abs.(extrema(standardized_z)))
V = l_x * l_y * l_z
ϵ_sup = V / (length(x) * length(y) * length(z))

cam = (-30, round(atand(1 / √2); digits=3))
plot(
    D_std[:, 1, 1],
    D_std[:, 2, 1],
    D_std[:, 3, 1];
    color=:grey,
    label="trajectory",
    α=0.2,
    xlabel="x₁",
    ylabel="x₂",
    linewidth=2,
    camera=cam,
)
plot!(
    D_std[:, 1, 2:end],
    D_std[:, 2, 2:end],
    D_std[:, 3, 2:end];
    color=:grey,
    label=false,
    α=0.2,
    linewidth=2,
    camera=cam,
)
scatter!(
    [D_std[1, 1, 1]],
    [D_std[1, 2, 1]],
    [D_std[1, 3, 1]],
    color=:red,
    label="initial condition",
    camera=cam,
)
scatter!(
    D_std[1, 1, 2:end],
    D_std[1, 2, 2:end],
    D_std[1, 3, 2:end],
    color=:red,
    label=false,
    camera=cam,
)

npzwrite("data/LORENZ_TEST_DATA.npy", D_std)