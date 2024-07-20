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

function duffing(du, u, p, t)
    du[1] = u[2]
    du[2] = -0.5 * u[2] - u[1] * (-1 + 0.1 * u[1]^2)
    return nothing
end

#provide observed time
datasize = 4000
tspan = (0.0f0, 40.0f0)
t = Float32.(range(tspan[1], tspan[2], length=datasize))

#Define DS
u0 = [1.0, 2.0]
p = [1.0]
true_prob = ODEProblem(duffing, u0, tspan, p)

x = range(-4.0, 4.0, 10)
y = range(-2.5, 2.5, 10)
N = Vector{Vector{Float64}}()

for i = 1:length(x)
    for j = 1:length(y)
        push!(N, [x[i], y[j]])
    end
end

D = generate_trajectories(datasize, N, tspan, duffing, p)

# standardize by training data
m = -0.10079668f0
s = 2.1933328f0
D_std = (D .- m) ./ s

# e_sup
standardized_x = (x .- m) ./ s
standardized_y = (y .- m) ./ s
l_x = sum(abs.(extrema(standardized_x)))
l_y = sum(abs.(extrema(standardized_y)))
V = l_x * l_y
ϵ_sup = V / (length(x) * length(y))

# plot
plot(D_std[:, 1, 1], D_std[:, 2, 1]; color=:grey, label="trajectory", α=0.2, xlabel="x₁", ylabel="x₂", linewidth=3)
plot!(D_std[:, 1, 2:end], D_std[:, 2, 2:end]; color=:grey, label=false, α=0.2, linewidth=3)
scatter!([D_std[1, 1, 1]], [D_std[1, 2, 1]], color=:red, label="initial condition")
scatter!(D_std[1, 1, 2:end], D_std[1, 2, 2:end], color=:red, label=false)

#npzwrite("data/DUFFING_TEST_GRID.npy", D_std)
