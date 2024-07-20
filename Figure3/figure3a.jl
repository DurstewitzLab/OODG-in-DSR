using NPZ, ReconstructionMeasures, Statistics, CairoMakie, ColorSchemes
fontsize_theme = Theme(fontsize=20)
set_theme!(fontsize_theme)

PWD = pwd()

X = npzread(joinpath(PWD, "data/DUFFING_FIG_3_TRAJS.npy"))

# GT
tr_indices = npzread(joinpath(PWD, "Figure3/B1_INDICES.npy"))
test_indices = setdiff(collect(axes(X, 3)), tr_indices)
X_train = X[:, :, tr_indices]
X_test = X[:, :, test_indices]

# RC
X_RC = npzread(joinpath(PWD, "Figure3/RC/rc_generated.npy"))[10:end, :, :]
X_RC_train = X_RC[:, :, tr_indices]
X_RC_test = X_RC[:, :, test_indices]

# NODE
X_node = npzread(joinpath(PWD, "Figure3/NODE/node_generated.npy"))
X_node_train = X_node[:, :, tr_indices]
X_node_test = X_node[:, :, test_indices]

# shallowPLRNN
X_sh = npzread(joinpath(PWD, "Figure3/RNN/shplrnn_generated.npy"))
X_sh_train = X_sh[:, :, tr_indices]
X_sh_test = X_sh[:, :, test_indices]

# global min/max in stsp
min_x, max_x = minimum(
    [vec(X[:, 1, :]); vec(X_RC[:, 1, :]); vec(X_node[:, 1, :]); vec(X_sh[:, 1, :])],
),
maximum([vec(X[:, 1, :]); vec(X_RC[:, 1, :]); vec(X_node[:, 1, :]); vec(X_sh[:, 1, :])])
min_y, max_y = minimum(
    [vec(X[:, 2, :]); vec(X_RC[:, 2, :]); vec(X_node[:, 2, :]); vec(X_sh[:, 2, :])],
),
maximum([vec(X[:, 2, :]); vec(X_RC[:, 2, :]); vec(X_node[:, 2, :]); vec(X_sh[:, 2, :])])
fig = Figure(size=(600, 200))

# UPPER LEFT
ax11 = Axis(fig[1, 1], xgridvisible=false, ygridvisible=false)
[
    lines!(ax11, Point2f.(eachrow(traj)), color=ColorSchemes.devon10[3], linewidth=2)
    for traj ∈ eachslice(X_train, dims=3)
]
[
    lines!(
        ax11,
        Point2f.(eachrow(traj)),
        color=(ColorSchemes.Greys[end-3], 0.6),
        linewidth=1,
        linestyle=:solid,
    ) for traj ∈ eachslice(X_test, dims=3)
]
scatter!(
    ax11,
    X[1, 1, tr_indices],
    X[1, 2, tr_indices],
    color=ColorSchemes.devon10[3],
    markersize=5,
)
[
    scatter!(ax11, x[1], x[2], color=(ColorSchemes.Greys[end-3], 0.6), markersize=5) for
    x ∈ eachcol(X[1, :, test_indices])
]
xlims!(ax11, min_x - 1.0f-1, max_x + 1.0f-1)
ylims!(ax11, min_y - 1.0f-1, max_y + 1.0f-1)
hidedecorations!(ax11)
hidespines!(ax11)
fig
save("DUFFING_GT.pdf", fig, pt_per_unit=2)

# UPPER RIGHT
#fig = Figure()
fig = Figure(size=(600, 200))
ax12 = Axis(fig[1, 1], xgridvisible=false, ygridvisible=false)
[
    lines!(ax12, Point2f.(eachrow(traj)), color=ColorSchemes.dracula[end], linewidth=2)
    for traj ∈ eachslice(X_RC_train, dims=3)
]
[
    lines!(
        ax12,
        Point2f.(eachrow(traj)),
        color=(ColorSchemes.Greys[end-3], 0.6),
        linewidth=1,
        linestyle=:solid,
    ) for traj ∈ eachslice(X_RC_test, dims=3)
]
scatter!(
    ax12,
    X[1, 1, tr_indices],
    X[1, 2, tr_indices],
    color=ColorSchemes.dracula[end],
    markersize=5,
)
[
    scatter!(ax12, x[1], x[2], color=(ColorSchemes.Greys[end-3], 0.6), markersize=5) for
    x ∈ eachcol(X[1, :, test_indices])
]
xlims!(ax12, min_x - 1.0f-1, max_x + 1.0f-1)
ylims!(ax12, min_y - 1.0f-1, max_y + 1.0f-1)
hidedecorations!(ax12)
hidespines!(ax12)
fig
save("DUFFING_RC.pdf", fig, pt_per_unit=2)

# LOWER LEFT
fig = Figure(size=(600, 200))
ax21 = Axis(fig[1, 1], xgridvisible=false, ygridvisible=false)
[
    lines!(ax21, Point2f.(eachrow(traj)), color=ColorSchemes.dracula[end], linewidth=2)
    for traj ∈ eachslice(X_node_train, dims=3)
]
[
    lines!(
        ax21,
        Point2f.(eachrow(traj)),
        color=(ColorSchemes.Greys[end-3], 0.6),
        linewidth=1,
        linestyle=:solid,
    ) for traj ∈ eachslice(X_node_test, dims=3)
]
scatter!(
    ax21,
    X[1, 1, tr_indices],
    X[1, 2, tr_indices],
    color=ColorSchemes.dracula[end],
    markersize=5,
)
[
    scatter!(ax21, x[1], x[2], color=(ColorSchemes.Greys[end-3], 0.6), markersize=5) for
    x ∈ eachcol(X[1, :, test_indices])
]
xlims!(ax21, min_x - 1.0f-1, max_x + 1.0f-1)
ylims!(ax21, min_y - 1.0f-1, max_y + 1.0f-1)
hidedecorations!(ax21)
hidespines!(ax21)
fig
save("DUFFING_NODE.pdf", fig, pt_per_unit=2)

# LOWER RIGHT
fig = Figure(size=(600, 200))
ax22 = Axis(fig[1, 1], xgridvisible=false, ygridvisible=false)
[
    lines!(ax22, Point2f.(eachrow(traj)), color=ColorSchemes.dracula[end], linewidth=2)
    for traj ∈ eachslice(X_sh_train, dims=3)
]
[
    lines!(
        ax22,
        Point2f.(eachrow(traj)),
        color=(ColorSchemes.Greys[end-3], 0.6),
        linewidth=1,
        linestyle=:solid,
    ) for traj ∈ eachslice(X_sh_test, dims=3)
]
scatter!(
    ax22,
    X[1, 1, tr_indices],
    X[1, 2, tr_indices],
    color=ColorSchemes.dracula[end],
    markersize=5,
)
[
    scatter!(ax22, x[1], x[2], color=(ColorSchemes.Greys[end-3], 0.6), markersize=5) for
    x ∈ eachcol(X[1, :, test_indices])
]
xlims!(ax22, min_x - 1.0f-1, max_x + 1.0f-1)
ylims!(ax22, min_y - 1.0f-1, max_y + 1.0f-1)
hidedecorations!(ax22)
hidespines!(ax22)
fig
save("DUFFING_SH.pdf", fig, pt_per_unit=2)