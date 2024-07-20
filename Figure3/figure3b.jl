using MKL, LinearAlgebra
BLAS.set_num_threads(1)
using NPZ,
    ReconstructionMeasures,
    Statistics,
    ColorSchemes,
    ThreadsX,
    LaTeXStrings,
    Plots,
    Measures,
    StatsBase

PWD = pwd()
RC_PATH = joinpath(PWD, "Figure3/RC")
RNN_PATH = joinpath(PWD, "Figure3/RNN")
NODE_PATH = joinpath(PWD, "Figure3/NODE")

RC_stat_errors = [
    npzread(joinpath(RC_PATH, "stat_errors_1-25.npy"))
    npzread(joinpath(RC_PATH, "stat_errors_26-50.npy"))
]
RC_top_errors = [
    npzread(joinpath(RC_PATH, "top_errors_1-25.npy"))
    npzread(joinpath(RC_PATH, "top_errors_26-50.npy"))
]
RNN_stat_errors = npzread(joinpath(RNN_PATH, "stat_errors.npy"))
RNN_top_errors = npzread(joinpath(RNN_PATH, "top_errors.npy"))
NODE_stat_errors = npzread(joinpath(NODE_PATH, "stat_errors.npy"))
NODE_top_errors = npzread(joinpath(NODE_PATH, "top_errors.npy"))

RC_stat_ECDF = ecdf(RC_stat_errors)
RC_top_ECDF = ecdf(RC_top_errors)
RNN_stat_ECDF = ecdf(RNN_stat_errors)
RNN_top_ECDF = ecdf(RNN_top_errors)
NODE_stat_ECDF = ecdf(NODE_stat_errors)
NODE_top_ECDF = ecdf(NODE_top_errors)

# find max_stat error
max_stat_error =
    maximum([maximum(RC_stat_errors), maximum(RNN_stat_errors), maximum(NODE_stat_errors)])

x_stat = 0:0.005:max_stat_error
y_stat_rc = RC_stat_ECDF.(x_stat)
y_stat_rnn = RNN_stat_ECDF.(x_stat)
y_stat_node = NODE_stat_ECDF.(x_stat)

x_top = 0:0.005:1
y_top_rc = RC_top_ECDF.(x_top)
y_top_rnn = RNN_top_ECDF.(x_top)
y_top_node = NODE_top_ECDF.(x_top)

c1 = ColorSchemes.starrynight[3]
c2 = ColorSchemes.starrynight[end]
#c3 = ColorSchemes.valentine[3]
c3 = ColorSchemes.fruitpunch[end]
fillalpha = 0.1

plot(
    x_stat,
    y_stat_rc,
    label="RC",
    linewidth=3,
    color=c1,
    framestyle=:box,
    grid=false,
    yticks=[0, 0.5, 1],
    xticks=[0.0, 0.5, 1.0],
    legendfontsize=12,
    xlabelfontsize=14,
    ylabelfontsize=14,
    xlabel=L"$\mathcal{E}_{\mathrm{stat}}$",
    ylabel="eCDF",
    dpi=1000,
    size=(500, 200),
    leftmargin=3mm,
    bottommargin=4mm,
)
plot!(
    x_stat,
    y_stat_rc,
    ribbon=(y_stat_rc, 0),
    fillalpha=fillalpha,
    label=false,
    linewidth=3,
    color=c1,
    alpha=0.000,
)
plot!(x_stat, y_stat_node, label="N-ODE", linewidth=3, color=c3)
plot!(
    x_stat,
    y_stat_node,
    ribbon=(y_stat_node, 0),
    fillalpha=fillalpha,
    label=false,
    linewidth=3,
    color=c3,
)
plot!(x_stat, y_stat_rnn, label="shPLRNN", linewidth=3, color=c2)
plot!(
    x_stat,
    y_stat_rnn,
    ribbon=(y_stat_rnn, 0),
    fillalpha=fillalpha,
    label=false,
    linewidth=3,
    color=c2,
    alpha=0.000,
    grid=false,
)
savefig("eCDFs_duffing_stat_error.pdf")

plot(
    x_top,
    y_top_rc,
    label=false,
    linewidth=3,
    color=c1,
    framestyle=:box,
    grid=false,
    yticks=[0, 0.5, 1],
    xticks=[0.0, 0.5, 1.0],
    legendfontsize=12,
    xlabelfontsize=14,
    ylabelfontsize=14,
    xlabel=L"$\mathcal{E}_{\mathrm{top}}$",
    ylabel="eCDF",
    dpi=1000,
    size=(500, 200),
    leftmargin=3mm,
    bottommargin=4mm,
)
plot!(
    x_top,
    y_top_rc,
    ribbon=(y_top_rc, 0),
    fillalpha=fillalpha,
    label=false,
    linewidth=3,
    color=c1,
    alpha=0.000,
)
plot!(x_top, y_top_node, label=false, linewidth=3, color=c3)
plot!(
    x_top,
    y_top_node,
    ribbon=(y_top_node, 0),
    fillalpha=fillalpha,
    label=false,
    linewidth=3,
    color=c3,
)
plot!(x_top, y_top_rnn, label=false, linewidth=3, color=c2)
plot!(
    x_top,
    y_top_rnn,
    ribbon=(y_top_rnn, 0),
    fillalpha=fillalpha,
    label=false,
    linewidth=3,
    color=c2,
    alpha=0.000,
    grid=false,
)
savefig("eCDFs_duffing_top_error.pdf")
