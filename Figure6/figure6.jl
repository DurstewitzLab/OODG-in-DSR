using NPZ
using Plots, StatsBase,ColorSchemes,LaTeXStrings



X1=npzread("Figure6/generalzing_models.npy")
X2=npzread("Figure6/not_generalizing_models.npy")


g1 = mean(X_total,dims=1)[1,:]
gcdf = ecdf(g1)
plot(x -> gcdf(x), 0, 0.001,lw=3,c=ColorSchemes.cool[end],label="generalizing models")
p=plot!([0.001, 0.002], [1, 1],lw=3,c=ColorSchemes.cool[end],label="", thickness_scaling=1.2,dpi=1000)
g2 = mean(X_total2,dims=1)[1,:]
gcdf = ecdf(g2)
pl=plot!(x -> gcdf(x), xlims=(0.0,0.002),lw=3,c=ColorSchemes.cool[1],yticks=[0,0.5,1.0],xlabel=L"r(\theta)",
grid=false,xticks=[0,0.005,0.0015],framestyle=:box,widen=true,size=(600,250),ylabel=L"eCDF",label="non-generalizing models",legend=:bottomright,dpi=1000)

savefig(pl,"ecdf_duff01.pdf")
