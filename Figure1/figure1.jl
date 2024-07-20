using DifferentialEquations, MAT ,ColorSchemes
using CairoMakie
using NeuralODETraining
using    DiffEqFlux,
    Lux,
    Random,
    Plots,
    DifferentialEquations,
    NPZ,
    Statistics,
    LinearAlgebra
rng=Random.default_rng()

function TraningData(traj_length::Int64, init_cond::Vector{Vector{Float64}},tspan::Tuple,Vectorfield,params::Vector{Float64})
    
    t = Float32.(range(tspan[1], tspan[2], length = traj_length))
    X=Array{Float32}(undef,traj_length,length(init_cond),length(init_cond[1]))

    for i in 1:length(init_cond)
        true_prob = ODEProblem(Vectorfield, init_cond[i], tspan,params)
        X[:,i,:]= Float32.(Array(solve(true_prob, Vern9(), saveat = t)))'  #Tsit5()
    end
    return  permutedims(X, (1,3,2))
end

function dVdt3(u)
    I, C, gL, EL, gNa, ENa, VhNa, kNa, gK, EK, VhK, kK, tcK, gM, VhM, kM, tcM, gNMDA, Vsinf, hfix = 
    [0.0, 6.0, 8.0, -80.0, 20.0, 60.0, -20.0, 15.0, 10.0, -90.0, -25.0, 5.0, 1.0, 25.0, -15.0, 5.0, 35.0, 10.2, 0.0,0.06]

    minf = 1 / (1 + exp((VhNa - u[1]) / kNa))
    ninf = 1 / (1 + exp((VhK - u[1]) / kK))

    sinf = if Vsinf == 0
        1 / (1 + 0.33 * exp(-0.0625 * u[1]))
    else
        1 / (1 + 0.33 * exp(-0.0625 * Vsinf))
    end

    return ( (I - gL * (u[1] - EL) - gNa * minf * (u[1] - ENa) - gK * u[2] * (u[1] - EK) -
             gM * hfix * (u[1] - EK) - gNMDA * sinf * u[1]) / C), ((ninf - u[2]) / tcK)
    
end

function vdp_VF(du,u,p,t) #multi
    du[1]=3.0*u[2]
    du[2]=1.5*(1-u[1]^2)*u[2]-u[1]
    return nothing
end

function vdp(u) #multi
    return (3.0*u[2],1.5*(1-u[1]^2)*u[2]-u[1])
end


# define vector fields
f_neuron(u)=Point2f(dVdt3( u))
f_vdp(u)=Point2f(vdp( u))
params = [0.0, 6.0, 8.0, -80.0, 20.0, 60.0, -20.0, 15.0, 10.0, -90.0, -25.0, 5.0, 1.0, 25.0, -15.0, 5.0, 35.0, 10.2, 0.0,0.06]


#### plot left side
# plot 1.1 fig

cg4=cgrad(range(ColorSchemes.devon10[3], stop=ColorSchemes.devon10[3], length=100))
yel=colorant"rgb(252,226,5)"
datasize =10000
tspan = (0.0f0,13.0f0)
t = Float32.(range(tspan[1], tspan[2], length = datasize))
#norm mit 30
ini=[[-73.69,-0.5],[-67.88,-0.5]]
D_b=TraningData(datasize,ini,tspan,dVdt,params)


cg4=cgrad(range(ColorSchemes.devon10[1], stop=ColorSchemes.devon10[1], length=100))
fig= Figure()
ax2=Axis(fig[1, 1], width = 500, height = 300, xautolimitmargin = (0.01,0.01),yautolimitmargin = (0.01,0.025))

streamplot!(f_neuron, -80:0.01:10, -0.5:0.01:0.8, colormap = cg4,density=1000.8,linewidth=0.9,arrow_size=10.5,maxsteps=10000)
lines!(D_b[:,1,1],D_b[:,2,1],color=:black,linewidth=5, linestyle = Linestyle([0.1, 0.2, 3.2, 6.8]))
lines!(D_b[:,1,2],D_b[:,2,2],color=:black,linewidth=5,linestyle = Linestyle([0.1, 0.2, 3.2, 6.8]))
lines!(D_neuron[end-700:end,1,2],D_neuron[end-700:end,2,2],color=yel,linewidth=5)
CairoMakie.scatter!(D_neuron[end,1,1],D_neuron[end,2,1],color=ColorSchemes.Paired_12[10],markersize=14,strokewidth=0)
hidedecorations!(ax2)
    hidespines!(ax2)
CairoMakie.activate!(type="png",px_per_unit = 10.0)
fig
save("0103_1.png", fig,pt_per_unit = 100)



#### plot 1.2 fig
cg4=cgrad([:white,ColorSchemes.devon10[4],ColorSchemes.devon10[3]], [ 0.01,0.1])

fig= Figure(backgroundcolor = :transparent)
ax2=Axis(fig[1, 1], width = 500, height = 200, xautolimitmargin = (0.01,0.01),yautolimitmargin = (0.025,0.025),backgroundcolor = :transparent)
ini=[[-10,-0.5],[-80,0.5],[-70,-0.3]]
D=TraningData(datasize,ini,tspan,dVdt,params)
lines!(D[1:end,1,1],D[1:end,2,1],color=t,colormap=cg4,linewidth=7)
lines!(D[1:end,1,2],D[1:end,2,2],color=t,colormap=cg4,linewidth=7)
lines!(D[1:end,1,3],D[1:end,2,3],color=t,colormap=cg4,linewidth=7)
fig
hidedecorations!(ax2)
    hidespines!(ax2)
CairoMakie.scatter!(D[1,1,1],D[1,2,1],color=:white,markersize=5,strokewidth=8,strokecolor=ColorSchemes.devon10[3])
CairoMakie.scatter!(D[1,1,2],D[1,2,2],color=:white,markersize=5,strokewidth=8,strokecolor=ColorSchemes.devon10[3])
CairoMakie.scatter!(D[1,1,3],D[1,2,3],color=:white,markersize=5,strokewidth=8,strokecolor=ColorSchemes.devon10[3])
CairoMakie.scatter!(D_neuron[end,1,1],D_neuron[end,2,1],color=ColorSchemes.Paired_12[8],markersize=14,strokewidth=0)
CairoMakie.activate!(type="png",px_per_unit = 10.0)
fig
save("1802_train1.png", fig,pt_per_unit = 100)


#plot 1.3 fig 

cg4=cgrad(range(ColorSchemes.devon10[3], stop=ColorSchemes.devon10[3], length=100))
m1=-67.36355f0
s1=5.629799f0
m2=0.005419275f0
s2=0.053270325f0
x_range=(collect(-80:0.01:10).-m1)/s1
y_range=(collect(-0.5:0.01:0.8).-m2)/s2

model, p, st=load_model("Results/fig1_epoch250k/0702-errtol_0.001-data_neur_3init_fig1_0702.npy-H_[30, 30, 30]-seqL_50-solver_Tsit5/003",epoch=100000)

function node_neuron(u,model)
    x,y=Lux.apply(model,u, p, st)
    return (x[1],x[2])
end
NODE_neuron(u)=Point2f(node_neuron( u,model))


 fig= Figure()
 ax2=Axis(fig[1, 1], width = 500, height = 200, xautolimitmargin = (0.00,0.00),yautolimitmargin = (0.0,0.00))

streamplot!(NODE_neuron,x_range,y_range , colormap = cg4,density=1000.8,linewidth=0.9,arrow_size=10.5,maxsteps=10000)
D_b_norm=similar(D_b)
D_b_norm[:,1,:]=(D_b[:,1,:].-m1)/s1
D_b_norm[:,2,:]=(D_b[:,2,:].-m2)/s2
lines!(D_b_norm[:,1,1],D_b_norm[:,2,1],color=:black,linewidth=5, linestyle = Linestyle([0.1, 0.2, 3.2, 6.8]))
lines!(D_b_norm[:,1,2],D_b_norm[:,2,2],color=:black,linewidth=5,linestyle = Linestyle([0.1, 0.2, 3.2, 6.8]))
fig
CairoMakie.scatter!((D_neuron[end,1,1].-m1)/s1,(D_neuron[end,2,1].-m2)/s2,color=ColorSchemes.Paired_12[8],markersize=12,strokewidth=0)
fig
hidedecorations!(ax2)
    hidespines!(ax2)
fig

save("1802_VFgen.png", fig,pt_per_unit = 100)


#### fig 1 right side
# plot 2.1 fig
CairoMakie.activate!(type="png")
datasize =4000
tspan = (0.0f0,40.0f0)
t = Float32.(range(tspan[1], tspan[2], length = datasize))

ini=[[-10,-0.5],[-1,-0.3],[-1.8,1.2]]
D=TraningData(datasize,ini,tspan,vdp_VF,params)
cg4=cgrad(range(ColorSchemes.dracula[end], stop=ColorSchemes.dracula[end], length=100))

fig= Figure()
ax2=Axis(fig[1, 1], width = 500, height = 300,xautolimitmargin = (0.00,0.00),yautolimitmargin = (0.0,0.00))
streamplot!(f_vdp, -4:0.01:4, -2.5:0.01:2.5, colormap = cg4,density=1000.8,linewidth=0.9,arrow_size=10.5,maxsteps=10000)

lines!(D[2000:3900,1,2],D[2000:3900,2,2],color=yel,linewidth=5)
hidedecorations!(ax2)
    hidespines!(ax2)
fig
save("rebuttle_2.png", fig,pt_per_unit = 100)

#plot 2.2 fig
cg4=cgrad([:white,ColorSchemes.dracula[end]], [ 0.01,0.1])
fig= Figure(backgroundcolor = :transparent)
ax2=Axis(fig[1, 1], width = 500, height = 200, xautolimitmargin = (0.01,0.01),yautolimitmargin = (0.025,0.025),backgroundcolor = :transparent)
ini=[[-0.5,-2.5],[-2,2],[2.5,1.5]]

D=TraningData(datasize,ini,tspan,vdp_VF,params)
lines!(D[1:end,1,1],D[1:end,2,1],color=t,colormap=cg4,linewidth=7)
lines!(D[1:end,1,2],D[1:end,2,2],color=t,colormap=cg4,linewidth=7)
lines!(D[1:end,1,3],D[1:end,2,3],color=t,colormap=cg4,linewidth=7)
fig
hidedecorations!(ax2)
    hidespines!(ax2)
CairoMakie.scatter!(D[1,1,1],D[1,2,1],color=:white,markersize=5,strokewidth=8,strokecolor=ColorSchemes.dracula[end])
CairoMakie.scatter!(D[1,1,2],D[1,2,2],color=:white,markersize=5,strokewidth=8,strokecolor=ColorSchemes.dracula[end])
CairoMakie.scatter!(D[1,1,3],D[1,2,3],color=:white,markersize=5,strokewidth=8,strokecolor=ColorSchemes.dracula[end])
CairoMakie.activate!(type="png",px_per_unit = 10.0)
fig

save("1902_train2.png", fig,pt_per_unit = 100)


#plot 2.3 fig
model, p, st=load_model("Results/fig1_epoch250k/0702-errtol_0.001-data_vdP_fig1_0702.npy-H_[30, 30, 30]-seqL_50-solver_Tsit5/003",epoch=100000)

function node_neuron(u,model)
    x,y=Lux.apply(model,u, p, st)
    return (x[1],x[2])
end
NODE_neuron(u)=Point2f(node_neuron( u,model))


CairoMakie.activate!(type="png")
datasize =4000
tspan = (0.0f0,40.0f0)
t = Float32.(range(tspan[1], tspan[2], length = datasize))
#norm mit 30
ini=[[-10,-0.5],[-1,-0.3],[-10,-0.3]]
D=TraningData(datasize,ini,tspan,vdp_VF,params)
cg4=cgrad(range(ColorSchemes.dracula[end], stop=ColorSchemes.dracula[end], length=100))


fig= Figure()
ax2=Axis(fig[1, 1], width = 500, height = 200,xautolimitmargin = (0.00,0.00),yautolimitmargin = (0.0,0.00))
streamplot!(NODE_neuron, -4:0.01:4, -2.5:0.01:2.5, colormap = cg4,density=1000.8,linewidth=0.9,arrow_size=10.5,maxsteps=10000)
lines!(D[end-700:end,1,2],D[end-700:end,2,2],color=yel,linewidth=2)
fig

hidedecorations!(ax2)
    hidespines!(ax2)

    fig
save("1902_vdpgen_n2.png", fig,pt_per_unit = 100)

