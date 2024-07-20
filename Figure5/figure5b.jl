#The actual plot
using NPZ,CairoMakie,ColorSchemes,LaTeXStrings

X_1=log.(npzread("Figure5/Loss1.npy"))
X_2=log.(npzread("Figure5/Loss2.npy"))
a=-0.2:0.005:1.2

X_n=Vector{Float32}(undef,201)
for i in 0:200
    X_n[i+1]=X_1[41+i,241-i]
end


fig = Figure(size = (400, 400))
ax1=Axis3(fig[1, 1], azimuth = 4.9,elevation=0.38,aspect=(1.5,1.8,1.0))
surface!(a,a,X_1; shading = NoShading, colormap = my_cg)# colormap = :linear_wcmr_100_45_c42_n256)
lines!(zeros(201),a[41:241],X_1[41:241,41],color=:black,linewidth=4,linestyle = :dash)
lines!(a[41:241],zeros(201),X_1[41,41:241],color=:black,linewidth=4,linestyle = :dash)
lines!(a[41:241],a[241:-1:41],X_n,color=:black,linewidth=4,linestyle = :dash)
scatter!([a[241]],[a[41]],[X_1[241,41]],strokewidth=0,markersize=18,color=ColorSchemes.Paired_12[4])
scatter!([a[41]],[a[41]],[X_1[41,41]],strokewidth=0,markersize=18,color=ColorSchemes.Paired_12[4])
scatter!([a[41]],[a[241]],[X_1[41,241]],strokewidth=0,markersize=18,color=ColorSchemes.Paired_12[4])

hidedecorations!(ax1)  # hides ticks, grid and lables#hidespines!(ax1)
fig
save("loss1.png", fig,pt_per_unit = 20)



X_n=Vector{Float32}(undef,201)
for i in 0:200
    X_n[i+1]=X_2[41+i,241-i]
end


fig = Figure(size = (400, 400))
ax1=Axis3(fig[1, 1], azimuth = 4.9,elevation=0.38,aspect=(1.5,1.8,1.0))

surface!(a,a,X_2; shading = NoShading, colormap = my_cg)
lines!(zeros(201),a[41:241],X_2[41:241,41],color=:black,linewidth=4,linestyle = :dash)
lines!(a[41:241],zeros(201),X_2[41,41:241],color=:black,linewidth=4,linestyle = :dash)
lines!(a[41:241],a[241:-1:41],X_n,color=:black,linewidth=4,linestyle = :dash)
scatter!([a[241]],[a[41]],[X_2[241,41]],strokewidth=0,markersize=18,color=ColorSchemes.Paired_12[4])
scatter!([a[41]],[a[41]],[X_2[41,41]],markersize=18,strokewidth=0,color=ColorSchemes.Paired_12[4])
scatter!([a[41]],[a[241]],[X_2[41,241]],markersize=18,strokewidth=0,color=ColorSchemes.Paired_12[4])

hidedecorations!(ax1)  # hides ticks, grid and lables#hidespines!(ax1)
fig
save("loss2.png", fig,pt_per_unit = 20)

#How to creat a similar looking plot

using Flux
using Plots
using Statistics
using BSON
using LinearAlgebra
using GTF,ColorSchemes,NPZ

function CalcX(model::clippedShallowPLRNN)
    A=model.A
    la=length(A)
    W1=model.W₁
    lW1_1,lW1_2=size(W1)
    W2=model.W₂
    lW2_1,lW2_2=size(W1)
    b1=model.h₁
    lb1=length(b1)
    b2=model.h₂
    lb1=length(b1)
    W1_vec=reshape(W1,(lW1_1*lW1_2))
    W2_vec=reshape(W2,(lW2_1*lW2_2))

    x=Vector{Float32}[]
    push!(x,A,W1_vec,b1,W2_vec,b2)
    return x
end



function ConvToModel(x::Vector{Vector{Float32}},model::clippedShallowPLRNN)
    lW1_1,lW1_2=size(model.W₁)
    lW2_1,lW2_2=size(model.W₂)
    model.A .= x[1]
    model.W₁ .= reshape(x[2],(lW1_1,lW1_2))
    model.h₁ .= x[3]
    model.W₂ .= reshape(x[4],(lW2_1,lW2_2))
    model.h₂ .= x[5]
    return model
end


function Loss3D_TF(alpha::Float64,a::StepRangeLen,b,model1,model2,model3,model,Y,T,init_cond)
    b=a
    surf=Matrix{Vector{Vector{Float32}}}(undef,length(a),length(b))
    for i in 1:length(a)
        for j in 1:length(b)
        surf[i,j]=Float32(a[i])*(CalcX(model3) - CalcX(model1)) +Float32(b[j])*(CalcX(model2) - CalcX(model1))+ CalcX(model1)
        end
    end
    Loss3D=Matrix{Float32}(undef,length(a),length(b))
    for i in 1:length(a)
        for j in 1:length(b)
            Loss3D[i,j]=Flux.Losses.mse((1-alpha)*generate(ConvToModel(surf[i,j],model),init_cond,T).+alpha*Y,Y)
        end
    end
    return Loss3D
end


##read in loss landscape
Y1=npzread("Figure5/duff_basin1.npy")
z0_1=(Y1[1,:,:])
Y2=npzread("Figure5/duff_basin_both.npy")
z0_2=(Y2[1,:,:])

#load models, e.g.: (not included in this repository)
model4,_=load_model("model_2_4_duff20GT.bson")
m1,_=load_model("Results/paper_2201_perturb__radamdecay_hist/duff_1e-3decay1e-6_bpe=50_ssi=5-ℳ_clippedShallowPLRNN-γ_0.999-𝒪_Identity-bs_64-wtf_0.05-α_method_constant-M_2-H_100-T̃_30-model_model_2_4_duff20GT.bson/001/checkpoints/model_2500.bson")
m2,_=load_model("Results/paper_2201_perturb__radamdecay_hist/duff_1e-3decay1e-6_bpe=50_ssi=5-ℳ_clippedShallowPLRNN-γ_0.999-𝒪_Identity-bs_64-wtf_0.05-α_method_constant-M_2-H_100-T̃_30-model_model_2_4_duff20GT.bson/001/checkpoints/model_5000.bson")

model=clippedShallowPLRNN(2,100)
a=-0.2:0.005:1.2
b=0:0.1:1
i=10
b=a
Loss=Loss3D_TF(0.05,a,b,model4,m1,m2,model,Y1[i:i+29,:,:],30,Y1[i,:,:])
my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
    p=Plots.surface(a, b, log.(Loss); colorbar=false,
        xlabel="a", ylabel="b", zlabel="loss",title="m",
        camera=(25, 10), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=700,zlims=(-15.0,10))    
    a0=findall(a .==0)[1]
    a1=findall(a .==1)[1]
    scatter!([a[a0]],[a[a0]],[log(Loss[a0,a0])],label=log(Loss[a0,a0]))
    scatter!([a[a0]],[a[a1]],[log(Loss[a0,a1])],label=log(Loss[a0,a1]))
    scatter!([a[a1]],[a[a0]],[log(Loss[a1,a0])],label=log(Loss[a1,a0]))









