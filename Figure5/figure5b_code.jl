using Flux
using Plots
using Statistics
using BSON
using LinearAlgebra
using GTF,ColorSchemes

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

CalcX(model1)

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

function Means(model1::clippedShallowPLRNN,model2::clippedShallowPLRNN)
    m1=mean([norm(model1.W₁-model2.W₁),norm(model1.W₂-model2.W₂),norm(model1.A-model2.A),norm(model1.h₁-model2.h₁),
    norm(model1.h₂-model2.h₂)])
    x1=reduce(vcat,(CalcX(model1)))
    x2=reduce(vcat,(CalcX(model2)))
    m2=norm(x1-x2)
    return [m1,m2]
end
Means(model1_1,model1_2)


using NPZ
Y=npzread("Training_Data/paper_1512/duff_basin1.npy")
model1=load_model("model_2_4_duff20GT.bson")[1]
model2=load_model("model_2_9_duff20GT.bson")[1]
model3=load_model("model_2_10_duff20GT.bson")[1]
model=load_model("model_2_14_duff20GT.bson")[1]
a=0:0.01:1
Loss,m=Loss3D_TF(0.05,a,b,model1,model2,model3,model,Y,4000,Y[1,:,:])
ConvToModel(CalcX(model1),model)

Flux.Losses.mse(generate(model1,Y[1,:,:],4000),Y)
al=0.05
Flux.Losses.mse((1-al)*generate(ConvToModel(CalcX(model1),model),Y[1,:,:],4000).+al*Y,Y)
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
a[1]
using ColorSchemes
###normal plot
a=0:0.01:1
b=0:0.01:1
b=a

Loss=Loss3D_TF(0.05,a,model1,model2,model3,model,Y[1020:1049,:,:],30,Y[1020,:,:])
Loss[1,1]
Y[120:150,:,:]
my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
surface(a, b, log.(Loss); colorbar=false,
    xlabel="a", ylabel="b", zlabel="loss",title=m,
    camera=(20, 20), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=1000,zlims=(-15.0,10))
    scatter!([a[1]],[b[1]],[log(Loss[1,1])],label=log(Loss[1,1]))
    scatter!([a[1]],[b[101]],[log(Loss[1,101])],label=log(Loss[1,101]))
    scatter!([a[101]],[b[1]],[log(Loss[101,1])],label=log(Loss[101,1]))
    plot!(zeros(101),a[1:101],log.(Loss[1:101,1]),color=:red)
    plot!( a[1:101],zeros(101),log.(Loss[1,1:101]),color=:red)
    #zlims=(-5.0,10)

#savefig("m11m23m32.png")

###hessian


Y1=npzread("Training_Data/paper_1512/duff_basin1.npy")
z0_1=(Y1[1,:,:])
Y2=npzread("Training_Data/paper_1512/duff_basin_both.npy")
z0_2=(Y2[1,:,:])

#models=["model_2_duff_16GT.bson","model_3_duff_16GT.bson","model_6_duff_16GT.bson","model_10_duff_16GT.bson","model_19_duff_16GT.bson"]
model4,_=load_model("model_2_29_duff20GT.bson")
38_i=10
i=10

    model=model4
  
  

        z0_1=(Y1[i,:,:])
        z0_2=(Y2[i,:,:])


        A,W1,W2,b1,b2=GetModel(model)
        P=reduce(vcat,CalcX(model))

        H1 = Zygote.hessian(θ -> L2(z0_1, θ,Y1[i:i+30,:,:],2,100,30), P)
eigvals(H1)
        h1=real.(eigvals(H1))
println(h1)
      length(findall(x->x>0,h1))
      length(findall(x->x<0,h1))
    length(findall(x->x==0.0,h1))
        maximum(h1)
minimum(h1)

eigvecs(H_full)[:,1]

Zygote.hessian(θ -> L2(z0_2, θ,Y2[i:i+30,:,:],2,100,30), P)
    H_full = Zygote.hessian(θ -> L2(z0_2, θ,Y2[i:i+30,:,:],2,100,30), P)

        h_full=real.(eigvals(H_full))

        length(findall(x->x>0,h_full))
      length(findall(x->x<0,h_full))
        length(findall(x->x==0.0,h_full))
        maximum(h_full)
        minimum(h_full)

    
maximum(abs.(h1-h_full))
x
for i in 1:504

    if h1[i]== 0 && h_full[i] >0
        println(i)
    end
    
end


h1[155:168]
h_full[252]

eigvecs(H_full)[:,251]
eigvecs(H1)[:,251]

reduce(vcat,CalcX(model4))

function ConvToModel2(x::Vector{Float32})
    model=clippedShallowPLRNN(2,100)
    lW1_1,lW1_2=size(model.W₁)
    lW2_1,lW2_2=size(model.W₂)
    l_A=length(model.A)
    l_h1=length(model.h₁)
    l_h2=length(model.h₂)

    model.A .= x[1:l_A]
    k=l_A
    model.W₁ .= reshape(x[k+1:k+lW1_1*lW1_2],(lW1_1,lW1_2))
    k=k+lW1_1*lW1_2
    model.h₁ .= x[k+1:k+l_h1]
    k=k+l_h1
    model.W₂ .= reshape(x[k+1:k+lW2_1*lW2_2],(lW2_1,lW2_2))
    k=k+lW2_1*lW2_2
    model.h₂ .= x[k+1:k+l_h2]
    return model
end


rn=252
    model4_full_min=ConvToModel2(reduce(vcat,CalcX(model4))+1* real(eigvecs(H_full)[:,rn]))

    #model4_notfull_min=ConvToModel2(reduce(vcat,CalcX(model4))+1* real(eigvecs(H1)[:,100]))


    model4_notfull_min=ConvToModel2(reduce(vcat,CalcX(model4))+1* real(eigvecs(H_full)[:,rn]))


    model4_full_max=ConvToModel2(reduce(vcat,CalcX(model4))+ real(eigvecs(H_full)[:,end]))

    model4_notfull_max=ConvToModel2(reduce(vcat,CalcX(model4))+ real(eigvecs(H1)[:,end]))
    model=clippedShallowPLRNN(2,100)
    a=-3.0:0.1:2
    b=-1:0.1:1
    i=10
    b=a
    Loss=Loss3D_TF(0.05,a,b,model4,model4_notfull_min,model4_notfull_max,model,Y1[i:i+29,:,:],30,Y1[i,:,:])

    my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
    p=Plots.surface(a, b, log.(Loss); colorbar=false,
        xlabel="a", ylabel="b", zlabel="loss",title="m",
        camera=(20, 20), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=700,zlims=(-20.0,10))
        #savefig(p,"plots_min/mono_$rn .png")

        Loss=Loss3D_TF(0.05,a,b,model4,model4_full_min,model4_full_max,model,Y2[i:i+29,:,:],30,Y2[i,:,:])
        
        my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
        p=Plots.surface(a, b, log.(Loss); colorbar=false,
            xlabel="a", ylabel="b", zlabel="loss",title="m",
            camera=(20, 20), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=700,zlims=(-15.0,10))    
        #savefig(p,"plots_min/multi_$rn .png")





#######gradient



model4,_=load_model("model_2_29_duff20GT.bson")
model=model4
 i=10 
  z0_1=(Y1[i,:,:])
z0_2=(Y2[i,:,:])


A,W1,W2,b1,b2=GetModel(model)
P=reduce(vcat,CalcX(model))
H1 = Zygote.gradient(θ -> L2(z0_1, θ,Y1[i:i+30,:,:],2,100,30), P)
grad=H1[1]
H1hess = Zygote.hessian(θ -> L2(z0_1, θ,Y1[i:i+30,:,:],2,100,30), P)



model4_full_min=ConvToModel2(reduce(vcat,CalcX(model4))+10* grad)

#model4_notfull_min=ConvToModel2(reduce(vcat,CalcX(model4))+1* real(eigvecs(H1)[:,100]))


model4_notfull_min=ConvToModel2(reduce(vcat,CalcX(model4))+10* grad)


model4_full_max=ConvToModel2(reduce(vcat,CalcX(model4))+ real(eigvecs(H_full)[:,end]))

model4_notfull_max=ConvToModel2(reduce(vcat,CalcX(model4))+ real(eigvecs(H1hess)[:,end]))
model=clippedShallowPLRNN(2,100)
a=-0.5:0.1:1
b=-1:0.1:1
i=10
b=a
Loss=Loss3D_TF(0.05,a,b,model4,model4_notfull_min,model4_notfull_max,model,Y1[i:i+29,:,:],30,Y1[i,:,:])

my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
p=Plots.surface(a, b, log.(Loss); colorbar=false,
    xlabel="a", ylabel="b", zlabel="loss",title="m",
    camera=(20, 20), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=700,zlims=(-20.0,10))
    #savefig(p,"plots_min/mono_$rn .png")

    Loss=Loss3D_TF(0.05,a,b,model4,model4_full_min,model4_full_max,model,Y2[i:i+29,:,:],30,Y2[i,:,:])
    
    my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
    p=Plots.surface(a, b, log.(Loss); colorbar=false,
        xlabel="a", ylabel="b", zlabel="loss",title="m",
        camera=(20, 20), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=700,zlims=(-15.0,10))    
    #savefig(p,"plots_min/multi_$rn .png")


####
33ok 
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
    #savefig(p,"plots_min/multi_$rn .png")
    a0=findall(a .==0)[1]
    a1=findall(a .==1)[1]
    scatter!([a[a0]],[a[a0]],[log(Loss[a0,a0])],label=log(Loss[a0,a0]))
    scatter!([a[a0]],[a[a1]],[log(Loss[a0,a1])],label=log(Loss[a0,a1]))
    scatter!([a[a1]],[a[a0]],[log(Loss[a1,a0])],label=log(Loss[a1,a0]))
npzwrite("Loss1.npy",Loss)    
    plot(Loss[:,6])
heatmap(a,a,log.(Loss))
    a[end]
Loss=Loss3D_TF(0.05,a,b,model4,m1,m2,model,Y2[i:i+29,:,:],30,Y2[i,:,:])
    plot(Loss[:,6])
    my_cg = cgrad([ColorSchemes.devon10[3],ColorSchemes.dracula[end]])
    p=Plots.surface(a, b, log.(Loss); colorbar=false,
        xlabel="a", ylabel="b", zlabel="loss",title="m",
        camera=(20, 20), color=my_cg,display_option=Plots.GR.OPTION_SHADED_MESH,alpha = 0.7,dpi=700,zlims=(-15.0,10))    
        npzwrite("Loss2.npy",Loss)  
        a0=findall(a .==0)[1]
        a1=findall(a .==1)[1]
        scatter!([a[a0]],[a[a0]],[log(Loss[a0,a0])],label=log(Loss[a0,a0]))
        scatter!([a[a0]],[a[a1]],[log(Loss[a0,a1])],label=log(Loss[a0,a1]))
        scatter!([a[a1]],[a[a0]],[log(Loss[a1,a0])],label=log(Loss[a1,a0]))
        heatmap(a,a,log.(Loss))

        plot(Loss[:,6])



###scatter

function Loss1D_TF(alpha::Float64,a::StepRangeLen,model1,model2,model,Y,T,init_cond)
    b=a
    surf=Vector{Vector{Vector{Float32}}}(undef,length(a))
    for i in 1:length(a)
        
        surf[i]=Float32(a[i])*(CalcX(model2) - CalcX(model1))+ CalcX(model1)
        
    end
    Loss3D=Vector{Float32}(undef,length(a))
    for i in 1:length(a)
        
            Loss3D[i]=Flux.Losses.mse((1-alpha)*generate(ConvToModel(surf[i],model),init_cond,T).+alpha*Y,Y)
    end
    return Loss3D
end

using NaturalSort
modelinit,_=load_model("model_2_14_duff20GT.bson")
m1,_=load_model("Results/paper_2101_perturb_sortedbatch/duff_5e-4_bpe=1_ssi=5-ℳ_clippedShallowPLRNN-γ_0.999-𝒪_Identity-bs_64-wtf_0.05-α_method_constant-M_2-H_100-T̃_30-model_model_2_14_duff20GT.bson/001/checkpoints/model_5.bson")
i=10
dir=readdir("Results/paper_2201_perturb__radamdecay_hist/duff_1e-3decay1e-6_bpe=50_ssi=5-ℳ_clippedShallowPLRNN-γ_0.999-𝒪_Identity-bs_64-wtf_0.05-α_method_constant-M_2-H_100-T̃_30-model_model_2_14_duff20GT.bson/001/checkpoints",join=true)
dir=sort(dir,lt=natural)[1:end]
model=clippedShallowPLRNN(2,100)
L_all=Array{Float32}(undef,100,length(a),2)
a=-0:0.005:1
length(a)
Threads.@threads for m in 1:length(dir)
    println(m)
    modele,_=load_model(dir[m])
    L_all[m,:,2]=Loss1D_TF(0.05,a,modelinit,modele,model,Y2[i:i+29,:,:],30,Y2[i,:,:])
    L_all[m,:,1]=Loss1D_TF(0.05,a,modelinit,modele,model,Y1[i:i+29,:,:],30,Y1[i,:,:])
end
scatter()
for i in 1:100
    plot!(L_all[i,:,2],label=false)
end
current()

npzwrite("L_all.npy",L_all)



modelinit,_=load_model("model_2_14_duff20GT.bson")
m1,_=load_model("Results/paper_2101_perturb_sortedbatch/duff_5e-4_bpe=1_ssi=5-ℳ_clippedShallowPLRNN-γ_0.999-𝒪_Identity-bs_64-wtf_0.05-α_method_constant-M_2-H_100-T̃_30-model_model_2_14_duff20GT.bson/001/checkpoints/model_5.bson")
i=10
dir=readdir("Results/paper_2201_perturb__radamdecay_hist/duff_1e-3decay1e-6_bpe=50_ssi=5-ℳ_clippedShallowPLRNN-γ_0.999-𝒪_Identity-bs_64-wtf_0.05-α_method_constant-M_2-H_100-T̃_30-model_model_2_14_duff20GT.bson/001/checkpoints",join=true)
dir=sort(dir,lt=natural)[1:end]
model=clippedShallowPLRNN(2,100)
L_all=Array{Float32}(undef,99,length(a),2)
a=-0:0.005:1
length(a)
 for m in 1:length(dir)-1
    println(m)
    modele1,_=load_model(dir[m])
    modele2,_=load_model(dir[m+1])

    L_all[m,:,2]=Loss1D_TF(0.05,a,modele1,modele2,model,Y2[i:i+29,:,:],30,Y2[i,:,:])
    L_all[m,:,1]=Loss1D_TF(0.05,a,modele1,modele2,model,Y1[i:i+29,:,:],30,Y1[i,:,:])
end
scatter()
for i in 1:100
    plot!(L_all[i,:,1],label=false)
end
current()

npzwrite("L_all.npy",L_all)