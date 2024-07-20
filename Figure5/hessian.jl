
using LinearAlgebra,Zygote,ForwardDiff,GTF,NPZ,Flux

BLAS.set_num_threads(1)

f2(z, A, W₁ ,h₁, W₂,h₂) = A .* z .+ W₁ * (relu.(W₂*z .+ h₂) .- relu.(W₂*z)) .+ h₁


function CalcX(model)
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
    return reduce(vcat,x)
end

function GetModel(model)
    A=model.A
    W1=model.W₁
    W2=model.W₂
    #W2=Matrix{Float64}(I, 50, 3)
    b1=model.h₁
    b2=model.h₂
    return A,W1,W2,b1,b2
end




function L2(z0, θ,Y,N,H,long_T)
    A = θ[1:N]
    W1 = reshape(θ[N+1:N+N*H],(N,H))
    b1 = θ[N+N*H+1:N+N*H+N]
    W2= reshape(θ[N+N*H+N+1:N+N*H+N+H*N],(H,N))
    b2 = θ[N+N*H+N+H*N+1:N+N*H+N+H*N+H]
    T = size(Y,1)
    b=size(Y,3)
    Z = Array{Float32}(undef,long_T,N,b) 
    z=z0
    Loss = 0.0f0
    α = 0.05f0
    for t = 2:long_T
        z = f2(z,A, W1,b1, W2,b2)
        Loss += Flux.mse(z, Y[t, :,:]) #+ 50*norm(f2([1.4877174f0,0.04595573f0],A, W1,b1, W2,b2)-[1.4877174f0,0.04595573f0])
        z = (1 - α) * z + α * Y[t,:,:]
    end
    return Loss / T
end


Y1=npzread("Figure5/duff_basin1.npy")
z0_1=(Y1[1,:,:])
Y2=npzread("Figure5/duff_basin_both.npy")
z0_2=(Y2[1,:,:])

#load models here, e.g.: (not included in this repository)
models=["model_2_duff_16GT.bson","model_3_duff_16GT.bson","model_6_duff_16GT.bson","model_10_duff_16GT.bson","model_19_duff_16GT.bson"]


X_all=Array{Float32}(undef,5,135,2,5)
for m in 1:length(models)
    model,_=load_model(models[m])
    Threads.@threads for j in 1:135
        println(j)

        i=rand(1:3970)
        z0_1=(Y1[i,:,:])
        z0_2=(Y2[i,:,:])


        A,W1,W2,b1,b2=GetModel(model)
        P=reduce(vcat,CalcX(model))

        H1 = Zygote.hessian(θ -> L2(z0_1, θ,Y1[i:i+30,:,:],2,100,30), P)

        h1=real.(eigvals(H1))

        X_all[m,j,1,1]=length(findall(x->x>0,h1))
        X_all[m,j,1,2]=length(findall(x->x<0,h1))
        X_all[m,j,1,3]=length(findall(x->x==0.0,h1))
        X_all[m,j,1,4]=maximum(h1)
        X_all[m,j,1,5]=minimum(h1)



        H1 = Zygote.hessian(θ -> L2(z0_2, θ,Y2[i:i+30,:,:],2,100,30), P)

        h1=real.(eigvals(H1))

        X_all[m,j,2,1]=length(findall(x->x>0,h1))
        X_all[m,j,2,2]=length(findall(x->x<0,h1))
        X_all[m,j,2,3]=length(findall(x->x==0.0,h1))
        X_all[m,j,2,4]=maximum(h1)
        X_all[m,j,2,5]=minimum(h1)

    end
    npzwrite("hessian_1801_l=50.0.npy",X_all)
end
