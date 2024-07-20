using NPZ,Plots
using NaturalSort
using LaTeXStrings,ColorSchemes
using StatsPlots



function m_nan(X::Vector,th)
    M=Vector{Float32}()
    for k in X
        if k >-0.01 
            if k != Inf && k<th
                push!(M,k)
            end
        end
    end
    return M
end



X_all_GT=npzread("Figure5/models_trained_0error.npy")

X_all_GT_2=Matrix{Float64}(undef,18,2)
for i in 1:4
    X_all_GT_2[i,:]=X_all_GT[i,:]
    println(i)
end
for i in 5:7
    X_all_GT_2[i,:]=X_all_GT[i+1,:]
    println(i+1)
end
X_all_GT_2[8,:]=X_all_GT[10,:]
for i in 9:18
    X_all_GT_2[i,:,:]=X_all_GT[i+2,:]
    println(i+2)
end

X_all=npzread("Figure5/models_retrained.npy")

X_all_2=Array{Float64}(undef,18,20,2)
for i in 1:4
    X_all_2[i,:,:]=X_all[i,:,:]
    println(i)
end
for i in 5:7
    X_all_2[i,:,:]=X_all[i+1,:,:]
    println(i+1)
end
X_all_2[8,:,:]=X_all[10,:,:]
for i in 9:18
    X_all_2[i,:,:]=X_all[i+2,:,:]
    println(i+2)
end

X_all_2=reshape(X_all_2,(360,2))


violin(X_all_GT[:,1],c=ColorSchemes.Paired_12[4],lw=2,label="",xticks=(1,[""]))
p=violin!(X_all_2[:,1],c=ColorSchemes.Paired_12[10],lw=2,ylim=(0.029,0.1),yticks=[0.03,0.06,0.09],size=(300,150),label="",framestyle=:box,grid=false,xticks=([1,2],["",""]),dpi=1000)
savefig(p,"violin1.png")
violin(X_all_GT[:,2],c=ColorSchemes.Paired_12[4],lw=2,label="")
p=violin!(X_all_2[:,2],c=ColorSchemes.Paired_12[10],yticks=[0.1,1],yscale=:log10,lw=2,size=(300,150),label="",framestyle=:box,grid=false,xticks=([1,2],["",""]),dpi=1000)
savefig(p,"violin2.png")
