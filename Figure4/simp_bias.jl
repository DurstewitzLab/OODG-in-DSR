using GTF, Plots, Flux, NPZ, LinearAlgebra, ColorSchemes, DynamicalSystems
BLAS.set_num_threads(1)  

# Define a clipped shallow PLRNN 
function clippedShallowPLRNN(M::Int, hidden_dim::Int, s)
    A = Flux.glorot_normal(M; gain=0.2)
    h₁ = Flux.glorot_normal(M; gain=s)
    h₂ = Flux.glorot_normal(hidden_dim; gain=s)
    W₁ = Flux.glorot_normal(M, hidden_dim; gain=s)
    W₂ = Flux.glorot_normal(hidden_dim, M; gain=s)
  
    return clippedShallowPLRNN(A, W₁, W₂, h₁, h₂, nothing)
end

# Define bin size for rectangular binning
ε = 0.005
est = VisitationFrequency(RectangularBinning(ε))

# Define a set of variances
s_all = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]

# Define PLRNN dimensions
hidden_dim = 100
N = 2

# number of iterations to calculate the plot
num_iter = 1000

# Initialize matrix to store results
X_v = Matrix{Float64}(undef, length(s_all), num_iter)

# Loop over variances and iterations for calculating entropies
for k in 1:length(s_all)
    Threads.@threads for i in 1:num_iter
        m = clippedShallowPLRNN(N, hidden_dim, s_all[k])

        # Generate data using the model
        X = generate(m, rand(-3:0.01:3, 2, 1000), 8000)

        # Check for diverging models
        if sum(isnan.(X)) > 0 || sum(isinf.(X)) > 0 || sum(findall(x -> abs(x) > 1e3, vcat(X...))) > 0
            X_v[k, i] = NaN
        else
            Y = X[end, :, :]' |> StateSpaceSet
            try
                println(k, i)

                # Compute visitation frequency and entropy
                p = probabilities(est, Y)
                X_v[k, i] = information(Renyi(q=1.0, base=MathConstants.e), p)  # Calculate entropy
             
            catch y # Filter values which give an error
                if isa(y, InexactError)
                    println("inexact error")
                    X_v[k, i] = NaN
                end
            end
        end
    end
    npzwrite("monostab_bias/simp_bias.npy", X_v)
end

using Plots, NPZ, LaTeXStrings, ColorSchemes, Statistics, StatsPlots

# Filter out NaN values from the data
function m_nan(X::Vector, th)
    M = Vector{Float32}()
    for k in X
        if k > -0.01 
            if k != Inf && k < th
                push!(M, k)
            end
        end
    end
    return M
end

# Load and process results

X=npzread("Figure4/glorot_normal.npy")

X_1=Matrix{Float64}(undef,14,3)
for i in 1:14
    X_1[i,1]=   mean(m_nan(X[i,:],100))
    X_1[i,2]=   std(m_nan(X[i,:],100))/sqrt(length(m_nan(X[i,:],100)))
end


X=npzread("Figure4/glorot_uniform.npy")

X_2 = Matrix{Float64}(undef, size(X, 1), 2)
for i in 1:size(X, 1)
    X_2[i, 1] = mean(m_nan(X[i, :], 100))
    X_2[i, 2] = std(m_nan(X[i, :], 100)) / sqrt(length(m_nan(X[i, :], 100)))
end

# Density plot / Fig 4.a)
density(m_nan(X[2,:],100),label=L"\sigma=0.3",bandwidth=0.05,fill=(0, 0.25,ColorSchemes.YlGn_9[end]),xlims=(0,2),size=(600,250),
color=ColorSchemes.YlGn_9[end],lw=3,grid=false,xlabel=L"e", ylabel=L"L_{\mathcal{E}_{stat}}")
plot!([0.2; 3], [0; 0], lw=3,label="",color=ColorSchemes.YlGn_9[end])
density!(m_nan(X[9,:],100),label=L"\sigma=3.0",bandwidth=0.04,fill=(0, 0.25,ColorSchemes.YlGn_9[end-3]),xlims=(0,3),size=(600,250),
color=ColorSchemes.YlGn_9[end-3],lw=3,grid=false,xlabel=L"e", ylabel=L"L_{\mathcal{E}_{stat}}")
p=density!(m_nan(X[13,:],100),label=L"\sigma=5.0",bandwidth=0.05,fill=(0, 0.25,ColorSchemes.YlGn_9[end-5]),xlims=(0,2),
color=ColorSchemes.YlGn_9[end-5],lw=3,grid=false,xlabel=L"H", ylabel=L"density",thickness_scaling=1.4,size=(600,200),framestyle=:box,legend=:bottomright,legendfont=font(7))

# Plot results / Fig 4.b) for glorot uniform
p=plot(s_all,X_1[:,1],ribbon=X_1[:,2],lw=2,c=ColorSchemes.Paired_12[8]	
,label="Glorot uniform",grid=false,xlabel="gain",ylabel=L"H",thickness_scaling=1.4,size=(600,200),framestyle=:box,
yticks=[0,1,2,3],guidefontsize=9,xticks=[0,2,4,6])
p=plot!(s_all,X_2[:,1],ribbon=X_2[:,2],lw=2,c=ColorSchemes.coffee[1],	
label="Glorot normal",grid=false,xlabel="gain",ylabel=L"H",thickness_scaling=1.0,size=(600,200),framestyle=:box,
yticks=[0,1,2,3],guidefontsize=9,xticks=[0,2,4,6],dpi=1000)


