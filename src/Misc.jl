module Misc

using Statistics, Clustering
using JSON
import DataStructures: PriorityQueue, peek
import SpecialFunctions: lbeta #, logabsgamma
import Base: sum, maximum, minimum, argmax, argmin
import Statistics: mean
import LinearAlgebra: norm
import PyPlot

export nanmean, nanstd, nanmax, nansum, nanmin 
export sum, mean, maximum, minimum, argmin, argmax
export logmeanexp, logvarexp, logsumexp, lbeta #, lgamma
export KL, rmse, mae, hoyer
export discretize, categorize, cooccurrences
export finucan, SNR, ROC, AUC

export save_json, load_json
export generate_params, plot_pair

@inline sum(X::AbstractArray,dims) = dropdims(sum(X;dims=dims); dims=dims)
@inline mean(X::AbstractArray,dims) = dropdims(mean(X;dims=dims); dims=dims)
@inline maximum(X::AbstractArray,dims) = dropdims(maximum(X;dims=dims); dims=dims)
@inline minimum(X::AbstractArray,dims) = dropdims(minimum(X;dims=dims); dims=dims)

@inline _nanmean(X::AbstractArray) = mean(filter(!isnan,X))
nanmean(X::AbstractArray; dims=nothing) = dims == nothing ? _nanmean(X) : mapslices(_nanmean, X; dims=dims)
nanmean(X::AbstractArray, dims) = dropdims(nanmean(X; dims=dims),dims=dims)

@inline _nanstd(X::AbstractArray) = std(filter(!isnan,X))
nanstd(X::AbstractArray; dims=nothing) = dims == nothing ? _nanstd(X) : mapslices(_nanstd, X; dims=dims)
nanstd(X::AbstractArray, dims) = dropdims(nanstd(X; dims=dims),dims=dims)

@inline _nansum(X::AbstractArray) = sum(filter(!isnan,X))
nansum(X::AbstractArray; dims=nothing) = dims == nothing ? _nansum(X) : mapslices(_nansum, X; dims=dims)
nansum(X::AbstractArray, dims) = dropdims(nansum(X; dims=dims),dims=dims)

@inline _nanmax(X::AbstractArray) = maximum(filter(!isnan,X))
nanmax(X::AbstractArray; dims=nothing) = dims == nothing ? _nanmax(X) : mapslices(_nanmax, X; dims=dims)
nanmax(X::AbstractArray, dims) = dropdims(nanmax(X; dims=dims),dims=dims)

@inline _argmax(X::AbstractArray) = findmax(X)[2]
argmax(X::AbstractArray; dims=nothing) = dims == nothing ? _argmax(X) : mapslices(_argmax, X; dims=dims)
argmax(X::AbstractArray, dims) = dropdims(argmax(X; dims=dims),dims=dims)

@inline _argmin(X::AbstractArray) = findmin(X)[2]
argmin(X::AbstractArray; dims=nothing) = dims == nothing ? _argmin(X) : mapslices(_argmin, X; dims=dims)
argmin(X::AbstractArray, dims) = dropdims(argmin(X; dims=dims),dims=dims)

@inline _nanmin(X::AbstractArray) = minimum(filter(!isnan,X))
nanmin(X::AbstractArray; dims=nothing) = dims == nothing ? _nanmin(X) : mapslices(_nanmin, X; dims=dims)
nanmin(X::AbstractArray, dims) = dropdims(nanmin(X; dims=dims),dims=dims)

@inline _logsumexp(X::AbstractArray) = begin x⁺ = maximum(X); log(sum(exp, X .- x⁺)) + x⁺ end
logsumexp(X::AbstractArray; dims=nothing) = dims == nothing ? _logsumexp(X) : mapslices(_logsumexp, X; dims=dims)
logsumexp(X::AbstractArray, dims) = dropdims(logsumexp(X; dims=dims),dims=dims)

@inline _logmeanexp(X::AbstractArray) = _logsumexp(X) - log(length(X))
logmeanexp(X::AbstractArray; dims=nothing) = dims == nothing ? _logmeanexp(X) : mapslices(_logmeanexp, X; dims=dims)
logmeanexp(X::AbstractArray, dims) = dropdims(logmeanexp(X; dims=dims),dims=dims)

@inline _logvarexp(X::AbstractArray; μ=_logmeanexp(X)) = begin EX = _logsumexp(X); log(var(exp.(X .- EX); mean=exp(μ - EX))) + 2.0 * EX end
logvarexp(X::AbstractArray; dims=nothing) = dims == nothing ? _logvarexp(X) : mapslices(_logvarexp, X; dims=dims)
logvarexp(X::AbstractArray, dims) = dropdims(logvarexp(X; dims=dims),dims=dims)

@inline _lbeta(X::AbstractArray) =  sum(lgamma, X) - lgamma(sum(X))
lbeta(X::AbstractArray; dims=nothing) = dims == nothing ? _lbeta(X) : mapslices(_lbeta, X; dims=dims)
lbeta(X::AbstractArray, dims) = dropdims(lbeta(X; dims=dims),dims=dims)

@inline _lbeta(γ::Number,I::Integer) = I*lgamma(γ) - lgamma(I*γ)
lbeta(γ::Number, sz::Vararg{Integer}; dims=nothing) = dims == nothing ? _lbeta(γ,prod(sz)) : fill(_lbeta(γ,prod(sz[dims])), map(t -> t[1] in dims ? 1 : t[2], enumerate(sz))...)

#lgamma(x::Real) = logabsgamma(x)[1]

    
function rmse(X::AbstractArray, Xᵖ::AbstractArray)
    return sqrt(nanmean((X .- Xᵖ) .^ 2 ))
end
    
function mae(X::AbstractArray, Xᵖ::AbstractArray)
    return nanmean(abs.(X .- Xᵖ))
end

function ROC(score,labels,truelabel; weights=ones(length(score)))
    N = length(score)
    thresholds = [Inf, sort(score[:])[end:-1:1]...]
    TP, TN, FP, FN = zeros(N+1), zeros(N+1), zeros(N+1), zeros(N+1)
    for (n,th) ∈ enumerate(thresholds)
        TP[n] = sum(((labels .== truelabel) .& (score .>= th)) .* weights)
        TN[n] = sum(((labels .!= truelabel) .& (score .< th)) .* weights)
        FP[n] = sum(((labels .!= truelabel) .& (score .>= th)) .* weights)
        FN[n] = sum(((labels .== truelabel) .& (score .< th)) .* weights)
    end
 
    TPR = TP ./ (TP .+ FN)
    FPR = FP ./ (FP .+ TN)
    return FPR, TPR
end

function AUC(score,labels,truelabel; weights=ones(length(score)))
    area = 0.0
    FPR, TPR = ROC(score,labels,truelabel; weights=weights)
    for n ∈ 2:length(FPR)
        area += (FPR[n] - FPR[n-1])*(TPR[n]+TPR[n-1])/2
    end
    return area
end

function KL(X::AbstractArray, Xᵖ::AbstractArray)
    KL_div = 0.0
    for (x,xᵖ) in zip(X,Xᵖ)
        if !isnan(x) && !isnan(xᵖ)
            KL_div += (x ≈ 0.0) ? 0.0 : x * (log(x) - log(xᵖ)) + xᵖ - x
        end
    end
    return KL_div
end

function SNR(X::AbstractArray, Xᵖ::AbstractArray; base=10.0)
    return 2.0*base*(log(base,norm(X)) - log(base,norm(X .- Xᵖ)))
end

function hoyer(X::AbstractArray)
    N = length(X)
    return (sqrt(N) - sum(abs,X)/sqrt(sum(X .* X))) / (sqrt(N) - 1)
end

function cooccurrences(Tab::AbstractArray{Ƶ,2},dims) where {Ƶ <: Integer}
    T, N = size(Tab)
    X = zeros(Ƶ,dims...)
    for t in 1:T
        X[Tab[t,:]...] += 1
    end
    return X
end

function cooccurrences(Tab::AbstractArray{Ƶ,2}) where {Ƶ <: Integer}
	I = nanmax(Tab;dims=1)[1,:]
	return cooccurrences(Tab,I)
end

function save_json(filename; variables...)
    json_str = JSON.json(Dict(variables));
    open(filename, "w") do f
        write(f, json_str)
    end
    return json_str
end

function load_json(filename)
    json_str = nothing
    open(filename, "r") do f
        json_str = read(f, String)
    end
    return JSON.parse(json_str)
end

function plot_pair(pair::Dict{Symbol,Any};ax=PyPlot.gca(), normalize=false, kwargs...)
	rel = '>' in pair[:relationship] ? "\$\\to\$" : "\$\\leftarrow\$"
	x, y = pair[:X],pair[:Y]
	xⁿ = normalize ? (x .- mean(x)) / std(x) : x
    yⁿ = normalize ? (y .- mean(y)) / std(y) : y
    ax.scatter(xⁿ, yⁿ; marker=".", label="X $(rel) Y", kwargs...)
    ax.set_title("$(pair[:id]): $(pair[:dataset])")
    ax.set_xlabel("X: $(pair[:X_label])")
    ax.set_ylabel("Y: $(pair[:Y_label])")
    ax.legend()
end

function generate_params(A₁,A₂,B₁,B₂;PATH="./")
    num = 0
    for a₁ ∈ A₁, a₂ ∈ A₂, b₁ ∈ B₁, b₂ ∈ B₂
        if a₁ >= b₁ && a₂ >= b₂
            num += 1
            setup = Dict(:name => "params-$num", :P => 1000, :Rs => 1:5, :Ms => 2:4, 
                         :γ => 10.0, :m₁ => 0.0, :m₂ => 0.0, :λ₁ => 0.1, 
                         :λ₂ => 0.1, :a₁ => a₁, :a₂ => a₂, :b₁ => b₁, :b₂ => b₂);
            save_json("$PATH/params-$num.json"; setup...)
        end
    end
    return num
end

end
