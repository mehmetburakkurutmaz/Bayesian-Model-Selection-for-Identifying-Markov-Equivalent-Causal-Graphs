module Experiments

include("Misc.jl");
include("BayesianCausality.jl");

using .Misc, .BayesianCausality #, .Tokenizer

using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import .Misc: lgamma
import Base.Iterators: product
import Clustering: kmeans
using GaussianMixtures: GMM
using ScikitLearn: fit!, predict_proba
import Random: randperm

export v_bayes_zero_pseudodata
export causal_likelihoods, vb_causal_likelihoods
export conduct_vb, get_pseudodata


# function v_bayes_pseudodata(X::Array{ℜ,2}, R::Ƶ=1; M::Ƶ=1, γ::Array{ℜ,1}, m₁::Array{ℜ,1}, m₂::Array{ℜ,2},
#         Λ₁::Array{ℜ,1}, Λ₂::Array{ℜ,3}, a₁::Array{ℜ,1}, a₂::Array{ℜ,1}, 
#         b₁::Array{ℜ,1}, b₂::Array{ℜ,1}, EPOCHS=1) where {ℜ<:Real, Ƶ<:Integer}
#     T, N = size(X)
#     ELBO = zeros(EPOCHS)
    
#     p = Particle(X, R; M=M)
#     p.r .=  T < R ? [1:T...] : R > 1 ? kmeans(Array(X'),R).assignments : ones(Int,T)
#     X₁, X₂ = X[:,1], X[:,2]
    
#     γʰ = zeros(R)
#     Λ₁ʰ, m₁ʰ = zeros(R), zeros(R)
#     a₁ʰ, b₁ʰ = zeros(R), zeros(R)
#     Λ₂ʰ, m₂ʰ = zeros(R,M,M), zeros(R,M) 
#     a₂ʰ, b₂ʰ = zeros(R), zeros(R)
    
#     ϕ, log_q, q = zeros(M), zeros(R), zeros(R)
#     ρ₁, log_ρ₁, ρ₂, log_ρ₂, log_θ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
    
#     for (r, x₁, x₂) ∈ zip(p.r, X₁, X₂)
#         ϕ .= x₁.^(0:M-1)
#         p.Σ_R[r] += 1.0
#         p.Σ_X₁[r] += x₁
#         p.Σ_X₁²[r] += x₁*x₁
#         p.Σ_X₂²[r] += x₂*x₂
#         p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
#         p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
#     end
    
#     for eph ∈ 1:EPOCHS
#         for r ∈ 1:R
#             γʰ[r] = γ[r] + p.Σ_R[r]
#             Λ₁ʰ[r] = Λ₁[r] + p.Σ_R[r]
#             m₁ʰ[r] = (Λ₁[r]*m₁[r] + p.Σ_X₁[r]) / Λ₁ʰ[r]
#             a₁ʰ[r] = a₁[r] + p.Σ_R[r]/2
#             b₁ʰ[r] = b₁[r] + (p.Σ_X₁²[r] - Λ₁ʰ[r]*m₁ʰ[r]*m₁ʰ[r] + Λ₁[r]*m₁[r]*m₁[r])/2
        
#             Λ₂ʰ[r,:,:] .= Λ₂[r,:,:] .+ p.Σ_ϕϕᵀ[r,:,:]
#             m₂ʰ[r,:] .= Λ₂ʰ[r,:,:] \ (Λ₂[r,:,:]*m₂[r,:] .+ p.Σ_X₂ϕ[r,:])
#             a₂ʰ[r] = a₂[r] + p.Σ_R[r]/2
#             b₂ʰ[r] = b₂[r] + (p.Σ_X₂²[r] - m₂ʰ[r,:]'*Λ₂ʰ[r,:,:]*m₂ʰ[r,:] + m₂[r,:]'*Λ₂[r,:,:]*m₂[r,:])/2
            
#             log_ρ₁[r] = digamma(a₁ʰ[r]) - log(b₁ʰ[r])
#             ρ₁[r] = a₁ʰ[r]/b₁ʰ[r]
#             log_ρ₂[r] = digamma(a₂ʰ[r]) - log(b₂ʰ[r])
#             ρ₂[r] = a₂ʰ[r]/b₂ʰ[r]
#             log_θ[r] = digamma(γʰ[r]) - digamma(sum(γ)+T)
            
#             ELBO[eph] += (γ[r]-γʰ[r])*log_θ[r]
            
#             ELBO[eph] += a₁[r]*log(b₁[r]) - a₁ʰ[r]*log(b₁ʰ[r])
#             ELBO[eph] += lgamma(a₁ʰ[r]) - lgamma(a₁[r])
#             ELBO[eph] += (log(Λ₁[r]) - log(Λ₁ʰ[r]))/2
#             ELBO[eph] += (1 - Λ₁[r]/Λ₁ʰ[r])/2
#             ELBO[eph] += (a₁[r] - a₁ʰ[r])*log_ρ₁[r] - (b₁[r] - b₁ʰ[r])*ρ₁[r]
#             ELBO[eph] -= (ρ₁[r]*Λ₁[r]*(m₁ʰ[r] - m₁[r])^2)/2
                
#             ELBO[eph] += a₂[r]*log(b₂[r]) - a₂ʰ[r]*log(b₂ʰ[r])
#             ELBO[eph] += lgamma(a₂ʰ[r]) - lgamma(a₂[r])
#             ELBO[eph] += (logdet(Λ₂[r,:,:]) - logdet(Λ₂ʰ[r,:,:]))/2
#             ELBO[eph] += (M - tr(Λ₂ʰ[r,:,:] \ Λ₂[r,:,:]))/2
#             ELBO[eph] += (a₂[r] - a₂ʰ[r])*log_ρ₂[r] - (b₂[r] - b₂ʰ[r])*ρ₂[r]
#             ELBO[eph] -= ρ₂[r]*((m₂ʰ[r,:] - m₂[r,:])'*Λ₂[r,:,:]*(m₂ʰ[r,:] - m₂[r,:]))/2
#         end
        
#         ELBO[eph] += lgamma(sum(γ)) - sum(lgamma,γ)
#         ELBO[eph] -= lgamma(sum(γ)+T) - sum(lgamma,γʰ)
        
#         p.Σ_R .= 0.0
#         p.Σ_X₁ .= 0.0
#         p.Σ_X₁² .= 0.0
#         p.Σ_X₂² .= 0.0
#         p.Σ_X₂ϕ .= 0.0
#         p.Σ_ϕϕᵀ .= 0.0
        
#         for (t,(x₁, x₂)) ∈ enumerate(zip(X₁, X₂))
#             ϕ .= x₁.^(0:M-1)
#             for r ∈ 1:R
#                 log_q[r] = log_θ[r] 
#                 log_q[r] += (log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / Λ₁ʰ[r])/2
#                 log_q[r] += (log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r,:])^2 - ϕ'* (Λ₂ʰ[r,:,:] \ ϕ))/2
#             end
            
#             log_q .-= logsumexp(log_q)
#             q .= exp.(log_q)
            
#             for r ∈ 1:R                
#                 p.Σ_R[r] += q[r]
#                 p.Σ_X₁[r] += q[r]*x₁
#                 p.Σ_X₁²[r] += q[r]*x₁*x₁
#                 p.Σ_X₂²[r] += q[r]*x₂*x₂
#                 p.Σ_X₂ϕ[r,:] .+= q[r]*x₂*ϕ
#                 p.Σ_ϕϕᵀ[r,:,:] .+= q[r]*ϕ*ϕ'
                
#                 ELBO[eph] += q[r]*(log_θ[r] - log_q[r])
#                 ELBO[eph] += q[r]*(log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / Λ₁ʰ[r])/2
#                 ELBO[eph] += q[r]*(log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r,:])^2 - ϕ'* (Λ₂ʰ[r,:,:] \ ϕ))/2
#             end
#             _, p.r[t] = findmax(log_q)
#         end
                
#         ELBO[eph] -= T*log(2π)
#         if eph > 1 && ELBO[eph] - ELBO[eph-1] < 1e-6
#             ELBO = ELBO[1:eph]
#             break
#         end
#     end
#     return ELBO, p.r
# end

function conduct_vb(X::Array{ℜ,2}, R::Int, M::Int, Xᵖ::Array{ℜ,3}; EPOCHS::Int) where {ℜ <: Real}
    params_new = make_param(Xᵖ,M)
    # we have to correct for spurious case
    if M == 1
        params_new[:a₂] = params_new[:a₁]
        params_new[:b₂] = params_new[:b₁]
    end
    return v_bayes_pseudodata(X, R; M=M, EPOCHS=EPOCHS, params_new...)
end

function conduct_vb(X::Array{ℜ,2}, R::Int, M::Int; γ, m₁, m₂, 
        λ₁, λ₂, a₁, a₂, b₁, b₂, EPOCHS) where {ℜ <: Real}
    params_original = make_param(R, M; γᵣ=γ, m₁ᵣ=m₁, m₂ᵣ=m₂, λ₁ᵣ=λ₁, λ₂ᵣ=λ₂,
        a₁ᵣ=a₁, a₂ᵣ=a₂, b₁ᵣ=b₁, b₂ᵣ=b₂)
    # we have to correct for spurious case
    if M == 1
        params_original[:a₂] = params_original[:a₁]
        params_original[:b₂] = params_original[:b₁]
    end
    return BayesianCausality.v_bayes_pseudodata(X, R; M=M, EPOCHS=EPOCHS, params_original...)
end

function make_param(Xᵖ::Array{ℜ,3},M::Int) where {ℜ <: Real}
    R, T = size(Xᵖ)[1:end-1]
    # FIXME: Fix this check
    # if T != Int((M+1)*M/2)
    # throw("Prior data size not correct")
    # end
    X = reshape(Xᵖ,:,2)
    p = Particle(X, R; M=M)
    ϕ = zeros(M)
    
    for r ∈ 1:R, t ∈ 1:T
        x₁, x₂ = Xᵖ[r,t,:]
        ϕ .= x₁.^(0:M-1)
        p.Σ_R[r] += 1.0
        p.Σ_X₁[r] += x₁
        p.Σ_X₁²[r] += x₁*x₁
        p.Σ_X₂²[r] += x₂*x₂
        p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
        p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
    end
    
    γ = zeros(R)
    Λ₁, m₁ = zeros(R), zeros(R)
    a₁, b₁ = zeros(R), zeros(R)
    Λ₂, m₂ = zeros(R,M,M), zeros(R,M) 
    a₂, b₂ = zeros(R), zeros(R)
    
    for r ∈ 1:R
        γ[r] = p.Σ_R[r]
        Λ₁[r] = p.Σ_R[r]
        m₁[r] = p.Σ_X₁[r] / Λ₁[r]
        a₁[r] = p.Σ_R[r]/2
        b₁[r] = (p.Σ_X₁²[r] - Λ₁[r]*m₁[r]*m₁[r])/2
        
        Λ₂[r,:,:] .= p.Σ_ϕϕᵀ[r,:,:]
        m₂[r,:] .= Λ₂[r,:,:] \ p.Σ_X₂ϕ[r,:]
        a₂[r] = p.Σ_R[r]/2
        b₂[r] = (p.Σ_X₂²[r] - m₂[r,:]'*Λ₂[r,:,:]*m₂[r,:])/2
    end
    
    param = Dict(:γ=>γ, :m₁=>m₁, :m₂=>m₂, :Λ₁=>Λ₁, 
    :Λ₂=>Λ₂, :a₁=>a₁, :a₂=>a₂, :b₁=>b₁, :b₂=>b₂);
    
    return param
end

function make_param(R::Int, M::Int; γᵣ, m₁ᵣ, m₂ᵣ, λ₁ᵣ, λ₂ᵣ, a₁ᵣ, a₂ᵣ, b₁ᵣ, b₂ᵣ)
    return  Dict(:γ=>fill(γᵣ,R), :m₁=>zeros(R), :m₂=>zeros(R,M), :Λ₁=>fill(λ₁ᵣ,R), 
        :Λ₂=>zeros(R,M,M) .+ λ₂ᵣ*reshape(diagm(0 => ones(M)),1,M,M), :a₁=>fill(a₁ᵣ,R), 
        :a₂=>fill(a₂ᵣ,R), :b₁=>fill(b₁ᵣ,R), :b₂=>fill(b₂ᵣ,R));
end

function vb_causal_likelihoods(x::AbstractArray,y::AbstractArray,R::Int; M, EPOCHS,
                            parameters_pair, norm, pseudodata_type, m_zero)
    """
    Uses the previously optimized hyperparameters in the parameters dictionary, or obtains pseudodata.
     """
    if m_zero
        m₁, m₂ = 0., 0.
    end
    xⁿ = norm ? (x .- mean(x)) / std(x) : x
    yⁿ = norm ? (y .- mean(y)) / std(y) : y
    # We enforce M to be larger than 1 since M == 1 is how conduct_vb detects the
    # spurious case and makes the relevant hyperparameter adjustments for a and b
    if M == 1
        throw("M has to be larger than 1")
    end
    if pseudodata_type != "none"
        Xᵖ, X = get_pseudodata([xⁿ yⁿ], R, M; pseudodata_type=pseudodata_type)
        Pxry = -1e10
        # To save time, spurious relationships are not calculated for now
        # Pxry = conduct_vb([xⁿ yⁿ], R, 1, Xᵖ; EPOCHS=EPOCHS)[1][end]
        Px2y = conduct_vb([xⁿ yⁿ], R, M, Xᵖ; EPOCHS=EPOCHS)[1][end]
        Xᵖ[:,:,1], Xᵖ[:,:,2] = Xᵖ[:,:,2], Xᵖ[:,:,1];
        Py2x = conduct_vb([yⁿ xⁿ], R, M, Xᵖ; EPOCHS=EPOCHS)[1][end]
    else
        γ, λ₁, λ₂, a₁, a₂, b₁, b₂ = get_params(parameters_pair, string(R),"^")
        γᵣ, m₁ᵣ, m₂ᵣ, λ₁ᵣ, λ₂ᵣ, a₁ᵣ, a₂ᵣ, b₁ᵣ, b₂ᵣ
        # a_2 and b_2 in the spurious case is now handled within the conduct_vb functino
        Pxry = conduct_vb([xⁿ yⁿ], R, 1; m₁=m₁,m₂=m₂, γ=γ, λ₁=λ₁, λ₂=λ₂, 
            a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂, EPOCHS=EPOCHS)[1][end]
        γ, λ₁, λ₂, a₁, a₂, b₁, b₂ = get_params(parameters_pair, string(R),">")
        Px2y = conduct_vb([xⁿ yⁿ], R, M; m₁=m₁,m₂=m₂, γ=γ, λ₁=λ₁, λ₂=λ₂, 
            a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂, EPOCHS=EPOCHS)[1][end]
        γ, λ₁, λ₂, a₁, a₂, b₁, b₂ = get_params(parameters_pair, string(R),"<")
        Py2x = conduct_vb([yⁿ xⁿ], R, M; m₁=m₁,m₂=m₂, γ=γ, λ₁=λ₁, λ₂=λ₂, 
            a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂, EPOCHS=EPOCHS)[1][end]
    end
    return Px2y, Py2x, Pxry

end


function causal_likelihoods(x::AbstractArray,y::AbstractArray,R::Int,P::Int=1; M=1, 
                            γ=1.0, m₁=0.0, m₂=0.0, λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, norm=true)
    xⁿ = norm ? (x .- mean(x)) / std(x) : x
    yⁿ = norm ? (y .- mean(y)) / std(y) : y
    Pxry = smc_weight([xⁿ yⁿ],R,P;M=1,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₂,a₁=a₁,a₂=a₁,b₁=b₁,b₂=b₁)[1]
    Px2y = smc_weight([xⁿ yⁿ],R,P;M=M,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₂,a₁=a₁,a₂=a₂,b₁=b₁,b₂=b₂)[1]
    Py2x = smc_weight([yⁿ xⁿ],R,P;M=M,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₂,a₁=a₁,a₂=a₂,b₁=b₁,b₂=b₂)[1]
    return Px2y, Py2x, Pxry
end

function vb_causal_likelihoods(x::AbstractArray,y::AbstractArray,R::Int, omit_spurious::Bool=false; M=1, EPOCHS=1,
                            γ=1.0, m₁=0.0, m₂=0.0, λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, norm=true)
    xⁿ = norm ? (x .- mean(x)) / std(x) : x
    yⁿ = norm ? (y .- mean(y)) / std(y) : y
    Pxry = omit_spurious ? 1e-10 : v_bayes([xⁿ yⁿ],R;M=1,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₁,a₁=a₁,a₂=a₁,b₁=b₁,b₂=b₁,EPOCHS=EPOCHS)[1][end]
    Px2y = v_bayes([xⁿ yⁿ],R;M=M,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₂,a₁=a₁,a₂=a₂,b₁=b₁,b₂=b₂,EPOCHS=EPOCHS)[1][end]
    Py2x = v_bayes([yⁿ xⁿ],R;M=M,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₂,a₁=a₁,a₂=a₂,b₁=b₁,b₂=b₂,EPOCHS=EPOCHS)[1][end]
    return Px2y, Py2x, Pxry
end

function get_params(d, R, direction)
    if R in keys(d)
        return d[R][direction]["γᵣ"], d[R][direction]["λ₁ᵣ"], d[R][direction]["λ₂ᵣ"], d[R][direction]["a₁ᵣ"], 
             d[R][direction]["a₂ᵣ"], d[R][direction]["b₁ᵣ"], d[R][direction]["b₂ᵣ"]
    else
        return d[direction]["γ"], d[direction]["λ₁"], d[direction]["λ₂"], d[direction]["a₁"], 
            d[direction]["a₂"], d[direction]["b₁"], d[direction]["b₂"]
    end
end

function get_pseudodata(X::Array{ℜ,2}, R::Int, M::Int; pseudodata_type) where {ℜ <: Real}
    if pseudodata_type == "zeros"
        return zeros(R, M+1, 2), X
    elseif pseudodata_type == "random"
        return randn(R, M+1, 2), X
    elseif pseudodata_type == "kmeans"
        Xᵖ = zeros(R, M+1, 2)
        if R == 1
            Xᵖ[1,:,:] = rand(MvNormal(vec(mean(X, dims=1)), I), M + 1)'
        else
            result = kmeans(X', R)
            for r in 1:R
                Xᵖ[r,:,:] = rand(MvNormal(result.centers[:,r], I), M+1)'
            end
        end
        return Xᵖ, X
    elseif pseudodata_type == "improper"
        ratio = 0.05
        N = size(X)[1]
        T = Int(floor((N*ratio)/R))
        Xᵖ = zeros(R,T,2)
        #gmm = fit!(GMM(n_components=R, kind=:full), X)
        #clusters = argmax(predict_proba(gmm, X),2)
        unchosen = 1:size(X)[1]
        if R == 1 
            chosen = map(x->x[1],randperm(N)[1:T])
            unchosen = setdiff(unchosen,chosen)
            Xᵖ[1,:,:] .= X[chosen,:] 
        else
            result = kmeans(X', R)
            clusters = result.assignments
            for r ∈ 1:R
                cluster_r = findall(x->x==r, clusters)
                chosen = map(x->x[1],randperm(length(cluster_r))[1:T])
                unchosen = setdiff(unchosen,chosen)
                Xᵖ[r,:,:] .= X[chosen,:] 
            end
        end
        X = X[unchosen,:] 
        return Xᵖ, X
    end
end


end
