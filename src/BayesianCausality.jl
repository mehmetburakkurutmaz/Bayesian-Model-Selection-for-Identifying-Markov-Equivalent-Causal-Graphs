module BayesianCausality

include("Misc.jl");

using .Misc

using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import SpecialFunctions: lgamma
import Base.Iterators: product
import Clustering: kmeans
import Random: randperm

export generate, log_Pᵣₓ, log_Qᵣ, log_marginal
export Particle, systematic_resample, multinomial_resample, smc_weight, p_gibbs
export gibbs, v_bayes, dual_em 

function generate(T::Int, R::Int; M=1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0)
    
    θ = rand(Dirichlet(R,γ))
    ρ₁, ρ₂ = rand(Gamma(a₁,1/b₁),R), rand(Gamma(a₂,1/b₂),R)
    w₁ = [rand(Normal(m₁, (λ₁ * ρ₁[r])^(-1/2))) for r ∈ 1:R]
    w₂ = [rand(MultivariateNormal(m₂*ones(M), Diagonal(ones(M)/(λ₂*ρ₂[r])))) for r ∈ 1:R]
    
    rs = rand(Categorical(θ),T)
    X₁ = [rand(Normal(w₁[r], ρ₁[r]^(-1/2))) for r ∈ rs]
    X₂ = [rand(Normal(w₂[r]'*(X₁[t].^(0:M-1)), ρ₂[r]^(-1/2))) for (t,r) ∈ enumerate(rs)]
    return X₁, X₂
end

mutable struct Particle{ℜ <: Real}
    r::Array{Int,1}
    Σ_R::Array{ℜ,1}
    Σ_X₁::Array{ℜ,1}
    Σ_X₁²::Array{ℜ,1}
    Σ_X₂²::Array{ℜ,1}
    Σ_X₂ϕ::Array{ℜ,2}
    Σ_ϕϕᵀ::Array{ℜ,3}
    log_πₜ::ℜ
    
    function Particle(X::Array{ℜ,2}, R::Int; M::Int=1) where {ℜ <: Real}
        T, N = size(X)
        return new{ℜ}(zeros(Int,T), zeros(ℜ,R), zeros(ℜ,R), zeros(ℜ,R),
                      zeros(ℜ,R), zeros(ℜ,R,M), zeros(ℜ,R,M,M), 0.0)
    end
    
    function Particle(r::Array{Int,1}, Σ_R::Array{Int,1}, Σ_X₁::Array{ℜ,1}, Σ_X₁²::Array{ℜ,1},
                      Σ_X₂²::Array{ℜ,1}, Σ_X₂ϕ::Array{ℜ,2}, Σ_ϕϕᵀ::Array{ℜ,3}, log_πₜ::ℜ) where {ℜ <: Real}
        return new{ℜ}(copy(r), copy(Σ_R), copy(Σ_X₁), copy(Σ_X₁²),
                      copy(Σ_X₂²), copy(Σ_X₂ϕ), copy(Σ_ϕϕᵀ), log_πₜ)
    end
    
    function Particle(p::Particle{ℜ}) where {ℜ <: Real}
        return new{ℜ}(copy(p.r), copy(p.Σ_R), copy(p.Σ_X₁), copy(p.Σ_X₁²),
                      copy(p.Σ_X₂²), copy(p.Σ_X₂ϕ), copy(p.Σ_ϕϕᵀ), p.log_πₜ)
    end 
end

function log_Pᵣₓ(p::Particle{ℜ}; γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0) where {ℜ <: Real}
    (R, M), t = size(p.Σ_X₂ϕ), sum(p.Σ_R)
    
    Λ₂ᵣᵗ = Array{Float64}(undef, M, M)
    m₂ᵣᵗ = Array{Float64}(undef, M)
    
    log_P = -t*log(2π) - lgamma(R*γ+t) + lgamma(R*γ)
    
    for r ∈ 1:R
        γᵣᵗ = γ + p.Σ_R[r]
        λ₁ᵣᵗ = λ₁ + p.Σ_R[r]
        m₁ᵣᵗ = (λ₁*m₁ + p.Σ_X₁[r]) / λ₁ᵣᵗ
        a₁ᵣᵗ = a₁ + p.Σ_R[r]/2
        b₁ᵣᵗ = b₁ + (p.Σ_X₁²[r] - λ₁ᵣᵗ*m₁ᵣᵗ*m₁ᵣᵗ + λ₁*m₁*m₁)/2
        
        Λ₂ᵣᵗ .= λ₂*I + p.Σ_ϕϕᵀ[r,:,:]
        m₂ᵣᵗ .= Λ₂ᵣᵗ \ (λ₂*m₂ .+ p.Σ_X₂ϕ[r,:])
        a₂ᵣᵗ = a₂ + p.Σ_R[r] /2
        b₂ᵣᵗ = b₂ + (p.Σ_X₂²[r] - m₂ᵣᵗ'*Λ₂ᵣᵗ*m₂ᵣᵗ + M*λ₂*m₂*m₂)/2
 
        log_P += lgamma(γᵣᵗ) - lgamma(γ) + a₁*log(b₁) - a₁ᵣᵗ*log(b₁ᵣᵗ) +
                 a₂*log(b₂) - a₂ᵣᵗ*log(b₂ᵣᵗ) + lgamma(a₁ᵣᵗ) - lgamma(a₁) +
                 lgamma(a₂ᵣᵗ) - lgamma(a₂) + (log(λ₁) - log(λ₁ᵣᵗ))/2 +
                 (M*log(λ₂) - logdet(Λ₂ᵣᵗ))/2
    end
    return log_P
end

function log_Qᵣ(x₁::ℜ, x₂::ℜ, p::Particle{ℜ}; γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0) where {ℜ <: Real}
    (R, M), t = size(p.Σ_X₂ϕ), sum(p.Σ_R)+1
    
    ϕ = x₁.^(0:M-1)
    Λ₂ᵣᵗ = Array{Float64}(undef, M, M)
    m₂ᵣᵗ = Array{Float64}(undef, M)
    
    log_Q = fill(-t*log(2π) - lgamma(R*γ+t) + lgamma(R*γ), R)
    
    for rᶜ ∈ 1:R
        p.Σ_R[rᶜ] += 1
        p.Σ_X₁[rᶜ] += x₁
        p.Σ_X₁²[rᶜ] += x₁*x₁
        p.Σ_X₂²[rᶜ] += x₂*x₂
        p.Σ_X₂ϕ[rᶜ,:] .+= x₂*ϕ
        p.Σ_ϕϕᵀ[rᶜ,:,:] .+= ϕ*ϕ'
        
        for r ∈ 1:R
            γᵣᵗ = γ + p.Σ_R[r]
            λ₁ᵣᵗ = λ₁ + p.Σ_R[r]
            m₁ᵣᵗ = (λ₁*m₁ + p.Σ_X₁[r]) / λ₁ᵣᵗ
            a₁ᵣᵗ = a₁ + p.Σ_R[r]/2
            b₁ᵣᵗ = b₁ + (p.Σ_X₁²[r] - λ₁ᵣᵗ*m₁ᵣᵗ*m₁ᵣᵗ + λ₁*m₁*m₁)/2
        
            Λ₂ᵣᵗ .= λ₂*I + p.Σ_ϕϕᵀ[r,:,:]
            m₂ᵣᵗ .= Λ₂ᵣᵗ \ (λ₂*m₂ .+ p.Σ_X₂ϕ[r,:])
            a₂ᵣᵗ = a₂ + p.Σ_R[r]/2
            b₂ᵣᵗ = b₂ + (p.Σ_X₂²[r] - m₂ᵣᵗ'*Λ₂ᵣᵗ*m₂ᵣᵗ + M*λ₂*m₂*m₂)/2
        
            log_Q[rᶜ] += lgamma(γᵣᵗ) - lgamma(γ) + a₁*log(b₁) - a₁ᵣᵗ*log(b₁ᵣᵗ) +
                    a₂*log(b₂) - a₂ᵣᵗ*log(b₂ᵣᵗ) + lgamma(a₁ᵣᵗ) - lgamma(a₁) +
                    lgamma(a₂ᵣᵗ) - lgamma(a₂) + (log(λ₁) - log(λ₁ᵣᵗ))/2 +
                    (M*log(λ₂) - logdet(Λ₂ᵣᵗ))/2
        end
        p.Σ_R[rᶜ] -= 1
        p.Σ_X₁[rᶜ] -= x₁
        p.Σ_X₁²[rᶜ] -= x₁*x₁
        p.Σ_X₂²[rᶜ] -= x₂*x₂
        p.Σ_X₂ϕ[rᶜ,:] .-= x₂*ϕ
        p.Σ_ϕϕᵀ[rᶜ,:,:] .-= ϕ*ϕ'
    end
    return log_Q
end

function log_marginal(X::Array{ℜ,2}, R::Ƶ; M=1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0) where {ℜ<:Real, Ƶ<:Integer}
    T, N = size(X)
    
    p = Particle(X, R; M=M)
    ϕ = zeros(M)
    
    log_PX = -Inf
    log_Z = 0.0
    
    for rs ∈ product(fill(1:R,T)...)
        p.Σ_R .= 0.0
        p.Σ_X₁ .= 0.0
        p.Σ_X₁² .= 0.0
        p.Σ_X₂² .= 0.0
        p.Σ_X₂ϕ .= 0.0
        p.Σ_ϕϕᵀ .= 0.0
        for (t,r) ∈ enumerate(rs)
            x₁, x₂ = X[t,1], X[t,2]
            ϕ .= x₁.^(0:M-1)
            
            p.Σ_R[r] += 1.0
            p.Σ_X₁[r] += x₁
            p.Σ_X₁²[r] += x₁*x₁
            p.Σ_X₂²[r] += x₂*x₂
            p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
            p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
        end
        log_Z = log_Pᵣₓ(p; γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, 
                           a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)
        log_PX = logsumexp([log_PX,log_Z])
    end
    return log_PX
end

function systematic_resample(W::Array,Π::Array{Particle{ℜ}},u=rand()) where {ℜ <: Real} # systematic resampling
    P = length(W)
    j = 0
    cum_Wⱼ = cum_Wᵢ = -u
    for i ∈ 1:P
        rᵢ = ceil(cum_Wᵢ + P*W[i]) - ceil(cum_Wᵢ) # number of replicas for ith particle
        for _ ∈ 2.0:rᵢ
            j+=1
            while ceil(cum_Wⱼ+ P*W[j]) - ceil(cum_Wⱼ) > 0 # find next j to be replaced
                cum_Wⱼ += P*W[j]
                j+=1
            end
            cum_Wⱼ += P*W[j]
            
            # replace j by i
            Π[j].r .= Π[i].r
            Π[j].Σ_R .= Π[i].Σ_R
            Π[j].Σ_X₁ .= Π[i].Σ_X₁
            Π[j].Σ_X₁² .= Π[i].Σ_X₁²
            Π[j].Σ_X₂² .= Π[i].Σ_X₂²
            Π[j].Σ_X₂ϕ .= Π[i].Σ_X₂ϕ
            Π[j].Σ_ϕϕᵀ .= Π[i].Σ_ϕϕᵀ
            Π[j].log_πₜ = Π[i].log_πₜ
        end
        cum_Wᵢ += P*W[i]
    end
end

function multinomial_resample(W::Array,Π::Array{Particle{ℜ}}) where {ℜ <: Real} # systematic resampling
    P = length(W)
    j = 0
    replicas = rand(Multinomial(P,W))
    for i ∈ 1:P
        rᵢ = replicas[i] # number of replicas for ith particle
        for _ ∈ 2.0:rᵢ
            j+=1
            while replicas[j] > 0 # find next j to be replaced
                j+=1
            end
            
            # replace j by i
            Π[j].r .= Π[i].r
            Π[j].Σ_R .= Π[i].Σ_R
            Π[j].Σ_X₁ .= Π[i].Σ_X₁
            Π[j].Σ_X₁² .= Π[i].Σ_X₁²
            Π[j].Σ_X₂² .= Π[i].Σ_X₂²
            Π[j].Σ_X₂ϕ .= Π[i].Σ_X₂ϕ
            Π[j].Σ_ϕϕᵀ .= Π[i].Σ_ϕϕᵀ
            Π[j].log_πₜ = Π[i].log_πₜ
        end
    end
end

function smc_weight(X::Array{ℜ,2}, R::Ƶ, P::Ƶ=1; M=1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, resampling=true, adaptive=true) where {ℜ<:Real, Ƶ<:Integer}
    T, N = size(X)
    ESS = P
    
    Π = [Particle(X, R; M=M) for p=1:P]
    log_Z = 0.0

    ϕ, log_W, W = zeros(M), zeros(P), fill(1.0/P,P)
    log_q, q = zeros(R), zeros(R)
    
    for t ∈ 1:T
        x₁, x₂ = X[t,1], X[t,2]
        ϕ .= x₁.^(0:M-1)
        for p ∈ 1:P
            log_q .= log_Qᵣ(x₁, x₂, Π[p]; γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, 
                                           a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)

            log_ν = logsumexp(log_q)
            log_q .-= log_ν
            
            q .= exp.(log_q)
            r = rand(Categorical(q))
            
            log_W[p] += log_ν - Π[p].log_πₜ
            
            Π[p].r[t] = r
            Π[p].Σ_R[r] += 1
            Π[p].Σ_X₁[r] += x₁
            Π[p].Σ_X₁²[r] += x₁*x₁
            Π[p].Σ_X₂²[r] += x₂*x₂
            Π[p].Σ_X₂ϕ[r,:] .+= x₂*ϕ
            Π[p].Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'    
            Π[p].log_πₜ = log_ν + log_q[r]
        end
           
        W .= exp.(log_W .- logsumexp(log_W))
        ESS = 1.0/sum(W .* W)
        
        if resampling && (!adaptive || ESS < P/2)
            systematic_resample(W,Π)
            log_Z += logmeanexp(log_W)
            log_W .= 0.0
        end
        
    end
    log_Z += logmeanexp(log_W)
    p = rand(Categorical(W))
    return log_Z, Π[p]
end

function p_gibbs(X::Array{ℜ,2}, Πᵖ::Particle{ℜ}, P::Ƶ=1; γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, resampling=true) where {ℜ<:Real, Ƶ<:Integer}
    T, N = size(X)
    R, M, ESS = length(Πᵖ.Σ_R), size(Πᵖ.Σ_X₂ϕ)[2], P
    
    Π = [Particle(X, R; M=M) for p=1:P]

    ϕ, log_W, W = zeros(M), zeros(P), fill(1.0/P,P)
    log_q, q = zeros(R), zeros(R)
    
    for t ∈ 1:T
        x₁, x₂ = X[t,1], X[t,2]
        ϕ .= x₁.^(0:M-1)
        for p ∈ 1:P
            log_q .= log_Qᵣ(x₁, x₂, Π[p]; γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, 
                                           a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)
            
            log_ν = logsumexp(log_q)
            log_q .-= log_ν
            
            q .= exp.(log_q)
            r = p==1 ? Πᵖ.r[t] : rand(Categorical(q))

            log_W[p] += log_ν - Π[p].log_πₜ
            
            Π[p].r[t] = r
            Π[p].Σ_R[r] += 1
            Π[p].Σ_X₁[r] += x₁
            Π[p].Σ_X₁²[r] += x₁*x₁
            Π[p].Σ_X₂²[r] += x₂*x₂
            Π[p].Σ_X₂ϕ[r,:] .+= x₂*ϕ
            Π[p].Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
            Π[p].log_πₜ = log_ν + log_q[r]
        end
        
        W .= exp.(log_W .- logsumexp(log_W)) .* (1-1/P)
        W[1] += 1/P
        ESS = 1.0/sum(W .* W)
        
        if resampling && ESS < P/2
            systematic_resample(W,Π)
            log_W .= 0.0
        end
        
    end
    p = rand(Categorical(W))
    return Π[p]
end

function gibbs(X::Array{ℜ,2}, R::Ƶ=1; M=1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, EPOCHS=50) where {ℜ<:Real, Ƶ<:Integer}
    
    T, N = size(X)
    p = Particle(X, R; M=M)
    p.r .= kmeans(Array(X'),R).assignments
    
    γˢ, λ₁ᵣˢ, m₁ᵣˢ, a₁ᵣˢ, b₁ᵣˢ = zeros(R), 0.0, 0.0, 0.0, 0.0
    Λ₂ᵣˢ, m₂ᵣˢ, a₂ᵣˢ, b₂ᵣˢ = zeros(M,M), zeros(M), 0.0, 0.0
    ϕ = zeros(M)
    
    for (r, x₁, x₂) ∈ zip(p.r, X₁, X₂)
        ϕ .= x₁.^(0:M-1)
        p.Σ_R[r] += 1.0
        p.Σ_X₁[r] += x₁
        p.Σ_X₁²[r] += x₁*x₁
        p.Σ_X₂²[r] += x₂*x₂
        p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
        p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
    end

    θ, ρ₁, ρ₂ = zeros(R), zeros(R), zeros(R)
    w₁, w₂ = zeros(R), [zeros(M) for r ∈ 1:R]
    log_q, q = zeros(R), zeros(R)
    
    for eph ∈ 1:EPOCHS
        for r ∈ 1:R
            γˢ[r] = γ + p.Σ_R[r]
            λ₁ᵣˢ = λ₁ + p.Σ_R[r]
            m₁ᵣˢ = (λ₁*m₁ + p.Σ_X₁[r]) / λ₁ᵣˢ
            a₁ᵣˢ = a₁ + p.Σ_R[r]/2
            b₁ᵣˢ = b₁ + (p.Σ_X₁²[r] - λ₁ᵣˢ*m₁ᵣˢ*m₁ᵣˢ + λ₁*m₁*m₁)/2
        
            Λ₂ᵣˢ .= λ₂*I + p.Σ_ϕϕᵀ[r,:,:]
            m₂ᵣˢ .= Λ₂ᵣˢ \ (λ₂*m₂ .+ p.Σ_X₂ϕ[r,:])
            a₂ᵣˢ = a₂ + p.Σ_R[r] /2
            b₂ᵣˢ = b₂ + (p.Σ_X₂²[r] .- m₂ᵣˢ'*Λ₂ᵣˢ*m₂ᵣˢ .+ M*λ₂*m₂*m₂)/2
            
            ρ₁[r], ρ₂[r] = rand(Gamma(a₁ᵣˢ,1/b₁ᵣˢ)), rand(Gamma(a₂ᵣˢ,1/b₂ᵣˢ))
            w₁[r] = rand(Normal(m₁ᵣˢ, (λ₁ᵣˢ * ρ₁[r])^(-1/2)))
            w₂[r] .= m₂ᵣˢ .+ cholesky(Λ₂ᵣˢ).U \ rand(Normal(0, 1/sqrt(ρ₂[r])),M)
        end
        
        θ .= rand(Dirichlet(γˢ))
        
        p.Σ_R .= 0.0
        p.Σ_X₁ .= 0.0
        p.Σ_X₁² .= 0.0
        p.Σ_X₂² .= 0.0
        p.Σ_X₂ϕ .= 0.0
        p.Σ_ϕϕᵀ .= 0.0
        
        for (t,(x₁, x₂)) ∈ enumerate(zip(X₁,X₂))
            ϕ .= x₁.^(0:M-1)
            for r ∈ 1:R
                log_q[r] = log(θ[r]) + (log(ρ₁[r]) + log(ρ₂[r]))/2 
                            - (ρ₁[r]*(x₁ - w₁[r])^2 + ρ₂[r]*(x₂ - ϕ'*w₂[r])^2)/2
            end
            q .= exp.(log_q .- logsumexp(log_q))
            rᵗ = rand(Categorical(q))
            
            p.r[t] = rᵗ
            p.Σ_R[rᵗ] += 1.0
            p.Σ_X₁[rᵗ] += x₁
            p.Σ_X₁²[rᵗ] += x₁*x₁
            p.Σ_X₂²[rᵗ] += x₂*x₂
            p.Σ_X₂ϕ[rᵗ,:] .+= x₂*ϕ
            p.Σ_ϕϕᵀ[rᵗ,:,:] .+= ϕ*ϕ' 
        end
    end
    return p.r
end

function v_bayes(X::Array{ℜ,2}, R::Ƶ; M=1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, EPOCHS=1) where {ℜ<:Real, Ƶ<:Integer}
    
    T, N = size(X)
    ELBO = zeros(EPOCHS)
    
    p = Particle(X, R; M=M)
    p.r .=  T < R ? [1:T...] : R > 1 ? kmeans(Array(X'),R).assignments : ones(Int,T)
    X₁, X₂ = X[:,1], X[:,2]
    
    γʰ, λ₁ʰ, m₁ʰ, a₁ʰ, b₁ʰ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
    Λ₂ʰ, m₂ʰ = [zeros(M,M) for r ∈ 1:R], [zeros(M) for r ∈ 1:R] 
    a₂ʰ, b₂ʰ = zeros(R), zeros(R)
    
    ϕ, log_q, q = zeros(M), zeros(R), zeros(R)
    ρ₁, log_ρ₁, ρ₂, log_ρ₂, log_θ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
    
    for (r, x₁, x₂) ∈ zip(p.r, X₁, X₂)
        ϕ .= x₁.^(0:M-1)
        p.Σ_R[r] += 1.0
        p.Σ_X₁[r] += x₁
        p.Σ_X₁²[r] += x₁*x₁
        p.Σ_X₂²[r] += x₂*x₂
        p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
        p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
    end
    
    for eph ∈ 1:EPOCHS
        for r ∈ 1:R
            γʰ[r] = γ + p.Σ_R[r]
            λ₁ʰ[r] = λ₁ + p.Σ_R[r]
            m₁ʰ[r] = (λ₁*m₁ + p.Σ_X₁[r]) / λ₁ʰ[r]
            a₁ʰ[r] = a₁ + p.Σ_R[r]/2
            b₁ʰ[r] = b₁ + (p.Σ_X₁²[r] - λ₁ʰ[r]*m₁ʰ[r]*m₁ʰ[r] + λ₁*m₁*m₁)/2
        
            Λ₂ʰ[r] .= λ₂*I + p.Σ_ϕϕᵀ[r,:,:]
            m₂ʰ[r] .= Λ₂ʰ[r] \ (λ₂*m₂ .+ p.Σ_X₂ϕ[r,:])
            a₂ʰ[r] = a₂ + p.Σ_R[r] /2
            b₂ʰ[r] = b₂ + (p.Σ_X₂²[r] - m₂ʰ[r]'*Λ₂ʰ[r]*m₂ʰ[r] + M*λ₂*m₂*m₂)/2
            
            log_ρ₁[r] = digamma(a₁ʰ[r]) - log(b₁ʰ[r])
            ρ₁[r] = a₁ʰ[r]/b₁ʰ[r]
            log_ρ₂[r] = digamma(a₂ʰ[r]) - log(b₂ʰ[r])
            ρ₂[r] = a₂ʰ[r]/b₂ʰ[r]
            log_θ[r] = digamma(γʰ[r]) - digamma(R*γ+T)
            
            ELBO[eph] += (γ-γʰ[r])*log_θ[r] + lgamma(γʰ[r])
            ELBO[eph] += (a₁ - a₁ʰ[r])*log_ρ₁[r] - (b₁ - b₁ʰ[r])*ρ₁[r]
            ELBO[eph] -= a₁ʰ[r]*log(b₁ʰ[r]) - lgamma(a₁ʰ[r]) + log(λ₁ʰ[r])/2
            ELBO[eph] -= (ρ₁[r]*λ₁*(m₁ʰ[r] - m₁)^2 + λ₁/λ₁ʰ[r] - 1)/2
            ELBO[eph] += (a₂ - a₂ʰ[r])*log_ρ₂[r] - (b₂ - b₂ʰ[r])*ρ₂[r]
            ELBO[eph] -= a₂ʰ[r]*log(b₂ʰ[r]) - lgamma(a₂ʰ[r]) + logdet(Λ₂ʰ[r])/2
            ELBO[eph] -= (ρ₂[r]*λ₂*(m₂ʰ[r] .- m₂)'*(m₂ʰ[r] .- m₂) + tr(Λ₂ʰ[r] \ (λ₂*I)) - M)/2
        end
        
        ELBO[eph] += lgamma(R*γ) - lgamma(R*γ+T) - R*lgamma(γ) - T*log(2π) 
        ELBO[eph] += R*(a₁*log(b₁) - lgamma(a₁) + log(λ₁)/2)
        ELBO[eph] += R*(a₂*log(b₂) - lgamma(a₂) + M*log(λ₂)/2)
        
        p.Σ_R .= 0.0
        p.Σ_X₁ .= 0.0
        p.Σ_X₁² .= 0.0
        p.Σ_X₂² .= 0.0
        p.Σ_X₂ϕ .= 0.0
        p.Σ_ϕϕᵀ .= 0.0
        
        for (t,(x₁, x₂)) ∈ enumerate(zip(X₁, X₂))
            ϕ .= x₁.^(0:M-1)
            for r ∈ 1:R
                log_q[r] = log_θ[r] 
                log_q[r] += (log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / λ₁ʰ[r])/2
                log_q[r] += (log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r])^2 - ϕ'* (Λ₂ʰ[r] \ ϕ))/2
            end
            
            log_q .-= logsumexp(log_q)
            q .= exp.(log_q)
            
            for r ∈ 1:R                
                p.Σ_R[r] += q[r]
                p.Σ_X₁[r] += q[r]*x₁
                p.Σ_X₁²[r] += q[r]*x₁*x₁
                p.Σ_X₂²[r] += q[r]*x₂*x₂
                p.Σ_X₂ϕ[r,:] .+= q[r]*x₂*ϕ
                p.Σ_ϕϕᵀ[r,:,:] .+= q[r]*ϕ*ϕ'
                
                ELBO[eph] += q[r]*(log_θ[r] - log_q[r])
                ELBO[eph] += q[r]*(log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / λ₁ʰ[r])/2
                ELBO[eph] += q[r]*(log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r])^2 - ϕ'* (Λ₂ʰ[r] \ ϕ))/2
            end
            _, p.r[t] = findmax(log_q)
        end
        if eph > 1 && ELBO[eph] - ELBO[eph-1] < 1e-10
            ELBO = ELBO[1:eph]
            break
        end
    end
    return ELBO, p.r
end

function v_bayes(X::Array{ℜ,2}, R::Ƶ, full::Bool; M::Ƶ=1, γ::Array{ℜ,1}, m₁::Array{ℜ,1}, m₂::Array{ℜ,2},
        Λ₁::Array{ℜ,1}, Λ₂::Array{ℜ,3}, a₁::Array{ℜ,1}, a₂::Array{ℜ,1}, 
        b₁::Array{ℜ,1}, b₂::Array{ℜ,1}, EPOCHS=1) where {ℜ<:Real, Ƶ<:Integer}
    T, N = size(X)
    ELBO = zeros(EPOCHS)
    
    p = Particle(X, R; M=M)
    p.r .=  T < R ? [1:T...] : R > 1 ? kmeans(Array(X'),R).assignments : ones(Int,T)
    X₁, X₂ = X[:,1], X[:,2]
    
    γʰ = zeros(R)
    Λ₁ʰ, m₁ʰ = zeros(R), zeros(R)
    a₁ʰ, b₁ʰ = zeros(R), zeros(R)
    Λ₂ʰ, m₂ʰ = zeros(R,M,M), zeros(R,M) 
    a₂ʰ, b₂ʰ = zeros(R), zeros(R)
    
    ϕ, log_q, q = zeros(M), zeros(R), zeros(R)
    ρ₁, log_ρ₁, ρ₂, log_ρ₂, log_θ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
    
    for (r, x₁, x₂) ∈ zip(p.r, X₁, X₂)
        ϕ .= x₁.^(0:M-1)
        p.Σ_R[r] += 1.0
        p.Σ_X₁[r] += x₁
        p.Σ_X₁²[r] += x₁*x₁
        p.Σ_X₂²[r] += x₂*x₂
        p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
        p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
    end
    
    for eph ∈ 1:EPOCHS
        for r ∈ 1:R
            γʰ[r] = γ[r] + p.Σ_R[r]
            Λ₁ʰ[r] = Λ₁[r] + p.Σ_R[r]
            m₁ʰ[r] = (Λ₁[r]*m₁[r] + p.Σ_X₁[r]) / Λ₁ʰ[r]
            a₁ʰ[r] = a₁[r] + p.Σ_R[r]/2
            b₁ʰ[r] = b₁[r] + (p.Σ_X₁²[r] - Λ₁ʰ[r]*m₁ʰ[r]*m₁ʰ[r] + Λ₁[r]*m₁[r]*m₁[r])/2
        
            Λ₂ʰ[r,:,:] .= Λ₂[r,:,:] .+ p.Σ_ϕϕᵀ[r,:,:]
            m₂ʰ[r,:] .= Λ₂ʰ[r,:,:] \ (Λ₂[r,:,:]*m₂[r,:] .+ p.Σ_X₂ϕ[r,:])
            a₂ʰ[r] = a₂[r] + p.Σ_R[r]/2
            b₂ʰ[r] = b₂[r] + (p.Σ_X₂²[r] - m₂ʰ[r,:]'*Λ₂ʰ[r,:,:]*m₂ʰ[r,:] + m₂[r,:]'*Λ₂[r,:,:]*m₂[r,:])/2
            
            log_ρ₁[r] = digamma(a₁ʰ[r]) - log(b₁ʰ[r])
            ρ₁[r] = a₁ʰ[r]/b₁ʰ[r]
            log_ρ₂[r] = digamma(a₂ʰ[r]) - log(b₂ʰ[r])
            ρ₂[r] = a₂ʰ[r]/b₂ʰ[r]
            log_θ[r] = digamma(γʰ[r]) - digamma(sum(γ)+T)
            
            ELBO[eph] += (γ[r]-γʰ[r])*log_θ[r]
            
            ELBO[eph] += a₁[r]*log(b₁[r]) - a₁ʰ[r]*log(b₁ʰ[r])
            ELBO[eph] += lgamma(a₁ʰ[r]) - lgamma(a₁[r])
            ELBO[eph] += (log(Λ₁[r]) - log(Λ₁ʰ[r]))/2
            ELBO[eph] += (1 - Λ₁[r]/Λ₁ʰ[r])/2
            ELBO[eph] += (a₁[r] - a₁ʰ[r])*log_ρ₁[r] - (b₁[r] - b₁ʰ[r])*ρ₁[r]
            ELBO[eph] -= (ρ₁[r]*Λ₁[r]*(m₁ʰ[r] - m₁[r])^2)/2
                
            ELBO[eph] += a₂[r]*log(b₂[r]) - a₂ʰ[r]*log(b₂ʰ[r])
            ELBO[eph] += lgamma(a₂ʰ[r]) - lgamma(a₂[r])
            ELBO[eph] += (logdet(Λ₂[r,:,:]) - logdet(Λ₂ʰ[r,:,:]))/2
            ELBO[eph] += (M - tr(Λ₂ʰ[r,:,:] \ Λ₂[r,:,:]))/2
            ELBO[eph] += (a₂[r] - a₂ʰ[r])*log_ρ₂[r] - (b₂[r] - b₂ʰ[r])*ρ₂[r]
            ELBO[eph] -= ρ₂[r]*((m₂ʰ[r,:] - m₂[r,:])'*Λ₂[r,:,:]*(m₂ʰ[r,:] - m₂[r,:]))/2
        end
        
        ELBO[eph] += lgamma(sum(γ)) - sum(lgamma,γ)
        ELBO[eph] -= lgamma(sum(γ)+T) - sum(lgamma,γʰ)
        
        p.Σ_R .= 0.0
        p.Σ_X₁ .= 0.0
        p.Σ_X₁² .= 0.0
        p.Σ_X₂² .= 0.0
        p.Σ_X₂ϕ .= 0.0
        p.Σ_ϕϕᵀ .= 0.0
        
        for (t,(x₁, x₂)) ∈ enumerate(zip(X₁, X₂))
            ϕ .= x₁.^(0:M-1)
            for r ∈ 1:R
                log_q[r] = log_θ[r] 
                log_q[r] += (log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / Λ₁ʰ[r])/2
                log_q[r] += (log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r,:])^2 - ϕ'* (Λ₂ʰ[r,:,:] \ ϕ))/2
            end
            
            log_q .-= logsumexp(log_q)
            q .= exp.(log_q)
            
            for r ∈ 1:R                
                p.Σ_R[r] += q[r]
                p.Σ_X₁[r] += q[r]*x₁
                p.Σ_X₁²[r] += q[r]*x₁*x₁
                p.Σ_X₂²[r] += q[r]*x₂*x₂
                p.Σ_X₂ϕ[r,:] .+= q[r]*x₂*ϕ
                p.Σ_ϕϕᵀ[r,:,:] .+= q[r]*ϕ*ϕ'
                
                ELBO[eph] += q[r]*(log_θ[r] - log_q[r])
                ELBO[eph] += q[r]*(log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / Λ₁ʰ[r])/2
                ELBO[eph] += q[r]*(log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r,:])^2 - ϕ'* (Λ₂ʰ[r,:,:] \ ϕ))/2
            end
            _, p.r[t] = findmax(log_q)
        end
                
        ELBO[eph] -= T*log(2π)
        if eph > 1 && ELBO[eph] - ELBO[eph-1] < 1e-6
            ELBO = ELBO[1:eph]
            break
        end
    end
    return ELBO, p.r
end


function dual_em(X::Array{ℜ,2}, R::Ƶ=1; M=1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=1.0, λ₂=1.0, a₁=1.0, a₂=1.0, b₁=1.0, b₂=1.0, EPOCHS=1) where {ℜ<:Real, Ƶ<:Integer}
    
    T, N = size(X)
    
    p = Particle(X, R; M=M)
    p.r .=  T < R ? [1:T...] : R > 1 ? kmeans(Array(X'),R).assignments : ones(Int,T)
    X₁, X₂ = X[:,1], X[:,2]
    
    γʰ, λ₁ʰ, m₁ʰ, a₁ʰ, b₁ʰ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
    Λ₂ʰ, m₂ʰ = [zeros(M,M) for r ∈ 1:R], [zeros(M) for r ∈ 1:R] 
    a₂ʰ, b₂ʰ = zeros(R), zeros(R)
    
    ϕ, log_q = zeros(M), zeros(R)
    ρ₁, log_ρ₁, ρ₂, log_ρ₂, log_θ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
    
    for (r, x₁, x₂) ∈ zip(p.r, X₁, X₂)
        ϕ .= x₁.^(0:M-1)
        p.Σ_R[r] += 1.0
        p.Σ_X₁[r] += x₁
        p.Σ_X₁²[r] += x₁*x₁
        p.Σ_X₂²[r] += x₂*x₂
        p.Σ_X₂ϕ[r,:] .+= x₂*ϕ
        p.Σ_ϕϕᵀ[r,:,:] .+= ϕ*ϕ'
    end
    
    for eph ∈ 1:EPOCHS
        for r ∈ 1:R
            γʰ[r] = γ + p.Σ_R[r]
            λ₁ʰ[r] = λ₁ + p.Σ_R[r]
            m₁ʰ[r] = (λ₁*m₁ + p.Σ_X₁[r]) / λ₁ʰ[r]
            a₁ʰ[r] = a₁ + p.Σ_R[r]/2
            b₁ʰ[r] = b₁ + (p.Σ_X₁²[r] - λ₁ʰ[r]*m₁ʰ[r]*m₁ʰ[r] + λ₁*m₁*m₁)/2
        
            Λ₂ʰ[r] .= λ₂*I + p.Σ_ϕϕᵀ[r,:,:]
            m₂ʰ[r] .= Λ₂ʰ[r] \ (λ₂*m₂ .+ p.Σ_X₂ϕ[r,:])
            a₂ʰ[r] = a₂ + p.Σ_R[r] /2
            b₂ʰ[r] = b₂ + (p.Σ_X₂²[r] - m₂ʰ[r]'*Λ₂ʰ[r]*m₂ʰ[r] + M*λ₂*m₂*m₂)/2
            
            log_ρ₁[r] = digamma(a₁ʰ[r]) - log(b₁ʰ[r])
            ρ₁[r] = a₁ʰ[r]/b₁ʰ[r]
            log_ρ₂[r] = digamma(a₂ʰ[r]) - log(b₂ʰ[r])
            ρ₂[r] = a₂ʰ[r]/b₂ʰ[r]
            log_θ[r] = digamma(γʰ[r]) - digamma(R*γ+T)
        end
        
        p.Σ_R .= 0.0
        p.Σ_X₁ .= 0.0
        p.Σ_X₁² .= 0.0
        p.Σ_X₂² .= 0.0
        p.Σ_X₂ϕ .= 0.0
        p.Σ_ϕϕᵀ .= 0.0
        
        for (t,(x₁, x₂)) ∈ enumerate(zip(X₁, X₂))
            ϕ .= x₁.^(0:M-1)
            for r ∈ 1:R
                log_q[r] = log_θ[r] 
                log_q[r] += (log_ρ₁[r] - ρ₁[r]*(x₁ - m₁ʰ[r])^2 - 1.0 / λ₁ʰ[r])/2
                log_q[r] += (log_ρ₂[r] - ρ₂[r]*(x₂ - ϕ'*m₂ʰ[r])^2 - ϕ'* (Λ₂ʰ[r] \ ϕ))/2
            end
            
            log_q .-= logsumexp(log_q)
            _, p.r[t] = findmax(log_q)
                         
            p.Σ_R[p.r[t]] += 1.0
            p.Σ_X₁[p.r[t]] += x₁
            p.Σ_X₁²[p.r[t]] += x₁*x₁
            p.Σ_X₂²[p.r[t]] += x₂*x₂
            p.Σ_X₂ϕ[p.r[t],:] .+= x₂*ϕ
            p.Σ_ϕϕᵀ[p.r[t],:,:] .+= ϕ*ϕ'
        end
    end
    return p.r
end

end