function v_bayes(X::Array{ℜ,2}, R::Ƶ=1; M::Ƶ=1, γ::Array{ℜ,1}, m₁::Array{ℜ,1}, m₂::Array{ℜ,2},
        Λ₁::Array{ℜ,1}, Λ₂::Array{ℜ,3}, a₁::Array{ℜ,1}, a₂::Array{ℜ,1}, 
        b₁::Array{ℜ,1}, b₂::Array{ℜ,1}, EPOCHS=1) where {ℜ<:Real, Ƶ<:Integer}
    T, N = size(X)
    ELBO = zeros(EPOCHS)
    
    p = Particle(X, R; M=M)
    p.r .=  T < R ? [1:T...] : R > 1 ? kmeans(Array(X'),R).assignments : ones(Int,T)
    X₁, X₂ = X[:,1], X[:,2]
    
    γʰ, Λ₁ʰ, m₁ʰ, a₁ʰ, b₁ʰ = zeros(R), zeros(R), zeros(R), zeros(R), zeros(R)
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
            b₂ʰ[r] = b₂[r] + (p.Σ_X₂²[r] - m₂ʰ[r,:]'*Λ₂ʰ[r,:,:]*m₂ʰ[r,:,:] + m₂[r,:]'*Λ₂[r,:,:]*m₂[r,:,:])/2
            
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
        if eph > 1 && ELBO[eph] - ELBO[eph-1] < 1e-10
            ELBO = ELBO[1:eph]
            break
        end
    end
    return ELBO, p.r
end
