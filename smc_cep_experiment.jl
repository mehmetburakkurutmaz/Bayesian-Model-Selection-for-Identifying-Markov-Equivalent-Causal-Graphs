include("src/Misc.jl");
include("src/Datasets.jl");
include("src/BayesianCausality.jl");

using .Misc, .BayesianCausality
import .Datasets

using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import Base.Iterators: product
import Random: randperm, seed!

using PyPlot
import Base.Filesystem: mkpath

seed!(8457)
RESULTS_PATH = "./results/tuebingen/smc"
PARAMS_PATH = "./params"

function causal_accuracy(data,T_max=Inf; name="experiment", M=1, P=1, Rs=1:1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=10.0, λ₂=0.01, a₁=1.0, a₂=10.0, b₁=1.0, b₂=1.0)
    mkpath("$RESULTS_PATH/$name")
    DIRECTION = ['>','<', '^']
    
    total_score, total_weight = 0, 0
    for (id,pair) ∈ enumerate(data)
    	print(pair[:id],"...\t\t\t")
        T = length(pair[:X])
        perm = randperm(T)[1:Int(min(T,T_max))]
        x, y = pair[:X][perm], pair[:Y][perm]
        likelihoods = reshape([p for R ∈ Rs for p ∈ causal_likelihoods(x, y, R, P; M=M, 
                               γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)],3,:)
                        
        save_json("$RESULTS_PATH/$name/$(pair[:id]).json", scores=likelihoods);
        
        model = DIRECTION[findmax(likelihoods)[2][1]]
        
        total_weight += pair[:weight]
        total_score += (model in pair[:relationship])*pair[:weight]
                        
        println(model in pair[:relationship])
    end
    return total_score/total_weight
end

println("TUEBINGEN PAIRS")

valid_pairs = setdiff(1:100,[52,53,54,55,71])
tuebingen_data = Datasets.tuebingen(valid_pairs)

params = Dict(Symbol("$p") => v for (p,v) ∈ load_json("$PARAMS_PATH/params-$(ARGS[1]).json"))
println("PARAMETERS")
println(params)

@time score = causal_accuracy(tuebingen_data; params...)
println("Accuracy = $score")