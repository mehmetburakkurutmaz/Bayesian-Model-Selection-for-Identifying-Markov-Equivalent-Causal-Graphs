include("src/Misc.jl");
include("src/Datasets.jl");
include("src/BayesianCausality.jl");

using .Misc, .BayesianCausality
import .Datasets

using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import Base.Iterators: product
import Random: randperm

using PyPlot
import Base.Filesystem: mkpath

RESULTS_PATH = "./results/tuebingen"
PARAMS_PATH = "./params/tuebingen"

function causal_accuracy(data,T_max=Inf; name="experiment", K=1, P=1, Rs=1:1, α=1.0, m₁=0.0, m₂=0.0,
        λ₁=10.0, λ₂=0.01, a₁=1.0, a₂=10.0, b₁=1.0, b₂=1.0)
    mkpath("$RESULTS_PATH/$name")
    DIRECTION = ["->","<-", "^"]
    
    total_score, total_weight = 0, 0
    spurious_pairs = falses(length(data))
    for (id,pair) ∈ enumerate(data)
    	print(pair[:id],"...\t\t\t")
        T = length(pair[:X])
        perm = randperm(T)[1:Int(min(T,T_max))]
        x, y = pair[:X][perm], pair[:Y][perm]
        likelihoods = reshape([p for R ∈ Rs for p ∈ causal_likelihoods(x, y, R, P; K=K, 
                               α=α, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)],3,:)
                        
        save_json("$RESULTS/$name/$(pair[:id]).json", scores=likelihoods);
        model = DIRECTION[findmax(likelihoods)[2][1]]
        
        total_weight += pair[:weight]
        if model == pair[:relationship]
            total_score += pair[:weight]
        elseif model == "^"
            spurious_pairs[id] = true
            model = DIRECTION[findmax(likelihoods[1:2,:])[2][1]]
            total_score += pair[:weight] * (model == pair[:relationship])
        end
        println("Done.")
    end
    return total_score/total_weight, spurious_pairs
end

println("TUEBINGEN PAIRS")

valid_pairs = setdiff(parse(Int64, ARGS[2]):parse(Int64, ARGS[3]),[52,53,54,55,71])
tuebingen_data = Datasets.tuebingen(valid_pairs)

params = Dict(Symbol("$p") => v for (p,v) ∈ load_json("$PARAMS_PATH/params-$(ARGS[1]).json"))
println("PARAMETERS")
println(params)

@time score, spurious_pairs = causal_accuracy(tuebingen_data; params...)

println("Accuracy = $score")
println("Spurious Pairs => $(valid_pairs[spurious_pairs])")
