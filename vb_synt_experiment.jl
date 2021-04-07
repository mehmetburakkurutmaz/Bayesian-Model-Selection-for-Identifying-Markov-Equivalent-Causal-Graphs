include("src/Misc.jl");
include("src/BayesianCausality.jl");

using .Misc, .BayesianCausality

using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import Base.Iterators: product
import Random: randperm, seed!

using PyPlot
import Base.Filesystem: mkpath

seed!(8457)

RESULTS_PATH = "./results/synthetic/vb"
PARAMS_PATH = "./params"
DATA_PATH = "./data/synthetic"

function causal_accuracy(data, params; name="experiment", EPOCHS=1)
    mkpath("$RESULTS_PATH/$name")
    DIRECTION = ["->","<-","^"]
    
    total_score, total_weight = 0, 0
    for (id,pair) ∈ enumerate(data)
        print(pair[:id],"...\t\t\t")
        pr = params[pair[:dataset]]
        Rs, P, M, γ = pr[:Rs], pr[:P], pr[:M], pr[:γ]
        m₁, m₂, λ₁, λ₂ = pr[:m₁], pr[:m₂], pr[:λ₁], pr[:λ₂]
        a₁, a₂, b₁, b₂ = pr[:a₁], pr[:a₂], pr[:b₁], pr[:b₂]

        T = length(pair[:X])
        perm = randperm(T)
        x, y = pair[:X][perm], pair[:Y][perm]

        likelihoods = reshape([p for R ∈ Rs for p ∈ vb_causal_likelihoods(x, y, R; M=M, EPOCHS=EPOCHS, norm=false,
                               γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)],3,:)  
    
        save_json("$RESULTS_PATH/$name/$(pair[:id]).json", scores=likelihoods);
        model = DIRECTION[findmax(likelihoods)[2][1]]
        
        total_weight += pair[:weight]
        total_score += (model == pair[:relationship])*pair[:weight]
                        
        println(model == pair[:relationship])
    end
    return total_score/total_weight
end

println("SYNTHETIC DATA")

N_PARAMS = 36
N₁, N₂ = parse(Int,ARGS[1]), parse(Int,ARGS[2])

params = [Dict(Symbol(String(p)) => v for (p,v) ∈ load_json("$PARAMS_PATH/params-$pr.json"))
                                      for pr ∈ 1:N_PARAMS]

synt_data = [Dict(Symbol(String(p)) => v for (p,v) ∈ load_json("$DATA_PATH/synt-$id.json")) 
                                    for id ∈ N₁:N₂];

@time score = causal_accuracy(synt_data, params; name="all",EPOCHS=50)
println("Accuracy = $score")