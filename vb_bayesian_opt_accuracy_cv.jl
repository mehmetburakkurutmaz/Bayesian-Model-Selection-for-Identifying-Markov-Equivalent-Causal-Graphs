"""
This is a script for CV using accuracy. MxK CV conducted according to accuracy on the training folds.
"""

include("src/BayesianCausalityOptimization.jl");
include("src/BayesianCausality.jl");
include("src/Datasets.jl");
include("src/Misc.jl");

using .Misc, .BayesianCausality, .BayesianCausalityOptimization
using .Datasets
using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import Base.Iterators: product
import Random: randperm, seed!
using BayesOpt
import Base.Iterators: enumerate
import Base.Filesystem: mkpath

function split_data(N,K)
    splits = fill(Int(floor(N/K)),K)
    splits[1:(N%K)] .+= 1
    return [0; cumsum(splits)]
end
"""
function bayesian_optimization(data, T_max; Rs, EPOCHS, M, P, n_iterations)
    config = ConfigParameters()        # calls initialize_parameters_to_default of the C API
    config.n_iterations = n_iterations;
    set_kernel!(config, "kMaternARD5")  # calls set_kernel of the C API
    config.sc_type = SC_MAP;
    f((γ, λ₁, λ₂, a₁, a₂, b₁, b₂)) = -vb_causal_accuracy(data, T_max; M=M, P=P, Rs=Rs, γ=exp(γ), m₁=0.0, m₂=0.0, λ₁=exp(λ₁), λ₂=exp(λ₂), a₁=exp(a₁), a₂=exp(a₂), b₁=exp(b₁), b₂=exp(b₂), EPOCHS=EPOCHS)
    lowerbound = [log(.0001), log(.0001), log(.0001), log(.0001), log(.0001), log(.0001), log(.0001)]; upperbound = [log(10000.), log(10000.), log(10000.), log(10000.), log(10000.), log(10000.), log(10000.)];
    @time optimizer, optimum = bayes_optimization(f, lowerbound, upperbound, config)
    return optimizer, optimum
end;
"""
function causal_accuracy_opt_cv(data,T_max=Inf; name="experiment", M=1, P=1, Rs=1:1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=10.0, λ₂=0.01, a₁=1.0, a₂=10.0, b₁=1.0, b₂=1.0, EPOCHS=1, tr_acc= 0.)
    """
    Differences:
    - results not printed
    - saving parameters
    """
    mkpath("$RESULTS_PATH/$name")
    DIRECTION = ['>','<', '^']
    params = Dict("seed_no"=> seed_no, "name"=> name, "M"=> M, "P"=> P, 
        "Rs"=> Rs, "γ"=> γ, "m₁"=> m₁, "m₂"=> m₂,
        "λ₁"=> λ₁, "λ₂"=> λ₂, "a₁"=> a₁, "a₂"=> a₂, "b₁"=> b₁, 
        "b₂"=> b₂, "EPOCHS"=> EPOCHS, "T_max"=> T_max, "tr_acc"=>tr_acc)
    save_json("$RESULTS_PATH/$(name)-params.json", params=params);
    total_score, total_weight = 0, 0
    println(name)
    for (id,pair) ∈ enumerate(data)
    	print(pair[:id],"...\t\t\t")
        T = length(pair[:X])
        perm = randperm(T)[1:Int(min(T,T_max))]
        x, y = pair[:X][perm], pair[:Y][perm]
        likelihoods = reshape([p for R ∈ Rs for p ∈ vb_causal_likelihoods(x, y, R; M=M, EPOCHS=EPOCHS,
                               γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂,
                               a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)],3,:)                
        save_json("$RESULTS_PATH/$name/$(pair[:id]).json", scores=likelihoods);
        
        model = DIRECTION[findmax(nanmax(likelihoods,2)[1:2])[2]]
        total_score += pair[:weight] * (model in pair[:relationship])
        total_weight += pair[:weight]
        println(model in pair[:relationship])

    end
    return total_score/total_weight
                    end;
                    
valid_pairs = setdiff(1:100,[52,53,54,55,71])
tuebingen_data = Datasets.tuebingen(valid_pairs);
N_data = length(valid_pairs);
                    
seed_no = parse(Int, ARGS[1])
seed!(seed_no)
RESULTS_PATH = string("./results/tuebingen/vb-seed/seed-",seed_no)
M_repeat_no, K = 1, 3
performance = zeros(M_repeat_no,K);
Rs = 1:5
EPOCHS = 200
T_max = 1000
M = 4
P = 100
m₁, m₂ = 0., 0.
n_iterations = 200

splits = split_data(length(valid_pairs),K);

for m ∈ 1:M_repeat_no    
    perm_pairs = randperm(N_data)
    for k ∈ 1:K
        name = string("m_",m,"_k_",k)
        test = perm_pairs[splits[k]+1:splits[k+1]]
        train = setdiff(perm_pairs,test)
        train_data = tuebingen_data[train]
        test_data = tuebingen_data[test]
        optimizer, tr_acc = bayesian_optimization_accuracy(train_data, T_max;Rs=Rs, 
                                                    EPOCHS=EPOCHS, M=M, P=P, n_iterations=n_iterations)
        γ, λ₁, λ₂, a₁, a₂, b₁, b₂ = exp.(optimizer)
        accuracy = causal_accuracy_opt_cv(test_data, T_max; name=name, M=M, P=P, Rs=Rs, γ=γ,
            m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂, EPOCHS=EPOCHS, tr_acc=tr_acc)
        performance[m,k] = accuracy
        save_json("$RESULTS_PATH/$(name)-split.json", split=Dict("train"=>train, "test"=>test));
    end
end
save_json("$RESULTS_PATH/accuracy.json", performance=performance);