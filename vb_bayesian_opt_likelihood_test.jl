"""
This script is for optimizing hyperparameters from the terminal.
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

test_method = ARGS[1]
T_max = parse(Float64, ARGS[2])
ratio = parse(Float64, ARGS[3])
M_start = parse(Int64, ARGS[4])
M_end = parse(Int64, ARGS[5])
pseudodata_type = ARGS[6]
imprecise_prior = ARGS[7]
elite_data = ARGS[8]
SEED = parse(Int, ARGS[9])

seed!(SEED)
RESULTS_PATH = "./results/tuebingen/vb-hp-opt"
PARAMS_PATH = "./params";

hp_folder = "$RESULTS_PATH/$(test_method)_$(T_max)_Ms_$(M_start)_$(M_end)_$(pseudodata_type)_seed_$(SEED)"

valid_pairs = setdiff(1:100,[52,53,54,55,71])
tuebingen_data = Datasets.tuebingen(valid_pairs);

if elite_data == "elite_data"
    pair_ids = ["001","007","016","025","034","049","064","072","073","078","086","087","088","096","100"]
    tuebingen_data = create_data(tuebingen_data, pair_ids, 1000);
end

accuracy = causal_accuracy_preset_parameters(tuebingen_data, T_max; hp_folder=hp_folder, Ms=M_start:M_end, 
    P=100, Rs=1:5, EPOCHS=200, seed_no=SEED, test_method=test_method, ratio=ratio, 
    m_zero=true, pseudodata_type=pseudodata_type, imprecise_prior=imprecise_prior)

println(accuracy)

