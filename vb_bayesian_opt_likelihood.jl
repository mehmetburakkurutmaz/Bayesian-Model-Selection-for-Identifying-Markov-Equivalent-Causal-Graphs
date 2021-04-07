"""
This script is for optimizing hyperparameters from the terminal.

TODO: Named arguments
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

RESULTS_PATH = "./results/tuebingen/vb-hp-opt"

# reading tuebingen data
valid_pairs = setdiff(1:100,[52,53,54,55,71])
tuebingen_data = Datasets.tuebingen(valid_pairs);

# setting arguments
SEED = parse(Int, ARGS[1])
hp_optimization_type = ARGS[2]
T_max = parse(Int, ARGS[3])
n_iterations = parse(Int, ARGS[4])
start_pair_index = parse(Int, ARGS[5])
finish_pair_index = parse(Int, ARGS[6])

vb_hyperparameter_optimization(tuebingen_data[start_pair_index : finish_pair_index], T_max; 
name="hp_opt_T_max_$(T_max)_seed_$SEED/optimization_results/hp_opt_likelihood_$(start_pair_index)_$(finish_pair_index)", 
M=4, P=100, Rs=1:5, EPOCHS=200, hp_optimization_type=hp_optimization_type, n_iterations=200, 
norm=true, SEED=SEED, RESULTS_PATH=RESULTS_PATH, m_zero=true, separate_Rs=true)