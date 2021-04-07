module BayesianCausalityOptimization

include("Misc.jl");
include("Datasets.jl");
include("BayesianCausality.jl");

using .Misc, .BayesianCausality
import .Datasets
using Statistics, Distributions
using SpecialFunctions, LinearAlgebra
import Base.Iterators: product
import Random: randperm, seed!
using BayesOpt
import Base.Iterators: enumerate

export bayesian_optimization, create_data, causal_accuracy_preset_parameters, vb_hyperparameter_optimization

function get_indices(tuebingen_data, pair_ids)
    pair_indices = zeros(Int,  length(pair_ids))
    current_id_index = 1
    for (i, data) in enumerate(tuebingen_data)
        if data[:id] == string("pair0",pair_ids[current_id_index])
            pair_indices[current_id_index] = Int(i)
            current_id_index += 1
        end
    end
    return pair_indices
end;

function create_data(tuebingen_data, pair_ids, max_N=1000; reweight=true)
    pair_indices = get_indices(tuebingen_data, pair_ids);
    selected_data = deepcopy(tuebingen_data[pair_indices])
    data_indices =  fill(Float64[], length(pair_indices))
    for data in selected_data
        if length(data[:X]) > max_N
            data[:indices] = randperm(length(data[:X]))[1:max_N]
        else
            data[:indices] = [i for i in 1:length(data[:X])]
        end
        if reweight
            data[:weight] = 1
        end
    end
    return selected_data
end;

"""
function smc_causal_accuracy(data,T_max=Inf; M=1, P=1, Rs=1:1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=10.0, λ₂=0.01, a₁=1.0, a₂=10.0, b₁=1.0, b₂=1.0, hold_sample_constant=false)
    
    1- results not written as JSON
    2- hold_sample_constant is available if desired
    3- println's removed, name removed
    
    DIRECTION = ['>','<', '^']
    total_score, total_weight = 0, 0
    for (id,pair) ∈ enumerate(data)
        if hold_sample_constant
            perm = [data[:indices]] 
        else
            T = length(pair[:X])
            perm = randperm(T)[1:Int(min(T,T_max))]
        end
        x, y = pair[:X][perm], pair[:Y][perm]
        likelihoods = reshape([p for R ∈ Rs for p ∈ causal_likelihoods(x, y, R, P; M=M, 
                               γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂)],3,:)             
        model = DIRECTION[findmax(likelihoods)[2][1]]
        total_weight += pair[:weight]
        print(model)
        total_score += (model in pair[:relationship])*pair[:weight]
    end
    return total_score/total_weight
end;
"""

function vb_causal_accuracy(data,T_max=Inf; M=1, P=1, Rs=1:1, γ=1.0, m₁=0.0, m₂=0.0,
        λ₁=10.0, λ₂=0.01, a₁=1.0, a₂=10.0, b₁=1.0, b₂=1.0, EPOCHS=1, hold_sample_constant=false, m_zero)
    """
    1- results not written as JSON
    2- hold_sample_constant is available if desired
    3- println's removed, name removed
    4- nanmax error handling
    """
    DIRECTION = ['>','<', '^']
    total_score, total_weight = 0, 0
    for (id,pair) ∈ enumerate(data)
        if hold_sample_constant
            perm = [data[:indices]] 
        else
            T = length(pair[:X])
            perm = randperm(T)[1:Int(min(T,T_max))]
        end        
        x, y = pair[:X][perm], pair[:Y][perm]
        likelihoods = reshape([p for R ∈ Rs for p ∈ vb_causal_likelihoods(x, y, R; M=M, EPOCHS=EPOCHS,
                               γ=γ, m₁=m₁, m₂=m₂, λ₁=λ₁, λ₂=λ₂, a₁=a₁, a₂=a₂, b₁=b₁, b₂=b₂, m_zero=m_zero)],3,:)                        
        if length(likelihoods) - sum(isnan.(likelihoods)) == 0 
            model = NaN
        else
            model = DIRECTION[findmax(nanmax(likelihoods,2)[1:2])[2]]
        end
        total_score += pair[:weight] * (model in pair[:relationship])
        total_weight += pair[:weight]
    end
    return total_score/total_weight
end;                                        
        
function bayesian_optimization_accuracy(data, T_max; Rs, EPOCHS, M, P, n_iterations)
    config = ConfigParameters()        # calls initialize_parameters_to_default of the C API
    config.n_iterations = n_iterations;
    set_kernel!(config, "kMaternARD5")  # calls set_kernel of the C API
    config.sc_type = SC_MAP;
    f((γ, λ₁, λ₂, a₁, a₂, b₁, b₂)) = -vb_causal_accuracy(data, T_max; M=M, P=P, Rs=Rs, γ=exp(γ), m₁=0.0, m₂=0.0, λ₁=exp(λ₁), λ₂=exp(λ₂), a₁=exp(a₁), a₂=exp(a₂), b₁=exp(b₁), b₂=exp(b₂), EPOCHS=EPOCHS)
    lowerbound = [log(.0001), log(.0001), log(.0001), log(.0001), log(.0001), log(.0001), log(.0001)]; upperbound = [log(10000.), log(10000.), log(10000.), log(10000.), log(10000.), log(10000.), log(10000.)];
    @time optimizer, optimum = bayes_optimization(f, lowerbound, upperbound, config)
    return optimizer, optimum
    end;
                    

function causal_accuracy_preset_parameters(data, T_max=Inf; hp_folder, Ms=2:5, P=1, Rs=1:1, EPOCHS=1, seed_no=100, test_method="vb", norm = true, ratio=1.0, pseudodata_type="none", m_zero, imprecise_prior="imprecise_prior")
    """
    Testing accuracy using prerecorded parameters (e.g. in unconstrained or pseudodata formulations)
    """
    function create_empty_results_dict(data)
        """This function cretaes an empty dictionary for storing results"""
        return Dict(:pair_ids=>["" for i in 1:length(data)], 
                    :accuracy=>0.0, 
                    :pair_weights=>zeros(length(data)),
                    :pair_correct=>[false for i in 1:length(data)],
                    :pair_scores=>zeros(length(data)),
                    :models_decision=>['0' for i in 1:length(data)],
                    :models_actual=>["0" for i in 1:length(data)]) 
    end      
    no_Rs = length(Rs)
    no_Ms = length(Ms)
    seed!(seed_no)
    DIRECTION = ['>','<', '^']   
    OUTPUT_PATH = (ratio==1.0) ? "$(hp_folder)/test_results/test_results_$(test_method)_Tmax_$(T_max)_seed_$(seed_no)" : "$(hp_folder)/test_results/test_results_$(test_method)_ratio_$(ratio)_seed_$(seed_no)" 
    mkpath(OUTPUT_PATH); mkpath("$(OUTPUT_PATH)/pair_results");
    if imprecise_prior == "imprecise_prior"
        prep_imprecise_prior(data, hp_folder, Rs)
    end
    # If we do pseudodata, then create an empty dictionary for parameters, else read the parameters
    parameters = pseudodata_type != "none" ? Dict(pair[:id]=>Dict() for pair in data) : load_json(string(hp_folder,"/parameters.json"))["parameters"]
    results = create_empty_results_dict(data)
    total_score, total_weight = 0, 0    
    for (id,pair) ∈ enumerate(data)
        print(pair[:id],"...\t\t\t")
        T = length(pair[:X]); 
        T_max_actual = ratio < 1.0 ? round(T*ratio) : round(T_max); # If we have a ratio less than 1, it determines the T_max
        perm = randperm(T)[1:Int(min(T,T_max_actual))]
        x, y = pair[:X][perm], pair[:Y][perm]
        likelihood_labels = reshape(["$(p)_M_$(M)_R_$(R)" for M ∈ Ms for R ∈ Rs for p ∈ DIRECTION],length(DIRECTION),no_Rs,:)
        likelihoods = reshape([p for M ∈ Ms for R ∈ Rs for p ∈ vb_causal_likelihoods(x, y, R; M=M, EPOCHS=EPOCHS, 
           parameters_pair=parameters[pair[:id]], norm=norm, m_zero=m_zero, 
           pseudodata_type=pseudodata_type)],length(DIRECTION),no_Rs,:)
        save_json("$(OUTPUT_PATH)/pair_results/$(pair[:id]).json", scores=Dict(zip(likelihood_labels, likelihoods)));
        model = DIRECTION[findmax(vec(nanmax(likelihoods,dims=(2,3)))[1:2])[2]]
        total_score += pair[:weight] * (model in pair[:relationship])
        total_weight += pair[:weight]
        # Write pair results
        results[:pair_ids][id], results[:pair_correct][id] = pair[:id], (model in pair[:relationship])       
        results[:models_decision][id], results[:models_actual][id]  = model, pair[:relationship]
        results[:pair_weights][id], results[:pair_scores][id] = pair[:weight], pair[:weight] * (model in pair[:relationship])
        println(model in pair[:relationship])
    end
    results[:accuracy] = total_score/total_weight
    test_metaparameters = Dict("Ms"=>Ms, "EPOCHS"=>EPOCHS, "P"=>P, "T_max"=>T_max, 
        "Rs"=>Rs, "hp_folder"=>hp_folder, "SEED"=>seed_no, "norm"=>norm, "pairs_included"=>results[:pair_ids],
        "ratio"=>ratio, "pseudodata_type"=>pseudodata_type)
    if (pseudodata_type == "none") & m_zero
        test_metaparameters["m_zero"] = true
    end
    save_json("$(OUTPUT_PATH)/test_metaparameters.json", test_metaparameters=test_metaparameters);
    save_json("$(OUTPUT_PATH)/results.json", results=results);
    return total_score/total_weight
end

function prep_imprecise_prior(data, hp_folder, Rs)
    DIRECTION = ['>','<', '^']
    parameters = Dict()
    for (id, pair) ∈ enumerate(data)
        parameters[pair[:id]] = Dict()
        for R in Rs
            parameters[pair[:id]][R] = Dict()
            for direction in DIRECTION
                parameters[pair[:id]][R][direction] = Dict()
                pd = parameters[pair[:id]][R][direction]
                pd[:γᵣ] = 1.
                pd[:λ₁ᵣ] = .001
                pd[:λ₂ᵣ] = .001
                pd[:a₁ᵣ] = 1.
                pd[:a₂ᵣ] = 1.
                pd[:b₁ᵣ] = 1.
                pd[:b₂ᵣ] = 1.
            end
        end
    end
    save_json("$(hp_folder)/parameters.json", parameters=parameters);
end
function prepare_empty_parameter_dictionary(Rs::UnitRange{Int64}, data::Array{Dict{Symbol,Any},1}, separate_Rs=true)
    DIRECTION = ['>','<', '^']
    resulting_parameters = Dict()
    likelihoods = Dict()
    for (id, pair) ∈ enumerate(data)
        resulting_parameters[pair[:id]] = Dict()
        likelihoods[pair[:id]] = Dict()
        for R in Rs
            resulting_parameters[pair[:id]][R] = Dict()
            likelihoods[pair[:id]][R] = Dict()
            for direction in DIRECTION
                resulting_parameters[pair[:id]][R][direction] = Dict()
                likelihoods[pair[:id]][R][direction] = Dict()
            end
        end
    end
    return resulting_parameters, likelihoods
end
                                                                
function prepare_empty_parameter_dictionary(Rs::UnitRange{Int64}, data::Array{Dict{Symbol,Any},1}, separate_Rs=true)
    DIRECTION = ['>','<', '^']
    resulting_parameters = Dict()
    likelihoods = Dict()
    for (id, pair) ∈ enumerate(data)
        resulting_parameters[pair[:id]] = Dict()
        likelihoods[pair[:id]] = Dict()
        for R in Rs
            resulting_parameters[pair[:id]][R] = Dict()
            likelihoods[pair[:id]][R] = Dict()
            for direction in DIRECTION
                resulting_parameters[pair[:id]][R][direction] = Dict()
                likelihoods[pair[:id]][R][direction] = Dict()
            end
        end
    end
    return resulting_parameters, likelihoods
end
                                                            
function bayesian_optimization_likelihood(xⁿ, yⁿ; Rs, EPOCHS, M, P, n_iterations, unconstrained, m_zero)
    """
    Conducting hyperparameter selection based on marginal likelihood optimization -- unconstrained hyperparameters
    """

    function get_f(M)
        function v_bayes_wrapper(X, Rs; M,γ,m₁,m₂,λ₁,λ₂,a₁,a₂,b₁,b₂,EPOCHS)
            results = zeros(length(Rs))
            for R in Rs
                results[R] = v_bayes(X,R;M=M,γ=γ,m₁=m₁,m₂=m₂,λ₁=λ₁,λ₂=λ₂,
                    a₁=a₁,a₂=a₂,b₁=b₁,b₂=b₂,EPOCHS=EPOCHS)[1][end]
            end
            return max(results...)
        end
        if M == 1
            f((γ, λ₁, λ₂, a₁, a₂, b₁, b₂)) = -v_bayes_wrapper([xⁿ yⁿ],Rs;M=1,γ=exp(γ),m₁=m₁,m₂=m₂,λ₁=exp(λ₁),λ₂=exp(λ₂),
                a₁=exp(a₁),a₂=exp(a₁),b₁=exp(b₁),b₂=exp(b₁),EPOCHS=EPOCHS)[1][end]
            return f
        elseif M > 1
            g((γ, λ₁, λ₂, a₁, a₂, b₁, b₂)) = -v_bayes_wrapper([xⁿ yⁿ],Rs;M=M,γ=exp(γ),m₁=m₁,m₂=m₂,λ₁=exp(λ₁),λ₂=exp(λ₂),
                a₁=exp(a₁),a₂=exp(a₂),b₁=exp(b₁),b₂=exp(b₂),EPOCHS=EPOCHS)[1][end]       
            return g
        end
    end                                                                
    if m_zero
        m₁, m₂ = 0., 0.
    end
    config = ConfigParameters(); config.n_iterations = n_iterations; config.verbose_level = 0; set_kernel!(config, "kMaternARD5"); config.sc_type = SC_MAP;
    f = get_f(M)
    param_keys = ["γ", "λ₁", "λ₂", "a₁", "a₂", "b₁", "b₂"]
    lowerbound = [log(.0001) for i in 1:7]; upperbound = [log(10000.) for i in 1:7];
    optimization_unfinished = true; global optimizer = []; global optimum = 0.;
    while optimization_unfinished
        @time optimizer, optimum = bayes_optimization(f, lowerbound, upperbound, config)
        optimization_unfinished = sum(convert(Array{Int64}, optimizer .== 0.00000))>0
        if optimization_unfinished
            println("optimization unsuccessful, reattempting")
        end
    end
    return Dict(zip(param_keys, optimizer)), optimum
    end;
                                                            
function bayesian_optimization_likelihood(xⁿ, yⁿ; R, EPOCHS, M, P, n_iterations, unconstrained, m_zero, separate_Rs=true)
    """
    Conducting hyperparameter selection based on marginal likelihood optimization -- unconstrained hyperparameters
    """
    function get_f(M)
        if M == 1
            f((γ, λ₁, λ₂, a₁, a₂, b₁, b₂)) = -v_bayes([xⁿ yⁿ],R;M=M,γ=exp(γ),m₁=m₁,m₂=m₂,λ₁=exp(λ₁),λ₂=exp(λ₂),
                a₁=exp(a₁),a₂=exp(a₁),b₁=exp(b₁),b₂=exp(b₁),EPOCHS=EPOCHS)[1][end]
            return f
        elseif M > 1
            g((γ, λ₁, λ₂, a₁, a₂, b₁, b₂)) = -v_bayes([xⁿ yⁿ],R;M=M,γ=exp(γ),m₁=m₁,m₂=m₂,λ₁=exp(λ₁),λ₂=exp(λ₂),
                a₁=exp(a₁),a₂=exp(a₂),b₁=exp(b₁),b₂=exp(b₂),EPOCHS=EPOCHS)[1][end]       
            return g
        end
    end
    if m_zero
        m₁, m₂ = 0., 0.
    end
    config = ConfigParameters(); config.n_iterations = n_iterations; config.verbose_level = 0; set_kernel!(config, "kMaternARD5"); config.sc_type = SC_MAP;
    f = get_f(M)
    param_keys = ["γ", "λ₁", "λ₂", "a₁", "a₂", "b₁", "b₂"]
    lowerbound = [log(.0001) for i in 1:7]; upperbound = [log(10000.) for i in 1:7];
    optimization_unfinished = true; global optimizer = []; global optimum = 0.;
    while optimization_unfinished
        @time optimizer, optimum = bayes_optimization(f, lowerbound, upperbound, config)
        optimization_unfinished = sum(convert(Array{Int64}, optimizer .== 0.00000))>0
        if optimization_unfinished
            println("optimization unsuccessful, reattempting")
        end
    end
    return Dict(zip(param_keys, optimizer)), optimum
    end;
     
function vb_hyperparameter_optimization(data,T_max=Inf; name="hp_opt_likelihood", M=1, P=1, Rs=1:1,
        EPOCHS=1, hp_optimization_type="unconstrained", n_iterations=200, norm=true, SEED=100, RESULTS_PATH, m_zero)
    # preparing output paths and directions
    OUTPUT_PATH = "$RESULTS_PATH/$(name)_$(hp_optimization_type)_seed_$(SEED)"
    mkpath(OUTPUT_PATH)
    DIRECTION = ['>','<', '^']
    # the empty dictionary below will be filled with those obtained from optimization
    parameters, likelihoods = prepare_empty_parameter_dictionary(Rs, data)
    for (id,pair) ∈ enumerate(data)
        println(pair[:id],"...\t\t\t")
        # a random permutation of data is obtained, if T_max is smaller than data size T is limited to T_max
        T = length(pair[:X])
        perm = randperm(T)[1:Int(min(T,T_max))]
        x, y = pair[:X][perm], pair[:Y][perm]
        xⁿ = norm ? (x .- mean(x)) / std(x) : x
        yⁿ = norm ? (y .- mean(y)) / std(y) : y
        # feeding Rs to the optimization likelihood function
        if hp_optimization_type=="unconstrained"
            parameters[pair[:id]]['^'], likelihoods[pair[:id]]['^'] = bayesian_optimization_likelihood(xⁿ, yⁿ; 
                Rs=Rs, EPOCHS=EPOCHS, M=1, P=P, n_iterations=n_iterations, unconstrained=true, m_zero=m_zero)
            parameters[pair[:id]]['>'], likelihoods[pair[:id]]['>'] = bayesian_optimization_likelihood(xⁿ, yⁿ; 
                Rs=Rs, EPOCHS=EPOCHS, M=M, P=P, n_iterations=n_iterations, unconstrained=true, m_zero=m_zero)
            parameters[pair[:id]]['<'], likelihoods[pair[:id]]['<'] = bayesian_optimization_likelihood(yⁿ, xⁿ; 
                Rs=Rs, EPOCHS=EPOCHS, M=M, P=P, n_iterations=n_iterations, unconstrained=true, m_zero=m_zero)     
        end   
    end
    metaparameters = Dict("M"=>M, "EPOCHS"=>EPOCHS, "P"=>P, "T_max"=>T_max, 
        "Rs"=>Rs, "hp_optimization_type"=>hp_optimization_type, "n_iterations"=>n_iterations, 
        "SEED"=>SEED, "norm"=>norm)
    if m_zero
        metaparameters["m₁"], metaparameters["m₂"] = 0., 0.
    end    
    save_json("$OUTPUT_PATH/metaparameters.json", metaparameters=metaparameters)
    save_json("$OUTPUT_PATH/parameters.json", parameters=parameters);
    save_json("$OUTPUT_PATH/likelihoods.json", likelihoods=likelihoods);
    end;
                                                            
                                                            
    function vb_hyperparameter_optimization(data,T_max=Inf; name="hp_opt_likelihood", M=1, P=1, Rs=1:1, EPOCHS=1, hp_optimization_type="unconstrained",         n_iterations=200, norm=true, SEED=100, RESULTS_PATH, m_zero, separate_Rs=true)
        # preparing output paths and directions
        OUTPUT_PATH = "$RESULTS_PATH/$(name)_$(hp_optimization_type)_seed_$(SEED)"
        mkpath(OUTPUT_PATH)
        DIRECTION = ['>','<', '^']
        # the empty dictionary below will be filled with those obtained from optimization
        parameters, likelihoods = prepare_empty_parameter_dictionary(Rs, data, separate_Rs)
        # for each data pair
        for (id,pair) ∈ enumerate(data)
            println(pair[:id],"...\t\t\t")
            # a random permutation of data is obtained, if T_max is smaller than data size T is limited to T_max
            T = length(pair[:X])
            perm = randperm(T)[1:Int(min(T,T_max))]
            x, y = pair[:X][perm], pair[:Y][perm]
            xⁿ = norm ? (x .- mean(x)) / std(x) : x
            yⁿ = norm ? (y .- mean(y)) / std(y) : y
            for R in Rs
                println(" ", R)
                # for each R calling the bayesian optimization likelihood function    
                if hp_optimization_type=="unconstrained"
                    parameters[pair[:id]][R]['^'], likelihoods[pair[:id]][R]['^'] = bayesian_optimization_likelihood(xⁿ, yⁿ; 
                        R=R, EPOCHS=EPOCHS, M=1, P=P, n_iterations=n_iterations,unconstrained=true, m_zero=m_zero,separate_Rs=true)
                    parameters[pair[:id]][R]['>'], likelihoods[pair[:id]][R]['>'] = bayesian_optimization_likelihood(xⁿ, yⁿ; 
                        R=R, EPOCHS=EPOCHS, M=M, P=P, n_iterations=n_iterations,unconstrained=true, m_zero=m_zero,separate_Rs=true)
                    parameters[pair[:id]][R]['<'], likelihoods[pair[:id]][R]['<'] = bayesian_optimization_likelihood(yⁿ, xⁿ; 
                        R=R, EPOCHS=EPOCHS, M=M, P=P, n_iterations=n_iterations,unconstrained=true, m_zero=m_zero,separate_Rs=true) 
                end
            end        
        end
        metaparameters = Dict("M"=>M, "EPOCHS"=>EPOCHS, "P"=>P, "T_max"=>T_max, 
            "Rs"=>Rs, "hp_optimization_type"=>hp_optimization_type, "n_iterations"=>n_iterations, 
            "SEED"=>SEED, "norm"=>norm)
        if m_zero
            metaparameters["m₁"], metaparameters["m₂"] = 0., 0.
        end 
        save_json("$OUTPUT_PATH/metaparameters.json", metaparameters=metaparameters)
        save_json("$OUTPUT_PATH/parameters.json", parameters=parameters);
        save_json("$OUTPUT_PATH/likelihoods.json", likelihoods=likelihoods);
    end;
                                                            
end;
              
                                                                                
                                                                                
                    
                    
                    
                    