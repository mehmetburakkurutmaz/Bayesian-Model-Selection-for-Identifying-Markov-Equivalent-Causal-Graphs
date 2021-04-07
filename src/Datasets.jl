module Datasets

include("Misc.jl")
using .Misc

import NPZ, CSV
import DataFrames: dropmissing!, names!, readtable

export DataDesc, ngram, load, tuebingen, abalone

struct DataDesc
    path::String
    features::Array{Symbol}
    types::Array{DataType}
end

descriptions = Dict(
    "Abalone"       => DataDesc("data/abalone/abalone.data",
                         [:sex, :length, :diameter, :height, :whole, :shucked, :viscera, :shell, :rings],
                         [String,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int]),

    "Contraception" => DataDesc("data/contraception/cmc.data",
                         [:age, :education, :heducation, :children, :religion, :working, :hoccupation, :living, :exposure, :method],
                         [Float64,Int,Int,Float64,Int,Int,Int,Int,Int,Int]),

    "Citation"      => DataDesc("data/citation/citation.data",
                         [:source, :target],
                         [Int,Int]),

    "LastFM"        => DataDesc("data/lastfm/user_artists.dat",
                         [:user, :artist],
                         [String,String]),

    "Thyroid"        => DataDesc("data/thyroid/thyroid.data",
                         [:class, :T3RU, :Thyroxin, :Triiodothyronine, :TSH, :ΔTSH],
                         [Int,Float64,Float64,Float64,Float64,Float64])
    )



function load(name::String)
    desc = descriptions[name]
    df = CSV.read(desc.path; delim=',', datarow=1, types=desc.types, ignorerepeated=true)
    
    T,N = size(df)
    ℜ = promote_type(desc.types...)
    tab = Array{ℜ,2}(undef, T, N)
    
    for n in 1:N
        tab[:,n] .= df[!,n]
    end
    
    return tab
end

function load(name::String,features::Array{Symbol})
    desc = descriptions[name]
    df = CSV.read(desc.path; delim=',', datarow=1, types=desc.types, ignorerepeated=true)
    names!(df,desc.features)

    T,N = size(df)
    f_inds = [findfirst(desc.features .== feat) for feat in features]

    ℜ = promote_type(desc.types[f_inds]...)
    tab = Array{ℜ,2}(undef, T, length(features))

    for n in 1:length(features)
        tab[:,n] .= df[!,f_inds[n]]
    end
    
    return tab
end

function load(name::String,features::Array{String})
    return load(name, Symbol.(features))
end

function load(name::String,features::AbstractArray{Int})
    return load(name, descriptions[name].features[features])
end

function ngram(n::Int, T::Int)
    freq = NPZ.npzread("data/letters/transitions_$(n)L.npy");
    θ = freq ./ sum(freq)
    return finucan(Float64(T),θ)
end

function tuebingen(pairs::Array{Int})
    metadata = CSV.read("data/tuebingen/METADATA.csv"; delim='\t', datarow=1, ignorerepeated=true)
    names!(metadata,[:fname,:X_label,:Y_label,:dataset,:relationship,:weight])

    data = [Dict{Symbol,Any}() for id ∈ pairs]
    for (n,id) ∈ enumerate(pairs)
        pair = readtable("data/tuebingen/$(metadata.fname[id]).txt";separator=' ', header=false)
        pair = pair[:,1:2]
        dropmissing!(pair; disallowmissing=true)
        names!(pair,[:X,:Y])

        data[n][:id] = metadata.fname[id]
        data[n][:X_label] = metadata.X_label[id]
        data[n][:Y_label] = metadata.Y_label[id]
        data[n][:dataset] = metadata.dataset[id]
        data[n][:weight] = metadata.weight[id]
        data[n][:relationship] = '>' in metadata.relationship[id] ? "->" : "<-"
        data[n][:X] = pair.X
        data[n][:Y] = pair.Y
    end
    return data
end

function tuebingen()
    metadata = CSV.read("data/tuebingen/METADATA.csv"; delim='\t', datarow=1, ignorerepeated=true)
    names!(metadata,[:fname,:X_label,:Y_label,:dataset,:relationship,:weight])

    N = size(metadata,1)
    data = [Dict{Symbol,Any}() for n ∈ 1:N]
    for n ∈ 1:N
        pair = readtable("data/tuebingen/$(metadata.fname[n]).txt";separator=' ', header=false)
        pair = pair[:,1:2]
        dropmissing!(pair; disallowmissing=true)
        names!(pair,[:X,:Y])

        data[n][:id] = metadata.fname[n]
        data[n][:X_label] = metadata.X_label[n]
        data[n][:Y_label] = metadata.Y_label[n]
        data[n][:dataset] = metadata.dataset[n]
        data[n][:weight] = metadata.weight[n]
        data[n][:relationship] = '>' in metadata.relationship[id] ? "->" : "<-"
        data[n][:X] = pair.X
        data[n][:Y] = pair.Y
    end
    return data
end

end