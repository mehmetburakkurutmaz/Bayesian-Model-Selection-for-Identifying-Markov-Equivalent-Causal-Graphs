#!/bin/bash


for i in {1..9}
do
	bmlingam-causality bmlingam/pairs/pair000$i.csv --result_dir=bmlingam/results/pair000$i --col_names="x1,x2" --seed=42 --cs='0,.4,.8' --L_cov_21s='[-0.9,-0.5,0,.5,.9]' --betas_noise='.5,1.' --n_mc_samples 1000
done

for i in {10..51}
do
	bmlingam-causality bmlingam/pairs/pair00$i.csv --result_dir=bmlingam/results/pair00$i --col_names="x1,x2" --seed=42 --cs='0,.4,.8' --L_cov_21s='[-0.9,-0.5,0,.5,.9]' --betas_noise='.5,1.' --n_mc_samples 1000
done

for i in {56..70}
do
	bmlingam-causality bmlingam/pairs/pair00$i.csv --result_dir=bmlingam/results/pair00$i --col_names="x1,x2" --seed=42 --cs='0,.4,.8' --L_cov_21s='[-0.9,-0.5,0,.5,.9]' --betas_noise='.5,1.' --n_mc_samples 1000
done

for i in {72..99}
do
	bmlingam-causality bmlingam/pairs/pair00$i.csv --result_dir=bmlingam/results/pair00$i --col_names="x1,x2" --seed=42 --cs='0,.4,.8' --L_cov_21s='[-0.9,-0.5,0,.5,.9]' --betas_noise='.5,1.' --n_mc_samples 1000 
done

bmlingam-causality bmlingam/pairs/pair0100.csv --result_dir=bmlingam/results/pair0100 --col_names="x1,x2" --seed=42 --cs='0,.4,.8' --L_cov_21s='[-0.9,-0.5,0,.5,.9]' --betas_noise='.5,1.' --n_mc_samples 1000 

