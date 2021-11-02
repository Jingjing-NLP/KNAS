#!/bin/bash
dataset=$1
LR=$2
seed=$3
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/REINFORCE-${dataset}-${LR}

python ./exps/algos/reinforce.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} \
	--search_space_name ${space} \
	--arch_nas_dataset "../../a.pth" \
	--time_budget 12000 \
	--learning_rate ${LR} --EMA_momentum 0.9 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
