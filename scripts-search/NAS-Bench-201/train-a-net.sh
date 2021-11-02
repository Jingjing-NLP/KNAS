#!/bin/bash
# bash ./scripts-search/NAS-Bench-201/train-a-net.sh resnet 16 5
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for network, channel, num-of-cells"
  exit 1
fi
#if [ "$TORCH_HOME" = "" ]; then
#  echo "Must set TORCH_HOME envoriment variable for data dir saving"
#  exit 1
#else
#  echo "TORCH_HOME : $TORCH_HOME"
#fi

model=$1
channel=$2
num_cells=$3

save_dir=./output/NAS-BENCH-201-single

python ./exps/NAS-Bench-201/main.py \
	--mode specific-${model} --save_dir ${save_dir} --max_node 1 \
	--datasets cifar100 \
	--use_less 0 \
	--splits                0 \
	--xpaths ../../cifar.python \
	--channel ${channel} --num_cells ${num_cells} \
	--workers 1 \
	--seeds 777
