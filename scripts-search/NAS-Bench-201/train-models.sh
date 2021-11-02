#!/bin/bash
# bash ./scripts-search/train-models.sh 0/1 0 100 -1 '777 888 999'
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for use-less-or-not, start-and-end, arch-index, and seeds"
  exit 1
fi

use_less=$1
xstart=$2
xend=$3
arch_index=$4
all_seeds=$5

save_dir=./output/NAS-BENCH-201-4/

if [ ${arch_index} == "-1" ]; then
  mode=new
else
  mode=cover
fi


python3 ./exps/NAS-Bench-201/main.py \
	--mode ${mode} --save_dir ${save_dir} --max_node 4 \
	--use_less ${use_less} \
	--datasets cifar10 cifar100 ImageNet16-120 \
	--splits        0       0        0 \
	--xpaths cifar.python \
		 cifar.python \
		 cifar.python/ImageNet16 \
	--channel 16 --num_cells 5 \
	--workers 0 \
	--srange ${xstart} ${xend} --arch_index ${arch_index} \
	--seeds ${all_seeds}
