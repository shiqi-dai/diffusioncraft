#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DETAIL=DEBUG

day=$(date "+%Y%m%d")
input_area_name="vanilla_mineshaft" # same as train.yaml
export LOGDIR="OUTPUT/$input_area_name-$day"

mpiexec -n 2 python train.py  
