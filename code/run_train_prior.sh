#!/bin/bash -l
#  Copyright (c) Meta Platforms, Inc. and affiliates.

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=500
#SBATCH --mem=0
#SBATCH --time=30-24:00:00

export CUDA_HOME=/mnt/zicfan/packages/cuda118
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CPATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64/:$LD_LIBRARY_PATH"
export TORCH_HOME="/home/zicfan/torch_data"
export TORCH_MODEL_ZOO="/home/zicfan/torch_data"
export CUDA_LAUNCH_BLOCKING=1

source /mnt/zicfan/condas/palm/bin/activate

cd /home/zicfan/palm
torchrun --nproc_per_node=8 --standalone launch.py dataset=prior.yaml gpu=\"0,1,2,3,4,5,6,7\" model.batch_size=128 model.loader_size=16
