#!/bin/bash
#  Copyright (c) Meta Platforms, Inc. and affiliates.

# set -e
seq_name=$1
ckpt_name=$2
shape_code_basis_path=$3
appearance_code_basis_path=$4

val_check_interval=1
batch_size=1
loader_size=1
num_pixels=3
resume_weights_only=True
pose_lr=0.000000001

echo $seq_name


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/zicfan/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/zicfan/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/zicfan/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/zicfan/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


echo "Starting"
conda activate palm


python launch.py dataset=personalize.yaml gpu=\"0\" model.batch_size=$batch_size \
    model.loader_size=$loader_size resume_weights_only=$resume_weights_only personalize=True \
    trainer.val_check_interval=$val_check_interval no_mesh=True checkpoint.save_top_k=0 \
     resume=\"$ckpt_name\" sampler.num_sample=$num_pixels system.optimizer.params.pose_correction.lr=$pose_lr \
     seq_name=$seq_name \
     dataset.opt.val.subject_ids=['_val'] \
     trainer.max_steps=1 \
     render_interhand=True \
     shape_code_basis_path=$shape_code_basis_path \
     appearance_code_basis_path=$appearance_code_basis_path
