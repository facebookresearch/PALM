#!/bin/bash
#  Copyright (c) Meta Platforms, Inc. and affiliates.

set -e
export CUDA_VISIBLE_DEVICES=0

# checkpoint of the prior model
shape_code_basis_path=\"pca_basis/0311-0719-135/epoch=0-step=24000.shape.pca.npy\"
appearance_code_basis_path=\"pca_basis/0311-0719-135/epoch=0-step=24000.appearance.pca.npy\"

seq_name=ih_c0_ROM03_RT_No_Occlusion_cam400262
ckpt_name=./exp/0314-0220-275/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c0_ROM03_RT_No_Occlusion_cam400451
ckpt_name=./exp/0314-0220-257/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c0_ROM04_RT_Occlusion_cam400275
ckpt_name=./exp/0314-0220-212/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c0_ROM04_RT_Occlusion_cam400418
ckpt_name=./exp/0314-0220-068/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c0_ROM05_RT_Wrist_ROM_cam400270
ckpt_name=./exp/0314-0220-261/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c0_ROM05_RT_Wrist_ROM_cam400488
ckpt_name=./exp/0314-0220-216/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c1_ROM03_RT_No_Occlusion_cam400456
ckpt_name=./exp/0314-0219-541/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c1_ROM03_RT_No_Occlusion_cam400486
ckpt_name=./exp/0314-0220-223/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c1_ROM04_RT_Occlusion_cam400266
ckpt_name=./exp/0314-0220-129/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c1_ROM04_RT_Occlusion_cam400439
ckpt_name=./exp/0314-0220-147/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c1_ROM05_RT_Wrist_ROM_cam400314
ckpt_name=./exp/0314-0220-369/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path

seq_name=ih_c1_ROM05_RT_Wrist_ROM_cam400469
ckpt_name=./exp/0314-0220-175/ckpt/0/it005100.ckpt
./run_render_interhand.sh $seq_name $ckpt_name $shape_code_basis_path $appearance_code_basis_path
