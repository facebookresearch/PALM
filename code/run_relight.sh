#  Copyright (c) Meta Platforms, Inc. and affiliates.

CKPT_PATH=$1
MOTION_DATA=$2
HDRI=$3
samples_per_pixel=4096
secondary_shader_chunk=20000
shape_code_basis_path=pca_basis/0301-1442-124/epoch=0-step=24000.shape.pca.npy
appearance_code_basis_path=pca_basis/0301-1442-124/epoch=0-step=24000.appearance.pca.npy
python launch.py mode=test dataset=personalize.yaml \
        light=envlight_tensor model.render_mode=light model.global_illumination=false \
        model.samples_per_pixel=$samples_per_pixel model.resample_light=false \
        model.add_emitter=true enable_phys=True model.loader_size=1 \
        resume_weights_only=True  \
        model.secondary_shader_chunk=$secondary_shader_chunk \
        seq_name=$MOTION_DATA relight=True \
        resume=\"$CKPT_PATH\" \
        shape_code_basis_path=\"$shape_code_basis_path\" \
        appearance_code_basis_path=\"$appearance_code_basis_path\"
