

## Training

```bash
cd ./code
ln -s ../common
```

### Multi-subject prior training 

If you want to train our multi-subject prior on PALM-Net, run the following:


```bash
torchrun --nproc_per_node=8 --standalone launch.py dataset=prior.yaml gpu=\"0,1,2,3,4,5,6,7\" model.batch_size=128 model.loader_size=16
```

This will generate an experiment folder (for example, `./exp/aaaa-aaaa-aaa`) for the training. All artifacts such as checkpoints, visualizations will be stored under this folder. If you use SLURM, you can use this example SLURM script `run_train_prior.sh` and update the paths to the ones on your machine.

Once the training is done, you can extract the PCA space of the prior latent codes as follows:

```bash
python pca_decomposition.py --sd_p exp/aaaa-aaaa-aaa/ckpt/0/epoch=0-step=24000.ckpt
```

This will create the PCA basis like the following:

```bash
Explained variance ratio: [0.529188   0.14620578 0.11323046 0.09396036 0.06889796]
Saved PCA at:  ./pca_basis/aaaa-aaaa-aaa/epoch=0-step=100.shape.pca.npy
Explained variance ratio: [0.4520226  0.18015076 0.09320185 0.0458548  0.02672015]
Saved PCA at:  ./pca_basis/aaaa-aaaa-aaa/epoch=0-step=100.appearance.pca.npy
```

### Personalization

Suppose that the prior you trained above is stored under this path `ckpt_name=exp/aaaa-aaaa-aaa/ckpt/0/epoch=0-step=24000.ckpt` and you want to personalize the model using a preprocessed image at `./load/personalize/ih_c0_ROM03_RT_No_Occlusion_cam400262` (see [here](../generator/README.md) for preprocessing data), you can run the following:

```bash
IMG_NAME=ih_c0_ROM03_RT_No_Occlusion_cam400262
CKPT_PATH=exp/aaaa-aaaa-aaa/ckpt/0/epoch=0-step=24000.ckpt
./run_personalize.sh $IMG_NAME $CKPT_PATH
```

This will launch personalization on the specified image (in this case `ih_c0_ROM03_RT_No_Occlusion_cam400262`). 

### Relighting and novel pose rendering

Suppose the personalized model from the previous step stores at `exp/bbbb-bbbb-bbb/ckpt/0/epoch=0-step=5400.ckpt` and you want to use the poses at `load/personalize/ih_c1_ROM03_RT_No_Occlusion_cam400456` to drive the new model so that you can render using the environment map at `envs/flamingo_pan_4k` (can be downloaded [here](https://polyhaven.com/a/flamingo_pan)). You can run the following:

```bash
CKPT_PATH=exp/bbbb-bbbb-bbb/ckpt/0/epoch=0-step=5400.ckpt
MOTION_DATA=ih_c1_ROM03_RT_No_Occlusion_cam400456
HDRI=flamingo_pan_4k
./run_relight.sh $CKPT_PATH $MOTION_DATA $HDRI
```

Note: make sure to update `shape_code_basis_path` and `appearance_code_basis_path` in `run_relight.sh` to use the right PCA basis from your prior. 

You can find the relit results under `exp/$EXP_ID/render/pbr_rgb`. 
