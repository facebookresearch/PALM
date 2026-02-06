## Data documentation

### Data download

After registering an account at [here](https://forms.gle/zbNPFtJDaZP4cQmz7) and then you will recieve a link to download PALM-Net/PALM and have access to the following files: 

```bash
./PALM/XR20A/zips/0117.zip
./PALM/XR20A/zips/0201.zip
./PALM/XR20A/zips/0140.zip
./PALM/XR20A/zips/0251.zip
...
./PALM/XR20A/quality_dict_feb_27.npy
./PALM/XR20A/scan_ids.txt
./PALM/XR20A/MCU_04_env.exr
./PALM/XR20A/meshes/0221.zip
./PALM/XR20B/zips/0221.zip
./PALM/XR20B/zips/0226.zip
./PALM/XR20B/zips/0251.zip
./PALM/XR20B/zips/0003.zip
./PALM/XR20B/zips/0049.zip

...
./PALM/XR20B/quality_dict_feb_27.npy
./priors/0301-1442-124.zip
./pca_basis/0301-1442-124/epoch=0-step=24000.appearance.pca.npy
./pca_basis/0301-1442-124/epoch=0-step=24000.shape.pca.npy
./pretrained/flat_hand.right.dim32.7subj.feb.23.pt
./personalize_ckpts/0306-0218-596.zip
./personalize_ckpts/0306-1503-223.zip
./gt_checksums.json
```

There are two versions of PALM: XR20A and XR20B. The XR20A (137GB) version contains high-resolution 2K images while the XR20B (27GB) version resized the images for faster training. The camera intrinsics for XR20B are resized accordingly. Everything else in these two versions are the identical. We also have 3dMD scans for each frame at PALM. All 3D annotations are in meters. 

- `./PALM/XR20A/zips/0117.zip`: subject 0117 of PALM dataset with version XR20A.
- `./PALM/XR20A/quality_dict_feb_27.npy`: fitting error for each frame. 
- `./PALM/XR20A/MCU_04_env.exr`: calibrated environment map
- `./PALM/XR20A/meshes/0221.zip`: Hand scans for subject 221
- `./priors/0301-1442-124.zip`: Pretrained multi-subject prior with version `0301-1442-124` (see Checkpoints below)
- `./pca_basis/0301-1442-124/epoch=0-step=24000.appearance.pca.npy`: PCA appearance space of the prior with version `0301-1442-124`
- `./pca_basis/0301-1442-124/epoch=0-step=24000.shape.pca.npy`: PCA shape space of the prior with version `0301-1442-124`
- `./pretrained/flat_hand.right.dim32.7subj.feb.23.pt`: pretrained geometry network on 7 subjects using MANO SDF values; we found using a pretrained hand geometry network stablizes the PALM model training. 
- `./personalize_ckpts/0306-0218-596.zip`: Personalization checkpoints in our InterHand2.6M experiments (see Checkpoints below)
- `./gt_checksums.json`: checksums of the files to check for file corruption.

About the calibrated environment map, we put a metal sphere in our 3dMD capture setup and fit a sphere to this metal sphere to unwrap an environment map from RGB observations. 

### Checkpoints

We offer three versions of priors trained on 14%, 50% and 100% of frames at PALM: 

|     Priors    | Number of frames |
|:-------------:|:----------------:|
| 0301-1442-124 |        14% (ckpt in paper)|
| 0311-0806-124 |        50%       |
| 0311-0910-389 |       100%       |

Here are the personalized checkpoints for our InterHand2.6M experiments: `ih_c0_ROM03_RT_No_Occlusion_cam400262` denotes the sequence with name `ROM03_RT_No_Occlusion` in capture 0 on the test set; `cam400262` is the camera view. 

| InterHand2.6M sequences               | Personalized checkpoints |
|---------------------------------------|--------------------------|
| ih_c0_ROM03_RT_No_Occlusion_cam400262 | 0306-0218-303            |
| ih_c0_ROM03_RT_No_Occlusion_cam400451 | 0306-0218-212            |
| ih_c0_ROM04_RT_Occlusion_cam400275    | 0306-0218-171            |
| ih_c0_ROM04_RT_Occlusion_cam400418    | 0306-0218-458            |
| ih_c0_ROM05_RT_Wrist_ROM_cam400270    | 0306-0218-299            |
| ih_c0_ROM05_RT_Wrist_ROM_cam400488    | 0306-0218-402            |
| ih_c1_ROM03_RT_No_Occlusion_cam400456 | 0306-0218-443            |
| ih_c1_ROM03_RT_No_Occlusion_cam400486 | 0306-0219-214            |
| ih_c1_ROM04_RT_Occlusion_cam400266    | 0306-0218-596            |
| ih_c1_ROM04_RT_Occlusion_cam400439    | 0306-0219-262            |
| ih_c1_ROM05_RT_Wrist_ROM_cam400314    | 0306-0218-439            |
| ih_c1_ROM05_RT_Wrist_ROM_cam400469    | 0306-0218-495            |

### Environment maps

Environment maps can be found [here](https://polyhaven.com/a/flamingo_pan). You should put them under `code/envs/*.hdri` and `code/envs/*.exr`. 

