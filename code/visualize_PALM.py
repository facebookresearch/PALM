"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from glob import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from smplx import MANO
import os.path as op
import argparse
from transforms import transform_points_batch, project2d_batch


def visualize_palmDB(data_p, cam_id, idx):
    """
    create a simple visualization by projecting the vertices to the image plane
    """
    root_path = op.dirname(data_p)
    poses = np.load(data_p, allow_pickle=True).item()

    mano_layer = MANO(
        './data/body_models/',
        use_pca=False, flat_hand_mean=False, is_rhand=True)

    for key, val in poses.items():
        poses[key] = torch.FloatTensor(val)

    cam = np.load(op.join(root_path, 'cameras.npy'), allow_pickle=True).item()['cameras'][cam_id]
    K = cam['K']
    w2c = cam['Rt']
    image_paths = sorted(glob(op.join(root_path, 'images', cam_id, '*')))
    mask_paths = sorted(glob(op.join(root_path, 'masks', cam_id, '*')))
    assert len(image_paths) > 0


    num_frames = poses['global_orient'].shape[0]
    betas = poses['betas'].repeat(num_frames, 1)

    with torch.no_grad():
        outputs = mano_layer(
            betas=betas, hand_pose=poses['hand_pose'], global_orient=poses['global_orient'], transl=poses['transl'])
        j3d_w = outputs.joints 
        v3d_w = outputs.vertices
        
        j3d_c = transform_points_batch(torch.FloatTensor(w2c)[None, :, :].repeat(num_frames, 1, 1), j3d_w)
        v3d_c = transform_points_batch(torch.FloatTensor(w2c)[None, :, :].repeat(num_frames, 1, 1), v3d_w)
        v2d = project2d_batch(torch.FloatTensor(K)[None, :, :].repeat(num_frames, 1, 1), v3d_c).numpy()

    plt.imshow(Image.open(image_paths[idx]))
    plt.imshow(Image.open(mask_paths[idx]), alpha=0.2)
    plt.scatter(v2d[idx, :, 0], v2d[idx, :, 1], s=1)
    plt.savefig("vis_palm.png")
    plt.close()
    print('Checkout visualization at ', "./vis_palm.png")

def main():
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('-c', '--cam_id', type=str, help='camera id', default='MCU_02')
    parser.add_argument('-id', '--id', type=str, help='subject id', required=True)
    parser.add_argument('-frame', '--frame', type=int, help='frame id', default=0)
    # Parse the arguments
    args = parser.parse_args()

    visualize_palmDB(op.join('./PALMDB_highres', 'zips', str(args.id), 'poses.npy'), args.cam_id, args.frame)


if __name__ == "__main__":
    main()
