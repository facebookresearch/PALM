import os
import torch
from glob import glob
from PIL import Image
import numpy as np
import trimesh
from tqdm import tqdm
import os.path as op
import cv2
from smplx import MANO
from pytorch3d.transforms import matrix_to_axis_angle

import sys

sys.path = [".."] + sys.path
sys.path = ["."] + sys.path
import common.transforms as tf
from src.building.build_utils import save_visualizations

device = "cuda:0"


mano_models = {
    "right": MANO(
        "../code/data/body_models", is_rhand=True, flat_hand_mean=True, use_pca=False
    ).to(device),
    "left": MANO(
        "../code/data/body_models", is_rhand=False, flat_hand_mean=True, use_pca=False
    ).to(device),
}


def get_out_dir(seq_name):
    out_dir = f'../code/load/personalize/{seq_name}/folders/_train'
    return out_dir


def copy_images(rgb_ps, mask_ps, out_dir):
    cam_id = 'MCU_02'
    num_frames = len(rgb_ps)
    remap_old2new = {}

    for idx in tqdm(range(num_frames)):
        rgb_p = rgb_ps[idx]
        mask_p = mask_ps[idx]
        remap_old2new[rgb_p.split("/")[-1].split(".")[0]] = idx

        image = Image.open(rgb_p)
        mask = np.array(Image.open(mask_p))
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

        out_mask_p = op.join(out_dir, "masks", cam_id, f"{idx+1:04}.png")
        os.makedirs(op.dirname(out_mask_p), exist_ok=True)
        mask.save(out_mask_p)

        # Create the normal image
        # Convert the single-channel mask to a 3-channel image
        mask = np.array(Image.open(mask_p))
        normal_array = np.stack([mask, mask, mask, mask], axis=-1)
        normal = Image.fromarray(normal_array)
        # Save the normal image
        out_normal_p = os.path.join(out_dir, "normals", cam_id, f"{idx+1:04}.png")
        os.makedirs(os.path.dirname(out_normal_p), exist_ok=True)
        normal.save(out_normal_p)

        # save mano image
        out_mano_p = os.path.join(out_dir, "mano", cam_id, f"{idx+1:04}.png")
        os.makedirs(os.path.dirname(out_mano_p), exist_ok=True)
        normal.save(out_mano_p)

        out_image_p = op.join(out_dir, "images", cam_id, f"{idx+1:04}.png")
        os.makedirs(op.dirname(out_image_p), exist_ok=True)
        image.save(out_image_p)
        # break

    corres_out_p = op.join(out_dir, "corres.txt")
    # write rgb_ps into corres.txt
    base_ps = [op.basename(p) for p in rgb_ps]
    with open(corres_out_p, "w") as f:
        for rgb_p in base_ps:
            f.write(rgb_p + "\n")


def convert_parameters(mano_fit, K, o2w_all, num_frames, T_hip, normalize_shift=None):
    output_trans_r = []
    output_pose_r = []
    output_P = {}
    output_object_trans = []
    output_object_pose = []
    if normalize_shift is not None:
        R_cv, T_cv = tf.convert_gl2cv(np.eye(3)[None, :, :], np.zeros(3)[None, :])
        normalize_shift = normalize_shift.reshape(3, 1)
        R_cv = R_cv.reshape(3, 3)
        T_cv = T_cv.reshape(3, 1)
        normalize_shift = R_cv @ normalize_shift + T_cv
        normalize_shift = -normalize_shift.squeeze()
        normalize_shift += np.array([0, 0, -1])  # put object in front of camera
    else:
        # shift used in cvpr paper
        normalize_shift = np.array([-0.0085238, -0.01372686, 0.42570806])
    for idx in range(num_frames):
        # prepare right hand
        mano_poses = np.concatenate(
            (mano_fit["hand_rot"][idx], mano_fit["hand_pose"][idx]), axis=0
        )
        mano_trans_r = mano_fit["hand_transl"][idx]
        mano_rot_r = mano_poses[:3]
        mano_rot_r, mano_trans_r = tf.cv2gl_mano(mano_rot_r, mano_trans_r, T_hip)
        mano_poses[:3] = mano_rot_r

        obj_rot = o2w_all[idx, :3, :3].numpy()
        obj_trans = o2w_all[idx, :3, 3]

        Rt_o = np.eye(4)
        Rt_o[:3, :3] = obj_rot
        Rt_o[:3, 3] = obj_trans
        Rt_o[1:3] *= -1

        obj_rot = matrix_to_axis_angle(torch.FloatTensor(Rt_o[:3, :3])).numpy()
        obj_trans = Rt_o[:3, 3]

        trans_r = mano_trans_r + normalize_shift
        trans_obj_normalized = obj_trans + normalize_shift

        # static camera position
        target_extrinsic = np.eye(4)
        target_extrinsic[1:3] *= -1
        target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (
            target_extrinsic[:3, :3] @ normalize_shift
        )

        # view matrix
        K_pad = np.eye(4)
        K_pad[:3, :3] = K.numpy()
        P = K_pad @ target_extrinsic

        output_trans_r.append(trans_r)
        output_pose_r.append(mano_poses)

        output_object_trans.append(trans_obj_normalized)
        output_object_pose.append(obj_rot)
        output_P[f"cam_{idx}"] = P

    hand_poses_r = np.array(output_pose_r)
    hand_trans_r = np.array(output_trans_r)
    return (
        output_P,
        output_object_trans,
        output_object_pose,
        hand_poses_r,
        hand_trans_r,
        normalize_shift,
    )


def process_seq(
    seq_name, rebuild, scene_bounding_sphere, max_radius_ratio, no_vis, no_fixed_shift
):
    out_dir = get_out_dir(seq_name)
    print("Starting new build")
    print(out_dir)
    if rebuild:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
    if op.exists(out_dir):
        print(
            "Output directory already exists, skipping; Use --rebuild to remove and rebuild"
        )
        return
    rgb_ps = sorted(glob(f"./data/{seq_name}/images/*"))
    mask_ps = sorted(glob(f"./data/{seq_name}/processed/masks/*"))

    assert len(rgb_ps) == len(mask_ps)

    hold_fit = np.load(
        f"./data/{seq_name}/processed/hold_fit.init.npy", allow_pickle=True
    ).item()
    

    num_frames = len(mask_ps)

    copy_images(rgb_ps, mask_ps, out_dir)
    hand = "right"
    hand_fit = hold_fit[hand]
    if not no_vis:
        save_visualizations(
            rgb_ps, mask_ps, hand_fit, num_frames, out_dir, flag=hand
        )
    poses = {}
    poses['betas'] = hand_fit['betas'].reshape(1, 10)
    hand_mean = MANO("../code/data/body_models", is_rhand=True, flat_hand_mean=False, use_pca=False).hand_mean.numpy()
    hand_pose = hand_fit['hand_pose'].reshape(1, 45) - hand_mean[None, :]
    poses['hand_pose'] = hand_pose
    poses['transl'] = hand_fit['transl'].reshape(1, 3)
    poses['global_orient'] = hand_fit['global_orient'].reshape(1, 3)
    np.save(op.join(out_dir, "poses.npy"), poses)

    im_width, im_height = Image.open(rgb_ps[0]).size

    cameras = {}
    cam = {}
    cam['K'] = np.load(f"./data/{seq_name}/processed/K_hamer.npy")
    cam['Rt'] = np.eye(4)
    cam['height'] = im_height
    cam['width'] = im_width
    cameras['MCU_02'] = cam
    np.save(op.join(out_dir, "cameras.npy"), {"cameras": cameras})


def zip_seq(seq_name):
    import os
    import zipfile

    def zip_directory_with_exclusions(source_dir, zip_path, exclusions):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                # Filter out directories that should be excluded
                dirs[:] = [d for d in dirs if not any(excl in d for excl in exclusions)]
                for file in files:
                    if not any(excl in file for excl in exclusions):
                        file_path = os.path.join(root, file)
                        # Store the file relative path in the zip, including the seq_name directory
                        arcname = os.path.join(
                            seq_name, os.path.relpath(file_path, start=source_dir)
                        )
                        zipf.write(file_path, arcname)

    # Example usage
    directory_to_zip = f"./data/{seq_name}/"
    output_zip_file = f"./data/{seq_name}.zip"
    exclusion_keywords = ["processed", "video.mp4", "vis", "images.zip", "images"]
    zip_directory_with_exclusions(directory_to_zip, output_zip_file, exclusion_keywords)
    print("Exported zipfile to", output_zip_file)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--no_zip", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--build_all", action="store_true")
    parser.add_argument("--max_radius_ratio", type=float, default=3.0)
    parser.add_argument("--scene_bounding_sphere", type=float, default=6.0)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--no_fixed_shift", action="store_true")
    return parser.parse_args()

def build_a_dataset(seq_name, args):
    process_seq(
        seq_name,
        args.rebuild,
        args.scene_bounding_sphere,
        args.max_radius_ratio,
        args.no_vis,
        args.no_fixed_shift,
    )
    if not args.no_zip:
        zip_seq(seq_name)

def main():
    args = parse_args()
    if args.build_all:
        from glob import glob
        fnames = sorted(glob('./data/*'))
        import pdb; pdb.set_trace()
        seq_names = [op.basename(fname) for fname in fnames]
        for seq_name in seq_names:
            build_a_dataset(seq_name, args)
    else:
        build_a_dataset(args.seq_name, args)


if __name__ == "__main__":
    main()
