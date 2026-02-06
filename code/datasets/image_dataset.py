import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import glob
import pytorch_lightning as pl
import os.path as op
import random
import sys

sys.path = [".."] + sys.path

from torch.utils.data import ConcatDataset
import datasets.utils as utils

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, subject, split, config, mode, hdri_filepath=None):
        root = op.join(root, subject)
        print("Dataset: ", root, split)
        cam_info = np.load(os.path.join(root, "cameras.npy"), allow_pickle=True).item()
        cameras = cam_info["cameras"]
        self.version = root.split('/')[3]
        K_dict, c2w_dict, height_dict, width_dict = utils.load_cameras(cameras)
        self.cam_ids = sorted(list(K_dict.keys()))
        if mode == "test" and hdri_filepath is not None:
            self.hdri_filepath = hdri_filepath
            assert os.path.exists(self.hdri_filepath), "HDRI not found: {}".format(
                self.hdri_filepath
            )

        self.downscale = config.downscale

        if self.downscale > 1:
            for cam_id in self.cam_ids:
                height_dict[cam_id] = int(height_dict[cam_id] / self.downscale)
                width_dict[cam_id] = int(width_dict[cam_id] / self.downscale)
                K_dict[cam_id][:2] /= self.downscale
        self.xy_all = utils.get_ray_directions(
            height_dict[self.cam_ids[0]], width_dict[self.cam_ids[0]]
            ).astype(np.float32)

        self.K_dict = K_dict
        self.c2w_dict = c2w_dict
        self.width_dict = width_dict
        self.height_dict = height_dict
        self.subject_ids = sorted(config.subject_ids)
        self.sid = subject
        is_personalize = config.personalize
        self.is_personalize = is_personalize

        self.mano_params = utils.load_mano_param(os.path.join(root, "poses.npy"))
        if is_personalize:
            self.mano_params['betas'] = utils.load_mano_param(os.path.join(root.replace('_val', '_train'), "poses.npy"))['betas']
            

        self.split = split
        self.mode = mode
        self.downscale = config.downscale
        self.near = config.get("near", None)  ## None
        self.far = config.get("far", None)  ## None
        self.use_vgg = config.use_vgg
        img_lists = sorted(glob.glob(op.join(root, f"images/*/*")))
        if mode == "train":
            from utils.sampler import EdgeSampler
            self.sampler = EdgeSampler(num_sample=3, ratio_mask=0.6, ratio_edge=0.3, kernel_size=16)
            if is_personalize:
                self.img_lists = img_lists*100000 # repeat for faster training
            else:
                quality_dict = np.load(f'./load/PALM/{self.version}/quality_dict_feb_27.npy', allow_pickle=True).item()
                def find_best_k(quality_dict, best_k):
                    # Sort the dictionary items by value in ascending order
                    sorted_items = sorted(quality_dict.items(), key=lambda item: item[1])
                    # Select the first k items
                    best_k_items = sorted_items[:best_k]
                    # Create a new dictionary from the selected items
                    best_k_dict = dict(best_k_items)
                    return best_k_dict

                # num of frames to train on
                best_dict = find_best_k(quality_dict, best_k=1848) # top 30%
                keys = list(best_dict.keys())
                
                keys = [tuple(key.split('/')) for key in keys]
                basenames = [basename for sid, basename in keys if sid == subject]
                if len(basenames) > 0:
                    # random.seed(1)
                    views = ["MCU_01",
                            "MCU_02",
                            "MCU_03",
                            "MCU_04",
                            "MCU_05",
                            "MCU_06",
                            "MCU_07"]
                    pairs = [(random.choice(views), basename) for basename in basenames]
                    folder_path = '/'.join(img_lists[0].split('/')[:7])
                    self.img_lists = [op.join(folder_path, view, basename.replace('.jpg', '.png')) for view, basename in pairs]
                    self.img_lists = self.img_lists *2000 ## repeat img_lists to speed up dataloader
                else: 
                    self.img_lists = []
        elif split in ["val"]:
            random.seed(4127)
            if 'ih_' in root: # interhand seen env novel pose eval
                self.img_lists = sorted(img_lists)
            else:
                if is_personalize: # personalization training
                    img_lists = random.sample(img_lists, 1)
                else: # prior training
                    img_lists = random.sample(img_lists, 1)
                    
                    img_lists += [
                        f'./load/PALM/{self.version}/folders/{self.sid}/images/MCU_03/000001.png',
                        f'./load/PALM/{self.version}/folders/{self.sid}/images/MCU_05/000012.png',
                    ]

                for im_p in img_lists:
                    assert op.exists(im_p), im_p
                self.img_lists = sorted(img_lists)
        elif split in ["test"]:
            img_lists = sorted(img_lists)
            self.img_lists = sorted(img_lists)
        else:
            assert False

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        idx = idx % len(self.img_lists)

        rgb_p = self.img_lists[idx]
        
        mid = int(rgb_p.split("/")[-1].split(".")[0])
        subject_id = rgb_p.split("/")[-4]
        midx = mid - 1
        
        mask_p = rgb_p.replace("/images/", "/masks/").replace(".jpg", ".png")
        normal_p = rgb_p.replace("/images/", "/normals/").replace(".jpg", ".png")
        cam_id = rgb_p.split("/")[-2]

        cam_idx = self.cam_ids.index(cam_id)

        img = utils.load_image(rgb_p)
        msk, msk_dilate = utils.load_mask(mask_p)
        if op.exists(normal_p):
            normal = utils.load_normal(normal_p)####
        else:
            normal = np.copy(img).astype(np.float32)*np.nan
        if self.downscale > 1:
            img = cv2.resize(
                img, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
            )
            msk = cv2.resize(
                msk, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
            )
            msk_dilate = cv2.resize(
                msk_dilate, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
            )
            normal = cv2.resize(
                normal, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
            )

        img = (img[..., :3] / 255.0).astype(np.float32)
        msk = msk.astype(np.float32)
        msk_dilate = msk_dilate.astype(np.float32)
        normal = normal.astype(np.float32)
        xy_all = np.copy(self.xy_all)
        if self.mode == "train":
            if self.use_vgg:
                if self.is_personalize:
                    VGG_SPECS = [(32, 1), (64, 2), (128, 4)]
                else:
                    VGG_SPECS = [(64, 2), (128, 4)]
                patch_size, k = random.choice(VGG_SPECS)
                img_patch, msk_patch, normal_patch, msk_dilate_patch, xy_patch = (
                    utils.sample_and_downsample_patches(
                        img,
                        msk,
                        normal,
                        msk_dilate,
                        xy_all,
                        patch_size=patch_size,
                        k=k,
                    )
                )
            (normal, msk, msk_dilate, img, xy) = self.sampler.sample(
                normal, msk, msk_dilate, img, xy_all
            )
            msk_dilate = msk_dilate.reshape(-1)
            if self.use_vgg:
                img = np.concatenate((img, img_patch.reshape(-1, 3)), axis=0)
                msk = np.concatenate((msk, msk_patch.reshape(-1)), axis=0)
                normal = np.concatenate((normal, normal_patch.reshape(-1, 3)), axis=0)
                msk_dilate = np.concatenate((msk_dilate, msk_dilate_patch.reshape(-1)), axis=0)
                xy = np.concatenate((xy, xy_patch.reshape(-1, 3)), axis=0)
        else:
            xy = xy_all.reshape(-1, 3)
            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)
            msk_dilate = msk_dilate.reshape(-1)
            normal = normal.reshape(-1, 3)
        rays_o, rays_d = utils.make_rays(
            self.K_dict[cam_id],
            self.c2w_dict[cam_id],
            xy,
        )
        assert idx < len(self.img_lists)
        if subject_id == '_val' and 'train' not in self.split:
            subject_id = '_train'
        datum = {
            # NeRF
            "rgb": img.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,
            "betas": self.mano_params["betas"][0],
            "global_orient": self.mano_params["global_orient"][midx],
            "body_pose": self.mano_params["body_pose"][midx],
            "transl": self.mano_params["transl"][midx],
            # auxiliary
            "alpha": msk,
            "msk_dilate": msk_dilate,
            "valid_mask": msk.astype(np.bool).reshape(-1),
            "normal": normal,
            "frame_idx": midx,
            "fname": rgb_p,
            "img_id": op.basename(rgb_p).split(".")[0],
            "K": self.K_dict[cam_id],
            "c2w": self.c2w_dict[cam_id].astype(np.float32),
            "width": self.width_dict[cam_id],
            "height": self.height_dict[cam_id],
            "cam_idx": cam_idx,
            "subject_id": subject_id,
        }
        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            # TODO: we could replace it with bbox in the canonical space
            dist = np.sqrt(np.square(self.mano_params["transl"][midx]).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)
        if self.mode == "test" and hasattr(self, "hdri_filepath"):
            import pyexr
            hdri = pyexr.open(self.hdri_filepath).get()[:, :, :3]
            datum["hdri"] = hdri.astype(np.float32)
        return datum

class ImageDatasetDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        data_root = self.config.dataroot
        if stage in [None, "fit"]:
            split = self.config.train_split
            subject_ids = self.config.opt.train.subject_ids                
            print("Training subject ids: ", subject_ids)
            datasets = [
                ImageDataset(
                    data_root,
                    subject_id,
                    split,
                    self.config.opt.get(split),
                    mode="train",
                )
                for subject_id in subject_ids
            ]
            self.train_dataset = ConcatDataset(datasets)
        if stage in [None, "fit", "validate"]:
            split = self.config.val_split
            subject_ids = sorted(self.config.opt.val.subject_ids)
            datasets = [
                ImageDataset(
                    data_root,
                    subject_id,
                    split,
                    self.config.opt.get(split),
                    mode="val",
                )
                for subject_id in subject_ids
            ]
            self.val_dataset = ConcatDataset(datasets)
        if stage in [None, "test"]:
            split = self.config.test_split
            subject_ids = sorted(self.config.opt.test.subject_ids)
            hdri_filepath = self.config.get("hdri_filepath", None)
            datasets = [
                ImageDataset(
                    data_root,
                    subject_id,
                    split,
                    self.config.opt.get(split),
                    mode="test",
                    hdri_filepath=hdri_filepath,
                )
                for subject_id in subject_ids
            ]
            self.test_dataset = ConcatDataset(datasets)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        if hasattr(self, "train_dataset"):
            return DataLoader(
                self.train_dataset,
                shuffle=self.config.opt.train.shuffle,
                num_workers=self.config.opt.train.num_workers,
                persistent_workers=True and self.config.opt.train.num_workers > 0,
                pin_memory=True,
                batch_size=self.config.opt.train.batch_size
                # batch_size=4,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            return DataLoader(
                self.val_dataset,
                shuffle=False,
                num_workers=self.config.opt.val.num_workers,
                persistent_workers=True and self.config.opt.val.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                shuffle=False,
                num_workers=self.config.opt.test.num_workers,
                persistent_workers=True and self.config.opt.test.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().test_dataloader()

    def predict_dataloader(self):
        if hasattr(self, "predict_dataset"):
            return DataLoader(
                self.predict_dataset,
                shuffle=False,
                num_workers=self.config.opt.test.num_workers,
                persistent_workers=True and self.config.opt.test.num_workers > 0,
                pin_memory=True,
                batch_size=1,
            )
        else:
            return super().predict_dataloader()
