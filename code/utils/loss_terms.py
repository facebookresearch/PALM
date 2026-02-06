#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F

import lpips
from systems.criterions import binary_cross_entropy

lpips_net = None


def compute_rgb_rf_loss(loss_dict, out, batch):
    if out["rays_valid_full"].sum() > 0:
        loss_rgb_l1 = F.l1_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )
        loss_dict["loss/rgb/l1"] = loss_rgb_l1
    return loss_dict


def compute_zero_density_loss(loss_dict, out, batch):
    far_index = out['gt_sdf'].detach().view(-1) > 0.02 # 2 cm
    far_density = out['pred_density'].view(-1)[far_index]
    loss_dict["loss/zero_density"] = far_density.mean()
    return loss_dict


def compute_rgb_pbr_loss(loss_dict, out, batch, add_emitter):
    loss_rgb_phys_l1 = F.l1_loss(
        out["comp_rgb_phys_full"][out["rays_valid_phys_full"][..., 0]],
        batch["rgb"][out["rays_valid_phys_full"][..., 0]],
    )
    loss_dict["loss/rgb/pbr/l1"] = loss_rgb_phys_l1 * 0.2
    return loss_dict


def compute_eikonal_loss(loss_dict, out, batch):
    ## always apply
    loss_eikonal = (
        (torch.linalg.norm(out["sdf_grad_samples"], ord=2, dim=-1) - 1.0) ** 2
    ).mean()
    loss_dict["loss/eikonal"] = loss_eikonal * 0.00001
    return loss_dict


def compute_vgg_loss(loss_dict, out, batch):
    global lpips_net
    if lpips_net is None:
        lpips_net = lpips.LPIPS(net="vgg").to("cuda")

    def scale_to_vgg(img):
        # img in [0, 1]
        img_scaled = img * 2 - 1  # -1, 1
        return img_scaled
    # patch_size = 64
    patch_size = 32
    patch_pred = scale_to_vgg(
        out["comp_rgb_full"][-(patch_size**2) :].view(1, patch_size, patch_size, -1)
    ).permute(0, 3, 1, 2)
    patch_gt = scale_to_vgg(
        batch["rgb"][-(patch_size**2) :].view(1, patch_size, patch_size, -1)
    ).permute(0, 3, 1, 2)

    patch_pred = F.interpolate(patch_pred, scale_factor=2, mode='bilinear')
    patch_gt = F.interpolate(patch_gt, scale_factor=2, mode='bilinear')

    loss_vgg = lpips_net(patch_pred, patch_gt).view(-1)[0] * 0.1
    loss_dict["loss/vgg"] = loss_vgg
    return loss_dict


def compute_normal_loss(loss_dict, out, batch):
    def get_normal_loss(normal_pred, normal_gt, mask):
        is_fg = mask.bool()
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.nanmean(torch.abs(normal_pred - normal_gt).sum(dim=-1)[is_fg])
        cos = torch.nanmean(1.0 - torch.sum(normal_pred * normal_gt, dim=-1)[is_fg])
        return l1, cos

    loss_normal_l1, loss_normal_cos = get_normal_loss(
        out["fg_normal"], batch["normal"], batch["alpha"]
    )
    loss_dict["loss/normal/l1"] = loss_normal_l1 * 0.1
    return loss_dict


def compute_off_surface_loss(loss_dict, out, batch):
    bg_idx = ~batch['msk_dilate'].bool().view(-1)
    if bg_idx.sum() == 0:
        return loss_dict
    loss_off_surface_pix = out['opacity'].view(-1)[bg_idx].mean()
    loss_dict["loss/off_surface"] = loss_off_surface_pix
    return loss_dict

def compute_mask_loss(loss_dict, out, batch):
    opacity = torch.clamp(out["opacity"].squeeze(-1), 1.0e-3, 1.0 - 1.0e-3)
    loss_mask_bce = binary_cross_entropy(opacity, batch["alpha"].float())
    loss_dict["loss/mask/bce"] = loss_mask_bce * 0.1
    return loss_dict

def compute_latent_loss(loss_dict, out, batch, normal_started):
    shape_id = out['shape_id']
    app_id = out['appearance_id']
    loss_shape = F.mse_loss(shape_id, torch.zeros_like(shape_id), reduction='none').sum(-1).mean()
    loss_app = F.mse_loss(app_id, torch.zeros_like(app_id), reduction='none').sum(-1).mean()
    weight = 1e-3 ## tune this
    if normal_started:
        loss_dict["loss/shape_id"] = loss_shape * weight
    loss_dict["loss/app_id"] = loss_app * weight
    return loss_dict
