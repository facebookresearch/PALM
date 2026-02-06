#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F

from skimage.metrics import structural_similarity


class WeightedLoss(nn.Module):
    @property
    def func(self):
        raise NotImplementedError

    def forward(self, inputs, targets, weight=None, reduction="mean"):
        assert reduction in ["none", "sum", "mean", "valid_mean"]
        loss = self.func(inputs, targets, reduction="none")
        if weight is not None:
            while weight.ndim < inputs.ndim:
                weight = weight[..., None]
            loss *= weight.float()
        if reduction == "none":
            return loss
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "valid_mean":
            return loss.sum() / weight.float().sum()


class MSELoss(WeightedLoss):
    @property
    def func(self):
        return F.mse_loss


class L1Loss(WeightedLoss):
    @property
    def func(self):
        return F.l1_loss


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction="mean"):
        assert reduction in ["mean", "none"]
        value = (inputs - targets) ** 2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == "mean":
            return -10 * torch.log10(torch.mean(value))
        elif reduction == "none":
            return -10 * torch.log10(
                torch.mean(value, dim=tuple(range(value.ndim)[1:]))
            )



# Use scipy package
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        if valid_mask is not None:
            assert valid_mask.dtype == torch.bool
            valid_mask = (valid_mask.byte() * 255).detach().cpu().numpy()
            x, y, w, h = cv2.boundingRect(valid_mask)
            inputs = inputs[y : y + h, x : x + w]
            targets = targets[y : y + h, x : x + w]

        value = structural_similarity(inputs, targets, multichannel=True)

        return value


class LPIPS(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # self.loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")

    def forward(self, inputs, targets, loss_fn_vgg, valid_mask=None):
        assert inputs.shape == targets.shape
        assert len(inputs.shape) == 3
        assert len(targets.shape) == 3

        if valid_mask is not None:
            assert valid_mask.dtype == torch.bool
            valid_mask = (valid_mask.byte() * 255).detach().cpu().numpy()
            x, y, w, h = cv2.boundingRect(valid_mask)
            inputs = inputs[y : y + h, x : x + w]
            targets = targets[y : y + h, x : x + w]

        inputs = inputs.unsqueeze(0).permute(0, 3, 1, 2)
        targets = targets.unsqueeze(0).permute(0, 3, 1, 2)
        value = loss_fn_vgg(inputs, targets, normalize=True)

        return value


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()
