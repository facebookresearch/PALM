#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn.functional as F
import torch

PBR_ENABLE_STEP = 2000

from common.xdict import xdict
import utils.loss_terms as loss_terms
import numpy as np

PBR_ENABLE_STEP = 1


def compute_zero_loss(out, batch, global_step, enable_phys, add_emitter, self):
    loss_dict = {}

    # Radiance field losses
    loss_dict = loss_terms.compute_rgb_rf_loss(loss_dict, out, batch)
    for key, val in loss_dict.items():
        loss_dict[key] *= 0.0
    loss_dict = xdict(loss_dict)
    return loss_dict
