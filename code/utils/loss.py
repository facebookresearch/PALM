#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn.functional as F
import torch

from common.xdict import xdict
import utils.loss_terms as loss_terms
import numpy as np

class WeightScheduler:
    def __init__(self, start_val, end_val, converged_steps):
        self.alphas = np.linspace(start_val, end_val, converged_steps)
    
    def __getitem__(self, global_step):
        if global_step < len(self.alphas):
            return self.alphas[global_step]
        return self.alphas[-1]

def compute_training_loss(out, batch, global_step, enable_phys, add_emitter, self):
    is_personalize = self.config.personalize
    loss_dict = xdict()
    normal_scheduler = WeightScheduler(5.0, 1.0, 24000)
    ## SDF MANO loss
    if self.config.pretrain:
        ## assume gt_sdf in meter
        sdf_loss = F.mse_loss(out["pred_sdf"], out["gt_sdf"], reduction='none')
        bound = 1e-3*20 # 20mm
        loss_dict["loss/sdf"] = torch.clamp(sdf_loss, 0, bound**2).mean()*10000

        loss_dict = loss_terms.compute_eikonal_loss(loss_dict, out, batch)
        return loss_dict
    
    # Radiance field losses
    loss_dict = loss_terms.compute_rgb_rf_loss(loss_dict, out, batch)

    # Eikonal loss
    if not is_personalize:
        loss_dict = loss_terms.compute_eikonal_loss(loss_dict, out, batch)

    ## VGG
    if global_step > self.config.model.vgg_start_step:
        loss_dict = loss_terms.compute_vgg_loss(loss_dict, out, batch)
    normal_started = global_step > self.config.model.normal_start_step
    if global_step > self.config.model.normal_start_step and not is_personalize:
        alpha = normal_scheduler[global_step]
        loss_dict = loss_terms.compute_normal_loss(loss_dict, out, batch)
        loss_dict.overwrite('loss/normal/l1', loss_dict['loss/normal/l1']*alpha)

        lambda_curvature = 1e-1
        loss_curvature = out["sdf_laplace_samples"].abs().mean()
        loss_dict["loss/curvature"] = loss_curvature * lambda_curvature

    ## mask loss
    loss_dict = loss_terms.compute_mask_loss(loss_dict, out, batch)

    loss_dict = loss_terms.compute_latent_loss(loss_dict, out, batch, normal_started)

    ## regularization loss
    loss_dict.update(self.model.regularizations(out))

    if enable_phys:
        loss_dict = loss_terms.compute_rgb_pbr_loss(loss_dict, out, batch, add_emitter)


    return loss_dict
