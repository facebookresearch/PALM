import torch.nn.functional as F
import torch

PBR_ENABLE_STEP = 2000

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

def compute_mse_loss(shape_pca, weight):
    # Compute the mean squared error loss
    mse_loss = torch.mean(shape_pca ** 2)
    
    # Apply the weight to the loss
    loss = mse_loss * weight
    
    return loss


def compute_personalize_loss(out, batch, global_step, enable_phys, add_emitter, self):
    is_personalize = self.config.personalize
    loss_dict = xdict()
    normal_scheduler = WeightScheduler(5.0, 1.0, 24000)

    # Radiance field losses
    loss_dict = loss_terms.compute_rgb_rf_loss(loss_dict, out, batch)

    # Eikonal loss
    if not is_personalize:
        loss_dict = loss_terms.compute_eikonal_loss(loss_dict, out, batch)

    loss_dict = loss_terms.compute_vgg_loss(loss_dict, out, batch)

    lambda_curvature = 1e-1
    loss_curvature = out["sdf_laplace_samples"].abs().mean()
    loss_dict["loss/curvature"] = loss_curvature * lambda_curvature

    loss_dict = loss_terms.compute_mask_loss(loss_dict, out, batch)

    loss_dict = loss_terms.compute_latent_loss(loss_dict, out, batch, normal_started=True)

    if global_step > PBR_ENABLE_STEP:
        loss_dict = loss_terms.compute_rgb_pbr_loss(loss_dict, out, batch, add_emitter)

    return loss_dict