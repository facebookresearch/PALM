import torch
import torch.nn as nn
from models.utils import get_activation
from models.network_utils import get_mlp


class VolumeMaterial(nn.Module):
    def __init__(self, config):
        super(VolumeMaterial, self).__init__()
        self.config = config
        self.n_output_dims = self.config.get("n_output_dim", 5)
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(
            self.n_input_dims, self.n_output_dims, self.config.mlp_network_config
        )
        self.network = network

        self.albedo_bias = self.config.get("albedo_bias", 0.03)
        self.albedo_scale = self.config.get("albedo_scale", 0.77)
        self.roughness_bias = self.config.get("roughness_bias", 0.09)
        self.roughness_scale = self.config.get("roughness_scale", 0.9)
        self.metallic_bias = self.config.get("metallic_bias", 0.0)
        self.metallic_scale = self.config.get("metallic_scale", 1.0)

    def forward(self, features, *args):
        network_inp = torch.cat(
            [features.view(-1, features.shape[-1])]
            + [arg.view(-1, arg.shape[-1]) for arg in args],
            dim=-1,
        )
        material = (
            self.network(network_inp)
            .view(*features.shape[:-1], self.n_output_dims)
            .float()
        )
        if "material_activation" in self.config:
            material = get_activation(self.config.material_activation)(material)
        albedo = material[..., :3] * self.albedo_scale + self.albedo_bias
        roughness = (
            material[..., 3:4] * self.roughness_scale + self.roughness_bias
        )
        
        metallic = material[..., 4:] * self.metallic_scale + self.metallic_bias
        return torch.cat([albedo, roughness, metallic], dim=-1)

    def regularizations(self, out):
        return {}
