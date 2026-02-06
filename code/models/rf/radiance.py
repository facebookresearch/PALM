import torch
import numpy as np

import models
from models.base import BaseModel
from models.utils import get_activation, reflect
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


class BaseImplicitRadiance(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def prepare_bbox(self, bbox):
        # if hasattr(self, "bbox"):
        #     return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox


class VolumeRadiance(BaseImplicitRadiance):
    def setup(self):
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        xyz_encoding_config = self.config.get('xyz_encoding_config', None)
        xyz_encoding = (
            get_encoding(3, xyz_encoding_config)
            if xyz_encoding_config is not None
            else None
        )
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
        if xyz_encoding is not None:
            self.n_input_dims += xyz_encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.xyz_encoding = xyz_encoding
        self.dir_encoding = dir_encoding
        self.network = network

    def forward(self, points, features, dirs, *args, feature_only=False):
        if self.xyz_encoding is not None:
            points = (points - self.center) / self.scale + 0.5
            xyz_embd = self.xyz_encoding(points.view(-1, 3))
        else:
            xyz_embd = torch.empty(
                features.shape[:1] + (0,), dtype=features.dtype
            ).cuda()

        network_inp = [xyz_embd, features.view(-1, features.shape[-1])]
        if feature_only:
            return xyz_embd

        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims))
        network_inp.append(dirs_embd)
        network_inp = torch.cat(
            network_inp + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1
        )
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        # Return color and geometric features
        return color, xyz_embd

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

    def regularizations(self, out):
        if hasattr(self.network, 'regularizations'):
            assert False, 'not being called'
            return self.network.regularizations()
        else:
            return {}

## register this constructor into `models` so that we can `make` it
class VolumeRefDirRadiance(BaseImplicitRadiance):
    def setup(self):
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        xyz_encoding_config = self.config.get('xyz_encoding_config', None)
        xyz_encoding = (
            get_encoding(3, xyz_encoding_config)
            if xyz_encoding_config is not None
            else None
        )
        # xyz_encoding
        # CompositeEncoding(
        # (encoding): ProgressiveBandHashGrid(
        #     (encoding): Encoding(n_input_dims=3, n_output_dims=64, seed=1337, dtype=torch.float32, hyperparams={'base_resolution': 16, 'hash': 'CoherentPrime', 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 32, 'otype': 'Grid', 'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
        # )
        # )

        # () dir_encoding
        # CompositeEncoding(
        # (encoding): Encoding(n_input_dims=3, n_output_dims=16, seed=1337, dtype=torch.float32, hyperparams={'degree': 4, 'otype': 'SphericalHarmonics'})
        # )
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims
        if xyz_encoding is not None:
            self.n_input_dims += xyz_encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)

        # network
        # VanillaMLP(
        # (layers): Sequential(
        #     (0): Linear(in_features=99, out_features=64, bias=True)
        #     (1): ReLU(inplace=True)
        #     (2): Linear(in_features=64, out_features=64, bias=True)
        #     (3): ReLU(inplace=True)
        #     (4): Linear(in_features=64, out_features=3, bias=True)
        # )
        # )

        self.xyz_encoding = xyz_encoding ## hash grid encoding for position
        self.dir_encoding = dir_encoding ## encode viewing direction
        self.network = network

        self.register_buffer(
            "sh_mask",
            torch.zeros(1, self.dir_encoding.n_output_dims, dtype=torch.float32),
        )
        # Default parameters for progressive SH bands
        # i.e. no progressive SH
        # self.start_step = self.config.get('start_step', 0)
        self.start_step = self.config.xyz_encoding_config.get('start_step', 0) ### my fix
        
        # self.full_band_step = self.config.get('full_band_step', 1) ## 1
        self.full_band_step = self.config.xyz_encoding_config.get('full_band_step', 1) ### my fix


    def forward(self, points, features, dirs, *args, feature_only=False):
        
        if self.xyz_encoding is not None:
            points = (points - self.center) / self.scale + 0.5
            xyz_embd = self.xyz_encoding(points.view(-1, 3))
        else:
            xyz_embd = torch.empty(
                features.shape[:1] + (0,), dtype=features.dtype, device=features.device
            )

        network_inp = [xyz_embd, features.view(-1, features.shape[-1])]
        if feature_only:
            return xyz_embd
        
        dirs = reflect(-dirs, args[0])
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims)) * self.sh_mask
        network_inp.append(dirs_embd)
        
        network_inp = torch.cat(
            network_inp + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1
        )
        
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        # Return color and geometric features
        return color, xyz_embd

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)
        update_module_step(self.xyz_encoding, epoch, global_step)

        # Progressively enabling different bands of spherical harmonics
        t = max(global_step - self.start_step, 0.0)
        N = self.full_band_step - self.start_step
        m = self.config.dir_encoding_config.degree ## my fix
        # m = 4
        alpha = m * t / N

        idx = 0
        for deg in range(m):
            w = (
                1.0 - np.cos(np.pi * min(max(alpha - deg, 0.0), 1.0))
            ) / 2.0
            next_idx = idx + deg * 2 + 1
            self.sh_mask[..., idx:next_idx] = w
            idx = next_idx

    def regularizations(self, out):
        if hasattr(self.network, 'regularizations'):
            return self.network.regularizations()
        else:
            return {}

