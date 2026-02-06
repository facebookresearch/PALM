import torch
import torch.nn as nn
import numpy as np
class PCACode(nn.Module):
    def __init__(self, subject_ids, config):
        super().__init__()
        self.config = config
        self.latent_dim = 5
        codec = {}
        for subject_id in subject_ids:
            codec[subject_id] = torch.zeros(1, self.latent_dim)
        self.codec = nn.ParameterDict(codec)
        self.enable_latent_code = False

    def forward(self, subject_id):
        self.basis.requires_grad = False
        self.enable_latent_code = True
        if self.enable_latent_code:
            pca_code = torch.cat([self.codec[sid] for sid in subject_id], dim=0)
            print(pca_code[:1])
            subject_code = pca_code @ self.basis
        else:
            assert False
        return pca_code, subject_code

    def load_basis(self, basis_path):
        self.basis = nn.Parameter(torch.FloatTensor(np.load(basis_path)), requires_grad=False).cuda()
        print('Loaded basis: ', basis_path)

    def update_step(self, epoch, global_step):
        if global_step > self.config.latent_code_start_step:
            self.enable_latent_code = True
