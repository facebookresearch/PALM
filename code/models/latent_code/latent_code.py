#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

class LatentCode(nn.Module):
    def __init__(self, subject_ids, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        codec = {}
        for subject_id in subject_ids:
            codec[subject_id] = torch.randn(1, self.latent_dim)*1e-8
        self.codec = nn.ParameterDict(codec)
        self.enable_latent_code = False

    def forward(self, subject_id):
        self.enable_latent_code = True
        if self.enable_latent_code:
            subject_code = torch.cat([self.codec[sid] for sid in subject_id], dim=0)
        else:
            subject_code = torch.zeros((len(subject_id), self.latent_dim)).to(self.codec[subject_id[0]].device)
        return subject_code

    def update_step(self, epoch, global_step):
        if global_step > self.config.latent_code_start_step:
            self.enable_latent_code = True
