#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn


# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder(nn.Module):
    def __init__(self, input_dims, num_freq, include_input=True, log_sampling=True):
        super().__init__()
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = num_freq - 1
        self.num_freq = num_freq
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]
        self.construct()

    def construct(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freq

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def step(self):
        pass

    def eval(self):
        pass

    def embed(self, inputs):
        encoded_x = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        return encoded_x
