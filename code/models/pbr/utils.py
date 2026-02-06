#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

from torch import Tensor


from lib.nerfacc import (
    pack_info,
    unpack_info,
    ray_resampling,
)


def sample_volume_interaction(
    rays_o: Tensor,
    rays_d: Tensor,
    cond_pack: Tensor,
    img_indices: Tensor,
    ray_indices: Tensor,
    t_starts: Tensor,
    t_ends: Tensor,
    n_rays: int,
    samples_per_pixel: int,
    transmittance_map: Tensor,
    extras: dict,
):
    """
    Sample interaction points for volumetric scattering, without re-evaluating density/weight/material/normal. We assume
    density/weight/material/normal/position of any point in an interval equals to the density/weight/material/normal/position
    of the midpoint of the interval.

    Args:
        rays_o (Tensor): Ray origins of shape (n_rays, 3).
        rays_d (Tensor): Normalized ray directions of shape (n_rays, 3).
        ray_indices (Tensor): Ray indices of the flattened samples. LongTensor with shape (all_samples).
        t_starts (Tensor): Per-sample start distance. Tensor with shape (all_samples,).
        t_ends (Tensor): Per-sample end distance. Tensor with shape (all_samples,).
        n_rays (int): Number of rays. Only useful when `ray_indices` is provided.
        samples_per_pixel (int): Number of samples per ray/pixel.
        transmittance_map (Tensor): Transmittance map of shape (n_rays, 1).
        extras (dict): Extra information for sample, including `weights`, `normals`, `albedo`, `roughness`, `metallic`.
            each of the value is a Tensor of shape (all_samples, ...).

    Returns:
        resampled_packed_info (Tensor): Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. Tensor with shape (n_rays, 2).
        resampled_ray_indices (Tensor): Resampled ray indices of shape (n_resamples,).
        resampled_weights (Tensor): Resampled weights of shape (n_resamples, 1).
        fg_indices (Tensor): Indices of foreground samples of shape (n_fg_resamples,).
        bg_indices (Tensor): Indices of background samples of shape (n_bg_resamples,).
        resampled_extras (dict): Resampled extra information for sample, including `positions`, `normals`, `albedo`,
            `roughness`, `metallic`, `t_dirs`. Each of the value is a Tensor of shape (n_resamples, ...).
    """
    # Importance sample points for volumetric scattering
    weights = extras["weights"]
    sdfs = extras["sdf"]
    with torch.no_grad():
        ## packed_info[idx] -> sample_idx, num_samples
        # this means: 
        # 3D samples for ray idx (rays_o[idx]) are located in ray_indices[sample_idx:sample_idx+num_samples]

        ## resamples
        packed_info = pack_info(ray_indices, n_rays)
        (
            resampled_packed_info,
            resampled_midpoints,
            resampled_offsets,
            sampled_indices,
            sampled_fg_counts,
            sampled_bg_counts,
            surface_idx,
        ) = ray_resampling(
            packed_info,
            t_starts[..., None],
            t_ends[..., None],
            weights,
            sdfs,
            samples_per_pixel,
        )

        fg_indices = torch.nonzero(resampled_offsets < 1e4)[..., 0]
        bg_indices = torch.nonzero(resampled_offsets >= 1e4)[..., 0]
        resampled_ray_indices = unpack_info(
            resampled_packed_info, len(resampled_midpoints)
        )
        fg_resampled_ray_indices = resampled_ray_indices[fg_indices]
        bg_resampled_ray_indices = resampled_ray_indices[bg_indices]
        surface_idx = surface_idx[fg_resampled_ray_indices]
        # fg_resampled_midpoints = resampled_midpoints[fg_indices]
        # fg_resampled_offsets = resampled_offsets[fg_indices]
        fg_sampled_indices = sampled_indices[fg_indices]
        # bg_sampled_indices = sampled_indices[bg_indices]

    # Assemble resampled weights, positions, normals, albedo, roughness, metallic, t_dirs
    resampled_extras = {}
    if fg_sampled_indices.numel() > 0:
        resampled_weights = torch.zeros_like(resampled_midpoints[..., 0])
        # Weight of a resampled foreground point is the weight of the sampled
        # interval divided by the number of foreground re-samples in the interval
        resampled_fg_weights = (
            weights[fg_sampled_indices] / sampled_fg_counts[fg_sampled_indices].float()
        )
        # Weight of a resampled background point is the weight of the sampled
        # interval (here it is simply the transmittance of the ray) divided by
        # the number of background re-samples in the interval
        resampled_bg_weights = (
            transmittance_map[bg_resampled_ray_indices][..., 0]
            / sampled_bg_counts[bg_resampled_ray_indices].float()
        )
        resampled_weights.scatter_(0, fg_indices, resampled_fg_weights)
        resampled_weights.scatter_(0, bg_indices, resampled_bg_weights)

        t_dirs = rays_d[fg_resampled_ray_indices]
        cond = cond_pack[fg_resampled_ray_indices]
        _img_indices = img_indices[fg_resampled_ray_indices]
        # # For rays that we found zero-crossing point, we use the zero-crossing
        # # point as the resampled point for all samples. For rays that we did not
        # # find zero-crossing point, we use actual resampled points.
        # mask = surface_idx >= 0

        # t_coarse = ((t_starts + t_ends) / 2.0)
        # t = torch.zeros(
        #     (fg_sampled_indices.size(0),), device=t_coarse.device, dtype=t_coarse.dtype
        # )
        # t[mask] = t_coarse[surface_idx[mask]]
        # t[~mask] = t_coarse[fg_sampled_indices][~mask]

        # Ignore previous code-bloack. Zero-crossing on primary rays are now handled
        # inside the CUDA kernel.
        t = resampled_midpoints[fg_indices]

        # normals_coarse = extras["normals"]
        # normals = torch.zeros(
        #     fg_sampled_indices.size(0),
        #     3,
        #     device=normals_coarse.device,
        #     dtype=normals_coarse.dtype,
        # )
        # normals[mask] = normals_coarse[surface_idx[mask]]
        # normals[~mask] = normals_coarse[fg_sampled_indices][~mask]
        normals = extras["normals"][fg_sampled_indices]

        resampled_extras["sdf"] = sdfs[fg_sampled_indices]
        resampled_extras["alphas"] = extras["alphas"][fg_sampled_indices]
        resampled_extras["dists"] = (t_ends - t_starts)[..., None][fg_sampled_indices]
        resampled_extras["positions"] = (
            rays_o[fg_resampled_ray_indices]
            + rays_d[fg_resampled_ray_indices]
            * t
        )  # (n_fg_resamples, 3)

        resampled_extras["normals"] = normals
        resampled_extras["albedo"] = extras["albedo"][fg_sampled_indices]
        resampled_extras["roughness"] = extras["roughness"][fg_sampled_indices]
        resampled_extras["metallic"] = extras["metallic"][fg_sampled_indices]

        # (Inverse) incidence directions
        resampled_extras["t_dirs"] = t_dirs

        resampled_extras['cond'] = cond
        resampled_extras['img_indices'] = _img_indices
    else:
        resampled_weights = torch.zeros((0, 1), device=rays_o.device)

        resampled_extras["sdf"] = torch.zeros((0,), device=rays_o.device)
        resampled_extras["alphas"] = torch.zeros((0,), device=rays_o.device)
        resampled_extras["dists"] = torch.zeros((0, 1), device=rays_o.device)
        resampled_extras["positions"] = torch.zeros((0, 3), device=rays_o.device)
        resampled_extras["normals"] = torch.zeros((0, 3), device=rays_o.device)
        resampled_extras["albedo"] = torch.zeros((0, 3), device=rays_o.device)
        resampled_extras["roughness"] = torch.zeros((0, 1), device=rays_o.device)
        resampled_extras["metallic"] = torch.zeros(
            (0, extras["metallic"].size(-1)), device=rays_o.device
        )
        resampled_extras["t_dirs"] = torch.zeros((0, 3), device=rays_o.device)

    return (
        resampled_packed_info,
        resampled_ray_indices,
        resampled_weights,
        fg_indices,
        bg_indices,
        resampled_extras,
    )
