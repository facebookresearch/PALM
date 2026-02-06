#  Copyright (c) Meta Platforms, Inc. and affiliates.

import gc
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn

# from torch_scatter import scatter_min


def pack_without_loop(
        img_indices,
        num_images,
        positions=None, cond=None, t_dirs=None, t_starts=None, t_ends=None):
    """
    Pack inputs with zeros to enable batched operations (loop-free version).

    Args:
        positions (torch.Tensor): Positions tensor.
        cond (torch.Tensor): Condition tensor.
        t_dirs (torch.Tensor): Directions tensor.
        t_starts (torch.Tensor): Start points tensor.
        t_ends (torch.Tensor): End points tensor.
        img_indices (torch.Tensor): Image indices tensor.

    Returns:
        tuple: Packed tensors.
    """
    num_pts = img_indices.shape[0]
    assert img_indices.dim() == 1, "img_indices should be a 1D tensor"
    if positions is not None:
        assert num_pts == positions.shape[0]
        assert 3 == positions.shape[1]
    
    if cond is not None:
        assert cond.dim() == 2, "cond should be a 2D tensor"
        assert cond.shape[0] == num_images
    
    if t_dirs is not None:
        assert t_dirs.shape[0] == num_pts
        assert t_dirs.dim() == 2 and t_dirs.size(1) == 3, "t_dirs should be a 2D tensor with shape (N, 3)"
    
    if t_starts is not None:
        assert t_starts.shape[0] == num_pts
        assert t_starts.dim() == 1, "t_starts should be a 1D tensor"
    
    if t_ends is not None:
        assert t_ends.shape[0] == num_pts
        assert t_ends.dim() == 1, "t_ends should be a 1D tensor"

    # num_images = cond.shape[0]
    counts = torch.bincount(img_indices)
    max_count, _ = counts.max(), counts.argmax()
    
    # Create a tensor with cumulative counts
    cum_counts = torch.cumsum(counts, dim=0)
    cum_counts = torch.cat((torch.tensor([0]).cuda(), cum_counts))

    # Use advanced indexing to fill packed tensors
    idxs = torch.arange(img_indices.shape[0]).cuda()
    img_idxs = img_indices[idxs]
    pos_idxs = idxs - cum_counts[img_idxs]
    out = {}
    positions_valid = torch.zeros((num_images, max_count)).cuda().bool()
    positions_valid[img_idxs, pos_idxs] = True        
    out['positions_valid'] = positions_valid

    if positions is not None:
        positions_packed = torch.zeros((num_images, max_count, 3)).cuda()
        positions_packed[img_idxs, pos_idxs] = positions[idxs]
        out['positions_packed'] = positions_packed

    if cond is not None:
        cond_packed = torch.zeros((num_images, max_count, cond.shape[1])).cuda()
        cond_packed[img_idxs, pos_idxs] = cond[img_idxs]
        out['cond_packed'] = cond_packed

    if t_dirs is not None:
        t_dirs_packed = torch.zeros((num_images, max_count, 3)).cuda()
        t_dirs_packed[img_idxs, pos_idxs] = t_dirs[idxs]
        out['t_dirs_packed'] = t_dirs_packed


    if t_starts is not None:
        t_starts_packed = torch.zeros((num_images, max_count)).cuda()
        t_starts_packed[img_idxs, pos_idxs] = t_starts[idxs]
        out['t_starts_packed'] = t_starts_packed

    if t_ends is not None:
        t_ends_packed = torch.zeros((num_images, max_count)).cuda()
        t_ends_packed[img_idxs, pos_idxs] = t_ends[idxs]
        out['t_ends_packed'] = t_ends_packed
    return out, max_count

def bbox2scale(bbox):
    assert len(bbox.shape) == 2
    assert bbox.shape[0] == 2
    assert bbox.shape[1] == 3
    center = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max() /2.0
    return center, scale



def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[1]
            break
    out = defaultdict(list)
    out_type = None
    # from tqdm import tqdm
    
    for i in range(0, B, chunk_size):
    # for i in tqdm(range(0, B, chunk_size)):
        out_chunk = func(
            *[
                (
                    arg[:, i : i + chunk_size]
                    if isinstance(arg, torch.Tensor) and arg.shape[1] == B
                    else arg
                )
                for arg in args
            ],
            **kwargs,
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)

    if out_type is None:
        return
    # out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    out = {k: torch.cat(v, dim=1) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


def chunk_batch_dim0(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[1]
            break
    out = defaultdict(list)
    out_type = None
    # from tqdm import tqdm
    
    for i in range(0, B, chunk_size):
    # for i in tqdm(range(0, B, chunk_size)):
        out_chunk = func(
            *[
                (
                    arg[:, i : i + chunk_size]
                    if isinstance(arg, torch.Tensor) and arg.shape[1] == B
                    else arg
                )
                for arg in args
            ],
            **kwargs,
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)

    if out_type is None:
        return
    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    # out = {k: torch.cat(v, dim=1) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out

def chunk_batch_inference(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[1]
            break
    out = defaultdict(list)
    out_type = None
    from tqdm import tqdm
    # for i in range(0, B, chunk_size):
    for i in tqdm(range(0, B, chunk_size)):
        out_chunk = func(
            *[
                (
                    arg[:, i : i + chunk_size]
                    if isinstance(arg, torch.Tensor) and arg.shape[1] == B
                    else arg
                )
                for arg in args
            ],
            **kwargs,
        )

        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)

    if out_type is None:
        return
    
    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out

def chunk_batch_original(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(
            *[
                (
                    arg[i : i + chunk_size]
                    if isinstance(arg, torch.Tensor) and arg.shape[0] == B
                    else arg
                )
                for arg in args
            ],
            **kwargs,
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)

    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name.startswith("scale"):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0.0, scale_factor) / scale_factor
    elif name.startswith("clamp"):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0.0, clamp_max)
    elif name.startswith("mul"):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name.startswith("+") or name.startswith("-"):
        return lambda x: x + float(name)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (
            torch.arange(bins, device=sigma.device, dtype=sigma.dtype) + 0.5
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = (
            torch.exp(-0.5 * (x / self.sigma) ** 2)
            / (self.sigma * np.sqrt(np.pi * 2))
            * self.delta
        )
        x = x.sum(dim=1)
        return x


# Utilities from InstantAvatar codebase.
# https://github.com/tijiang13/InstantAvatar
def max_connected_component(grid):
    grid = grid[None]
    comp = (
        torch.arange(1, grid.numel() + 1, device=grid.device)
        .reshape(grid.shape)
        .float()
    )
    comp[~grid] = 0
    for _ in range(grid.shape[-1] * 3):
        comp = F.max_pool3d(comp, kernel_size=3, stride=1, padding=1)
        comp *= grid
    return comp.squeeze(0).squeeze(0)


# Utilities for spherical harmonics. Adapeted from RefNeRF codebase.
# https://github.com/google-research/multinerf
def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.
    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).
    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).
    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return (
        (-1) ** m
        * 2**l
        * np.math.factorial(l)
        / np.math.factorial(k)
        / np.math.factorial(l - k - m)
        * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return np.sqrt(
        (2.0 * l + 1.0)
        * np.math.factorial(l - m)
        / (4.0 * np.pi * np.math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    # Default from RefNeRF, l = [1, 2, 4, 8, 16, ..., 2**(deg_view-1)]
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array


def get_complex_to_real(deg_view):
    sh_complex_to_real = []
    ls = [2**deg for deg in range(deg_view)]
    # Nonnegative m
    for l in ls:
        coeff = np.ones(l + 1, dtype=np.float32)
        for m in range(1, l + 1):
            coeff[m] = np.sqrt(2) * (-1) ** m

        sh_complex_to_real.append(coeff)

    # Negative m
    for l in ls:
        coeff = np.ones(l, dtype=np.float32)
        for m in range(l):
            coeff[m] = np.sqrt(2) * (-1) ** (m + 1)

        sh_complex_to_real.append(coeff)

    sh_complex_to_real = np.concatenate(sh_complex_to_real, axis=0)

    return sh_complex_to_real


def get_perspective(fov, theta, phi, height, width):
    #
    # theta is left/right angle, phi is up/down angle, both in degree
    #

    f = 0.5 * width * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        np.float32,
    )

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    R = R2 @ R1

    return K.astype(np.float32), R.T.astype(np.float32)


def compute_albedo_rescale_factor(gt_albedo, pred_albedo, gt_mask):
    # Align predicted albedo with GT in linear RGB space
    three_channel_ratio = []
    for i in range(gt_albedo.shape[-1]):
        x = gt_albedo[gt_mask][:, i]
        x_hat = pred_albedo[gt_mask][:, i]
        scale = torch.sum(x * x_hat) / torch.sum(x_hat * x_hat)
        three_channel_ratio.append(scale)

    return torch.stack(three_channel_ratio, dim=0)


import numpy as np
import kaolin
import torch
from typing import Union
from torch import Tensor
from nerfacc import (
    RayIntervals,
    render_weight_from_alpha,
)

from lib.nerfacc import (
    ray_resampling_merge,
    pack_info,
    unpack_info,
)
from models.utils import pack_without_loop



def _meshgrid3d(
    res: Tensor, device: Union[torch.device, str] = "cpu"
) -> Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.long),
                torch.arange(res[1], dtype=torch.long),
                torch.arange(res[2], dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).to(device)


def compute_mano_cano_sdf(mesh_v_cano, mesh_f_cano, mesh_face_vertices, x_cano):
    distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        x_cano.contiguous(), mesh_face_vertices
    )

    distance = torch.sqrt(distance)  # kaolin outputs squared distance

    # inside or not
    sign = kaolin.ops.mesh.check_sign(mesh_v_cano, mesh_f_cano, x_cano).float()

    # inside: 1 -> 1 - 2 = -1, negative
    # outside: 0 -> 1 - 0 = 1, positive
    sign = 1 - 2 * sign
    signed_distance = sign * distance  # SDF of points to mesh
    return signed_distance

def subdivide_cano(mesh_vh_cano, mesh_fh_cano):
    import trimesh
    cano_v = mesh_vh_cano[0].cpu().detach().numpy()
    mano_f = mesh_fh_cano.cpu().detach().numpy()
    cano_v_div, mano_f_div = trimesh.remesh.subdivide_loop(cano_v, mano_f, iterations=1)

    mesh_v_cano_div = torch.FloatTensor(cano_v_div).cuda()
    mesh_f_cano_div = torch.LongTensor(mano_f_div).cuda()

    return mesh_v_cano_div[None, :, :], mesh_f_cano_div



def sample_on_barycentric_mesh(verts, faces, num_samples):
    # Ensure that the random tensors are created on the same device as the input tensors
    device = verts.device

    batch_size, num_verts, _ = verts.shape
    num_faces = faces.shape[0]

    # Randomly select faces
    face_indices = torch.randint(0, num_faces, (batch_size, num_samples), device=device)

    # Gather the vertices corresponding to the faces
    sampled_faces = faces[face_indices]

    # Gather the coordinates of the vertices of the sampled faces
    v0 = torch.gather(verts, 1, sampled_faces[..., 0].unsqueeze(-1).expand(-1, -1, 3))
    v1 = torch.gather(verts, 1, sampled_faces[..., 1].unsqueeze(-1).expand(-1, -1, 3))
    v2 = torch.gather(verts, 1, sampled_faces[..., 2].unsqueeze(-1).expand(-1, -1, 3))

    # Sample random barycentric coordinates
    u = torch.rand((batch_size, num_samples, 1), device=device)
    v = torch.rand((batch_size, num_samples, 1), device=device)

    # If the sum of u, v exceeds 1, we flip the coordinates to ensure the points lie within the triangle
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]

    # Compute the sampled points using the barycentric coordinates
    samples = u * v0 + v * v1 + (1 - u - v) * v2

    return samples




class PointInSpace:
    def __init__(self, global_sigma=0.5, global_sigma_xyz=None, local_sigma=0.01):
        # self.global_sigma = global_sigma
        if global_sigma_xyz is None:
            self.global_sigma_xyz = torch.ones(3) * global_sigma
        else:
            self.global_sigma_xyz = torch.FloatTensor(np.array(global_sigma_xyz))
        self.local_sigma = local_sigma

    def get_points(self, pc_input=None, local_sigma=None, global_ratio=0.125):
        """Sample one point near each of the given point + 1/8 uniformly.
        Args:
            pc_input (tensor): sampling centers. shape: [B, N, D]
        Returns:
            samples (tensor): sampled points. shape: [B, N + N / 8, D]
        """
        if self.global_sigma_xyz.device != pc_input.device:
            self.global_sigma_xyz = self.global_sigma_xyz.to(pc_input.device)

        batch_size, sample_size, dim = pc_input.shape
        if local_sigma is None:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
        sample_global = (
            torch.rand(
                batch_size, int(sample_size * global_ratio), dim, device=pc_input.device
            )
            * (self.global_sigma_xyz * 2)
        ) - self.global_sigma_xyz

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample
    


def get_alpha(density_fn, sdf, dists):
    density = density_fn(sdf)
    alpha = 1.0 - torch.exp(-density * dists[:, 0])
    return alpha


def coarse_alpha_fn(
        rays_o, rays_d, intervals, geometry, cond, render_step_size, density, deformer, training,
        num_images, _img_indices,
        dists=None, sdf_prev=None, is_resampled=None):
    
    if is_resampled is None:
        ray_indices = intervals.ray_indices
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * intervals.vals[..., None]
    else:
        ray_indices = intervals.ray_indices[is_resampled]
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * intervals.vals[is_resampled][..., None]
    if t_origins.shape[0] == 0:
        return torch.zeros((0,), device=t_origins.device), torch.zeros(
            (0,), device=t_origins.device
        )
    
    img_indices = _img_indices.clone()
    img_indices = img_indices[ray_indices]
    packed, max_count = pack_without_loop(img_indices, num_images, positions=positions, cond=cond)
    positions_packed = packed['positions_packed']
    positions_valid = packed['positions_valid']
    cond_packed = packed['cond_packed']    

    def geometry_fn(x, valid, cond):
        return geometry(
            x[valid],
            cond[:, :, None, :].repeat(1, 1, x.shape[2], 1)[valid],
            with_grad=False, with_feature=False, with_laplace=False
        )
    
    _, sdf_curr, *others = deformer(
        positions_packed,
        positions_valid,
        cond_packed,
        geometry_fn,
        with_jac=False,
        eval_mode=not training,
    )
    sdf_curr = sdf_curr[positions_valid]
    if sdf_prev is not None:
        sdf = torch.ones_like(intervals.vals) * 1e10
        sdf[is_resampled] = sdf_curr
        sdf[~is_resampled] = sdf_prev
    else:
        sdf = sdf_curr

    sdf_merge = torch.ones_like(intervals.vals) * 1e10
    sdf_min = torch.minimum(sdf[intervals.is_left], sdf[intervals.is_right])
    sdf_merge[intervals.is_left] = sdf_min

    if dists is None:
        dists = (render_step_size * torch.ones_like(intervals.vals))

    # VolSDF does not need normal and t_dirs to compute alpha
    alphas = get_alpha(density, sdf_merge, dists[..., None])
    return alphas, sdf_merge

def alpha_fn(
        rays_o, 
        rays_d, 
        intervals,
        geometry,
        cond,
        deformer,
        training,
        num_images, _img_indices,
        density
        ):
    ray_indices = intervals.ray_indices[intervals.is_left]
    t_starts = intervals.vals[intervals.is_left]
    t_ends = intervals.vals[intervals.is_right]
    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]
    positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
    if t_origins.shape[0] == 0:
        return torch.zeros((0,), device=t_origins.device)

    img_indices = _img_indices.clone()
    img_indices = img_indices[ray_indices]

    packed, max_count = pack_without_loop(img_indices, num_images, positions=positions, cond=cond)
    positions_packed = packed['positions_packed']
    positions_valid = packed['positions_valid']
    cond_packed = packed['cond_packed']   
    
    def geometry_fn(x, valid, cond):
        return geometry(
            x[valid],
            cond[:, :, None, :].repeat(1, 1, x.shape[2], 1)[valid],
            with_grad=False, with_feature=False, with_laplace=False
        )

    _, sdf_curr, *others = deformer(
        positions_packed,
        positions_valid,
        cond_packed,
        geometry_fn,
        with_jac=False,
        eval_mode=not training,
    )
    sdf_curr = sdf_curr[positions_valid]

    sdf = torch.ones_like(intervals.vals) * 1e10
    sdf[intervals.is_left] = sdf_curr
    dists = torch.zeros_like(intervals.vals)
    dists[intervals.is_left] = t_ends - t_starts

    # VolSDF does not need normal and t_dirs to compute alpha
    alphas = get_alpha(density, sdf, dists[..., None])
    return alphas


def importance_upsampling(
    rays_o, rays_d, geometry_fn, cond, render_step_size, density_fn, deformer, training,intervals, num_images, img_indices,
):
    n_rays = rays_o.shape[0]
    resampled_dists = None
    sdf = None
    is_resampled = None
    for upsample_iter in range(2):
        with torch.no_grad():
            if upsample_iter == 0:
                alphas, sdf = coarse_alpha_fn(
                    rays_o, rays_d, 
                    intervals, geometry_fn, cond, render_step_size, density_fn, deformer, training,
                    num_images, img_indices,
                    resampled_dists, sdf, is_resampled
                )
            else:
                alphas = alpha_fn(
                    rays_o, rays_d, intervals, geometry_fn, cond, deformer, training, num_images, img_indices, density_fn)
            # if alphas.numel() == 0 or sdf.numel() == 0:
            #     break
            weights, _ = render_weight_from_alpha(
                alphas, ray_indices=intervals.ray_indices, n_rays=n_rays
            )
            # packed_info = pack_info(ray_indices=ray_indices, n_rays=n_rays)
            (
                resampled_packed_info,
                resampled_vals,
                resampled_dists,
                resampled_is_left,
                resampled_is_right,
                is_resampled,
                is_fg_sample,
            ) = ray_resampling_merge(
                intervals.packed_info.int(),
                intervals.vals,
                intervals.is_left,
                intervals.is_right,
                weights,
                n_samples=16,
            )

        # Keep only foreground samples
        resampled_ray_indices = unpack_info(
            resampled_packed_info, len(resampled_vals)
        )[is_fg_sample]
        resampled_packed_info = pack_info(resampled_ray_indices, n_rays)
        resampled_dists = resampled_dists[is_fg_sample]
        is_resampled = is_resampled[is_fg_sample]

        # t_starts = resampled_vals[resampled_is_left]
        # t_ends = resampled_vals[resampled_is_right]
        # ray_indices = resampled_ray_indices[resampled_is_left[is_fg_sample]]

        intervals = RayIntervals(
                vals=resampled_vals[is_fg_sample],
                is_left=resampled_is_left[is_fg_sample],
                is_right=resampled_is_right[is_fg_sample],
                ray_indices=resampled_ray_indices,
                packed_info=resampled_packed_info,
            )

    return intervals


def visualize_sdf(sdf_vals, samples, midx):
    from common.xmesh import color_pointcloud
    import matplotlib.pyplot as plt
    cmap = plt.cm.bwr
    sdf_colors = sdf_vals.detach().cpu().numpy().reshape(-1)*10
    sdf_colors += 0.5
    sdf_colors = cmap(sdf_colors)
    mymesh = color_pointcloud(samples, sdf_colors, radius=0.001)
    mymesh.export(f'./0_scratch/sdf_{midx:05d}.obj')
