#  Copyright (c) Meta Platforms, Inc. and affiliates.

from pytorch3d import ops
import torch
from torch import einsum
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

def skinning(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    
    x_h = F.pad(x, (0, 1), value=1.0)
    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs) #### CHECK!!
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
    return x_h[:, :, :3], w_tf[:, :, :3, :3]

class ForwardDeformer(torch.nn.Module):
    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt
        self.soft_blend = 20
        self.init_bones = list(range(16)) ## https://github.com/Skype-line/X-Avatar/blob/7a071b62bce646f49a44b5459e0f956d52e1ffae/code/lib/model/deformer.py#L57
        self.init_bones_cuda = torch.tensor(self.init_bones).int()

        # the bounding box should be slighter larger than the actual mesh
        self.global_scale = 1.2
        self.version = opt.get("version", 1)
        cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
        print('Compiling JIT')
        self.fuse_kernel = load(name='fuse_cuda',
                        extra_cuda_cflags=[],
                        sources=[f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
                                    f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel_fast.cu'])
        self.filter_cuda = load(name='filter',
                        sources=[f'{cuda_dir}/filter/filter.cpp',
                                    f'{cuda_dir}/filter/filter.cu'])
        self.precompute_cuda = load(name='precompute',
                            sources=[f'{cuda_dir}/precompute/precompute.cpp',
                                        f'{cuda_dir}/precompute/precompute.cu'])
        print('Done compiling')
        ## NOTE: if stuck above, run this find ~/.cache/torch_extensions/ -delete
        ## this needs to be clean and recompile for different numbers of gpus on nodes



    def forward(self, xd, cond, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        xc_opt, others = self.search(xd, cond, tfs, eval_mode=True)
        
        if eval_mode:
            xc_opt_ = xc_opt.detach()
            xc_opt_[~others['valid_ids']] = 0
            n_batch, n_point, n_init, n_dim = xc_opt_.shape

            mask = others['valid_ids']
            _, fwd_tfs = self.forward_skinning(xc_opt_, cond, tfs, mask=mask)
            others['fwd_tfs'] = fwd_tfs
            return xc_opt, others

        if self.version == 1:
            xc_opt = xc_opt.detach()
            xc_opt[~others['valid_ids']] = 0
            n_batch, n_point, n_init, n_dim = xc_opt.shape

            mask = others['valid_ids']
            xd_opt, fwd_tfs = self.forward_skinning(xc_opt, cond, tfs, mask=mask)

            grad_inv = others['J_inv']#[others['valid_ids']]
            others['fwd_tfs'] = fwd_tfs
            correction = xd_opt - xd_opt.detach()
            # correction = bmv(-grad_inv, correction.unsqueeze(-1)).squeeze(-1)
            correction = bmv(-grad_inv.view(-1, 3, 3), correction.view(-1, 3, 1)).view(n_batch, n_point, n_init, n_dim)

            # trick for implicit diff with autodiff:
            # xc = xc_opt + 0 and xc' = correction'
            xc = xc_opt
            xc += correction
            xc[~others['valid_ids']] = 0
            xc = xc.reshape(n_batch, n_point, n_init, n_dim)
            return xc, others
        else:
            assert False

    def precompute(self, tfs):
        # b, c, d, h, w = tfs.shape[0], 3, self.resolution // 4, self.resolution, self.resolution
        # b, d, h, w = tfs.shape[0], self.resolution // 2, self.resolution, self.resolution
        b, d, h, w = tfs.shape[0], self.resolution, self.resolution, self.resolution
        voxel_d = torch.zeros((b, 3, d, h, w)).cuda()
        voxel_J = torch.zeros((b, 12, d, h, w)).cuda()
        self.precompute_cuda.precompute(self.lbs_voxel_final, tfs, voxel_d, voxel_J, self.offset_kernel, self.scale_kernel)
        self.voxel_d = voxel_d
        self.voxel_J = voxel_J

    def search(self, xd, cond, tfs, eval_mode=False):
        """Search correspondences.
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        with torch.no_grad():
            result = self.broyden_cuda(xd, self.voxel_d, self.voxel_J, tfs)
        return result['result'], result

    def broyden_cuda(self, xd_tgt, voxel, voxel_J_inv, tfs, cvg_thresh=1e-5, dvg_thresh=1e-1):
        b, n, _ = xd_tgt.shape
        n_init = self.init_bones_cuda.shape[0]
        self.init_bones_cuda = self.init_bones_cuda.cuda()

        xc_init_IN = torch.zeros((b, n, n_init, 3), dtype=torch.float32).cuda()
        J_inv_init_IN = torch.zeros((b, n, n_init, 3, 3), dtype=torch.float32).cuda()
        is_valid = torch.zeros((b, n, n_init), dtype=torch.bool).cuda()
        self.fuse_kernel.fuse_broyden(xc_init_IN, xd_tgt, voxel, voxel_J_inv, tfs,
                                 self.init_bones_cuda, True, J_inv_init_IN,
                                 is_valid, self.offset_kernel,
                                 self.scale_kernel, cvg_thresh, dvg_thresh)
        mask = self.filter_cuda.filter(xc_init_IN, is_valid)
        return {
            "result": xc_init_IN,
            'valid_ids': mask,
            'J_inv': J_inv_init_IN
        }

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point
        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        weights = self.query_weights(xc, cond, mask=mask)
        batch_size, num_pts, num_jts = xc.shape[:3]
        ### butchering it!!
        x_h, w_tfs = skinning(
            xc.view(batch_size, num_pts*num_jts, 3),
            weights.view(batch_size, num_pts*num_jts, 16),
            tfs, inverse=False)
        x_h = x_h.view(batch_size, num_pts, num_jts, 3)
        w_tfs = w_tfs.view(batch_size, num_pts, num_jts, 3, 3)
        x_h[~mask] = 0
        w_tfs[~mask]  = 0
        return x_h, w_tfs    

    def switch_to_explicit(self, resolution=32, mano_verts=None, mano_weights=None):
        self.resolution = resolution
        # convert to voxel grid
        d, h, w = resolution, resolution, resolution
        b = mano_verts.shape[0]

        x_range = (torch.linspace(-1, 1, steps=w).cuda()).view(1, 1, 1, w).expand(1, d, h, w)
        y_range = (torch.linspace(-1, 1, steps=h).cuda()).view(1, 1, h, 1).expand(1, d, h, w)
        z_range = (torch.linspace(-1, 1, steps=d).cuda()).view(1, d, 1, 1).expand(1, d, h, w)
        grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3, -1).permute(0, 2, 1)

        ## tight box around MANO in canonical
        gt_bbox = torch.FloatTensor([[-0.0935, -0.0365, -0.0859],
                [ 0.1366,  0.0417,  0.1158]]).cuda()
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5 ## center of bbox
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * self.global_scale ## size of bbox with 20% padding

        corner = torch.ones_like(offset[0]) * scale
        min_vert = (offset - corner).reshape(1, 3)
        max_vert = (offset + corner).reshape(1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=0)

        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

        self.register_buffer('offset_kernel', -self.offset)
        scale_kernel = torch.zeros_like(self.offset)
        scale_kernel[...] = 1. / self.scale
        self.register_buffer('scale_kernel', scale_kernel)

        def normalize(x):
            x_normalized = x.clone()
            x_normalized -= self.offset
            x_normalized /= self.scale
            return x_normalized
        
        def denormalize(x):
            x_denormalized = x.clone()
            x_denormalized *= self.scale
            x_denormalized += self.offset
            return x_denormalized

        self.normalize = normalize
        self.denormalize = denormalize
        
        grid_denorm = self.denormalize(grid).repeat(b, 1, 1)
        weights = construct_weight_grid(grid_denorm,
                                        mano_verts=mano_verts.detach().clone(),
                                        mano_weights=mano_weights.detach().clone(),
                                        resolution=resolution).detach().clone()
        

        self.register_buffer('lbs_voxel_final', weights.detach())
        self.register_buffer('grid_denorm', grid_denorm)

        def query_weights(my_xc, cond=None, mask=None, mode='bilinear'):
            """
            Query skinning weights from points.
            
            Args:
                my_xc (torch.Tensor): Input points with shape (B, N, M, 3) where B is the batch size, 
                                    N is the number of points per sample, and M is another dimension.
                cond (None): Not used in this function.
                mask (None): Not used in this function.
                mode (str): Mode for grid sampling. Defaults to 'bilinear'.
            
            Returns:
                torch.Tensor: Skinning weights with shape (B, N, M, K) where K is the number of bones.
            """
            # Get the batch size and other dimensions
            B, N, M, _ = my_xc.shape
            
            # Reshape my_xc to (B, -1, 3) for grid sampling
            xc = my_xc.view(B, -1, 3)
            
            # Expand self.lbs_voxel_final to match the batch size
            lbs_voxel_final_expanded = self.lbs_voxel_final.expand(B, -1, -1, -1, -1)
            
            # Normalize xc
            normalized_xc = self.normalize(xc)[:, :, None, None]
            
            # Perform grid sampling
            w = F.grid_sample(
                lbs_voxel_final_expanded,
                normalized_xc, 
                align_corners=True,
                mode=mode,
                padding_mode='border'
            )
            
            # Squeeze and permute w to get the desired shape
            w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            
            # Reshape w to match the original shape of my_xc
            w = w.view(B, N, M, -1)
            return w
        self.query_weights = query_weights

def skinning_mask(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    p, n = w.shape
    w_tf = einsum("pn,nij->pij", w, tfs.squeeze(0))
    x_h = x_h.view(p, 1, 4).expand(p, 4, 4)
    x_h = (w_tf * x_h).sum(-1)
    return x_h[:, :3], w_tf[:, :3, :3]


def bmv(m, v):
    return (m * v.transpose(-1, -2).expand(-1, 3, -1)).sum(-1, keepdim=True)



def construct_weight_grid(x, mano_verts, mano_weights, resolution=None):
    
    dist, idx, _ = ops.knn_points(x, mano_verts.detach(), K=5)
    dist = dist.sqrt().clamp_(0.0001, 1.)
    weights = mano_weights[0, idx]
    
    ws = 1. / dist
    ws = ws / ws.sum(-1, keepdim=True)
    
    weights = (ws[..., None] * weights).sum(-2)
    num_bones = 16
    c, d, h, w = num_bones, resolution, resolution, resolution
    b = weights.shape[0]
    weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
    return weights.detach()
