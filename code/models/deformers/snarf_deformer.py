#  Copyright (c) Meta Platforms, Inc. and affiliates.

from .fast_snarf.deformer_torch import ForwardDeformer
from .smplx import MANO
import torch
import torch.nn.functional as F

from torchgeometry.core import conversions

class SNARFDeformer():
    def __init__(self, config) -> None:
        self.is_rhand = True
        mano_layer = MANO(config.model_path, use_pca=False, flat_hand_mean=True, is_rhand=self.is_rhand, seal=True)


        ## faster if we remove this
        from .smplx.manohd.subdivide import sub_mano
        mano_layer ,_ ,_ = sub_mano(mano_layer, 2)
        self.body_model = mano_layer
        self.deformer = ForwardDeformer(config.deformer_config)
        # self.initialized = False
        self.opt = config.deformer_config

    def initialize(self, betas):        
        body_pose_t = torch.zeros((len(betas), 45)).cuda()
        global_orient = torch.zeros((len(betas), 3)).cuda()
        outputs = self.body_model(betas=betas, hand_pose=body_pose_t, global_orient=global_orient)
        
        ## inverse trans. mat.
        self.tfs_inv_t = torch.inverse(outputs.A.float().detach())

        ## canonical verts
        self.vs_template = self.body_model.v_template[None, :, :] + outputs.shape_offsets
        self.vs_template_seal = self.body_model.seal_mesh(self.vs_template, None, self.is_rhand)
        self.faces = self.body_model.faces
        self.faces_seal = self.body_model.faces_seal
        
        self.deformer.switch_to_explicit(resolution=self.opt.resolution,
                                         mano_verts=outputs.vertices.float().detach(),
                                         mano_weights=self.body_model.lbs_weights.clone()[None].detach())
        self.bbox = self.deformer.bbox
        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)

    def prepare_deformer(self, mano_params):
        if self.opt.optimize_betas:
            betas = mano_params["betas"] + mano_params["betas_correction"]
        else:
            betas = mano_params["betas"]
        self.body_model = self.body_model.cuda()
        # skin weights should be different per subject
        self.initialize(betas.detach())

        ## fwd MANO parameters -> posed mesh
        body_pose = mano_params["body_pose"] + mano_params["pose_correction"]
        global_orient = (
            mano_params["global_orient"] + mano_params["global_orient_correction"]
        )
        transl = mano_params["transl"] + mano_params["transl_correction"]
        outputs = self.body_model(
            betas=betas, hand_pose=body_pose, global_orient=global_orient, transl=transl
        )
        s2w = outputs.A[:, 0].float() ## MANO to world (using root tf mat)
        w2s = torch.inverse(s2w) ## world to MANO (using root tf mat)

        tfs = (w2s[:, None] @ outputs.A.float() @ self.tfs_inv_t).type(self.dtype)
        self.deformer.precompute(tfs)

        self.w2s = w2s

        ## posed vertices (converted into MANO space)
        self.vertices = (outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[
            :, None, :3, 3
        ]

        self.v3d_w = outputs.vertices
        
        num_bones = 16 ## MANO
        self.basic_joints = (
            outputs.joints[:, :num_bones] @ w2s[:, :3, :3].permute(0, 2, 1)
        ) + w2s[:, None, :3, 3]


        ## use corrected poses
        rot_mats = conversions.angle_axis_to_rotation_matrix(
            body_pose.reshape(-1, 3)
        )[:, :3, :3].reshape(-1, num_bones - 1, 3, 3)

        self.rot_mats = torch.cat(
            [
                torch.eye(3)[None].repeat(rot_mats.shape[0], 1, 1, 1).cuda(),
                rot_mats,
            ],
            dim=1,
        ).reshape(-1, num_bones, 9)
        self.tfs = tfs
        self.mano_outputs = outputs

    def transform_rays_w2s(self, rays):
        """transform rays from world to MANO coordinate system"""
        w2s = self.w2s.detach()
        # assert (w2s.shape[0] == 1)
        b = w2s.shape[0]
        rays = rays.view(b, -1, 8)

        w2s = w2s[:, None, :, :].repeat(1, rays.shape[1], 1, 1)
        rays = rays.view(-1, 8)
        w2s = w2s.view(-1, 4, 4)

        rays_o = torch.bmm(rays[:, :3][:, None, :], w2s[:, :3, :3].permute(0, 2, 1))[:, 0] + w2s[:, :3, 3]
        rays_d = torch.bmm(rays[:, 3:6][:, None, :], w2s[:, :3, :3].permute(0, 2, 1))[:, 0]
        d = torch.linalg.norm(rays_o, dim=-1, keepdim=True)

        near = d - 1
        far = d + 1

        return torch.cat([rays_o, rays_d, near, far], dim=-1)

    def transform_dirs_w2s(self, d):
        """transform directions from world to MANO coordinate system"""
        w2s = self.w2s.detach()
        assert w2s.shape[0] == 1
        return F.normalize(
            (d @ w2s[0, :3, :3].permute(1, 0)).to(d), dim=-1, eps=1e-6
        )


    def transform_dirs_s2w(self, d):
        """transform directions from MANO to world coordinate system"""
        if len(d.shape) == 2:
            d = d.unsqueeze(0)
        w2s = self.w2s.detach()
        return F.normalize(torch.bmm(d, w2s[:, :3, :3]).to(d), dim=-1, eps=1e-6)


    def transform_rots_s2w(self, J_inv):
        """apply MANO->world rotation to J_inv"""
        w2s = self.w2s.detach()
        return (w2s[:, :3, :3].permute(0, 2, 1) @ J_inv).to(J_inv)

    def get_bbox_deformed(self):
        voxel = self.deformer.voxel_d[0].reshape(3, -1)
        return [voxel.min(dim=1).values, voxel.max(dim=1).values]

    def deform_(self, pts, eval_mode):
        """transform pts to canonical space"""
        batch_size, point_size = pts.shape[:2]
        betas = self.mano_outputs.betas
        assert len(pts.shape) == 3
        assert pts.shape[0] == betas.shape[0]
        assert pts.shape[2] == 3

        pts_cano, others = self.deformer.forward(pts, cond=None, tfs=self.tfs, eval_mode=eval_mode)
        valid = others["valid_ids"].reshape(batch_size, point_size, -1) ## torch.Size([7, 21788, 16])

        if self.opt.use_j_inv:
            # Use the (approximated) inverse Jacobian to transform normals
            J_inv = others["J_inv"].reshape(batch_size, point_size, -1, 3, 3)
        else:
            # Use the linearly blended bone transforms to transform normals
            J_inv = others["fwd_tfs"].reshape(batch_size, point_size, -1, 3, 3)
        return pts_cano.reshape(batch_size, point_size, -1, 3), J_inv, valid


    def deform(self, pts, pts_valid, cond, model, eval_mode):
        ## world -> canonical
        pts_cano_all, J_inv, valid = self.deform_(pts.type(self.dtype), eval_mode=eval_mode)
        valid = valid & pts_valid[:, :, None] ## bug fix
        c2w = J_inv[valid]

        sdf_cano = torch.ones_like(pts_cano_all[..., 0]).float() * 1e5  # canonical SDF
        sdf_grad = None # SDF gradient in observation space
        sdf_grad_cano = None    # SDF gradient in canonical space
        features = None # features from the canonical SDF field
        laplace = None  # Laplacian of the canonical SDF field

        if not torch.isfinite(pts_cano_all).all():
            print("WARNING: NaN found in pts_cano_all")
        # Note that model should also take care of the case where input Tensor is empty
        # (i.e. all points are invalid), and return empty tensors

        model_ret, J_inv_nr = model(pts_cano_all, valid, cond)
        c2w = c2w @ J_inv_nr
        any_valid = valid.any()
        
        if isinstance(model_ret, tuple) or isinstance(model_ret, list):
            if any_valid:
                sdf_cano[valid] = model_ret[0]
            if len(model_ret) > 1:
                sdf_grad_cano = torch.tensor(
                    [0, 0, 1], dtype=torch.float32
                ).cuda().repeat((*pts_cano_all.shape[:3], 1))
                if any_valid:
                    sdf_grad_cano[valid] = model_ret[1]
                assert c2w is not None
                sdf_grad = torch.tensor(
                    [0, 0, 1], dtype=torch.float32
                ).cuda().repeat((*pts_cano_all.shape[:3], 1))
                if any_valid:
                    ### C->D I think
                    sdf_grad[valid] = torch.einsum(
                        "bij,bj->bi", c2w, sdf_grad_cano[valid]
                    )
            if len(model_ret) > 2:
                features = torch.zeros(
                    (*pts_cano_all.shape[:3], model_ret[2].size(-1)),
                    dtype=torch.float32,
                ).cuda()
                
                if any_valid:
                    features[valid] = model_ret[2]
            if len(model_ret) > 3:
                laplace = torch.zeros_like(pts_cano_all[..., 0]).float()
                if any_valid:
                    laplace[valid] = model_ret[3]
        elif isinstance(model_ret, torch.Tensor):
            if any_valid:
                sdf_cano[valid] = model_ret
        else:
            raise ValueError("Invalid return type from model")
        batch_size, num_pts, num_inits = sdf_cano.shape
        sdf_cano = sdf_cano.view(batch_size*num_pts, num_inits)
        
        pts_cano_all = pts_cano_all.view(batch_size*num_pts, num_inits, 3)
        sdf_cano, idx = torch.min(sdf_cano, dim=-1)
        pts_cano = torch.gather(pts_cano_all, 1, idx[:, None, None].repeat(1, 1, 3))
        pts_cano = pts_cano.view(batch_size, num_pts, 3)
        sdf_cano = sdf_cano.view(batch_size, num_pts)

        valid_cano = valid.any(dim=-1)
        valid_cano = valid_cano.view(batch_size, num_pts)

        out = [pts_cano, sdf_cano, valid_cano]


        if sdf_grad is not None:
            sdf_grad = sdf_grad.view(batch_size*num_pts, num_inits, 3)
            sdf_grad = torch.gather(sdf_grad, 1, idx[:, None, None].repeat(1, 1, 3))
            sdf_grad = sdf_grad.view(batch_size, num_pts, 3)
            out.append(sdf_grad)
        if sdf_grad_cano is not None:
            sdf_grad_cano = sdf_grad_cano.view(batch_size*num_pts, num_inits, 3)
            sdf_grad_cano = torch.gather(sdf_grad_cano, 1, idx[:, None, None].repeat(1, 1, 3))
            sdf_grad_cano = sdf_grad_cano.view(batch_size, num_pts, 3)
            out.append(sdf_grad_cano)
        if features is not None:
            features = features.view(batch_size*num_pts, num_inits, features.shape[-1])
            features = torch.gather(
                features, 1, idx[:, None, None].repeat(1, 1, features.size(-1))
            ).reshape(-1, features.size(-1))
            features = features.view(batch_size, num_pts, features.shape[-1])
            out.append(features)
        if laplace is not None:
            laplace = laplace.view(batch_size*num_pts, num_inits)
            laplace = torch.gather(laplace, 1, idx[:, None])
            laplace = laplace.view(batch_size, num_pts)
            out.append(laplace)
        return out

    def __call__(self, pts, pts_valid, cond, model, eval_mode=True):
        return self.deform(pts, pts_valid, cond, model, eval_mode)
