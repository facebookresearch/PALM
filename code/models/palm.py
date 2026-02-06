#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base import BaseModel
from models.utils import chunk_batch_inference
from systems.utils import update_module_step
from lib.torch_pbr.utils.nvdiffrecmc_util import rgb_to_srgb
from pytorch3d.ops import knn_points

# nerfacc-0.5.3 - for basic occupancy grid functionalities and general volume rendering
from nerfacc import (
    accumulate_along_rays,
)
from models.occ_grid.temporal_occ_grid import TemporalOccGridEstimator
from models.volrend import (
    rendering_with_normals_sdf,
    rendering_with_normals_mats_sdf,
)
from models.pbr.utils import sample_volume_interaction
from lib.nerfacc import (
    pack_data,
    unpack_data,
)
from models.utils import (
    get_alpha,
    importance_upsampling,
)
from common.transforms import transform_points_batch
from models.utils import pack_without_loop

class PALMNet(BaseModel):
    def setup(self):
        # Radiance field modules
        ## geometry.name == 'volume-sdf'
        from models.rf.geometry import VolumeSDF
        self.geometry = VolumeSDF(self.config.geometry)

        ## density.name == 'learned-laplace-density'
        from models.rf.density import LearnedLaplaceDensity
        self.density = LearnedLaplaceDensity(self.config.density)

        ## radiance.name == 'volume-ref-dir-radiance'
        from models.rf.radiance import VolumeRefDirRadiance
        self.radiance = VolumeRefDirRadiance(self.config.radiance)

        from models.pose.pose_encoder import FlatPoseEncoder
        self.pose_encoder = FlatPoseEncoder(self.config)

        ## SNARFDeformer
        from models.deformers.deformer import SNARFDeformer
        self.config.deformer.pose_dim = self.config.pose_dim
        self.deformer = SNARFDeformer(self.config.deformer)

        from models.pose.pose_correction import PoseCorrection
        
        self.pose_correction = PoseCorrection(
            self.config.subject_ids, self.config.pose_correction
        )
        if len(self.config.subject_ids) == 1 and self.config.subject_ids[0] == '_train':
            # use this for personalization and relighting
            from models.latent_code.pca_code import PCACode
            from models.latent_code.latent_code import LatentCode
            
            self.shape_code = PCACode(self.config.subject_ids, self.config.shape_code)
            # self.appearance_code = PCACode(self.config.subject_ids, self.config.shape_code) ## try later
            self.appearance_code = PCACode(self.config.subject_ids, self.config.appearance_code)
        else:
            from models.latent_code.latent_code import LatentCode
            self.shape_code = LatentCode(self.config.subject_ids, self.config.shape_code)
            self.appearance_code = LatentCode(self.config.subject_ids, self.config.appearance_code)

        # PBR modules
        from models.pbr.material import VolumeMaterial
        self.material = VolumeMaterial(self.config.material)

        from lib.torch_pbr.bxdf import MultiLobe
        self.scatterer = MultiLobe(self.config.scatterer)

        from lib.torch_pbr.light import EnvironmentLightSG, EnvironmentLightTensor
        if self.config.light.name == "envlight-tensor":
            self.emitter = EnvironmentLightTensor(self.config.light)
        elif self.config.light.name == "envlight-SG":
            self.emitter = EnvironmentLightSG(self.config.light)
        else:
            assert False

        self.material_feature = self.config.get("material_feature", "hybrid")
        assert self.material_feature in ["geometry", "radiance", "hybrid"], (
            "material_feature must be one of "
            + "['geometry', 'radiance', 'hybrid'], got %s" % self.material_feature
        )

        self.register_buffer(
            "scene_aabb",
            torch.as_tensor(
                self.config.scene_aabb,
                dtype=torch.float32,
            ),
        )
        scene_diag_len = torch.norm(self.scene_aabb[3:] - self.scene_aabb[:3]).item()
        if self.config.grid_prune:

            self.occupancy_grid = TemporalOccGridEstimator(
                roi_aabb=self.scene_aabb[None, ...],
                resolution=64,
                levels=1,
            )

        # Hyperparameters
        self.randomized = self.config.randomized
        self.background_color = None
        self.samples_per_pixel = self.config.samples_per_pixel
        self.render_step_size = scene_diag_len / self.config.num_samples_per_ray
        self.num_samples_per_secondary_ray = self.config.get(
            "num_samples_per_secondary_ray", 64
        )
        self.secondary_near_plane = self.config.get("secondary_near_plane", 0.0)
        self.secondary_far_plane = self.config.get("secondary_far_plane", 1.5)
        self.secondary_shader_chunk = self.config.get("secondary_shader_chunk", 160000)
        self.secondary_importance_sample = self.config.get(
            "secondary_importance_sample", True
        )

        self.enable_phys = False if self.config.phys_kick_in_step > 0 else True
        self.add_emitter = self.config.get("add_emitter", False)

        self.albedo_only = False

        self.mesh_vh_cano = None
        self.mesh_fh_cano = None

    def update_step(self, epoch, global_step):
        # Update states of all submodules
        ### step here means: the number of times self.optimizer.step() is called; not the number of iteration.
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.radiance, epoch, global_step)
        update_module_step(self.density, epoch, global_step)
        update_module_step(self.pose_correction, epoch, global_step)
        update_module_step(self.shape_code, epoch, global_step)
        update_module_step(self.appearance_code, epoch, global_step)
        update_module_step(self.deformer, epoch, global_step)

        # Switch on/off physically based rendering and importance sampling
        if global_step >= self.config.phys_kick_in_step:
            self.enable_phys = True
        else:
            self.enable_phys = False

        if global_step > self.config.importance_sample_kick_in_step:
            self.importance_sample = True
        else:
            self.importance_sample = False

    def prepare(self, batch):
        batch.update(self.pose_correction(batch["frame_idx"], batch["subject_id"]))
        from models.latent_code.pca_code import PCACode
        if isinstance(self.shape_code, PCACode):
            pca_code, latent_code = self.shape_code(batch["subject_id"])
            batch.update({'shape_id': latent_code, 'shape_pca': pca_code})

            pca_code, latent_code = self.appearance_code(batch["subject_id"])
            batch.update({'appearance_id': latent_code, 'appearance_pca': pca_code})
        else:
            latent_code = self.shape_code(batch["subject_id"])
            batch.update({'shape_id': latent_code})

            latent_code = self.appearance_code(batch["subject_id"])
            batch.update({'appearance_id': latent_code})

        # self.deformer.prepare(batch
        self.deformer.rigid_deformer.prepare_deformer(batch)
        
        self.geometry.prepare_bbox(self.deformer.rigid_deformer.bbox)
        self.radiance.prepare_bbox(self.deformer.rigid_deformer.bbox)

        self.w2c = torch.inverse(batch["c2w"])
        self.v3d_s = self.deformer.get_vertices()
        v3d_w = self.deformer.get_vertices_world()
        v3d_c = transform_points_batch(self.w2c, v3d_w)
        faces = self.deformer.get_faces()
        batch["v3d_c"] = v3d_c
        batch["faces"] = faces

        def occ_eval_fn(x):
            thres = 15.0 * 0.001  # 15mm
            thres = thres**2
            
            dist = knn_points(x[None, :, :], self.v3d_s.view(1, -1, 3)).dists.view(-1)
            alpha = (dist < thres).float()
            return alpha
        self.occupancy_grid.update(occ_eval_fn)
        self.batch = batch

        if not self.training:
            resample_light = self.config.resample_light
            if not resample_light:
                # If `resample_light` is off,  then we only need to sample the light once
                resample_light = not hasattr(self, "secondary_rays_d")
            if self.enable_phys and "hdri" in batch and resample_light:
                assert self.config.light.name == "envlight-tensor", (
                    "envlight-tensor is required for physically based rendering"
                    + " when hdri is provided"
                )
                self.emitter.base = nn.Parameter(batch["hdri"])
                self.emitter.pdf_scale = (
                    self.emitter.base.shape[0] * self.emitter.base.shape[1]
                ) / (2 * np.pi * np.pi)
                self.emitter.update_pdf()
                self.secondary_rays_d = self.emitter.sample(self.samples_per_pixel)

    def forward_(self, rays):
        
        num_images, num_rays_per_image = rays.shape[:2]
        rays = rays.view(-1, 8)
        
        img_indices_og = torch.arange(num_images).cuda()[:, None].repeat(1, num_rays_per_image).view(-1)
        
        ## convert rays (centers + dir) from world to MANO canonical space
        rays = self.deformer.rigid_deformer.transform_rays_w2s(rays)
        n_rays = rays.shape[0]
        rays_o, rays_d, _, far = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]

        def rgb_normal_alpha_fn(t_starts, t_ends, ray_indices, img_indices):

            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros((0, 3)).cuda(), torch.zeros(
                    (0,).cuda()
                )
            

            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0 ## 3D sample points

            packed, max_count = pack_without_loop(
                img_indices, self.cond.shape[0], positions, self.cond, t_dirs, t_starts, t_ends)
            positions_packed = packed['positions_packed'] ## (num_images, max_3d_samples, 3)
            positions_valid = packed['positions_valid'] ## (num_images, max_3d_samples)
            cond_packed = packed['cond_packed'] ## (num_images, max_3d_samples, 576) ## too high? tune down during personalization
            t_dirs_packed = packed['t_dirs_packed'] ## (num_images, max_3d_samples, 3)
            t_starts_packed = packed['t_starts_packed'] ## (num_images, max_3d_samples)
            t_ends_packed = packed['t_ends_packed'] ## (num_images, max_3d_samples)

            
            def geometry_fn(x, valid, cond):
                return self.geometry(
                    x[valid],
                    cond[:, :, None, :].repeat(1, 1, x.shape[2], 1)[valid],
                    with_grad=True,
                    with_feature=True,
                    with_laplace=self.training and self.with_curvature_loss,
                )

            ret = self.deformer(
                positions_packed,
                positions_valid,
                cond_packed,
                geometry_fn,
                with_jac=True,
                eval_mode=not self.training,
            )

            
            positions, sdf, valid, sdf_grad, sdf_grad_cano, feature = ret[:6]
            laplace = ret[6] if len(ret) > 6 else torch.zeros_like(sdf)
            dists = (t_ends_packed - t_starts_packed)[..., None]
            normal_mano = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            normal_world = self.deformer.rigid_deformer.transform_dirs_s2w(sdf_grad)
            t_dirs_packed = self.deformer.rigid_deformer.transform_dirs_s2w(t_dirs_packed)
            alphas = get_alpha(self.density, sdf.view(-1), dists.view(-1, 1)).view(num_images, max_count)

            positions = positions[positions_valid]
            feature = feature[positions_valid]
            t_dirs_packed = t_dirs_packed[positions_valid]
            normal_world = normal_world[positions_valid]
            normal_mano = normal_mano[positions_valid]
            cond_packed = cond_packed[positions_valid]
            alphas = alphas[positions_valid]
            sdf = sdf[positions_valid]
            sdf_grad = sdf_grad[positions_valid]
            laplace = laplace[positions_valid]
            valid = valid[positions_valid]
            
            cond_pose, cond_s_shape, cond_s_appear = torch.split(cond_packed, [self.config.pose_dim, self.config.shape_code.latent_dim, self.config.appearance_code.latent_dim], dim=1)
            cond_rad = torch.cat((cond_pose, cond_s_appear), dim=1)
            rgbs, *_ = self.radiance(positions, feature, t_dirs_packed, normal_world, cond_rad)
            return (
                positions,
                valid,
                rgbs,
                normal_mano,
                normal_world,
                alphas,
                sdf,
                sdf_grad,
                laplace,
            )

        def rgb_normal_mats_alpha_fn(t_starts, t_ends, ray_indices, img_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros((0, 3)).cuda(), torch.zeros((0,).cuda()
                )
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            
            packed, max_count = pack_without_loop(
                img_indices, self.cond.shape[0], positions, self.cond, t_dirs, t_starts, t_ends)
            positions_packed = packed['positions_packed']
            positions_valid = packed['positions_valid']
            cond_packed = packed['cond_packed']
            t_dirs_packed = packed['t_dirs_packed']
            t_starts_packed = packed['t_starts_packed']
            t_ends_packed = packed['t_ends_packed']

            def geometry_fn(x, valid, cond):
                return self.geometry(
                    x[valid],
                    cond[:, :, None, :].repeat(1, 1, x.shape[2], 1)[valid],
                    with_grad=True,
                    with_feature=True,
                    with_laplace=self.training and self.with_curvature_loss,
                )
            
            ret = self.deformer(
                # positions,
                positions_packed,
                positions_valid,
                cond_packed,
                geometry_fn,
                with_jac=True,
                eval_mode=not self.training,
            )
            positions, sdf, valid, sdf_grad, sdf_grad_cano, feature = ret[:6]

            laplace = ret[6] if len(ret) > 6 else torch.zeros_like(sdf)
            dists = (t_ends_packed - t_starts_packed)[..., None]
            normal_mano = F.normalize(sdf_grad, p=2, dim=-1, eps=1e-6)
            normal_world = self.deformer.rigid_deformer.transform_dirs_s2w(sdf_grad)
            t_dirs_packed = self.deformer.rigid_deformer.transform_dirs_s2w(t_dirs_packed)
            alphas = get_alpha(self.density, sdf.view(-1), dists.view(-1, 1)).view(num_images, max_count)
            
            positions = positions[positions_valid]
            feature = feature[positions_valid]
            t_dirs_packed = t_dirs_packed[positions_valid]
            normal_world = normal_world[positions_valid]
            normal_mano = normal_mano[positions_valid]
            cond_packed = cond_packed[positions_valid]
            alphas = alphas[positions_valid]
            sdf = sdf[positions_valid]
            sdf_grad = sdf_grad[positions_valid]
            laplace = laplace[positions_valid]
            valid = valid[positions_valid]
            # Get radiance

            cond_pose, cond_s_shape, cond_s_appear = torch.split(cond_packed, [self.config.pose_dim, self.config.shape_code.latent_dim, self.config.appearance_code.latent_dim], dim=1)
            cond_rad = torch.cat((cond_pose, cond_s_appear), dim=1)
            rgbs, rgb_feature = self.radiance(
                positions, feature, t_dirs_packed, normal_world, cond_rad
            )
            
            cond_pose, cond_s_shape, cond_s_appear = torch.split(cond_packed, [self.config.pose_dim, self.config.shape_code.latent_dim, self.config.appearance_code.latent_dim], dim=1)
            cond_mat = torch.cat((cond_pose, cond_s_appear), dim=1)
            feature = torch.cat([rgb_feature, feature, cond_mat], dim=-1)

            # Get materials
            materials = self.material(feature)
            if not self.training and hasattr(self, "albedo_align_ratio"):
                materials[..., :3] = materials[..., :3] * self.albedo_align_ratio
            materials_jitter = torch.zeros_like(materials)

            return (
                positions,
                valid,
                rgbs,
                normal_mano,
                normal_world,
                materials,
                materials_jitter,
                alphas,
                sdf,
                sdf_grad,
                laplace,
            )

        occ_grid = self.occupancy_grid

        intervals, ray_indices, t_starts, t_ends = occ_grid.sampling(
            rays_o,
            rays_d,
            render_step_size=self.render_step_size,
            t_idx=0,
            stratified=self.randomized,
            cone_angle=0.0,
            alpha_thre=0.0,
        )
        
        # Importance sampling
        if self.importance_sample and t_starts.numel() > 0:
            intervals = importance_upsampling(
                rays_o,
                rays_d,
                self.geometry,
                self.cond,
                self.render_step_size,
                self.density,
                self.deformer,
                self.training,
                intervals,
                num_images, img_indices_og.clone(),
            )

        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = intervals.ray_indices[intervals.is_left]
        img_indices = img_indices_og.clone()[ray_indices] ## which image each 3D sample belongs to

        midpoints = (t_starts + t_ends) / 2.0
        intervals = t_ends - t_starts

        if self.enable_phys:
            (
                rgb_map,
                fg_normal_map,
                normal_map,
                albedo_map,
                roughness_map,
                metallic_map,
                acc_map,
                depth_map,
                extras,
            ) = rendering_with_normals_mats_sdf(
                t_starts,
                t_ends,
                img_indices,
                ray_indices=ray_indices,
                n_rays=n_rays,
                rgb_alpha_fn=rgb_normal_mats_alpha_fn,
                render_bkgd=None,
                material_dim=self.material.n_output_dims,
            )
            rgb_phys_map = self.background_color[None, ...].expand((n_rays, -1))
            if self.config.render_mode == "uniform_light":
                visibility_map = torch.zeros(n_rays, 1).cuda()
        else:

            (
                rgb_map,
                fg_normal_map,
                normal_map,
                acc_map,
                depth_map,
                extras,
            ) = rendering_with_normals_sdf(
                t_starts,
                t_ends,
                img_indices,
                ray_indices=ray_indices,
                n_rays=n_rays,
                rgb_alpha_fn=rgb_normal_alpha_fn,
                render_bkgd=None,
            )
        
        depth_map = depth_map + (1.0 - acc_map) * far[..., None]

        # Material-based volume scattering
        if ray_indices.numel() > 0 and self.enable_phys and not self.albedo_only:
            # Importance sample points for volumetric scattering
            cond_pack = self.cond[:, None, :].repeat(1, num_rays_per_image, 1).view(-1, self.cond.shape[1])
            img_indices_og = torch.arange(self.cond.shape[0], device=rays_o.device)[:, None].repeat(1, num_rays_per_image).view(-1)
            (
                resampled_packed_info, 
                resampled_ray_indices,
                resampled_weights,
                fg_indices,
                bg_indices, ## prob for env map
                resampled_extras,
            ) = sample_volume_interaction(
                rays_o,
                rays_d,
                cond_pack,
                img_indices_og,
                ray_indices,
                t_starts,
                t_ends,
                n_rays,
                self.samples_per_pixel,
                1.0 - acc_map,
                extras,
            )
            ## at least one point is fg
            if fg_indices.numel() > 0:
                
                Lo = torch.zeros(len(resampled_ray_indices), 3).cuda()
                if self.config.render_mode == "uniform_light":
                    
                    visibility = torch.zeros(
                        len(resampled_ray_indices), 3
                    ).cuda()
                if self.add_emitter:
                    # Evaluate the environment map for bg rays
                    rays_indices_bg, inverse_indices = torch.unique(
                        resampled_ray_indices[bg_indices], return_inverse=True
                    )
                    em_li_bg = self.emitter.eval(
                        self.deformer.rigid_deformer.transform_dirs_s2w(
                            rays_d[rays_indices_bg]
                        ).view(-1, 3)
                    )  # [len(rays_indices_bg), 3]
                    Lo.scatter_(
                        0,
                        bg_indices[..., None].expand(-1, 3),
                        em_li_bg[inverse_indices],  # [len(bg_indices), 3]
                    )
                else:
                    Lo.scatter_(
                        0,
                        bg_indices[..., None].expand(-1, 3),
                        self.background_color[None, ...].expand(
                            (bg_indices.shape[0], -1)
                        ),
                    )

                Lo_demod = Lo.clone() ### now this has background colors
                if self.config.render_mode == "light":
                    
                    # Shuffle samples along each ray independently to avoid bias
                    shuffled_indices = torch.arange(self.samples_per_pixel)
                    shuffled_indices = shuffled_indices.cuda()
                    shuffled_indices = shuffled_indices.reshape(1, -1).repeat(n_rays, 1)
                    
                    col_indices = torch.argsort(
                        torch.rand(n_rays, self.samples_per_pixel).cuda(), dim=-1
                    )
                    row_indices = torch.arange(n_rays)[..., None]
                    
                    shuffled_indices = shuffled_indices[row_indices, col_indices]
                    
                    # Get shuffled_indice for foreground samples
                    mask = (
                        unpack_data(
                            resampled_packed_info,
                            torch.ones_like(resampled_ray_indices[..., None]),
                            self.samples_per_pixel,
                        )
                        .bool()
                        .squeeze(-1)
                    )
                    
                    shuffled_indices, _ = pack_data(shuffled_indices[..., None], mask)
                    shuffled_fg_indices = shuffled_indices[fg_indices].squeeze(-1)

                    fg_Lo, fg_Lo_diff, fg_Lo_spec = self.pbr_light_forward(
                        resampled_extras["normals"],
                        resampled_extras["albedo"],
                        resampled_extras["roughness"],
                        resampled_extras["metallic"],
                        resampled_extras["positions"],
                        resampled_extras["t_dirs"],
                        shuffled_indices=shuffled_fg_indices,
                        img_indices=resampled_extras['img_indices'],
                        n_rays=n_rays,
                    )
                elif self.config.render_mode == "uniform_light":
                    assert self.samples_per_pixel == 512
                    # Shuffle samples along each ray independently to avoid bias
                    shuffled_indices = (
                        torch.arange(512).cuda()
                        .reshape(1, -1)
                        .repeat(n_rays, 1)
                    )
                    col_indices = torch.argsort(torch.rand(n_rays, 512), dim=-1)
                    row_indices = torch.arange(n_rays)[..., None].cuda()
                    shuffled_indices = shuffled_indices[row_indices, col_indices]
                    # Get shuffled_indice for foreground samples
                    mask = (
                        unpack_data(
                            resampled_packed_info,
                            torch.ones_like(resampled_ray_indices[..., None]),
                            512,
                        )
                        .bool()
                        .squeeze(-1)
                    )
                    shuffled_indices, _ = pack_data(shuffled_indices[..., None], mask)
                    shuffled_fg_indices = shuffled_indices[fg_indices].squeeze(-1)

                    assert self.config.render_mode == "uniform_light"

                    fg_Lo, fg_Lo_diff, fg_Lo_spec, fg_vis = self.pbr_uniform_light_forward(
                        resampled_extras["normals"],
                        resampled_extras["albedo"],
                        resampled_extras["roughness"],
                        resampled_extras["metallic"],
                        resampled_extras["positions"],
                        resampled_extras["t_dirs"],
                        shuffled_indices=shuffled_fg_indices,
                        img_indices=resampled_extras['img_indices'],
                        n_rays=n_rays,
                    )
                    visibility.scatter_(0, fg_indices[..., None].expand(-1, 3), fg_vis)
                    visibility_map = accumulate_along_rays(
                        resampled_weights,
                        visibility,
                        resampled_ray_indices,
                        n_rays,
                    ).mean(dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(
                        f"Render mode {self.config.render_mode} not supported."
                    )
                Lo.scatter_(0, fg_indices[..., None].expand(-1, 3), fg_Lo)
                Lo_demod.scatter_(
                    0, fg_indices[..., None].expand(-1, 3), fg_Lo_diff + fg_Lo_spec
                )

                rgb_phys_map = accumulate_along_rays(
                    resampled_weights,
                    Lo,
                    resampled_ray_indices,
                    n_rays,
                )
                demod_phys_map = accumulate_along_rays(
                    resampled_weights,
                    Lo_demod,
                    resampled_ray_indices,
                    n_rays,
                )
                bg_indices = torch.nonzero(resampled_packed_info[..., 1] <= 0)[..., 0]
                if self.add_emitter:
                    # Evaluate the environment map for bg rays
                    em_li_bg = self.emitter.eval(
                        self.deformer.rigid_deformer.transform_dirs_s2w(
                            rays_d[bg_indices]
                        ).view(-1, 3)
                    )
                    rgb_phys_map[bg_indices] = em_li_bg
                    demod_phys_map[bg_indices] = em_li_bg
                else:
                    rgb_phys_map[bg_indices] = self.background_color[None, :]
                    demod_phys_map[bg_indices] = self.background_color[None, :]
            else:
                if self.add_emitter:
                    # Evaluate the environment map for bg rays
                    rgb_phys_map = self.emitter.eval(
                        self.deformer.rigid_deformer.transform_dirs_s2w(rays_d).view(-1, 3)
                    )
                    demod_phys_map = rgb_phys_map
                else:
                    rgb_phys_map = self.background_color[None, ...] * torch.ones(
                        (n_rays, 3)
                    ).cuda()
                    demod_phys_map = rgb_phys_map
        elif self.enable_phys:
            if self.add_emitter:
                # Evaluate the environment map for bg rays

                rgb_phys_map = self.emitter.eval(
                    self.deformer.rigid_deformer.transform_dirs_s2w(rays_d).view(-1, 3)
                )
                demod_phys_map = rgb_phys_map
            else:
                rgb_phys_map = self.background_color[None, ...] * torch.ones(
                    (n_rays, 3)
                ).cuda()
                demod_phys_map = rgb_phys_map
        out = {
            "comp_rgb": rgb_map,
            "comp_normal": normal_map,
            "fg_normal": fg_normal_map,
            "opacity": acc_map,
            "depth": depth_map,
            "rays_valid": acc_map > 0,
            "rays_valid_phys": (
                (acc_map > 0)
                if self.enable_phys
                else torch.zeros_like(acc_map, dtype=torch.bool)
            ),
            "num_samples": torch.as_tensor(
                [len(t_starts)], dtype=torch.int32
            ).cuda(),
        }

        if self.enable_phys:
            out.update(
                {
                    "comp_rgb_phys": rgb_phys_map,
                    "comp_demod_phys": demod_phys_map,
                    "comp_albedo": albedo_map,
                    "comp_metallic": metallic_map,
                    "comp_roughness": roughness_map,
                }
            )
            if self.config.render_mode == "uniform_light":
                out.update({"visibility": visibility_map})

        if self.training:
            weights = extras["weights"]
            sdf = extras["sdf"]
            sdf_grad = extras["sdf_grad"]
            sdf_laplace = extras["laplace"]
            out.update(
                {
                    "sdf_samples": sdf,
                    "sdf_grad_samples": sdf_grad,
                    "sdf_laplace_samples": sdf_laplace,
                    "weights": weights.view(-1),
                    # TODO: following variables are useful for unbounded scenes
                    "points": midpoints.view(-1),
                    "intervals": intervals.view(-1),
                    "ray_indices": ray_indices.view(-1),
                }
            )

            if ray_indices.numel() > 0 and self.enable_phys:
                normals = extras["normals"]
                albedo = extras["albedo"]
                roughness = extras["roughness"]
                metallic = extras["metallic"]
                albedo_jitter = extras["albedo_jitter"]
                roughness_jitter = extras["roughness_jitter"]
                metallic_jitter = extras["metallic_jitter"]

                # Normal orientation loss and material smoothness losses
                normals_orientation_loss = torch.sum(
                    rays_d[ray_indices] * normals, dim=-1, keepdim=True
                ).clamp(min=0)
                albedo_smoothness_loss = self.compute_relative_smoothness_loss(
                    albedo, albedo_jitter
                )
                roughness_smoothness_loss = self.compute_relative_smoothness_loss(
                    roughness, roughness_jitter
                )
                metallic_smoothness_loss = self.compute_relative_smoothness_loss(
                    metallic, metallic_jitter
                )

                normals_orientation_loss_map = accumulate_along_rays(
                    weights,
                    values=normals_orientation_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                albedo_smoothness_loss_map = accumulate_along_rays(
                    weights,
                    values=albedo_smoothness_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                roughness_smoothness_loss_map = accumulate_along_rays(
                    weights,
                    values=roughness_smoothness_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                metallic_smoothness_loss_map = accumulate_along_rays(
                    weights,
                    values=metallic_smoothness_loss,
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
            else:
                normals_orientation_loss_map = torch.zeros_like(rgb_map[..., :1])
                albedo_smoothness_loss_map = torch.zeros_like(rgb_map[..., :1])
                roughness_smoothness_loss_map = torch.zeros_like(rgb_map[..., :1])
                metallic_smoothness_loss_map = torch.zeros_like(rgb_map[..., :1])

            out.update(
                {
                    "normals_orientation_loss_map": normals_orientation_loss_map,
                    "albedo_smoothness_loss_map": albedo_smoothness_loss_map,
                    "roughness_smoothness_loss_map": roughness_smoothness_loss_map,
                    "metallic_smoothness_loss_map": metallic_smoothness_loss_map,
                }
            )

        out_bg = {
            "comp_rgb": self.background_color[None, :].expand(*rgb_map.shape),
            "num_samples": torch.zeros_like(out["num_samples"]),
            "rays_valid": torch.zeros_like(out["rays_valid"]),
            "rays_valid_phys": torch.zeros_like(
                out["rays_valid_phys"], dtype=torch.bool
            ),
        }
        if self.enable_phys:
            out_bg.update(
                {
                    "comp_albedo": torch.zeros(
                        (1, 3), dtype=torch.float32
                    ).cuda().expand(*albedo_map.shape),
                    "comp_metallic": self.background_color[None, :]
                    .mean(-1, keepdim=True)
                    .expand(metallic_map.shape),
                    "comp_roughness": self.background_color[None, :]
                    .mean(-1, keepdim=True)
                    .expand(roughness_map.shape),
                }
            )

        out_full = {
            "comp_rgb": rgb_to_srgb(
                out["comp_rgb"] + out_bg["comp_rgb"] * (1.0 - out["opacity"])
            ).clamp(0, 1),
            "num_samples": out["num_samples"] + out_bg["num_samples"],
            "rays_valid": out["rays_valid"] | out_bg["rays_valid"],
            "rays_valid_phys": out["rays_valid_phys"] | out_bg["rays_valid_phys"],
        }
        if self.enable_phys:
            out_full.update(
                {
                    "comp_rgb_phys": rgb_to_srgb(out["comp_rgb_phys"]).clamp(0, 1),
                    "comp_demod_phys": rgb_to_srgb(out["comp_demod_phys"]).clamp(0, 1),
                    "comp_albedo": out["comp_albedo"]
                    + out_bg["comp_albedo"] * (1.0 - out["opacity"]),
                    "comp_metallic": out["comp_metallic"]
                    + out_bg["comp_metallic"] * (1.0 - out["opacity"]),
                    "comp_roughness": out["comp_roughness"]
                    + out_bg["comp_roughness"] * (1.0 - out["opacity"]),
                }
            )

        out.update(self.prepare_sdf_targets())

        # rotate normal to camera view
        out["comp_normal"] = out["comp_normal"].view(num_images, -1, 3)
        out["comp_normal"] = torch.bmm(out["comp_normal"], self.w2c[:, :3, :3].transpose(1, 2)) ## bug fix
        out["comp_normal"] = out["comp_normal"] * torch.tensor(
            [1.0, -1.0, -1.0]
        ).cuda()


        out["fg_normal"] = out["fg_normal"].view(num_images, -1, 3)
        out["fg_normal"] = torch.bmm(out["fg_normal"], self.w2c[:, :3, :3].transpose(1, 2)) ## bug fix
        out["fg_normal"] = out["fg_normal"] * torch.tensor(
            [1.0, -1.0, -1.0]
        ).cuda()
        out["fg_normal"] = (out["fg_normal"] + 1.0) / 2.0

        out["comp_normal"] =  out["comp_normal"].view(-1, 3)
        out["fg_normal"] =  out["fg_normal"].view(-1, 3)


        myout = {
            **out,
            **{k + "_bg": v for k, v in out_bg.items()},
            **{k + "_full": v for k, v in out_full.items()},
        }
        return myout

    def forward(self, batch):
        rays = batch["rays"]
        self.img_id = batch['img_id']

        self.cond = self.pose_encoder(
            self.deformer.get_axis_angles(), self.deformer.get_joints()
        )  ##  pose conditioning
        self.cond = torch.cat((self.cond, batch["shape_id"], batch["appearance_id"]), dim=1)

        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch_inference(
                self.forward_,
                self.config.ray_chunk,
                True,
                rays,
            )
        out['shape_id'] = batch['shape_id']
        out['appearance_id'] = batch['appearance_id']
        if 'shape_pca' in batch:
            out['shape_pca'] = batch['shape_pca']
            out['appearance_pca'] = batch['appearance_pca']
        return {**out, "beta": self.density.get_beta()}

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def isosurface(self):
        mesh = self.geometry.isosurface(self.cond[:, :self.config.pose_dim+self.config.shape_code.latent_dim])
        return mesh
    
    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        return mesh
