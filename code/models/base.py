import torch
import torch.nn as nn
import numpy as np
from models.utils import (
    chunk_batch,
)
from models.volrend import (
    rendering,
)
from models.utils import (
    sample_on_barycentric_mesh,
    compute_mano_cano_sdf,
    PointInSpace,
    get_alpha,
)

from lib.nerfacc import (
    ray_resampling_sdf_fine,
    pack_info,
    unpack_info,
)
from models.utils import pack_without_loop

pt_in_space_sampler_h = PointInSpace(global_sigma_xyz=[0.18, 0.9, 0.18])


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup()
        if self.config.get("weights", None):
            self.load_state_dict(torch.load(self.config.weights))

    def setup(self):
        raise NotImplementedError

    def update_step(self, epoch, global_step, *args):
        pass

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        if self.enable_phys:  ## false
            losses.update(self.material.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(
                self.geometry,
                export_config.chunk_size,
                False,
                mesh["v_pos"].cuda(),
                with_grad=True,
                with_feature=True,
            )
        return mesh

    def pbr_mis_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
            roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Scatterer sampling
            scatter_dirs = self.scatterer.sample(
                n=normal_interleaved,
                wi=wi_interleaved,
                alpha_x=roughness_interleaved.squeeze(-1),
                alpha_y=roughness_interleaved.squeeze(-1),
                albedo=albedo_interleaved,
                metallic=metallic_interleaved,
                attenuation=attenuation_interleaved,
            )
            # Light sampling
            if self.training:
                self.emitter.update_pdf()
            light_dirs = self.deformer.rigid_deformer.transform_dirs_w2s(
                self.emitter.sample(len(scatter_dirs))
            )
            # Concatenate all sampled directions
            secondary_rays_d = torch.cat([scatter_dirs, light_dirs], dim=0)
            secondary_rays_o = positions.repeat(2, 1)

            secondary_tr, secondary_rgb_map = self.compute_indirect_radiance(
                secondary_rays_o,
                secondary_rays_d,
            )

        # Compute pdf values of all rays (scatter + light)
        pdf_scatter = self.scatterer.pdf(
            n=normal_interleaved.repeat(2, 1),
            wi=wi_interleaved.repeat(2, 1),
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1).repeat(2),
            alpha_y=roughness_interleaved.squeeze(-1).repeat(2),
            albedo=albedo_interleaved.repeat(2, 1),
            metallic=metallic_interleaved.repeat(2, 1),
            attenuation=attenuation_interleaved.repeat(2, 1),
        )
        pdf_light = self.emitter.pdf(
            self.deformer.rigid_deformer.transform_dirs_s2w(secondary_rays_d)
        )
        # Compute scatter ratio of all rays (scatter + light)
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff, spec = self.scatterer.eval(
            wi=wi_interleaved.repeat(2, 1),
            n=normal_interleaved.repeat(2, 1),
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1).repeat(2),
            alpha_y=roughness_interleaved.squeeze(-1).repeat(2),
            albedo=albedo_interleaved.repeat(2, 1),
            metallic=metallic_interleaved.repeat(2, 1),
            attenuation=attenuation_interleaved.repeat(2, 1),
        )
        # Query the environment map with all rays (scatter + light)
        em_li = self.emitter.eval(
            self.deformer.rigid_deformer.transform_dirs_s2w(secondary_rays_d)
        )
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        # Compute MIS weights
        # We don't backpropagate through MIS weights, so we can use torch.where
        # without handling divide-by-zero cases
        mis_weights = torch.where(
            pdf_scatter + pdf_light > 1e-6,
            torch.reciprocal(pdf_scatter + pdf_light),
            torch.zeros_like(pdf_scatter),
        )
        # Combine scatterer and light samples using MIS weights
        # Note the PDF terms of MC integration in the denominator is
        # cancelled out by MIS weights
        Lo_diff = (Li * diff) * mis_weights
        Lo_spec = (Li * spec) * mis_weights

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd.repeat(2, 1) * Lo_diff + ks.repeat(2, 1) * Lo_spec

        # Sum over sampling strategies
        Lo = Lo.reshape(2, -1, 3).sum(dim=0)
        Lo_diff = Lo_diff.reshape(2, -1, 3).sum(dim=0)
        Lo_spec = Lo_spec.reshape(2, -1, 3).sum(dim=0)

        return Lo, Lo_diff, Lo_spec

    def pbr_uniform_light_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
        shuffled_indices,
        img_indices,
        n_rays,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
            roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Stratified light sampling
            # Since we sample uniformly on the light sphere, we do not need to
            # convert the sampled directions to the MANO space
            n_rows = 16
            n_cols = 32
            (
                secondary_rays_d,
                inv_pdf,
            ) = self.emitter.sample_uniform_sphere_stratified(
                n_rays,
                n_rows,
                n_cols,
                device=albedo_interleaved.device,
            )
            secondary_rays_d = secondary_rays_d[shuffled_indices]
            inv_pdf = inv_pdf[shuffled_indices]

            secondary_rays_o = positions.reshape(-1, 3)
            cosine_mask = (normal_interleaved * secondary_rays_d).sum(dim=-1) > 1e-6

            secondary_tr = torch.zeros(len(cosine_mask), 1, device=cosine_mask.device)
            secondary_rgb_map = torch.zeros(
                len(cosine_mask), 3, device=cosine_mask.device
            )

            if cosine_mask.sum() > 0:
                (
                    secondary_tr[cosine_mask],
                    secondary_rgb_map[cosine_mask],
                ) = self.compute_indirect_radiance(
                    secondary_rays_o[cosine_mask], secondary_rays_d[cosine_mask], img_indices[cosine_mask]
                )
                secondary_tr.clamp_(0.0, 1.0)

            tr_mask = secondary_tr[..., 0] > 0.0

        # Compute scatter ratio of all rays
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff = torch.zeros_like(albedo_interleaved[..., :1])
        spec = torch.zeros_like(albedo_interleaved)
        if cosine_mask.sum() > 0:
            diff[cosine_mask], spec[cosine_mask] = self.scatterer.eval(
                wi=wi_interleaved[cosine_mask],
                n=normal_interleaved[cosine_mask],
                wo=secondary_rays_d[cosine_mask],
                alpha_x=roughness_interleaved.squeeze(-1)[cosine_mask],
                alpha_y=roughness_interleaved.squeeze(-1)[cosine_mask],
                albedo=albedo_interleaved[cosine_mask],
                metallic=metallic_interleaved[cosine_mask],
                attenuation=attenuation_interleaved[cosine_mask],
            )
        # Query the environment map with all rays
        em_li = torch.zeros_like(secondary_rgb_map)
        # if cosine_mask.sum() > 0:
        if (cosine_mask & tr_mask).sum() > 0:
            my_rays_d = secondary_rays_d[cosine_mask & tr_mask]
            my_img_indices = img_indices[cosine_mask & tr_mask]
            packed, max_count = pack_without_loop(my_img_indices, self.cond.shape[0], t_dirs=my_rays_d)
            em_li[cosine_mask & tr_mask] = self.emitter.eval(
                self.deformer.rigid_deformer.transform_dirs_s2w(
                    packed['t_dirs_packed']
                )[packed["positions_valid"]]
            )
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        # Compute diffuse/specular component of the outgoing radiance
        Lo_diff = Li * diff * inv_pdf
        Lo_spec = Li * spec * inv_pdf
        vis = 2 * torch.ones_like(em_li) * secondary_tr

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd * Lo_diff + ks * Lo_spec

        return Lo, Lo_diff, Lo_spec, vis

    def pbr_light_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
        shuffled_indices,
        img_indices,
        n_rays,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
            roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Stratified light sampling
            if self.training:
                self.emitter.update_pdf()

            if self.training:
                secondary_rays_d = self.deformer.rigid_deformer.transform_dirs_w2s(
                    self.emitter.sample(len(albedo_interleaved))
                )
            else:
                secondary_rays_d = self.deformer.rigid_deformer.transform_dirs_w2s(
                    self.secondary_rays_d
                )
                secondary_rays_d = secondary_rays_d.repeat(n_rays, 1)[shuffled_indices]

            secondary_rays_o = positions.reshape(-1, 3)
            cosine_mask = (normal_interleaved * secondary_rays_d).sum(dim=-1) > 1e-6

            secondary_tr = torch.zeros(len(cosine_mask), 1, device=cosine_mask.device)
            secondary_rgb_map = torch.zeros(
                len(cosine_mask), 3, device=cosine_mask.device
            )

            if cosine_mask.sum() > 0:
                (
                    secondary_tr[cosine_mask],
                    secondary_rgb_map[cosine_mask],
                ) = self.compute_indirect_radiance(
                    secondary_rays_o[cosine_mask], secondary_rays_d[cosine_mask], img_indices[cosine_mask]
                )
                secondary_tr.clamp_(0.0, 1.0)

            tr_mask = secondary_tr[..., 0] > 0.0

        # Compute scatter ratio of all rays
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff = torch.zeros_like(albedo_interleaved[..., :1])
        spec = torch.zeros_like(albedo_interleaved)
        if cosine_mask.sum() > 0:
            diff[cosine_mask], spec[cosine_mask] = self.scatterer.eval(
                wi=wi_interleaved[cosine_mask],
                n=normal_interleaved[cosine_mask],
                wo=secondary_rays_d[cosine_mask],
                alpha_x=roughness_interleaved.squeeze(-1)[cosine_mask],
                alpha_y=roughness_interleaved.squeeze(-1)[cosine_mask],
                albedo=albedo_interleaved[cosine_mask],
                metallic=metallic_interleaved[cosine_mask],
                attenuation=attenuation_interleaved[cosine_mask],
            )
        # Query the environment map with all rays
        em_li = torch.zeros_like(secondary_rgb_map)
        if cosine_mask.sum() > 0:
            
            em_li[cosine_mask & tr_mask] = self.emitter.eval(
                self.deformer.rigid_deformer.transform_dirs_s2w(
                    secondary_rays_d[cosine_mask & tr_mask]
                ).view(-1, 3)
            )
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr

        pdf = torch.ones_like(em_li[..., :1])
        pdf[cosine_mask & tr_mask] = self.emitter.pdf(
            self.deformer.rigid_deformer.transform_dirs_s2w(
                secondary_rays_d[cosine_mask & tr_mask]
            ).view(-1, 3)
        )
        # Avoid divide-by-zero cases
        pdf = torch.where(pdf > 0, pdf, torch.ones_like(pdf))
        # Compute diffuse/specular component of the outgoing radiance
        Lo_diff = Li * diff / pdf
        Lo_spec = Li * spec / pdf

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd * Lo_diff + ks * Lo_spec

        return Lo, Lo_diff, Lo_spec

    def pbr_mats_forward(
        self,
        normal,
        albedo,
        roughness,
        metallic,
        positions,
        dirs,
    ):
        albedo_interleaved = albedo.reshape(-1, 3)
        roughness_interleaved = roughness.reshape(-1, 1)
        normal_interleaved = normal.reshape(-1, 3)
        wi_interleaved = -dirs.reshape(-1, 3)
        attenuation_interleaved = torch.zeros_like(
            roughness_interleaved
        )  # no attenuation for now
        metallic_interleaved = metallic.reshape(len(roughness_interleaved), -1)
        with torch.no_grad():
            # Scatterer sampling
            secondary_rays_d = self.scatterer.sample(
                n=normal_interleaved,
                wi=wi_interleaved,
                alpha_x=roughness_interleaved.squeeze(-1),
                alpha_y=roughness_interleaved.squeeze(-1),
                albedo=albedo_interleaved,
                metallic=metallic_interleaved,
                attenuation=attenuation_interleaved,
            )
            secondary_rays_o = positions.reshape(-1, 3)

            secondary_tr, secondary_rgb_map = self.compute_indirect_radiance(
                secondary_rays_o,
                secondary_rays_d,
            )

        # Compute pdf values of all rays
        pdf = self.scatterer.pdf(
            n=normal_interleaved,
            wi=wi_interleaved,
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1),
            alpha_y=roughness_interleaved.squeeze(-1),
            albedo=albedo_interleaved,
            metallic=metallic_interleaved,
            attenuation=attenuation_interleaved,
        )
        # Avoid divide-by-zero cases
        pdf = torch.where(pdf > 0, pdf, torch.ones_like(pdf))
        # Compute scatter ratio of all rays
        # Note that diff, and spec all include the cosine foreshortening factor,
        # if the scatterer is a BRDF/BSDF.
        diff, spec = self.scatterer.eval(
            wi=wi_interleaved,
            n=normal_interleaved,
            wo=secondary_rays_d,
            alpha_x=roughness_interleaved.squeeze(-1),
            alpha_y=roughness_interleaved.squeeze(-1),
            albedo=albedo_interleaved,
            metallic=metallic_interleaved,
            attenuation=attenuation_interleaved,
        )
        # Query the environment map with all rays (scatter + light)
        em_li = self.emitter.eval(
            self.deformer.rigid_deformer.transform_dirs_s2w(secondary_rays_d)
        )
        if self.config.global_illumination:
            Li = em_li * secondary_tr + secondary_rgb_map
        else:
            Li = em_li * secondary_tr
        # Combine scatterer and light samples using MIS weights
        # Note the PDF terms of MC integration in the denominator is
        # cancelled out by MIS weights
        Lo_diff = Li * diff / pdf
        Lo_spec = Li * spec / pdf

        # Compose blended radiance in linear RGB space
        if metallic_interleaved.size(-1) == 1:
            # Surface scattering
            kd = (1.0 - metallic_interleaved) * albedo_interleaved
            ks = torch.ones_like(kd)
        else:
            # Volume scattering
            kd = albedo_interleaved
            ks = metallic_interleaved

        Lo = kd * Lo_diff + ks * Lo_spec

        return Lo, Lo_diff, Lo_spec

    def compute_indirect_radiance(self, rays_o, rays_d, _img_indices):
        n_rays = rays_o.shape[0]
        img_indices = _img_indices.clone()

        def coarse_alpha_sdf_fn(t_starts, t_ends, ray_indices, img_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            img_indices = img_indices[ray_indices] 
            positions = t_origins + t_dirs * t_starts[..., None]
            if t_origins.shape[0] == 0:
                return torch.zeros((0,), device=t_origins.device)
            
            num_images = self.cond.shape[0]
            packed, max_count = pack_without_loop(
                img_indices, num_images, positions=positions, cond=self.cond, t_starts=t_starts, t_ends=t_ends)
            positions_packed = packed['positions_packed']
            positions_valid = packed['positions_valid']
            cond_packed = packed['cond_packed']
            assert num_images == positions_packed.shape[0]
            assert num_images == positions_valid.shape[0]
            assert max_count == positions_packed.shape[1]
            assert max_count == positions_valid.shape[1]
            assert len(positions_valid.shape) == 2
            t_starts_packed = packed['t_starts_packed']
            t_ends_packed = packed['t_ends_packed']   

            def geometry_fn(x, valid, cond):
                return self.geometry(
                    x[valid],
                    cond[:, :, None, :].repeat(1, 1, x.shape[2], 1)[valid],
                    with_grad=False, with_feature=False, with_laplace=False
                )

            # # Use chunk_batch to avoid OOM
            _, sdf, *others = chunk_batch(
                self.deformer,
                self.secondary_shader_chunk, 
                False,
                positions_packed,
                positions_valid,
                cond_packed,
                geometry_fn,
                with_jac=False,
                eval_mode=not self.training,
            )
            # sdf = torch.minimum(sdf[intervals.is_left], sdf[intervals.is_right])

            dists = (t_ends_packed - t_starts_packed)[..., None]
            # VolSDF does not need normal and t_dirs to compute alpha
            # alphas = get_alpha(self.density, sdf, dists)
            alphas = get_alpha(self.density, sdf.view(-1), dists.view(-1, 1)).view(num_images, max_count)
            alphas = alphas[positions_valid]
            sdf = sdf[positions_valid]
            return alphas, sdf

        def rgb_alpha_fn(t_starts, t_ends, ray_indices, img_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            if t_origins.shape[0] == 0:
                return torch.zeros((0, 3), device=t_origins.device), torch.zeros(
                    (0,), device=t_origins.device
                )
            num_images = self.cond.shape[0]
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
                    with_grad=True, with_feature=True, with_laplace=False
                )

            positions, sdf, valid, sdf_grad, _, feature = self.deformer(
                positions_packed,
                positions_valid,
                cond_packed,
                geometry_fn,
                with_jac=True,
                eval_mode=True,
            )

            dists = (t_ends_packed - t_starts_packed)[..., None]
            t_dirs_packed = self.deformer.rigid_deformer.transform_dirs_s2w(t_dirs_packed)
            normal_world = self.deformer.rigid_deformer.transform_dirs_s2w(sdf_grad)
            alphas = get_alpha(self.density, sdf.view(-1), dists.view(-1, 1)).view(num_images, max_count)

            positions = positions[positions_valid]
            feature = feature[positions_valid]
            t_dirs_packed = t_dirs_packed[positions_valid]
            normal_world = normal_world[positions_valid]
            cond_packed = cond_packed[positions_valid]

            alphas = alphas[positions_valid]
            sdf = sdf[positions_valid]

            cond_pose, cond_s_shape, cond_s_appear = torch.split(cond_packed, [32, self.config.shape_code.latent_dim, self.config.appearance_code.latent_dim], dim=1)
            cond_rad = torch.cat((cond_pose, cond_s_appear), dim=1)
            rgbs, *_ = self.radiance(positions, feature, t_dirs_packed, normal_world, cond_rad)
            return sdf, rgbs, alphas

        secondary_render_step_size = (
            self.secondary_far_plane - self.secondary_near_plane
        ) / (self.num_samples_per_secondary_ray - 1)

        intervals, ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
            rays_o,
            rays_d,
            near_plane=self.secondary_near_plane,
            far_plane=self.secondary_far_plane,
            t_idx=0,
            render_step_size=secondary_render_step_size,
            stratified=False,
        )

        # Importance sampling for secondary rays. Note that different to primary
        # rays, we only upsample once and use only importance samples while
        # discarding uniform samples
        if t_starts.numel() > 0 and self.secondary_importance_sample:
            for _ in range(1):
                with torch.no_grad():
                    alphas, sdfs = coarse_alpha_sdf_fn(t_starts, t_ends, ray_indices, img_indices)
                    packed_info = pack_info(ray_indices=ray_indices, n_rays=n_rays)
                    (
                        resampled_packed_info,
                        resampled_starts,
                        resampled_ends,
                        is_fg_sample,
                    ) = ray_resampling_sdf_fine(
                        packed_info,
                        t_starts[..., None],
                        t_ends[..., None],
                        alphas,
                        sdfs,
                        n_samples=4,
                    )
                    
                # Keep only foreground samples
                resampled_ray_indices = unpack_info(
                    resampled_packed_info, len(resampled_starts)
                )
                ray_indices = resampled_ray_indices[is_fg_sample]
                t_starts = resampled_starts[is_fg_sample, 0]
                t_ends = resampled_ends[is_fg_sample, 0]
        img_indices = _img_indices.clone()[ray_indices] ### image id of each ray
        (
            rgb_map,
            acc_map,
            _,
            extras,
        ) = rendering(
            t_starts,
            t_ends,
            img_indices,
            ray_indices=ray_indices,
            n_rays=n_rays,
            rgb_alpha_fn=rgb_alpha_fn,
            render_bkgd=None,
            chunk_size=self.secondary_shader_chunk,
        )

        return 1.0 - acc_map, rgb_map

    def compute_relative_smoothness_loss(self, values, values_jittor):

        base = torch.maximum(values, values_jittor).clip(min=1e-6)
        difference = torch.sum(
            ((values - values_jittor) / base) ** 2, dim=-1, keepdim=True
        )  # [..., 1]

        return difference

    def prepare_sdf_targets(self):
        
        out = {}
        ## this is MANO zero pose shape, not cursed pose
        mesh_v_cano = self.deformer.rigid_deformer.vs_template_seal ## v-template of cursed pose (no pose dependent effect)
        pose_offsets = self.deformer.rigid_deformer.mano_outputs.pose_offsets
        pose_offsets = self.deformer.rigid_deformer.body_model.seal_mesh(pose_offsets, None, self.deformer.rigid_deformer.is_rhand)
        
        faces_seal = torch.LongTensor(
            self.deformer.rigid_deformer.faces_seal.astype(np.int64)
        ).cuda()
        
        mesh_vh_cano = mesh_v_cano + pose_offsets
        mesh_fh_cano = faces_seal
        from kaolin.ops.mesh import index_vertices_by_faces
        
        # prepare mesh
        mesh_h = index_vertices_by_faces(mesh_vh_cano, mesh_fh_cano)
        
        num_samples = 5120*8
        mano_cano_samples = sample_on_barycentric_mesh(
            mesh_vh_cano, mesh_fh_cano, num_samples=num_samples
        )
        mano_cano_samples = pt_in_space_sampler_h.get_points(
            mano_cano_samples, local_sigma=0.005, global_ratio=0.10
        )

        # prepare input to compute sdf
        mesh_h = index_vertices_by_faces(mesh_vh_cano, mesh_fh_cano)


        gt_sdf = compute_mano_cano_sdf(
            mesh_vh_cano,
            mesh_fh_cano,
            mesh_h,
            mano_cano_samples,
        ).view(-1)
        cond = self.cond[:, None, :].repeat(1, mano_cano_samples.shape[1], 1)
        cond = cond[:, :, :self.config.pose_dim + self.config.shape_code.latent_dim] ### may change dim


        def sdf_fn(x):
            return self.geometry(
                x.view(-1, 3), cond.view(-1, cond.shape[-1]), with_grad=False, with_feature=False, with_laplace=False
            )
        
        pred_sdf = sdf_fn(mano_cano_samples)

        out["pred_sdf"] = pred_sdf
        out["gt_sdf"] = gt_sdf.detach()
        return out
