#  Copyright (c) Meta Platforms, Inc. and affiliates.


from systems.utils import update_module_step
from models.base import BaseModel


class BaseDeformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def prepare(self, batch):
        pass


class DummyNonRigidDeformer(BaseDeformer):
    def setup(self):
        pass

    def forward(self, points, cond, geometry, *args, with_jac=False, eval_mode=False):
        return points

    def set_initialized(self, is_initialized):
        pass

    def get_rot_mats(self):
        return None

    def get_joints(self):
        return None


class SNARFDeformer(BaseDeformer):
    def setup(self):
        self.n_input_dims = 3
        self.n_output_dims = 3

        from models.deformers.snarf_deformer import SNARFDeformer
        self.rigid_deformer = SNARFDeformer(self.config.rigid_deformer)

        from models.deformers.non_rigid_deformer import DummyNonRigidDeformer
        self.non_rigid_deformer = DummyNonRigidDeformer(self.config.non_rigid_deformer)


    def forward(self, points, points_valid, cond, geometry_fn, *args, with_jac=False, eval_mode=False):

        def non_rigid_geometry_fn(x, valid, cond):
            cond = cond[:, :, :self.config.pose_dim+32]
            x, J_inv = self.non_rigid_deformer(x, cond=cond, with_jac=with_jac)
            ret = geometry_fn(x, valid, cond)
            return ret, J_inv[valid]
        
        return self.rigid_deformer(
            points,
            points_valid,
            cond,
            non_rigid_geometry_fn,
            eval_mode=eval_mode, ## true
        )

    def set_initialized(self, is_initialized):
        if hasattr(self.rigid_deformer, "initialized"):
            self.rigid_deformer.initialized = is_initialized
        if hasattr(self.non_rigid_deformer, "initialized"):
            self.non_rigid_deformer.initialized = is_initialized

    def get_rot_mats(self):
        return self.rigid_deformer.rot_mats

    def get_axis_angles(self):
        rots = self.rigid_deformer.rot_mats
        from common.rot import rotation_matrix_to_angle_axis
        aa = rotation_matrix_to_angle_axis(rots.view(-1, 3, 3)).reshape(rots.shape[0], -1)
        return aa
    
    def get_joints(self):
        return self.rigid_deformer.basic_joints
    
    def get_vertices(self):
        ## MANO deform space
        return self.rigid_deformer.vertices

    def get_vertices_world(self):
        return self.rigid_deformer.v3d_w
    
    def get_faces(self):
        return self.rigid_deformer.faces

    def update_step(self, epoch, global_step):
        update_module_step(self.rigid_deformer, epoch, global_step)
        update_module_step(self.non_rigid_deformer, epoch, global_step)
