#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn



def init_param_dict(keys, num_rows, dim):
    param_dict = {}
    for subject_id in keys:
        param_dict[subject_id] = torch.zeros(num_rows, dim)
    param_dict = nn.ParameterDict(param_dict)
    return param_dict

class PoseCorrection(nn.Module):
    def __init__(self, subject_ids, config):
        super().__init__()
        self.config = config
        self.num_frames = 60

        # Pose correction
        pose_dim = 45 ## MANO
        self.pose_correct = init_param_dict(subject_ids, self.num_frames, pose_dim)
        self.shape_correct = init_param_dict(subject_ids, 1, 10)
        self.global_orient_correct = init_param_dict(subject_ids, self.num_frames, 3)
        self.transl_correct = init_param_dict(subject_ids, self.num_frames, 3)

        self.enable_pose_correction = False

    def forward(self, frame_idx, subject_id):
        assert torch.all(frame_idx < self.num_frames)
        if self.enable_pose_correction:
            betas_correction = []
            global_orient_correction = []
            transl_correction = []
            pose_correction = []
            for sid, fid in zip(subject_id, frame_idx):
                betas_correction.append(self.shape_correct[sid])
                global_orient_correction.append(self.global_orient_correct[sid][fid])
                transl_correction.append(self.transl_correct[sid][fid])
                pose_correction.append(self.pose_correct[sid][fid])
            betas_correction = torch.cat(betas_correction, dim=0)
            global_orient_correction = torch.stack(global_orient_correction, dim=0)
            transl_correction = torch.stack(transl_correction, dim=0)
            pose_correction = torch.stack(pose_correction, dim=0)
        else:
            betas_correction = torch.zeros(len(frame_idx), 10).cuda()
            global_orient_correction = torch.zeros(len(frame_idx), 3).cuda()
            transl_correction = torch.zeros(len(frame_idx), 3).cuda()
            pose_correction = torch.zeros(len(frame_idx), 45).cuda()

        return {
            "betas_correction": betas_correction,
            "global_orient_correction": global_orient_correction,
            "transl_correction": transl_correction,
            "pose_correction": pose_correction,
        }
    
    def update_step(self, epoch, global_step):
        if (
            self.config.enable_pose_correction
            and global_step > self.config.pose_correction_start_step
        ):
            self.enable_pose_correction = True
