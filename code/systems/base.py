#  Copyright (c) Meta Platforms, Inc. and affiliates.

import pytorch_lightning as pl
from systems.utils import parse_optimizer, parse_scheduler
from utils.mixins import SaverMixin
import time
import os
from omegaconf import OmegaConf
import torch
from lib.torch_pbr.utils.nvdiffrecmc_util import rgb_to_srgb
from systems.utils import update_module_step

class BaseSystem(pl.LightningModule, SaverMixin):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        from models.palm import PALMNet
        self.model = PALMNet(self.config.model)
        self.start_time = time.time()
        self.has_step_train = False
        self.file_names = set()

    def on_test_start(self) -> None:
        self.on_fit_start()

    def on_fit_start(self) -> None:
        callbacks = self.trainer.callbacks
        rank_str = str(self.global_rank)
        config = self.config
        config.exp_dir = os.path.join(config.exp_dir, self.trial_name)
        config.log_dir = os.path.join(config.exp_dir, "log", rank_str)
        config.save_dir = os.path.join(config.exp_dir, "save")
        config.render_dir = os.path.join(config.exp_dir, "render")
        config.ckpt_dir = os.path.join(config.exp_dir, "ckpt", rank_str)
        config.code_dir = os.path.join(config.exp_dir, "code", rank_str)
        config.config_dir = os.path.join(config.exp_dir, "config", rank_str)

        from pytorch_lightning.loggers import TensorBoardLogger
        # Set up TensorBoard logger

        if self.global_rank == 0:
            print(OmegaConf.to_yaml(config), flush=True)
            from common.sys_utils import copy_repo
            copy_repo(config.code_dir)

        if config.mode == 'train':
            from pytorch_lightning.callbacks import ModelCheckpoint
            callbacks += [
                ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
            ]
            
        tb_logger = (
            TensorBoardLogger(name="IA", save_dir=config.log_dir)
            if config.mode == "train"
            else None
        )
        self.trainer.logger = tb_logger
        self.trainer.callbacks = callbacks
        self.config = config
        print("Starting experiment", config.exp_dir, self.global_rank)
        if self.config.seq_idx != -1:
            from utils.status import status_updater
            status_updater.record(f'Relighting ckpt={config.exp_dir} with env={config.dataset.hdri_filepath}: job_id=[{config.job_id}] personlize=[{config.dataset.dataroot}]', config.seq_idx)


    def on_train_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.train_dataloader().dataset

    def on_validation_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.val_dataloader().dataset

    def on_test_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.test_dataloader().dataset

    def on_predict_epoch_start(self) -> None:
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        
    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            "optimizer": optim,
        }
        if "scheduler" in self.config.system:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(
                        self.config.system.scheduler, optim
                    ),
                }
            )
        return ret

    def export(self, img_id):
        try:
            mesh = self.model.export(self.config.export)
            self.save_mesh(
                f"mc/{img_id}.obj",
                **mesh,
            )
        except:
            print(f'WARNING: failed to mesh {img_id}')

    def condition_train(self, batch):
        from common.torch_utils import toggle_parameters
        warmup_steps = self.config.model.warmup_steps
        sanity = self.config.sanity
        pretrain_version = self.config.model.pretrain_version
        global_step = self.global_step
        pretrain = self.config.pretrain

        ## freeze vs. defrost
        if not pretrain:
            if global_step == 0:
                # freeze pretrained shape
                toggle_parameters(self.model.geometry, False)
                toggle_parameters(self.model.pose_encoder, False)
                toggle_parameters(self.model.shape_code, False)
                toggle_parameters(self.model.deformer, False)

            if global_step == warmup_steps:
                toggle_parameters(self.model.geometry, True)
                toggle_parameters(self.model.pose_encoder, True)
                toggle_parameters(self.model.shape_code, True)
                toggle_parameters(self.model.appearance_code, True)
                toggle_parameters(self.model.deformer, True)

        if sanity:
                toggle_parameters(self.model.geometry, True)
                toggle_parameters(self.model.pose_encoder, True)
                toggle_parameters(self.model.shape_code, True)
                toggle_parameters(self.model.appearance_code, True)
                toggle_parameters(self.model.deformer, True)                

        self.has_step_train = True
        if pretrain:
            update_module_step(self.model, self.current_epoch, self.global_step)
        else:    
            update_module_step(self.model, self.current_epoch, self.global_step)
            update_module_step(self.model.geometry, 10000, 1000000)

        if sanity:
            update_module_step(self.model, 10000, 1000000)
        
        self.log("sys/min", (time.time() - self.start_time) / 60, prog_bar=True)

        # forward batch and step
        self.preprocess_data(batch, "train")

        if global_step % 2000 == 0 and pretrain and self.global_rank == 0:
            print('Saving sd')
            sd = self.state_dict()
            torch.save(sd, f"./data/pretrained/{pretrain_version}")

        # turned on by default
        self.model.with_curvature_loss = True
        self.model.jitter_materials = False

        if global_step < self.config.model.vgg_start_step:
            self.num_forwards = 2
        else:
            self.num_forwards = 8

    def condition_val(self, batch):
        pretrain = self.config.pretrain
        if pretrain:
            update_module_step(self.model, self.current_epoch, self.global_step)
        else:
            update_module_step(self.model, self.current_epoch, self.global_step)

        self.preprocess_data(batch, "validation")

        if not hasattr(self, "num_forwards"):
            self.num_forwards = 2

    def condition_test(self, batch):
        update_module_step(self.model, 250, 25000)
        self.model.enable_phys = self.config.enable_phys
        self.model.pose_correction.enable_pose_correction = True
        self.model.importance_sample = True
        self.model.config.ray_chunk = 1024 * 4 * 2
        self.preprocess_data(batch, "test")

    def condition_personalize(self, batch):
        from common.torch_utils import toggle_parameters
        if self.config.render_interhand:
            self.model.enable_phys = True
            update_module_step(self.model, 10000, 1000000)
            toggle_parameters(self.model, False)
            from utils.zero_loss import compute_zero_loss
            self.compute_loss_fn = compute_zero_loss
        else:
            from utils.personalize_loss import PBR_ENABLE_STEP
            update_module_step(self.model, 10000, 1000000)
            toggle_parameters(self.model, False)
            toggle_parameters(self.model.pose_correction, True)
            toggle_parameters(self.model.shape_code, True)
            toggle_parameters(self.model.appearance_code, True)
            
            if self.global_step > PBR_ENABLE_STEP: 
                toggle_parameters(self.model.emitter, True)
                self.model.enable_phys = True
            else:
                self.model.enable_phys = False
            
            # stop shape code
            if self.global_step > 1500: 
                toggle_parameters(self.model.shape_code, False)

            from utils.personalize_loss import compute_personalize_loss
            self.compute_loss_fn = compute_personalize_loss


    def forward(self, batch):
        return self.model(batch)

    def condition_step(self, batch, mode):
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        from utils.loss import compute_training_loss
        self.compute_loss_fn = compute_training_loss 
        if mode == "train":
            self.condition_train(batch)
        elif mode == "val":
            self.condition_val(batch)
        elif mode == "test":
            self.condition_test(batch)
        else:
            assert False

        if self.config.personalize:
            self.condition_personalize(batch)


    def preprocess_data(self, batch, stage):
        mean_pose = self.model.deformer.rigid_deformer.body_model.hand_pose_offset.cuda()
        batch['body_pose'] += mean_pose[None, :]

        if "normal" in batch:
            batch["normal"] = batch["normal"].reshape(-1, 3)

        if "hdri" in batch:
            assert stage in ["test"]
            assert batch["hdri"].shape[0] == 1
            batch["hdri"] = batch["hdri"].squeeze(0)
        num_pix = batch['rgb'].shape[1]
        rays = torch.cat(
            [
                batch["rays_o"],  ## 3
                batch["rays_d"],  ## 3
                batch["near"][..., None],  ## 1
                batch["far"][..., None],  ## 1
            ],
            dim=-1,
        ).reshape(-1, num_pix, 8)
        batch.pop("rays_o")
        batch.pop("rays_d")
        batch.pop("near")
        batch.pop("far")
        batch["rays"] = rays

        if stage in ["train"]:
            ## training mode
            if self.config.model.background_color == "white":
                self.model.background_color = torch.ones(
                    (3,), dtype=torch.float32
                ).to('cuda')
            elif self.config.model.background_color == "black":
                self.model.background_color = torch.zeros(
                    (3,), dtype=torch.float32
                ).to('cuda')
            elif self.config.model.background_color == "random":
                self.model.background_color = torch.rand(
                    (3,), dtype=torch.float32
                ).to('cuda')
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones(
                (3,), dtype=torch.float32
            ).to('cuda')

        if "rgb" in batch:
            rgb = batch["rgb"].reshape(-1, 3).to('cuda')
            fg_mask = batch["alpha"].reshape(-1).to('cuda')
            rgb_wo_mask = rgb
            rgb = rgb * fg_mask[..., None] + rgb_to_srgb(
                self.model.background_color * (1 - fg_mask[..., None])
            )  ### mask out background with random background color
            batch["rgb"] = rgb
            batch["rgb_wo_mask"] = rgb_wo_mask
            batch["alpha"] = fg_mask

        self.model.prepare(batch)
