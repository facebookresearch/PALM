#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import os.path as op

from systems.base import BaseSystem
from utils.metrics import compute_metrics
import utils.vis_utils as vis_utils
from common.xdict import xdict
from torchvision import transforms



pil2tensor = transforms.ToTensor()

class PALM(BaseSystem):
    def training_step(self, batch, batch_idx):
        self.condition_step(batch, mode="train")
        out = self(batch)
        loss_dict = self.compute_loss_fn(
            out,
            batch,
            self.global_step,
            self.model.enable_phys,
            self.model.add_emitter,
            self,
        )
        loss = sum([loss_dict[k] for k in loss_dict.keys()])
        loss_dict["loss"] = loss

        self.log_dict(xdict(loss_dict).prefix("0-"))

        for fname in batch['fname']:
            self.file_names.add(fname)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.condition_step(batch, mode="val")
        out = self(batch)
        metric_dict = compute_metrics(out, batch)

        exp_id = self.save_dir.split('/')[1]
        img_dir = op.join(self.save_dir, '..', exp_id)
        
        vis_img = vis_utils.visualize_results(
            img_dir, self.global_step, batch, out
        )

        img_id = f"it{self.global_step:06}-{batch['subject_id'][0]}-{batch['img_id'][0]}-{int(batch['cam_idx'])}.png"
        self.logger.experiment.add_image(
            f"0-Images/" + img_id, pil2tensor(vis_img), self.global_step
        )
        if not self.config.no_mesh:
            self.export(img_id)
        
        return metric_dict

    def validation_epoch_end(self, out):
        if not self.has_step_train:
            return 
        # Save environment map
        if hasattr(self.model, "emitter"):
            env_map = self.model.emitter.generate_image()
            self.save_image_grid(
                f"it{self.global_step}-envmap.exr",
                [
                    {"type": "hdr", "img": env_map, "kwargs": {"data_format": "HWC"}},
                ],
            )

        if self.config.personalize:
            from utils.status import status_updater
            status_updater.record(f'validated {self.config.exp_dir} at step {self.global_step}', self.config.seq_idx)

        if self.global_rank == 0:
            self.trainer.save_checkpoint(op.join(self.config.ckpt_dir, f'it{self.global_step:06}.ckpt'))

    def test_step(self, batch, batch_idx):
        self.condition_step(batch, mode="test")
        with torch.no_grad():
            out = self(batch)
        vis_utils.visualize_results(self.render_dir, self.global_step, batch, out)
        return {}

    def test_epoch_end(self, out):
        from utils.status import status_updater
        status_updater.record(f'Finished relighting {self.config.exp_dir} at step {self.global_step}', self.config.seq_idx)
