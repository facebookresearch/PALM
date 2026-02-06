#  Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import random
import os.path as op

from pytorch_lightning.utilities.rank_zero import rank_zero_debug
import numpy as np

from datetime import datetime
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.callbacks import CustomProgressBar, TrialNameCallback

def parse_args(config):
    duration = random.random()*30
    print(f'Cooling down for {duration} second')
    # time.sleep(duration)
    from systems.utils import generate_trial_name
    config.trial_name = generate_trial_name()
    if "seed" not in config:
        config.seed = 4127 ## ensure same seed across gpus

    if config.personalize:
        if config.seq_idx != -1 and config.seq_name is None:
            config.dataset.dataroot = config.dataset.dataroot.replace('<<<SEQNAME>>>', seq_lists[config.seq_idx])
        elif config.seq_name is not None:
            config.dataset.dataroot = config.dataset.dataroot.replace('<<<SEQNAME>>>', config.seq_name)
        else:
            assert False, f"config.seq_name={config.seq_name}, config.seq_idx={config.seq_idx}"

        # config.trainer.max_steps = 5200
    else: 
        if config.relight:
            from systems.utils import read_file_to_tuples, get_hdri_path
            # seq_name, ckpt_name = runs[config.seq_idx]
            seq_name = config.seq_name
            # ckpt_name = config.resume
            hdri_path = get_hdri_path(seq_name)
            config.dataset.hdri_filepath = hdri_path
            # config.resume = f'exp/{ckpt_name}/ckpt/0/it005100.ckpt'
            assert op.exists(config.resume), config.resume
            config.dataset.dataroot = config.dataset.dataroot.replace('<<<SEQNAME>>>', seq_name)
        
    n_gpus = len(config.gpu.split(","))
    strategy = None
    if n_gpus > 1:
        strategy = 'ddp'
        config.dataset.opt.train.shuffle = False

    ## sanity run for OOM issues etc before longer training OR coding
    if config.sanity or config.fast:
        config.model.warmup_steps = 5
        config.model.normal_start_step = 5
        config.model.vgg_start_step=5
        config.model.importance_sample_kick_in_step=5
        config.model.phys_kick_in_step=5
        
    ## coding
    if config.fast:
        config.sampler.num_sample=128
        config.model.warmup_steps = 10000
        config.model.phys_kick_in_step=20
        config.trainer.val_check_interval=5
        config.dataset.opt.train.num_workers = 0

    ## rf only long training
    if config.rf_only:
        config.model.phys_kick_in_step=100000000

    config.dataset.opt.train.personalize = config.personalize
    config.dataset.opt.val.personalize = config.personalize

    ## cluster training
    if config.cluster:
        config.refresh_rate=50

    ## pretraining geometry network
    if config.pretrain:
        config.model.phys_kick_in_step=500000
        config.trainer.max_steps = 300000
        config.sampler.num_sample=32
        config.dataset.opt.train.use_vgg = False

    from systems.utils import parse_subject_ids
    config.dataset.opt.train.subject_ids = parse_subject_ids(config.dataset.opt.train.subject_ids)
    config.dataset.opt.val.subject_ids = parse_subject_ids(config.dataset.opt.val.subject_ids)
    config.dataset.opt.test.subject_ids = parse_subject_ids(config.dataset.opt.test.subject_ids)

    batch_size = config.model.batch_size # total batch size
    loader_batch_size = config.model.loader_size
    if batch_size < loader_batch_size:
        loader_batch_size = batch_size
        
    config.dataset.opt.train.batch_size = loader_batch_size # actual batch size each it
    num_acc_grad = batch_size//(n_gpus*loader_batch_size)
    assert num_acc_grad >= 1
    config.trainer.val_check_interval *= num_acc_grad
    return config, num_acc_grad, strategy, n_gpus

def get_callbacks(config):
    callbacks = []
    if config.mode == "train":
        callbacks += [
            LearningRateMonitor(logging_interval="step"),
            CustomProgressBar(refresh_rate=config.refresh_rate),
            TrialNameCallback(trial_name=config.trial_name)
        ]
    else:
        callbacks += [
            TrialNameCallback(trial_name=config.trial_name)
        ]
    return callbacks

def read_file_to_tuples(file_path):
    """
    Reads a text file where each line is a pair of strings separated by a tab,
    trims spaces, and returns a list of tuples.
    :param file_path: Path to the text file
    :return: List of tuples containing the string pairs
    """
    tuples_list = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the line and split it by tab
            parts = line.strip().split('\t')
            # Create a tuple from the parts and strip any extra spaces
            tuple_pair = (parts[0].strip(), parts[1].strip())
            # Append the tuple to the list
            tuples_list.append(tuple_pair)
    return tuples_list

def get_hdri_path(seq_name):
    # Example of HDRI URL:
    # https://polyhaven.com/a/preller_drive
    envs = [
        "christmas_photo_studio_01_4k",
        "city",
        "cobblestone_street_night_4k",
        "cyclorama_hard_light_4k",
        "flamingo_pan_4k",
        "industrial_sunset_puresky_4k",
        "lakeside_sunrise_4k",
        "little_paris_eiffel_tower_4k",
        "lonely_road_afternoon_puresky_8k",
        "moonless_golf_4k",
        "neon_photostudio_4k",
        "overcast_soil_puresky_4k",
        "photo_studio_loft_hall_4k",
        "pond_bridge_night_4k",
        "preller_drive_4k",
        "rosendal_plains_2_4k",
        "snowy_park_01_4k",
        "studio_small_04_4k",
        "studio_small_05_4k",
        "studio_small_07_4k",
        "victoria_sunset_4k",
        "warm_restaurant_night_4k",
    ]
    hdri_name = seq_name.split('__')[-1]
    hdri_name = random.choice(envs)
    if op.exists(f'./envs/{hdri_name}.exr'):
        return f'./envs/{hdri_name}.exr'
    assert False, f'Not found {hdri_name}'

def parse_subject_ids(subject_ids):
    if ':' in subject_ids[0]:
        all_ids = []
        for subj_id_range in subject_ids:
            from_id, to_id = subj_id_range.split(':')
            myids = [f"{sid:04}" for sid in range(int(from_id), int(to_id)+1)]
            all_ids = all_ids + myids
        subject_ids = all_ids
    subject_ids = sorted(subject_ids)
    return subject_ids

def generate_trial_name():
    now = datetime.now()
    trial_name = now.strftime("%m%d-%H%M-%S") + str(now.microsecond)[:1]
    if op.exists(op.join('exp', trial_name)):
        duration = random.random()*30
        print(f'Trial name exists. Cooling down for {duration} second')
        # time.sleep(duration)
        return generate_trial_name()
    return trial_name



def split_batch_into_k(batch, k):
    """
    Splits a batch into k smaller batches.
    Args:
        batch (dict): The input batch.
        k (int): The number of smaller batches to split the input batch into.
    Returns:
        list: A list of dictionaries, where each dictionary represents a smaller batch.
    """
    # Calculate the size of each smaller batch
    batch_size = None
    for key, val in batch.items():
        if hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] > 1:
            batch_size = val.shape[0]
            break
    if batch_size is None:
        raise ValueError("Could not determine batch size")

    base_batch_size_per_k, remainder = divmod(batch_size, k)

    # Initialize an empty list to store the smaller batches
    batch_list = [{} for _ in range(k)]

    # Iterate over each key-value pair in the input batch
    for key, val in batch.items():
        # If the value has a shape attribute and its first dimension is greater than 1,
        # it's likely a pixel-related entry, so we split it into k smaller batches
        if hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] > 1:
            start_idx = 0
            for i in range(k):
                batch_size_per_k = base_batch_size_per_k + (1 if i < remainder else 0)
                end_idx = start_idx + batch_size_per_k
                batch_list[i][key] = val[start_idx:end_idx]
                start_idx = end_idx
        # Otherwise, we simply copy the value into each smaller batch
        else:
            for i in range(k):
                batch_list[i][key] = val

    return batch_list


def merge_outputs(out_list):
    """
    Merges a list of outputs into a single output.
    Args:
        out_list (list): A list of dictionaries, where each dictionary represents an output.
    Returns:
        dict: The merged output.
    """
    # Initialize an empty dictionary to store the merged output
    merged_out = {}
    # Define keys for different merge strategies
    concat_dim0_keys = [
        "comp_rgb",
        "comp_normal",
        "fg_normal",
        "opacity",
        "depth",
        "rays_valid",
        "rays_valid_phys",
        "normals_orientation_loss_map",
        "albedo_smoothness_loss_map",
        "roughness_smoothness_loss_map",
        "metallic_smoothness_loss_map",
        "comp_rgb_bg",
        "rays_valid_bg",
        "rays_valid_phys_bg",
        "comp_rgb_full",
        "rays_valid_full",
        "rays_valid_phys_full",
        "sdf_samples",
        "sdf_grad_samples",
        "sdf_laplace_samples",
        "weights",
        "points",
        "intervals",
        "ray_indices",
        "rays_valid_phys_full",
        "comp_rgb_phys_full",
        "comp_demod_phys_full",
        "comp_albedo_full",
        "comp_metallic_full",
        "comp_roughness_full",
        "comp_rgb_phys",
        "comp_demod_phys",
        "comp_albedo",
        "comp_metallic",
        "comp_roughness",
        "visibility",
        "comp_albedo_bg",
        "comp_metallic_bg",
        "comp_roughness_bg",
    ]
    concat_dim1_sdf_samples_keys = ["pred_sdf", "gt_sdf", "pred_density"]
    sum_keys = ["num_samples_bg", "num_samples_full", "num_samples"]
    average_keys = ["gain", "bias", "beta"]
    # Iterate over each key-value pair in the first output
    for key, val in out_list[0].items():
        if key in concat_dim0_keys:
            merged_out[key] = torch.cat([out[key] for out in out_list], dim=0)
        elif key in concat_dim1_sdf_samples_keys:
            merged_out[key] = torch.cat([out[key] for out in out_list], dim=1)
        elif key in sum_keys:
            merged_out[key] = sum(out[key] for out in out_list)
        elif key in average_keys:
            merged_out[key] = sum(out[key] for out in out_list) / len(out_list)
        else:
            assert False, f"Unknown key: {key}"
    return merged_out


class ChainedScheduler(lr_scheduler._LRScheduler):
    """Chains list of learning rate schedulers. It takes a list of chainable learning
    rate schedulers and performs consecutive step() functions belong to them by just
    one call.

    Args:
        schedulers (list): List of chained schedulers.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.09     if epoch == 0
        >>> # lr = 0.081    if epoch == 1
        >>> # lr = 0.729    if epoch == 2
        >>> # lr = 0.6561   if epoch == 3
        >>> # lr = 0.59049  if epoch >= 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers):
        for scheduler_idx in range(1, len(schedulers)):
            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(
                        0, scheduler_idx
                    )
                )
        self._schedulers = list(schedulers)
        self.optimizer = optimizer

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class SequentialLR(lr_scheduler._LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially during
    optimization process and milestone points that provides exact intervals to reflect
    which scheduler is supposed to be called at a given epoch.

    Args:
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.

    Example:
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(self.opt, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        >>> scheduler = SequentialLR(self.opt, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(1, len(schedulers)):
            if schedulers[scheduler_idx].optimizer != schedulers[0].optimizer:
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    "got schedulers at index {} and {} to be different".format(
                        0, scheduler_idx
                    )
                )
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(
                    len(schedulers), len(milestones)
                )
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class ConstantLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False
    ):
        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters or (self.last_epoch != self.total_iters):
            return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch == self.total_iters:
            return [
                group["lr"] * (1.0 / self.factor)
                for group in self.optimizer.param_groups
            ]

    def _get_closed_form_lr(self):
        return [
            base_lr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]


class LinearLR(lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        verbose=False,
    ):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                "Starting multiplicative factor expected to be between 0 and 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]


custom_schedulers = ["ConstantLR", "LinearLR"]


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    elif name in custom_schedulers:
        return getattr(sys.modules[__name__], name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        rank_zero_debug("Specify optimizer params:", config.params)
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "Chained":
        scheduler = {
            "scheduler": ChainedScheduler(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler


def update_module_step(m, epoch, global_step):
    if hasattr(m, "update_step"):
        m.update_step(epoch, global_step)
