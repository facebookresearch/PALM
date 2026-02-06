
import logging
import torch
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import os.path as op

seq_lists = [
    "itw_red",
    "itw_anna",
    "itw_directional",
    "itw_hike",
    "itw_jeswin",
    "itw_kevin",
    "itw_light",
    "itw_nana",
    "itw_omair",
    "itw_phael",
    "itw_samuel",
    "itw_santiago",
    "itw_sky",
    "itw_thirdman",
    "itw_white",
    "itw_yoga",
]


def write_to_file(line, filename):
    """
    Write a line to a file.
    If the file does not exist, it will be created. Otherwise, the line will be appended to the existing file.
    Args:
        line (str): The line to write to the file.
        filename (str, optional): Name of the file to write to. Defaults to './exp_ids.txt'.
    """
    # Create the directory if it doesn't exist
    import os
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # Write or append the line to the file
    with open(filename, 'a+') as f:
        f.write(line + "\n")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print('Hello world')
    from systems.utils import parse_args, get_callbacks
    config, num_acc_grad, strategy, n_gpus = parse_args(config)
    logger = logging.getLogger("pytorch_lightning")
    if config.verbose:
        logger.setLevel(logging.DEBUG)
    pl.seed_everything(config.seed)
    
    print('Constructing dataset')
    from datasets.image_dataset import ImageDatasetDataModule
    dm = ImageDatasetDataModule(config.dataset)
    from systems.palm import PALM
    system = PALM(config).to("cuda")

    callbacks = get_callbacks(config)
    trainer = Trainer(
        devices=n_gpus,
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=0.5,
        strategy=strategy,
        accumulate_grad_batches=num_acc_grad,
        **config.trainer,
    )
    if config.seq_name != "" and config.seq_name is not None:
        write_to_file(config.seq_name + "\t" + config.trial_name, './exp_ids.txt')
    print('Starting trainer.fit()')
    if config.mode == "train":
        if config.resume and not config.resume_weights_only:
            trainer.fit(system, datamodule=dm, ckpt_path=config.resume)
        elif config.resume_weights_only: # personalization training OR render seen env on IH
            sd = torch.load(config.resume, map_location='cpu')['state_dict']
            new_sd = system.state_dict()
            keys = list(new_sd.keys())
            for key in keys:
                if key in sd.keys():
                    new_sd[key] = sd[key]
                else:
                    print("Skipping: ", key)
            system.load_state_dict(new_sd, strict=False)
            if config.shape_code_basis_path is None:
                ckpt_exp_id = config.resume.split('/')[1]
                shape_basename = config.resume.split('/')[-1].replace('.ckpt', '.shape.pca.npy')
                appearance_basename = config.resume.split('/')[-1].replace('.ckpt', '.appearance.pca.npy')
                shape_code_basis_path = op.join("pca_basis", ckpt_exp_id, shape_basename)
                appearance_code_basis_path = op.join("pca_basis", ckpt_exp_id, appearance_basename)
            else:
                shape_code_basis_path = config.shape_code_basis_path
                appearance_code_basis_path = config.appearance_code_basis_path
            system.model.shape_code.load_basis(shape_code_basis_path)
            system.model.appearance_code.load_basis(appearance_code_basis_path)
        else: # regular training
            if not config.pretrain:
                old_sd = torch.load(
                    f"./data/pretrained/{config.model.pretrain_version}", map_location="cpu"
                )
                from common.xdict import xdict
                old_sd = old_sd['state_dict'] if 'state_dict' in old_sd else old_sd
                old_sd = xdict(old_sd)
                new_sd = xdict()
                for key in ['geometry', 'density.beta', 'pose_encoder', 'shape_code']:
                    new_sd.merge(old_sd.search(key))
                assert len(new_sd) > 0
                system.load_state_dict(new_sd, strict=False)
        trainer.fit(system, datamodule=dm)            
    elif config.mode == "test":
        from common.xdict import xdict
        checkpoint = torch.load(config.resume, map_location="cpu")
        sd = xdict(checkpoint["state_dict"])
        sd = sd.rm("occupancy_grid", verbose=True)

        system.load_state_dict(sd, strict=False)
        shape_code_basis_path = config.shape_code_basis_path
        appearance_code_basis_path = config.appearance_code_basis_path
        system.model.shape_code.load_basis(shape_code_basis_path)
        system.model.appearance_code.load_basis(appearance_code_basis_path)
        trainer.test(system, datamodule=dm, ckpt_path=None)
    else:
        assert False, "Invalid mode"


if __name__ == "__main__":
    main()
