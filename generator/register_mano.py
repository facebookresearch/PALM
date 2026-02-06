import os
import numpy as np
import os.path as op
import torch
import smplx
from tqdm import tqdm
import sys

sys.path = ["."] + sys.path
from src.hand_pose.registration import fit_frame

sys.path = [".."] + sys.path
from common.ld_utils import ld2dl
from common.xdict import xdict
MANO_DIR_L = "../code/data/body_models/MANO_LEFT.pkl"
MANO_DIR_R = "../code/data/body_models/MANO_RIGHT.pkl"

mano_layers = {
    "right": smplx.create(
        model_path=MANO_DIR_R, model_type="mano", use_pca=False, is_rhand=True, flat_hand_mean=True
    ),
    "left": smplx.create(
        model_path=MANO_DIR_L, model_type="mano", use_pca=False, is_rhand=False, flat_hand_mean=True
    ),
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--use_beta_loss", action="store_true")
    parser.add_argument("--hand_type", type=str, default=None)
    args = parser.parse_args()
    return args


def fit_single_hand(v3d_ra_list, seq_name, args, is_right):
    import copy

    pbar = tqdm(enumerate(v3d_ra_list), total=len(v3d_ra_list))
    prev_out = None
    out_list = []
    for iteration, v3d_ra in pbar:
        pbar.set_description(
            "Processing %s [%d/%d]" % (seq_name, iteration + 1, len(v3d_ra_list))
        )
        is_valid = np.isnan(v3d_ra).sum() == 0
        if not is_valid:
            out = {}
            out["global_orient"] = torch.zeros(1, 3).cuda() * np.nan
            out["hand_pose"] = torch.zeros(1, 45).cuda() * np.nan
            out["betas"] = torch.zeros(1, 10).cuda() * np.nan
            out["transl"] = torch.zeros(1, 3).cuda() * np.nan
        else:
            out = fit_frame(
                mano_layers,
                seq_name,
                v3d_ra,
                save_mesh=args.save_mesh,
                init_params=prev_out,
                iteration=iteration,
                is_right=is_right,
                first_frame=prev_out is None,
                use_beta_loss=args.use_beta_loss,
            )
            prev_out = copy.deepcopy(out)
        out_list.append(out)

    out_dict = ld2dl(out_list)
    out_dict = dict(
        xdict({key: torch.cat(val, axis=0) for key, val in out_dict.items()}).to_np()
    )
    return out_dict


def register_sequence(seq_name, args):
    data = np.load(f"./data/{seq_name}/processed/v3d.npy", allow_pickle=True).item()
    data = xdict(data).search("v3d.")

    if args.hand_type is not None:
        data = data.search(args.hand_type)

    out_dict = {}
    for key, val in data.items():
        print("Processing " + key)
        flag = key.split(".")[1]
        is_right = flag == "right"

        mydict = fit_single_hand(val, seq_name, args, is_right=is_right)
        out_dict[flag] = mydict
    out_p = f"./data/{seq_name}/processed/hold_fit.init.npy"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    np.save(out_p, out_dict)
    print(f"Saved to {out_p}")

def main():
    args = parse_args()
    seq_name = args.seq_name
    if seq_name != "":
        register_sequence(seq_name, args)
    else:
        from glob import glob
        fnames = sorted(glob('./data/*'))
        seq_names = [op.basename(fname) for fname in fnames]
        for seq_name in seq_names:
            register_sequence(seq_name, args)



if __name__ == "__main__":
    main()
