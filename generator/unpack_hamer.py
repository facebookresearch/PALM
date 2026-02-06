import os.path as op
import numpy as np
import os

def main(hpe_path):
    all_out = np.load(hpe_path, allow_pickle=True).item()

    keys = list(all_out.keys())

    seq_names = [op.basename(key).replace('-', '_').replace('.jpg', '').replace('pexels_', '').split('_')[0] for key in keys]
    seq_names = ['itw_' + sn for sn in seq_names]
    assert len(seq_names) == len(set(seq_names))

    pairs = list(zip(keys, seq_names))

    for key, seq_name in pairs:
        out = all_out[key]
        
        out_p = f'./data/{seq_name}/images/0001.jpg'
        
        os.makedirs(op.dirname(out_p), exist_ok=True)
        out['im'].save(out_p)
        
        out_p = out_p.replace('/images/', '/crop_image/')
        
        os.makedirs(op.dirname(out_p), exist_ok=True)
        out['im'].save(out_p)
        
        myv3d = {}
        myv3d['v3d.right'] = out['v3d_cam'][:1]
        myv3d['im_paths'] = out_p
        
        out_process_p = f'./data/{seq_name}/processed/v3d.npy'
        os.makedirs(op.dirname(out_process_p), exist_ok=True)
        np.save(out_process_p, myv3d)
        
        out_process_p = f'./data/{seq_name}/processed/K_hamer.npy'
        print(out_process_p)
        np.save(out_process_p, out['K'])



def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpe_path", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.hpe_path)
