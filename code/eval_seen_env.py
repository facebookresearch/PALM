import numpy as np
import torch
from glob import glob
from PIL import Image
import lpips
from systems.criterions import PSNR, SSIM, LPIPS
from systems.utils import read_file_to_tuples
import sys
sys.path = ['..'] + sys.path
from common.ld_utils import ld2dl
criterions = {
    "psnr": PSNR(),
    "ssim": SSIM(),
    'lpips': LPIPS(),
}
loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
loss_fn_vgg.eval();


def set_non_foreground_to_white(im_gt, mask):
    """
    Sets non-foreground areas of the ground truth image to white based on the mask.
    Parameters:
    - im_gt: A PIL Image object of the ground truth image.
    - mask: A PIL Image object of the mask.
    Returns:
    - A PIL Image with non-foreground areas set to white.
    """
    # Convert the mask to a binary array (foreground is 255, background is 0)
    mask_arr = np.array(mask)
    foreground_mask = mask_arr == 255
    # Convert the ground truth image to a numpy array
    im_gt_arr = np.array(im_gt)
    # Set non-foreground areas to white (255, 255, 255)
    im_gt_arr[~foreground_mask] = [255, 255, 255]
    # Convert the modified array back to a PIL Image
    im_gt_white_bg = Image.fromarray(im_gt_arr)
    return im_gt_white_bg

def evaluate_image_pair(im_pred, im_gt, im_mask):
    H, W = im_gt.size
    im_pred = torch.FloatTensor(np.array(im_pred)).view(-1, 3)/255
    im_gt = torch.FloatTensor(np.array(im_gt)).view(-1, 3)/255
    
    
    mask = torch.BoolTensor(
        np.array(
            im_mask
        )==255
        )

    
    mask = mask.view(-1)
    # mask[:] = True
    im_gt[~mask] = 1.0 # remove background
    psnr = float(criterions["psnr"](
        im_pred,
        im_gt,
        valid_mask=mask,
    ).cpu())
    
    ssim = criterions["ssim"](
        im_pred.reshape(H, W, 3),
        im_gt.reshape(H, W, 3),
        valid_mask=mask.reshape(H, W)
    )
    im_pred_normalize = im_pred*2 - 1
    im_gt_normalize = im_gt*2 - 1
    
    lpips = float(criterions["lpips"](
        im_pred_normalize.reshape(H, W, 3).cuda(),
        im_gt_normalize.reshape(H, W, 3).cuda(),
        loss_fn_vgg,
        valid_mask=mask.reshape(H, W).cuda()
    ).view(-1)[0].cpu())
    metrics = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
    return metrics



def eval_loop(data):
    
    metrics_all = []
    for pred_im, gt_im, mask in data:
        metrics = evaluate_image_pair(pred_im, gt_im, mask)
        metrics_all.append(metrics)
        
    metrics_all = ld2dl(metrics_all)
    
    for key, val in metrics_all.items():
        metrics_all[key] = np.array(val)
    
    avg_metrics = {}
    for key, val in metrics_all.items():
        avg_metrics[key] = val.mean()
    return metrics_all, avg_metrics

def palm_getter(exp_key, seq_name):
    print(seq_name)
    
    front_fnames = sorted(glob(f'./exp/{exp_key}/{exp_key}/pbr_rgb/*'))
    front_gt_fnames = sorted(glob(f'./load/personalize/{seq_name}/folders/_val/images/MCU_02/*'))
    fnames = front_fnames
    gt_fnames = front_gt_fnames

    fnames = fnames
    gt_fnames = gt_fnames

    assert len(fnames) == len(gt_fnames)
    data = []
    for pred_im, gt_im in zip(fnames, gt_fnames):
        im_pred = Image.open(pred_im)
        im_gt = Image.open(gt_im).convert('RGB')
        mask = Image.open(gt_im.replace('/images/', '/masks_crop/').replace('.jpg', '.png'))

        im_gt = set_non_foreground_to_white(im_gt, mask)
        data.append((im_pred, im_gt, mask))
    return data



runs = read_file_to_tuples('./exp_ids.txt')
assert len(runs) == 12, "We used 12 sequences to evaluate in our paper"

palm_all = {}

for run in runs:
    seq_name, exp_id = run
    palm_data = palm_getter(exp_id, seq_name)
    palm_metrics_all, palm_avg_metrics = eval_loop(palm_data)
    palm_all[seq_name] = palm_avg_metrics


def calculate_averages(data, keys=None):
    # Initialize sums for each metric
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    # Determine which entries to process
    entries_to_process = data.keys() if keys is None else keys
    # Number of entries
    num_entries = len(entries_to_process)
    # Sum up each metric for the specified keys
    for key in entries_to_process:
        if key in data:
            entry = data[key]
            psnr_sum += entry['psnr']
            ssim_sum += entry['ssim']
            lpips_sum += entry['lpips']
    # Calculate averages
    psnr_avg = psnr_sum / num_entries
    ssim_avg = ssim_sum / num_entries
    lpips_avg = lpips_sum / num_entries
    # Return the averages as a dictionary
    return {
        'psnr_avg': psnr_avg,
        'ssim_avg': ssim_avg,
        'lpips_avg': lpips_avg
    }

avg_all = {}
avg_all['palm'] = calculate_averages(palm_all)

print('Method\tPSNR\tSSIM\tLPIPS')
for method, metrics in avg_all.items():
    out_str = method + "\t" 
    for mname, mval in metrics.items():
        out_str += f"{mval:.2f}\t"
    print(out_str)