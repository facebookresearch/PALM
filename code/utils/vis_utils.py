import os
import os.path as op
from utils.mixins import SaverMixin
from PIL import Image, ImageDraw, ImageFont
import torch
import trimesh
import numpy as np
from common.rend_utils import Renderer

renderer = Renderer(1600)

mixin = SaverMixin()


def render_mano(renderer, v3d_c, faces, K, height, width):
    mesh = trimesh.Trimesh(v3d_c[0].cpu().detach().numpy(), faces=faces, process=False)
    K_np = K[0]
    rend_img = renderer.render_meshes_pose(
        [mesh],
        K=K_np,
    )
    rend_img = rend_img[:height, :width]
    return rend_img


def stack_images_horizontally(ims, captions, img_id):
    """
    Stack a list of PIL images horizontally with captions.
    Args:
        ims (list): A list of PIL images.
        captions (list): A list of captions corresponding to each image.
    Returns:
        PIL Image: The stacked image with captions.
    """
    # Check if the number of images and captions match
    if len(ims) != len(captions):
        raise ValueError("The number of images and captions must match")
    # Add the captions to the images
    captioned_ims = []
    for im, caption in zip(ims, captions):
        width, height = im.size
        max_dim = max(width, height)
        font_size = int(max_dim * 0.05)
        new_im = Image.new(
            "RGB", (width, height + int(font_size * 1.5)), color=(255, 255, 255)
        )
        draw = ImageDraw.Draw(new_im)
        # Draw the text directly onto the new image
        draw.text(
            (0, 0),
            caption,
            font=ImageFont.truetype("./data/Roboto-Light.ttf", font_size),
            fill=(0, 0, 0),
        )
        # If 'arial.ttf' is not found, use the following line instead
        # draw.text((0, 0), caption, font=ImageFont.load_default(), fill=(0, 0, 0))
        new_im.paste(im, (0, int(font_size * 1.5)))
        captioned_ims.append(new_im)
    # Stack the images horizontally
    widths, heights = zip(*(i.size for i in captioned_ims))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in captioned_ims:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Draw img_id at the center bottom of the final image
    draw = ImageDraw.Draw(new_im)
    font_size = int(max_height * 0.05)
    font = ImageFont.truetype("./data/Roboto-Light.ttf", font_size)
    draw.text((0, font_size), img_id, font=font, fill=(0, 0, 0))
    return new_im

def compute_loss_image(rgb_pred, rgb_gt, mask):
    assert len(rgb_pred.shape) == 2
    assert len(rgb_gt.shape) == 2
    diff = torch.abs(rgb_pred - rgb_gt)
    # diff = diff.norm(dim=1)
    diff[~mask.bool()] = 0.0
    return diff

def visualize_results(save_dir, global_step, batch, out):
    img_id = f"it{global_step:06}-{batch['subject_id'][0]}-{batch['img_id'][0]}-{int(batch['cam_idx'])}.png"
    out_p = op.join(save_dir, 'combine', img_id)

    W = int(batch["width"])
    H = int(batch["height"])

    img_list = []

    # render MANO
    rend_img = render_mano(
        renderer,
        batch["v3d_c"],
        batch["faces"],
        batch["K"],
        batch["height"],
        batch["width"],
    )
    rend_img = torch.FloatTensor(rend_img) / 255
    if "rgb" in batch:
        rgb = batch["rgb"].view(H, W, 3).cpu()
        alpha = 0.7
        blend_img = alpha * rend_img + (1 - alpha) * rgb

    # GT RGB
    if "rgb" in batch:
        img_list.append(
            {
                "type": "rgb",
                "img": batch["rgb"].view(H, W, 3),
                "kwargs": {"data_format": "HWC"},
                "caption": "gt_rgb",
            }
        )
        

    # PRED RGB
    img_list.append(
        {
            "type": "rgb",
            "img": out["comp_rgb_full"].view(H, W, 3),
            "kwargs": {"data_format": "HWC"},
            "caption": "rf_rgb",
        }
    )

    if "comp_rgb_phys_full" in out:
        img_list.append(
            {
                "type": "rgb",
                "img": out["comp_rgb_phys_full"].view(H, W, 3),
                "kwargs": {"data_format": "HWC"},
                "caption": "pbr_rgb",
            }
        )



    if "rgb" in batch:
        img_list.append(
            {
                "type": "rgb",
                "img": blend_img,
                "kwargs": {"data_format": "HWC"},
                "caption": "overlay",
            }
        )

    img_list.append(
        {
            "type": "rgb",
            "img": rend_img,
            "kwargs": {"data_format": "HWC"},
            "caption": "mano",
        }
    )


    if 'comp_rgb_full' in out:
        loss_image = compute_loss_image(out['comp_rgb_full'], batch["rgb"].detach().cpu(), batch["alpha"])
        img_list.append(
            {
                "type": "rgb",
                "img": loss_image.view(H, W, 3),
                "kwargs": {"data_format": "HWC"},
                "caption": "loss_rf",
            }
        )

    if 'comp_rgb_phys_full' in out:
        loss_image = compute_loss_image(out['comp_rgb_phys_full'], batch["rgb"].detach().cpu(), batch["alpha"])
        img_list.append(
            {
                "type": "rgb",
                "img": loss_image.view(H, W, 3),
                "kwargs": {"data_format": "HWC"},
                "caption": "loss_pbr",
            }
        )

    # Pred Depth
    img_list.append(
        {
            "type": "grayscale",
            "img": out["depth"].view(H, W),
            "kwargs": {},
            "caption": "depth",
        }
    )

    # Normal
    img_list.append(
        {
            "type": "rgb",
            "img": out["comp_normal"].view(H, W, 3),
            "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
            "caption": "normal",
        }
    )

    # PBR
    if "comp_rgb_phys_full" in out:

        img_list.append(
            {
                "type": "rgb",
                "img": out["comp_demod_phys_full"].view(H, W, 3),
                "kwargs": {"data_format": "HWC"},
                "caption": "pbr_demod",
            }
        )

        img_list.append(
            {
                "type": "rgb",
                "img": out["comp_albedo_full"].view(H, W, 3),
                "kwargs": {"data_format": "HWC"},
                "caption": "pbr_albedo",
            }
        )

        img_list.append(
            {
                "type": "grayscale",
                "img": out["comp_metallic_full"].view(H, W),
                "kwargs": {"data_range": (0, 1), "cmap": None},
                "caption": "pbr_metallic",
            }
        )

        img_list.append(
            {
                "type": "grayscale",
                "img": out["comp_roughness_full"].view(H, W),
                "kwargs": {"data_range": (0, 1), "cmap": None},
                "caption": "pbr_roughness",
            }
        )

        if "visibility" in out:
            img_list.append(
                {
                    "type": "grayscale",
                    "img": out["visibility"].view(H, W),
                    "kwargs": {"data_range": (0, 1), "cmap": None},
                    "caption": "visibility",
                }
            )

    _, imgs = mixin.get_image_grid_(img_list)
    ims = [Image.fromarray(img) for img in imgs]
    captions = [img["caption"] for img in img_list]
    long_img = stack_images_horizontally(ims, captions, img_id)
    os.makedirs(op.dirname(out_p), exist_ok=True)
    long_img.save(out_p)
    for im, caption in zip(imgs, captions):
        img = Image.fromarray(im)
        my_out_p = out_p.replace('combine', caption)
        os.makedirs(op.dirname(my_out_p), exist_ok=True)
        img.save(my_out_p)

    return long_img
