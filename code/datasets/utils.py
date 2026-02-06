from PIL import Image
import numpy as np
import random
import cv2

class KeyIndex:
    def __init__(self, values, key=None):
        # If 'values' is a list of dictionaries, store it directly. Otherwise, create a list of
        # dictionaries from the values and key.
        if all(isinstance(v, dict) for v in values):
            self.data = values
        else:
            self.data = [{key: value} for value in values]

    def __mul__(self, other):
        if not isinstance(other, KeyIndex):
            raise ValueError("Operand must be an instance of KeyIndex")

        result = []
        for dict1 in self.data:
            for dict2 in other.data:
                merged_dict = {**dict1, **dict2}  # Merge dictionaries
                result.append(merged_dict)
        return KeyIndex(result)

    def __add__(self, other):
        if not isinstance(other, KeyIndex) or len(self.data) != len(other.data):
            raise ValueError(
                "Operands must be instances of KeyIndex with equal length data"
            )

        result = []
        for dict1, dict2 in zip(self.data, other.data):
            merged_dict = {**dict1, **dict2}  # Merge dictionaries
            result.append(merged_dict)
        return KeyIndex(result)

    def to_list(self):
        return self.data


def parse_custom_fids(config, end_fid):
    """
    Parse config string into a list of integers.
    Args:
        config (str): A string representing ranges of numbers, e.g., ':3,21:29,64,194:'.
        end_fid (int): The end frame ID.
    Returns:
        list: A list of integers.
    """
    if config == "":
        return []
    ranges = config.split(",")
    result = []
    for rng in ranges:
        if ":" not in rng:
            result.append(int(rng))
        else:
            parts = rng.split(":")
            if len(parts[0]) == 0:
                parts[0] = 0
            if len(parts[1]) == 0:
                parts[1] = end_fid

            start = int(parts[0])
            end = int(parts[1])
            result.extend(range(start, end + 1))

    return sorted(list(result))


def sample_patch_from_mask(img, msk, patch_size, margin=32):
    """
    Samples a patch of size patch_size by patch_size from the given image and mask.
    The patch is sampled from a cropped region around the mask with a specified margin.
    Args:
        img (np.ndarray): The input image.
        msk (np.ndarray): The input mask.
        patch_size (int): The size of the patch.
        margin (int, optional): The margin to add around the cropped mask. Defaults to 16.
    Returns:
        tuple: A tuple containing the sampled patch of the image, mask, and other arrays,
               along with their corresponding coordinates.
    """

    # Find the indices of non-zero elements in the mask
    rows = np.any(msk, axis=1)
    cols = np.any(msk, axis=0)

    # Get the indices of the first and last non-zero elements
    row_start = np.argmax(rows)
    row_end = len(rows) - 1 - np.argmax(rows[::-1])
    col_start = np.argmax(cols)
    col_end = len(cols) - 1 - np.argmax(cols[::-1])

    # Add the margin to the indices
    row_start = max(0, row_start - margin)
    row_end = min(len(rows), row_end + margin + 1)
    col_start = max(0, col_start - margin)
    col_end = min(len(cols), col_end + margin + 1)

    # Crop the image, mask, and other arrays
    img_cropped = img[row_start:row_end, col_start:col_end]
    msk_cropped = msk[row_start:row_end, col_start:col_end]

    # Check if the patch size is valid
    if patch_size > img_cropped.shape[0] or patch_size > img_cropped.shape[1]:
        raise ValueError("Patch size cannot be larger than the cropped image")

    # Generate random coordinates for the top-left corner of the patch
    x = np.random.randint(0, img_cropped.shape[1] - patch_size + 1)
    y = np.random.randint(0, img_cropped.shape[0] - patch_size + 1)

    # Sample the patch
    img_patch = img_cropped[y : y + patch_size, x : x + patch_size]
    msk_patch = msk_cropped[y : y + patch_size, x : x + patch_size]

    return img_patch, msk_patch, x, y, row_start, row_end, col_start, col_end


def sample_and_downsample_patches(img, msk, normal, msk_dilate, xy_all, patch_size, k):
    """
    Samples a patch from the given image and mask, and then downsamples it by a factor of k.
    Args:
        img (np.ndarray): The input image.
        msk (np.ndarray): The input mask.
        normal (np.ndarray): The input normal map.
        rays_o (np.ndarray): The input ray origins.
        rays_d (np.ndarray): The input ray directions.
        patch_size (int, optional): The size of the patch. Defaults to 512.
        k (int, optional): The downsampling factor. Defaults to 4.
    Returns:
        tuple: A tuple containing the downsampled patches of the image, mask, normal map, ray origins, and ray directions.
    """

    # Sample patches from the image and mask
    img_patch, msk_patch, x, y, row_start, row_end, col_start, col_end = (
        sample_patch_from_mask(img, msk, patch_size=patch_size)
    )

    # Sample patches from other arrays using the same coordinates
    normal_patch = normal[
        row_start + y : row_start + y + patch_size,
        col_start + x : col_start + x + patch_size,
    ]
    dilate_patch = msk_dilate[
        row_start + y : row_start + y + patch_size,
        col_start + x : col_start + x + patch_size,
    ]
    xy_patch = xy_all[
        row_start + y : row_start + y + patch_size,
        col_start + x : col_start + x + patch_size,
    ]

    # Downsample the patches
    img_patch = img_patch[::k, ::k, :]
    msk_patch = msk_patch[::k, ::k]
    
    normal_patch = normal_patch[::k, ::k, :]
    dilate_patch = dilate_patch[::k, ::k]
    xy_patch = xy_patch[::k, ::k, :]

    return img_patch, msk_patch, normal_patch, dilate_patch, xy_patch


random.seed(778)


# def load_mask(mask_p):
#     im = Image.open(mask_p)
#     mask_np = (np.array(im) / 255.0).astype(np.uint8)
#     return mask_np


def load_mask(mask_p):
    mask_np = (np.array(Image.open(mask_p)) / 255.0).astype(np.uint8)

    mano_p = mask_p.replace('/masks/', '/mano/')
    mano = np.array(Image.open(mano_p))
    if len(mano.shape) == 3:
        mano = mano[:, :, 3]
    mano_np =(mano/255.0).astype(np.uint8)

    ksize = 21
    kernel = np.ones((ksize, ksize), np.uint8)
    mano_corrode = cv2.erode(mano_np, kernel, iterations=2)
    ksize = 11
    kernel = np.ones((ksize, ksize), np.uint8)
    mano_dilate = cv2.dilate(mano_np, kernel, iterations=2)

    final_np = mask_np.copy()
    final_np = final_np * mano_dilate
    final_np = ((final_np + mano_corrode) > 0).astype(np.uint8)

    final_np = cv2.medianBlur(final_np, 7)

    kernel = np.ones((5, 5), np.uint8)
    mano_dilate = cv2.dilate(final_np, kernel, iterations=2)
    return final_np, mano_dilate

def load_image(rgb_p):
    img = np.array(Image.open(rgb_p))
    return img


def encode_depth(depth):
    z_max = 1.8
    MAX_VAL_16bits = 2**16 - 1
    depth[depth < 1e-6] = np.nan
    depth_normalized = (depth / z_max * MAX_VAL_16bits).astype(
        np.uint16
    )  # Scale to 16-bit
    return depth_normalized


def decode_depth(depth_normalized):
    Z_MAX = 1.8
    MAX_VAL_16bits = 2**16 - 1
    # Handle NaN values that were set to zero

    depth_normalized = depth_normalized.astype(
        np.float32
    )  # Convert to float for NaN handling
    depth_normalized[depth_normalized == 0] = np.nan
    # Reverse the normalization

    depth_decode = depth_normalized / MAX_VAL_16bits * Z_MAX
    return depth_decode


def load_normal(normal_p):
    im = Image.open(normal_p)
    normal_np = np.array(im) * 1.0
    normal_np[normal_np == 0.0] = np.nan
    normal_np = normal_np / 255.0
    normal_np = normal_np[:, :, :3]
    # normal_np = 2*normal_np - 1.0
    return normal_np

def load_depth(depth_p):
    depth_im = np.array(Image.open(depth_p))
    depth_val = decode_depth(depth_im) # meters
    return depth_val

def load_cameras(cameras):
    K_dict = {}
    c2w_dict = {}
    height_dict = {}
    width_dict = {}
    for cam_id, cam_dict in cameras.items():
        K_dict[cam_id] = cam_dict["K"]
        c2w_dict[cam_id] = np.linalg.inv(cam_dict["Rt"])
        height_dict[cam_id] = cam_dict["height"]
        width_dict[cam_id] = cam_dict["width"]
    return K_dict, c2w_dict, height_dict, width_dict


def _rgb_to_srgb(f: np.ndarray) -> np.ndarray:
    return np.where(
        f <= 0.0031308,
        f * 12.92,
        np.power(np.clip(f, 0.0031308, None), 1.0 / 2.4) * 1.055 - 0.055,
    )


def get_ray_directions(H, W):
    x, y = np.meshgrid(np.arange(W)+0.5, np.arange(H)+0.5, indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1)
    return xy


def make_rays(K, c2w, xy):
    # xy = get_ray_directions(H, W).reshape(-1, 3).astype(np.float32) ## cache
    d_c = xy @ np.linalg.inv(K).T
    d_w = d_c @ c2w[:3, :3].T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    o_w = np.tile(c2w[:3, 3], (len(d_w), 1))
    o_w = o_w.reshape(-1, 3)
    d_w = d_w.reshape(-1, 3)
    return o_w.astype(np.float32), d_w.astype(np.float32)


def load_mano_param(path):
    mano_params = np.load(path, allow_pickle=True).item()
    return {
        "betas": mano_params["betas"][:1].astype(np.float32).reshape(1, 10),
        "body_pose": mano_params["hand_pose"].astype(np.float32),
        "global_orient": mano_params["global_orient"].astype(np.float32),
        "transl": mano_params["transl"].astype(np.float32),
        # "fitting_error": mano_params["fitting_error"].astype(np.float32),
    }
