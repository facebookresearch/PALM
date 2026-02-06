import torch
import numpy as np
import os
from sklearn.decomposition import PCA

def pca_decomposition(sd, num_components=5, save_path='pca_5.npy', keyword='shape_code'):
    """
    Perform PCA decomposition on shape codes extracted from the input dictionary.

    Parameters:
    - sd: dict
        A dictionary containing shape codes.
    - num_components: int, optional
        The number of principal components to keep. Default is 5.
    - save_path: str, optional
        The file path to save the principal axes. Default is 'pca_5.npy'.

    Returns:
    - principal_components: np.ndarray
        The transformed data after PCA.
    - explained_variance_ratio: np.ndarray
        The amount of variance explained by each of the selected components.
    """
    # Extract shape codes from the dictionary
    shape_codes = [val for key, val in sd.items() if keyword in key]
    shape_codes = torch.cat(shape_codes)

    # Convert to numpy array
    latent_codes = np.array(shape_codes)

    # Initialize and fit PCA
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(latent_codes)

    # Print explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio:", explained_variance_ratio)

    # Ensure the directory for the save path exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save principal axes
    principal_axes = pca.components_
    np.save(save_path, principal_axes)
    print('Saved PCA at: ', save_path)

    return principal_components, explained_variance_ratio


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_p", required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    import os.path as op
    sd_p = args.sd_p
    exp_id = sd_p.split('/')[1]
    basename = op.basename(sd_p).replace('.ckpt', '.shape.pca.npy')
    save_p = f'./pca_basis/{exp_id}/{basename}'
    sd = torch.load(sd_p, map_location='cpu')['state_dict']
    pca_decomposition(sd, num_components=5, save_path=save_p, keyword='shape_code')

    basename = op.basename(sd_p).replace('.ckpt', '.appearance.pca.npy')
    save_p = f'./pca_basis/{exp_id}/{basename}'
    sd = torch.load(sd_p, map_location='cpu')['state_dict']
    pca_decomposition(sd, num_components=5, save_path=save_p, keyword='appearance_code')
    

