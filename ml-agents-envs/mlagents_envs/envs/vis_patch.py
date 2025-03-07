import matplotlib.pyplot as plt
import numpy as np
from patchify import patchify, unpatchify
from PIL import Image


def get_patchified_mask(water_percentage_threshold: float = 0) -> np.ndarray:
    # Patchify this mask to shape (patch_row_num, patch_col_num, patch_size_x, patch_size_y)
    patchified_mask = patchify(mask_arr, (patch_dim_x, patch_dim_y), step=patch_step)

    # Calculate the percentage of 255 (water) pixels in each (patch_size_x, patch_size_y) patch
    percentage_255 = np.mean(patchified_mask == 255, axis=(2, 3))
    print(f'{percentage_255.shape=}')

    # Create a 2D binary array where each element is 255 if the percentage exceeds 50%, otherwise 0
    condensed_mask = np.where(percentage_255 > water_percentage_threshold, 255, 0)
    print(f'{condensed_mask=}')

    return condensed_mask


if __name__ == '__main__':
    """
    Visualize patchification of a binary mask.
    """

    mask_path = 'sim_images/masks/0000.png'
    mask = Image.open(mask_path)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    mask_arr = mask_arr[..., 0]
    print(f'{mask_arr.shape=}')

    # patch_dim_x = 8
    # patch_dim_y = 8
    # patch_step = 8

    patch_dim_x = 16
    patch_dim_y = 16
    patch_step = 16

    patchified_mask = get_patchified_mask()

    patches = patchify(mask_arr, (patch_dim_x, patch_dim_y), step=patch_step)
    patches_clone = np.copy(patches)
    print(f'{patches.shape=}')
    patch_row, patch_col, pixel_row, pixel_col = patches.shape

    gap_dim = 4
    binary_patches = np.zeros((mask_arr.shape[0] + gap_dim * (patch_row - 1),
                               mask_arr.shape[1] + gap_dim * (patch_col - 1)))
    print(f'{binary_patches.shape=}')
    for pi in range(patch_row):
        for pj in range(patch_col):
            if np.any(patches[pi, pj]):
                patches_clone[pi, pj, ...] = 1

    reconstructed_mask = unpatchify(patches_clone, mask_arr.shape)
    print(f'{reconstructed_mask.shape=}')

    inflated_recon_mask = np.insert(reconstructed_mask, np.arange(pixel_row, mask_arr.shape[0], pixel_row), 1, axis=0)
    inflated_recon_mask = np.insert(inflated_recon_mask, np.arange(pixel_col, mask_arr.shape[1], pixel_col), 1, axis=1)

    plt.imshow(inflated_recon_mask, cmap='gray')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
