import sys
import numpy as np
import rasterio
from skimage.transform import resize

def read_raster(path):
    with rasterio.open(path) as src:
        img = src.read(1)
        profile = src.profile
    return img, profile

def save_raster(path, img, profile):
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(img.astype(np.uint8), 1)

def multiscale_validate(change_map):
    # Original scale
    scale1 = change_map

    # Downscale (50%)
    small = resize(
        change_map,
        (change_map.shape[0] // 2, change_map.shape[1] // 2),
        preserve_range=True,
        anti_aliasing=True
    )

    # Upscale back
    scale2 = resize(
        small,
        change_map.shape,
        preserve_range=True,
        anti_aliasing=True
    )

    # Binary consistency check
    final = np.logical_and(scale1 > 0, scale2 > 0)

    return final.astype(np.uint8) * 255

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    change, profile = read_raster(input_path)
    validated = multiscale_validate(change)

    save_raster(output_path, validated, profile)
    print("[✓] Multi-scale validated change map saved:", output_path)
