import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import os
import sys

def spatial_filter(input_tif, output_tif, min_area=50):
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        profile = src.profile

    binary = (data > 0).astype(np.uint8)

    labeled = label(binary, connectivity=2)
    filtered = np.zeros_like(binary)

    for region in regionprops(labeled):
        if region.area >= min_area:
            for (r, c) in region.coords:
                filtered[r, c] = 1

    profile.update(dtype=rasterio.uint8, count=1)

    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(filtered, 1)

    return filtered


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    filtered = spatial_filter(input_path, output_path)

    # Save visualization
    fig_dir = "outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(6,6))
    plt.imshow(filtered, cmap="hot")
    plt.title("Spatially Filtered Persistent Change")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(os.path.join(fig_dir, "persistent_change_filtered.png"), dpi=300)
    plt.close()

    print("Spatial filtering completed.")
