import rasterio
import numpy as np
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python scripts/compute_change.py <year1> <year2>")
    sys.exit(1)

year1 = sys.argv[1]
year2 = sys.argv[2]

img1_path = f"data/raw/{year1}/image_aligned.tif"
img2_path = f"data/raw/{year2}/image_aligned.tif"

out_dir = "data/processed"
os.makedirs(out_dir, exist_ok=True)

out_path = f"{out_dir}/change_{year1}_{year2}.tif"

with rasterio.open(img1_path) as src1, rasterio.open(img2_path) as src2:
    img1 = src1.read().astype(np.float32)
    img2 = src2.read().astype(np.float32)

    # Absolute difference per band
    diff = np.abs(img2 - img1)

    # Mean change across RGB bands
    change_map = np.mean(diff, axis=0)

    profile = src1.profile
    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(change_map, 1)

print(f"Change map created: {out_path}")
