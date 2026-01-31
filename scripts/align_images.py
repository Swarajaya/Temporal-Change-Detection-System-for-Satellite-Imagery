import rasterio
import numpy as np
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/align_images.py <year>")
    sys.exit(1)

year = sys.argv[1]   # 2019 / 2020 / 2021
base_path = f"data/raw/{year}"

input_path = os.path.join(base_path, "image.tif")
output_path = os.path.join(base_path, "image_aligned.tif")

with rasterio.open(input_path) as src:
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",          # ✅ FORCE GEOTIFF
        dtype=rasterio.float32   # ✅ SAFE FOR ALIGNMENT
    )

    data = src.read().astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)

print(f"Aligned image created for {year}: image_aligned.tif")
