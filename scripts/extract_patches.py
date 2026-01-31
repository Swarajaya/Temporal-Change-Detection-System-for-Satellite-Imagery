import rasterio
import numpy as np
import os
from tqdm import tqdm

PATCH_SIZE = 256
STRIDE = 128

def extract(image_path, out_dir, prefix):
    with rasterio.open(image_path) as src:
        img = src.read()
        h, w = img.shape[1], img.shape[2]

        count = 0
        for y in range(0, h - PATCH_SIZE, STRIDE):
            for x in range(0, w - PATCH_SIZE, STRIDE):
                patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                if patch.shape[1] != PATCH_SIZE or patch.shape[2] != PATCH_SIZE:
                    continue

                out_path = os.path.join(out_dir, f"{prefix}_{count}.npy")
                np.save(out_path, patch)
                count += 1

        print(f"{count} patches saved from {image_path}")

# ---- CHANGE THESE PATHS ----
pairs = [
    ("2019", "2020", "train"),
    ("2020", "2021", "val")
]

for t1, t2, split in pairs:
    extract(f"data/raw/{t1}/image_aligned.tif", f"data/dl/{split}/t1", f"{t1}_{t2}")
    extract(f"data/raw/{t2}/image_aligned.tif", f"data/dl/{split}/t2", f"{t1}_{t2}")
    extract(f"data/processed/change_{t1}_{t2}_binary.tif", f"data/dl/{split}/masks", f"{t1}_{t2}")
