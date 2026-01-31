import os
import rasterio
import numpy as np

# Paths
c1_path = "outputs/binary/change_2019_2020_binary.tif"
c2_path = "outputs/binary/change_2020_2021_binary.tif"

out_dir = "outputs/temporal"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "persistent_change_2019_2021.tif")

# Load binary maps
with rasterio.open(c1_path) as src1:
    c1 = src1.read(1)
    meta = src1.meta.copy()

with rasterio.open(c2_path) as src2:
    c2 = src2.read(1)

# Ensure binary
c1 = (c1 > 0).astype(np.uint8)
c2 = (c2 > 0).astype(np.uint8)

# Temporal persistence (AND operation)
persistent = (c1 & c2).astype(np.uint8)

# Save
meta.update(dtype="uint8", count=1)

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(persistent, 1)

print("[✓] Persistent change map created")
print(f"Saved to: {out_path}")
