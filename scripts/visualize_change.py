import os
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Usage check
# ----------------------------
if len(sys.argv) != 2:
    print("Usage: python scripts/visualize_change.py <path_to_tif>")
    sys.exit(1)

path = sys.argv[1]

# ----------------------------
# Read raster
# ----------------------------
with rasterio.open(path) as src:
    img = src.read(1)

# Handle NaNs safely
img = np.nan_to_num(img)

# ----------------------------
# Create output directory
# ----------------------------
os.makedirs("outputs/figures", exist_ok=True)

# Output file name
out_name = os.path.basename(path).replace(".tif", ".png")
out_path = os.path.join("outputs", "figures", out_name)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="hot")
plt.colorbar(label="Change Intensity")
plt.title(f"Change Map: {os.path.basename(path)}")
plt.axis("off")

# ----------------------------
# Save & close
# ----------------------------
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[✓] Visualization saved to: {out_path}")
