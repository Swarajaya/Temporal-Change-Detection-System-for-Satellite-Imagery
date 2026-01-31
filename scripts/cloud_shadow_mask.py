import sys
import os
import numpy as np
import rasterio

def adaptive_cloud_shadow_mask(input_tif, output_tif):
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    with rasterio.open(input_tif) as src:
        img = src.read().astype(np.float32)
        meta = src.meta.copy()

    # Normalize to [0,1] safely
    img_norm = img / 255.0

    # Brightness map
    brightness = img_norm.mean(axis=0)

    # Adaptive thresholds (percentiles)
    cloud_thresh = np.percentile(brightness, 97)   # top 3% bright
    shadow_thresh = np.percentile(brightness, 3)   # bottom 3% dark

    cloud_mask = brightness >= cloud_thresh
    shadow_mask = brightness <= shadow_thresh

    invalid_mask = cloud_mask | shadow_mask

    # Apply mask
    img_norm[:, invalid_mask] = 0

    cleaned = (img_norm * 255).astype(np.uint8)

    meta.update(dtype="uint8")

    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(cleaned)

    print("[✓] Adaptive cloud-shadow masking done")
    print(f"    Cloud threshold  : {cloud_thresh:.3f}")
    print(f"    Shadow threshold : {shadow_thresh:.3f}")
    print(f"    Saved → {output_tif}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cloud_shadow_mask.py <input_norm> <output_clean>")
        sys.exit(1)

    adaptive_cloud_shadow_mask(sys.argv[1], sys.argv[2])
