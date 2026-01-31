import sys
import os
import numpy as np
import rasterio

def threshold_change(input_tif, output_tif, threshold=0.3):
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    with rasterio.open(input_tif) as src:
        change_map = src.read(1)
        meta = src.meta.copy()

    binary_change = (change_map > threshold).astype(np.uint8)

    meta.update({
        "dtype": "uint8",
        "count": 1
    })

    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(binary_change, 1)

    print(f"[✓] Binary change map saved to: {output_tif}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python threshold_change.py <input_tif> <output_tif>")
        sys.exit(1)

    input_tif = sys.argv[1]
    output_tif = sys.argv[2]

    threshold_change(input_tif, output_tif)
