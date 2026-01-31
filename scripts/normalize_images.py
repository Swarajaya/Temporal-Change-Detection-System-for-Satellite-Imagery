import rasterio
import numpy as np
import os

def normalize_image(input_path, output_path):
    with rasterio.open(input_path) as src:
        img = src.read().astype(np.float32)
        meta = src.meta.copy()

    # Min-Max normalization per band
    for b in range(img.shape[0]):
        band = img[b]
        min_val = np.percentile(band, 2)
        max_val = np.percentile(band, 98)

        band = np.clip(band, min_val, max_val)
        img[b] = (band - min_val) / (max_val - min_val + 1e-6)

    # Scale to uint8
    img = (img * 255).astype(np.uint8)

    meta.update(dtype="uint8")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(img)

    print(f"[✓] Normalized image saved to: {output_path}")


if __name__ == "__main__":
    inputs = {
        "2019": "data/raw/2019/image_aligned.tif",
        "2020": "data/raw/2020/image_aligned.tif",
        "2021": "data/raw/2021/image_aligned.tif"
    }

    for year, in_path in inputs.items():
        out_path = f"data/processed/normalized/{year}_norm.tif"
        normalize_image(in_path, out_path)
