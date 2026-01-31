import rasterio
import numpy as np
import csv
import os
import sys

change_map_path = sys.argv[1]
confidence_map_path = sys.argv[2]
change_type_path = sys.argv[3]
output_csv = sys.argv[4]

with rasterio.open(change_map_path) as src:
    change = src.read(1)
    transform = src.transform
    pixel_area = abs(transform.a * transform.e)  # m²

with rasterio.open(confidence_map_path) as src:
    confidence = src.read(1)

with rasterio.open(change_type_path) as src:
    change_type = src.read(1)

pixel_area_km2 = pixel_area / 1e6

stats = {}

for ctype in np.unique(change_type):
    if ctype == 0:
        continue  # no change

    mask = (change_type == ctype)
    pixel_count = np.sum(mask)
    area = pixel_count * pixel_area_km2
    avg_conf = np.mean(confidence[mask]) if pixel_count > 0 else 0

    stats[int(ctype)] = {
        "pixels": int(pixel_count),
        "area_km2": float(area),
        "avg_confidence": float(avg_conf)
    }

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Change_Type", "Pixel_Count", "Area_km2", "Avg_Confidence"])

    for k, v in stats.items():
        writer.writerow([k, v["pixels"], round(v["area_km2"], 4), round(v["avg_confidence"], 3)])

print("[✓] Change statistics saved to:", output_csv)
