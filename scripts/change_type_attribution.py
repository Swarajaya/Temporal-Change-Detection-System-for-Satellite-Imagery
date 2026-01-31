import sys
import numpy as np
import rasterio
import os

def load_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile

def save_tif(path, data, profile):
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8), 1)

def classify_change(persistent, multiscale, confidence):
    output = np.zeros(persistent.shape, dtype=np.uint8)

    # Thresholds (can be tuned later)
    high_conf = confidence > 0.7
    mid_conf  = (confidence > 0.4) & (confidence <= 0.7)

    sudden = (multiscale > 0.6) & (persistent == 1)
    gradual = (multiscale > 0.3) & (multiscale <= 0.6)
    permanent = persistent == 1

    output[gradual & mid_conf] = 1      # Gradual change
    output[sudden & mid_conf] = 2       # Sudden change
    output[permanent & high_conf] = 4   # Permanent high-confidence change
    output[mid_conf & ~permanent] = 3   # Seasonal / uncertain

    return output

if __name__ == "__main__":
    persistent_path = sys.argv[1]
    multiscale_path = sys.argv[2]
    confidence_path = sys.argv[3]
    output_path = sys.argv[4]

    persistent, profile = load_tif(persistent_path)
    multiscale, _ = load_tif(multiscale_path)
    confidence, _ = load_tif(confidence_path)

    change_type = classify_change(persistent, multiscale, confidence)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_tif(output_path, change_type, profile)

    print("[✓] Change type attribution completed")
    print("Legend:")
    print("0 = No change")
    print("1 = Gradual change")
    print("2 = Sudden change")
    print("3 = Seasonal / uncertain")
    print("4 = Permanent high-confidence change")
