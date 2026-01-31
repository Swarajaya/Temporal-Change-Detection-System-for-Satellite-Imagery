import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile

def save_tif(path, data, profile):
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python temporal_confidence.py change1.tif change2.tif output.tif")
        sys.exit(1)

    change1_path = sys.argv[1]
    change2_path = sys.argv[2]
    output_path = sys.argv[3]

    c1, profile = load_tif(change1_path)
    c2, _ = load_tif(change2_path)

    # Ensure binary
    c1 = (c1 > 0).astype(np.float32)
    c2 = (c2 > 0).astype(np.float32)

    # Confidence score (0, 0.5, 1.0)
    confidence = (c1 + c2) / 2.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_tif(output_path, confidence, profile)

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Temporal Change Confidence")
    plt.title("Temporal Confidence Map (2019–2021)")
    plt.axis("off")

    fig_path = output_path.replace(".tif", ".png").replace("confidence", "figures")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[✓] Temporal confidence map saved:")
    print(" →", output_path)
