import rasterio
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def load_binary(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8).flatten()

# Paths (UPDATED)
gt_path = "outputs/binary/change_2019_2020_binary.tif"
pred_path = "outputs/temporal/persistent_change_2019_2021.tif"

# Load data
y_true = load_binary(gt_path)
y_pred = load_binary(pred_path)

# Ensure same size
min_len = min(len(y_true), len(y_pred))
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

# Metrics
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
iou = jaccard_score(y_true, y_pred, zero_division=0)

print("\n=== Evaluation Metrics ===")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"IoU       : {iou:.4f}")
