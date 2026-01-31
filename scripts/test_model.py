import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Dummy Siamese UNet (sanity model)
# --------------------------------
class SiameseUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # [B, 6, H, W]
        return self.encoder(x)

# --------------------------------
# Create output folders
# --------------------------------
os.makedirs("outputs/model_outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# --------------------------------
# Load model
# --------------------------------
device = "cpu"
model = SiameseUNet().to(device)
model.eval()

# --------------------------------
# Dummy input (replace later with real patches)
# --------------------------------
t1 = torch.rand(1, 3, 256, 256).to(device)
t2 = torch.rand(1, 3, 256, 256).to(device)

# --------------------------------
# Forward pass
# --------------------------------
with torch.no_grad():
    output = model(t1, t2)

# --------------------------------
# Convert to numpy
# --------------------------------
output_np = output.squeeze().cpu().numpy()

# --------------------------------
# Save raw output (.npy)
# --------------------------------
np.save("outputs/model_outputs/change_prediction.npy", output_np)

# --------------------------------
# Save visualization (.png)
# --------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(output_np, cmap="hot")
plt.colorbar(label="Predicted Change Probability")
plt.title("Model Output (Sanity Check)")
plt.axis("off")

plt.savefig(
    "outputs/figures/model_output_sanity.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------
# Print stats (VERY IMPORTANT)
# --------------------------------
print("[✓] Model test completed")
print("Output shape:", output.shape)
print("Min:", output_np.min())
print("Max:", output_np.max())
print("Mean:", output_np.mean())

print("[✓] Saved:")
print(" - outputs/model_outputs/change_prediction.npy")
print(" - outputs/figures/model_output_sanity.png")
