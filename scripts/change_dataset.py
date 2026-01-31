import os
import torch
import rasterio
from torch.utils.data import Dataset

class ChangeDataset(Dataset):
    def __init__(self, root):
        self.t1 = os.path.join(root, "t1")
        self.t2 = os.path.join(root, "t2")
        self.mask = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.t1))

    def __len__(self):
        return len(self.files)

    def read(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype("float32") / 255.0
            return torch.tensor(img)

    def __getitem__(self, idx):
        name = self.files[idx]
        return (
            self.read(os.path.join(self.t1, name)),
            self.read(os.path.join(self.t2, name)),
            self.read(os.path.join(self.mask, name))[0:1],
        )
