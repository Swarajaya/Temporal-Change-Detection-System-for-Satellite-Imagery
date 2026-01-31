import os
import torch
import rasterio
from torch.utils.data import Dataset

class ChangeDataset(Dataset):
    def __init__(self, root_dir):
        self.t1_dir = os.path.join(root_dir, "t1")
        self.t2_dir = os.path.join(root_dir, "t2")
        self.mask_dir = os.path.join(root_dir, "masks")

        self.files = sorted(os.listdir(self.t1_dir))

    def __len__(self):
        return len(self.files)

    def read_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype("float32")
            return torch.tensor(img)

    def __getitem__(self, idx):
        fname = self.files[idx]

        t1 = self.read_tif(os.path.join(self.t1_dir, fname))
        t2 = self.read_tif(os.path.join(self.t2_dir, fname))
        mask = self.read_tif(os.path.join(self.mask_dir, fname))

        return t1, t2, mask
