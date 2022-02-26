import glob
import json
import os

import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from csip.datasets.augmentations import (
    default_ssl_augmentations,
    default_ssl_rgb_augmentations,
)


class OverheadGeopose(Dataset):
    def __init__(self, root="data", transforms=None):
        self.root = root
        self.transforms = transforms
        self.files = self._load_files()

    def _load_files(self):
        images = sorted(glob.glob(os.path.join(self.root, "*RGB.j2k")))
        agls = sorted(glob.glob(os.path.join(self.root, "*AGL.tif")))
        vflows = sorted(glob.glob(os.path.join(self.root, "*VFLOW.json")))
        files = []
        for image, agl, vflow in zip(images, agls, vflows):
            files.append(dict(image=image, agl=agl, vflow=vflow))
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        files = self.files[idx]
        rgb = self._load_image(files["image"])
        agl = self._load_agl(files["agl"])
        image = torch.cat([rgb, agl], dim=0)
        scale, angle = self._load_target(files["vflow"])

        sample = {"image": image, "scale": scale, "angle": angle}
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path):
        x = np.array(Image.open(path))
        x = torch.from_numpy(x).to(torch.float)
        x = x.permute(2, 0, 1)
        return x

    def _load_agl(self, path):
        x = tifffile.imread(path).astype(np.float32)
        x = torch.from_numpy(x)
        x = x.unsqueeze(dim=0)
        return x

    def _load_target(self, path):
        with open(path) as f:
            vflow = json.load(f)

        scale = torch.tensor(vflow["scale"])
        angle = torch.tensor(vflow["angle"])
        return scale, angle


class OverheadGeoposeSSLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size=2,
        num_workers=0,
        num_prefetch=2,
        patch_size=512,
        augmentations=None,
        rgb_augmentations=None,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_prefetch = num_prefetch
        self.patch_size = patch_size

        if augmentations is None:
            self.augmentations = default_ssl_augmentations(patch_size)
        if rgb_augmentations is None:
            self.rgb_augmentations = default_ssl_rgb_augmentations(strength=0.01)

    def preprocess(self, sample):
        del sample["scale"]
        del sample["angle"]
        sample["image"][:3] = sample["image"][:3] / 255.0
        sample["image"][-1] = sample["image"][-1] / 65535.0
        return sample

    def setup(self, stage=None):
        transforms = T.Compose([self.preprocess])
        self.dataset = OverheadGeopose(self.root, transforms=transforms)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=True,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["image"] = self.augmentations(batch["image"])
        rgb = batch["image"][:, :3, ...]
        agl = batch["image"][:, -1, ...].unsqueeze(dim=1)
        rgb = self.rgb_augmentations(rgb)
        batch = dict(x_a=rgb, x_b=agl)
        return batch
