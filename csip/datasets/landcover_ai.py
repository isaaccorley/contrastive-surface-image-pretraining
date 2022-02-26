import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_segmentation_masks

from csip.datasets.augmentations import default_augs


class LandCoverAI(Dataset):
    classes = ["background", "building", "woodland", "water", "road"]
    colormap = ["#a3ff72", "#9c9c9c", "#267200", "#00c5ff", "#000000"]

    def __init__(self, root="data", split="train", transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files()

    def __getitem__(self, idx):
        files = self.files[idx]
        image = self._load_image(files["image"])
        mask = self._load_target(files["mask"])

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_files(self):
        image_root = os.path.join(self.root, "output")
        split_path = os.path.join(self.root, f"{self.split}.txt")
        with open(split_path) as f:
            filenames = f.read().strip().splitlines()

        files = []
        for filename in sorted(filenames):
            image = os.path.join(image_root, f"{filename}.jpg")
            mask = os.path.join(image_root, f"{filename}_m.png")
            files.append(dict(image=image, mask=mask))
        return files

    def _load_image(self, path):
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            tensor = tensor.permute((2, 0, 1))
            tensor = tensor.to(torch.float)
            return tensor

    def _load_target(self, path):
        with Image.open(path) as img:
            array = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor


class LandCoverAIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size=2,
        num_workers=0,
        num_prefetch=2,
        augmentations=default_augs(),
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_prefetch = num_prefetch
        self.augmentations = augmentations

    def preprocess(self, sample):
        sample["image"] = sample["image"] / 255
        sample["image"] = torch.clamp(sample["image"], min=0.0, max=1.0)
        sample["mask"] = rearrange(sample["mask"], "h w -> () h w")
        return sample

    def setup(self, stage=None):
        transforms = T.Compose([self.preprocess])

        self.train_dataset = LandCoverAI(
            self.root, split="train", transforms=transforms
        )
        self.val_dataset = LandCoverAI(self.root, split="val", transforms=transforms)
        self.test_dataset = LandCoverAI(self.root, split="test", transforms=transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)
        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(self, x, y):
        x = (x.cpu() * 255).to(torch.uint8)
        y = y.cpu().unsqueeze(dim=0)
        classes = torch.tensor([1, 2, 3, 4])
        class_masks = y == classes[:, None, None]
        image = draw_segmentation_masks(
            x, class_masks, alpha=0.5, colors=self.train_dataset.colormap
        )
        return image
