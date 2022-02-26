import glob
import os

import kornia.augmentation as K
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_segmentation_masks

from csip.datasets.augmentations import default_augs
from csip.datasets.utils import dataset_split


class OSCD(Dataset):
    image_root = "Onera Satellite Change Detection dataset - Images"
    target_root = "Onera Satellite Change Detection dataset - {split} Labels"
    classes = ["no-change", "change"]

    def __init__(self, root="data", split="train", transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.files = self._load_files()

    def __getitem__(self, idx):
        files = self.files[idx]
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
        mask = self._load_target(files["mask"])

        image = torch.stack(tensors=[image1, image2], dim=0)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_files(self):
        image_root = os.path.join(self.root, self.image_root)
        target_root = os.path.join(
            self.root, self.target_root.format(split=self.split.capitalize())
        )
        folders = [
            folder.split(os.sep)[-2]
            for folder in glob.glob(os.path.join(target_root, "*/"))
        ]

        files = []
        for folder in sorted(folders):
            mask = os.path.join(target_root, folder, "cm", "cm.png")
            image1 = os.path.join(image_root, folder, "pair", "img1.png")
            image2 = os.path.join(image_root, folder, "pair", "img2.png")
            files.append(dict(image1=image1, image2=image2, mask=mask))

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
            tensor = torch.clip(tensor, min=0, max=1)
            tensor = tensor.to(torch.long)
            return tensor


class OSCDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size=2,
        num_workers=0,
        num_prefetch=2,
        val_split_pct=0.2,
        patch_size=512,
        augmentations=default_augs(),
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_prefetch = num_prefetch
        self.val_split_pct = val_split_pct
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.random_crop = K.AugmentationSequential(
            K.RandomCrop(
                (self.patch_size, self.patch_size),
                p=1.0,
                pad_if_needed=True,
                return_transform=False,
            ),
            data_keys=["input", "mask"],
        )

    def preprocess(self, sample):
        sample["image"] = sample["image"] / 255
        sample["image"] = torch.clamp(sample["image"], min=0.0, max=1.0)
        sample["image"] = rearrange(sample["image"], "t c h w -> (t c) h w")
        sample["mask"] = rearrange(sample["mask"], "h w -> () h w")
        return sample

    def crop(self, sample):
        sample["mask"] = sample["mask"].to(torch.float)
        sample["image"], sample["mask"] = self.random_crop(
            sample["image"], sample["mask"]
        )
        sample["mask"] = sample["mask"].to(torch.long)
        sample["image"] = rearrange(sample["image"], "() c h w -> c h w")
        sample["mask"] = rearrange(sample["mask"], "() c h w -> c h w")
        return sample

    def setup(self, stage=None):
        transforms = T.Compose([self.preprocess, self.crop])
        test_transforms = T.Compose([self.preprocess])

        dataset = OSCD(self.root, split="train", transforms=transforms)
        self.train_dataset, self.val_dataset, _ = dataset_split(
            dataset, val_pct=self.val_split_pct, test_pct=0.0
        )
        self.test_dataset = OSCD(self.root, split="test", transforms=test_transforms)

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
            batch_size=1,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
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
        batch["image"] = (batch["image"][:, :3, ...], batch["image"][:, 3:, ...])
        return batch

    def plot(self, x, y):
        x = (x.cpu() * 255).to(torch.uint8)
        y = y.cpu().to(torch.bool)
        image = draw_segmentation_masks(x, y, alpha=0.5, colors="red")
        return image
