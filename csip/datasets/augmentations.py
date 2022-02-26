import kornia.augmentation as K
import torch.nn as nn
from einops.layers.torch import Rearrange
from kornia.contrib import ExtractTensorPatches


def default_augs():
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        data_keys=["input", "mask"],
    )


def default_ssl_augmentations(input_shape):
    kernel_size = [int(input_shape * 0.1), int(input_shape * 0.1)]
    if kernel_size[0] % 2 == 0:
        kernel_size[0] += 1
    if kernel_size[1] % 2 == 0:
        kernel_size[1] += 1
    return nn.Sequential(
        ExtractTensorPatches(window_size=input_shape, stride=input_shape),
        Rearrange("b t c h w -> (b t) c h w"),
        K.RandomResizedCrop(
            size=(input_shape, input_shape),
            scale=(0.5, 1.0),
            resample="bilinear",
            p=0.5,
        ),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(degrees=(0, 90), p=0.5),
        K.RandomGaussianBlur(kernel_size=tuple(kernel_size), sigma=(0.1, 2.0), p=0.2),
    )


def default_ssl_rgb_augmentations(strength):
    return nn.Sequential(
        K.ColorJitter(
            brightness=0.8 * strength,
            contrast=0.8 * strength,
            saturation=0.8 * strength,
            hue=0.2 * strength,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
    )
