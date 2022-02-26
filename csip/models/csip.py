"""Inspired by OpenAI's CLIP https://github.com/openai/CLIP."""
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXent(nn.Module):
    def forward(self, z1, z2, t):
        batch_size = z1.shape[0]
        device = z1.device
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity = torch.matmul(z1, z2.T)
        similarity = similarity * torch.exp(t)
        targets = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(similarity, targets)
        return loss


class Encoder(nn.Sequential):
    def __init__(self, backbone, num_channels, pretrained=True):
        model = timm.create_model(
            backbone, in_chans=num_channels, num_classes=0, pretrained=pretrained
        )
        self.num_features = model.num_features
        super().__init__(model)


class DualEncoders(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet18",
        proj_dim=256,
        num_channels_a=3,
        num_channels_b=1,
        pretrained=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_a = Encoder(backbone, num_channels_a, pretrained)
        self.encoder_b = Encoder(backbone, num_channels_b, pretrained)
        self.proj_a = nn.Linear(self.encoder_a.num_features, proj_dim)
        self.proj_b = nn.Linear(self.encoder_b.num_features, proj_dim)
        self.t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_fn = NTXent()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def forward(self, x_a, x_b):
        return self.encoder_a(x_a), self.encoder_b(x_b)

    def training_step(self, batch, batch_idx):
        x_a, x_b = batch["x_a"], batch["x_b"]
        e_a, e_b = self(x_a, x_b)
        z_a, z_b = self.proj_a(e_a), self.proj_b(e_b)
        loss = self.loss_fn(z_a, z_b, self.t)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def embed_a(self, x):
        return F.normalize(self.encoder_a(x), dim=-1)

    @torch.no_grad()
    def embed_b(self, x):
        return F.normalize(self.encoder_b(x), dim=-1)
