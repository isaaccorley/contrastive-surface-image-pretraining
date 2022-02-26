import kornia.augmentation as K
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics

import csip.models.fcsiam


def extract_encoder(path, prefix="encoder_a.0."):
    state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
    state_dict = {
        k.replace(prefix, ""): v for k, v in state_dict.items() if prefix in k
    }
    return state_dict


class FocalDiceLoss(nn.Module):
    def __init__(self, mode="multiclass", ignore_index=None, normalized=True):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(mode=mode, normalized=normalized)
        self.dice_loss = smp.losses.DiceLoss(mode=mode, ignore_index=ignore_index)

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.dice_loss(preds, targets)


class FocalJaccardLoss(nn.Module):
    def __init__(self, num_classes, mode="multiclass", normalized=True):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(mode=mode, normalized=normalized)
        self.jaccard_loss = smp.losses.JaccardLoss(mode=mode, classes=num_classes)

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.jaccard_loss(preds, targets)


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model="Unet",
        backbone="resnet18",
        num_channels=3,
        num_classes=2,
        weights=None,
        learning_rate=1e-3,
        freeze_encoder=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        model = getattr(smp, model)(
            encoder_name=backbone,
            encoder_weights=None if weights != "imagenet" else "imagenet",
            in_channels=num_channels,
            classes=num_classes,
        )
        if weights is not None and weights != "imagenet":
            print(f"loading pretrained backbone from {weights}")
            state_dict = extract_encoder(weights)
            model.encoder.load_state_dict(state_dict)
            if freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = False

        self.model = model
        self.loss_fn = FocalJaccardLoss(
            num_classes=num_classes, mode="multiclass", normalized=True
        )
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "OverallAccuracy": torchmetrics.Accuracy(
                    num_classes=num_classes, average="micro", mdmc_average="global"
                ),
                "OverallPrecision": torchmetrics.Precision(
                    num_classes=num_classes, average="micro", mdmc_average="global"
                ),
                "OverallRecall": torchmetrics.Recall(
                    num_classes=num_classes, average="micro", mdmc_average="global"
                ),
                "OverallF1Score": torchmetrics.FBetaScore(
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                    mdmc_average="global",
                ),
                "AverageAccuracy": torchmetrics.Accuracy(
                    num_classes=num_classes, average="macro", mdmc_average="global"
                ),
                "AveragePrecision": torchmetrics.Precision(
                    num_classes=num_classes, average="macro", mdmc_average="global"
                ),
                "AverageRecall": torchmetrics.Recall(
                    num_classes=num_classes, average="macro", mdmc_average="global"
                ),
                "AverageF1Score": torchmetrics.FBetaScore(
                    num_classes=num_classes,
                    beta=1.0,
                    average="macro",
                    mdmc_average="global",
                ),
                "IoU": torchmetrics.JaccardIndex(
                    num_classes=num_classes, ignore_index=0
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.train_metrics(y_hat_hard, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.val_metrics(y_hat_hard, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False)

        if batch_idx < 5:
            image = self.trainer.datamodule.plot(x[0], y_hat_hard[0])
            self.logger.experiment.add_image(
                "predictions/val", image, global_step=self.global_step + batch_idx
            )
            if self.current_epoch == 0:
                image = self.trainer.datamodule.plot(x[0], y[0])
                self.logger.experiment.add_image(
                    "ground-truth/val", image, global_step=self.global_step + batch_idx
                )

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.test_metrics(y_hat_hard, y)
        self.log("test_loss", loss, on_step=True, on_epoch=False)

        if batch_idx < 50:
            image = self.trainer.datamodule.plot(x[0], y_hat_hard[0])
            self.logger.experiment.add_image(
                "predictions/test", image, global_step=batch_idx
            )
            image = self.trainer.datamodule.plot(x[0], y[0])
            self.logger.experiment.add_image(
                "ground-truth/test", image, global_step=batch_idx
            )

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()


class ChangeDetectionModel(SegmentationModel):
    def __init__(
        self,
        model="FCSiamDiff",
        backbone="resnet18",
        num_channels=3,
        num_classes=2,
        weights=None,
        learning_rate=1e-3,
        freeze_encoder=True,
        pad=False,
    ):
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters()
        model = getattr(csip.models.fcsiam, model)(
            encoder_name=backbone,
            encoder_weights=None if weights != "imagenet" else "imagenet",
            in_channels=num_channels,
            classes=num_classes,
        )
        if weights is not None and weights != "imagenet":
            print(f"loading pretrained backbone from {weights}")
            state_dict = extract_encoder(weights)
            model.encoder.load_state_dict(state_dict)
            if freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = False

        self.model = model

        if pad:
            self.pad_to = K.PadTo(size=(2048, 2048), pad_mode="constant", pad_value=0.0)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]

        if self.hparams.pad:
            h, w = y.shape[-2:]
            x = list(x)
            x[0] = self.pad_to(x[0])
            x[1] = self.pad_to(x[1])

        y_hat = self(x)

        if self.hparams.pad:
            x[0] = x[0][..., :h, :w]
            x[1] = x[1][..., :h, :w]
            y_hat = y_hat[..., :h, :w]

        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.val_metrics(y_hat_hard, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False)

        if batch_idx < 5:
            image = self.trainer.datamodule.plot(x[1][0], y_hat_hard[0])
            self.logger.experiment.add_image(
                "predictions/val", image, global_step=self.global_step + batch_idx
            )
            if self.current_epoch == 0:
                image = self.trainer.datamodule.plot(x[1][0], y[0])
                self.logger.experiment.add_image(
                    "ground-truth/val", image, global_step=self.global_step + batch_idx
                )

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]

        if self.hparams.pad:
            h, w = y.shape[-2:]
            x = list(x)
            x[0] = self.pad_to(x[0])
            x[1] = self.pad_to(x[1])

        y_hat = self(x)

        if self.hparams.pad:
            x[0] = x[0][..., :h, :w]
            x[1] = x[1][..., :h, :w]
            y_hat = y_hat[..., :h, :w]

        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.test_metrics(y_hat_hard, y)
        self.log("test_loss", loss, on_step=True, on_epoch=False)

        if batch_idx < 50:
            image = self.trainer.datamodule.plot(x[1][0], y_hat_hard[0])
            self.logger.experiment.add_image(
                "predictions/test", image, global_step=batch_idx
            )
            image = self.trainer.datamodule.plot(x[1][0], y[0])
            self.logger.experiment.add_image(
                "ground-truth/test", image, global_step=batch_idx
            )
