import segmentation_models_pytorch as smp
import torch


class FCSiamDiff(smp.Unet):
    def __init__(self, *args, **kwargs):
        kwargs["aux_params"] = None
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x1, x2 = x
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [features2[i] - features1[i] for i in range(1, len(features1))]
        features.insert(0, features2[0])
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks


class FCSiamConc(smp.base.SegmentationModel):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None,
        in_channels=3,
        classes=1,
        activation=None,
    ):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_out_channels = [c * 2 for c in self.encoder.out_channels[1:]]
        encoder_out_channels.insert(0, self.encoder.out_channels[0])
        self.decoder = smp.unet.decoder.UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        x1, x2 = x
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [
            torch.cat([features2[i], features1[i]], dim=1)
            for i in range(1, len(features1))
        ]
        features.insert(0, features2[0])
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks
