import torch
import torch.nn as nn
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        self.backbone = SwinTransformerSys(
            img_size=256,
            patch_size=4,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)
