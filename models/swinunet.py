import torch
import torch.nn as nn
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # IMPORTANT:
        # Swin Transformer requires that (img_size / patch_size) % window_size == 0
        # With patch_size=4 and window_size=7, only img_size=224 works correctly.
        # DO NOT change this unless you rewrite the entire Swin architecture.
        self.backbone = SwinTransformerSys(
            img_size=224,         # REQUIRED (do NOT change to 256)
            patch_size=4,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)
