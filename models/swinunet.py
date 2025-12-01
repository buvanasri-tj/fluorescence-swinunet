# models/swinunet.py
import torch.nn as nn
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # img_size MUST be 224 with window_size=7 + patch_size=4
        self.backbone = SwinTransformerSys(
            img_size=224,
            patch_size=4,
            in_chans=in_channels,   # 3-channel fluorescence
            num_classes=num_classes,  # binary mask
        )

    def forward(self, x):
        return self.backbone(x)
