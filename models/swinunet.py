import torch
import torch.nn as nn
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()

        # Swin-Unet cannot handle arbitrary sizes.
        # img_size must be divisible by (4 * 2 * 2 * window_size = 56).
        # 224 is correct.
        self.backbone = SwinTransformerSys(
            img_size=img_size,
            patch_size=4,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)
