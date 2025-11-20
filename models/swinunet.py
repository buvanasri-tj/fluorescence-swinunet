# models/swinunet.py
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
def get_swinunet(num_classes=1, **kwargs):
    return SwinTransformerSys(num_classes=num_classes, **kwargs)
