# config.py
import yaml
from ml_collections import ConfigDict
import argparse

def load_yaml_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return raw_cfg

def dict_to_configdict(d):
    """Convert nested Python dicts into ml-collections ConfigDict."""
    if isinstance(d, dict):
        cd = ConfigDict()
        for k, v in d.items():
            cd[k] = dict_to_configdict(v)
        return cd
    else:
        return d

def get_config(args):
    """
    Loads the YAML config file and merges it into a ConfigDict
    similar to the original Swin-UNet implementation.
    """
    yaml_cfg = load_yaml_cfg(args.cfg)
    cfg = dict_to_configdict(yaml_cfg)

    # Attach TRAIN and DATA sections if missing
    if "TRAIN" not in cfg:
        cfg.TRAIN = ConfigDict()
        cfg.TRAIN.USE_CHECKPOINT = False

    if "DATA" not in cfg:
        cfg.DATA = ConfigDict()
        cfg.DATA.IMG_SIZE = args.img_size

    # Store checkpoint path
    if "MODEL" in cfg and "PRETRAIN_CKPT" in cfg.MODEL:
        cfg.MODEL.PRETRAIN_CKPT = cfg.MODEL.PRETRAIN_CKPT

    return cfg
