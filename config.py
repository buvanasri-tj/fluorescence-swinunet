import yaml
import argparse

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))  # Recursively wrap nested dicts
            else:
                setattr(self, key, value)

    def override(self, overrides):
        if overrides:
            for k, v in zip(overrides[0::2], overrides[1::2]):
                keys = k.split(".")
                target = self
                for key in keys[:-1]:
                    target = getattr(target, key, None)
                    if target is None:
                        break
                if target:
                    setattr(target, keys[-1], v)

def get_config(args):
    """
    Loads YAML config and merges CLI overrides.
    Supports nested keys using dot notation in --opts.
    """
    with open(args.cfg, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = Config(cfg_dict)
    config.override(args.opts)

    return config
