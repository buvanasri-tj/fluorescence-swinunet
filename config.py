# config.py (safe version without ml_collections)
import yaml
import argparse

def get_config(args):
    """
    Loads YAML config and merges CLI overrides.
    """

    # Load YAML
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Convert YAML dictionary to a simple Python dictionary
    class Config:
        pass
    config = Config()

    # Turn nested YAML into attributes
    for key, value in cfg.items():
        setattr(config, key, value)

    # Apply CLI overrides from args.opts
    if args.opts:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            setattr(config, k, v)

    return config
