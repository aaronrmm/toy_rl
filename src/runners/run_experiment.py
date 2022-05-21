import json
import os
from typing import Dict, Any
import wandb

import yaml

from runners.load_config import load_config

config: Dict = None


def run_experiment():
    pass


if __name__ == "__main__":
    import argparse

    overwrite_config_path = f"{__file__}/../../../configs/defaults.yml"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=overwrite_config_path,
        help=f"Config yaml file to use. Defaults to {overwrite_config_path}",
    )
    args = parser.parse_args()

    config: Dict[str, Any] = load_config(config_path=args.config_path)
    overwrite_config: Dict[str, Any] = load_config(config_path=overwrite_config_path)
    config.update(overwrite_config)
    print(config)
