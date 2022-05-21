from typing import Dict, Any
import os
import yaml


def load_config(
    config_path=f"{__file__}/../../../configs/overwrite.yml",
) -> Dict[str, Any]:
    assert os.path.isfile(
        config_path
    ), f"No config file found at {os.path.abspath(config_path)}"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
        return config


if __name__ == "__main__":
    config = load_config()
    assert type(config) == dict
    print("Loaded")
    print(config)
    quit()
    import argparse

    import yaml

    config_path = os.getenv("local_config_path", "./configs/dev.yaml")
    parser = argparse.ArgumentParser(
        description=f"Run the pipelines from {config_path}."
    )
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=config_path)

    args = parser.parse_args()
    print(f"loading config from {args.config_path}")
    assert os.path.isfile(
        args.config_path
    ), f"No config found at {os.path.abspath(args.config_path)}"
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if args.input_dir:
        config["pipelines"][0]["input_dir"] = args.input_dir
    print("Input directory:", config["pipelines"][0]["input_dir"])

    if args.output_dir:
        config["pipelines"][-1]["output_dir"] = args.output_dir
    print("Output directory:", config["pipelines"][-1]["output_dir"])

    print(config)
    run_pipelines(config)
