import argparse
import yaml
import logging
import sys
import os

from ultralytics import YOLO, settings
from ray import tune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ray Tune for YOLO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    space = {}
    for key, val in config.get("space", {}).items():
        if val["type"] == "choice":
            space[key] = tune.choice(val["values"])
        elif val["type"] == "uniform":
            space[key] = tune.uniform(val["min"], val["max"])

    tune_args = config.get("tune_args", {})
    model_path = tune_args.pop("model")

    datasets_dir = config.get("datasets_dir", os.getcwd())
    settings.update({"datasets_dir": datasets_dir})

    train_args = config.get("train_args", {})
    if "data" in train_args:
        train_args["data"] = os.path.abspath(train_args["data"])

    model = YOLO(model_path)  # TODO: replace with download_model

    logger.info("Starting Ray Tune with 4 parallel trials...")

    results = model.tune(space=space, **tune_args, **train_args)


if __name__ == "__main__":
    main()
