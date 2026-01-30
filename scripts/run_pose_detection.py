import os
import argparse
import json
import logging
import numpy as np
from typing import List, Dict

from src.processors.base import PoseModel
from src.processors.yolo import YoloProcessor
from src.processors.mediapipe import MediaPipeProcessor
from src.utils.file_discovery import get_video_files
from src.utils.get_path import get_mirror_path
from src.utils.load_config import load_config
from src.utils.filtering import filter_main_runner

logger = logging.getLogger(__name__)


def save_pose_data(data_list: List[Dict], output_path: str):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(data_list, f, default=convert, indent=4)


def process_video(video_file: str, output_json_path: str, processor: PoseModel):
    full_run_data = processor.process_video(video_file)
    if isinstance(processor, YoloProcessor):
        full_run_data = filter_main_runner(
            full_run_data
        )  # TODO all None value handling

    save_pose_data(full_run_data, output_json_path)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run Pose Estimation Analysis")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return

    model_type = cfg["model"]["type"]
    model_filename = cfg["model"]["name"]
    model_dir = cfg["paths"]["models"]
    model_path = os.path.join(model_dir, model_filename)

    if model_type == "yolo":
        logger.info(f"Initializing YOLO ({cfg['model']['device']})...")
        processor_class = lambda path: YoloProcessor(
            model_path=path, device=cfg["model"]["device"]
        )
    elif model_type == "mediapipe":
        logger.info("Initializing MediaPipe...")
        processor_class = MediaPipeProcessor
    else:
        raise ValueError(f"Model {model_type} not supported")

    input_path = cfg["paths"]["input"]

    try:
        video_files = get_video_files(input_path)
    except FileNotFoundError as e:
        logger.error(e)
        return

    if not video_files:
        logger.warning(f"No video files found in {input_path}")
        return

    output_root = cfg["paths"]["output"]

    for video_file in video_files:
        target_json_path = get_mirror_path(video_file, input_path, output_root, ".json")

        os.makedirs(os.path.dirname(target_json_path), exist_ok=True)

        processor = processor_class(model_path)
        logger.info(f"Processing {video_file}")
        process_video(
            video_file=video_file,
            output_json_path=target_json_path,
            processor=processor,
        )
        del processor


if __name__ == "__main__":
    main()
