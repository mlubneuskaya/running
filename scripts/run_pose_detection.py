import os
import argparse
import cv2
import json
import logging
import numpy as np
from typing import List, Dict

from src.processors.base import PoseModel
from src.processors.yolo import YoloProcessor
from src.processors.mediapipe import MediaPipeProcessor
from src.utils.file_discovery import get_video_files
from src.utils.load_config import load_config

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


def process_video(video_file: str, output_dir: str, processor: PoseModel):
    filename = os.path.basename(video_file)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logger.error(f"Could not open {video_file}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    full_run_data = []

    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = (frame_index / fps) * 1000.0

        data = processor.process_frame(frame, timestamp_ms)

        full_run_data.append(data)

        frame_index += 1

    processor.reset()

    cap.release()

    name_only = os.path.splitext(filename)[0]
    json_filename = f"{name_only}.json"
    json_path = os.path.join(output_dir, json_filename)

    save_pose_data(full_run_data, json_path)


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
        processor = YoloProcessor(model_path=model_path, device=cfg["model"]["device"])
    elif model_type == "mediapipe":
        logger.info("Initializing MediaPipe...")
        processor = MediaPipeProcessor(model_path=model_path)
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

    for video_file in video_files:
        process_video(video_file, cfg["paths"]["output"], processor, cfg)


if __name__ == "__main__":
    main()
