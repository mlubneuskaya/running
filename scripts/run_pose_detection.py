import csv
import os
import argparse
import json
import logging
import time

import numpy as np
from typing import List, Dict, Tuple, Optional

from src.detectors.mediapipe_detector import MediaPipeROIDetector
from src.detectors.yolo_detector import YoloROIDetector
from src.processors.base import PoseModel
from src.processors.yolo import YoloProcessor
from src.processors.mediapipe import MediaPipeProcessor
from src.utils.file_discovery import get_video_files
from src.utils.get_path import get_mirror_path
from src.utils.load_config import load_config
from src.utils.detection.filtering import filter_main_runner
from src.utils.roi import ROI
from src.utils.setup_logger import setup_run_logging

logger = logging.getLogger(__name__)


def save_pose_data(
    data_list: List[Dict], skeleton_connections: List[Tuple[str, str]], output_path: str
):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    data = {"connections": skeleton_connections, "pose_data": data_list}
    with open(output_path, "w") as f:
        json.dump(data, f, default=convert, indent=4)


def process_video(video_file: str, output_json_path: str, processor: PoseModel=None, rois: Optional[List[ROI]] = None) -> int:
    if isinstance(processor, MediaPipeProcessor):
        full_run_data = processor.process_video(video_file, rois=rois)
    else:
        full_run_data = processor.process_video(video_file)

    if full_run_data is None:
        logger.warning(f"No data returned for {video_file}")
        return

    if isinstance(processor, YoloProcessor):
        full_run_data = filter_main_runner(full_run_data)

    save_pose_data(full_run_data, processor.connections, output_json_path)

    return len(full_run_data)


def main():

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

    output_root = cfg["paths"]["logs"]
    run_dir, log_path = setup_run_logging(logger, output_root, cfg)
    csv_log_path = os.path.join(run_dir, "inference_stats.csv")
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Frames", "ROI Time (s)", "Pose Time (s)", "Total Time (s)", "Time per Frame (s)"])

    model_type = cfg["model"]["type"]
    model_filename = cfg["model"]["name"]
    model_dir = cfg["paths"]["models"]
    model_path = os.path.join(model_dir, model_filename)

    device = cfg["model"].get("device", "cpu")
    roi_detector_class = None

    if model_type == "yolo":
        logger.info(f"Initializing YOLO Processor ({device})...")
        processor_class = lambda: YoloProcessor(
            model_path=model_path, device=cfg["model"]["device"]
        )

    elif model_type == "mediapipe":
        logger.info("Initializing MediaPipe Processor...")

        if "roi_detector" in cfg:
            roi_cfg = cfg["roi_detector"]
            roi_type = roi_cfg.get("type")
            roi_model_name = roi_cfg.get("name")
            roi_model_path = os.path.join(model_dir, roi_model_name) if roi_model_name else None

            if roi_type == "yolo" and roi_model_path:
                logger.info(f" - Attaching YOLO ROI Detector: {roi_model_name}")
                roi_detector_class = lambda: YoloROIDetector(
                    model_path=roi_model_path,
                    device=device,
                    conf=roi_cfg.get("conf", 0.25)
                )
            elif roi_type == "mediapipe" and roi_model_path:
                logger.info(f" - Attaching MediaPipe ROI Detector: {roi_model_name}")
                roi_detector_class = lambda: MediaPipeROIDetector(
                    model_path=roi_model_path,
                    score_threshold=roi_cfg.get("score_threshold", 0.25)
                )
            else:
                logger.warning("ROI detector configured but type unknown or path missing.")

        processor_class = lambda: MediaPipeProcessor(model_path=model_path)

    else:
        raise ValueError(f"Model {model_type} not supported")

    input_paths_cfg = cfg["paths"]["input"]
    input_roots = [input_paths_cfg] if isinstance(input_paths_cfg, str) else input_paths_cfg

    try:
        video_files = get_video_files(input_roots)
    except FileNotFoundError as e:
        logger.error(e)
        return

    if not video_files:
        logger.warning(f"No video files found in {input_roots}")
        return

    output_root = cfg["paths"]["output"]

    for video_file in video_files:

        source_root = next(
            (root for root in sorted(input_roots, key=len, reverse=True)
             if os.path.abspath(video_file).startswith(os.path.abspath(root))),
            None
        )

        if source_root is None:
            logger.warning(f"Skipping {video_file}: Could not determine its source root from config.")
            continue

        target_json_path = get_mirror_path(video_file, source_root, output_root, ".json")

        os.makedirs(os.path.dirname(target_json_path), exist_ok=True)

        logger.info(f"Processing {video_file}")

        t_start_total = time.perf_counter()

        rois = None
        if roi_detector_class is not None:
            logger.info("Detecting ROIs...")
            roi_detector = roi_detector_class()
            rois = roi_detector.scan_video(video_file)
            logger.info(f"  - Found ROIs for {len(rois)} frames")
        t_roi_end = time.perf_counter()

        t_pose_start = time.perf_counter()
        logger.info("Estimating Pose...")
        processor = processor_class()
        frame_count = process_video(
            video_file=video_file,
            output_json_path=target_json_path,
            processor=processor,
            rois=rois
        )
        t_total_end = time.perf_counter()

        roi_time = t_roi_end - t_start_total
        pose_time = t_total_end - t_pose_start
        total_time = t_total_end - t_start_total

        with open(csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.basename(video_file),
                f"{frame_count}",
                f"{roi_time:.4f}",
                f"{pose_time:.4f}",
                f"{total_time:.4f}",
                f"{total_time/frame_count:.4f}",
            ])

        processor.reset()

    logger.info(f"Run Complete. Logs saved to: {run_dir}")


if __name__ == "__main__":
    main()