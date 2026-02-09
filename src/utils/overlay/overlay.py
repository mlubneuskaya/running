import cv2
import json
import os
import logging

from src.utils.overlay.drawing import draw_runner_skeleton

logger = logging.getLogger(__name__)


def create_overlay_video(video_path: str, json_path: str, output_path: str):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            pose_data = data["pose_data"]
            skeleton_connections = data["connections"]
    except FileNotFoundError:
        logger.error(f"JSON file not found at {json_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_index < len(pose_data):
            frame_data = pose_data[frame_index]

            if frame_data is not None:
                try:
                    draw_runner_skeleton(frame, frame_data, skeleton_connections)
                except Exception as e:
                    logger.warning(f"Error drawing frame {frame_index}: {e}")
        frame_index += 1

        out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Saved overlay to {output_path}")
