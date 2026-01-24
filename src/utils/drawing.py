from typing import Dict, Any

import cv2
import numpy as np


def draw_runner_skeleton(
    frame: np.ndarray, data: Dict[str, Any], confidence_threshold: float = 0.5
) -> np.ndarray:
    connections = [
        ("left_shoulder", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_shoulder", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ]

    for start_name, end_name in connections:
        if start_name in data and end_name in data:
            start_pt = data[start_name]
            end_pt = data[end_name]

            if start_pt[2] > confidence_threshold and end_pt[2] > confidence_threshold:
                cv2.line(
                    frame,
                    (int(start_pt[0]), int(start_pt[1])),
                    (int(end_pt[0]), int(end_pt[1])),
                    (0, 255, 255),
                    2,
                )

    for name, val in data.items():
        if isinstance(val, (list, tuple)) and len(val) == 3:
            x, y, conf = val
            if conf > confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    return frame
