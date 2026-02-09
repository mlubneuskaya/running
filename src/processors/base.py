from abc import ABC, abstractmethod

import numpy as np


class PoseModel(ABC):

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

    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp_ms: float) -> dict:
        """
        Must return a standardized dictionary of keypoints.
        Returns None if no person is detected.
        """
        pass

    def process_video(self, video_file: str) -> list:
        pass

    def reset(self):
        pass

    def close(self):
        pass
