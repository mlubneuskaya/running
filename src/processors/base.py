from abc import ABC, abstractmethod

import numpy as np


class PoseModel(ABC):
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
