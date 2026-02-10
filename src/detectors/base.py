from abc import abstractmethod, ABC
from typing import Optional, List

import cv2
import numpy as np

from src.utils.roi import ROI


class ROIDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[ROI]:
        pass

    def scan_video(self, video_path: str) -> List[Optional[ROI]]:
        rois = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            rois.append(self.detect(frame))
        cap.release()
        return rois