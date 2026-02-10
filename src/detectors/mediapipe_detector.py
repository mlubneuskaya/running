from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from src.detectors.base import ROIDetector
from src.utils.roi import ROI


class MediaPipeROIDetector(ROIDetector):

    def __init__(self, model_path: str, score_threshold: float = 0.25):
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            max_results=5,
            score_threshold=score_threshold,
            running_mode=VisionRunningMode.IMAGE,
            category_allowlist=["person"]
        )

        self.detector = ObjectDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[ROI]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return None

        h_img, w_img, _ = frame.shape
        best_box = None
        max_area = 0

        for detection in detection_result.detections:
            bbox = detection.bounding_box

            w = bbox.width
            h = bbox.height
            area = w * h

            if area > max_area:
                max_area = area
                best_box = (bbox.origin_x, bbox.origin_y, w, h)

        return best_box