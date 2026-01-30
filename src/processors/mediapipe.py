import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from src.processors.base import PoseModel


class MediaPipeProcessor(PoseModel):
    KEYPOINT_MAP = {
        11: "left_shoulder",
        12: "right_shoulder",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }

    def __init__(self, model_path: str):
        super().__init__(model_path)

        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
        )

        self.landmarker = self.PoseLandmarker.create_from_options(self.options)

    def process_video(self, video_path: str) -> Optional[List[Dict[str, Any]]]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        full_run_data = []

        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = (frame_index / fps) * 1000.0

            data = self.process_frame(frame, timestamp_ms)

            full_run_data.append(data)

            frame_index += 1

        self.reset()

        cap.release()

        return full_run_data

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: float
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a frame using MediaPipe and returns standardized pixel coordinates.
        """
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not detection_result.pose_landmarks:
            return None

        landmarks = detection_result.pose_landmarks[0]

        clean_landmarks = []
        for lm in landmarks:
            clean_landmarks.append(
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            )

        processed_data = {
            "timestamp_ms": timestamp_ms,
            "raw_landmarks": clean_landmarks,
        }

        for idx, name in self.KEYPOINT_MAP.items():
            lm = landmarks[idx]

            pixel_x = int(lm.x * w)
            pixel_y = int(lm.y * h)

            processed_data[name] = (pixel_x, pixel_y, lm.visibility)

        return processed_data

    def reset(self):
        """Closes the current instance and starts a fresh one."""
        if self.landmarker:
            self.landmarker.close()
        self.landmarker = self.PoseLandmarker.create_from_options(self.options)

    def close(self):
        """Clean up MediaPipe resources"""
        if self.landmarker:
            self.landmarker.close()
