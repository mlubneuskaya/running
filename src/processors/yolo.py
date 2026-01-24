import numpy as np
from ultralytics import YOLO
from typing import Dict, Optional, Any

from src.processors.base import PoseModel


class YoloProcessor(PoseModel):
    # Mapping YOLOv8 COCO keypoint indices to standard names
    # 5:L-Shoulder, 6:R-Shoulder, 11:L-Hip, 12:R-Hip,
    # 13:L-Knee, 14:R-Knee, 15:L-Ankle, 16:R-Ankle
    KEYPOINT_MAP = {
        5: "left_shoulder",
        6: "right_shoulder",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }

    def __init__(self, model_path: str, device: str = "mps"):
        super().__init__(model_path)
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: float
    ) -> Optional[Dict[str, Any]]:
        """
        Runs inference, filters for the main runner, and standardizes output.
        Returns None if no runner is detected.
        """
        results = self.model(frame, verbose=False)

        if not results or results[0].boxes is None:
            return None

        boxes = results[0].boxes.xywh.cpu().numpy()

        best_person_idx = -1
        max_area = 0

        for i, box in enumerate(boxes):
            area = box[2] * box[3]
            if area > max_area:
                max_area = area
                best_person_idx = i

        if best_person_idx == -1:
            return None

        raw_keypoints = results[0].keypoints.data[best_person_idx].cpu().numpy()

        processed_data = {
            "timestamp_ms": timestamp_ms,
            "raw_keypoints": raw_keypoints,
            "bbox": boxes[best_person_idx],
        }

        for idx, name in self.KEYPOINT_MAP.items():
            kp = raw_keypoints[idx]
            processed_data[name] = (float(kp[0]), float(kp[1]), float(kp[2]))

        return processed_data
