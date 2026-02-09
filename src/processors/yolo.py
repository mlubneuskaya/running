import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional

from src.processors.base import PoseModel


class YoloProcessor(PoseModel):
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

    def process_video(self, video_path: str) -> Optional[List[List[Dict[str, Any]]]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0:
            fps = 30.0

        results_generator = self.model.track(  # TODO move model parameters to config
            source=video_path,
            persist=True,
            verbose=False,
            device=self.device,
            tracker="./configs/botsort_tracker_yolo.yaml",
            conf=0.1,
            iou=0.7,
            stream=True,
            visualize=False,
            batch=4,
        )

        all_frames_data = []

        for frame_idx, r in enumerate(
            results_generator
        ):  # TODO frame postprocessing -> function
            frame_candidates = []

            timestamp_ms = (frame_idx / fps) * 1000.0

            if r.boxes and r.boxes.id is not None:
                boxes = r.boxes.xywh.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy()

                if r.keypoints is not None:
                    all_keypoints = r.keypoints.data.cpu().numpy()
                else:
                    all_keypoints = None

                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    area = w * h

                    candidate = {
                        "timestamp_ms": timestamp_ms,
                        "track_id": int(track_ids[i]),
                        "bbox_area": float(area),
                        "bbox": box.tolist(),
                    }

                    if all_keypoints is not None:
                        person_kpts = all_keypoints[i]
                        candidate["raw_keypoints"] = person_kpts

                        for idx, name in self.KEYPOINT_MAP.items():
                            kp = person_kpts[idx]
                            candidate[name] = (float(kp[0]), float(kp[1]), float(kp[2]))

                    frame_candidates.append(candidate)

            all_frames_data.append(frame_candidates)

        return all_frames_data

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: float
    ) -> List[Dict[str, Any]]:
        """
        Runs tracking and returns ALL detected people in the frame.
        Global filtering (selecting the main runner) happens later in main.py.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            verbose=False,
            device=self.device,
            tracker="./configs/botsort_tracker_yolo.yaml",
            conf=0.1,
            iou=0.7,
        )

        if not results or not results[0].boxes:
            return []

        r = results[0]
        boxes = r.boxes.xywh.cpu().numpy()

        if r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy()
        else:
            track_ids = [-1] * len(boxes)

        if r.keypoints is None:
            return []

        all_keypoints = r.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)

        frame_candidates = []

        for i, box in enumerate(boxes):
            _, _, w, h = box
            area = w * h

            person_kpts = all_keypoints[i]

            candidate = {
                "timestamp_ms": timestamp_ms,
                "track_id": int(track_ids[i]),
                "bbox_area": float(area),
                "bbox": box.tolist(),
                "raw_keypoints": person_kpts,
            }

            for idx, name in self.KEYPOINT_MAP.items():
                kp = person_kpts[idx]
                candidate[name] = (float(kp[0]), float(kp[1]), float(kp[2]))

            frame_candidates.append(candidate)

        return frame_candidates

    def reset(self):  # TODO processor deleting -> here?
        """Resets the tracker state between videos."""
        try:
            self.model.predictor = None
        except AttributeError:
            pass

        try:
            self.model.trackers = None
        except AttributeError:
            pass
