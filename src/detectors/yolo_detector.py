from typing import Optional, List
import collections

import numpy as np
from ultralytics import YOLO

from src.detectors.base import ROIDetector
from src.utils.roi import ROI


class YoloROIDetector(ROIDetector):
    def __init__(self, model_path: str, conf: float = 0.25, device: str = "cpu"):
        self.model = YOLO(model_path)  #TODO auto download
        self.conf = conf
        self.device = device
        self.model.to(device)

    def scan_video(self, video_path: str) -> List[Optional[ROI]]:

        results_generator = self.model.track(
            source=video_path,
            conf=self.conf,
            classes=[0],
            persist=True,
            verbose=False,
            stream=True,
            device=self.device,
            tracker="./configs/botsort_tracker_yolo.yaml",
            iou=0.7,
        )

        all_detections = []

        id_scores = collections.defaultdict(float)

        frame_count = 0
        for result in results_generator:
            frame_detections = []

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy()

                for i, box in enumerate(boxes):
                    tid = int(track_ids[i])
                    w, h = box[2], box[3]
                    area = w * h

                    id_scores[tid] += area

                    frame_detections.append({
                        "id": tid,
                        "box": box,
                        "area": area
                    })

            all_detections.append(frame_detections)
            frame_count += 1

        if not id_scores:
            return [None] * frame_count

        main_id = max(id_scores, key=id_scores.get)

        final_rois = []

        for frame_dets in all_detections:
            chosen_box = None

            for det in frame_dets:
                if det["id"] == main_id:
                    chosen_box = det["box"]
                    break

            if chosen_box is not None:
                cx, cy, w, h = chosen_box
                x = int(cx - w / 2)
                y = int(cy - h / 2)

                final_rois.append((max(0, x), max(0, y), int(w), int(h)))
            else:
                final_rois.append(None)

        return final_rois

    def detect(self, frame: np.ndarray) -> Optional[ROI]:
        results = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)
        return self._extract_best_box(results)

    def _extract_best_box(self, results) -> Optional[ROI]:
        """Helper to find the largest person box in a result."""
        if not results or len(results[0].boxes) == 0:
            return None

        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        best_box = None
        max_area = 0

        for box in boxes_xywh:
            cx, cy, w, h = box
            area = w * h
            if area > max_area:
                max_area = area
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                best_box = (max(0, x), max(0, y), int(w), int(h))

        return best_box