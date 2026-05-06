"""CNN backbone feature extraction from video frames using pose bounding boxes.

For each video frame that has a detection in the pose JSON, crops the bounding
box region (with optional padding), resizes to the backbone input size, applies
ImageNet normalization, and extracts the pooled feature vector.

Frames with no detection (empty entry in pose_data) receive a zero vector.
"""

from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_backbone(name: str) -> tuple[nn.Module, int]:
    """Return (backbone, feature_dim) with classifier head replaced by Identity."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        return m, 512
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        return m, 1280
    raise ValueError(f"Unknown backbone '{name}'. Supported: resnet18, efficientnet_b0")


def _xywh_to_xyxy(
    bbox: list[float], img_h: int, img_w: int, padding: float
) -> tuple[int, int, int, int]:
    """Convert YOLO [cx, cy, w, h] pixel bbox to padded [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox
    pw, ph = w * padding, h * padding
    x1 = max(0,     int(cx - w / 2 - pw))
    y1 = max(0,     int(cy - h / 2 - ph))
    x2 = min(img_w, int(cx + w / 2 + pw))
    y2 = min(img_h, int(cy + h / 2 + ph))
    return x1, y1, x2, y2


class ImageFeatureExtractor:
    """Extract per-frame CNN features from video crops defined by pose bboxes.

    Parameters
    ----------
    backbone : str
        "resnet18" (512-dim) or "efficientnet_b0" (1280-dim).
    device : str
        PyTorch device string ("cpu", "cuda", "mps").
    img_size : int
        Height and width to resize crops before the backbone (224 for ResNet/EfficientNet).
    bbox_padding : float
        Fractional padding added to each side of the detected bbox.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        device: str = "cpu",
        img_size: int = 224,
        bbox_padding: float = 0.1,
    ):
        self.device = torch.device(device)
        self.img_size = img_size
        self.bbox_padding = bbox_padding

        model, self.feature_dim = build_backbone(backbone)
        self.model = model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        logger.info(
            "ImageFeatureExtractor: backbone=%s  dim=%d  device=%s  img_size=%d  padding=%.2f",
            backbone, self.feature_dim, self.device, img_size, bbox_padding,
        )

    @torch.no_grad()
    def extract_video(self, video_path: str, pose_json_path: str) -> np.ndarray:
        """Extract features for every frame listed in the pose JSON.

        Frames with no detection get a zero vector so that the returned array
        has shape (T, feature_dim) where T == len(pose_data).

        Parameters
        ----------
        video_path : str
        pose_json_path : str

        Returns
        -------
        np.ndarray  shape (T, feature_dim), float32
        """
        with open(pose_json_path) as f:
            pose_data = json.load(f)["pose_data"]

        n_frames = len(pose_data)
        features = np.zeros((n_frames, self.feature_dim), dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        n_detected = 0

        try:
            while frame_idx < n_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    logger.warning(
                        "Video ended early at frame %d/%d: %s",
                        frame_idx, n_frames, video_path,
                    )
                    break

                frame_data = pose_data[frame_idx]
                if frame_data and "bbox" in frame_data:
                    h, w = frame_bgr.shape[:2]
                    x1, y1, x2, y2 = _xywh_to_xyxy(
                        frame_data["bbox"], h, w, self.bbox_padding
                    )
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        raise ValueError(
                            f"Empty crop at frame {frame_idx} in {video_path} "
                            f"(bbox={frame_data['bbox']}, img={w}x{h})"
                        )
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                    features[frame_idx] = self.model(tensor).squeeze(0).cpu().numpy()
                    n_detected += 1

                frame_idx += 1
        finally:
            cap.release()

        logger.info(
            "%s  frames=%d  detected=%d  (%.1f%%)",
            os.path.basename(video_path), n_frames, n_detected,
            100.0 * n_detected / max(n_frames, 1),
        )
        return features
