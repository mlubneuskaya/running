#!/usr/bin/env python3
"""Smoke tests for the image-based gait detection pipeline.

No real data required — all tests use synthetic frames, pose JSONs, and videos
written to a temporary directory.

Run locally
-----------
    python tests/smoke_img.py
    pytest tests/smoke_img.py -v

Tests
-----
1.  Backbone builds correctly and output has expected shape (resnet18 → 512-dim)
2.  _xywh_to_xyxy clips correctly to image boundaries
3.  extract_video processes a synthetic video and returns (T, 512)
4.  extract_video raises on missing video
5.  load_image_dataset loads records with correct feature dim and label length
6.  load_image_dataset raises FileNotFoundError on missing .npy
7.  GaitWindowDataset works with 512-dim image features
8.  GaitSequenceDataset collate pads correctly for 512-dim features
9.  TCN(n_features=512) forward pass produces correct output shape
10. Two-epoch training loop completes with finite loss on image features
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.gait.detection.model import TCN
from src.gait.detection.train import Trainer, TrainerConfig
from src.gait.gait_data.dataset import (
    GaitSequenceDataset,
    GaitWindowDataset,
    VideoRecord,
    compute_class_weights,
)
from src.gait.image.extractor import ImageFeatureExtractor, _build_backbone, _xywh_to_xyxy
from src.gait.image.dataset import load_image_dataset

N_FRAMES   = 40
IMG_H, IMG_W = 240, 320
BBOX       = [160.0, 120.0, 80.0, 100.0]   # cx, cy, w, h — stays inside image
FPS        = 30.0


# ── synthetic helpers ──────────────────────────────────────────────────────────

def _write_video(path: str, n_frames: int, h: int, w: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, FPS, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = (rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        out.write(frame)
    out.release()


def _write_pose_json(path: str, n_frames: int, bbox: list[float]) -> None:
    pose_data = []
    for i in range(n_frames):
        if i % 5 == 0:           # every 5th frame has no detection
            pose_data.append({})
        else:
            pose_data.append({"bbox": bbox, "timestamp_ms": i * 1000 / FPS})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"pose_data": pose_data}, f)


def _write_features(path: str, T: int, feat_dim: int, seed: int = 0) -> np.ndarray:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((T, feat_dim)).astype(np.float32)
    np.save(path, feats)
    return feats


def _write_annotations_csv(path: str, video_path: str, n_frames: int) -> None:
    """Write a minimal annotations CSV for one video."""
    rows = []
    for i in range(10, n_frames - 10):
        if i % 3 == 0:
            rows.append({"video_path": video_path, "frame_number": i, "label": "contact", "side": "left"})
        elif i % 3 == 1:
            rows.append({"video_path": video_path, "frame_number": i, "label": "contact", "side": "right"})
        else:
            rows.append({"video_path": video_path, "frame_number": i, "label": "flight", "side": ""})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_image_record(T: int = 200, feat_dim: int = 512, seed: int = 0) -> VideoRecord:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((T, feat_dim)).astype(np.float32)
    phase    = np.linspace(0, 6 * np.pi, T)
    labels   = np.where(np.sin(phase) > 0.4, 0,
               np.where(np.sin(phase) < -0.4, 1, 2)).astype(np.int64)
    return VideoRecord(video_path="synthetic", athlete=f"athlete_{seed}",
                       features=features, labels=labels)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_backbone_output_shape():
    backbone, dim = _build_backbone("resnet18")
    assert dim == 512
    backbone.eval()
    with torch.no_grad():
        x   = torch.randn(1, 3, 224, 224)
        out = backbone(x)
    assert out.shape == (1, 512), f"unexpected shape: {out.shape}"


def test_xywh_to_xyxy_clipping():
    # bbox at top-left corner: should clip to 0
    x1, y1, x2, y2 = _xywh_to_xyxy([5.0, 5.0, 20.0, 20.0], img_h=100, img_w=100, padding=0.5)
    assert x1 == 0 and y1 == 0, f"expected clip to 0,0 got {x1},{y1}"
    # bbox near bottom-right: should clip to img bounds
    x1, y1, x2, y2 = _xywh_to_xyxy([95.0, 95.0, 20.0, 20.0], img_h=100, img_w=100, padding=0.5)
    assert x2 == 100 and y2 == 100, f"expected clip to 100,100 got {x2},{y2}"


def test_extract_video_shape():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "test.mp4")
        pose_path  = os.path.join(tmp, "test.json")
        _write_video(video_path, N_FRAMES, IMG_H, IMG_W)
        _write_pose_json(pose_path, N_FRAMES, BBOX)

        extractor = ImageFeatureExtractor(backbone="resnet18", device="cpu",
                                          img_size=224, bbox_padding=0.1)
        features  = extractor.extract_video(video_path, pose_path)

    assert features.shape == (N_FRAMES, 512), f"unexpected shape: {features.shape}"
    assert features.dtype == np.float32
    # Frames without bbox should be zero
    assert np.all(features[0] == 0.0), "frame 0 (no detection) should be zeros"
    # Frames with bbox should be non-zero
    assert not np.all(features[1] == 0.0), "frame 1 (has detection) should not be zeros"


def test_extract_video_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        pose_path = os.path.join(tmp, "test.json")
        _write_pose_json(pose_path, N_FRAMES, BBOX)

        extractor = ImageFeatureExtractor(backbone="resnet18", device="cpu")
        raised = False
        try:
            extractor.extract_video("/nonexistent/video.mp4", pose_path)
        except RuntimeError:
            raised = True
    assert raised, "extract_video should raise RuntimeError for missing video"


def test_load_image_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        video_path   = "data/input/optojump/study_0/athlete_0.mov"
        feat_path    = os.path.join(tmp, "features", "study_0", "athlete_0.npy")
        csv_path     = os.path.join(tmp, "annotations.csv")
        _write_features(feat_path, N_FRAMES, 512)
        _write_annotations_csv(csv_path, video_path, N_FRAMES)

        records = load_image_dataset(
            annotations_csv=csv_path,
            features_dir=os.path.join(tmp, "features"),
            video_input_dir="data/input/optojump",
        )

    assert len(records) == 1
    rec = records[0]
    assert rec.features.shape[1] == 512
    assert len(rec.features) == len(rec.labels)
    assert rec.features.dtype == np.float32
    assert rec.labels.dtype == np.int64


def test_load_image_dataset_missing_npy():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = "data/input/optojump/study_0/athlete_0.mov"
        csv_path   = os.path.join(tmp, "annotations.csv")
        _write_annotations_csv(csv_path, video_path, N_FRAMES)

        raised = False
        try:
            load_image_dataset(
                annotations_csv=csv_path,
                features_dir=os.path.join(tmp, "features"),
                video_input_dir="data/input/optojump",
            )
        except FileNotFoundError:
            raised = True
    assert raised, "load_image_dataset should raise FileNotFoundError for missing .npy"


def test_window_dataset_with_image_features():
    records = [_make_image_record(T=200, feat_dim=512, seed=i) for i in range(5)]
    ds      = GaitWindowDataset(records, window_size=60, feature_idx=None)
    feats, labels, mask = ds[0]
    assert feats.shape  == (60, 512)
    assert labels.shape == (60,)
    assert mask.shape   == (60,)


def test_sequence_dataset_collate_image_features():
    records = [_make_image_record(T=t, feat_dim=512, seed=i) for i, t in enumerate([100, 150, 200])]
    ds      = GaitSequenceDataset(records)
    loader  = DataLoader(ds, batch_size=3, collate_fn=GaitSequenceDataset.collate)
    feats, labels, masks = next(iter(loader))
    assert feats.shape == (3, 200, 512), f"unexpected shape: {feats.shape}"
    assert masks[0, :100].all() and not masks[0, 100:].any()


def test_tcn_512_forward():
    model = TCN(n_features=512, n_blocks=2, n_filters=16, kernel_size=3)
    x     = torch.randn(2, 60, 512)
    out   = model(x)
    assert out.shape == (2, 60, 3), f"unexpected shape: {out.shape}"
    sums  = out.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_training_two_epochs_image_features():
    records = [_make_image_record(T=200, feat_dim=512, seed=i) for i in range(5)]
    model   = TCN(n_features=512, n_blocks=2, n_filters=16, kernel_size=3)
    weights = compute_class_weights(records)

    train_ds = GaitWindowDataset(records[:-1], window_size=60, feature_idx=None)
    val_ds   = GaitSequenceDataset(records[-1:])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=GaitSequenceDataset.collate, num_workers=0)

    cfg    = TrainerConfig(max_epochs=2, batch_size=2, lr=1e-3,
                           early_stopping_patience=10, window_size=60)
    result = Trainer(model, weights, cfg).fit(train_loader, val_loader)
    assert np.isfinite(result["best_val_loss"]), "val loss is not finite"
    assert result["epochs_trained"] == 2


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suite = [
        test_backbone_output_shape,
        test_xywh_to_xyxy_clipping,
        test_extract_video_shape,
        test_extract_video_missing_file,
        test_load_image_dataset,
        test_load_image_dataset_missing_npy,
        test_window_dataset_with_image_features,
        test_sequence_dataset_collate_image_features,
        test_tcn_512_forward,
        test_training_two_epochs_image_features,
    ]

    passed, failed = 0, 0
    for fn in suite:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
