"""Stage 1 — Validate labels and features with fast baselines.

Experiments
-----------
1. Kinematic baseline: detect_landings / detect_liftoffs + heel-y side assignment
   (sanity check — must exceed 80% accuracy)
2. XGBoost, all 22 features
3. XGBoost feature group ablation:
   A: position only (6 features)
   B: A + velocities (16 features)
   C: B + angles (20 features)
   D: all 22 features (same as Exp 2)

Usage
-----
    python -m experiments.gait_detection.stage1_baselines
    python -m experiments.gait_detection.stage1_baselines --config_overrides '{"n_val_athletes_tuning": 3}'

Output
------
    experiments/gait_detection/results/stage1_baselines.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import xgboost as xgb

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.detectors import KinematicDetector
from src.pose.utils.load_config import load_config
from src.gait.detection.metrics import timing_error_full, per_class_f1, confusion_matrix
from src.gait.detection.postprocess import derive_events
from src.gait.gait_data.dataset import tuning_split, load_dataset


# ── feature group slices (indices into the 22-feature vector) ────────────────
# 0–5   : norm. y-positions  (L/R heel, big_toe, ankle)
# 6–11  : y-velocities
# 12–15 : x-velocities
# 16–19 : joint angles
# 20–21 : hip y + hip dy/dt

SLICE_POSITION = list(range(0, 6))
SLICE_VELOCITY = list(range(6, 16))    # both y and x vels
SLICE_ANGLES   = list(range(16, 20))
SLICE_HIP      = list(range(20, 22))

FEATURE_GROUPS = {
    "A_position":            SLICE_POSITION,
    "B_position_velocity":   SLICE_POSITION + SLICE_VELOCITY,
    "C_position_vel_angles": SLICE_POSITION + SLICE_VELOCITY + SLICE_ANGLES,
    "D_all":                 list(range(22)),
}


def _flatten(records, feature_idx=None):
    """Flatten records to (X, y) numpy arrays."""
    X_list, y_list = [], []
    for r in records:
        feats = r.features if feature_idx is None else r.features[:, feature_idx]
        X_list.append(feats)
        y_list.append(r.labels)
    return np.vstack(X_list), np.concatenate(y_list)


def _eval_detector(detector, val_records, cfg: ExperimentConfig) -> dict:
    """Run a detector on all val records, collect frame-level and timing metrics."""
    all_true, all_pred = [], []
    timing: dict[str, list[dict]] = {
        "left_landing": [], "left_takeoff": [],
        "right_landing": [], "right_takeoff": [],
    }

    for rec in val_records:
        pred = detector.predict(rec.features, cfg.fps)
        all_true.append(rec.labels)
        all_pred.append(pred)

        gt_ev   = derive_events(rec.labels, cfg.fps)
        pred_ev = derive_events(pred,        cfg.fps)
        for key in timing:
            err = timing_error_full(pred_ev[key], gt_ev[key], cfg.fps)
            timing[key].append(err)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    acc = float((y_true == y_pred).mean())
    f1  = per_class_f1(y_true, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
    cm  = confusion_matrix(y_true, y_pred, n_classes=cfg.n_classes).tolist()

    def _mean_timing(lst: list[dict], unit: str) -> float:
        vals = [d[unit] for d in lst if not np.isnan(d[unit])]
        return float(np.mean(vals)) if vals else float("nan")

    timing_summary = {
        key: {
            "ms":     _mean_timing(timing[key], "ms"),
            "frames": _mean_timing(timing[key], "frames"),
        }
        for key in timing
    }

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm, "timing_error": timing_summary}


# ── 1. Kinematic baseline ─────────────────────────────────────────────────────

def run_kinematic(val_records, cfg: ExperimentConfig) -> dict:
    detector = KinematicDetector()
    result = _eval_detector(detector, val_records, cfg)
    return result


# ── 2. XGBoost ───────────────────────────────────────────────────────────

def run_xgboost(train_records, val_records, feature_idx, cfg: ExperimentConfig) -> dict:
    X_train, y_train = _flatten(train_records, feature_idx)
    X_val,   y_val   = _flatten(val_records,   feature_idx)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = clf.predict(X_val)

    acc = (y_val == y_pred).mean()
    f1  = per_class_f1(y_val, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
    cm  = confusion_matrix(y_val, y_pred, n_classes=cfg.n_classes).tolist()

    full_importances = np.zeros(22)
    indices = list(range(22)) if feature_idx is None else feature_idx
    full_importances[indices] = clf.feature_importances_

    return {
        "accuracy": float(acc),
        "f1": f1,
        "confusion_matrix": cm,
        "feature_importances": full_importances.tolist(),
    }


def main(cfg: ExperimentConfig | None = None) -> None:
    all_results = {}

    cfg = cfg or ExperimentConfig()
    records = load_dataset(cfg.annotations_csv, fps=cfg.fps)

    train_records, val_records = tuning_split(
        records, n_val_athletes=cfg.n_val_athletes_tuning, seed=cfg.tuning_seed
    )

    all_results["kinematic"] = run_kinematic(val_records, cfg)

    ablation = {}
    for name, feat_idx in FEATURE_GROUPS.items():
        ablation[name] = run_xgboost(train_records, val_records, feat_idx, cfg)
    all_results["xgboost_ablation"] = ablation

    all_results['data'] = {}
    all_results['data']['train_records'] = len(train_records)
    all_results['data']['val_records'] = len(val_records)

    out_path = cfg.results_path("stage1_baselines")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)


DEFAULT_CONFIG = "configs/experiments/stage1_baselines.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to YAML config file (default: configs/experiments/stage1_baselines.yaml)",
    )
    args = parser.parse_args()

    raw = load_config(args.config)
    cfg = ExperimentConfig(**raw.get("experiment", {}))
    main(cfg)
