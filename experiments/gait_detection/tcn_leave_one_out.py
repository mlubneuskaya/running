"""Stage 4 — Full leave-one-athlete-out cross-validation.

Reads best hyperparameters from Stage 2 (lr, dropout) and Stage 3 (architecture).
Falls back to ExperimentConfig defaults if result files are not present.

Usage
-----
    python -m experiments.gait_detection.stage4_loao_cv
    python -m experiments.gait_detection.stage4_loao_cv --config_overrides '{"max_epochs": 50}'

Output
------
    experiments/gait_detection/results/stage4_loao_cv.json
    experiments/gait_detection/checkpoints/loao_<athlete>.pt  (one per fold)
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.metrics import timing_error_full, per_class_f1, confusion_matrix, aggregate_confusion_matrices
from src.gait.detection.model import TCN
from src.gait.detection.postprocess import derive_events, min_duration_filter
from src.gait.detection.train import TrainerConfig, Trainer
from src.gait.gait_data.dataset import load_dataset, compute_class_weights, loao_splits, GaitWindowDataset, \
    GaitSequenceDataset
from src.pose.utils.load_config import load_config


_REQUIRED_PARAMS = {"lr", "dropout", "n_blocks", "n_filters", "kernel_size"}


def _load_best_params(best_params_path: str) -> dict:
    if not best_params_path:
        raise ValueError("best_params_path is not set in the YAML config.")

    if not os.path.exists(best_params_path):
        raise FileNotFoundError(
            f"Best params file not found: {best_params_path}\n"
            "Run stage2_optuna.py first."
        )

    with open(best_params_path) as f:
        d = json.load(f)

    if "best_params" not in d:
        raise KeyError(
            f"'best_params' key missing in {best_params_path}."
        )

    params = d["best_params"]
    missing = _REQUIRED_PARAMS - params.keys()
    if missing:
        raise KeyError(
            f"Missing required parameters in {best_params_path}: {sorted(missing)}"
        )

    logger.info("Loaded best params from %s: %s", best_params_path, params)
    return params


@torch.no_grad()
def _predict(model: torch.nn.Module, rec, device: torch.device, feature_idx) -> np.ndarray:
    model.eval()
    feats = rec.features if feature_idx is None else rec.features[:, feature_idx]
    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    log_probs = model(x)
    return log_probs.squeeze(0).argmax(dim=-1).cpu().numpy()


def main(cfg: ExperimentConfig | None = None, best_params_path: str | None = None) -> None:
    cfg = cfg or ExperimentConfig()
    params = _load_best_params(best_params_path)

    logger.info("Loading dataset …")
    records = load_dataset(cfg.annotations_csv, fps=cfg.fps)
    logger.info("%d videos loaded.", len(records))

    fold_results = []
    all_cms = []

    for fold_i, (train_records, val_records, athlete) in enumerate(loao_splits(records), 1):
        logger.info("[Fold %d/%d] Held out: %s (%d videos)",
                    fold_i, len({r.athlete for r in records}), athlete, len(val_records))

        class_weights = compute_class_weights(train_records, n_classes=cfg.n_classes)

        n_features = len(cfg.feature_idx) if cfg.feature_idx is not None else 22

        model = TCN(
            n_features=n_features,
            n_blocks=params["n_blocks"],
            n_filters=params["n_filters"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
        )
        ckpt = cfg.checkpoint_path(f"loao_{athlete}")
        trainer_cfg = TrainerConfig(
            lr=params["lr"],
            dropout=params["dropout"],
            batch_size=cfg.batch_size,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.early_stopping_patience,
            lr_schedule_factor=cfg.lr_schedule_factor,
            lr_schedule_patience=cfg.lr_schedule_patience,
            max_grad_norm=cfg.max_grad_norm,
            window_size=cfg.window_size,
            checkpoint_path=ckpt,
        )
        trainer = Trainer(model, class_weights, trainer_cfg)

        train_ds = GaitWindowDataset(train_records, window_size=cfg.window_size, feature_idx=cfg.feature_idx)  # TODO extract method?
        val_ds   = GaitSequenceDataset(val_records, feature_idx=cfg.feature_idx)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                                  collate_fn=GaitSequenceDataset.collate, num_workers=0)

        def print_epoch(epoch, train_loss, val_loss):
            if epoch % 10 == 0:
                logger.info("Epoch %3d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

        train_result = trainer.fit(train_loader, val_loader, epoch_callback=print_epoch)

        # ── per-video evaluation ──────────────────────────────────────────────
        device = next(model.parameters()).device
        fold_y_true, fold_y_pred = [], []

        timing_errs: dict[str, list[dict]] = {
            "left_landing": [], "left_takeoff": [],
            "right_landing": [], "right_takeoff": [],
        }

        for rec in val_records:
            raw_pred = _predict(model, rec, device, cfg.feature_idx)
            pred = min_duration_filter(raw_pred, min_frames=3)
            fold_y_true.append(rec.labels)
            fold_y_pred.append(pred)

            gt_events   = derive_events(rec.labels, fps=cfg.fps)
            pred_events = derive_events(pred,        fps=cfg.fps)
            for key in timing_errs:
                err = timing_error_full(pred_events[key], gt_events[key], cfg.fps)
                timing_errs[key].append(err)

        y_true = np.concatenate(fold_y_true)
        y_pred = np.concatenate(fold_y_pred)
        f1     = per_class_f1(y_true, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
        cm     = confusion_matrix(y_true, y_pred, n_classes=cfg.n_classes)
        all_cms.append(cm)

        def _mean_timing(lst: list[dict], unit: str) -> float:
            vals = [d[unit] for d in lst if not np.isnan(d[unit])]
            return float(np.mean(vals)) if vals else float("nan")

        fold_result = {
            "athlete":          athlete,
            "n_val_videos":     len(val_records),
            "f1":               f1,
            "confusion_matrix": cm.tolist(),
            "training":         train_result,
            "timing_error": {
                key: {
                    "ms":     _mean_timing(timing_errs[key], "ms"),
                    "frames": _mean_timing(timing_errs[key], "frames"),
                }
                for key in timing_errs
            },
        }
        fold_results.append(fold_result)
        logger.info("F1 macro=%.3f  left=%.3f  right=%.3f  flight=%.3f",
                    f1["macro"], f1["left_stance"], f1["right_stance"], f1["flight"])

    # ── aggregate ────────────────────────────────────────────────────────────
    macro_f1s = [r["f1"]["macro"] for r in fold_results]
    total_cm  = aggregate_confusion_matrices(all_cms)

    summary = {
        "n_folds":          len(fold_results),
        "macro_f1_mean":    float(np.mean(macro_f1s)),
        "macro_f1_std":     float(np.std(macro_f1s)),
        "per_class_f1_mean": {
            cls: float(np.mean([r["f1"][cls] for r in fold_results]))
            for cls in cfg.class_names
        },
        "total_confusion_matrix": total_cm.tolist(),
        "best_hyperparams":       params,
        "folds":                  fold_results,
    }

    out_path = cfg.results_path("stage4_loao_cv")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", out_path)
    logger.info("Macro F1: %.3f ± %.3f", summary["macro_f1_mean"], summary["macro_f1_std"])


DEFAULT_CONFIG = "configs/experiments/stage4_loao_cv.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to YAML config file (default: configs/experiments/stage4_loao_cv.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)
    cfg = ExperimentConfig(**raw.get("experiment", {}))
    best_params_path = raw.get("best_params_path", None)
    main(cfg, best_params_path)
