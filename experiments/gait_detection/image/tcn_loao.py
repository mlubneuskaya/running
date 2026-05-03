"""Image pipeline Stage 4 — leave-one-athlete-out cross-validation + MLflow.

Mirrors tcn_leave_one_out.py but uses pre-extracted CNN image features.
Each LOAO run is logged as an MLflow parent run; each fold is a nested child.

When trial_ids is not set and best_params_json points at the stage 2 output,
all trials listed under "top_trials" are evaluated automatically.  After all
trials complete the winner (highest macro_f1_mean) is written to loao_best.json
for stage 5 to consume.

Usage
-----
    python -m experiments.gait_detection.img_tcn_loao
    python -m experiments.gait_detection.img_tcn_loao --config configs/experiments/image/tcn_leave_one_out.yaml

Output
------
    <output_dir>/loao_trial_<id>.json   — per-fold metrics for each trial
    <output_dir>/loao_best.json         — best trial selected by macro F1
    <checkpoint_dir>/loao_<id>_<athlete>.pt
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig
from experiments.gait_detection.study_utils import load_study, params_from_trial
from src.gait.detection.metrics import (
    aggregate_confusion_matrices,
    confusion_matrix,
    per_class_f1,
)
from src.gait.detection.model import TCN
from src.gait.detection.postprocess import min_duration_filter
from src.gait.detection.train import Trainer, TrainerConfig, seed_everything
from src.gait.gait_data.dataset import (
    GaitSequenceDataset,
    GaitWindowDataset,
    compute_class_weights,
    loao_splits,
    train_test_split,
)
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



@torch.no_grad()
def _predict(model: torch.nn.Module, rec, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(rec.features).unsqueeze(0).to(device)
    return model(x).squeeze(0).argmax(dim=-1).cpu().numpy()


def run_loao(
    trial_id: int,
    params: dict,
    records: list,
    cfg: ExperimentConfig,
    n_features: int,
) -> dict:
    n_athletes   = len({r.athlete for r in records})
    fold_results = []
    all_cms      = []

    with mlflow.start_run(run_name=f"img_tcn_loao_trial_{trial_id}"):
        mlflow.log_params(params)
        mlflow.log_param("n_features", n_features)

        for fold_i, (train_records, val_records, athlete) in enumerate(loao_splits(records), 1):
            logger.info(
                "[Trial %d | Fold %d/%d] Held out: %s (%d videos)",
                trial_id, fold_i, n_athletes, athlete, len(val_records),
            )

            dilations = getattr(dilation_schedules, params["dilation_schedule"])(params["n_blocks"])
            model = TCN(
                n_features=n_features,
                n_blocks=params["n_blocks"],
                n_filters=params["n_filters"],
                kernel_size=params["kernel_size"],
                dropout=params["dropout"],
                dilations=dilations,
            )

            class_weights = compute_class_weights(train_records, cfg.n_classes)
            ckpt = cfg.checkpoint_path(f"loao_{trial_id}_{athlete}")
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

            train_ds = GaitWindowDataset(train_records, window_size=cfg.window_size, feature_idx=None)
            val_ds   = GaitSequenceDataset(val_records, feature_idx=None)

            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=1,              shuffle=False,
                                      collate_fn=GaitSequenceDataset.collate, num_workers=0)

            def _epoch_cb(epoch, train_loss, val_loss):
                if epoch % 10 == 0:
                    logger.info("  Epoch %3d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

            train_result = trainer.fit(train_loader, val_loader, epoch_callback=_epoch_cb)

            device = next(model.parameters()).device
            fold_y_true, fold_y_pred = [], []

            for rec in val_records:
                raw_pred = _predict(model, rec, device)
                pred     = min_duration_filter(raw_pred, min_frames=3)
                fold_y_true.append(rec.labels)
                fold_y_pred.append(pred)

            y_true = np.concatenate(fold_y_true)
            y_pred = np.concatenate(fold_y_pred)
            f1     = per_class_f1(y_true, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
            cm     = confusion_matrix(y_true, y_pred, n_classes=cfg.n_classes)
            all_cms.append(cm)

            fold_result = {
                "athlete":          athlete,
                "n_val_videos":     len(val_records),
                "f1":               f1,
                "confusion_matrix": cm.tolist(),
                "training":         train_result,
            }
            fold_results.append(fold_result)

            with mlflow.start_run(run_name=f"fold_{athlete}", nested=True):
                mlflow.log_metric("macro_f1",      f1["macro"])
                mlflow.log_metric("best_epoch",    train_result["best_epoch"])
                mlflow.log_metric("best_val_loss", train_result["best_val_loss"])
                for cls in cfg.class_names:
                    mlflow.log_metric(f"f1_{cls}", f1[cls])

            logger.info(
                "  F1 macro=%.3f  left=%.3f  right=%.3f  flight=%.3f",
                f1["macro"], f1["left_stance"], f1["right_stance"], f1["flight"],
            )

        macro_f1s = [r["f1"]["macro"] for r in fold_results]
        total_cm  = aggregate_confusion_matrices(all_cms)

        summary = {
            "trial_id":               trial_id,
            "params":                 params,
            "n_folds":                len(fold_results),
            "macro_f1_mean":          float(np.mean(macro_f1s)),
            "macro_f1_std":           float(np.std(macro_f1s)),
            "per_class_f1_mean": {
                cls: float(np.mean([r["f1"][cls] for r in fold_results]))
                for cls in cfg.class_names
            },
            "total_confusion_matrix": total_cm.tolist(),
            "folds":                  fold_results,
        }

        mlflow.log_metric("macro_f1_mean", summary["macro_f1_mean"])
        mlflow.log_metric("macro_f1_std",  summary["macro_f1_std"])

    logger.info("Trial %d  Macro F1: %.3f ± %.3f", trial_id, summary["macro_f1_mean"], summary["macro_f1_std"])
    return summary


def main(cfg: ExperimentConfig, study_cfg: dict, trial_ids: list[int], features_dir: str, video_input_dir: str) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_tcn_loao")

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records)
    logger.info("Test set excluded: %s (%d train records)", test_athletes, len(records))

    n_features = records[0].features.shape[1]
    logger.info("n_features: %d", n_features)

    study        = load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    summaries = []
    for trial_id in trial_ids:
        if trial_id not in trials_by_id:
            raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")
        params = params_from_trial(trials_by_id[trial_id])
        logger.info("Starting LOAO for trial %d  %s", trial_id, params)

        summary  = run_loao(trial_id, params, records, cfg, n_features)
        summaries.append(summary)
        out_path = cfg.results_path(f"loao_trial_{trial_id}")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", out_path)

    best = max(summaries, key=lambda s: s["macro_f1_mean"])
    loao_best = {
        "best_trial_id":  best["trial_id"],
        "best_params":    best["params"],
        "macro_f1_mean":  best["macro_f1_mean"],
        "macro_f1_std":   best["macro_f1_std"],
    }
    best_path = cfg.results_path("loao_best")
    with open(best_path, "w") as f:
        json.dump(loao_best, f, indent=2)
    logger.info(
        "Best trial: %d  macro_F1=%.3f ± %.3f  → %s",
        best["trial_id"], best["macro_f1_mean"], best["macro_f1_std"], best_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg       = raw.get("study", {})
    trial_ids       = raw.get("trial_ids", [])
    best_params_json= raw.get("best_params_json")
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")

    if not trial_ids:
        if best_params_json:
            with open(best_params_json) as _f:
                _bp = json.load(_f)
        #     top_trials = _bp.get("top_trials")
        #     if top_trials:
        #         trial_ids = [t["trial_id"] for t in top_trials]
        #         logger.info(
        #             "Auto-selected %d trials from top_trials in %s: %s",
        #             len(trial_ids), best_params_json, trial_ids,
        #         )
        #     else:
                trial_ids = [_bp["best_trial_id"]]
                logger.info("Auto-selected trial %d from %s", trial_ids[0], best_params_json)
        else:
            raise ValueError(
                "'trial_ids' is empty and 'best_params_json' is not set."
            )
    if not study_cfg:
        raise ValueError("'study' section is missing in the config.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, trial_ids, features_dir, video_input_dir)
