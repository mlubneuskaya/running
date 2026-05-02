"""Leave-one-athlete-out cross-validation for selected Optuna trials.

Loads hyperparameters from the Optuna study by trial ID, runs a full LOAO CV
for each, and saves one result file per trial.  Intended to compare the
generalisation of the top-N hyperparameter configurations found in tuning.

Usage
-----
    python -m experiments.gait_detection.tcn_leave_one_out
    python -m experiments.gait_detection.tcn_leave_one_out --config configs/experiments/tcn_leave_one_out.yaml

Output
------
    <output_dir>/loao_trial_<id>.json         — metrics for each trial
    <checkpoint_dir>/loao_<id>_<athlete>.pt   — one checkpoint per fold per trial
"""

from __future__ import annotations

import argparse
import json
import logging
import os

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
    timing_error_full,
)
from src.gait.detection.model import TCN
from src.gait.detection.postprocess import derive_events, min_duration_filter
from src.gait.detection.train import TrainerConfig, Trainer
from src.gait.gait_data.dataset import (
    GaitSequenceDataset,
    GaitWindowDataset,
    compute_class_weights,
    load_dataset,
    loao_splits,
    train_test_split,
)
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model: torch.nn.Module, rec, device: torch.device, feature_idx) -> np.ndarray:
    model.eval()
    feats = rec.features if feature_idx is None else rec.features[:, feature_idx]
    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    return model(x).squeeze(0).argmax(dim=-1).cpu().numpy()


# ── single-trial LOAO CV ──────────────────────────────────────────────────────

def run_loao(
    trial_id: int,
    params: dict,
    records: list,
    cfg: ExperimentConfig,
) -> dict:
    """Run a full LOAO CV for one set of hyperparameters.

    Returns the summary dict (same schema as before, plus ``trial_id``).
    """
    n_athletes   = len({r.athlete for r in records})
    n_features   = len(cfg.feature_idx) if cfg.feature_idx is not None else 22
    fold_results = []
    all_cms      = []

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

        class_weights = compute_class_weights(train_records, n_classes=cfg.n_classes)
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

        train_ds = GaitWindowDataset(train_records, window_size=cfg.window_size, feature_idx=cfg.feature_idx)
        val_ds   = GaitSequenceDataset(val_records, feature_idx=cfg.feature_idx)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                                  collate_fn=GaitSequenceDataset.collate, num_workers=0)

        def _epoch_cb(epoch, train_loss, val_loss):
            if epoch % 10 == 0:
                logger.info("  Epoch %3d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

        train_result = trainer.fit(train_loader, val_loader, epoch_callback=_epoch_cb)

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
            pred_events = derive_events(pred, fps=cfg.fps)
            for key in timing_errs:
                timing_errs[key].append(timing_error_full(pred_events[key], gt_events[key], cfg.fps))

        y_true = np.concatenate(fold_y_true)
        y_pred = np.concatenate(fold_y_pred)
        f1     = per_class_f1(y_true, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
        cm     = confusion_matrix(y_true, y_pred, n_classes=cfg.n_classes)
        all_cms.append(cm)

        def _mean(lst, unit):
            vals = [d[unit] for d in lst if not np.isnan(d[unit])]
            return float(np.mean(vals)) if vals else float("nan")

        fold_results.append({
            "athlete":          athlete,
            "n_val_videos":     len(val_records),
            "f1":               f1,
            "confusion_matrix": cm.tolist(),
            "training":         train_result,
            "timing_error": {
                key: {
                    "ms":            _mean(timing_errs[key], "ms"),
                    "frames":        _mean(timing_errs[key], "frames"),
                    "signed_ms":     _mean(timing_errs[key], "signed_ms"),
                    "signed_frames": _mean(timing_errs[key], "signed_frames"),
                }
                for key in timing_errs
            },
        })
        logger.info(
            "  F1 macro=%.3f  left=%.3f  right=%.3f  flight=%.3f",
            f1["macro"], f1["left_stance"], f1["right_stance"], f1["flight"],
        )

    macro_f1s = [r["f1"]["macro"] for r in fold_results]
    total_cm  = aggregate_confusion_matrices(all_cms)

    summary = {
        "trial_id":           trial_id,
        "params":             params,
        "n_folds":            len(fold_results),
        "macro_f1_mean":      float(np.mean(macro_f1s)),
        "macro_f1_std":       float(np.std(macro_f1s)),
        "per_class_f1_mean":  {
            cls: float(np.mean([r["f1"][cls] for r in fold_results]))
            for cls in cfg.class_names
        },
        "total_confusion_matrix": total_cm.tolist(),
        "folds":              fold_results,
    }
    logger.info(
        "Trial %d  Macro F1: %.3f ± %.3f",
        trial_id, summary["macro_f1_mean"], summary["macro_f1_std"],
    )
    return summary


# ── entry point ───────────────────────────────────────────────────────────────

def main(cfg: ExperimentConfig, study_cfg: dict, trial_ids: list[int],
         auto_mode: bool = False) -> None:
    logger.info("Loading dataset …")
    all_records = load_dataset(cfg.annotations_csv, fps=cfg.fps)
    logger.info("%d videos loaded.", len(all_records))

    records, test_records, test_athletes = train_test_split(all_records)
    logger.info(
        "Test set excluded from LOAO: %s (%d videos). Training on %d videos.",
        test_athletes, len(test_records), len(records),
    )

    logger.info("Loading Optuna study …")
    study = load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    for trial_id in trial_ids:
        if trial_id not in trials_by_id:
            raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")

        trial  = trials_by_id[trial_id]
        params = params_from_trial(trial)
        logger.info(
            "Starting LOAO for trial %d  (val_loss=%.4f  %s)",
            trial_id, trial.value, params,
        )

        summary  = run_loao(trial_id, params, records, cfg)
        # In auto mode (single best trial from pipeline) write a stable loao.json
        # so downstream stages have a predictable dependency path.
        out_name = "loao" if auto_mode and len(trial_ids) == 1 else f"loao_trial_{trial_id}"
        out_path = cfg.results_path(out_name)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)

    cfg              = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg        = raw.get("study", {})
    trial_ids        = raw.get("trial_ids", [])
    best_params_json = raw.get("best_params_json")

    if not trial_ids:
        if best_params_json:
            with open(best_params_json) as _f:
                _bp = json.load(_f)
            trial_ids = [_bp["best_trial_id"]]
            logger.info("Auto-selected trial %d from %s", trial_ids[0], best_params_json)
        else:
            raise ValueError(
                "'trial_ids' is empty and 'best_params_json' is not set. "
                "Either list trial IDs in the config or point 'best_params_json' "
                "at the stage2 best_params.json file."
            )
    if not study_cfg:
        raise ValueError("'study' section is missing in the config.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, trial_ids, auto_mode=bool(best_params_json))
