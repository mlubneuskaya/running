"""Image pipeline — full ResNet fine-tuning on all train athletes.

Loads best hyperparameters from the tuning stage and trains the complete
backbone + head for the recorded epoch count.

Usage
-----
    python -m experiments.gait_detection.image.resnet_train \
        --config configs/experiments/image/resnet_train.yaml

Output
------
    <checkpoint_dir>/checkpoint.pt
    <output_dir>/resnet_training.json
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.train import get_device, seed_everything
from src.gait.gait_data.dataset import train_test_split
from src.gait.image.dataset import load_image_records_for_finetune
from src.gait.image.finetune import FrameCropDataset, ResNetFinetune
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def main(
    cfg: ExperimentConfig,
    best_params: dict,
    best_epoch: int,
    backbone_name: str,
    img_size: int,
    bbox_padding: float,
    pose_dir: str,
    video_input_dir: str,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_resnet_training")

    logger.info("Loading image records for fine-tuning …")
    all_records_info = load_image_records_for_finetune(
        cfg.annotations_csv, pose_dir, video_input_dir
    )
    all_records = [r for r, _, _ in all_records_info]

    _, _, test_athletes = train_test_split(all_records)
    test_set     = set(test_athletes)
    train_info   = [t for t in all_records_info if t[0].athlete not in test_set]
    logger.info("Test set excluded: %s  (%d train records)", test_athletes, len(train_info))

    logger.info("Loading crops into RAM …")
    train_ds = FrameCropDataset(train_info, img_size=img_size, bbox_padding=bbox_padding)
    loader   = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    device = get_device()
    model  = ResNetFinetune(
        backbone_name=backbone_name,
        n_classes=cfg.n_classes,
        dropout=best_params["dropout"],
    ).to(device)

    backbone_lr = best_params["lr"] * best_params["backbone_lr_factor"]
    head_params = [*model.drop.parameters(), *model.fc.parameters()]
    optimizer   = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": head_params,                 "lr": best_params["lr"]},
        ],
        weight_decay=best_params["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []

    with mlflow.start_run(run_name="resnet_train"):
        mlflow.log_params(best_params)
        mlflow.log_params({
            "max_epochs":    best_epoch,
            "backbone_name": backbone_name,
        })

        for epoch in range(1, best_epoch + 1):
            model.train()
            total = 0.0
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                total += loss.item()
            epoch_loss = total / max(len(loader), 1)
            train_losses.append(epoch_loss)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            if epoch % 10 == 0:
                logger.info("Epoch %3d/%d  train_loss=%.4f", epoch, best_epoch, epoch_loss)

        ckpt_path = cfg.checkpoint_path("checkpoint")
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_metric("final_loss", train_losses[-1])
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
        logger.info("Checkpoint saved → %s", ckpt_path)

    out = {
        "params":        best_params,
        "epochs":        best_epoch,
        "backbone_name": backbone_name,
        "img_size":      img_size,
        "bbox_padding":  bbox_padding,
        "final_loss":    train_losses[-1],
        "checkpoint":    ckpt_path,
    }
    out_path = cfg.results_path("resnet_training")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Summary saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    pose_dir        = raw["pose_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")

    best_params_json = raw.get("best_params_json")
    if not best_params_json:
        raise ValueError("'best_params_json' is required.")

    with open(best_params_json) as _f:
        _bp = json.load(_f)

    best_params   = _bp["best_params"]
    best_epoch    = _bp["best_epoch"]
    backbone_name = _bp.get("backbone_name", raw.get("backbone_name", "resnet18"))
    img_size      = _bp.get("img_size",      raw.get("img_size",      224))
    bbox_padding  = _bp.get("bbox_padding",  raw.get("bbox_padding",  0.1))

    logger.info(
        "Loaded best params from %s  (epoch=%d  val_macro_f1=%.4f)",
        best_params_json, best_epoch, _bp["best_val_macro_f1"],
    )

    main(cfg, best_params, best_epoch, backbone_name, img_size, bbox_padding, pose_dir, video_input_dir)
