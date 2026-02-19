import os
import csv
import logging
import yaml
import sys
import argparse

from ultralytics import YOLO

from src.yolo.model_download import download_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def init_csv_file(path):
    headers = [
        "epoch",
        "time",
        "train/box_loss",
        "train/pose_loss",
        "train/kobj_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "train/rle_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(P)",
        "metrics/recall(P)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(P)",
        "val/box_loss",
        "val/pose_loss",
        "val/kobj_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "val/rle_loss",
        "lr/pg0",
        "lr/pg1",
        "lr/pg2",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        logger.info(f"Initialized metrics file at: {os.path.abspath(path)}")
    else:
        logger.info(
            f"Found existing metrics file at: {os.path.abspath(path)}. Appending new data."
        )


def log_metrics_to_csv(trainer, metrics_path):
    train_losses = trainer.label_loss_items(trainer.tloss, prefix="train")

    val_metrics = trainer.metrics

    lr = trainer.lr

    row = [
        trainer.epoch + 1,
        getattr(trainer, "epoch_time", 0.0),
        train_losses.get("train/box_loss", 0.0),
        train_losses.get("train/pose_loss", 0.0),
        train_losses.get("train/kobj_loss", 0.0),
        train_losses.get("train/cls_loss", 0.0),
        train_losses.get("train/dfl_loss", 0.0),
        train_losses.get("train/rle_loss", 0.0),
        val_metrics.get("metrics/precision(B)", 0.0),
        val_metrics.get("metrics/recall(B)", 0.0),
        val_metrics.get("metrics/mAP50(B)", 0.0),
        val_metrics.get("metrics/mAP50-95(B)", 0.0),
        val_metrics.get("metrics/precision(P)", 0.0),
        val_metrics.get("metrics/recall(P)", 0.0),
        val_metrics.get("metrics/mAP50(P)", 0.0),
        val_metrics.get("metrics/mAP50-95(P)", 0.0),
        val_metrics.get("val/box_loss", 0.0),
        val_metrics.get("val/pose_loss", 0.0),
        val_metrics.get("val/kobj_loss", 0.0),
        val_metrics.get("val/cls_loss", 0.0),
        val_metrics.get("val/dfl_loss", 0.0),
        val_metrics.get("val/rle_loss", 0.0),
        lr.get("lr/pg0", 0.0),
        lr.get("lr/pg1", 0.0),
        lr.get("lr/pg2", 0.0),
    ]

    with open(metrics_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with open(args.config, "r") as f:
            training_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file '{args.config}' not found.")

    logger.info(f"--- Starting Training Pipeline using config: {args.config} ---")

    metrics_path = training_config["paths"]["metrics_file"]
    model_name = training_config["paths"]["model_name"]
    model_dir = training_config["paths"]["model_dir"]

    resume_checkpoint = training_config["paths"].get("resume_path")

    model_path = None
    is_resuming = False

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        logger.info(f"Resuming from: {resume_checkpoint}")
        model_path = resume_checkpoint
        is_resuming = True

    elif resume_checkpoint and not os.path.exists(resume_checkpoint):
        logger.error(
            f"Config 'resume_path' is set but file does not exist: {resume_checkpoint}"
        )
        return

    else:
        logger.info(f"Finetuning {model_name}.")

        try:
            init_csv_file(metrics_path)
            model_path = download_model(model_name, model_dir)
        except Exception as e:
            logger.error(f"Model prep failed: {e}")
            return

    if not model_path:
        return

    logger.info(f"Loading model weights: {model_path}")
    model = YOLO(model_path)

    model.add_callback(
        "on_fit_epoch_end", lambda trainer: log_metrics_to_csv(trainer, metrics_path)
    )

    logger.info("Launching training...")

    try:
        if is_resuming:
            train_args = training_config["resuming"]
            model.train(resume=True, **train_args)
        else:
            train_args = training_config["training"]
            model.train(**train_args)

        logger.info("Training finished successfully.")

    except Exception as e:
        logger.exception(f"Training crashed: {e}")


if __name__ == "__main__":
    main()
