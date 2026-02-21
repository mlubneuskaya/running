import os
import csv
import logging
import yaml
import sys
import argparse
import multiprocessing as mp

from ultralytics import YOLO
from ray.tune import ExperimentAnalysis

from src.yolo.model_download import download_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def init_csv_file(path):
    headers = [
        "epoch", "time", "train/box_loss", "train/pose_loss",
        "train/kobj_loss", "train/cls_loss", "train/dfl_loss", "train/rle_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/precision(P)", "metrics/recall(P)", "metrics/mAP50(P)", "metrics/mAP50-95(P)",
        "val/box_loss", "val/pose_loss", "val/kobj_loss", "val/cls_loss",
        "val/dfl_loss", "val/rle_loss", "lr/pg0", "lr/pg1", "lr/pg2",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        logger.info(f"Initialized metrics file at: {os.path.abspath(path)}")
    else:
        logger.info(f"Found existing metrics file at: {os.path.abspath(path)}. Appending new data.")


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


def train_worker(model_path, train_args, metrics_path, is_resuming=False):
    logger.info(f"Initializing training worker for metrics: {metrics_path}")

    try:
        if not is_resuming:
            init_csv_file(metrics_path)

        logger.info(f"Loading model weights: {model_path}")
        model = YOLO(model_path)

        model.add_callback(
            "on_fit_epoch_end", lambda trainer: log_metrics_to_csv(trainer, metrics_path)
        )

        logger.info(f"Launching training with args: {train_args}")

        if is_resuming:
            model.train(resume=True, **train_args)
        else:
            model.train(**train_args)

        logger.info(f"Training finished successfully for {metrics_path}.")

    except Exception as e:
        logger.exception(f"Training crashed in worker: {e}")


def get_top_ray_tune_configs(experiment_dir, metric, mode, top_n):
    try:
        experiment_dir = os.path.abspath(experiment_dir)
        logger.info(f"Parsing Ray Tune directory: {experiment_dir}")
        analysis = ExperimentAnalysis(experiment_dir)
        df = analysis.dataframe()
    except Exception as e:
        logger.error(f"Failed to load ExperimentAnalysis: {e}")
        return []

    if df is None or df.empty:
        logger.error("Ray Tune results dataframe is empty.")
        return []

    if metric not in df.columns:
        logger.error(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        return []

    df = df.dropna(subset=[metric])
    ascending = (mode == "min")
    top_df = df.sort_values(by=metric, ascending=ascending).head(top_n)

    configs = []
    for _, row in top_df.iterrows():
        # Extract hyperparameters (Ray Tune prefixes them with 'config/')
        trial_config = {k.replace('config/', ''): v for k, v in row.items() if k.startswith('config/')}
        trial_id = row.get('trial_id', 'unknown')
        score = row.get(metric)
        configs.append((trial_id, trial_config, score))

    return configs


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def main():
    # Force 'spawn' method for CUDA multiprocessing safety
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    try:
        with open(args.config, "r") as f:
            training_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file '{args.config}' not found.")
        return

    logger.info(f"--- Starting Training Pipeline using config: {args.config} ---")

    metrics_path = training_config["paths"]["metrics_file"]
    model_name = training_config["paths"]["model_name"]
    model_dir = training_config["paths"]["model_dir"]

    resume_checkpoint = training_config["paths"].get("resume_path")

    is_resuming = False

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        logger.info(f"Resuming from: {resume_checkpoint}")
        model_path = resume_checkpoint
        is_resuming = True
    elif resume_checkpoint and not os.path.exists(resume_checkpoint):
        logger.error(f"Config 'resume_path' is set but file does not exist: {resume_checkpoint}")
        return
    else:
        logger.info(f"Preparing base model {model_name}.")
        try:
            model_path = download_model(model_name, model_dir)
        except Exception as e:
            logger.error(f"Model prep failed: {e}")
            return

    if not model_path:
        logger.error("Invalid model path. Exiting.")
        return

    ray_cfg = training_config.get("ray_tune")

    if ray_cfg:
        logger.info("Ray Tune configuration found. Launching multi-GPU batch training.")

        experiment_dir = ray_cfg.get("experiment_dir")
        top_n = ray_cfg.get("top_n", 1)
        metric = ray_cfg.get("metric", "metrics/mAP50-95(P)")
        mode = ray_cfg.get("mode", "max")
        gpus = ray_cfg.get("gpus", [0])

        base_train_args = training_config.get("training", {})

        top_trials = get_top_ray_tune_configs(experiment_dir, metric, mode, top_n)

        if not top_trials:
            logger.error("No valid trials found. Aborting batch run.")
            return

        processes = []
        for i, (trial_id, trial_config, score) in enumerate(top_trials):
            gpu_id = gpus[i % len(gpus)]

            logger.info(f"Preparing Trial {trial_id} (Score: {score}) for GPU {gpu_id}")

            run_args = base_train_args.copy()
            run_args.update(trial_config)
            run_args["device"] = str(gpu_id)

            base_name = run_args.get("name", "yolo_run")
            run_args["name"] = f"{base_name}_{trial_id}"

            name, ext = os.path.splitext(metrics_path)
            run_metrics_path = f"{name}_{trial_id}{ext}"

            p = mp.Process(target=train_worker, args=(model_path, run_args, run_metrics_path, False))
            p.start()
            processes.append(p)

        logger.info(f"Launched {len(processes)} background training processes.")

        for p in processes:
            p.join()

        logger.info("All batch training processes have completed.")

    else:
        logger.info("Standard single-run mode detected.")
        train_args = training_config["resuming"] if is_resuming else training_config.get("training", {})
        train_worker(model_path, train_args, metrics_path, is_resuming)


if __name__ == "__main__":
    main()