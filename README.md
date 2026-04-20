# Running Gait Analysis

Automated detection of gait events (left/right stance, flight phase) from treadmill running videos. Two parallel pipelines: pose-keypoint-based TCN and image-crop-based CNN+TCN.

---

## Repository layout

```
configs/                 YAML configs for every stage
data/                    All data (not in git — tracked by DVC)
  input/optojump/        Raw .mov recordings
  input/video/high_fps/  High-fps recordings for model comparison
  models/                Pretrained & fine-tuned weights
  output/                All pipeline outputs
experiments/             Runnable experiment scripts
  gait_detection/        Pose-based and image-based TCN stages
scripts/                 YOLO fine-tuning and pose extraction entry points
src/
  gait/                  Dataset, model, training, metrics, postprocessing
  pose/                  YOLO and MediaPipe pose processors
tests/                   Smoke tests (no real data required)
hpc/                     SLURM job scripts for PLGrid
dvc.yaml                 Full DVC pipeline definition
```

---

## Prerequisites

- Python 3.11
- CUDA 12.1+ for GPU runs (CPU works for inference and feature extraction)
- [DVC](https://dvc.org) for data versioning
- Docker (optional, recommended for cluster/cloud runs)

---

## Local setup

```bash
git clone <repo>
cd running

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

# Pull data from remote (configure DVC remote first)
dvc pull
```

Run smoke tests to verify the environment:

```bash
python tests/smoke_tcn.py
python tests/smoke_img.py
```

---

## DVC pipeline

The full pipeline is defined in `dvc.yaml`. All stage commands, dependencies, parameters and outputs are tracked.

```bash
dvc dag                   # visualise the full DAG
dvc repro                 # run everything end-to-end
dvc repro pose_extract    # run one stage (and its deps if stale)
dvc metrics show          # compare metrics across git commits
```

### Registering already-computed outputs (no re-run)

The YOLO fine-tuning stages were run on the HPC cluster. Register their outputs locally without re-running:

```bash
dvc commit yolo_hp_tune yolo_finetune nb_yolo_comparison
```

### Pipeline stages

| Stage | Script | Output |
|---|---|---|
| `yolo_hp_tune` | `scripts/yolo_finetuning/tune_hyperparameters_yolo.py` | Ray Tune trial dirs |
| `yolo_finetune` | `scripts/yolo_finetuning/finetune_yolo.py` | `data/models/yolo/tuned/` |
| `pose_extract_yolo_comparison` | `scripts/run_pose_detection.py` (loop) | `data/output/pose/tuned_yolo/` |
| `pose_extract_mediapipe` | `scripts/run_pose_detection.py` | `data/output/pose/mediapipe/` |
| `nb_yolo_comparison` | papermill notebook | `data/output/pose/notebooks/` |
| `pose_extract` | `scripts/run_pose_detection.py` | `data/output/pose/optojump/` |
| `stage1_baselines` | `experiments/gait_detection/baselines.py` | `data/output/gait/stage1/` |
| `stage2_tcn_tune` | `experiments/gait_detection/tcn_tuning.py` | `data/output/gait/stage2/` |
| `stage3_xgb_tune` | `experiments/gait_detection/xgb_tuning.py` | `data/output/gait/stage3/` |
| `stage4_loao` | `experiments/gait_detection/tcn_leave_one_out.py` | `data/output/gait/stage4/` |
| `stage5_train` | `experiments/gait_detection/tcn_training.py` | `data/output/gait/stage5/` |
| `stage6_infer` | `experiments/gait_detection/tcn_inference.py` | `data/output/gait/stage6/` |
| `img_feature_extract` | `experiments/gait_detection/img_feature_extract.py` | `data/output/gait/image_features/` |
| `img_baselines` | `experiments/gait_detection/img_baselines.py` | `data/output/gait/image/stage1/` |
| `img_tcn_tune` | `experiments/gait_detection/img_tcn_tune.py` | `data/output/gait/image/stage2/` |
| `img_tcn_loao` | `experiments/gait_detection/img_tcn_loao.py` | `data/output/gait/image/stage4/` |
| `img_tcn_train` | `experiments/gait_detection/img_tcn_train.py` | `data/output/gait/image/stage5/` |
| `img_tcn_infer` | `experiments/gait_detection/img_tcn_infer.py` | `data/output/gait/image/stage6/` |

### Changing the active YOLO model

After reviewing `nb_yolo_comparison`, edit `configs/config_yolo.yaml`:

```yaml
paths:
  models: "data/models/yolo/tuned/yolo26l_full_d725f_000342/weights/"
```

DVC tracks `paths.models` as a param of `pose_extract`, so changing it automatically invalidates `pose_extract` and all downstream gait stages on the next `dvc repro`.

---

## Docker

### Build

```bash
docker build -t running:latest .
```

The build smoke-tests all imports. If it fails, the image is not produced.

### Run a stage

Data lives outside the container and is mounted at runtime:

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/mlruns:/app/mlruns \
  running:latest \
  python -m experiments.gait_detection.tcn_tuning \
    --config configs/experiments/tcn_tuning.yaml
```

### Run smoke tests inside the container

```bash
docker run --rm running:latest python tests/smoke_tcn.py
docker run --rm running:latest python tests/smoke_img.py
```

### Selecting the device

Set the device in the relevant config before building or override at runtime:

```bash
# Pose extraction with CUDA
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs \
  running:latest \
  python scripts/run_pose_detection.py --config configs/config_yolo.yaml
```

For CPU-only runs (e.g. inference or feature validation), omit `--gpus all`.

---

## PLGrid (SLURM + Singularity)

HPC clusters cannot run Docker directly. Convert the image to Singularity and submit via SLURM:

```bash
# On your local machine or a build node with Docker
docker push your-registry/running:latest

# On PLGrid login node
singularity pull running.sif docker://your-registry/running:latest
```

Submit a job (example — adapt resource requests as needed):

```bash
#!/bin/bash
#SBATCH --job-name=tcn_tune
#SBATCH --account=plgroomagine-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8

singularity exec --nv \
  --bind $SCRATCH/running/data:/app/data \
  --bind $SCRATCH/running/configs:/app/configs \
  running.sif \
  python -m experiments.gait_detection.tcn_tuning \
    --config configs/experiments/tcn_tuning.yaml
```

The `--nv` flag passes through the host GPU. `--bind` mounts your scratch data directory.

For parallel Optuna jobs, launch an array job where all tasks point at the same study journal file (already on a shared filesystem):

```bash
#SBATCH --array=0-4
singularity exec --nv --bind $SCRATCH/running/data:/app/data \
  running.sif python -m experiments.gait_detection.tcn_tuning \
    --config configs/experiments/tcn_tuning.yaml
```

Existing SLURM scripts for the non-containerised workflow are in `hpc/`.

---

## Azure

Use Azure ML or Azure Container Instances.

### Azure Container Instances (one-off jobs)

```bash
# Push image to Azure Container Registry
az acr build --registry <your-registry> --image running:latest .

# Run a job
az container create \
  --resource-group <rg> \
  --name tcn-tune \
  --image <your-registry>.azurecr.io/running:latest \
  --gpu-count 1 --gpu-sku V100 \
  --azure-file-volume-account-name <storage-account> \
  --azure-file-volume-account-key <key> \
  --azure-file-volume-share-name running-data \
  --azure-file-volume-mount-path /app/data \
  --command-line "python -m experiments.gait_detection.tcn_tuning --config configs/experiments/tcn_tuning.yaml"
```

### Azure ML (managed jobs with experiment tracking)

Use the Docker image as the environment and point MLflow at the Azure ML tracking URI:

```bash
export MLFLOW_TRACKING_URI=$(az ml workspace show --query mlflow_tracking_uri -o tsv)
```

---

## MLflow

Experiment metrics from the image-based TCN pipeline are logged to MLflow automatically. Start the UI locally:

```bash
mlflow ui --port 5000
```

Inside Docker, expose the port and mount the mlruns directory:

```bash
docker run -p 5000:5000 -v $(pwd)/mlruns:/app/mlruns running:latest mlflow ui --host 0.0.0.0
```

---

## YOLO hyperparameter tuning config

`configs/pose/ray_tune.yaml` accepts an optional `datasets_dir` field that tells Ultralytics where to look for dataset files. Set it if your dataset YAML uses relative paths:

```yaml
datasets_dir: "/path/to/project/root"   # optional; defaults to cwd
```

This replaces the previously hardcoded PLGrid absolute path.
