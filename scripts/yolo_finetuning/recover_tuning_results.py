import os

import plotly.express as px
from ray.tune import ExperimentAnalysis


relative_path = "./runs/detect/runs/train/yolo26l_tune2"
experiment_dir = os.path.abspath(relative_path)


try:
    analysis = ExperimentAnalysis(experiment_dir)
    df = analysis.results_df
    print(f"Successfully recovered data for {len(df)} trials!")
except Exception as e:
    print(f"Failed to parse directory: {e}")
    exit()

target_metric = "metrics/mAP50-95(P)"
df.sort_values(by=[target_metric]).to_csv(f"{relative_path}/summary.csv")

df = df.dropna(subset=[target_metric])
df = df[df["epoch"] == df.epoch.max()]

best_trial = analysis.get_best_trial(metric=target_metric, mode="max")

fig_parallel = px.parallel_coordinates(
    df,
    color=target_metric,
    dimensions=[
        "config/lr0",
        "config/lrf",
        "config/momentum",
        "config/pose",
        "config/weight_decay",
    ],
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="YOLO26l Pose - Hyperparameter Tuning Routes",
)
fig_parallel.write_html(f"{relative_path}/recovered_parallel_plot.html")


fig_scatter = px.scatter(
    df,
    x="config/lr0",
    y=target_metric,
    color="config/pose",
    size="config/momentum",
    hover_data=[
        "config/weight_decay",
        "config/lrf",
        "config/freeze",
        "config/pose",
        "config/lr0",
    ],
    title="Learning Rate vs Accuracy (Colored by Pose Weight)",
)
fig_scatter.write_html(f"{relative_path}/recovered_scatter_plot.html")
