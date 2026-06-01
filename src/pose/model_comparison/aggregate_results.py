import json
import os

import numpy as np
import pandas as pd

from src.pose.model_comparison.jitter_analysis import calculate_link_metrics, calculate_jitter_metrics
from src.pose.model_comparison.tracking_analysis import count_detection_gaps


def expand_lists_to_cols(df):
    expanded_data = {}

    for col in df.columns:
        expanded_data[f"{col}_x"] = df[col].apply(
            lambda val: val[0] if isinstance(val, (tuple, list, np.ndarray)) else np.nan
        )
        expanded_data[f"{col}_y"] = df[col].apply(
            lambda val: val[1] if isinstance(val, (tuple, list, np.ndarray)) else np.nan
        )

    return pd.DataFrame(expanded_data)


def aggregate_experiment_results(
    path_to_fps_dict,
    detector,
    links_dict,
    frame_ranges: "dict[str, tuple[int, int]] | None" = None,
):
    """Aggregate jitter, link-stability and detection-gap metrics across videos.

    Parameters
    ----------
    path_to_fps_dict : dict[str, float]
        Mapping from pose-JSON path to video FPS.
    detector : str
        Detector name written into the result DataFrames.
    links_dict : dict[str, tuple[str, str]]
        Skeleton link definitions (name → (kp_a, kp_b)).
    frame_ranges : dict[str, tuple[int, int]] | None
        Optional mapping from pose-JSON path to ``(first_visible, last_visible)``
        (inclusive, 0-based video frame indices).  When provided, only frames in
        ``[first_visible, last_visible]`` are included in the analysis.
        Pass ``None`` to use all frames (legacy behaviour).
    """
    global_links = []
    global_jitter = []
    global_gaps = []

    for path, fps in path_to_fps_dict.items():
        video_key = os.path.basename(path).split(".")[0]

        with open(path, "r") as f:
            raw_json = json.load(f)

        keypoints_list = list(set([x for xs in raw_json["connections"] for x in xs]))
        pose_data = raw_json["pose_data"]

        # ── optionally restrict to the annotated visibility window ──────────
        if frame_ranges is not None and path in frame_ranges:
            first_f, last_f = frame_ranges[path]
            pose_data = pose_data[first_f : last_f + 1]

        records = [row if row is not None else {} for row in pose_data]
        df_raw = pd.DataFrame(records)
        missing = [k for k in keypoints_list if k not in df_raw.columns]
        if missing:
            print(f"  ⚠ skipping {video_key}: no detections in analysis window "
                  f"(frame_range={frame_ranges.get(path) if frame_ranges else 'all'}, "
                  f"pose_len={len(raw_json['pose_data'])})")
            continue
        df_raw = df_raw[keypoints_list]
        df_coords = expand_lists_to_cols(df_raw)

        df_links = calculate_link_metrics(df_coords, links=links_dict)
        scale = df_links["scale_factor"][0]
        df_jitter = calculate_jitter_metrics(
            df_coords, keypoints=keypoints_list, fps=fps, scale_factor=scale
        )

        gap_stats = count_detection_gaps(df_coords, keypoints=keypoints_list)

        for df in [df_links, df_jitter, gap_stats]:
            df["path"] = path
            df["video_id"] = video_key
            df["model"] = detector

        global_links.append(df_links)
        global_jitter.append(df_jitter)
        global_gaps.append(gap_stats)

    links_df = pd.concat(global_links, ignore_index=True)
    jitter_df = pd.concat(global_jitter, ignore_index=True)
    gaps_df = pd.concat(global_gaps, ignore_index=True)

    return links_df, jitter_df, gaps_df
