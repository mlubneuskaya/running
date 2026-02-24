import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def get_valid_frames(pose_data: list[dict], keypoints: list) -> list[int]:
    valid_indices = []
    for i, frame in enumerate(pose_data):
        if frame is None:
            continue
        is_valid = all(
            kpt_name in frame and frame[kpt_name] is not None for kpt_name in keypoints
        )
        if is_valid:
            valid_indices.append(i)
    return valid_indices


def flatten_pose_data(pose_data: list[dict], keys_to_exclude: set[str]) -> pd.DataFrame:
    flattened = []
    for entry in pose_data:
        if entry is None:
            continue
        row = {}
        for key, value in entry.items():
            if key in keys_to_exclude:
                continue
            if isinstance(value, list):
                if len(value) == 3:
                    row[f"{key}_x"] = value[0]
                    row[f"{key}_y"] = value[1]
                    row[f"{key}_conf"] = value[2]
                elif len(value) == 4:
                    for i, coord in enumerate(["x1", "y1", "w", "h"]):
                        row[f"bbox_{coord}"] = value[i]
            else:
                row[key] = value
        flattened.append(row)
    return pd.DataFrame(flattened)


def acceleration(pose_data: pd.DataFrame, fps) -> pd.DataFrame:
    coords_df = pose_data.filter(regex=r".+_(x|y)$").interpolate(
        method="polynomial", order=2, limit_direction="both"
    )

    accel_components = coords_df.apply(lambda x: np.gradient(np.gradient(x, 1/fps), 1/fps))

    keypoint_names = set(col.rsplit("_", 1)[0] for col in accel_components.columns)

    magnitude_df = pd.DataFrame(index=pose_data.index)
    for kp in keypoint_names:
        ax = accel_components[f"{kp}_x"]
        ay = accel_components[f"{kp}_y"]
        magnitude_df[f"{kp}_y"] = np.sqrt(ax**2 + ay**2)
    magnitude_df["timestamp_ms"] = pose_data["timestamp_ms"]

    return magnitude_df


def remove_outliers(
    acceleration_df: pd.DataFrame, pose_df: pd.DataFrame
) -> pd.DataFrame:
    accel_data = acceleration_df.drop(columns=["timestamp_ms"], errors="ignore").abs()

    Q1 = accel_data.quantile(0.25)
    Q3 = accel_data.quantile(0.75)
    IQR = Q3 - Q1
    upper_bounds = Q3 + 1.5 * IQR

    outlier_mask = accel_data > upper_bounds

    common_cols = outlier_mask.columns.intersection(pose_df.columns)

    pose_df[common_cols] = pose_df[common_cols].mask(outlier_mask[common_cols])
    return pose_df.interpolate(method="linear", limit_direction="both")


def apply_butterworth(pose_df, fps, cutoff=6.0):
    nyquist = 0.5 * fps
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype="low", analog=False)

    refined_df = pose_df.copy()
    coord_cols = refined_df.filter(regex=r"_[xy]$").columns

    for col in coord_cols:
        refined_df[col] = filtfilt(b, a, refined_df[col])

    return refined_df


def smooth_pose_data(
    pose_data: list[dict],
    keypoints: list[str],
    keys_to_exclude: set[str],
    fps: float,
    cutoff: float,
) -> pd.DataFrame:
    valid_indices = get_valid_frames(pose_data, keypoints)

    if not valid_indices:
        return pd.DataFrame()

    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]

    pose_slice = pose_data[start_idx : end_idx + 1]

    pose_df = flatten_pose_data(pose_slice, keys_to_exclude=keys_to_exclude)

    accel_df = acceleration(pose_df, fps=fps)

    cleaned_df = remove_outliers(accel_df, pose_df)

    smoothed_df = apply_butterworth(cleaned_df, fps=fps, cutoff=cutoff)

    return smoothed_df
