import numpy as np
import pandas as pd


def calculate_link_metrics(df_coords, links):
    torso_l = np.sqrt(
        (df_coords["left_shoulder_x"] - df_coords["left_hip_x"]) ** 2
        + (df_coords["left_shoulder_y"] - df_coords["left_hip_y"]) ** 2
    )
    torso_r = np.sqrt(
        (df_coords["right_shoulder_x"] - df_coords["right_hip_x"]) ** 2
        + (df_coords["right_shoulder_y"] - df_coords["right_hip_y"]) ** 2
    )

    scale_factor = pd.concat([torso_l, torso_r]).median()

    if pd.isna(scale_factor) or scale_factor is not None == 0:
        scale_factor = 1.0

    link_results = []
    for name, (kp1, kp2) in links.items():
        dx = df_coords[f"{kp1}_x"] - df_coords[f"{kp2}_x"]
        dy = df_coords[f"{kp1}_y"] - df_coords[f"{kp2}_y"]
        dist = np.sqrt(dx**2 + dy**2)

        mean_len = dist.mean()
        cv = dist.std() / mean_len if mean_len != 0 else np.nan

        link_results.append(
            {
                "link": name,
                "cv": cv,
                "mean_length_normalized": mean_len / scale_factor,
                "scale_factor": scale_factor,
            }
        )

    return pd.DataFrame(link_results)


def calculate_jitter_metrics(df_coords, keypoints, fps, scale_factor):
    def get_trimmed_series(series):
        first = series.first_valid_index()
        last = series.last_valid_index()

        if first is not None and last is not None:
            return series.loc[first:last]
        return None

    def get_outlier_percentage(arr):
        q1 = np.nanpercentile(acceleration, 25)
        q3 = np.nanpercentile(acceleration, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outlier_count = (acceleration > upper_bound).sum()

        total_valid = (~np.isnan(acceleration)).sum()
        return (outlier_count / total_valid) * 100 if total_valid > 0 else 0

    jitter_results = []

    for kp in keypoints:
        v = (
            np.sqrt(df_coords[f"{kp}_x"].diff() ** 2 + df_coords[f"{kp}_y"].diff() ** 2)
            / scale_factor
            * fps
        )

        acceleration = v.diff().abs()
        acceleration = get_trimmed_series(acceleration)
        acceleration = acceleration.interpolate(method="linear")

        jitter_95th_val = np.nanpercentile(acceleration, 95) if len(acceleration) > 0 else 0

        jitter_results.append({
            "keypoint": kp,
            "jitter_95th_magnitude": jitter_95th_val,
            "jitter_outlier_percentage": get_outlier_percentage(acceleration),
        })

    return pd.DataFrame(jitter_results)


def aggregate_link_results(list_of_link_dfs):
    if not list_of_link_dfs:
        return pd.DataFrame()

    master_links = pd.concat(list_of_link_dfs, ignore_index=True)

    cols = [
        col
        for col in ["Model", "Video_ID", "Link", "CV"]
        if col in master_links.columns
    ]
    return master_links[cols]
