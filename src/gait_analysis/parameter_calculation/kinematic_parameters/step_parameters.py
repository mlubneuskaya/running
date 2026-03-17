import pandas as pd
import numpy as np

from src.gait_analysis.parameter_calculation.utils.height import get_segmental_height_px


def _create_step_timeline(
    left_gct: pd.DataFrame, right_gct: pd.DataFrame
) -> pd.DataFrame:
    left_landings = left_gct[["landing_time"]].copy()
    left_landings["landing_foot"] = "left"

    right_landings = right_gct[["landing_time"]].copy()
    right_landings["landing_foot"] = "right"

    timeline = (
        pd.concat([left_landings, right_landings])
        .sort_values(by="landing_time")
        .reset_index(drop=True)
    )

    timeline["prev_foot"] = timeline["landing_foot"].shift(1)
    timeline["prev_time"] = timeline["landing_time"].shift(1)

    return timeline


def calculate_cadence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["step_time_ms"] = df["landing_time"] - df["prev_time"]
    df["cadence_spm"] = np.where(
        df["step_time_ms"] > 0, 60000.0 / df["step_time_ms"], np.nan
    )
    return df


def calculate_step_length(
        timeline_df: pd.DataFrame, pose_df: pd.DataFrame, runner_height_m: float
) -> pd.DataFrame:
    """
    Calculates step length as the distance between the current landing foot
    at contact and the previous landing foot's position at its contact time.
    """
    merged = pd.merge(
        timeline_df, pose_df, left_on="landing_time", right_on="timestamp_ms", how="inner"
    )

    merged = pd.merge(
        merged, pose_df, left_on="prev_time", right_on="timestamp_ms",
        how="left", suffixes=('', '_prev')
    )

    height_in_pixels = get_segmental_height_px(merged, head_neck_factor=1.16)
    height_in_pixels = merged['bbox_h'] * 1.2
    px_to_m_ratio = runner_height_m / height_in_pixels

    def get_step_dist(row, marker='heel'):
        curr_foot = row['landing_foot']  # 'left' or 'right'
        prev_foot = row['prev_foot']  # 'right' or 'left'

        if pd.isna(prev_foot): return np.nan

        curr_x = row[f'{curr_foot}_{marker}_x']
        curr_y = row[f'{curr_foot}_{marker}_y']
        prev_x = row[f'{prev_foot}_{marker}_x_prev']
        prev_y = row[f'{prev_foot}_{marker}_y_prev']

        return np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

    merged["step_length_heel_px"] = merged.apply(lambda r: get_step_dist(r, 'heel'), axis=1)
    merged["step_length_toe_px"] = merged.apply(lambda r: get_step_dist(r, 'big_toe'), axis=1)

    merged["step_length_heel_m"] = merged["step_length_heel_px"] * px_to_m_ratio
    merged["step_length_toe_m"] = merged["step_length_toe_px"] * px_to_m_ratio

    cols_to_keep = [c for c in merged.columns if not c.endswith('_prev')]
    return merged[cols_to_keep]


def calculate_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    step_time_s = df["step_time_ms"] / 1000.0

    df["speed_m_s"] = np.where(
        step_time_s > 0, df["step_length_heel_m"] / step_time_s, np.nan
    )
    df["speed_km_h"] = df["speed_m_s"] * 3.6
    df["pace_min_km"] = np.where(
        df["speed_m_s"] > 0, 1000.0 / (df["speed_m_s"] * 60.0), np.nan
    )
    df["speed_px_s"] = np.where(
        step_time_s > 0, df["step_length_heel_px"] / step_time_s, np.nan
    )

    df["pace_s_100px"] = np.where(
        df["speed_px_s"] > 0, 100.0 / df["speed_px_s"], np.nan
    )

    return df

def calculate_step_metrics(
    left_gct: pd.DataFrame,
    right_gct: pd.DataFrame,
    pose_df: pd.DataFrame,
    runner_height_m: float = 1.80,
) -> pd.DataFrame:
    timeline_df = _create_step_timeline(left_gct, right_gct)

    cadence_df = calculate_cadence(timeline_df)

    length_df = calculate_step_length(cadence_df, pose_df, runner_height_m)

    final_df = calculate_speed(length_df)

    columns_to_keep = [
        "landing_time",
        "landing_foot",
        "step_time_ms",
        "cadence_spm",
        "step_length_heel_px",
        "step_length_toe_px",
        "step_length_heel_m",
        "step_length_toe_m",
        "speed_m_s",
        "speed_km_h",
        "pace_min_km",
        "speed_px_s",
        "pace_s_100px",
    ]

    return final_df[columns_to_keep]
