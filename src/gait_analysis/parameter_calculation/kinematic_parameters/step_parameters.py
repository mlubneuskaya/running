import pandas as pd
import numpy as np


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

    valid_steps = (
        timeline[timeline["landing_foot"] != timeline["prev_foot"]]
        .dropna(subset=["prev_foot"])
        .copy()
    )
    return valid_steps


def calculate_cadence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["step_time_ms"] = df["landing_time"] - df["prev_time"]
    df["cadence_spm"] = np.where(
        df["step_time_ms"] > 0, 60000.0 / df["step_time_ms"], np.nan
    )
    return df


def calculate_step_length(
    df: pd.DataFrame, pose_df: pd.DataFrame, runner_height_m: float
) -> pd.DataFrame:
    merged = pd.merge(
        df, pose_df, left_on="landing_time", right_on="timestamp_ms", how="inner"
    )

    lead_heel_x = np.where(
        merged["landing_foot"] == "left", merged["left_heel_x"], merged["right_heel_x"]
    )
    trail_toe_x = np.where(
        merged["landing_foot"] == "left",
        merged["right_big_toe_x"],
        merged["left_big_toe_x"],
    )

    step_length_px = np.abs(lead_heel_x - trail_toe_x)

    running_posture_factor = 1  # 0.88
    effective_bbox_height_m = runner_height_m * running_posture_factor

    px_to_m_ratio = effective_bbox_height_m / merged["bbox_h"]

    merged["step_length_px"] = step_length_px
    merged["step_length_m"] = step_length_px * px_to_m_ratio

    return merged


def calculate_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    step_time_s = df["step_time_ms"] / 1000.0

    df["speed_m_s"] = np.where(
        step_time_s > 0, df["step_length_m"] / step_time_s, np.nan
    )

    df["speed_km_h"] = df["speed_m_s"] * 3.6

    df["pace_min_km"] = np.where(
        df["speed_m_s"] > 0, 1000.0 / (df["speed_m_s"] * 60.0), np.nan
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
        "step_length_m",
        "step_length_px",
        "speed_m_s",
        "speed_km_h",
        "pace_min_km",
    ]

    return final_df[columns_to_keep]
