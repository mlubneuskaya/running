import numpy as np
import pandas as pd


def calculate_interior_angle(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    v1_x = p1_x - p2_x
    v1_y = p1_y - p2_y

    v2_x = p3_x - p2_x
    v2_y = p3_y - p2_y

    dot_product = (v1_x * v2_x) + (v1_y * v2_y)
    mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
    mag_v2 = np.sqrt(v2_x**2 + v2_y**2)

    cos_theta = dot_product / (mag_v1 * mag_v2 + 1e-8)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def calculate_interior_joint_angle(
    df: pd.DataFrame, side: str, p1_name: str, vertex_name: str, p3_name: str
) -> float:
    return calculate_interior_angle(
        df[f"{side}_{p1_name}_x"],
        df[f"{side}_{p1_name}_y"],
        df[f"{side}_{vertex_name}_x"],
        df[f"{side}_{vertex_name}_y"],
        df[f"{side}_{p3_name}_x"],
        df[f"{side}_{p3_name}_y"],
    )


def calculate_trunk_lean(df: pd.DataFrame, side: str) -> float:
    dx = df[f"{side}_shoulder_x"] - df[f"{side}_hip_x"]
    dy = df[f"{side}_hip_y"] - df[f"{side}_shoulder_y"]

    lean_rad = np.arctan2(dx, dy)
    lean_deg = np.degrees(lean_rad)

    runs_right = df[f"{side}_hip_x"].iloc[-1] > df[f"{side}_hip_x"].iloc[0]
    if not runs_right:
        lean_deg = -lean_deg

    return lean_deg


def calculate_kinematic_angles(df: pd.DataFrame, side: str = "left") -> pd.DataFrame:
    angles_data = {
        "timestamp_ms": df["timestamp_ms"].values,
        f"{side}_hip_angle": 180
        - calculate_interior_joint_angle(df, side, "shoulder", "hip", "knee"),
        f"{side}_knee_angle": 180
        - calculate_interior_joint_angle(df, side, "hip", "knee", "ankle"),
        f"{side}_ankle_angle": 90
        - calculate_interior_joint_angle(df, side, "knee", "ankle", "big_toe"),
        f"{side}_trunk_lean": calculate_trunk_lean(df, side),
    }

    return pd.DataFrame(angles_data)
