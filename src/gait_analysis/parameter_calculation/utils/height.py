import numpy as np
import pandas as pd

def get_segmental_height_px(df: pd.DataFrame, head_neck_factor: float = 1.16) -> pd.Series:
    """
    Calculates the runner's height in pixels by summing bone segments:
    (Ankle-Knee) + (Knee-Hip) + (Hip-Shoulder) averaged across both sides.
    """
    def dist(p1, p2):
        return np.sqrt((df[f"left_{p1}_x"] - df[f"left_{p2}_x"])**2 +
                       (df[f"left_{p1}_y"] - df[f"left_{p2}_y"])**2), \
               np.sqrt((df[f"right_{p1}_x"] - df[f"right_{p2}_x"])**2 +
                       (df[f"right_{p1}_y"] - df[f"right_{p2}_y"])**2)

    left_shin, right_shin = dist("ankle", "knee")
    left_thigh, right_thigh = dist("knee", "hip")
    left_torso, right_torso = dist("hip", "shoulder")

    avg_shin = (left_shin + right_shin) / 2
    avg_thigh = (left_thigh + right_thigh) / 2
    avg_torso = (left_torso + right_torso) / 2

    return (avg_shin + avg_thigh + avg_torso) * head_neck_factor
