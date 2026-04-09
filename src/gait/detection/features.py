"""Feature extraction for gait phase detection.

Produces a (T, 22) float32 array from a smoothed pose DataFrame.
Feature layout (22 total):
    0–5   : normalised vertical positions — L/R heel, big_toe, ankle y
    6–11  : vertical velocities — L/R heel, big_toe, ankle dy/dt
    12–15 : horizontal velocities — L/R heel, big_toe dx/dt
    16–17 : knee angles — L/R
    18–19 : ankle angles — L/R
    20    : hip y (normalised)
    21    : hip dy/dt
"""

import numpy as np
import pandas as pd


_POS_KPS = ["heel", "big_toe", "ankle"]  # L/R pairs for y-pos and y-vel
_VEL_X_KPS = ["heel", "big_toe"]          # L/R pairs for x-vel


def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance between two (T, 2) arrays."""
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=float)


def _angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Angle at `vertex` for the triplet p1–vertex–p2, shape (T,)."""
    v1 = p1 - vertex
    v2 = p2 - vertex
    cos = (v1 * v2).sum(axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _joint_xy(df: pd.DataFrame, side: str, joint: str) -> np.ndarray:
    return np.stack(
        [_col(df, f"{side}_{joint}_x"), _col(df, f"{side}_{joint}_y")], axis=1
    )


def compute_leg_length(df: pd.DataFrame) -> float:
    """Mean (hip–knee + knee–ankle) Euclidean distance across all frames."""
    total = np.zeros(len(df))
    for side in ("left", "right"):
        hip = _joint_xy(df, side, "hip")
        knee = _joint_xy(df, side, "knee")
        ankle = _joint_xy(df, side, "ankle")
        total += _euclidean(hip, knee) + _euclidean(knee, ankle)
    return float(total.mean() / 2)  # average of left and right


def extract_features(df: pd.DataFrame, fps: float) -> np.ndarray:
    """Extract 22 per-frame features from a smoothed pose DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Smoothed pose data with columns like ``left_heel_x``, ``left_heel_y``, etc.
        Expected keypoints per side: hip, knee, ankle, heel, big_toe.
    fps : float
        Recording frame rate used to compute velocities (frames/second).

    Returns
    -------
    np.ndarray
        Shape (T, 22) float32.  Rows correspond to the rows of ``df``.
    """
    T = len(df)
    dt = 1.0 / fps

    leg_length = compute_leg_length(df)
    if leg_length < 1e-6:
        leg_length = 1.0  # TODO exit or warning

    # Use mean of left/right hip_y as the reference vertical
    hip_y = (_col(df, "left_hip_y") + _col(df, "right_hip_y")) / 2.0

    def norm_y(raw_y: np.ndarray) -> np.ndarray:
        return (raw_y - hip_y) / leg_length

    def vel(x: np.ndarray) -> np.ndarray:
        """Central differences, forward/backward at edges."""
        return np.gradient(x, dt)

    features = []

    # ── vertical positions (normalised) ─────────────────────────────────────
    for side in ("left", "right"):
        for kp in _POS_KPS:
            y = norm_y(_col(df, f"{side}_{kp}_y"))
            features.append(y)

    # ── vertical velocities ──────────────────────────────────────────────────
    for side in ("left", "right"):
        for kp in _POS_KPS:
            y = norm_y(_col(df, f"{side}_{kp}_y"))
            features.append(vel(y))

    # ── horizontal velocities ────────────────────────────────────────────────
    for side in ("left", "right"):
        for kp in _VEL_X_KPS:
            x = _col(df, f"{side}_{kp}_x") / leg_length
            features.append(vel(x))

    # ── joint angles ─────────────────────────────────────────────────────────
    for side in ("left", "right"):
        hip = _joint_xy(df, side, "hip")
        knee = _joint_xy(df, side, "knee")
        ankle = _joint_xy(df, side, "ankle")
        toe = _joint_xy(df, side, "big_toe")

        knee_angle = _angle(hip, knee, ankle)       # hip–knee–ankle
        ankle_angle = _angle(knee, ankle, toe)      # knee–ankle–toe
        features.append(knee_angle)
        features.append(ankle_angle)

    # ── hip features ─────────────────────────────────────────────────────────
    hip_y_norm = norm_y(hip_y)
    features.append(hip_y_norm)
    features.append(vel(hip_y_norm))

    arr = np.stack(features, axis=1).astype(np.float32)  # (T, 22)
    assert arr.shape == (T, 22), f"Expected (T, 22), got {arr.shape}"
    return arr
