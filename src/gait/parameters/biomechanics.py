"""Biomechanical parameter extraction from gait phase predictions and pose data.

Label convention (matches dataset.py):
    0 = left_stance
    1 = right_stance
    2 = flight
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.gait.parameters.angles import (
    calculate_interior_joint_angle,
    calculate_trunk_lean,
)

LEFT_STANCE = 0
RIGHT_STANCE = 1
FLIGHT = 2

_SIDE_NAME = {LEFT_STANCE: "left", RIGHT_STANCE: "right"}


def _xy(df: pd.DataFrame, side: str, joint: str) -> np.ndarray:
    return df[[f"{side}_{joint}_x", f"{side}_{joint}_y"]].to_numpy()


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def _segments(labels: np.ndarray) -> list[dict]:
    """Return contiguous label segments as {label, start, end} (end exclusive)."""
    if len(labels) == 0:
        return []
    segs = []
    s = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segs.append({"label": int(labels[s]), "start": s, "end": i})
            s = i
    segs.append({"label": int(labels[s]), "start": s, "end": len(labels)})
    return segs


def compute_pixel_height(
    df: pd.DataFrame,
    runner_height_m: float = 2.0,
) -> tuple[float, float]:
    """Mean shoulder→ankle pixel height and meters-per-pixel scale.

    Pixel height = mean over frames of (shoulder–hip + hip–knee + knee–ankle),
    averaged over left and right sides.  Head and neck are excluded.

    Returns
    -------
    (pixel_height, meters_per_pixel)
    """
    total = np.zeros(len(df))
    for side in ("left", "right"):
        shoulder = _xy(df, side, "shoulder")
        hip = _xy(df, side, "hip")
        knee = _xy(df, side, "knee")
        ankle = _xy(df, side, "ankle")
        total += _dist(shoulder, hip) + _dist(hip, knee) + _dist(knee, ankle)
    pixel_height = float(np.nanmean(total / 2))
    m_per_px = runner_height_m / pixel_height if pixel_height > 1e-6 else 0.0
    return pixel_height, m_per_px


def gait_intervals(labels: np.ndarray, fps: float) -> dict:
    """Extract per-step contact and flight intervals from gait phase labels.

    Parameters
    ----------
    labels : np.ndarray
        1D integer array (0=left_stance, 1=right_stance, 2=flight).
    fps : float
        Recording frame rate.

    Returns
    -------
    dict with two keys:
        "contact" : list of dicts
            {"side", "start_s", "end_s", "duration_s", "start_frame", "end_frame"}
        "flight"  : list of dicts
            {"start_s", "end_s", "duration_s", "start_frame", "end_frame"}
    """
    contact, flight = [], []
    for seg in _segments(labels):
        s_s = seg["start"] / fps
        e_s = seg["end"] / fps
        base = {
            "start_s": s_s,
            "end_s": e_s,
            "duration_s": e_s - s_s,
            "start_frame": seg["start"],
            "end_frame": seg["end"],
        }
        if seg["label"] in (LEFT_STANCE, RIGHT_STANCE):
            contact.append({"side": _SIDE_NAME[seg["label"]], **base})
        else:
            flight.append(base)
    return {"contact": contact, "flight": flight}


def compute_cadence(labels: np.ndarray, fps: float) -> dict:
    """Cadence from gait phase labels.

    Returns
    -------
    dict: {"steps_per_second", "steps_per_minute", "n_steps", "duration_s"}
    """
    ivs = gait_intervals(labels, fps)
    n_steps = len(ivs["contact"])
    duration = len(labels) / fps
    sps = n_steps / duration if duration > 0 else 0.0
    return {
        "steps_per_second": sps,
        "steps_per_minute": sps * 60.0,
        "n_steps": n_steps,
        "duration_s": duration,
    }


def compute_step_lengths(
    labels: np.ndarray,
    df: pd.DataFrame,
    fps: float,
    runner_height_m: float = 2.0,
) -> pd.DataFrame:
    """Per-step step length in metres, from consecutive initial contact positions.

    Step length = horizontal distance between consecutive landing events
    (flight→stance transitions), calibrated using the runner's pixel height.

    The first step has NaN step_length_m (no previous landing to compare).

    Parameters
    ----------
    labels : np.ndarray
        1D array (T,) aligned with df rows.
    df : pd.DataFrame
        Smoothed pose DataFrame, same length as labels.
    fps : float
        Recording frame rate.
    runner_height_m : float
        Assumed height for pixel-to-metre calibration.

    Returns
    -------
    pd.DataFrame
        Columns: step_number, side, landing_frame, landing_s, step_length_m
    """
    _, m_per_px = compute_pixel_height(df, runner_height_m)
    df_r = df.reset_index(drop=True)

    landings: list[dict] = []
    for i in range(len(labels) - 1):
        prev, curr = int(labels[i]), int(labels[i + 1])
        if prev == FLIGHT and curr in (LEFT_STANCE, RIGHT_STANCE):
            frame = i + 1
            side = _SIDE_NAME[curr]
            col = (
                f"{side}_heel_x"
                if f"{side}_heel_x" in df_r.columns
                else f"{side}_ankle_x"
            )
            x_px = float(df_r[col].iloc[frame])
            landings.append({"side": side, "frame": frame, "x_px": x_px})

    rows = []
    for i, lnd in enumerate(landings):
        sl = np.nan if i == 0 else abs(lnd["x_px"] - landings[i - 1]["x_px"]) * m_per_px
        rows.append(
            {
                "step_number": i + 1,
                "side": lnd["side"],
                "landing_frame": lnd["frame"],
                "landing_s": lnd["frame"] / fps,
                "step_length_m": sl,
            }
        )
    return pd.DataFrame(rows)


def compute_joint_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame knee, ankle, and torso angles for both sides.

    Angles
    ------
    - ``{side}_knee_angle``  : interior angle at knee (hip–knee–ankle), degrees
    - ``{side}_ankle_angle`` : interior angle at ankle (knee–ankle–big_toe), degrees
    - ``{side}_torso_lean``  : forward trunk lean from vertical (positive = forward
      in the direction of travel), degrees
    - ``torso_lean``         : mean of left and right torso lean

    The running direction (left→right or right→left) is inferred from the
    net horizontal displacement of the hip centroid and used to sign torso lean
    so that positive always means leaning forward.

    Returns
    -------
    pd.DataFrame with one row per frame, optionally including timestamp_ms and
    frame_index if present in *df*.
    """
    result: dict = {}
    for col in ("timestamp_ms", "frame_index"):
        if col in df.columns:
            result[col] = df[col].to_numpy()

    for side in ("left", "right"):
        result[f"{side}_knee_angle"] = calculate_interior_joint_angle(
            df, side, "hip", "knee", "ankle"
        )
        result[f"{side}_ankle_angle"] = calculate_interior_joint_angle(
            df, side, "knee", "ankle", "big_toe"
        )
        result[f"{side}_torso_lean"] = calculate_trunk_lean(df, side)

    result["torso_lean"] = (result["left_torso_lean"] + result["right_torso_lean"]) / 2

    return pd.DataFrame(result)
