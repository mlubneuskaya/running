"""Concrete GaitDetector implementations.

Both implement the same interface:
    predict(features: np.ndarray, fps: float) -> np.ndarray
    features: (T, 22) float32 — output of extract_features()
    returns:  (T,)   int64   — 0=left_stance, 1=right_stance, 2=flight

Feature layout (relevant indices used here)
-------------------------------------------
    0  left_heel_y_norm       3  right_heel_y_norm
    6  left_heel_dy/dt        9  right_heel_dy/dt
(image coordinates: larger y = lower in frame = closer to ground)
"""

from __future__ import annotations

import numpy as np

from src.gait.detection.postprocess import min_duration_filter
from src.gait.events.detection import detect_liftoffs, detect_landings, enforce_alternating

# Feature indices (see features.py layout)
_LEFT_HEEL_Y   = 0
_RIGHT_HEEL_Y  = 3
_LEFT_HEEL_DY  = 6
_RIGHT_HEEL_DY = 9


def _events_to_contact_mask(
    landing_times_ms: np.ndarray,
    liftoff_times_ms: np.ndarray,
    T: int,
    fps: float,
) -> np.ndarray:
    """Convert landing/liftoff event times (ms) to a boolean contact mask.

    Handles partial recordings (leading liftoff → contact from frame 0;
    trailing landing → contact to last frame), mirroring marks_to_labels.
    """
    contact = np.zeros(T, dtype=bool)

    def _frame(t_ms: float) -> int:
        return min(T - 1, max(0, int(round(t_ms / 1000.0 * fps))))

    all_events: list[tuple[float, str]] = sorted(
        [(t, "land") for t in landing_times_ms] +
        [(t, "lift") for t in liftoff_times_ms]
    )

    i = 0
    while i < len(all_events):
        t, kind = all_events[i]
        if kind == "lift":
            # Leading liftoff: contact from recording start
            contact[0: _frame(t) + 1] = True
            i += 1
        else:  # land
            start = _frame(t)
            if i + 1 < len(all_events) and all_events[i + 1][1] == "lift":
                end = _frame(all_events[i + 1][0])
                contact[start: end + 1] = True
                i += 2
            else:
                # Trailing landing: contact to recording end
                contact[start:] = True
                i += 1

    return contact


def _foot_contact_mask(
    vel_image: np.ndarray,
    timestamps_ms: np.ndarray,
    T: int,
    fps: float,
    min_distance_ms: float,
) -> np.ndarray:
    """Detect contact intervals for a single foot from its heel y-velocity.

    Parameters
    ----------
    vel_image : np.ndarray
        Heel y-velocity in image coordinates (positive = moving downward).
    """
    # Convert to upward-positive convention expected by the detection functions
    vel_up = -vel_image
    landings  = detect_landings(vel_up, timestamps_ms)
    liftoffs  = detect_liftoffs(vel_up, timestamps_ms, min_distance_ms)
    landings, liftoffs = enforce_alternating(landings, liftoffs)
    return _events_to_contact_mask(landings, liftoffs, T, fps)


class KinematicDetector:
    """Deterministic gait phase detector using kinematic rules.

    Runs detect_landings / detect_liftoffs / enforce_alternating independently
    on the left and right heel vertical velocity, then resolves conflicts via
    heel y-position comparison.

    Parameters
    ----------
    min_distance_ms : float
        Minimum time between consecutive liftoff peaks per foot (ms).
    min_frames : int
        Post-processing: remove label runs shorter than this.
    """

    def __init__(self, min_distance_ms: float = 200.0, min_frames: int = 3):
        self.min_distance_ms = min_distance_ms
        self.min_frames = min_frames

    def predict(self, features: np.ndarray, fps: float) -> np.ndarray:
        T = len(features)
        timestamps_ms = np.arange(T) / fps * 1000.0

        left_contact  = _foot_contact_mask(
            features[:, _LEFT_HEEL_DY],  timestamps_ms, T, fps, self.min_distance_ms
        )
        right_contact = _foot_contact_mask(
            features[:, _RIGHT_HEEL_DY], timestamps_ms, T, fps, self.min_distance_ms
        )

        left_y  = features[:, _LEFT_HEEL_Y]
        right_y = features[:, _RIGHT_HEEL_Y]

        labels = np.full(T, 2, dtype=np.int64)  # default: flight

        # Assign non-conflicting contacts
        labels[left_contact  & ~right_contact] = 0
        labels[right_contact & ~left_contact]  = 1

        # Conflict: both feet detected as contact → choose by heel y (lower = on ground)
        conflict = left_contact & right_contact
        labels[conflict & (left_y >= right_y)] = 0
        labels[conflict & (right_y >  left_y)] = 1

        return min_duration_filter(labels, self.min_frames)
