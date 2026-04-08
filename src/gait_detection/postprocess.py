"""Post-processing of per-frame TCN predictions.

Steps
-----
1. ``min_duration_filter`` — remove label runs shorter than ``min_frames``
   (these are noise, not real gait events).
2. ``derive_events`` — extract landing / takeoff timestamps from state transitions.
"""

from __future__ import annotations

import numpy as np


def min_duration_filter(labels: np.ndarray, min_frames: int = 3) -> np.ndarray:
    """Remove label segments shorter than ``min_frames`` by absorbing them into
    the surrounding class.

    Short isolated segments (single or dual frames of a different class) are
    collapsed into the preceding class.  This is applied iteratively until no
    short segments remain.

    Parameters
    ----------
    labels : np.ndarray
        1D integer array of per-frame class predictions.
    min_frames : int
        Segments shorter than this are removed (default 3).

    Returns
    -------
    np.ndarray
        Filtered label array of the same shape and dtype.
    """
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(labels):
            j = i + 1
            while j < len(labels) and labels[j] == labels[i]:
                j += 1
            seg_len = j - i
            if seg_len < min_frames:
                # Replace with preceding class (or following if at start)
                replacement = labels[i - 1] if i > 0 else labels[j] if j < len(labels) else labels[i]
                labels[i:j] = replacement
                changed = True
            i = j
    return labels


def derive_events(
    labels: np.ndarray,
    fps: float,
) -> dict[str, np.ndarray]:
    """Derive gait events from label state transitions.

    Transition rules
    ----------------
    - ``flight → left_stance``  : left landing
    - ``left_stance → flight``  : left takeoff
    - ``flight → right_stance`` : right landing
    - ``right_stance → flight`` : right takeoff

    Parameters
    ----------
    labels : np.ndarray
        1D integer array (0=left_stance, 1=right_stance, 2=flight).
    fps : float
        Recording frame rate, used to convert frame indices to seconds.

    Returns
    -------
    dict
        Keys: ``"left_landing"``, ``"left_takeoff"``,
              ``"right_landing"``, ``"right_takeoff"``.
        Values: np.ndarray of times in seconds.
    """
    FLIGHT = 2
    LEFT = 0
    RIGHT = 1

    events: dict[str, list[float]] = {
        "left_landing": [],
        "left_takeoff": [],
        "right_landing": [],
        "right_takeoff": [],
    }

    for i in range(len(labels) - 1):
        prev, curr = int(labels[i]), int(labels[i + 1])
        t = (i + 1) / fps  # time at the transition frame
        if prev == FLIGHT and curr == LEFT:
            events["left_landing"].append(t)
        elif prev == LEFT and curr == FLIGHT:
            events["left_takeoff"].append(t)
        elif prev == FLIGHT and curr == RIGHT:
            events["right_landing"].append(t)
        elif prev == RIGHT and curr == FLIGHT:
            events["right_takeoff"].append(t)

    return {k: np.array(v) for k, v in events.items()}
