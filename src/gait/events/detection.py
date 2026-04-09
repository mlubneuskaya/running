import numpy as np
from scipy.signal import find_peaks


def detect_landings(vel: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Return timestamps where vertical velocity crosses zero from negative to positive.

    This corresponds to the moment a keypoint reaches its lowest point and begins
    moving upward — i.e. initial ground contact.

    Parameters
    ----------
    vel : np.ndarray
        Vertical velocity computed on a y-axis where *upward is positive*
        (raw pixel y multiplied by -1 before differentiation).
    timestamps : np.ndarray
        Timestamps in ms, same length as ``vel``.

    Returns
    -------
    np.ndarray
        Timestamps (ms) of detected landing events.
    """
    crossings = np.where((vel[:-1] < 0) & (vel[1:] >= 0))[0]
    result = []
    for i in crossings:
        dv = vel[i + 1] - vel[i]
        frac = (-vel[i] / dv) if dv != 0 else 0.0
        result.append(timestamps[i] + frac * (timestamps[i + 1] - timestamps[i]))
    return np.array(result)


def detect_liftoffs(vel: np.ndarray, timestamps: np.ndarray, min_distance_ms: float = 200.0) -> np.ndarray:
    """Return timestamps of local velocity maxima, one per liftoff event.

    The moment of liftoff coincides with peak upward velocity as the foot
    pushes off the ground.

    Parameters
    ----------
    vel : np.ndarray
        Vertical velocity computed on a y-axis where *upward is positive*
        (raw pixel y multiplied by -1 before differentiation).
    timestamps : np.ndarray
        Timestamps in ms.
    min_distance_ms : float
        Minimum time between successive liftoff events (ms).  Default 200 ms
        guards against detecting multiple peaks within the same push-off.

    Returns
    -------
    np.ndarray
        Timestamps (ms) of detected liftoff events.
    """
    dt_ms = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    min_samples = max(1, int(min_distance_ms / dt_ms))
    peaks, _ = find_peaks(vel, distance=min_samples)
    return timestamps[peaks]


def enforce_alternating(landing_times: np.ndarray, liftoff_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter landing and liftoff timestamps so they strictly alternate.

    Merges both sets of events, sorts them by time, then greedily accepts
    the first event of each expected type.  The sequence starts with whichever
    event type appears first chronologically.

    Parameters
    ----------
    landing_times, liftoff_times : np.ndarray
        Timestamps in ms as returned by ``detect_landings`` / ``detect_liftoffs``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered (landing_times, liftoff_times) that strictly alternate.
    """
    all_events = sorted(
        [(t, "landing") for t in landing_times] +
        [(t, "liftoff") for t in liftoff_times]
    )

    if not all_events:
        return np.array([]), np.array([])

    result_landings: list[float] = []
    result_liftoffs: list[float] = []
    expected = all_events[0][1]

    for t, kind in all_events:
        if kind == expected:
            if expected == "landing":
                result_landings.append(t)
                expected = "liftoff"
            else:
                result_liftoffs.append(t)
                expected = "landing"

    return np.array(result_landings), np.array(result_liftoffs)
