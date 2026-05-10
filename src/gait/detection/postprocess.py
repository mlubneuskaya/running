"""Post-processing of per-frame TCN predictions.

Steps
-----
1. ``min_duration_filter`` — remove label runs shorter than ``min_frames``
   (these are noise, not real gait events).
2. ``derive_events`` — extract landing / takeoff timestamps from state transitions.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


# ── Viterbi helper (defined at module level to avoid per-call class creation) ─

def _make_log_prob_hmm():
    from hmmlearn.base import BaseHMM

    class _LogProbHMM(BaseHMM):
        """HMM whose emission log-likelihood is supplied directly as input."""

        def _compute_log_likelihood(self, X):
            return X  # X is (T, n_classes) log-probs — already what we want

        def _get_n_fit_scalars_per_param(self):
            return {}

    return _LogProbHMM


_LogProbHMM = _make_log_prob_hmm()


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


def majority_vote_filter(labels: np.ndarray, window: int = 5) -> np.ndarray:
    """Replace each frame label with the majority class in a sliding window.

    The window is centred on each frame (symmetric), so this method introduces
    no systematic directional bias.  ``window`` should be odd; if even, it is
    rounded up to the next odd number.

    Applied iteratively: each pass reads from the previous pass's output and
    writes to a fresh array.  Iteration stops when no frame changes, ensuring
    that residual short segments created by one pass (e.g. a single-frame
    artefact introduced when two isolated segments are both voted out) are
    cleaned up in subsequent passes.

    Parameters
    ----------
    labels : np.ndarray
        1D integer array of per-frame class predictions.
    window : int
        Number of frames in the sliding window (default 5).

    Returns
    -------
    np.ndarray
        Filtered label array of the same shape and dtype.
    """
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = len(labels)
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        out = labels.copy()
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            vals, counts = np.unique(labels[lo:hi], return_counts=True)
            majority = vals[np.argmax(counts)]
            if majority != labels[i]:
                out[i] = majority
                changed = True
        labels = out
    return labels


def smooth_probs_argmax(probs: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth per-frame softmax probabilities then take argmax, iterated until stable.

    Each pass applies a symmetric uniform (box) filter along the time axis and
    takes argmax.  The result is converted to one-hot vectors and used as input
    to the next pass.  Iteration stops when no frame label changes.

    The first pass uses the original softmax scores (rich probability information);
    subsequent passes use one-hot vectors and behave like iterated majority vote,
    cleaning up any short residual segments introduced by the previous pass.

    Parameters
    ----------
    probs : np.ndarray
        2D array of shape ``(T, n_classes)`` with softmax probabilities.
    window : int
        Uniform filter width (default 5).

    Returns
    -------
    np.ndarray
        1D integer array of shape ``(T,)`` with the predicted class per frame.
    """
    n_classes = probs.shape[1]
    current = probs.astype(np.float32)
    labels = None
    while True:
        smoothed   = uniform_filter1d(current, size=window, axis=0, mode="nearest")
        new_labels = smoothed.argmax(axis=1).astype(np.int64)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels  = new_labels
        current = np.eye(n_classes, dtype=np.float32)[labels]
    return labels


# Allowed transitions in the cyclic 4-state gait model.
# States: 0=S_L, 1=F_L→R, 2=S_R, 3=F_R→L
# Each state may self-loop or advance one step clockwise only.
_CYCLIC_MASK_4 = np.array([
    [1, 1, 0, 0],  # S_L   → {S_L, F_L→R}
    [0, 1, 1, 0],  # F_L→R → {F_L→R, S_R}
    [0, 0, 1, 1],  # S_R   → {S_R, F_R→L}
    [1, 0, 0, 1],  # F_R→L → {S_L, F_R→L}
], dtype=np.float64)


def labels_3_to_4(labels: np.ndarray) -> np.ndarray:
    """Split the flight class into direction-aware states.

    3-class input  : 0=left_stance, 1=right_stance, 2=flight
    4-state output : 0=S_L, 1=F_L→R, 2=S_R, 3=F_R→L

    Each flight frame is assigned based on the most recent preceding stance.
    Leading flight frames (before any stance) are resolved by looking at the
    first stance that follows them.
    """
    result = np.empty(len(labels), dtype=np.int64)
    result[labels == 0] = 0  # left_stance  → S_L
    result[labels == 1] = 2  # right_stance → S_R

    # Forward pass: assign flight frames from preceding stance
    last_stance = None
    for i in range(len(labels)):
        if labels[i] == 0:
            last_stance = 0
        elif labels[i] == 1:
            last_stance = 1
        else:
            if last_stance == 0:
                result[i] = 1   # F_L→R
            elif last_stance == 1:
                result[i] = 3   # F_R→L
            else:
                result[i] = 1   # placeholder for leading flight

    # Fix leading flight segment (before any stance) using first stance after it
    if len(labels) > 0 and labels[0] == 2:
        first_stance = next((labels[i] for i in range(len(labels)) if labels[i] != 2), None)
        i = 0
        while i < len(labels) and labels[i] == 2:
            result[i] = 3 if first_stance == 1 else 1
            i += 1

    return result


def estimate_hmm_params(
    label_sequences: list[np.ndarray],
    smoothing: float = 0.0,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 4-state HMM parameters from 3-class label sequences.

    Converts each sequence to 4 states (S_L, F_L→R, S_R, F_R→L) via
    ``labels_3_to_4``, counts transitions, then normalises.

    When ``symmetric=True`` (default) left/right symmetry is enforced by
    averaging each entry with its mirror under the mapping 0↔2, 1↔3
    (S_L↔S_R, F_L→R↔F_R→L).  Row sums remain 1 after averaging.

    Returns
    -------
    startprob : np.ndarray, shape (4,)
    transmat  : np.ndarray, shape (4, 4)
    """
    seqs_4 = [labels_3_to_4(seq) for seq in label_sequences]

    start_counts = np.ones(4) * smoothing
    trans_counts = _CYCLIC_MASK_4 * smoothing  # smoothing only on allowed transitions
    for seq in seqs_4:
        if len(seq) == 0:
            continue
        start_counts[int(seq[0])] += 1
        for i in range(len(seq) - 1):
            a, b = int(seq[i]), int(seq[i + 1])
            if _CYCLIC_MASK_4[a, b]:  # ignore annotation noise violating the cycle
                trans_counts[a, b] += 1

    startprob = start_counts / start_counts.sum()
    transmat  = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    if symmetric:
        mirror = np.array([2, 3, 0, 1])
        transmat  = (transmat + transmat[np.ix_(mirror, mirror)]) / 2
        startprob = (startprob + startprob[mirror]) / 2
        startprob /= startprob.sum()

    return startprob, transmat


def viterbi_decode(
    probs: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
) -> np.ndarray:
    """Viterbi decoding with a 4-state cyclic HMM (S_L, F_L→R, S_R, F_R→L).

    The 3-class model output is mapped to 4-state emissions before decoding
    and the result is mapped back to 3-class labels:

        Emission mapping  (3-class column → 4-state row):
            S_L    ← P(left_stance)   [col 0]
            F_L→R  ← P(flight)        [col 2]
            S_R    ← P(right_stance)  [col 1]
            F_R→L  ← P(flight)        [col 2]

        Decoding mapping  (4-state → 3-class):
            0 S_L   → 0 left_stance
            1 F_L→R → 2 flight
            2 S_R   → 1 right_stance
            3 F_R→L → 2 flight

    Parameters
    ----------
    probs : np.ndarray
        Shape ``(T, 3)`` — per-frame softmax probabilities from the model.
    transmat : np.ndarray
        Shape ``(4, 4)`` — 4-state row-stochastic transition matrix.
    startprob : np.ndarray
        Shape ``(4,)`` — 4-state initial state distribution.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` int64 — decoded labels in 3-class space.
    """
    emissions = np.stack([
        probs[:, 0],  # S_L   ← P(left_stance)
        probs[:, 2],  # F_L→R ← P(flight)
        probs[:, 1],  # S_R   ← P(right_stance)
        probs[:, 2],  # F_R→L ← P(flight)
    ], axis=1).astype(np.float32)

    log_emissions = np.log(np.clip(emissions, 1e-9, 1.0))
    model = _LogProbHMM(n_components=4, init_params="", params="")
    model.startprob_ = startprob
    model.transmat_  = transmat
    _, decoded = model.decode(log_emissions, algorithm="viterbi")
    return np.array([0, 2, 1, 2], dtype=np.int64)[decoded]


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
