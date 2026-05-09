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


def estimate_hmm_params(
    label_sequences: list[np.ndarray],
    n_classes: int,
    smoothing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate start probabilities and transition matrix from label sequences.

    Parameters
    ----------
    label_sequences : list of np.ndarray
        Per-recording 1-D integer label arrays.
    n_classes : int
        Number of label classes.
    smoothing : float
        Laplace smoothing count added to every cell before normalisation.

    Returns
    -------
    startprob : np.ndarray, shape (n_classes,)
    transmat  : np.ndarray, shape (n_classes, n_classes)
    """
    start_counts = np.ones(n_classes) * smoothing
    trans_counts = np.ones((n_classes, n_classes)) * smoothing

    for seq in label_sequences:
        if len(seq) == 0:
            continue
        start_counts[int(seq[0])] += 1
        for i in range(len(seq) - 1):
            trans_counts[int(seq[i]), int(seq[i + 1])] += 1

    startprob = start_counts / start_counts.sum()
    transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    return startprob, transmat


def viterbi_decode(
    probs: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
) -> np.ndarray:
    """Viterbi decoding using per-frame softmax emission probabilities.

    The model's softmax outputs are used directly as emission log-likelihoods
    inside an HMM whose transition structure was estimated from annotations.
    Uses hmmlearn for the Viterbi pass.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``(T, n_classes)`` — per-frame softmax probabilities.
    transmat : np.ndarray
        Shape ``(n_classes, n_classes)`` — row-stochastic transition matrix.
    startprob : np.ndarray
        Shape ``(n_classes,)`` — initial state distribution.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` int64 — Viterbi-decoded class labels.
    """
    n_classes = probs.shape[1]
    log_probs = np.log(np.clip(probs, 1e-9, 1.0))

    model = _LogProbHMM(n_components=n_classes, init_params="", params="")
    model.startprob_ = startprob
    model.transmat_ = transmat
    _, decoded = model.decode(log_probs, algorithm="viterbi")
    return decoded.astype(np.int64)


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
