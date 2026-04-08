"""Evaluation metrics for gait phase detection.

per_class_f1
    Compute F1 score per class and macro average.

timing_error
    Mean absolute timing error between predicted and ground-truth events.

confusion_matrix
    Unnormalized confusion matrix (n_classes × n_classes).
"""

from __future__ import annotations

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
    class_names: list[str] | None = None,
) -> dict:
    """Compute per-class and macro F1.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        1D integer arrays of true / predicted labels.
    n_classes : int
    class_names : list[str] | None
        Optional labels for the output dict keys.

    Returns
    -------
    dict
        ``{"class_0": float, ..., "macro": float}``
    """
    if class_names is None:
        class_names = [str(c) for c in range(n_classes)]

    result = {}
    f1_scores = []
    for c in range(n_classes):
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        result[class_names[c]] = f1
        f1_scores.append(f1)

    result["macro"] = float(np.mean(f1_scores))
    return result


def timing_error(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
) -> float:
    """Mean absolute timing error, matching predictions to ground truth greedily.

    Each ground-truth event is matched to its nearest unmatched prediction.
    Unmatched GT events contribute ``np.nan`` and are excluded from the mean.

    Parameters
    ----------
    pred_times, gt_times : np.ndarray
        1D arrays of event times (seconds or ms — same unit).

    Returns
    -------
    float
        Mean absolute error, or ``np.nan`` if no matches possible.
    """
    if len(pred_times) == 0 or len(gt_times) == 0:
        return float("nan")

    errors = []
    used = set()
    for gt in sorted(gt_times):
        dists = [(abs(gt - p), i) for i, p in enumerate(pred_times) if i not in used]
        if not dists:
            break
        dist, idx = min(dists)
        used.add(idx)
        errors.append(dist)

    return float(np.mean(errors)) if errors else float("nan")


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
) -> np.ndarray:
    """Unnormalized confusion matrix, shape (n_classes, n_classes).

    Rows = true class, columns = predicted class.
    """
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[int(t), int(p)] += 1
    return mat


def aggregate_confusion_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """Sum a list of confusion matrices (for averaging across LOAO folds)."""
    return np.sum(matrices, axis=0)


def timing_error_full(
    pred_times_s: np.ndarray,
    gt_times_s: np.ndarray,
    fps: float,
) -> dict[str, float]:
    """Mean absolute timing error reported in both milliseconds and frames.

    Uses the same greedy nearest-neighbour matching as ``timing_error``.
    Event times must be in **seconds**.

    Parameters
    ----------
    pred_times_s, gt_times_s : np.ndarray
        1D arrays of event times in seconds (as returned by ``derive_events``).
    fps : float
        Recording frame rate, used to convert seconds → frame count.

    Returns
    -------
    dict
        ``{"ms": float, "frames": float}`` — both are mean absolute errors.
        Values are ``nan`` when no matches are possible.
    """
    mae_s = timing_error(pred_times_s, gt_times_s)
    if np.isnan(mae_s):
        return {"ms": float("nan"), "frames": float("nan")}
    return {
        "ms":     mae_s * 1000.0,
        "frames": mae_s * fps,
    }
