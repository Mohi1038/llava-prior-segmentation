"""Metrics for indexed semantic segmentation masks."""

from collections.abc import Iterable

import numpy as np


def intersection_over_union(
    prediction: np.ndarray,
    target: np.ndarray,
    class_index: int,
    ignore_index: int | None = 255,
) -> float:
    """Return IoU for one class, or NaN when the class is absent from both masks."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = np.ones(target.shape, dtype=bool) if ignore_index is None else target != ignore_index
    predicted = (prediction == class_index) & valid
    expected = (target == class_index) & valid
    union = np.logical_or(predicted, expected).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(predicted, expected).sum() / union)


def mean_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    class_indices: Iterable[int],
    ignore_index: int | None = 255,
) -> tuple[float, dict[int, float]]:
    """Return mean IoU and per-class IoU for the requested indexed classes."""
    per_class = {
        class_index: intersection_over_union(prediction, target, class_index, ignore_index)
        for class_index in class_indices
    }
    values = np.asarray(list(per_class.values()), dtype=np.float64)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan"), per_class