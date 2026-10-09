"""Convert refined token masks into SAM-compatible geometric prompts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PromptSet:
    points: np.ndarray
    box: np.ndarray | None


def masks_to_prompts(
    mask: np.ndarray,
    image_height: int,
    image_width: int,
    min_component_area: int = 16,
    morphological_kernel: int = 5,
) -> PromptSet:
    """Extract component centroids and a tight image-coordinate bounding box."""
    import cv2

    binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((morphological_kernel, morphological_kernel), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    valid = [index for index in range(1, component_count) if stats[index, cv2.CC_STAT_AREA] >= min_component_area]
    if not valid:
        return PromptSet(np.empty((0, 2), dtype=np.float32), None)
    points = np.asarray([centroids[index] for index in valid], dtype=np.float32)
    ys, xs = np.where(np.isin(labels, valid))
    box = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, image_width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, image_height - 1)
    return PromptSet(points=points, box=box)