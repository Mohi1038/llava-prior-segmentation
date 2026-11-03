"""Stage B maths: cached LLaVA traces -> per-class activation maps and VGM prompts.

Defaults reproduce the authors' released code. Switches expose the places where that code
differs from the paper text, so each difference can be ablated:

``attention_combine``  ``"union"`` (code: the attention mask is OR-ed with the thresholded
                       response and only masks D outside both) or ``"product"`` (paper Eq. 5-8:
                       D is thresholded, then multiplied by the attention mask);
``distance``           ``"aligned"`` (hidden state at the class token; reproduces the paper's
                       68 mIoU) or ``"shifted"`` (state one position earlier, i.e. the one that
                       predicts the class token; a misreading of the generate() indexing that
                       costs ~21 mIoU on the train tuning split, kept as an ablation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from .trace import ClassTrace, ImageTrace


@dataclass(frozen=True)
class SegmentSettings:
    alpha: float = 0.75
    beta: float = 0.19  # official ``--tth``
    # Scale-free maps (ours): "median_peak" rescales each class map to (d - median) /
    # (max - median), so 0 = typical location and 1 = the class's peak. alpha/beta then have the
    # same meaning for every VLM (cosine levels differ: LLaVA-1.5 median 0.09, Qwen2.5-VL 0.47).
    normalize: str = "none"
    distance: str = "aligned"  # or "shifted", or "centered" (mean image-token state removed)
    # Layer study: average the stored per-layer responses over hidden-state indices
    # [layer_from, layer_to] (inclusive) instead of the traced 7..13. Needs per-layer traces.
    layer_from: int = -1
    layer_to: int = -1
    # Precision/coverage fusion (ours, motivated by the layer study): SAM's points still come from
    # the precise middle-layer region, but the box also covers the deeper-layer region
    # [coverage_from, coverage_to] wherever its connected components touch that region. A
    # location is in the coverage region when its deep-layer response is at least
    # coverage_alpha * the class's maximum and the class wins the deep-layer argmax there.
    # -1 disables. Needs per-layer traces.
    coverage_from: int = -1
    coverage_to: int = -1
    coverage_alpha: float = 0.75
    use_attention: bool = True
    attention_combine: str = "union"  # or "product"
    attention_min_sum: float = 0.24
    attention_top_k: int = 30
    softmax_temperature: float = 0.01
    background: str = "global"  # or "per_class" / "relative" (ours), see assign_with_background
    relative_background: float = 0.75
    use_prototype: bool = True
    # QAP decision: "greedy" (LLaVA's text answer), "oracle" (ground-truth classes; analysis
    # only), "p_yes" (first-token probability >= qap_threshold, calibrated on train) or
    # "ensemble" (mean P(yes) over the main and the alternative query wordings >= qap_threshold).
    qap: str = "greedy"
    qap_threshold: float = 0.5
    # Evidence filter (our addition): drop a QAP-positive class when LLaVA is unsure
    # (p_yes < filter_p_yes) AND its response wins at most filter_area of the token grid
    # after competing with the other positives and background. 0 disables the filter.
    filter_p_yes: float = 0.0
    filter_area: float = 0.0
    # SAM prompting
    sam_iterations: int = 3
    use_negatives: bool = True
    negative_label: int = -1  # official passes other classes' points as -1 (padding tokens)
    use_box: bool = True
    box_mode: str = "single"  # or "components" (ours), see run_sam
    component_min_fraction: float = 0.002
    use_points: bool = True
    point_min_area: int = 25
    point_kernel: int = 5
    point_open_iterations: int = 2
    min_prompt_pixels: int = 5
    overlap_policy: str = "lvm"  # official: overlaps fall back to the LLaVA argmax map


def spatial_entropy(values: np.ndarray, height: int, width: int) -> float:
    """Entropy of connected-region areas of ``values > mean`` (8-connectivity)."""
    binary = (values.reshape(height, width) > values.mean()).astype(int)
    labels, count = connected_components(binary, structure=np.ones((3, 3)))
    if count == 0:
        return 0.0
    areas = np.bincount(labels.ravel())[1:] / (height * width)
    return float(-(areas * np.log(areas)).sum())


def attention_mask(trace: ClassTrace, grid: tuple[int, int], s: SegmentSettings) -> np.ndarray | None:
    """Binary mask from the ``top_k`` lowest-entropy (layer, head) description attentions."""
    if trace.attention is None:
        return None
    keep = np.where(trace.attention_sum > s.attention_min_sum)[0]
    if len(keep) == 0:
        return None
    entropies = np.array([spatial_entropy(trace.attention[i], *grid) for i in keep])
    chosen = keep[np.argsort(entropies, kind="stable")[: s.attention_top_k]]
    mean_map = trace.attention[chosen].mean(0)
    return (mean_map > mean_map.mean()).astype(np.float32)


def normalize_map(distance: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return distance.copy()
    if mode == "median_peak":
        median = float(np.median(distance))
        return np.clip((distance - median) / max(float(distance.max()) - median, 1e-6), 0.0, None)
    raise ValueError(f"unknown normalize mode {mode!r}")


def class_response(trace: ClassTrace, grid: tuple[int, int], s: SegmentSettings) -> np.ndarray | None:
    """Reliable visual-class activation map D_f (flattened grid)."""
    if s.distance not in {"aligned", "shifted", "centered"}:
        raise ValueError(f"unknown distance variant {s.distance!r}")
    distance = {"aligned": trace.distance_aligned, "shifted": trace.distance, "centered": trace.distance_centered}[
        s.distance
    ]
    if s.layer_from >= 0:
        layers = trace.layer_distance_centered if s.distance == "centered" else trace.layer_distance
        if layers is None:
            raise ValueError("layer_from/layer_to need traces made with per-layer maps")
        distance = layers[s.layer_from : s.layer_to + 1].mean(0)
    if distance is None:
        return None
    response = normalize_map(distance.astype(np.float32), s.normalize)
    threshold = min(s.alpha * float(response.max()), s.beta)
    above = (response >= threshold).astype(np.float32)
    mask = attention_mask(trace, grid, s) if s.use_attention else None
    if s.attention_combine == "product":
        response = response * above
        if mask is not None:
            response = response * mask
        return response
    if mask is not None:
        union = np.maximum(mask, above)
        response[union == 0] = 0
    return response


def assign_with_background(maps: torch.Tensor, s: SegmentSettings) -> tuple[torch.Tensor, float]:
    """Label every location with background (0) or a class slot (1..K).

    ``global`` (released code): argmax over [b, D_1..D_K] with one constant
    b = min(alpha * max over all classes, beta). A class whose cosine responses are weaker
    overall (e.g. the generic word "person") then loses to background almost everywhere.
    ``per_class`` (ours): each class competes with its own threshold
    b_c = min(alpha * max D_c, beta); a location goes to argmax_c (D_c - b_c) if positive.
    ``relative`` (ours): maps are scaled by their own maximum and compared with
    ``relative_background``.
    """
    if s.background == "global":
        background = min(s.alpha * float(maps.max()), s.beta)
        logits = torch.cat([torch.full_like(maps[:1], background), maps], dim=0)
        # Softmax with a temperature does not change the argmax; kept for parity with the code.
        return torch.softmax(logits / s.softmax_temperature, dim=0).argmax(0), background
    peaks = maps.flatten(1).max(1).values
    active = peaks > 0
    shape = (-1,) + (1,) * (maps.ndim - 1)
    if s.background == "per_class":
        thresholds = torch.minimum(s.alpha * peaks, torch.full_like(peaks, s.beta))
        scores = maps - thresholds.reshape(shape)
    elif s.background == "relative":
        scores = maps / peaks.clamp_min(1e-8).reshape(shape) - s.relative_background
    else:
        raise ValueError(f"unknown background mode {s.background!r}")
    scores[~active] = float("-inf")
    best, slot = scores.max(0)
    return torch.where(best > 0, slot + 1, torch.zeros_like(slot)), float("nan")


def prototype_labels(
    clip_features: torch.Tensor,
    low_res_labels: torch.Tensor,
    class_count: int,
    out_size: tuple[int, int],
) -> torch.Tensor:
    """VGM prototype map: each location goes to its most similar class/background prototype.

    ``clip_features`` [Nv, C]; ``low_res_labels`` [h, w] with 0 = background and 1..K slots.
    Slots of classes without a prototype keep similarity 0, as in the released code.
    """
    h, w = low_res_labels.shape
    flat = low_res_labels.reshape(-1)
    normed = F.normalize(clip_features, dim=-1)
    sims = torch.zeros(class_count + 1, h * w)
    for slot in range(class_count + 1):
        member = (flat == slot).float()
        if slot > 0 and member.sum() == 0:
            continue
        prototype = (clip_features * member[:, None]).sum(0) / (member.sum() + 1e-8)
        sims[slot] = normed @ F.normalize(prototype, dim=0)
    sims = F.interpolate(sims.reshape(1, -1, h, w), size=out_size, mode="bilinear")[0]
    return sims.argmax(0)


@dataclass
class ClassPrompt:
    slot: int  # 1-based index into the candidate list
    region: np.ndarray  # [H, W] bool, VGM refined mask at image resolution
    points: np.ndarray  # [P, 2] (x, y)
    box: np.ndarray | None  # [4] x0, y0, x1, y1
    response_pixels: int


@dataclass
class ImagePrompts:
    prompts: list[ClassPrompt]
    lvm_labels: np.ndarray  # [H, W] argmax over class responses (no background), 1-based
    coarse_labels: np.ndarray  # [H, W] LLaVA-only prediction with background, 0 = bg
    present: list[str]


def select_classes(trace: ImageTrace, class_names: list[str], s: SegmentSettings, gt: set[str] | None) -> list[str]:
    chosen = []
    for name in class_names:
        entry = trace.classes.get(name)
        if entry is None:
            continue
        if s.qap == "greedy":
            keep = not entry.forced and "yes" in entry.response.lower()
        elif s.qap == "oracle":
            keep = gt is not None and name in gt
        elif s.qap == "p_yes":
            keep = entry.p_yes >= s.qap_threshold
        elif s.qap == "ensemble":  # mean P(yes) over the main and the alternative wordings
            if entry.p_yes_extra is None:
                raise ValueError("qap='ensemble' needs traces made with --extra-queries")
            keep = float(np.mean([entry.p_yes, *entry.p_yes_extra])) >= s.qap_threshold
        else:
            raise ValueError(f"unknown qap mode {s.qap!r}")
        distance = {"aligned": entry.distance_aligned, "shifted": entry.distance, "centered": entry.distance_centered}[s.distance]
        if keep and distance is not None:
            chosen.append(name)
    return chosen


def mask_to_points(mask: np.ndarray, s: SegmentSettings) -> np.ndarray:
    """Official ``mask_to_sam_points`` with one point per region (centroid if inside)."""
    import cv2

    mask_bin = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.point_kernel, s.point_kernel))
    cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=s.point_open_iterations)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= s.point_min_area]
    if not valid:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= s.point_min_area]
    points = []
    for contour in sorted(valid, key=cv2.contourArea, reverse=True):
        (_, _), radius = cv2.minEnclosingCircle(contour)
        delta = max(int(radius * 0.3), 5)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        candidates = [[cx, cy], [cx + delta, cy]]
        inside = [p for p in candidates if cv2.pointPolygonTest(contour, (p[0], p[1]), False) >= 0]
        points.extend(inside[:1])
    return np.asarray(points, dtype=np.int32).reshape(-1, 2)


def mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def evidence_filter(
    trace: ImageTrace, class_names: list[str], present: list[str], s: SegmentSettings
) -> list[str]:
    """Remove unsure positives whose activation map loses almost everywhere."""
    if not present:
        return present
    gh, gw = trace.grid
    maps = torch.zeros(len(class_names), gh * gw)
    for name in present:
        response = class_response(trace.classes[name], trace.grid, s)
        if response is not None:
            maps[class_names.index(name)] = torch.from_numpy(response)
    labels, _ = assign_with_background(maps.reshape(-1, gh, gw), s)
    kept = []
    for name in present:
        area = float((labels == class_names.index(name) + 1).float().mean())
        if trace.classes[name].p_yes < s.filter_p_yes and area <= s.filter_area:
            continue
        kept.append(name)
    return kept


def coverage_regions(
    trace: ImageTrace, class_names: list[str], present: list[str], s: SegmentSettings
) -> dict[str, np.ndarray]:
    """Per present class: image-resolution mask where its deep-layer response is high and wins."""
    height, width = trace.image_size
    gh, gw = trace.grid
    maps = torch.zeros(len(present), gh * gw)
    for k, name in enumerate(present):
        layers = trace.classes[name].layer_distance
        if layers is None:
            raise ValueError("coverage_from/coverage_to need traces made with per-layer maps")
        maps[k] = torch.from_numpy(layers[s.coverage_from : s.coverage_to + 1].mean(0))
    full = F.interpolate(maps.reshape(-1, 1, gh, gw), size=(height, width), mode="bilinear")[:, 0]
    winner = full.argmax(0)
    regions = {}
    for k, name in enumerate(present):
        high = full[k] >= s.coverage_alpha * full[k].max()
        regions[name] = (high & (winner == k)).numpy()
    return regions


def touching_components(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Union of the 8-connected components of ``mask`` that overlap ``seeds``."""
    labels, count = connected_components(mask, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    hit = np.unique(labels[seeds & mask])
    return np.isin(labels, hit[hit > 0])


def build_prompts(
    trace: ImageTrace,
    class_names: list[str],
    s: SegmentSettings,
    gt: set[str] | None = None,
) -> ImagePrompts:
    height, width = trace.image_size
    gh, gw = trace.grid
    present = select_classes(trace, class_names, s, gt)
    if s.filter_p_yes > 0:
        present = evidence_filter(trace, class_names, present, s)
    slots = {name: class_names.index(name) + 1 for name in present}
    empty = np.zeros((height, width), dtype=np.int64)
    if not present:
        return ImagePrompts([], empty, empty.copy(), [])

    maps = torch.zeros(len(class_names), gh * gw)
    for name in present:
        response = class_response(trace.classes[name], trace.grid, s)
        if response is not None:
            maps[slots[name] - 1] = torch.from_numpy(response)
    grid_maps = maps.reshape(-1, 1, gh, gw)
    full_maps = F.interpolate(grid_maps, size=(height, width), mode="bilinear")[:, 0]
    low_res_labels, _ = assign_with_background(grid_maps[:, 0], s)
    full_labels, _ = assign_with_background(full_maps, s)
    lvm_labels = (full_maps.argmax(0) + 1).numpy()
    if s.use_prototype:
        proto = prototype_labels(
            torch.from_numpy(trace.clip_features), low_res_labels, len(class_names), (height, width)
        ).numpy()
    full_np = full_labels.numpy()
    coverage = coverage_regions(trace, class_names, present, s) if s.coverage_from >= 0 else None

    prompts = []
    for name in present:
        slot = slots[name]
        region = full_np == slot
        response_pixels = int(region.sum())
        if s.use_prototype:
            refined = region & (proto == slot)
            if refined.sum() > 0:
                region = refined
        points = mask_to_points(region.astype(np.uint8), s)
        box = mask_to_box(region)
        if coverage is not None and region.any():
            extent = region | touching_components(coverage[name], region)
            box = mask_to_box(extent)
        prompts.append(ClassPrompt(slot, region, points, box, response_pixels))
    return ImagePrompts(prompts, lvm_labels, full_np.astype(np.int64), present)


def _sam_mask(predictor: object, points, labels, box, s: SegmentSettings) -> np.ndarray:
    mask_input = None
    for _ in range(s.sam_iterations):
        masks, _, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=False,
        )
        mask_input = logits
    return masks[0].astype(bool)


def region_components(region: np.ndarray, min_fraction: float) -> list[np.ndarray]:
    """8-connected components of ``region`` covering at least ``min_fraction`` of the image."""
    labels, count = connected_components(region, structure=np.ones((3, 3)))
    minimum = min_fraction * region.size
    return [labels == index for index in range(1, count + 1) if (labels == index).sum() >= minimum]


def run_sam(predictor: object, prompts: ImagePrompts, s: SegmentSettings, shape: tuple[int, int]) -> np.ndarray:
    """SAM refinement and merge; ``predictor`` must already hold the image embedding.

    ``box_mode="single"`` is the released code: one box around the whole class region.
    ``box_mode="components"`` (ours) prompts SAM once per connected region of the class
    (its own box and the positive points inside it) and takes the union, which suits
    classes that appear as several separated instances, e.g. groups of people.
    """
    final = np.zeros(shape, dtype=np.int64)
    flag = np.zeros(shape, dtype=np.int64)
    for prompt in prompts.prompts:
        if len(prompt.points) == 0 or prompt.box is None or prompt.response_pixels <= s.min_prompt_pixels:
            continue
        negatives = [
            other.points for other in prompts.prompts if other.slot != prompt.slot and len(other.points)
        ]
        negative_points = (
            np.concatenate(negatives).astype(np.float32) if negatives and s.use_negatives else None
        )

        def point_prompt(positive: np.ndarray, negative_points=negative_points):
            if not s.use_points:
                return None, None
            points = positive.astype(np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            if negative_points is not None:
                points = np.concatenate([points, negative_points])
                labels = np.concatenate([labels, np.full(len(negative_points), s.negative_label, np.int32)])
            return points, labels

        components = region_components(prompt.region, s.component_min_fraction) if s.box_mode == "components" else []
        if len(components) > 1:
            selected = np.zeros(shape, dtype=bool)
            for component in components:
                inside = component[prompt.points[:, 1], prompt.points[:, 0]]
                positive = prompt.points[inside]
                if len(positive) == 0:
                    ys, xs = np.nonzero(component)
                    positive = np.array([[int(xs.mean()), int(ys.mean())]])
                points, labels = point_prompt(positive)
                box = mask_to_box(component) if s.use_box else None
                selected |= _sam_mask(predictor, points, labels, box, s)
        else:
            points, labels = point_prompt(prompt.points)
            selected = _sam_mask(predictor, points, labels, prompt.box if s.use_box else None, s)
        final[selected] = prompt.slot
        flag[selected] += 1
    if s.overlap_policy == "lvm":
        overlap = flag >= 2
        final[overlap] = prompts.lvm_labels[overlap]
    return final
