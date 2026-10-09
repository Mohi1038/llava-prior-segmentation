"""Orchestration interfaces for a complete FSeg-LLaVA inference pass."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .config import PriorSegmentationConfig
from .prompts import masks_to_prompts
from .prototypes import prototype_mask, refined_mask, semantic_guidance
from .qap import ClassObservation, build_class_query, parse_class_response
from .response import attention_mask, cosine_distance_map, reliable_response, threshold_response


@dataclass(frozen=True)
class TraceBatch:
    """Features and attention captured from one LLaVA class query."""

    image_features: torch.Tensor
    class_feature: torch.Tensor
    description_attention: torch.Tensor
    vision_features: torch.Tensor
    spatial_shape: tuple[int, int]
    image_size: tuple[int, int]


class LlavaBackend(Protocol):
    def answer_with_trace(self, image: object, query: str) -> tuple[str, TraceBatch]: ...


class SamBackend(Protocol):
    def predict(self, image: object, points: torch.Tensor, box: torch.Tensor | None) -> torch.Tensor: ...


@dataclass(frozen=True)
class ClassPrediction:
    observation: ClassObservation
    response_map: torch.Tensor
    refined_token_mask: torch.Tensor
    sam_mask: torch.Tensor


def merge_class_masks(predictions: Sequence[ClassPrediction]) -> torch.Tensor:
    """Merge independent class masks into an indexed semantic mask.

    Background is encoded as zero and detected classes use their input order plus
    one. When classes overlap, the first class in the candidate list wins, making
    the policy deterministic and reproducible.
    """
    if not predictions:
        return torch.zeros((0, 0), dtype=torch.int64)
    shape = predictions[0].sam_mask.shape
    if len(shape) != 2:
        raise ValueError("SAM masks must have shape [height, width]")
    merged = torch.zeros(shape, dtype=torch.int64)
    for class_index, prediction in enumerate(predictions, start=1):
        if prediction.sam_mask.shape != shape:
            raise ValueError("all SAM masks must have the same shape")
        available = prediction.sam_mask.to(torch.bool) & (merged == 0)
        merged[available] = class_index
    return merged


def run_pipeline(
    image: object,
    class_names: Sequence[str],
    llava: LlavaBackend,
    sam: SamBackend,
    config: PriorSegmentationConfig,
) -> list[ClassPrediction]:
    """Run QAP, TVR, VGM, and SAM for all classes detected by QAP."""
    traces: list[tuple[ClassObservation, TraceBatch]] = []
    seen_classes: set[str] = set()
    for class_name in class_names:
        normalized_class = class_name.strip()
        if normalized_class in seen_classes:
            continue
        seen_classes.add(normalized_class)
        query = build_class_query(class_name, config.class_prompt)
        response, trace = llava.answer_with_trace(image, query)
        observation = parse_class_response(class_name, response)
        if observation.present:
            traces.append((observation, trace))
    if not traces:
        return []

    response_maps = []
    for _, trace in traces:
        distance = cosine_distance_map(
            trace.image_features,
            trace.class_feature,
            config.layer_start,
            config.layer_end,
        )
        filtered = threshold_response(distance, config.threshold_alpha, config.threshold_beta)
        attention = attention_mask(
            trace.description_attention,
            config.attention_top_k,
            trace.spatial_shape,
        )
        response_maps.append(reliable_response(filtered, attention))
    guidance = semantic_guidance(torch.stack(response_maps))

    predictions = []
    for class_index, ((observation, trace), distance) in enumerate(zip(traces, response_maps)):
        prototype = prototype_mask(trace.vision_features, guidance, class_index, config.epsilon)
        refined = refined_mask(prototype, guidance, class_index)
        height, width = trace.spatial_shape
        token_mask = refined.reshape(height, width).detach().cpu().numpy().astype(np.uint8)
        image_height, image_width = trace.image_size
        import cv2

        image_mask = cv2.resize(
            token_mask,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        prompts = masks_to_prompts(
            image_mask,
            image_height,
            image_width,
            min_component_area=config.min_component_area,
            morphological_kernel=config.morphological_kernel,
        )
        points = torch.from_numpy(prompts.points)
        box = torch.from_numpy(prompts.box) if prompts.box is not None else None
        sam_mask = sam.predict(image, points, box)
        predictions.append(ClassPrediction(observation, distance, refined, sam_mask))
    return predictions