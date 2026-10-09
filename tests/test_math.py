import numpy as np
import pytest
import torch

from llava_prior.metrics import intersection_over_union, mean_iou
from llava_prior.prototypes import prototype_mask, refined_mask, semantic_guidance
from llava_prior.response import (
    attention_mask,
    cosine_distance_map,
    dynamic_threshold,
    reliable_response,
    threshold_response,
)


def test_cosine_distance_and_threshold_are_deterministic():
    image = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    class_feature = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    response = cosine_distance_map(image, class_feature, 0, 1)
    assert torch.allclose(response, torch.tensor([0.0, 1.0]))
    assert dynamic_threshold(response, 0.75, 0.19).item() == pytest.approx(0.19)
    assert torch.allclose(threshold_response(response, 0.75, 0.19), torch.tensor([0.0, 1.0]))


def test_attention_mask_and_reliable_response_filter_tokens():
    attention = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    mask = attention_mask(attention, top_k=2)
    response = torch.tensor([0.4, 0.5, 0.6])
    assert mask.shape == response.shape
    assert torch.equal(reliable_response(response, mask), response * mask)


def test_prototype_guidance_and_refinement():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    response_maps = torch.tensor([[1.0, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.8]])
    guidance = semantic_guidance(response_maps)
    mask = prototype_mask(features, guidance, class_index=0)
    refined = refined_mask(mask, guidance, class_index=0)
    assert torch.equal(guidance, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(refined, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_iou_ignores_void_pixels_and_reports_absent_classes():
    prediction = torch.tensor([[1, 1, 0], [2, 0, 255]]).numpy()
    target = torch.tensor([[1, 2, 0], [2, 0, 255]]).numpy()
    assert intersection_over_union(prediction, target, 1) == pytest.approx(1 / 2)
    score, per_class = mean_iou(prediction, target, [1, 2, 3])
    assert score == pytest.approx((1 / 2 + 1 / 2) / 2)
    assert np.isnan(per_class[3])