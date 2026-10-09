"""Vision-prototype computations used by the VGM stage."""

import torch


def semantic_guidance(response_maps: torch.Tensor) -> torch.Tensor:
    """Return the winning class index at every visual-token location."""
    if response_maps.ndim != 2 or response_maps.shape[0] < 1:
        raise ValueError("expected response_maps [classes, visual_tokens]")
    return response_maps.argmax(dim=0)


def prototype_mask(
    vision_features: torch.Tensor,
    guidance: torch.Tensor,
    class_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compare each vision token with class and non-class prototypes."""
    if vision_features.ndim != 2 or guidance.ndim != 1:
        raise ValueError("expected vision_features [visual_tokens, channels] and guidance [N]")
    if vision_features.shape[0] != guidance.shape[0]:
        raise ValueError("guidance and vision features must have the same token count")
    positive = guidance == class_index
    negative = ~positive
    positive_prototype = vision_features[positive].sum(dim=0) / (positive.sum() + epsilon)
    negative_prototype = vision_features[negative].sum(dim=0) / (negative.sum() + epsilon)
    positive_similarity = torch.nn.functional.cosine_similarity(
        vision_features, positive_prototype.unsqueeze(0), dim=-1
    )
    negative_similarity = torch.nn.functional.cosine_similarity(
        vision_features, negative_prototype.unsqueeze(0), dim=-1
    )
    return (positive_similarity > negative_similarity).to(vision_features.dtype)


def refined_mask(prototype: torch.Tensor, guidance: torch.Tensor, class_index: int) -> torch.Tensor:
    """Keep prototype-positive tokens only where the class wins semantic guidance."""
    if prototype.shape != guidance.shape:
        raise ValueError("prototype and guidance must have the same shape")
    return prototype * (guidance == class_index).to(prototype.dtype)