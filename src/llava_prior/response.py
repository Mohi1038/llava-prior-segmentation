"""Model-independent implementations of the paper's TVR equations."""

import torch


def cosine_distance_map(
    image_features: torch.Tensor,
    class_feature: torch.Tensor,
    layer_start: int,
    layer_end: int,
) -> torch.Tensor:
    """Average cosine distance over selected layers.

    Args:
        image_features: Tensor shaped ``[layers, visual_tokens, channels]``.
        class_feature: Tensor shaped ``[layers, channels]``.
    Returns:
        Tensor shaped ``[visual_tokens]``. Higher values indicate stronger response,
        following the paper's use of cosine distance (one minus cosine similarity).
    """
    if image_features.ndim != 3 or class_feature.ndim != 2:
        raise ValueError("expected image_features [L,N,D] and class_feature [L,D]")
    if image_features.shape[0] != class_feature.shape[0]:
        raise ValueError("image and class features must have the same layer count")
    selected_images = image_features[layer_start : layer_end + 1]
    selected_class = class_feature[layer_start : layer_end + 1]
    if selected_images.shape[0] == 0:
        raise ValueError("selected layer range is empty")
    similarity = torch.nn.functional.cosine_similarity(
        selected_images, selected_class[:, None, :], dim=-1
    )
    return (1.0 - similarity).mean(dim=0)


def dynamic_threshold(response: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Compute the paper's ``min(alpha * max(response), beta)`` threshold."""
    if response.numel() == 0:
        raise ValueError("response cannot be empty")
    return torch.minimum(alpha * response.max(), response.new_tensor(beta))


def threshold_response(response: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Remove low-confidence visual tokens while preserving response magnitudes."""
    threshold = dynamic_threshold(response, alpha, beta)
    return torch.where(response > threshold, response, torch.zeros_like(response))


def attention_mask(
    attention: torch.Tensor,
    top_k: int,
    spatial_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Average top spatial-attention responses and threshold at their mean.

    ``attention`` is ``[text_tokens, visual_tokens]``. The paper selects responses
    with the lowest spatial entropy. ``spatial_shape`` validates that the visual
    tokens form the expected grid before entropy ranking.
    """
    if attention.ndim != 2 or attention.shape[1] == 0:
        raise ValueError("expected attention [text_tokens, visual_tokens]")
    if spatial_shape is not None:
        if spatial_shape[0] < 1 or spatial_shape[1] < 1:
            raise ValueError("spatial_shape dimensions must be positive")
        if spatial_shape[0] * spatial_shape[1] != attention.shape[1]:
            raise ValueError("spatial_shape does not match visual token count")
    probabilities = attention.clamp_min(0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    selected = attention[torch.topk(-entropy, k=min(top_k, attention.shape[0])).indices]
    visual_attention = selected.mean(dim=0)
    return (visual_attention > visual_attention.mean()).to(attention.dtype)


def reliable_response(response: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the attention-derived binary filter to a thresholded response."""
    if response.shape != mask.shape:
        raise ValueError("response and attention mask must have the same shape")
    return response * mask