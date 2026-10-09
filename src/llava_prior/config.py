from dataclasses import dataclass, field


@dataclass(frozen=True)
class PriorSegmentationConfig:
    """Inference settings matching the paper's Vicuna-7B default."""

    layer_start: int = 7
    layer_end: int = 13
    threshold_alpha: float = 0.75
    threshold_beta: float = 0.19
    attention_top_k: int = 5
    morphological_kernel: int = 5
    min_component_area: int = 16
    epsilon: float = 1e-6
    class_prompt: str = (
        "Does the image have the class of a {class_name}? Answer yes or no to this "
        "question. If no, just say 'No.' If yes, the answer has to be two sentences, "
        "the first sentence is 'Yes, it contains a {class_name}.' The second sentence "
        "is to describe {class_name}."
    )
    candidate_classes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.layer_start < 0 or self.layer_end < self.layer_start:
            raise ValueError("layer range must satisfy 0 <= layer_start <= layer_end")
        if not 0 < self.threshold_alpha <= 1:
            raise ValueError("threshold_alpha must be in (0, 1]")
        if self.threshold_beta <= 0:
            raise ValueError("threshold_beta must be positive")
        if self.attention_top_k < 1:
            raise ValueError("attention_top_k must be positive")
        if self.morphological_kernel < 1 or self.morphological_kernel % 2 == 0:
            raise ValueError("morphological_kernel must be a positive odd number")