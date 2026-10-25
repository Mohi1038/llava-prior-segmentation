import torch

from llava_prior.config import PriorSegmentationConfig
from llava_prior.pipeline import ClassPrediction, TraceBatch, merge_class_masks, run_pipeline


class FakeLlava:
    def answer_with_trace(self, image, query):
        trace = TraceBatch(
            image_features=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]),
            class_feature=torch.tensor([[1.0, 0.0]]),
            description_attention=torch.ones(1, 4),
            vision_features=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            spatial_shape=(2, 2),
            image_size=(8, 8),
        )
        return "Yes, it contains a bird. It is visible.", trace


class FakeSam:
    def __init__(self):
        self.points = None
        self.box = None
        self.mask_prior = None

    def predict(self, image, points, box, mask_prior):
        self.points = points
        self.box = box
        self.mask_prior = mask_prior
        return torch.zeros(8, 8, dtype=torch.bool)


def test_pipeline_passes_image_coordinate_prompts_to_sam():
    sam = FakeSam()
    predictions = run_pipeline(
        image=object(),
        class_names=["bird"],
        llava=FakeLlava(),
        sam=sam,
        config=PriorSegmentationConfig(
            layer_start=0,
            layer_end=0,
            attention_top_k=1,
            min_component_area=1,
            morphological_kernel=1,
        ),
    )
    assert len(predictions) == 1
    assert sam.points.shape[1] == 2
    assert sam.box.shape == (4,)
    assert sam.mask_prior.shape == (8, 8)
    assert sam.mask_prior.dtype == torch.bool
    assert torch.equal(predictions[0].sam_prior_mask, sam.mask_prior)
    assert torch.all(sam.points >= 0)
    assert torch.all(sam.points <= 7)


def test_pipeline_skips_duplicate_candidate_classes():
    sam = FakeSam()
    predictions = run_pipeline(
        image=object(),
        class_names=["bird", " bird "],
        llava=FakeLlava(),
        sam=sam,
        config=PriorSegmentationConfig(
            layer_start=0,
            layer_end=0,
            attention_top_k=1,
            min_component_area=1,
            morphological_kernel=1,
        ),
    )
    assert len(predictions) == 1


def test_merge_class_masks_uses_background_zero_and_first_class_wins():
    base = ClassPrediction(None, torch.empty(0), torch.empty(0), torch.tensor([[1, 0], [0, 1]]))
    overlap = ClassPrediction(None, torch.empty(0), torch.empty(0), torch.tensor([[1, 1], [0, 0]]))
    merged = merge_class_masks([base, overlap])
    assert torch.equal(merged, torch.tensor([[1, 2], [0, 1]]))


def test_merge_empty_class_masks_preserves_requested_image_size():
    merged = merge_class_masks([], (8, 10))
    assert merged.shape == (8, 10)
    assert not merged.any()