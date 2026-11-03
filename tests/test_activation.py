import numpy as np
import torch

from llava_prior.activation import (
    SegmentSettings,
    assign_with_background,
    build_prompts,
    class_response,
    mask_to_box,
    mask_to_points,
    prototype_labels,
    spatial_entropy,
)
from llava_prior.trace import ClassTrace, ImageTrace, class_token_indices, square_resize
from llava_prior.voc import confusion, summarize


def test_class_tokens_follow_first_sentence_and_repeats():
    # "Yes, it contains a cat. The cat is black." without the first token ("Yes")
    tokens = [",", "it", "contains", "a", "cat", ".", "The", "cat", "is", "black", "."]
    indices, first_end = class_token_indices(tokens)
    assert first_end == 5
    assert indices == [4, 7]


def test_multi_token_class_names_are_all_selected():
    tokens = [",", "it", "contains", "a", "d", "ining", "table", "."]
    indices, _ = class_token_indices(tokens)
    assert [tokens[i] for i in indices] == ["d", "ining", "table"]


def test_spatial_entropy_prefers_one_compact_region():
    compact = np.zeros((4, 4))
    compact[:2, :2] = 1
    scattered = np.zeros((4, 4))
    scattered[0, 0] = scattered[0, 3] = scattered[3, 0] = scattered[3, 3] = 1
    assert spatial_entropy(compact.ravel(), 4, 4) < spatial_entropy(scattered.ravel(), 4, 4)


def _trace(distance, attention=None):
    attention_sum = None if attention is None else attention.sum(-1)
    return ClassTrace("cat", "Yes, it contains a cat. A cat.", 0.9, False, distance, distance, attention, attention_sum)


def test_union_keeps_attended_low_scores_and_product_drops_them():
    distance = np.array([0.30, 0.10, 0.05, 0.25], dtype=np.float32)
    attention = np.array([[0.0, 0.5, 0.0, 0.0]], dtype=np.float32)  # attends token 1 only
    union = class_response(_trace(distance, attention), (2, 2), SegmentSettings(attention_min_sum=0.1))
    product = class_response(
        _trace(distance, attention), (2, 2), SegmentSettings(attention_min_sum=0.1, attention_combine="product")
    )
    # threshold = min(0.75 * 0.30, 0.19) = 0.19 -> tokens 0 and 3 pass
    np.testing.assert_allclose(union, [0.30, 0.10, 0.0, 0.25])
    np.testing.assert_allclose(product, [0.0, 0.0, 0.0, 0.0])


def test_background_is_constant_threshold():
    maps = torch.tensor([[[0.5, 0.1]], [[0.0, 0.0]]])
    labels, background = assign_with_background(maps, SegmentSettings())
    assert background == 0.19
    assert labels.tolist() == [[1, 0]]


def test_prototype_labels_follow_feature_similarity():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    low_res = torch.tensor([[1, 0], [0, 0]])  # one class token, rest background
    labels = prototype_labels(features, low_res, class_count=1, out_size=(2, 2))
    assert labels.tolist() == [[1, 1], [0, 0]]


def test_points_and_box_from_mask():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 5:25] = 1
    points = mask_to_points(mask, SegmentSettings())
    assert points.shape == (1, 2)
    assert 5 <= points[0, 0] < 25 and 10 <= points[0, 1] < 30
    np.testing.assert_array_equal(mask_to_box(mask), [5, 10, 24, 29])


def test_build_prompts_ignores_forced_traces_under_greedy_qap():
    grid = (2, 2)
    natural = _trace(np.array([0.3, 0.3, 0.0, 0.0], dtype=np.float32))
    forced = ClassTrace("dog", "Yes, it contains a dog.", 0.1, True, natural.distance, natural.distance)
    trace = ImageTrace("x", (20, 20), grid, np.eye(4, 8, dtype=np.float32), {"cat": natural, "dog": forced})
    prompts = build_prompts(trace, ["cat", "dog"], SegmentSettings())
    assert prompts.present == ["cat"]
    oracle = build_prompts(trace, ["cat", "dog"], SegmentSettings(qap="oracle"), gt={"dog"})
    assert oracle.present == ["dog"]


def test_square_resize_stretches_short_side():
    from PIL import Image

    assert square_resize(Image.new("RGB", (50, 30))).size == (50, 50)


def test_voc21_summary_includes_background_and_ignores_void():
    target = np.array([[0, 1], [255, 1]])
    pred = np.array([[0, 1], [1, 0]])
    result = summarize(confusion(pred, target, 3), ("background", "a", "b"))
    assert result["IoU"]["background"] == 0.5
    assert result["IoU"]["a"] == 0.5
    assert result["IoU"]["b"] is None
    assert result["mIoU"] == 0.5


def test_evidence_filter_drops_unsure_classes_that_lose_everywhere():
    strong = ClassTrace("cat", "Yes, it contains a cat.", 0.95, False, None, np.array([0.5, 0.5, 0.0, 0.0], np.float32))
    weak = ClassTrace("dog", "Yes, it contains a dog.", 0.55, False, None, np.array([0.3, 0.3, 0.0, 0.0], np.float32))
    trace = ImageTrace("x", (20, 20), (2, 2), np.eye(4, 8, dtype=np.float32), {"cat": strong, "dog": weak})
    kept = build_prompts(trace, ["cat", "dog"], SegmentSettings(filter_p_yes=0.7, filter_area=0.02)).present
    assert kept == ["cat"]
    sure = build_prompts(trace, ["cat", "dog"], SegmentSettings(filter_p_yes=0.5, filter_area=0.02)).present
    assert sure == ["cat", "dog"]


def test_per_class_background_lets_weak_classes_compete():
    # class 1 strong everywhere it fires; class 2 weak but clearly peaked at its own location
    maps = torch.tensor([[[0.40, 0.00, 0.0]], [[0.00, 0.18, 0.0]]])
    global_labels, _ = assign_with_background(maps, SegmentSettings())
    per_class, _ = assign_with_background(maps, SegmentSettings(background="per_class"))
    relative, _ = assign_with_background(maps, SegmentSettings(background="relative"))
    assert global_labels.tolist() == [[1, 0, 0]]  # 0.18 < global background 0.19
    assert per_class.tolist() == [[1, 2, 0]]
    assert relative.tolist() == [[1, 2, 0]]


class _FakePredictor:
    """Returns the prompt box as the mask, so the merge logic can be checked without SAM."""

    def __init__(self, shape):
        self.shape = shape
        self.boxes = []

    def predict(self, point_coords, point_labels, box, mask_input, multimask_output):
        self.boxes.append(box)
        mask = np.zeros(self.shape, dtype=bool)
        x0, y0, x1, y1 = box.astype(int)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        return mask[None], np.ones(1), np.zeros((1, 256, 256), np.float32)


def test_component_boxes_prompt_each_instance_separately():
    from llava_prior.activation import ClassPrompt, ImagePrompts, run_sam

    region = np.zeros((20, 40), dtype=bool)
    region[2:8, 2:8] = True
    region[12:18, 30:38] = True
    points = np.array([[4, 4], [33, 14]])
    prompt = ClassPrompt(1, region, points, mask_to_box(region), int(region.sum()))
    prompts = ImagePrompts([prompt], np.ones((20, 40), np.int64), np.zeros((20, 40), np.int64), ["person"])
    single = run_sam(_FakePredictor((20, 40)), prompts, SegmentSettings(sam_iterations=1), (20, 40))
    split = run_sam(
        _FakePredictor((20, 40)), prompts, SegmentSettings(sam_iterations=1, box_mode="components"), (20, 40)
    )
    assert single.sum() == 16 * 36  # one box spanning both instances
    assert split.sum() == 36 + 48  # union of the two instance boxes


def test_layer_range_averages_stored_per_layer_maps():
    layers = np.zeros((33, 4), dtype=np.float32)
    layers[3] = [0.4, 0.0, 0.0, 0.0]
    layers[4] = [0.0, 0.4, 0.0, 0.0]
    trace = ClassTrace("cat", "Yes.", 0.9, False, None, layers[7:14].mean(0), None, None, layers)
    one = class_response(trace, (2, 2), SegmentSettings(layer_from=3, layer_to=3))
    two = class_response(trace, (2, 2), SegmentSettings(layer_from=3, layer_to=4))
    np.testing.assert_allclose(one, [0.4, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(two, [0.2, 0.2, 0.0, 0.0])


def test_conversation_templates_follow_official_modes():
    from llava_prior.trace import build_prompt, conversation_template

    assert conversation_template("llava-hf/llava-v1.6-mistral-7b-hf") == "mistral_instruct"
    assert conversation_template("llava-hf/llava-1.5-7b-hf") == "llava_v1"
    before, after = build_prompt("dog", "mistral_instruct")
    assert before == "[INST] " and after.startswith("\nDoes the image have the class of a dog ?")
    assert after.endswith(" [/INST]")
    before, after = build_prompt("dog")
    assert before.endswith(" USER: ") and after.endswith(" ASSISTANT:")


def test_ensemble_qap_averages_wordings():
    from llava_prior.activation import select_classes

    yes_map = np.array([0.5, 0.5, 0.0, 0.0], np.float32)
    cat = ClassTrace("cat", "Yes, it contains a cat.", 0.9, False, None, yes_map, p_yes_extra=[0.1, 0.1, 0.2])
    dog = ClassTrace("dog", "No.", 0.4, True, None, yes_map, p_yes_extra=[0.9, 0.8, 0.9])
    trace = ImageTrace("x", (20, 20), (2, 2), np.eye(4, 8, dtype=np.float32), {"cat": cat, "dog": dog})
    assert select_classes(trace, ["cat", "dog"], SegmentSettings(), None) == ["cat"]
    assert select_classes(trace, ["cat", "dog"], SegmentSettings(qap="ensemble"), None) == ["dog"]


def test_touching_components_keeps_only_regions_overlapping_seeds():
    from llava_prior.activation import touching_components

    mask = np.zeros((10, 10), bool)
    mask[0:4, 0:4] = True  # touches the seed
    mask[7:9, 7:9] = True  # separate component
    seeds = np.zeros((10, 10), bool)
    seeds[1, 1] = True
    kept = touching_components(mask, seeds)
    assert kept[0:4, 0:4].all() and not kept[7:9, 7:9].any()


def test_median_peak_normalization_is_scale_free():
    from llava_prior.activation import normalize_map

    llava_like = np.array([0.05, 0.09, 0.09, 0.29], np.float32)
    qwen_like = 0.4 + 0.7 * llava_like  # same shape, different cosine level
    a, b = normalize_map(llava_like, "median_peak"), normalize_map(qwen_like, "median_peak")
    assert np.allclose(a, b, atol=1e-6) and a.max() == 1.0 and a.min() == 0.0
