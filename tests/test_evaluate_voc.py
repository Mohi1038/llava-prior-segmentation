import json

import numpy as np
import pytest
from PIL import Image

from llava_prior.evaluate import VOC_CLASS_IDS, evaluate_voc


def _write_voc_sample(root, output_root, image_id, target, prediction, predicted_classes):
    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / "SegmentationClass").mkdir(parents=True, exist_ok=True)
    output = output_root / image_id
    output.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (target.shape[1], target.shape[0])).save(
        root / "JPEGImages" / f"{image_id}.jpg"
    )
    Image.fromarray(target.astype(np.uint8), mode="L").save(
        root / "SegmentationClass" / f"{image_id}.png"
    )
    np.save(output / "semantic_mask.npy", prediction.astype(np.int64))
    (output / "predictions.json").write_text(
        json.dumps([{"class": name, "mask": f"{name}.npy"} for name in predicted_classes]),
        encoding="utf-8",
    )


def test_voc_class_ids_cover_official_foreground_classes():
    assert len(VOC_CLASS_IDS) == 20
    assert VOC_CLASS_IDS["dog"] == 12
    assert VOC_CLASS_IDS["person"] == 15
    assert set(VOC_CLASS_IDS.values()) == set(range(1, 21))


def test_evaluate_voc_maps_prediction_order_and_ignores_other_classes(tmp_path):
    voc_root = tmp_path / "VOC2012"
    output_root = tmp_path / "predictions"
    image_id = "2007_000001"
    target = np.array([[12, 12, 8, 7, 0, 255]], dtype=np.uint8)
    # Saved prediction order is dog, cat; requested/evaluation order is cat, dog.
    prediction = np.array([[1, 0, 2, 2, 0, 2]], dtype=np.uint8)
    _write_voc_sample(voc_root, output_root, image_id, target, prediction, ["dog", "cat"])
    split = tmp_path / "split.txt"
    split.write_text(f"{image_id}\n", encoding="utf-8")

    result = evaluate_voc(output_root, voc_root, ["cat", "dog"], split)

    assert result["image_count"] == 1
    assert result["classes"] == ["cat", "dog"]
    assert result["per_class_IoU"]["cat"] == pytest.approx(1.0)
    assert result["per_class_IoU"]["dog"] == pytest.approx(0.5)
    assert result["mIoU"] == pytest.approx(0.75)
    assert result["images"][0]["per_class_IoU"]["cat"] == pytest.approx(1.0)


def test_evaluate_voc_uses_dataset_level_intersection_over_union(tmp_path):
    voc_root = tmp_path / "VOC2012"
    output_root = tmp_path / "predictions"
    _write_voc_sample(
        voc_root,
        output_root,
        "image_a",
        np.array([[12, 12, 0]], dtype=np.uint8),
        np.array([[1, 0, 0]], dtype=np.uint8),
        ["dog"],
    )
    _write_voc_sample(
        voc_root,
        output_root,
        "image_b",
        np.array([[12, 0, 0]], dtype=np.uint8),
        np.array([[1, 0, 0]], dtype=np.uint8),
        ["dog"],
    )
    split = tmp_path / "split.txt"
    split.write_text("image_a\nimage_b\n", encoding="utf-8")

    result = evaluate_voc(output_root, voc_root, ["dog"], split)

    assert result["per_class_IoU"]["dog"] == pytest.approx(2 / 3)
    assert result["images"][0]["mIoU"] == pytest.approx(0.5)
    assert result["images"][1]["mIoU"] == pytest.approx(1.0)


def test_evaluate_voc_requires_supported_unique_classes(tmp_path):
    with pytest.raises(ValueError, match="unsupported VOC classes"):
        evaluate_voc(tmp_path, tmp_path, ["traffic light"])
    with pytest.raises(ValueError, match="must be unique"):
        evaluate_voc(tmp_path, tmp_path, ["dog", "Dog"])
    with pytest.raises(ValueError, match="must not overlap"):
        evaluate_voc(tmp_path, tmp_path, ["dog"], ignore_index=1)


def test_evaluate_voc_requires_matching_prediction_dimensions(tmp_path):
    voc_root = tmp_path / "VOC2012"
    output_root = tmp_path / "predictions"
    _write_voc_sample(
        voc_root,
        output_root,
        "image_a",
        np.array([[12, 0]], dtype=np.uint8),
        np.array([[1]], dtype=np.uint8),
        ["dog"],
    )
    split = tmp_path / "split.txt"
    split.write_text("image_a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dimensions differ"):
        evaluate_voc(output_root, voc_root, ["dog"], split)


def test_evaluate_voc_can_require_all_inference_candidates(tmp_path):
    voc_root = tmp_path / "VOC2012"
    output_root = tmp_path / "predictions"
    image_id = "image_a"
    _write_voc_sample(
        voc_root,
        output_root,
        image_id,
        np.array([[12, 0]], dtype=np.uint8),
        np.array([[1, 0]], dtype=np.uint8),
        ["dog"],
    )
    split = tmp_path / "split.txt"
    split.write_text(f"{image_id}\n", encoding="utf-8")
    metadata = output_root / image_id / "run_metadata.json"
    metadata.write_text(json.dumps({"candidate_classes": ["dog"]}), encoding="utf-8")

    result = evaluate_voc(
        output_root,
        voc_root,
        ["dog"],
        split,
        require_all_candidates=True,
    )
    assert result["mIoU"] == pytest.approx(1.0)

    metadata.write_text(json.dumps({"candidate_classes": ["cat"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match evaluation classes"):
        evaluate_voc(
            output_root,
            voc_root,
            ["dog"],
            split,
            require_all_candidates=True,
        )
