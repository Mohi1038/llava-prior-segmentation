import json

import numpy as np
import pytest
from PIL import Image

from llava_prior.evaluate import VOC_CLASS_IDS
from llava_prior.voc_subset import prepare_voc_random_subset, prepare_voc_subset


def _make_voc(root, image_ids, masks):
    (root / "ImageSets" / "Segmentation").mkdir(parents=True)
    (root / "JPEGImages").mkdir()
    (root / "SegmentationClass").mkdir()
    (root / "ImageSets" / "Segmentation" / "val.txt").write_text(
        "\n".join(image_ids) + "\n",
        encoding="utf-8",
    )
    for image_id, mask in zip(image_ids, masks):
        Image.new("RGB", (mask.shape[1], mask.shape[0])).save(
            root / "JPEGImages" / f"{image_id}.jpg"
        )
        Image.fromarray(mask.astype(np.uint8)).save(
            root / "SegmentationClass" / f"{image_id}.png"
        )


def test_prepare_voc_subset_creates_all_class_lists_and_class_map(tmp_path):
    voc_root = tmp_path / "VOC2012"
    mask_small = np.zeros((20, 20), dtype=np.uint8)
    mask_large = np.zeros((20, 20), dtype=np.uint8)
    for class_id in range(1, 21):
        y, x = divmod(class_id - 1, 5)
        mask_small[y * 4, x * 4] = class_id
        mask_large[y * 4 : y * 4 + 2, x * 4 : x * 4 + 2] = class_id
    _make_voc(voc_root, ["small", "large"], [mask_small, mask_large])

    output_dir = tmp_path / "curated"
    manifest = prepare_voc_subset(voc_root, output_dir)

    assert manifest["unique_image_count"] == 2
    assert manifest["class_image_associations"] == 40
    assert set(manifest["selected"]) == set(VOC_CLASS_IDS)
    for class_name in VOC_CLASS_IDS:
        ids = (output_dir / "classes" / class_name / "image_ids.txt").read_text().split()
        assert ids == ["small", "large"]
        assert (output_dir / "classes" / class_name / "images" / "small.jpg").is_file()
        assert (output_dir / "classes" / class_name / "masks" / "small.png").is_file()
    class_map = json.loads((output_dir / "class_map.json").read_text())
    assert len(class_map) == 2
    assert set(class_map["small"]) == set(VOC_CLASS_IDS)
    assert (output_dir / "images" / "small.jpg").is_file()
    assert (output_dir / "images" / "large.jpg").is_file()
    assert (output_dir / "image_ids.txt").read_text().split() == ["large", "small"]


def test_prepare_voc_subset_requires_two_instances_per_class(tmp_path):
    voc_root = tmp_path / "VOC2012"
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 12
    _make_voc(voc_root, ["only_dog"], [mask])

    with pytest.raises(ValueError, match="at least 2"):
        prepare_voc_subset(voc_root, tmp_path / "curated", samples_per_class=1)

    with pytest.raises(ValueError, match="only 0 examples for 'aeroplane'"):
        prepare_voc_subset(voc_root, tmp_path / "curated", samples_per_class=2)


def test_prepare_voc_random_subset_is_seeded_and_does_not_read_annotations(tmp_path):
    voc_root = tmp_path / "VOC2012"
    split = voc_root / "ImageSets" / "Segmentation"
    images = voc_root / "JPEGImages"
    split.mkdir(parents=True)
    images.mkdir()
    image_ids = [f"image_{index:02d}" for index in range(8)]
    (split / "val.txt").write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    for image_id in image_ids:
        Image.new("RGB", (4, 4)).save(images / f"{image_id}.jpg")

    first_dir = tmp_path / "random-a"
    second_dir = tmp_path / "random-b"
    first = prepare_voc_random_subset(voc_root, first_dir, image_count=4, seed=42)
    second = prepare_voc_random_subset(voc_root, second_dir, image_count=4, seed=42)

    assert first["image_ids"] == second["image_ids"]
    assert first["candidate_classes"] == list(VOC_CLASS_IDS)
    assert first["candidate_policy"] == "Query all 20 VOC foreground classes for every selected image."
    assert not (first_dir / "class_map.json").exists()
    assert all((first_dir / "images" / f"{image_id}.jpg").is_file() for image_id in first["image_ids"])
    assert not (voc_root / "SegmentationClass").exists()


def test_prepare_voc_random_subset_validates_count_and_output_directory(tmp_path):
    voc_root = tmp_path / "VOC2012"
    split = voc_root / "ImageSets" / "Segmentation"
    images = voc_root / "JPEGImages"
    split.mkdir(parents=True)
    images.mkdir()
    (split / "val.txt").write_text("one\n", encoding="utf-8")
    Image.new("RGB", (4, 4)).save(images / "one.jpg")

    with pytest.raises(ValueError, match="image_count must be between"):
        prepare_voc_random_subset(voc_root, tmp_path / "too-many", image_count=2)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").touch()
    with pytest.raises(FileExistsError, match="must be empty"):
        prepare_voc_random_subset(voc_root, nonempty, image_count=1)
