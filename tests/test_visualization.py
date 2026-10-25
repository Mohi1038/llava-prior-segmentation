import json

import numpy as np
from PIL import Image

from llava_prior.visualization import save_prediction_visualization


def test_visualization_contains_overlays_and_marks_empty_masks(tmp_path):
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    person = np.zeros((8, 8), dtype=bool)
    person[2:6, 2:6] = True
    dog = np.zeros((8, 8), dtype=bool)
    np.save(tmp_path / "person.npy", person)
    np.save(tmp_path / "dog.npy", dog)
    np.save(tmp_path / "semantic_mask.npy", person.astype(np.int64))
    (tmp_path / "predictions.json").write_text(
        json.dumps(
            [
                {"class": "person", "mask": "person.npy"},
                {"class": "dog", "mask": "dog.npy"},
            ]
        ),
        encoding="utf-8",
    )

    result = save_prediction_visualization(tmp_path / "image.png", tmp_path)

    with Image.open(result) as visualization:
        assert visualization.size == (2008, 2310)
        assert visualization.getpixel((512, 1750)) != (100, 100, 100)
        assert visualization.getpixel((1496, 1750)) == (100, 100, 100)


def test_visualization_supports_no_detected_classes(tmp_path):
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    np.save(tmp_path / "semantic_mask.npy", np.zeros((8, 8), dtype=np.int64))
    (tmp_path / "predictions.json").write_text("[]", encoding="utf-8")

    result = save_prediction_visualization(tmp_path / "image.png", tmp_path)

    with Image.open(result) as visualization:
        assert visualization.size == (2008, 1220)
