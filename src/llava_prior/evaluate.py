"""Evaluate indexed prediction masks against indexed ground-truth masks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import mean_iou


def load_indexed_mask(path: Path) -> np.ndarray:
    """Load a PNG ground-truth mask or NumPy prediction mask as integer labels."""
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"expected an indexed 2D mask: {path}")
    return mask.astype(np.int64, copy=False)


def evaluate_files(
    prediction_dir: Path,
    target_dir: Path,
    class_names: list[str],
    ignore_index: int | None = 255,
) -> dict[str, object]:
    """Evaluate matching ``semantic_mask.npy`` and target files by stem."""
    prediction_paths = sorted(prediction_dir.glob("*.npy"))
    if not prediction_paths:
        raise ValueError(f"no .npy predictions found in {prediction_dir}")
    per_image: list[dict[str, object]] = []
    class_indices = range(1, len(class_names) + 1)
    for prediction_path in prediction_paths:
        if prediction_path.name == "semantic_mask.npy":
            stem = prediction_path.stem
        else:
            continue
        if target_dir.is_file():
            target_path = target_dir
        else:
            target_path = target_dir / f"{stem}.png"
            if not target_path.exists():
                target_path = target_dir / f"{stem}.npy"
        if not target_path.exists():
            raise FileNotFoundError(f"no target mask for {prediction_path.name}")
        prediction = load_indexed_mask(prediction_path)
        target = load_indexed_mask(target_path)
        score, class_scores = mean_iou(prediction, target, class_indices, ignore_index)
        per_image.append({"image": stem, "mIoU": score, "per_class_IoU": class_scores})
    if not per_image:
        raise ValueError("prediction directory contains no semantic_mask.npy files")
    scores = [float(item["mIoU"]) for item in per_image]
    return {"mIoU": float(np.mean(scores)), "classes": class_names, "images": per_image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()
    result = evaluate_files(args.predictions, args.targets, args.classes, args.ignore_index)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()