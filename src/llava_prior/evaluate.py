"""Evaluate indexed prediction masks against indexed ground-truth masks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import mean_iou

VOC_CLASS_IDS = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}


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


def _voc_target_to_candidate_indices(
    target: np.ndarray,
    class_names: list[str],
    ignore_index: int,
) -> np.ndarray:
    mapped = np.full(target.shape, ignore_index, dtype=np.int64)
    mapped[target == 0] = 0
    for candidate_index, class_name in enumerate(class_names, start=1):
        mapped[target == VOC_CLASS_IDS[class_name]] = candidate_index
    mapped[target == 255] = ignore_index
    return mapped


def _prediction_to_candidate_indices(
    prediction: np.ndarray,
    prediction_classes: list[str],
    candidate_classes: list[str],
) -> np.ndarray:
    if prediction.ndim != 2:
        raise ValueError("VOC semantic predictions must be indexed 2D masks")
    if prediction.size and (prediction.min() < 0 or prediction.max() > len(prediction_classes)):
        raise ValueError("semantic prediction contains labels absent from predictions.json")
    mapped = np.zeros(prediction.shape, dtype=np.int64)
    candidate_indices = {name: index for index, name in enumerate(candidate_classes, start=1)}
    for prediction_index, raw_class_name in enumerate(prediction_classes, start=1):
        class_name = raw_class_name.strip().lower()
        if class_name not in candidate_indices:
            raise ValueError(
                f"predicted class {class_name!r} is not in the requested candidate classes"
            )
        mapped[prediction == prediction_index] = candidate_indices[class_name]
    return mapped


def evaluate_voc(
    prediction_dir: Path,
    voc_root: Path,
    class_names: list[str],
    split_file: Path | None = None,
    ignore_index: int = 255,
    require_all_candidates: bool = False,
) -> dict[str, object]:
    """Evaluate batch outputs against VOC's native segmentation labels.

    VOC classes not listed in ``class_names`` are ignored rather than treated as
    background. Dataset-level IoU is accumulated over all valid pixels before
    computing the mean across requested classes.
    """
    normalized_classes = [name.strip().lower() for name in class_names]
    if not normalized_classes or any(not name for name in normalized_classes):
        raise ValueError("at least one non-empty VOC class name is required")
    if len(set(normalized_classes)) != len(normalized_classes):
        raise ValueError("candidate class names must be unique")
    unsupported = [name for name in normalized_classes if name not in VOC_CLASS_IDS]
    if unsupported:
        raise ValueError(f"unsupported VOC classes: {unsupported}")
    if ignore_index in range(len(normalized_classes) + 1):
        raise ValueError("ignore_index must not overlap background or candidate class indices")

    voc_root = Path(voc_root)
    prediction_dir = Path(prediction_dir)
    split_path = Path(split_file) if split_file is not None else (
        voc_root / "ImageSets" / "Segmentation" / "val.txt"
    )
    if not split_path.is_file():
        raise FileNotFoundError(f"VOC split file does not exist: {split_path}")
    image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    image_ids = [image_id for image_id in image_ids if image_id]
    if not image_ids:
        raise ValueError(f"VOC split file is empty: {split_path}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"VOC split contains duplicate image IDs: {split_path}")

    intersections = np.zeros(len(normalized_classes), dtype=np.int64)
    unions = np.zeros(len(normalized_classes), dtype=np.int64)
    per_image: list[dict[str, object]] = []
    for image_id in image_ids:
        if Path(image_id).name != image_id or "\\" in image_id:
            raise ValueError(f"invalid VOC image ID in split: {image_id!r}")
        target_path = voc_root / "SegmentationClass" / f"{image_id}.png"
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        output_dir = prediction_dir / image_id
        prediction_path = output_dir / "semantic_mask.npy"
        metadata_path = output_dir / "predictions.json"
        for required in (target_path, image_path, prediction_path, metadata_path):
            if not required.is_file():
                raise FileNotFoundError(f"required VOC evaluation file is missing: {required}")
        run_metadata_path = output_dir / "run_metadata.json"
        if require_all_candidates and not run_metadata_path.is_file():
            raise FileNotFoundError(
                f"run metadata is required to verify candidate classes: {run_metadata_path}"
            )

        target_native = load_indexed_mask(target_path)
        prediction_native = load_indexed_mask(prediction_path)
        if target_native.shape != prediction_native.shape:
            raise ValueError(
                f"prediction and VOC target dimensions differ for {image_id}: "
                f"{prediction_native.shape} != {target_native.shape}"
            )
        target_labels = set(np.unique(target_native).tolist())
        invalid_labels = target_labels - set(range(21)) - {255}
        if invalid_labels:
            raise ValueError(
                f"invalid VOC labels for {image_id}: {sorted(invalid_labels)}"
            )
        prediction_records = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(prediction_records, list):
            raise TypeError(f"predictions metadata must be a list: {metadata_path}")
        if any(not isinstance(record, dict) for record in prediction_records):
            raise ValueError(f"every prediction record must be an object: {metadata_path}")
        prediction_classes = [record.get("class") for record in prediction_records]
        if any(not isinstance(name, str) for name in prediction_classes):
            raise ValueError(f"every prediction record must contain a class name: {metadata_path}")
        normalized_prediction_classes = [name.strip().lower() for name in prediction_classes]
        if len(set(normalized_prediction_classes)) != len(normalized_prediction_classes):
            raise ValueError(f"duplicate predicted classes in {metadata_path}")
        if require_all_candidates:
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            recorded_candidates = (
                run_metadata.get("candidate_classes")
                if isinstance(run_metadata, dict)
                else None
            )
            if not isinstance(recorded_candidates, list) or any(
                not isinstance(name, str) for name in recorded_candidates
            ):
                raise ValueError(
                    f"run metadata must contain a candidate_classes list: {run_metadata_path}"
                )
            normalized_candidates = [name.strip().lower() for name in recorded_candidates]
            if (
                len(set(normalized_candidates)) != len(normalized_candidates)
                or set(normalized_candidates) != set(normalized_classes)
            ):
                raise ValueError(
                    f"inference candidate classes for {image_id} do not match evaluation "
                    f"classes: {normalized_candidates} != {normalized_classes}"
                )

        target = _voc_target_to_candidate_indices(
            target_native, normalized_classes, ignore_index
        )
        prediction = _prediction_to_candidate_indices(
            prediction_native, prediction_classes, normalized_classes
        )
        valid = target != ignore_index
        class_scores: dict[str, float | None] = {}
        image_values = []
        for candidate_index, class_name in enumerate(normalized_classes, start=1):
            predicted = (prediction == candidate_index) & valid
            expected = target == candidate_index
            intersection = int(np.logical_and(predicted, expected).sum())
            union = int(np.logical_or(predicted, expected).sum())
            intersections[candidate_index - 1] += intersection
            unions[candidate_index - 1] += union
            score = intersection / union if union else float("nan")
            class_scores[class_name] = score if np.isfinite(score) else None
            if union:
                image_values.append(score)
        per_image.append(
            {
                "image": image_id,
                "mIoU": float(np.mean(image_values)) if image_values else None,
                "per_class_IoU": class_scores,
            }
        )

    per_class_iou = {
        class_name: (
            float(intersections[index] / unions[index]) if unions[index] else None
        )
        for index, class_name in enumerate(normalized_classes)
    }
    included = [value for value in per_class_iou.values() if value is not None]
    return {
        "dataset": "PASCAL VOC 2012",
        "split": str(split_path),
        "image_count": len(per_image),
        "classes": normalized_classes,
        "ignore_index": ignore_index,
        "mIoU": float(np.mean(included)) if included else None,
        "per_class_IoU": per_class_iou,
        "class_intersections": {
            name: int(intersections[index]) for index, name in enumerate(normalized_classes)
        },
        "class_unions": {
            name: int(unions[index]) for index, name in enumerate(normalized_classes)
        },
        "images": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()
    result = evaluate_files(args.predictions, args.targets, args.classes, args.ignore_index)
    print(json.dumps(result, indent=2))


def voc_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions on PASCAL VOC 2012.")
    parser.add_argument("predictions", type=Path, help="Batch output directory keyed by VOC image ID")
    parser.add_argument("voc_root", type=Path, help="Path to VOC2012, containing JPEGImages and SegmentationClass")
    parser.add_argument("--classes", nargs="+", required=True, help="VOC classes used during inference")
    parser.add_argument("--split-file", type=Path, help="Image-ID list; defaults to the VOC validation split")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument(
        "--require-all-candidates",
        action="store_true",
        help="require each image's run metadata to list exactly the evaluation classes",
    )
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON report")
    args = parser.parse_args()
    result = evaluate_voc(
        args.predictions,
        args.voc_root,
        args.classes,
        args.split_file,
        args.ignore_index,
        args.require_all_candidates,
    )
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()