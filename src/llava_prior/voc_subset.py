"""Prepare reproducible, class-balanced VOC validation subsets."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import VOC_CLASS_IDS, load_indexed_mask


def prepare_voc_subset(
    voc_root: Path,
    output_dir: Path,
    samples_per_class: int = 2,
) -> dict[str, Any]:
    """Select low- and high-coverage examples per class and stage their images.

    The class folders contain source IDs, and the staged image folder stores
    each selected image once. Ground-truth masks remain in the original VOC tree.
    """
    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    split_path = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"VOC validation split does not exist: {split_path}")
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2 to represent small and large objects")

    image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    image_ids = sorted({image_id for image_id in image_ids if image_id})
    if not image_ids:
        raise ValueError(f"VOC validation split is empty: {split_path}")

    areas: dict[str, list[tuple[str, int, int]]] = {name: [] for name in VOC_CLASS_IDS}
    for image_id in image_ids:
        if Path(image_id).name != image_id or "\\" in image_id:
            raise ValueError(f"invalid VOC image ID: {image_id!r}")
        mask_path = voc_root / "SegmentationClass" / f"{image_id}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"VOC segmentation mask is missing: {mask_path}")
        mask = load_indexed_mask(mask_path)
        image_pixels = int(mask.size)
        for class_name, class_id in VOC_CLASS_IDS.items():
            pixel_count = int(np.count_nonzero(mask == class_id))
            if pixel_count:
                areas[class_name].append((image_id, pixel_count, image_pixels))

    selections: dict[str, list[dict[str, Any]]] = {}
    for class_name, samples in areas.items():
        if len(samples) < samples_per_class:
            raise ValueError(
                f"VOC validation split has only {len(samples)} examples for {class_name!r}; "
                f"cannot select {samples_per_class}"
            )
        ordered = sorted(samples, key=lambda item: (item[1], item[0]))
        quantiles = np.linspace(0, len(ordered) - 1, samples_per_class)
        chosen_indices: list[int] = []
        for quantile in quantiles:
            target_index = round(float(quantile))
            if target_index not in chosen_indices:
                chosen_indices.append(target_index)
                continue
            available = sorted(
                (index for index in range(len(ordered)) if index not in chosen_indices),
                key=lambda index: (abs(index - target_index), index),
            )
            chosen_indices.append(available[0])
        selections[class_name] = [
            {
                "image_id": ordered[index][0],
                "class_pixel_count": ordered[index][1],
                "class_coverage": ordered[index][1] / ordered[index][2],
                "coverage_rank": "small" if sample_index == 0 else "large",
            }
            for sample_index, index in enumerate(sorted(chosen_indices))
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_images = output_dir / "images"
    staged_images.mkdir(parents=True, exist_ok=True)
    class_map: dict[str, list[str]] = {}
    unique_ids = sorted(
        {
            sample["image_id"]
            for class_samples in selections.values()
            for sample in class_samples
        }
    )
    for class_name, class_samples in selections.items():
        class_dir = output_dir / "classes" / class_name
        class_image_dir = class_dir / "images"
        class_mask_dir = class_dir / "masks"
        class_image_dir.mkdir(parents=True, exist_ok=True)
        class_mask_dir.mkdir(parents=True, exist_ok=True)
        class_ids = [sample["image_id"] for sample in class_samples]
        (class_dir / "image_ids.txt").write_text(
            "\n".join(class_ids) + "\n",
            encoding="utf-8",
        )
        for image_id in class_ids:
            class_map.setdefault(image_id, []).append(class_name)
            shutil.copy2(
                voc_root / "JPEGImages" / f"{image_id}.jpg",
                class_image_dir / f"{image_id}.jpg",
            )
            shutil.copy2(
                voc_root / "SegmentationClass" / f"{image_id}.png",
                class_mask_dir / f"{image_id}.png",
            )

    for image_id in unique_ids:
        source = voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"VOC image is missing: {source}")
        shutil.copy2(source, staged_images / source.name)
    (output_dir / "image_ids.txt").write_text(
        "\n".join(unique_ids) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset": "PASCAL VOC 2012",
        "source_split": str(split_path),
        "selection_rule": (
            "For each class, sort validation examples by class-pixel count then image ID; "
            "select evenly spaced ranks, including smallest and largest coverage."
        ),
        "samples_per_class": samples_per_class,
        "class_ids": VOC_CLASS_IDS,
        "selected": selections,
        "image_classes": {image_id: sorted(names) for image_id, names in sorted(class_map.items())},
        "unique_image_count": len(unique_ids),
        "class_image_associations": sum(len(values) for values in class_map.values()),
        "images_directory": str(staged_images),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (output_dir / "class_map.json").write_text(
        json.dumps(manifest["image_classes"], indent=2),
        encoding="utf-8",
    )
    return manifest


def prepare_voc_random_subset(
    voc_root: Path,
    output_dir: Path,
    image_count: int = 100,
    seed: int = 2026,
) -> dict[str, Any]:
    """Stage a deterministic VOC validation sample without consulting annotations."""
    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    split_path = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"VOC validation split does not exist: {split_path}")

    image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    if not image_ids:
        raise ValueError(f"VOC validation split is empty: {split_path}")
    if any(
        not image_id or Path(image_id).name != image_id or "\\" in image_id
        for image_id in image_ids
    ):
        raise ValueError(f"VOC validation split contains an invalid image ID: {split_path}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"VOC validation split contains duplicate image IDs: {split_path}")
    if image_count < 1 or image_count > len(image_ids):
        raise ValueError(f"image_count must be between 1 and {len(image_ids)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")

    selected_ids = sorted(
        image_ids,
        key=lambda image_id: (
            hashlib.sha256(f"{seed}:{image_id}".encode()).digest(),
            image_id,
        ),
    )[:image_count]
    staged_images = output_dir / "images"
    staged_images.mkdir(parents=True, exist_ok=True)
    for image_id in selected_ids:
        source = voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"VOC image is missing: {source}")
        shutil.copy2(source, staged_images / source.name)

    (output_dir / "image_ids.txt").write_text(
        "\n".join(selected_ids) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "PASCAL VOC 2012",
        "source_split": str(split_path),
        "selection_rule": (
            "Select the lowest SHA-256 ranks of '<seed>:<image_id>'; "
            "selection does not read image annotations."
        ),
        "seed": seed,
        "image_count": len(selected_ids),
        "image_ids": selected_ids,
        "candidate_classes": list(VOC_CLASS_IDS),
        "candidate_policy": "Query all 20 VOC foreground classes for every selected image.",
        "images_directory": str(staged_images),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voc_root", type=Path, help="VOC2012 root containing JPEGImages and SegmentationClass")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("class-balanced", "random"),
        default="class-balanced",
        help="class-balanced selection uses annotations; random selection does not",
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.selection == "random":
        manifest = prepare_voc_random_subset(
            args.voc_root,
            args.output_dir,
            args.image_count,
            args.seed,
        )
    else:
        manifest = prepare_voc_subset(args.voc_root, args.output_dir, args.samples_per_class)
    if args.selection == "random":
        summary = {
            "image_count": manifest["image_count"],
            "selection": args.selection,
            "seed": manifest["seed"],
            "output_dir": str(args.output_dir.resolve()),
        }
    else:
        summary = {
            "unique_image_count": manifest["unique_image_count"],
            "class_image_associations": manifest["class_image_associations"],
            "samples_per_class": manifest["samples_per_class"],
            "output_dir": str(args.output_dir.resolve()),
        }
    print(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
