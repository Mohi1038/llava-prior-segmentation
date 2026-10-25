"""Command-line entry point for single-image and batch FSeg-LLaVA inference."""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PriorSegmentationConfig
from .models import HuggingFaceLlavaBackend, SamPredictorBackend, choose_device
from .pipeline import merge_class_masks, run_pipeline
from .visualization import save_prediction_visualization

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _discover_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image format: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"input image or directory does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    images = sorted(
        (candidate for candidate in iterator if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda candidate: str(candidate).casefold(),
    )
    if not images:
        raise ValueError(f"no supported images found in {path}")
    return images


def _output_dir_for_image(
    image_path: Path,
    input_path: Path,
    output_root: Path,
) -> Path:
    if input_path.is_file():
        return output_root
    return output_root / image_path.relative_to(input_path).with_suffix("")


def _is_complete(output_dir: Path, image_path: Path, expected_run: dict[str, Any]) -> bool:
    required = ("predictions.json", "semantic_mask.npy", "visualization.png", "run_metadata.json")
    if not all((output_dir / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    matches = metadata.get("image") == str(image_path.resolve()) and all(
        metadata.get(key) == value for key, value in expected_run.items()
    )
    if not matches:
        return False
    if expected_run.get("sam_prompt_strategy") == "points_box_mask_prior_multimask_iou_v1":
        try:
            predictions = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(predictions, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("prior"), str)
            or not Path(item["prior"]).is_file()
            for item in predictions
        ):
            return False
    return True


def _write_prediction(
    image_path: Path,
    output_dir: Path,
    image: Image.Image,
    predictions: list[Any],
    args: argparse.Namespace,
    config: PriorSegmentationConfig,
    llava: HuggingFaceLlavaBackend,
    sam: SamPredictorBackend,
    started: float,
    model_load_seconds: float,
    candidate_classes: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for prediction in predictions:
        class_name = prediction.observation.class_name
        mask_path = output_dir / f"{class_name.replace(' ', '_')}.npy"
        prior_path = output_dir / f"{class_name.replace(' ', '_')}_prior.npy"
        np.save(mask_path, prediction.sam_mask.detach().cpu().numpy())
        if prediction.sam_prior_mask is None:
            raise RuntimeError(f"missing SAM prior mask for prediction {class_name!r}")
        np.save(prior_path, prediction.sam_prior_mask.detach().cpu().numpy())
        metadata.append(
            {
                "class": class_name,
                "response": prediction.observation.response,
                "mask": str(mask_path),
                "prior": str(prior_path),
            }
        )

    np.save(
        output_dir / "semantic_mask.npy",
        merge_class_masks(predictions, (image.height, image.width)).numpy(),
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_prediction_visualization(image_path, output_dir)
    run_metadata = {
        "image": str(image_path.resolve()),
        "candidate_classes": candidate_classes,
        "models": {
            "llava": args.llava_model,
            "sam_checkpoint": str(args.sam_checkpoint.resolve()),
            "sam_model_type": args.sam_model_type,
        },
        "devices": {"llava": str(llava.device), "sam": str(sam.device)},
        "quantization": args.quantize,
        "sam_prompt_strategy": "points_box_mask_prior_multimask_iou_v1",
        "config": asdict(config),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "software": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "numpy": version("numpy"),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "success",
        "image": str(image_path),
        "output_dir": str(output_dir),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FSeg-LLaVA on one image or a directory of images."
    )
    parser.add_argument("image", type=Path, help="An image file or a directory of images")
    parser.add_argument("--classes", nargs="+", help="Candidate foreground class names")
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sam-device",
        help="Device for SAM; defaults to --device. Use cpu to conserve GPU memory for LLaVA.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search input directories recursively"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip images with complete matching saved outputs"
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        help="JSON object mapping image stems to per-image candidate-class lists",
    )
    parser.add_argument(
        "--quantize",
        default="none",
        choices=["none", "int8"],
        help="Quantize LLaVA's language model (int8 needs optimum-quanto; useful with limited GPU memory)",
    )
    parser.add_argument("--version", action="version", version="llava-prior-segmentation 0.1.0")
    args = parser.parse_args()
    if not args.classes or args.sam_checkpoint is None:
        parser.error("--classes and --sam-checkpoint are required")

    try:
        image_paths = [path.resolve() for path in _discover_images(args.image, args.recursive)]
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    input_path = args.image.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    class_map: dict[str, list[str]] = {}
    if args.class_map:
        try:
            loaded_class_map = json.loads(args.class_map.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not read --class-map: {exc}")
        if not isinstance(loaded_class_map, dict):
            parser.error("--class-map must be a JSON object mapping image stems to class lists")
        for image_id, class_names in loaded_class_map.items():
            if (
                not isinstance(image_id, str)
                or not image_id
                or not isinstance(class_names, list)
                or not class_names
                or any(not isinstance(name, str) or not name.strip() for name in class_names)
            ):
                parser.error("--class-map entries must map non-empty image IDs to non-empty class lists")
            normalized_classes = [name.strip() for name in class_names]
            if len({name.casefold() for name in normalized_classes}) != len(normalized_classes):
                parser.error(f"--class-map contains duplicate classes for image {image_id!r}")
            unexpected = [name for name in normalized_classes if name not in args.classes]
            if unexpected:
                parser.error(
                    f"--class-map classes for {image_id!r} are absent from --classes: {unexpected}"
                )
            class_map[image_id] = normalized_classes
        missing_ids = [path.stem for path in image_paths if path.stem not in class_map]
        if missing_ids:
            parser.error(f"--class-map has no class list for image(s): {missing_ids[:5]}")
    sam_device = args.sam_device or args.device
    expected_shared = {
        "models": {
            "llava": args.llava_model,
            "sam_checkpoint": str(args.sam_checkpoint.resolve()),
            "sam_model_type": args.sam_model_type,
        },
        "devices": {
            "llava": str(choose_device(args.device)),
            "sam": str(choose_device(sam_device)),
        },
        "quantization": args.quantize,
        "sam_prompt_strategy": "points_box_mask_prior_multimask_iou_v1",
    }
    batch_started = time.perf_counter()
    results = []
    pending_images = []
    for index, image_path in enumerate(image_paths, start=1):
        output_dir = _output_dir_for_image(image_path, input_path, output_root)
        label = image_path.relative_to(input_path) if input_path.is_dir() else image_path.name
        image_classes = class_map.get(image_path.stem, args.classes)
        image_config = PriorSegmentationConfig(candidate_classes=tuple(image_classes))
        expected_config = asdict(image_config)
        expected_config["candidate_classes"] = list(image_config.candidate_classes)
        expected_run = {
            **expected_shared,
            "candidate_classes": image_classes,
            "config": expected_config,
        }
        if args.resume and _is_complete(output_dir, image_path, expected_run):
            print(f"[{index}/{len(image_paths)}] Skipping completed image: {label}", flush=True)
            results.append({"status": "skipped", "image": str(image_path), "output_dir": str(output_dir)})
            continue
        pending_images.append(
            (index, image_path, output_dir, label, image_classes, image_config)
        )

    if pending_images:
        llava = HuggingFaceLlavaBackend(
            model_id=args.llava_model, device_name=args.device, quantize=args.quantize
        )
        sam = SamPredictorBackend(args.sam_checkpoint, args.sam_model_type, sam_device)
    model_load_seconds = time.perf_counter() - batch_started if pending_images else 0.0
    for index, image_path, output_dir, label, image_classes, image_config in pending_images:
        print(f"[{index}/{len(image_paths)}] Processing: {label}", flush=True)
        started = time.perf_counter()
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            predictions = run_pipeline(image, image_classes, llava, sam, image_config)
            result = _write_prediction(
                image_path,
                output_dir,
                image,
                predictions,
                args,
                image_config,
                llava,
                sam,
                started,
                model_load_seconds,
                image_classes,
            )
            result["detected_classes"] = [prediction.observation.class_name for prediction in predictions]
            print(
                f"  Completed in {result['runtime_seconds']:.1f}s "
                f"({len(predictions)} detected class(es))",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - isolate and record each failed batch item
            result = {
                "status": "failed",
                "image": str(image_path),
                "output_dir": str(output_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED: {result['error']}", file=sys.stderr, flush=True)
        results.append(result)

    summary = {
        "input": str(input_path),
        "candidate_classes": args.classes,
        "image_count": len(image_paths),
        "success_count": sum(result["status"] == "success" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
        "completed_count": sum(
            result["status"] in {"success", "skipped"} for result in results
        ),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "runtime_seconds": round(time.perf_counter() - batch_started, 3),
        "results": results,
    }
    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Batch summary: {summary_path}")
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()