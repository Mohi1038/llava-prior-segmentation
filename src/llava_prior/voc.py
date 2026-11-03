"""VOC21 experiment drivers: trace (LLaVA, GPU), segment (SAM, GPU), evaluate (CPU).

Typical flow (see ``slurm/``)::

    llava-prior-voc trace   --split val --out runs/traces/val
    llava-prior-voc segment --split val --traces runs/traces/val --configs configs/voc_main.json --out runs/pred/val
    llava-prior-voc eval    --split val --pred runs/pred/val/<config>

Ground-truth masks are read during ``trace`` for one purpose only: LLaVA additionally gets a
forced "Yes" trace for classes it rejected but that are present, which the QAP-oracle analysis
needs. Greedy (non-oracle) configs ignore forced traces, so their predictions never depend on
labels. ``segment`` reads labels only for ``qap: oracle`` configs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .activation import SegmentSettings, build_prompts, run_sam
from .trace import ImageTrace, LlavaTracer, TraceSettings, is_complete

PROJECT = Path(__file__).resolve().parents[2]
VOC_ROOT = PROJECT / "datasets" / "VOC2012" / "VOCdevkit" / "VOC2012"
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)  # fmt: skip
VOC21_NAMES = ("background", *VOC_CLASSES)


def dataset_names(voc_root: Path) -> tuple[str, ...]:
    """Label names including background (label 0). A dataset converted to the VOC layout (e.g.
    COCO-Object) lists its foreground classes in ``<root>/classes.txt``; VOC itself has none."""
    path = Path(voc_root) / "classes.txt"
    if not path.is_file():
        return VOC21_NAMES
    return ("background", *[line.strip() for line in path.read_text().splitlines() if line.strip()])


def read_class_names(path: Path | None, voc_root: Path = VOC_ROOT) -> list[str]:
    """Query names, one per line, in label order (line i -> label i+1)."""
    labels = list(dataset_names(voc_root)[1:])
    if path is None:
        return labels
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(names) != len(labels):
        raise ValueError(f"{path} must list {len(labels)} names in label order")
    return names


def read_ids(voc_root: Path, split: str, split_file: Path | None) -> list[str]:
    path = split_file or voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def gt_classes(voc_root: Path, image_id: str, names: list[str]) -> set[str]:
    target = np.asarray(Image.open(voc_root / "SegmentationClass" / f"{image_id}.png"))
    return {names[label - 1] for label in np.unique(target) if 1 <= label <= len(names)}


def _shard(ids: list[str], start: int, limit: int | None) -> list[str]:
    return ids[start : None if limit is None else start + limit]


def cmd_trace(args: argparse.Namespace) -> None:
    names = read_class_names(args.class_file, args.voc_root)
    ids = _shard(read_ids(args.voc_root, args.split, args.split_file), args.start, args.limit)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    settings = TraceSettings(
        model_id=args.model,
        per_layer=args.per_layer,
        screen_p_yes=args.screen_p_yes,
        force_classes_with_p_yes=args.force_p_yes,
        capture_in_generation=args.capture,
        extra_queries=args.extra_queries,
        # Qwen2.5-VL overflows in fp16 (NaN logits); it is run in bf16.
        dtype="bfloat16" if "qwen" in args.model.lower() else "float16",
    )
    (out / "trace_config.json").write_text(json.dumps({"settings": asdict(settings), "classes": names}, indent=1))
    todo = [image_id for image_id in ids if not is_complete(out, image_id)]
    print(f"{len(ids) - len(todo)} cached, {len(todo)} to trace", flush=True)
    if not todo:
        return
    # --override person="person, people": re-query only these classes with a new name and
    # copy every other class from --base-traces (names keep their original keys).
    overrides = dict(item.split("=", 1) for item in args.override)
    unknown = set(overrides) - set(names)
    if unknown or (overrides and args.base_traces is None):
        raise SystemExit(f"--override needs --base-traces and known class names (unknown: {unknown})")
    if overrides:
        (out / "trace_config.json").write_text(
            json.dumps({"settings": asdict(settings), "classes": names, "overrides": overrides,
                        "base_traces": str(args.base_traces)}, indent=1)
        )
    if "qwen" in args.model.lower():
        from .qwen import QwenTracer

        tracer = QwenTracer(settings)
    else:
        tracer = LlavaTracer(settings)
    started = time.time()
    for index, image_id in enumerate(todo, 1):
        image = Image.open(args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        force = gt_classes(args.voc_root, image_id, names) if args.force_gt else set()
        t0 = time.time()
        if overrides:
            trace = ImageTrace.load(args.base_traces, image_id)
            queried = tracer.trace_image(
                image, image_id, list(overrides.values()), {overrides[n] for n in force if n in overrides}
            )
            for name, query in overrides.items():
                entry = queried.classes[query]
                entry.class_name = name
                trace.classes[name] = entry
            trace.settings["overrides"] = overrides
        else:
            trace = tracer.trace_image(image, image_id, names, force)
        trace.settings["seconds"] = round(time.time() - t0, 2)
        trace.save(out)
        said_yes = [n for n, c in trace.classes.items() if not c.forced and "yes" in c.response.lower()]
        rate = (time.time() - started) / index
        print(
            f"[{index}/{len(todo)}] {image_id} {time.time() - t0:.1f}s yes={said_yes} "
            f"eta={rate * (len(todo) - index) / 60:.0f}min",
            flush=True,
        )


def cmd_augment(args: argparse.Namespace) -> None:
    """Add P(yes) under the alternative query wordings to existing traces (in place)."""
    config = json.loads((args.traces / "trace_config.json").read_text())
    names = config["classes"]
    model = config["settings"]["model_id"]
    ids = _shard(read_ids(args.voc_root, args.split, args.split_file), args.start, args.limit)
    todo = [i for i in ids if ImageTrace.load(args.traces, i).classes[names[0]].p_yes_extra is None]
    print(f"{len(ids) - len(todo)} already augmented, {len(todo)} to augment", flush=True)
    if not todo:
        return
    settings = TraceSettings(**config["settings"])
    if "qwen" in model.lower():
        from .qwen import QwenTracer

        tracer = QwenTracer(settings)
    else:
        tracer = LlavaTracer(settings)
    started = time.time()
    for index, image_id in enumerate(todo, 1):
        trace = ImageTrace.load(args.traces, image_id)
        image = Image.open(args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB")
        for name, scores in tracer.extra_scores(image, names).items():
            trace.classes[name].p_yes_extra = scores
        trace.save(args.traces)
        if index % 50 == 0:
            rate = (time.time() - started) / index
            print(f"[{index}/{len(todo)}] eta={rate * (len(todo) - index) / 60:.0f}min", flush=True)


def load_configs(path: Path) -> dict[str, SegmentSettings]:
    raw = json.loads(path.read_text())
    base = SegmentSettings()
    return {name: replace(base, **overrides) for name, overrides in raw.items()}


def cmd_segment(args: argparse.Namespace) -> None:
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    names = json.loads((args.traces / "trace_config.json").read_text())["classes"]
    ids = _shard(read_ids(args.voc_root, args.split, args.split_file), args.start, args.limit)
    configs = load_configs(args.configs)
    for name, settings in configs.items():
        (args.out / name).mkdir(parents=True, exist_ok=True)
        (args.out / name / "settings.json").write_text(json.dumps(asdict(settings), indent=1))
        if args.save_coarse:
            (args.out / f"{name}_coarse").mkdir(parents=True, exist_ok=True)
    predictor = None
    if not args.coarse_only:
        sam = sam_model_registry[args.sam_type](checkpoint=str(args.sam_checkpoint))
        predictor = SamPredictor(sam.to("cuda" if torch.cuda.is_available() else "cpu"))
    started = time.time()
    for index, image_id in enumerate(ids, 1):
        todo = {n: c for n, c in configs.items() if not (args.out / n / f"{image_id}.png").is_file()}
        if not todo:
            continue
        trace = ImageTrace.load(args.traces, image_id)
        image = np.asarray(Image.open(args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB"))
        gt = gt_classes(args.voc_root, image_id, names) if any(c.qap == "oracle" for c in todo.values()) else None
        if args.coarse_only:  # LLaVA-only maps, no SAM (layer study)
            for config_name, settings in todo.items():
                labels = build_prompts(trace, names, settings, gt).coarse_labels.astype(np.uint8)
                Image.fromarray(labels).save(args.out / config_name / f"{image_id}.png")
            continue
        predictor.set_image(image)
        for config_name, settings in todo.items():
            prompts = build_prompts(trace, names, settings, gt)
            prediction = run_sam(predictor, prompts, settings, image.shape[:2])
            Image.fromarray(prediction.astype(np.uint8)).save(args.out / config_name / f"{image_id}.png")
            if args.save_coarse:  # LLaVA-only prediction (no SAM), for the ablation table
                coarse = Image.fromarray(prompts.coarse_labels.astype(np.uint8))
                coarse.save(args.out / f"{config_name}_coarse" / f"{image_id}.png")
        predictor.reset_image()
        if index % 25 == 0:
            rate = (time.time() - started) / index
            print(f"[{index}/{len(ids)}] eta={rate * (len(ids) - index) / 60:.0f}min", flush=True)


def confusion(pred: np.ndarray, target: np.ndarray, classes: int, ignore: int = 255) -> np.ndarray:
    valid = target != ignore
    index = target[valid].astype(np.int64) * classes + pred[valid].astype(np.int64)
    return np.bincount(index, minlength=classes * classes).reshape(classes, classes)


def summarize(matrix: np.ndarray, names: tuple[str, ...] = VOC21_NAMES) -> dict[str, object]:
    """mmseg ``IoUMetric`` aggregation: dataset-level IoU per class, NaN-aware mean."""
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / union
        acc = tp / matrix.sum(1)
    return {
        "mIoU": float(np.nanmean(iou)),
        "mAcc": float(np.nanmean(acc)),
        "aAcc": float(tp.sum() / matrix.sum()),
        "IoU": {name: (None if np.isnan(v) else float(v)) for name, v in zip(names, iou)},
    }


def evaluate_dir(pred_dir: Path, voc_root: Path, ids: list[str]) -> tuple[dict, np.ndarray]:
    names = dataset_names(voc_root)
    classes = len(names)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    missing = []
    for image_id in ids:
        path = pred_dir / f"{image_id}.png"
        if not path.is_file():
            missing.append(image_id)
            continue
        pred = np.asarray(Image.open(path))
        target = np.asarray(Image.open(voc_root / "SegmentationClass" / f"{image_id}.png"))
        matrix += confusion(pred, target, classes)
    result = summarize(matrix, names)
    result.update({"images": len(ids) - len(missing), "missing": len(missing)})
    return result, matrix


def cmd_eval(args: argparse.Namespace) -> None:
    ids = read_ids(args.voc_root, args.split, args.split_file)
    for pred_dir in args.pred:
        result, matrix = evaluate_dir(pred_dir, args.voc_root, ids)
        if result["missing"] and not args.allow_missing:
            raise SystemExit(f"{pred_dir}: {result['missing']} predictions missing")
        (pred_dir / "metrics.json").write_text(json.dumps(result, indent=1))
        np.save(pred_dir / "confusion.npy", matrix)
        print(f"{pred_dir.name:40s} mIoU {100 * result['mIoU']:.2f}  images {result['images']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--voc-root", type=Path, default=VOC_ROOT)
    common.add_argument("--split", default="val")
    common.add_argument("--split-file", type=Path)
    common.add_argument("--start", type=int, default=0)
    common.add_argument("--limit", type=int)

    trace = sub.add_parser("trace", parents=[common])
    trace.add_argument("--out", type=Path, required=True)
    trace.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    trace.add_argument("--class-file", type=Path)
    trace.add_argument("--no-force-gt", dest="force_gt", action="store_false")
    trace.add_argument("--override", nargs="*", default=[], metavar="CLASS=QUERY")
    trace.add_argument("--base-traces", type=Path)
    trace.add_argument("--per-layer", action="store_true", help="store per-layer maps (layer study)")
    trace.add_argument("--screen-p-yes", type=float, default=0.0, help="batched first-token screening (0 = off)")
    trace.add_argument("--extra-queries", action="store_true", help="P(yes) under alternative wordings")
    trace.add_argument("--capture", action="store_true", help="record states/attention during generation")
    trace.add_argument("--force-p-yes", type=float, default=0.05, help="force maps for 'no' classes above this P(yes)")
    trace.set_defaults(func=cmd_trace)

    augment = sub.add_parser("augment", parents=[common])
    augment.add_argument("--traces", type=Path, required=True)
    augment.set_defaults(func=cmd_augment)

    segment = sub.add_parser("segment", parents=[common])
    segment.add_argument("--traces", type=Path, required=True)
    segment.add_argument("--configs", type=Path, required=True)
    segment.add_argument("--out", type=Path, required=True)
    segment.add_argument("--sam-checkpoint", type=Path, default=PROJECT / "checkpoints" / "sam_vit_h_4b8939.pth")
    segment.add_argument("--sam-type", default="vit_h")
    segment.add_argument("--save-coarse", action="store_true")
    segment.add_argument("--coarse-only", action="store_true", help="skip SAM; save LLaVA-only maps")
    segment.set_defaults(func=cmd_segment)

    evaluate = sub.add_parser("eval", parents=[common])
    evaluate.add_argument("--pred", type=Path, nargs="+", required=True)
    evaluate.add_argument("--allow-missing", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
