"""Analysis for the report: QAP recognition metrics, result tables, qualitative figures.

    llava-prior-report qap     --traces runs/traces/val --split val
    llava-prior-report table   --pred runs/pred/val/* --out reports/table_val.md
    llava-prior-report figure  --pred runs/pred/val/official --ids 2007_000033 ... --out fig.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .trace import ImageTrace
from .voc import VOC21_NAMES, VOC_ROOT, dataset_names, gt_classes, read_ids


def voc_palette() -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    for label in range(256):
        value, r, g, b = label, 0, 0, 0
        for shift in range(8):
            r |= ((value >> 0) & 1) << (7 - shift)
            g |= ((value >> 1) & 1) << (7 - shift)
            b |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        palette[label] = (r, g, b)
    palette[255] = (255, 255, 255)
    return palette


def qap_metrics(traces: Path, voc_root: Path, ids: list[str], thresholds: list[float]) -> dict:
    names = json.loads((traces / "trace_config.json").read_text())["classes"]
    rows = []  # (class index, gt present, greedy yes, p_yes)
    for image_id in ids:
        trace = ImageTrace.load(traces, image_id)
        present = gt_classes(voc_root, image_id, names)
        for index, name in enumerate(names):
            entry = trace.classes[name]
            greedy = (not entry.forced) and "yes" in entry.response.lower()
            rows.append((index, name in present, greedy, entry.p_yes))
    data = np.array(rows, dtype=np.float64)
    cls, gt, greedy, p_yes = data.T

    def prf(pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
        m = np.ones_like(gt, bool) if mask is None else mask
        tp = float(((pred == 1) & (gt == 1) & m).sum())
        fp = float(((pred == 1) & (gt == 0) & m).sum())
        fn = float(((pred == 0) & (gt == 1) & m).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    result = {
        "images": len(ids),
        "greedy": prf(greedy),
        "p_yes": {str(t): prf((p_yes >= t).astype(float)) for t in thresholds},
        "greedy_per_class": {name: prf(greedy, cls == i) for i, name in enumerate(names)},
        "greedy_matches_p_yes_0.5": float(((p_yes >= 0.5) == (greedy == 1)).mean()),
    }
    return result


def cmd_qap(args: argparse.Namespace) -> None:
    ids = read_ids(args.voc_root, args.split, args.split_file)
    thresholds = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    result = qap_metrics(args.traces, args.voc_root, ids, thresholds)
    (args.traces / "qap_metrics.json").write_text(json.dumps(result, indent=1))
    g = result["greedy"]
    print(f"greedy  P {g['precision']:.3f} R {g['recall']:.3f} F1 {g['f1']:.3f}  (fp {g['fp']:.0f}, fn {g['fn']:.0f})")
    for t, m in result["p_yes"].items():
        print(f"p>={t:<5} P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}")
    worst = sorted(result["greedy_per_class"].items(), key=lambda kv: kv[1]["f1"])[:6]
    print("lowest-F1 classes:", ", ".join(f"{n} {m['f1']:.2f} (P {m['precision']:.2f} R {m['recall']:.2f})" for n, m in worst))


def cmd_table(args: argparse.Namespace) -> None:
    rows = []
    for pred_dir in args.pred:
        metrics_path = pred_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append((pred_dir.name, metrics))
    reference = dict(rows).get(args.reference)
    lines = ["| config | mIoU | Δ vs " + args.reference + " | mAcc | aAcc | images |", "|---|---|---|---|---|---|"]
    for name, m in rows:
        delta = "" if reference is None else f"{100 * (m['mIoU'] - reference['mIoU']):+.2f}"
        lines.append(f"| {name} | {100 * m['mIoU']:.2f} | {delta} | {100 * m['mAcc']:.2f} | {100 * m['aAcc']:.2f} | {m['images']} |")
    if args.per_class:
        lines += ["", "| class | " + " | ".join(n for n, _ in rows) + " |", "|---|" + "---|" * len(rows)]
        for cls in VOC21_NAMES:
            values = [m["IoU"][cls] for _, m in rows]
            lines.append(f"| {cls} | " + " | ".join("-" if v is None else f"{100 * v:.1f}" for v in values) + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


def per_image_confusions(pred_dir: Path, voc_root: Path, ids: list[str]) -> np.ndarray:
    from .voc import confusion

    classes = len(dataset_names(voc_root))
    out = np.zeros((len(ids), classes, classes), dtype=np.int64)
    for k, image_id in enumerate(ids):
        pred = np.asarray(Image.open(pred_dir / f"{image_id}.png"))
        target = np.asarray(Image.open(voc_root / "SegmentationClass" / f"{image_id}.png"))
        out[k] = confusion(pred, target, classes)
    return out


def miou(matrix: np.ndarray) -> float:
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(tp / union))


def cmd_compare(args: argparse.Namespace) -> None:
    """Paired bootstrap over images of the dataset-level mIoU difference (B - A)."""
    ids = read_ids(args.voc_root, args.split, args.split_file)
    a = per_image_confusions(args.a, args.voc_root, ids)
    b = per_image_confusions(args.b, args.voc_root, ids)
    observed = miou(b.sum(0)) - miou(a.sum(0))
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.samples):
        pick = rng.integers(0, len(ids), len(ids))
        diffs.append(miou(b[pick].sum(0)) - miou(a[pick].sum(0)))
    low, high = np.percentile(diffs, [2.5, 97.5])
    result = {"a": str(args.a), "b": str(args.b), "images": len(ids), "delta": observed,
              "ci95": [float(low), float(high)], "p_le_0": float(np.mean(np.array(diffs) <= 0))}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


def cmd_figure(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = voc_palette()
    columns = ["Image", "Ground truth", *[p.name for p in args.pred]]
    titles = args.titles or columns
    fig, axes = plt.subplots(len(args.ids), len(columns), figsize=(2.6 * len(columns), 2.1 * len(args.ids)))
    axes = np.atleast_2d(axes)
    for row, image_id in enumerate(args.ids):
        image = np.asarray(Image.open(args.voc_root / "JPEGImages" / f"{image_id}.jpg").convert("RGB"))
        target = np.asarray(Image.open(args.voc_root / "SegmentationClass" / f"{image_id}.png"))
        panels = [image, palette[target]]
        for pred_dir in args.pred:
            pred = np.asarray(Image.open(pred_dir / f"{image_id}.png"))
            panels.append((0.45 * image + 0.55 * palette[pred]).astype(np.uint8))
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--voc-root", type=Path, default=VOC_ROOT)
    common.add_argument("--split", default="val")
    common.add_argument("--split-file", type=Path)

    qap = sub.add_parser("qap", parents=[common])
    qap.add_argument("--traces", type=Path, required=True)
    qap.set_defaults(func=cmd_qap)

    table = sub.add_parser("table")
    table.add_argument("--pred", type=Path, nargs="+", required=True)
    table.add_argument("--reference", default="official")
    table.add_argument("--per-class", action="store_true")
    table.add_argument("--out", type=Path)
    table.set_defaults(func=cmd_table)

    compare = sub.add_parser("compare", parents=[common])
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument("--samples", type=int, default=2000)
    compare.add_argument("--out", type=Path)
    compare.set_defaults(func=cmd_compare)

    figure = sub.add_parser("figure", parents=[common])
    figure.add_argument("--pred", type=Path, nargs="+", required=True)
    figure.add_argument("--ids", nargs="+", required=True)
    figure.add_argument("--titles", nargs="+")
    figure.add_argument("--out", type=Path, required=True)
    figure.add_argument("--dpi", type=int, default=150)
    figure.set_defaults(func=cmd_figure)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
