"""Derived statistics for the NeurIPS-format report, computed from saved predictions and traces only.

CPU only; reads runs/pred, runs/official, runs/traces and the dataset ground truth.
Writes reports/neurips/data/stats.json (+ per-object rows for the size analysis).

Run from the project root:  .venv/bin/python reports/neurips/compute_stats.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "neurips" / "data"
VOC = ROOT / "datasets" / "VOC2012" / "VOCdevkit" / "VOC2012"
COCO = ROOT / "datasets" / "coco_object"

import sys

sys.path.insert(0, str(ROOT / "src"))
from llava_prior.voc import gt_classes


def qap_metrics(traces: Path, root: Path, ids: list[str]) -> dict:
    """Same counting as llava_prior.report.qap_metrics (greedy answer, forced = no), from the JSON only."""
    names = json.loads((traces / "trace_config.json").read_text())["classes"]
    rows = []
    for image_id in ids:
        meta = json.loads((traces / f"{image_id}.json").read_text())["classes"]
        gt = gt_classes(root, image_id, names)
        for i, n in enumerate(names):
            e = meta[n]
            rows.append((i, n in gt, (not e["forced"]) and "yes" in e["response"].lower()))
    cls, gt, yes = np.array(rows, dtype=float).T

    def prf(mask):
        tp = float(((yes == 1) & (gt == 1) & mask).sum())
        fp = float(((yes == 1) & (gt == 0) & mask).sum())
        fn = float(((yes == 0) & (gt == 1) & mask).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else float("nan"),
                "tp": tp, "fp": fp, "fn": fn}

    return {"greedy": prf(np.ones_like(gt, bool)), "greedy_per_class": {n: prf(cls == i) for i, n in enumerate(names)}}

SYSTEMS_VOC = {
    "ours": ROOT / "runs/pred/val_vocnames/base",
    "ours_coarse": ROOT / "runs/pred/val_vocnames/base_coarse",
    "ours_oracle": ROOT / "runs/pred/val_vocnames/qap_oracle",
    "official": ROOT / "runs/official/val/bg",
    "v16v": ROOT / "runs/pred/v16v_val/base",
    "v16v_oracle": ROOT / "runs/pred/v16v_val/qap_oracle",
    "v16m": ROOT / "runs/pred/v16m_val/base",
    "v16m_oracle": ROOT / "runs/pred/v16m_val/qap_oracle",
    "combo": ROOT / "runs/pred/val_vocnames/combo_small",
}
SYSTEMS_COCO = {
    "coco": ROOT / "runs/pred/coco_v15/base",
    "coco_coarse": ROOT / "runs/pred/coco_v15/base_coarse",
    "coco_oracle": ROOT / "runs/pred/coco_v15/qap_oracle",
}
PAIRS_VOC = [("ours", "official"), ("ours", "v16v"), ("ours", "v16m"), ("ours", "ours_oracle"),
             ("ours", "combo"), ("v16v", "v16m")]
PAIRS_COCO = [("coco", "coco_oracle")]


def ids_of(root: Path) -> list[str]:
    return (root / "ImageSets/Segmentation/val.txt").read_text().split()


def confusion(pred: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    keep = target != 255
    return np.bincount(n * target[keep].astype(np.int64) + pred[keep], minlength=n * n).reshape(n, n)


def _one(args: tuple[str, Path, list[Path], int]) -> np.ndarray:
    image_id, root, dirs, n = args
    target = np.asarray(Image.open(root / "SegmentationClass" / f"{image_id}.png"))
    return np.stack([confusion(np.asarray(Image.open(d / f"{image_id}.png")), target, n) for d in dirs])


def per_image(root: Path, systems: dict[str, Path], n: int) -> dict[str, np.ndarray]:
    ids = ids_of(root)
    dirs = list(systems.values())
    with ProcessPoolExecutor(24) as pool:
        mats = list(pool.map(_one, [(i, root, dirs, n) for i in ids], chunksize=32))
    stacked = np.stack(mats).astype(np.int32)  # images x systems x n x n
    return {name: stacked[:, k] for k, name in enumerate(systems)}


def miou(m: np.ndarray) -> float:
    tp = np.diag(m).astype(np.float64)
    union = m.sum(0) + m.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(tp / union))


def _weights(n: int, samples: int, seed: int) -> np.ndarray:
    """Bootstrap resamples as per-image counts (samples x n) = resampling images with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=samples).astype(np.float64)


def _mious(m: np.ndarray, w: np.ndarray) -> np.ndarray:
    """mIoU of every resample at once: one (samples x images) @ (images x classes^2) product."""
    k = m.shape[1]
    sums = (w @ m.reshape(len(m), -1).astype(np.float64)).reshape(len(w), k, k)
    tp = np.diagonal(sums, axis1=1, axis2=2)
    union = sums.sum(1) + sums.sum(2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nanmean(tp / union, axis=1)


def bootstrap(a: np.ndarray, b: np.ndarray, samples: int = 1000) -> dict:
    """Paired bootstrap: the same resampled images for both systems."""
    w = _weights(len(a), samples, 0)
    diffs = _mious(b, w) - _mious(a, w)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": 100 * (miou(b.sum(0)) - miou(a.sum(0))), "ci95": [100 * lo, 100 * hi]}


def system_ci(m: np.ndarray, samples: int = 1000) -> list[float]:
    vals = _mious(m, _weights(len(m), samples, 1))
    return [100 * float(v) for v in np.percentile(vals, [2.5, 97.5])]


def errors(m: np.ndarray) -> dict:
    """Pixel-level error decomposition of the summed confusion (rows = GT, cols = prediction)."""
    fg_gt = m[1:].sum()
    correct = np.trace(m[1:, 1:])
    to_bg = m[1:, 0].sum()
    wrong_cls = m[1:, 1:].sum() - correct
    false_fg = m[0, 1:].sum()
    return {"fg_correct": correct / fg_gt, "fg_to_background": to_bg / fg_gt, "fg_wrong_class": wrong_cls / fg_gt,
            "bg_marked_fg_per_fg": false_fg / fg_gt}


def per_object(mats: dict[str, np.ndarray], systems: list[str]) -> list[dict]:
    """One row per (image, GT class present): area fraction of the image and IoU per system."""
    first = mats[systems[0]]
    rows = []
    for k in range(len(first)):
        total = first[k].sum()
        for c in range(1, first.shape[1]):
            area = first[k][c].sum()
            if area == 0:
                continue
            row = {"img": k, "cls": c, "frac": float(area / total)}
            for s in systems:
                m = mats[s][k]
                tp = m[c, c]
                row[s] = float(tp / (m[c].sum() + m[:, c].sum() - tp))
            rows.append(row)
    return rows


def per_image_miou(m: np.ndarray) -> list[float]:
    out = []
    for x in m:
        tp = np.diag(x).astype(float)
        union = x.sum(0) + x.sum(1) - tp
        ok = union > 0
        out.append(float(np.mean(tp[ok] / union[ok])))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    # Recognition (QAP: LLaVA's greedy yes/no) per model and dataset
    stats["qap"] = {}
    for key, traces, root in [("ours", "val_vocnames", VOC), ("v16v", "v16v_val", VOC), ("v16m", "v16m_val", VOC),
                              ("coco", "coco_v15", COCO)]:
        r = qap_metrics(ROOT / "runs/traces" / traces, root, ids_of(root))
        stats["qap"][key] = {"greedy": r["greedy"], "per_class": r["greedy_per_class"]}
        print("qap", key, {k: round(v, 3) for k, v in r["greedy"].items()}, flush=True)

    voc = per_image(VOC, SYSTEMS_VOC, 21)
    print("voc confusions done", flush=True)
    coco = per_image(COCO, SYSTEMS_COCO, 81)
    print("coco confusions done", flush=True)

    stats["miou"] = {s: 100 * miou(m.sum(0)) for s, m in {**voc, **coco}.items()}
    stats["ci"] = {s: system_ci(m) for s, m in {**voc, **coco}.items()}
    stats["paired"] = {f"{a}->{b}": bootstrap(voc[a], voc[b]) for a, b in PAIRS_VOC}
    stats["paired"].update({f"{a}->{b}": bootstrap(coco[a], coco[b]) for a, b in PAIRS_COCO})
    stats["errors"] = {s: {k: float(v) for k, v in errors(m.sum(0)).items()} for s, m in {**voc, **coco}.items()}
    stats["per_image_miou"] = {s: per_image_miou(voc[s]) for s in ("ours", "official", "v16v", "v16m", "ours_oracle")}
    stats["per_image_miou"]["coco"] = per_image_miou(coco["coco"])
    stats["voc_ids"] = ids_of(VOC)
    stats["coco_ids"] = ids_of(COCO)
    (OUT / "stats.json").write_text(json.dumps(stats))
    (OUT / "objects_voc.json").write_text(json.dumps(per_object(voc, ["ours", "ours_coarse", "ours_oracle", "official", "v16m"])))
    (OUT / "objects_coco.json").write_text(json.dumps(per_object(coco, ["coco", "coco_coarse", "coco_oracle"])))
    print("miou check", {k: round(v, 2) for k, v in stats["miou"].items()})


if __name__ == "__main__":
    main()
