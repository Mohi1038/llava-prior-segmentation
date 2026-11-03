"""Build the report's LaTeX tables and plots from saved run outputs (metrics.json files).

Run from the project root:  python reports/make_assets.py
Writes reports/tables/*.tex and reports/figures/layers.pdf. Missing runs are skipped with a note.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "runs" / "pred"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"


def miou(run: str) -> float | None:
    path = PRED / run / "metrics.json"
    return 100 * json.loads(path.read_text())["mIoU"] if path.is_file() else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body)


def ablation_table() -> None:
    rows = [
        ("Reproduction (released-code settings)", "base", "base", "68.0"),
        ("Class-token state read one position early", "shifted_token", "shifted_token", "--"),
        ("Paper formula $D_f = D\\cdot M_A$ (code: union)", "paper_product", "paper_product", "--"),
        ("No attention mask ($D_f = D$)", "no_attention", "no_attention", "67.3"),
        ("No prototype refinement ($M_r=\\hat M_f$)", "no_prototype", "no_prototype", "67.3"),
        ("SAM: points only", "points_only", "points_only", "61.4"),
        ("SAM: box only", "box_only", "box_only", "57.8"),
        ("LLaVA maps only, no SAM", "base_coarse", "base_coarse", "--"),
        ("Oracle class list (QAP replaced by labels)", "qap_oracle", "qap_oracle", "--"),
    ]
    base_dev, base_val = miou("dev_vocnames/base"), miou("val_vocnames/base")
    lines = []
    for label, dev, val, paper in rows:
        d, v = miou(f"dev_vocnames/{dev}"), miou(f"val_vocnames/{val}")
        delta = "" if dev == "base" or v is None else f" ({v - base_val:+.2f})"
        lines.append(f"{label} & {fmt(d)} & {fmt(v)}{delta} & {paper} \\\\")
    write("ablations", "\n".join(lines) + "\n")
    _ = base_dev


def tuning_table() -> None:
    groups = [
        ("$\\alpha$ = 0.65 / 0.85", ["alpha0.65", "alpha0.85"]),
        ("$\\beta$ = 0.16 / 0.22", ["beta0.16", "beta0.22"]),
        ("QAP threshold $p(\\text{yes})\\ge$ 0.35 / 0.65 / 0.8", ["qap_p0.35", "qap_p0.65", "qap_p0.8"]),
        ("SAM: 1 refinement round (code: 3)", ["sam_iter1"]),
        ("Negative points labelled 0 (code: $-1$)", ["neg_label0"]),
        ("No negative points", ["no_negatives"]),
        ("Overlaps to later class (code: LLaVA argmax)", ["overlap_last"]),
        ("Evidence filter (best of 6 settings)", ["filter_p0.7_a0.02"]),
        ("Per-class background threshold", ["bg_per_class"]),
        ("Relative background 0.6 / 0.7", ["bg_rel0.6", "bg_rel0.7"]),
        ("Per-instance SAM boxes (min.\\ area 1\\%)", ["comp_f0.01"]),
        ("Query ``person, people'' for person", ["../dev_person_people/base"]),
        ("Combination: no prototype, no negatives, overlap to later", ["combo_small"]),
    ]
    lines = []
    for label, runs in groups:
        values = " / ".join(fmt(miou(f"dev_vocnames/{run}")) for run in runs)
        lines.append(f"{label} & {values} \\\\")
    write("tuning", "\n".join(lines) + "\n")


def per_class_table() -> None:
    paths = {name: PRED / "val_vocnames" / name / "metrics.json" for name in ("base", "qap_oracle", "base_coarse")}
    if not all(p.is_file() for p in paths.values()):
        return
    iou = {name: json.loads(p.read_text())["IoU"] for name, p in paths.items()}
    names = list(iou["base"])
    half = (len(names) + 1) // 2
    lines = []
    for left, right in zip(names[:half], names[half:] + [None]):
        cells = []
        for name in (left, right):
            if name is None:
                cells.append(" & & &")
                continue
            values = [100 * iou[k][name] for k in ("base", "qap_oracle", "base_coarse")]
            cells.append(f"{name} & {values[0]:.1f} & {values[1]:.1f} & {values[2]:.1f}")
        lines.append(" & ".join(cells) + " \\\\")
    write("per_class", "\n".join(lines) + "\n")


RANGES = ["5_9", "7_9", "7_11", "7_13", "7_17", "10_13", "15_22", "22_30"]
PAPER_RANGES = {"5_9": 63.7, "7_9": 65.8, "7_11": 66.7, "7_13": 68.0, "7_17": 66.8, "10_13": 67.6,
                "15_22": 49.1, "22_30": 15.2}


def precision_recall(run: str) -> tuple[float, float, float] | None:
    """Class-correct foreground precision/recall and predicted-foreground share of pixels."""
    import numpy as np

    path = PRED / run / "confusion.npy"
    if not path.is_file():
        return None
    m = np.load(path)
    tp = np.diag(m)[1:].sum()
    return tp / m[:, 1:].sum(), tp / m[1:, :].sum(), m[:, 1:].sum() / m.sum()


def layer_assets() -> None:
    tags = [t for t in ("dev", "val") if (PRED / f"{t}_layers").is_dir()]
    if not tags:
        print("no layer study yet")
        return
    lines = []
    for r in RANGES:
        lo, hi = r.split("_")
        cells = []
        for tag in ("dev", "val"):
            cells += [fmt(miou(f"{tag}_layers/map_layers_{r}"), 1), fmt(miou(f"{tag}_layers/layers_{r}"), 1)]
        lines.append(f"{lo}--{hi} & " + " & ".join(cells) + f" & {PAPER_RANGES[r]:.1f} \\\\")
    write("layer_ranges", "\n".join(lines) + "\n")

    tag = "val" if "val" in tags and precision_recall("val_layers/layers_22_30") else "dev"
    lines = []
    for r in ("7_13", "15_22", "22_30"):
        lo, hi = r.split("_")
        for prefix, label in (("map_layers_", "map only"), ("layers_", "with SAM")):
            pr = precision_recall(f"{tag}_layers/{prefix}{r}")
            if pr:
                lines.append(f"{lo}--{hi} & {label} & {pr[0]:.2f} & {pr[1]:.2f} & {100 * pr[2]:.0f}\\% \\\\")
    write("layer_precision", "\n".join(lines) + "\n")
    (TABLES / "layer_precision_split.tex").write_text("validation" if tag == "val" else "tuning")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    styles = {"dev": ("#8a8a8a", "tuning (200 train images)"), "val": ("#1f5aa6", "validation (1,449 images)")}
    best = {}
    for tag in tags:
        points = [(k, miou(f"{tag}_layers/layer_{k:02d}")) for k in range(33)]
        points = [(k, v) for k, v in points if v is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        color, label = styles[tag]
        ax.plot(xs, ys, marker="o", markersize=3, color=color, lw=1.4, label=label)
        best[tag] = max(points, key=lambda item: item[1])
    ax.axvspan(7, 13, color="#f2b134", alpha=0.25, lw=0, label="paper's layers 7--13")
    ax.set_xlabel("LLM layer (hidden-state index; 0 = input embeddings)")
    ax.set_ylabel("mIoU, map only")
    ax.set_xlim(-0.5, 32.5)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "layers.pdf")
    key = "val" if "val" in best else "dev"
    (TABLES / "layer_best.tex").write_text(f"layer {best[key][0]} ({best[key][1]:.1f} mIoU)")


def main() -> None:
    ablation_table()
    tuning_table()
    per_class_table()
    layer_assets()
    print("wrote", sorted(p.name for p in TABLES.glob("*.tex")))


if __name__ == "__main__":
    main()
